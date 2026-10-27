"""
Workspace-Scoped MCP Connection Manager Service (Milestone 6C).

Coordinates workspace-scoped MCP connection persistence, endpoint revalidation,
server-managed credential resolution, tool discovery caching, and default-deny tool allowlists.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
from typing import Any, Callable, Optional
from urllib.parse import urlparse
import time
import uuid

from fastapi import HTTPException, status
import httpx2
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.core.config import get_settings
from rag_app.crm.twenty_mcp_client import APPROVED_WRITE_BACK_TOOLS, READ_ONLY_ALLOWED_TOOLS
from rag_app.db.models import (
    CRMWriteBackApproval,
    Department,
    MCPToolOperationRecord,
    User,
    WorkflowRun,
    WorkflowTask,
    Workspace,
    WorkspaceMCPConnection,
    WorkspaceMember,
)
from rag_app.schemas.workflow import compute_canonical_draft_hash, WorkflowStatus
from rag_app.mcp.client import (
    GenericMCPClient,
    MCPAuthenticationError,
    MCPClientConfig,
    MCPClientError,
    MCPConnectionError,
    MCPPayloadTooLargeError,
    MCPProtocolError,
    MCPServerInfo,
    MCPSSRFError,
    MCPTimeoutError,
    MCPToolResult,
    MCPTransportType,
    validate_mcp_url,
)
from rag_app.schemas.mcp_connection import (
    MCPConnectionCreate,
    MCPConnectionResponse,
    MCPConnectionUpdate,
    MCPToolInvokeResponse,
)
from rag_app.security.authorization import authorize
from rag_app.security.encryption import (
    CredentialDecryptionError,
    CredentialEncryptionConfigurationError,
    CredentialEncryptionService,
)

logger = logging.getLogger(__name__)

MAX_CONNECTIONS_PER_WORKSPACE = 10

# Server-managed allowlist of approved MCP credential references and their endpoint constraints
APPROVED_MCP_CREDENTIAL_REFS: dict[str, dict[str, Any]] = {
    "SERVER_TWENTY_API_KEY": {
        "description": "Approved Twenty CRM API Key reference",
        "env_var": "TWENTY_API_KEY",
        "allowed_hosts": ("twenty.com", "localhost", "127.0.0.1", "mockserver"),
        "allow_subdomains": True,
    },
    "MCP_TEST_SERVER_KEY": {
        "description": "Integration test server credential reference",
        "env_var": "MCP_TEST_SERVER_KEY",
        "allowed_hosts": ("localhost", "127.0.0.1", "mockserver", "example.com"),
        "allow_subdomains": True,
    },
    "MCP_DEMO_API_KEY": {
        "description": "Approved demo MCP server credential reference",
        "env_var": "MCP_DEMO_API_KEY",
        "allowed_hosts": ("localhost", "127.0.0.1", "mockserver", "example.com"),
        "allow_subdomains": True,
    },
}

# Server-managed catalog of verified read-only MCP tools.
# Security Invariant: An untrusted workspace configuration can never declare an
# unverified or mutating tool as read-only. Only tools in this catalog may execute as read-only.
VERIFIED_READ_ONLY_MCP_TOOLS: frozenset[str] = frozenset(READ_ONLY_ALLOWED_TOOLS) | frozenset({
    "find_many_companies",
    "find_one_company",
    "find_many_people",
    "find_many_opportunities",
    "get_company",
    "get_contact",
    "list_companies",
    "list_contacts",
    "read_query",
    "get_search_status",
    "echo_read",
    "read_report",
})



class WorkspaceMCPConnectionService:
    """
    Manages external MCP server connections scoped strictly to a Workspace.
    Enforces multi-tenant workspace isolation, department boundary checks,
    default-deny tool execution, and server-managed credential references.
    """

    def __init__(
        self,
        session: AsyncSession,
        http_client_factory: Optional[Callable[..., httpx2.AsyncClient]] = None,
        encryption_service: Optional[CredentialEncryptionService] = None,
    ):
        self.session = session
        self.http_client_factory = http_client_factory
        self.settings = get_settings()
        self.encryption_service = encryption_service or CredentialEncryptionService()

    def _is_dev_or_test_env(self) -> bool:
        """
        Determines whether the service is executing in a dev or test environment,
        which allows private network loopback addresses (for local testing/smoke tests).
        In production, allow_private_hosts is strictly forbidden.
        """
        env_val = getattr(self.settings, "env", "production").lower()
        if env_val in ("dev", "development", "test", "testing"):
            return True
        if "disposable_test" in getattr(self.settings, "database_url", ""):
            return True
        if os.environ.get("PYTEST_CURRENT_TEST") is not None:
            return True
        return False

    @staticmethod
    def _sanitize_sensitive_payload(data: Any) -> Any:
        """
        Minimizes and redacts sensitive credentials, tokens, passwords, and private keys
        from durable operation arguments and result payloads.
        """
        if isinstance(data, dict):
            sanitized: dict[str, Any] = {}
            sensitive_pattern = re.compile(
                r"(password|secret|token|api[_-]?key|auth|bearer|private[_-]?key|credential)",
                re.IGNORECASE,
            )
            for k, v in data.items():
                if sensitive_pattern.search(str(k)):
                    sanitized[str(k)] = "[REDACTED]"
                else:
                    sanitized[str(k)] = WorkspaceMCPConnectionService._sanitize_sensitive_payload(v)
            return sanitized
        elif isinstance(data, list):
            return [WorkspaceMCPConnectionService._sanitize_sensitive_payload(item) for item in data]
        elif isinstance(data, str):
            masked = re.sub(
                r"(gsk_|sk-|tf_|live_|secret_|bearer\s+)[A-Za-z0-9_\-]+",
                "[REDACTED_SECRET]",
                data,
                flags=re.IGNORECASE,
            )
            return masked
        return data

    @classmethod
    def _validate_credential_ref(cls, ref: Optional[str]) -> Optional[str]:
        """
        Validates that credential_ref is in the explicit server-managed allowlist.
        Rejects raw API keys and arbitrary/server-wide environment variables.
        """
        if not ref or not ref.strip():
            return None
        cleaned = ref.strip()
        lower = cleaned.lower()
        if any(lower.startswith(prefix) for prefix in ("sk-", "gsk_", "tf_", "live_", "secret_", "bearer ")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Do not supply raw API keys in credential_ref. Provide an approved environment variable reference name.",
            )
        if cleaned not in APPROVED_MCP_CREDENTIAL_REFS:
            allowed_list = ", ".join(f"'{k}'" for k in sorted(APPROVED_MCP_CREDENTIAL_REFS.keys()))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Credential reference '{cleaned}' is not approved for MCP use. Permitted server references: [{allowed_list}].",
            )
        return cleaned

    @classmethod
    def _verify_credential_endpoint_compatibility(cls, credential_ref: str, endpoint_url: str) -> None:
        """
        Ensures an approved credential is only permitted to be sent to its authorized
        endpoint hosts, preventing forwarding server credentials to untrusted third parties.
        """
        if not credential_ref or credential_ref not in APPROVED_MCP_CREDENTIAL_REFS:
            return
        policy = APPROVED_MCP_CREDENTIAL_REFS[credential_ref]
        parsed = urlparse(endpoint_url)
        host = (parsed.hostname or "").lower()
        allowed_hosts = tuple(h.lower() for h in policy.get("allowed_hosts", ()))
        allow_subdomains = policy.get("allow_subdomains", False)

        matched = False
        for allowed in allowed_hosts:
            if host == allowed:
                matched = True
                break
            if allow_subdomains and host.endswith(f".{allowed}"):
                matched = True
                break

        if not matched:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Credential reference '{credential_ref}' cannot be bound to endpoint host '{host}'. It is only authorized for: {allowed_hosts}.",
            )

    @staticmethod
    def _is_masked_secret(val: Optional[str]) -> bool:
        """Returns True if the string is a masked placeholder and should never overwrite a secret."""
        if not val or not val.strip():
            return True
        v = val.strip()
        if all(c in ("•", "*", ".") for c in v):
            return True
        if v in ("[REDACTED]", "[REDACTED_SECRET]", "••••••••", "••••••••••••••••", "********"):
            return True
        return False

    @staticmethod
    def _mask_credential_ref(ref: Optional[str]) -> Optional[str]:
        """Never return raw credentials; return reference label or masked string."""
        if not ref:
            return None
        if any(ref.lower().startswith(p) for p in ("sk-", "gsk_", "tf_", "live_", "secret_")):
            return "••••••••••••••••"
        return ref

    async def _resolve_credential_secret(
        self,
        target: WorkspaceMCPConnection | Optional[str],
        endpoint_url: str,
    ) -> Optional[str]:
        """
        Resolves server-side API secret according to connection's explicit credential_mode:
        - 'default': Uses platform-managed credentials (e.g. Twenty CRM API key).
        - 'custom': Decrypts the workspace's encrypted custom API credential using MultiFernet.
        - 'env_ref': Resolves an approved server environment variable reference.
        Never exposes secrets in logs or responses.
        """
        if target is None:
            return None

        # Backward compatibility: if caller passed string credential_ref directly
        if isinstance(target, str):
            credential_ref = target
            if not credential_ref:
                return None
            if credential_ref not in APPROVED_MCP_CREDENTIAL_REFS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Credential reference '{credential_ref}' is not approved for MCP use.",
                )
            self._verify_credential_endpoint_compatibility(credential_ref, endpoint_url)
            policy = APPROVED_MCP_CREDENTIAL_REFS[credential_ref]
            env_var_name = policy.get("env_var", credential_ref)
            secret = os.environ.get(env_var_name) or os.environ.get(credential_ref)
            if secret:
                return secret
            attr_name = env_var_name.lower()
            if hasattr(self.settings, attr_name):
                val = getattr(self.settings, attr_name)
                if val:
                    return str(val)
            return None

        conn = target
        mode = conn.credential_mode

        if mode == "default":
            # Platform-managed credentials
            parsed = urlparse(endpoint_url)
            host = (parsed.hostname or "").lower()
            port = parsed.port
            # Check Twenty CRM default
            if (host in ("localhost", "127.0.0.1") and port == 3000) or ("twenty.com" in host):
                return getattr(self.settings, "twenty_api_key", None)
            return None

        elif mode == "custom":
            ciphertext = conn.encrypted_credential
            if not ciphertext:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Connection is configured for 'custom' credential mode, but no custom API key has been saved.",
                )
            try:
                decrypted = self.encryption_service.decrypt(ciphertext)
                # Key rotation check: if encrypted with older secondary key, rotate with primary key and persist
                try:
                    rotated_cipher, was_rotated = self.encryption_service.rotate(ciphertext)
                    if was_rotated:
                        conn.encrypted_credential = rotated_cipher
                        conn.credential_key_version = "v1"
                        self.session.add(conn)
                        await self.session.commit()
                except Exception as ex:
                    logger.warning("Failed to persist rotated credential ciphertext: %s", ex)
                return decrypted
            except CredentialEncryptionConfigurationError as cece:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Custom credential encryption service unavailable: {cece}",
                )
            except CredentialDecryptionError as cde:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to decrypt custom credential: ciphertext corrupted, tampered, or key mismatch.",
                )

        elif mode == "env_ref":
            credential_ref = conn.credential_ref
            if not credential_ref:
                return None
            if credential_ref not in APPROVED_MCP_CREDENTIAL_REFS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Credential reference '{credential_ref}' is not approved for MCP use.",
                )
            self._verify_credential_endpoint_compatibility(credential_ref, endpoint_url)
            policy = APPROVED_MCP_CREDENTIAL_REFS[credential_ref]
            env_var_name = policy.get("env_var", credential_ref)
            secret = os.environ.get(env_var_name) or os.environ.get(credential_ref)
            if secret:
                return secret
            attr_name = env_var_name.lower()
            if hasattr(self.settings, attr_name):
                val = getattr(self.settings, attr_name)
                if val:
                    return str(val)
            return None

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid credential_mode '{mode}'. Supported: 'default', 'custom', 'env_ref'.",
            )

    def _format_connection_response(self, conn: WorkspaceMCPConnection) -> MCPConnectionResponse:
        """Formats the ORM connection into a sanitized API response."""
        clean_config = dict(conn.configuration or {})
        # Security invariant: Never leak raw ciphertext or internal key versions in API responses
        clean_config.pop("encrypted_credential", None)
        clean_config.pop("credential_key_version", None)

        mode = conn.credential_mode
        return MCPConnectionResponse(
            id=conn.id,
            workspace_id=conn.workspace_id,
            name=conn.name,
            endpoint_url=conn.endpoint_url,
            transport=conn.transport,
            credential_mode=mode,
            has_custom_credential=conn.has_custom_credential,
            credential_ref=self._mask_credential_ref(conn.credential_ref) if mode == "env_ref" else None,
            is_enabled=conn.is_enabled,
            status=conn.status,
            error_message=conn.error_message,
            last_connected_at=conn.last_connected_at,
            last_synced_at=conn.last_synced_at,
            server_info=conn.server_info or {},
            discovered_tools=conn.discovered_tools or [],
            enabled_tools=conn.enabled_tools or [],
            configuration=clean_config,
            created_by=conn.created_by,
            created_at=conn.created_at,
            updated_at=conn.updated_at,
        )

    async def _check_workspace_access(
        self,
        user: User,
        workspace_id: uuid.UUID,
        require_write: bool = False,
    ) -> tuple[Workspace, str]:
        """
        Verifies that caller possesses active access to workspace and its parent department.
        ADMIN: always allowed (role 'ADMIN').
        OWNER / EDITOR: allowed read and write.
        VIEWER: allowed read only.
        Returns (Workspace, effective_role).
        """
        stmt = (
            select(Workspace)
            .options(selectinload(Workspace.department))
            .where(
                Workspace.id == workspace_id,
                Workspace.deleted_at.is_(None),
            )
        )
        ws = await self.session.scalar(stmt)
        if not ws:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{workspace_id}' not found.",
            )

        dept = ws.department
        if not dept or not dept.is_active or dept.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace department is inactive or deleted.",
            )

        if user.role == "ADMIN":
            return ws, "ADMIN"

        # Department boundary check
        is_dept_auth = await authorize(self.session, user, "VIEW", dept.code)
        if not is_dept_auth:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: You do not have access to department '{dept.code}'.",
            )

        # Workspace membership check
        member_stmt = select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user.id,
        )
        member = await self.session.scalar(member_stmt)

        if not member:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: You are not an active member of this workspace.",
            )

        if require_write and member.role not in ("OWNER", "EDITOR"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Only Workspace Owners or Editors can configure MCP connections.",
            )

        return ws, member.role

    async def create_connection(
        self,
        user: User,
        workspace_id: uuid.UUID,
        payload: MCPConnectionCreate,
    ) -> MCPConnectionResponse:
        """Registers a new external MCP connection within the specified workspace."""
        ws, _ = await self._check_workspace_access(user, workspace_id, require_write=True)

        is_dev = self._is_dev_or_test_env()
        allow_private = bool(payload.configuration.get("allow_private_hosts", False) and is_dev)
        try:
            validate_mcp_url(payload.endpoint_url, allow_private_hosts=allow_private)
        except MCPSSRFError as se:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid endpoint URL: {se}",
            )

        # Determine and validate credential_mode
        raw_mode = (payload.credential_mode or "").lower().strip()
        if not raw_mode:
            cred_mode = "env_ref" if payload.credential_ref else "default"
        else:
            cred_mode = raw_mode
        if cred_mode not in ("default", "custom", "env_ref"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid credential_mode '{cred_mode}'. Supported: 'default', 'custom', 'env_ref'.",
            )

        sanitized_config = dict(payload.configuration or {})
        if not is_dev:
            sanitized_config["allow_private_hosts"] = False
        sanitized_config["credential_mode"] = cred_mode

        clean_cred_ref = None
        if cred_mode == "env_ref":
            clean_cred_ref = self._validate_credential_ref(payload.credential_ref)
            if clean_cred_ref:
                self._verify_credential_endpoint_compatibility(clean_cred_ref, payload.endpoint_url)
            sanitized_config.pop("encrypted_credential", None)
            sanitized_config.pop("credential_key_version", None)
        elif cred_mode == "custom":
            clean_cred_ref = None
            if payload.custom_credential and not self._is_masked_secret(payload.custom_credential):
                try:
                    self.encryption_service.require_configured()
                    ciphertext = self.encryption_service.encrypt(payload.custom_credential.strip())
                    sanitized_config["encrypted_credential"] = ciphertext
                    sanitized_config["credential_key_version"] = "v1"
                except CredentialEncryptionConfigurationError as cece:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Cannot save custom credential: {cece}",
                    )
        elif cred_mode == "default":
            clean_cred_ref = None
            sanitized_config.pop("encrypted_credential", None)
            sanitized_config.pop("credential_key_version", None)

        # Enforce maximum connections per workspace bound
        count_stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        existing_conns = (await self.session.scalars(count_stmt)).all()
        if len(existing_conns) >= MAX_CONNECTIONS_PER_WORKSPACE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum connection limit reached ({MAX_CONNECTIONS_PER_WORKSPACE} connections per workspace).",
            )

        # Check unique name per workspace
        clean_name = payload.name.strip()
        if any(c.name.lower() == clean_name.lower() for c in existing_conns):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A connection named '{clean_name}' already exists in this workspace.",
            )

        transport = payload.transport.lower().strip()
        if transport not in ("streamable-http", "sse"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported transport '{transport}'. Supported: 'streamable-http', 'sse'.",
            )

        now = datetime.now(timezone.utc)
        conn = WorkspaceMCPConnection(
            workspace_id=workspace_id,
            name=clean_name,
            endpoint_url=payload.endpoint_url.strip(),
            transport=transport,
            credential_ref=clean_cred_ref,
            is_enabled=True,
            status="DISCONNECTED",
            error_message=None,
            server_info={},
            discovered_tools=[],
            enabled_tools=[],  # DEFAULT-DENY: empty allowlist
            configuration=sanitized_config,
            created_by=user.id,
            created_at=now,
            updated_at=now,
        )
        self.session.add(conn)
        await self.session.commit()
        await self.session.refresh(conn)

        logger.info(
            "Registered MCP connection '%s' (%s) for workspace '%s' by user '%s'",
            conn.name,
            conn.id,
            workspace_id,
            user.id,
        )
        return self._format_connection_response(conn)

    async def list_connections(
        self,
        user: User,
        workspace_id: uuid.UUID,
    ) -> list[MCPConnectionResponse]:
        """Lists all MCP connections for the authorized workspace."""
        ws, _ = await self._check_workspace_access(user, workspace_id, require_write=False)

        stmt = (
            select(WorkspaceMCPConnection)
            .where(WorkspaceMCPConnection.workspace_id == workspace_id)
            .order_by(WorkspaceMCPConnection.created_at.desc())
        )
        conns = (await self.session.scalars(stmt)).all()
        return [self._format_connection_response(c) for c in conns]

    async def get_connection(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> MCPConnectionResponse:
        """Retrieves a single connection by ID within the authorized workspace."""
        ws, _ = await self._check_workspace_access(user, workspace_id, require_write=False)

        stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.id == connection_id,
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        conn = await self.session.scalar(stmt)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP connection '{connection_id}' not found in workspace '{workspace_id}'.",
            )
        return self._format_connection_response(conn)

    async def update_connection(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
        payload: MCPConnectionUpdate,
    ) -> MCPConnectionResponse:
        """Updates configuration for an existing MCP connection."""
        ws, _ = await self._check_workspace_access(user, workspace_id, require_write=True)

        stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.id == connection_id,
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        conn = await self.session.scalar(stmt)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP connection '{connection_id}' not found in workspace '{workspace_id}'.",
            )

        new_endpoint = conn.endpoint_url
        if payload.endpoint_url is not None:
            allow_private = bool(
                (payload.configuration or conn.configuration).get("allow_private_hosts", False) and self._is_dev_or_test_env()
            )
            try:
                validate_mcp_url(payload.endpoint_url, allow_private_hosts=allow_private)
            except MCPSSRFError as se:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid endpoint URL: {se}",
                )
            new_endpoint = payload.endpoint_url.strip()
            conn.endpoint_url = new_endpoint

        if payload.name is not None:
            clean_name = payload.name.strip()
            # Check unique name
            dup_stmt = select(WorkspaceMCPConnection).where(
                WorkspaceMCPConnection.workspace_id == workspace_id,
                WorkspaceMCPConnection.name == clean_name,
                WorkspaceMCPConnection.id != connection_id,
            )
            if await self.session.scalar(dup_stmt):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A connection named '{clean_name}' already exists in this workspace.",
                )
            conn.name = clean_name

        if payload.transport is not None:
            tr = payload.transport.lower().strip()
            if tr not in ("streamable-http", "sse"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported transport '{tr}'. Supported: 'streamable-http', 'sse'.",
                )
            conn.transport = tr

        # 1. Update credential mode
        new_mode = conn.credential_mode
        if payload.credential_mode is not None:
            cand_mode = payload.credential_mode.lower().strip()
            if cand_mode not in ("default", "custom", "env_ref"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid credential_mode '{cand_mode}'. Supported: 'default', 'custom', 'env_ref'.",
                )
            new_mode = cand_mode
            conn.credential_mode = new_mode

        # 2. Clear custom credential if explicitly requested
        if payload.clear_custom_credential:
            conn.encrypted_credential = None
            conn.credential_key_version = None

        # 3. Update custom credential if provided and not masked
        if payload.custom_credential is not None:
            if not self._is_masked_secret(payload.custom_credential):
                try:
                    self.encryption_service.require_configured()
                    ciphertext = self.encryption_service.encrypt(payload.custom_credential.strip())
                    conn.encrypted_credential = ciphertext
                    conn.credential_key_version = "v1"
                except CredentialEncryptionConfigurationError as cece:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Cannot save custom credential: {cece}",
                    )

        # 4. Handle env_ref vs default/custom mode scoping
        if new_mode == "env_ref":
            if payload.credential_ref is not None:
                clean_ref = self._validate_credential_ref(payload.credential_ref)
                if clean_ref:
                    self._verify_credential_endpoint_compatibility(clean_ref, new_endpoint)
                conn.credential_ref = clean_ref
            elif payload.endpoint_url is not None and conn.credential_ref:
                self._verify_credential_endpoint_compatibility(conn.credential_ref, new_endpoint)
        else:
            if payload.credential_mode in ("default", "custom"):
                conn.credential_ref = None

        if payload.is_enabled is not None:
            conn.is_enabled = payload.is_enabled

        if payload.configuration is not None:
            is_dev = self._is_dev_or_test_env()
            sanitized_cfg = dict(payload.configuration)
            if not is_dev:
                sanitized_cfg["allow_private_hosts"] = False
            # Preserve internal security keys in configuration JSONB
            if conn.encrypted_credential and "encrypted_credential" not in sanitized_cfg:
                sanitized_cfg["encrypted_credential"] = conn.encrypted_credential
            if conn.credential_key_version and "credential_key_version" not in sanitized_cfg:
                sanitized_cfg["credential_key_version"] = conn.credential_key_version
            sanitized_cfg["credential_mode"] = conn.credential_mode
            conn.configuration = sanitized_cfg

        conn.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(conn)
        return self._format_connection_response(conn)

    async def delete_connection(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> None:
        """Deletes an MCP connection from the workspace."""
        await self._check_workspace_access(user, workspace_id, require_write=True)

        stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.id == connection_id,
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        conn = await self.session.scalar(stmt)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP connection '{connection_id}' not found in workspace '{workspace_id}'.",
            )

        await self.session.delete(conn)
        await self.session.commit()
        logger.info("Deleted MCP connection '%s' (%s) from workspace '%s'", conn.name, connection_id, workspace_id)

    async def sync_tools(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> MCPConnectionResponse:
        """
        Connects to the remote MCP server, captures server capabilities,
        and refreshes cached tool metadata. Discovered tools are NOT enabled automatically.
        """
        ws, _ = await self._check_workspace_access(user, workspace_id, require_write=True)

        stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.id == connection_id,
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        conn = await self.session.scalar(stmt)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP connection '{connection_id}' not found in workspace '{workspace_id}'.",
            )

        if not conn.is_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Connection is currently disabled.",
            )

        # 1. Re-validate endpoint URL before connecting
        allow_private = bool(conn.configuration.get("allow_private_hosts", False) and self._is_dev_or_test_env())
        try:
            validate_mcp_url(conn.endpoint_url, allow_private_hosts=allow_private)
        except MCPSSRFError as se:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid endpoint URL: {se}")

        # 2. Resolve server-side credentials with endpoint compatibility
        auth_headers: dict[str, str] = {}
        secret = await self._resolve_credential_secret(conn, conn.endpoint_url)
        if secret:
            auth_headers["Authorization"] = f"Bearer {secret}"

        # 2. Build Client Config
        allow_private = bool(conn.configuration.get("allow_private_hosts", False) and self._is_dev_or_test_env())

        timeout = min(float(conn.configuration.get("timeout_seconds", 10.0)), 30.0)
        read_timeout = min(float(conn.configuration.get("read_timeout_seconds", 30.0)), 60.0)
        transport_type = MCPTransportType(conn.transport)

        config = MCPClientConfig(
            endpoint_url=conn.endpoint_url,
            transport=transport_type,
            auth_headers=auth_headers,
            timeout_seconds=timeout,
            read_timeout_seconds=read_timeout,
            allow_private_hosts=allow_private,
        )

        now = datetime.now(timezone.utc)
        try:
            async with GenericMCPClient(config, http_client_factory=self.http_client_factory) as client:
                server_info: Optional[MCPServerInfo] = client.server_info
                tools = await client.list_tools()

                discovered_list: list[dict[str, Any]] = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                        "output_schema": t.output_schema,
                    }
                    for t in tools
                ]

                # Retain only enabled tools that still exist
                discovered_names = {t.name for t in tools}
                preserved_enabled = [name for name in (conn.enabled_tools or []) if name in discovered_names]

                conn.server_info = {
                    "name": server_info.name if server_info else "Unknown",
                    "version": server_info.version if server_info else None,
                    "protocol_version": server_info.protocol_version if server_info else None,
                    "instructions": server_info.instructions if server_info else None,
                    "capabilities": server_info.capabilities if server_info else {},
                }
                conn.discovered_tools = discovered_list
                conn.enabled_tools = preserved_enabled  # Preserves default-deny
                conn.status = "CONNECTED"
                conn.error_message = None
                conn.last_connected_at = now
                conn.last_synced_at = now

        except MCPAuthenticationError as ae:
            conn.status = "ERROR"
            conn.error_message = f"Authentication failure: {ae}"
        except MCPTimeoutError as te:
            conn.status = "ERROR"
            conn.error_message = f"Timeout: {te}"
        except MCPSSRFError as se:
            conn.status = "ERROR"
            conn.error_message = f"Security error: {se}"
        except MCPClientError as ce:
            conn.status = "ERROR"
            conn.error_message = f"MCP Connection error: {ce}"
        except Exception as ex:
            conn.status = "ERROR"
            conn.error_message = f"Unexpected failure: {type(ex).__name__}"

        conn.updated_at = now
        await self.session.commit()
        await self.session.refresh(conn)
        return self._format_connection_response(conn)

    async def test_connection(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Tests connectivity and credential authentication against the remote MCP server
        using the connection's configured credential mode without mutating enabled tools.
        """
        ws, _ = await self._check_workspace_access(user, workspace_id, require_write=True)

        stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.id == connection_id,
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        conn = await self.session.scalar(stmt)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP connection '{connection_id}' not found in workspace '{workspace_id}'.",
            )

        # 1. Re-validate endpoint URL before connecting
        allow_private = bool(conn.configuration.get("allow_private_hosts", False) and self._is_dev_or_test_env())
        try:
            validate_mcp_url(conn.endpoint_url, allow_private_hosts=allow_private)
        except MCPSSRFError as se:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid endpoint URL: {se}")

        # 2. Resolve credentials using configured mode
        auth_headers: dict[str, str] = {}
        secret = await self._resolve_credential_secret(conn, conn.endpoint_url)
        if secret:
            auth_headers["Authorization"] = f"Bearer {secret}"

        timeout = min(float(conn.configuration.get("timeout_seconds", 10.0)), 15.0)
        transport_type = MCPTransportType(conn.transport)

        config = MCPClientConfig(
            endpoint_url=conn.endpoint_url,
            transport=transport_type,
            auth_headers=auth_headers,
            timeout_seconds=timeout,
            read_timeout_seconds=timeout,
            allow_private_hosts=allow_private,
        )

        t0 = time.monotonic()
        try:
            async with GenericMCPClient(config, http_client_factory=self.http_client_factory) as client:
                tools = await client.list_tools()
                latency_ms = int((time.monotonic() - t0) * 1000)
                server_info = {
                    "name": client.server_info.name if client.server_info else "Unknown",
                    "version": client.server_info.version if client.server_info else None,
                    "protocol_version": client.server_info.protocol_version if client.server_info else None,
                }
                return {
                    "status": "SUCCESS",
                    "latency_ms": latency_ms,
                    "server_info": server_info,
                    "discovered_tool_count": len(tools),
                    "credential_mode": conn.credential_mode,
                    "message": f"Successfully connected via '{conn.credential_mode}' credentials. Discovered {len(tools)} tools.",
                }
        except MCPAuthenticationError as ae:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed with remote server: {ae}",
            )
        except MCPTimeoutError as te:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Connection timed out: {te}",
            )
        except MCPClientError as ce:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"MCP remote failure: {ce}",
            )

    async def toggle_tool(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
        tool_name: str,
        enabled: bool,
    ) -> MCPConnectionResponse:
        """Explicitly enables or disables an individual tool in the connection allowlist."""
        ws, _ = await self._check_workspace_access(user, workspace_id, require_write=True)

        stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.id == connection_id,
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        conn = await self.session.scalar(stmt)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP connection '{connection_id}' not found in workspace '{workspace_id}'.",
            )

        discovered = {t.get("name") for t in (conn.discovered_tools or [])}
        if tool_name not in discovered:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tool '{tool_name}' was not discovered on this MCP connection.",
            )

        current_enabled = set(conn.enabled_tools or [])
        if enabled:
            current_enabled.add(tool_name)
        else:
            current_enabled.discard(tool_name)

        conn.enabled_tools = sorted(list(current_enabled))
        conn.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(conn)
        return self._format_connection_response(conn)

    async def update_tool_allowlist(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
        allowed_tools: list[str],
    ) -> MCPConnectionResponse:
        """Replaces the entire enabled tool allowlist with a verified subset of discovered tools."""
        ws, _ = await self._check_workspace_access(user, workspace_id, require_write=True)

        stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.id == connection_id,
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        conn = await self.session.scalar(stmt)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP connection '{connection_id}' not found in workspace '{workspace_id}'.",
            )

        discovered = {t.get("name") for t in (conn.discovered_tools or [])}
        for name in allowed_tools:
            if name not in discovered:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot enable '{name}': tool was not discovered on this MCP connection.",
                )

        conn.enabled_tools = sorted(list(set(allowed_tools)))
        conn.updated_at = datetime.now(timezone.utc)
        await self.session.commit()
        await self.session.refresh(conn)
        return self._format_connection_response(conn)

    async def propose_tool_mutation(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
        tool_name: str,
        arguments: dict[str, Any],
        target_record_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        task_id: Optional[uuid.UUID] = None,
    ) -> CRMWriteBackApproval:
        """
        Milestone 6E: Proposes an exact mutation payload for human-in-the-loop approval.
        Validates target connection, tool allowlist, caller permissions, and arguments,
        then persists an approval record bound to normalized cryptographic draft hash.
        """
        # 1. Recheck workspace access
        ws, role = await self._check_workspace_access(user, workspace_id, require_write=False)

        # 2. Connection verification
        conn = await self.session.get(WorkspaceMCPConnection, connection_id)
        if not conn or conn.workspace_id != workspace_id or not conn.is_enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Active MCP connection '{connection_id}' not found in workspace.",
            )
        if conn.status != "CONNECTED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"MCP connection '{connection_id}' is in status '{conn.status}', not 'CONNECTED'.",
            )

        # 3. Allowlist & verified write tool verification
        enabled_tools = set(conn.enabled_tools or [])
        if tool_name not in enabled_tools:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Tool '{tool_name}' is not in the approved allowlist for this workspace connection.",
            )

        if tool_name not in APPROVED_WRITE_BACK_TOOLS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Tool '{tool_name}' is not an authorized write-back tool. Permitted write tools: {sorted(APPROVED_WRITE_BACK_TOOLS)}.",
            )

        # 4. Payload normalization & hash calculation
        markdown_body = (
            arguments.get("bodyV2", {}).get("markdown")
            if isinstance(arguments.get("bodyV2"), dict)
            else arguments.get("markdown") or arguments.get("body") or json.dumps(arguments, sort_keys=True)
        )
        canonical_draft_hash = compute_canonical_draft_hash(str(markdown_body))
        company_id = str(target_record_id or arguments.get("targetCompanyId") or arguments.get("company_id") or "company")

        now = datetime.now(timezone.utc)
        approval_id = uuid.uuid4()

        # 5. Persist CRMWriteBackApproval record bound to cryptographic hash
        approval = CRMWriteBackApproval(
            id=approval_id,
            operation_type="NOTE_CREATE",
            external_record_id=company_id,
            target_object="company",
            department_id=ws.department_id,
            requested_by=user.id,
            approved_by=user.id,
            status="PENDING",
            expires_at=now + timedelta(hours=24),
            created_at=now,
            approved_draft_hash=canonical_draft_hash,
            intended_action=tool_name,
            workflow_run_id=run_id,
        )
        self.session.add(approval)

        # 6. Persist initial PENDING MCPToolOperationRecord
        effective_idempotency_key = idempotency_key or f"prop-{approval_id.hex[:12]}"
        op_record = MCPToolOperationRecord(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            connection_id=connection_id,
            tool_name=tool_name,
            idempotency_key=effective_idempotency_key,
            status="PENDING",
            arguments=self._sanitize_sensitive_payload(arguments),
            approval_ticket_id=approval.id,
            run_id=run_id,
            task_id=task_id,
            created_by=user.id,
        )
        self.session.add(op_record)

        # 7. If linked to a workflow run, transition run to AWAITING_APPROVAL
        if run_id:
            run = await self.session.get(WorkflowRun, run_id)
            if run:
                run.status = WorkflowStatus.AWAITING_APPROVAL.value
                run.awaiting_approval_at = now
                run.draft_hash = canonical_draft_hash
                run.intended_action = tool_name
                run.target_record_id = company_id

        await self.session.commit()
        await self.session.refresh(approval)
        return approval

    async def invoke_workspace_tool(
        self,
        user: User,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
        tool_name: str,
        arguments: dict[str, Any],
        idempotency_key: Optional[str] = None,
        approval_ticket_id: Optional[uuid.UUID] = None,
        run_id: Optional[uuid.UUID] = None,
        task_id: Optional[uuid.UUID] = None,
    ) -> MCPToolInvokeResponse:
        """
        Executes an approved tool on the remote MCP server with strict security hardening:
        1. Default-deny allowlist enforcement.
        2. Role-based and server-verified risk classification:
           - Verified read-only tools: permitted to authorized callers (VIEWER+).
           - Mutating tools: require EDITOR/OWNER/ADMIN.
           - Destructive tools: require OWNER/ADMIN (and explicit approval where required).
           - Unknown or unclassified tools: fail closed until reviewed.
           - Workspace configuration CANNOT downgrade an unverified tool to read-only.
        3. Durable external operation state machine for mutating operations:
           - PENDING -> IN_FLIGHT -> CONFIRMED / UNKNOWN / RECONCILIATION_REQUIRED.
           - Timeouts after dispatch persist UNKNOWN; never marked FAILED; blind retry prevented.
           - Automated reconciliation via provider lookup if supported.
        4. Structured audit logging with correlation IDs and masked secrets.
        """
        ws, role = await self._check_workspace_access(user, workspace_id, require_write=False)

        stmt = select(WorkspaceMCPConnection).where(
            WorkspaceMCPConnection.id == connection_id,
            WorkspaceMCPConnection.workspace_id == workspace_id,
        )
        conn = await self.session.scalar(stmt)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MCP connection '{connection_id}' not found in workspace '{workspace_id}'.",
            )

        if not conn.is_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Connection is currently disabled.",
            )

        # ── 1. DEFAULT-DENY ALLOWLIST BOUNDARY ────────────────────────────────
        enabled_tools = set(conn.enabled_tools or [])
        if tool_name not in enabled_tools:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Tool '{tool_name}' is not in the approved allowlist for this workspace connection.",
            )

        # ── 2. ROLE & RISK-BASED AUTHORIZATION BOUNDARY (Milestone 6C.2 Part A) ──
        # Policy Hierarchy:
        # A. Server-verified read-only catalog (VERIFIED_READ_ONLY_MCP_TOOLS)
        # B. Workspace configuration CANNOT declare or downgrade an unverified tool to read-only.
        # C. Destructive tools require OWNER/ADMIN (and explicit approval where required).
        # D. Unclassified tools default to mutating (fail closed for VIEWERS; allowed only for EDITOR/OWNER).
        tool_policy = (conn.configuration or {}).get("tool_policies", {}).get(tool_name) or {}

        # Strict Server-Verified Read-Only Classification (Preflight Check 1)
        # An untrusted workspace configuration can never declare an unverified tool as read_only.
        # Only tools present in VERIFIED_READ_ONLY_MCP_TOOLS may execute as read_only.
        # Furthermore, workspace-level escalation to mutating or destructive is always respected.
        is_read_only = False
        if tool_name in VERIFIED_READ_ONLY_MCP_TOOLS:
            if tool_policy.get("risk_level") in ("destructive", "mutating") or tool_policy.get("requires_approval"):
                is_read_only = False
            else:
                is_read_only = True
        else:
            is_read_only = False

        if is_read_only:
            risk_level = "read_only"
        elif tool_policy.get("risk_level") == "destructive" or tool_policy.get("requires_approval"):
            risk_level = "destructive"
        else:
            risk_level = "mutating"

        # Enforce caller role against determined risk level
        if risk_level == "mutating" and role not in ("ADMIN", "OWNER", "EDITOR"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Forbidden: Tool '{tool_name}' is classified as side-effecting/mutating. "
                    "Workspace Viewers cannot execute mutating tools. Only Workspace Owners or Editors can execute this tool."
                ),
            )
        elif risk_level == "destructive" and role not in ("ADMIN", "OWNER"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Tool '{tool_name}' is classified as destructive. Only Workspace Owners or Admins can execute it.",
            )

        # Check explicit approval ticket requirement & validate ticket lifecycle
        if approval_ticket_id or (tool_policy and tool_policy.get("requires_approval")):
            if not approval_ticket_id and role not in ("ADMIN", "OWNER"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Execution rejected: Tool '{tool_name}' requires an approved human-approval ticket.",
                )

            if approval_ticket_id:
                approval = await self.session.get(CRMWriteBackApproval, approval_ticket_id)
                if not approval:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Approval ticket '{approval_ticket_id}' not found.",
                    )

                now = datetime.now(timezone.utc)
                if approval.status == "CONSUMED":
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Approval ticket '{approval_ticket_id}' has already been consumed and cannot be reused.",
                    )
                if approval.status == "REVOKED":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Approval ticket '{approval_ticket_id}' has been revoked.",
                    )
                if approval.status == "REJECTED":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Approval ticket '{approval_ticket_id}' was rejected.",
                    )
                if approval.expires_at <= now or approval.status == "EXPIRED":
                    approval.status = "EXPIRED"
                    await self.session.commit()
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Approval ticket '{approval_ticket_id}' has expired.",
                    )

                # Tool identity check
                if approval.intended_action and approval.intended_action != tool_name:
                    if not (approval.intended_action == "CRM_CREATE_NOTE" and tool_name == "create_one_note"):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Approval ticket '{approval_ticket_id}' intended action '{approval.intended_action}' cannot authorize tool '{tool_name}'.",
                        )

                # Target check
                target_val = str(arguments.get("targetCompanyId") or arguments.get("company_id") or "")
                if target_val and approval.external_record_id and approval.external_record_id not in (target_val, "company"):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Approval ticket for target '{approval.external_record_id}' cannot authorize target '{target_val}'.",
                    )

                # Draft content / payload verification (detect tampering)
                if approval.approved_draft_hash:
                    markdown_content = (
                        arguments.get("bodyV2", {}).get("markdown")
                        if isinstance(arguments.get("bodyV2"), dict)
                        else arguments.get("markdown") or arguments.get("body") or json.dumps(arguments, sort_keys=True)
                    )
                    calc_hash = compute_canonical_draft_hash(str(markdown_content))
                    alt_hash = compute_canonical_draft_hash(json.dumps(arguments, sort_keys=True))
                    if approval.approved_draft_hash not in (calc_hash, alt_hash):
                        approval.status = "REVOKED"
                        approval.revoked_at = now
                        await self.session.commit()
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=(
                                f"Proposed payload content has changed since approval was granted. "
                                f"Expected hash {approval.approved_draft_hash}, got {calc_hash}. Prior approval ticket has been revoked."
                            ),
                        )

                # Atomically consume ticket before external dispatch
                approval.status = "CONSUMED"
                approval.consumed_at = now
                approval.approved_by = user.id
                await self.session.commit()

        # ── 3. ENDPOINT & CREDENTIAL SECURITY (Milestone 6C.2 Part B) ─────────
        allow_private = bool(conn.configuration.get("allow_private_hosts", False) and self._is_dev_or_test_env())
        try:
            validate_mcp_url(conn.endpoint_url, allow_private_hosts=allow_private)
        except MCPSSRFError as se:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid endpoint URL: {se}")

        auth_headers: dict[str, str] = {}
        secret = await self._resolve_credential_secret(conn, conn.endpoint_url)
        if secret:
            auth_headers["Authorization"] = f"Bearer {secret}"

        timeout = min(float(conn.configuration.get("timeout_seconds", 10.0)), 30.0)
        read_timeout = min(float(conn.configuration.get("read_timeout_seconds", 30.0)), 60.0)
        transport_type = MCPTransportType(conn.transport)

        config = MCPClientConfig(
            endpoint_url=conn.endpoint_url,
            transport=transport_type,
            auth_headers=auth_headers,
            timeout_seconds=timeout,
            read_timeout_seconds=read_timeout,
            allow_private_hosts=allow_private,
        )

        start_time = time.monotonic()

        # ── 4. READ-ONLY TOOL EXECUTION PATH ─────────────────────────────────
        if risk_level == "read_only":
            try:
                async with GenericMCPClient(config, http_client_factory=self.http_client_factory) as client:
                    res: MCPToolResult = await client.call_tool(tool_name, arguments)
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    logger.info(
                        "MCP Read Tool Executed: ws=%s conn=%s tool=%s duration_ms=%d",
                        workspace_id, connection_id, tool_name, duration_ms,
                    )
                    return MCPToolInvokeResponse(
                        tool_name=tool_name,
                        is_error=res.is_error,
                        text=res.text,
                        structured_content=res.structured_content,
                        error_message=res.error_message,
                    )
            except MCPAuthenticationError as ae:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(ae))
            except MCPTimeoutError as te:
                raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(te))
            except MCPPayloadTooLargeError as pe:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(pe))
            except MCPClientError as ce:
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(ce))

        # ── 5. DURABLE MUTATING EXTERNAL OPERATION STATE MACHINE (Part C & E3) ──
        effective_idempotency_key = idempotency_key or str(uuid.uuid4())
        sanitized_arguments = self._sanitize_sensitive_payload(arguments or {})

        # Check existing operation record by (workspace_id, idempotency_key) or approval_ticket_id
        existing_op = None
        if effective_idempotency_key:
            existing_op_stmt = select(MCPToolOperationRecord).where(
                MCPToolOperationRecord.workspace_id == workspace_id,
                MCPToolOperationRecord.idempotency_key == effective_idempotency_key,
            )
            existing_op = await self.session.scalar(existing_op_stmt)

        if not existing_op and approval_ticket_id:
            op_by_app_stmt = select(MCPToolOperationRecord).where(
                MCPToolOperationRecord.approval_ticket_id == approval_ticket_id
            )
            existing_op = await self.session.scalar(op_by_app_stmt)

        if existing_op:
            # Preflight Check 2: Scoped to the intended tool operation
            if existing_op.tool_name != tool_name:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"Idempotency conflict: Idempotency key '{effective_idempotency_key}' was previously used "
                        f"with tool '{existing_op.tool_name}'."
                    ),
                )
            if existing_op.status == "CONFIRMED":
                # Idempotent replay: return cached confirmed response without re-executing remote tool
                cached_res = existing_op.result_payload or {}
                return MCPToolInvokeResponse(
                    tool_name=tool_name,
                    is_error=cached_res.get("is_error", False),
                    text=cached_res.get("text", ""),
                    structured_content={
                        **(cached_res.get("structured_content") or {}),
                        "is_replay": True,
                        "operation_id": str(existing_op.id),
                        "idempotency_key": existing_op.idempotency_key or effective_idempotency_key,
                    },
                    error_message=cached_res.get("error_message"),
                )
            elif existing_op.status in ("IN_FLIGHT", "UNKNOWN", "RECONCILIATION_REQUIRED"):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"Operation '{effective_idempotency_key}' is currently in state '{existing_op.status}'. "
                        "Blind retry or duplicate execution prevented. Perform reconciliation or inspect operation record."
                    ),
                )
            elif existing_op.status == "PENDING":
                # Re-use pre-created proposal operation record
                op_record = existing_op
                op_record.arguments = sanitized_arguments
                if idempotency_key:
                    op_record.idempotency_key = idempotency_key
                    effective_idempotency_key = idempotency_key
                else:
                    effective_idempotency_key = existing_op.idempotency_key or effective_idempotency_key
            else:
                op_record = existing_op
        else:
            # Persist new operation record as PENDING with race condition protection
            op_record = MCPToolOperationRecord(
                id=uuid.uuid4(),
                workspace_id=workspace_id,
                connection_id=connection_id,
                tool_name=tool_name,
                idempotency_key=effective_idempotency_key,
                status="PENDING",
                arguments=sanitized_arguments,
                approval_ticket_id=approval_ticket_id,
                run_id=run_id,
                task_id=task_id,
                created_by=user.id,
            )
            self.session.add(op_record)
            try:
                await self.session.commit()
                await self.session.refresh(op_record)
            except IntegrityError:
                await self.session.rollback()
                raced_op = await self.session.scalar(existing_op_stmt)
                if raced_op:
                    if raced_op.tool_name != tool_name:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail="Idempotency key conflict: concurrent operation created with different tool.",
                        )
                    if raced_op.status == "CONFIRMED":
                        cached_res = raced_op.result_payload or {}
                        return MCPToolInvokeResponse(
                            tool_name=tool_name,
                            is_error=cached_res.get("is_error", False),
                            text=cached_res.get("text", ""),
                            structured_content={
                                **(cached_res.get("structured_content") or {}),
                                "is_replay": True,
                                "operation_id": str(raced_op.id),
                                "idempotency_key": effective_idempotency_key,
                            },
                            error_message=cached_res.get("error_message"),
                        )
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"Operation '{effective_idempotency_key}' is currently in state '{raced_op.status}'. Blind retry or duplicate execution prevented.",
                        )
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Duplicate operation detected.")

        # Transition to IN_FLIGHT
        op_record.status = "IN_FLIGHT"
        op_record.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

        # Dispatch mutating tool to remote MCP server
        try:
            async with GenericMCPClient(config, http_client_factory=self.http_client_factory) as client:
                res: MCPToolResult = await client.call_tool(tool_name, arguments)
                duration_ms = int((time.monotonic() - start_time) * 1000)

                # Remote execution completed: persist outcome
                now = datetime.now(timezone.utc)
                if res.is_error:
                    op_record.status = "FAILED"
                    op_record.error_message = res.error_message or res.text or "Remote tool returned error"
                else:
                    op_record.status = "CONFIRMED"

                op_record.result_payload = self._sanitize_sensitive_payload({
                    "text": res.text,
                    "structured_content": res.structured_content,
                    "is_error": res.is_error,
                    "error_message": res.error_message,
                })
                op_record.completed_at = now
                op_record.updated_at = now
                await self.session.commit()

                if res.is_error:
                    logger.warning(
                        "MCP Mutating Tool FAILED: op_id=%s ws=%s conn=%s tool=%s duration_ms=%d error=%s",
                        op_record.id, workspace_id, connection_id, tool_name, duration_ms, op_record.error_message,
                        extra={
                            "event": "mcp_tool_invocation",
                            "operation_id": str(op_record.id),
                            "workspace_id": str(workspace_id),
                            "connection_id": str(connection_id),
                            "tool_name": tool_name,
                            "run_id": str(run_id) if run_id else None,
                            "task_id": str(task_id) if task_id else None,
                            "duration_ms": duration_ms,
                            "outcome": "FAILED",
                        },
                    )
                else:
                    logger.info(
                        "MCP Mutating Tool CONFIRMED: op_id=%s ws=%s conn=%s tool=%s duration_ms=%d",
                        op_record.id, workspace_id, connection_id, tool_name, duration_ms,
                        extra={
                            "event": "mcp_tool_invocation",
                            "operation_id": str(op_record.id),
                            "workspace_id": str(workspace_id),
                            "connection_id": str(connection_id),
                            "tool_name": tool_name,
                            "run_id": str(run_id) if run_id else None,
                            "task_id": str(task_id) if task_id else None,
                            "duration_ms": duration_ms,
                            "outcome": "CONFIRMED",
                        },
                    )
                return MCPToolInvokeResponse(
                    tool_name=tool_name,
                    is_error=res.is_error,
                    text=res.text,
                    structured_content={
                        **(res.structured_content or {}),
                        "operation_id": str(op_record.id),
                        "idempotency_key": effective_idempotency_key,
                    },
                    error_message=res.error_message,
                )

        except MCPTimeoutError as te:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            now = datetime.now(timezone.utc)
            # The tool was dispatched but response timed out: OUTCOME IS UNKNOWN!
            # DO NOT MARK AS FAILED! DO NOT BLINDLY RETRY!
            op_record.status = "UNKNOWN"
            op_record.error_message = f"Remote invocation timed out after dispatch: {te}"
            op_record.updated_at = now
            await self.session.commit()

            logger.warning(
                "MCP Mutating Tool TIMEOUT (Ambiguous): op_id=%s ws=%s tool=%s duration_ms=%d outcome=UNKNOWN",
                op_record.id, workspace_id, tool_name, duration_ms,
                extra={
                    "event": "mcp_tool_invocation",
                    "operation_id": str(op_record.id),
                    "workspace_id": str(workspace_id),
                    "tool_name": tool_name,
                    "duration_ms": duration_ms,
                    "outcome": "UNKNOWN",
                },
            )

            # Check if tool policy provides a reconciliation lookup tool
            reconciliation_tool = (tool_policy or {}).get("reconciliation_tool")
            if reconciliation_tool:
                try:
                    key_param = (tool_policy or {}).get("reconciliation_key_param", "id")
                    lookup_val = (arguments or {}).get(key_param) or effective_idempotency_key
                    async with GenericMCPClient(config, http_client_factory=self.http_client_factory) as recon_client:
                        recon_res = await recon_client.call_tool(reconciliation_tool, {key_param: lookup_val})
                        if not recon_res.is_error and recon_res.text and "not found" not in recon_res.text.lower():
                            # Successfully reconciled! The operation committed on the remote server
                            op_record.status = "CONFIRMED"
                            op_record.reconciliation_attempts = 1
                            op_record.last_reconciled_at = datetime.now(timezone.utc)
                            op_record.result_payload = {
                                "text": recon_res.text,
                                "structured_content": recon_res.structured_content,
                                "is_error": False,
                            }
                            op_record.completed_at = op_record.last_reconciled_at
                            await self.session.commit()
                            return MCPToolInvokeResponse(
                                tool_name=tool_name,
                                is_error=False,
                                text=recon_res.text,
                                structured_content={
                                    "reconciled": True,
                                    "operation_id": str(op_record.id),
                                    "idempotency_key": effective_idempotency_key,
                                },
                                error_message=None,
                            )
                        else:
                            # Lookup returned no record: could be in-flight or eventual consistency delay
                            # Retain RECONCILIATION_REQUIRED; do NOT mark FAILED, do NOT retry!
                            op_record.status = "RECONCILIATION_REQUIRED"
                            op_record.reconciliation_attempts = 1
                            op_record.last_reconciled_at = datetime.now(timezone.utc)
                            await self.session.commit()
                            raise HTTPException(
                                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                detail=(
                                    f"Operation '{effective_idempotency_key}' timed out after dispatch. "
                                    "Immediate reconciliation returned no record (eventual consistency or still in-flight). "
                                    "Outcome is uncertain (RECONCILIATION_REQUIRED). Blind retry prevented."
                                ),
                            )
                except HTTPException:
                    raise
                except Exception as recon_err:
                    op_record.status = "RECONCILIATION_REQUIRED"
                    op_record.reconciliation_attempts = 1
                    op_record.last_reconciled_at = datetime.now(timezone.utc)
                    await self.session.commit()
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=(
                            f"Operation '{effective_idempotency_key}' timed out after dispatch. "
                            f"Reconciliation check failed ({type(recon_err).__name__}). "
                            "Outcome is uncertain (RECONCILIATION_REQUIRED). Blind retry prevented."
                        ),
                    )
            else:
                # No automated reconciliation mechanism available for this tool
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=(
                        f"Operation '{effective_idempotency_key}' timed out after dispatch. "
                        "Outcome is UNKNOWN. Remote tool provides no automated reconciliation mechanism. "
                        "Blind retry prevented."
                    ),
                )

        except (MCPAuthenticationError, MCPSSRFError) as pre_dispatch_err:
            op_record.status = "FAILED"
            op_record.error_message = str(pre_dispatch_err)
            op_record.updated_at = datetime.now(timezone.utc)
            await self.session.commit()
            if isinstance(pre_dispatch_err, MCPAuthenticationError):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(pre_dispatch_err))
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(pre_dispatch_err))
        except HTTPException:
            raise
        except Exception as exc:
            op_record.status = "UNKNOWN"
            op_record.error_message = f"Remote invocation encountered error: {exc}"
            op_record.updated_at = datetime.now(timezone.utc)
            await self.session.commit()
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    async def reconcile_operation(
        self,
        user: User,
        workspace_id: uuid.UUID,
        operation_id: uuid.UUID,
        manual_resolution: Optional[str] = None,
    ) -> MCPToolOperationRecord:
        """
        Reconciles an UNKNOWN or RECONCILIATION_REQUIRED mutating operation.
        Supports automated reconciliation using the configured tool policy,
        or authorized manual resolution by Workspace Owners / Admins.
        """
        ws, role = await self._check_workspace_access(user, workspace_id, require_write=True)

        stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.id == operation_id,
            MCPToolOperationRecord.workspace_id == workspace_id,
        )
        op_record = await self.session.scalar(stmt)
        if not op_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Operation record '{operation_id}' not found in workspace '{workspace_id}'.",
            )

        if op_record.status == "CONFIRMED":
            return op_record

        # Authorized manual resolution
        if manual_resolution in ("CONFIRMED", "FAILED"):
            if role not in ("ADMIN", "OWNER"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only Workspace Owners or Admins can manually resolve uncertain operations.",
                )
            now = datetime.now(timezone.utc)
            op_record.status = manual_resolution
            op_record.last_reconciled_at = now
            op_record.updated_at = now
            if manual_resolution == "CONFIRMED":
                op_record.completed_at = now
            await self.session.commit()
            await self.session.refresh(op_record)
            return op_record

        # Automated reconciliation using connection tool policy
        conn = await self.session.get(WorkspaceMCPConnection, op_record.connection_id)
        if not conn:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Associated MCP connection not found.",
            )

        tool_policy = (conn.configuration or {}).get("tool_policies", {}).get(op_record.tool_name, {})
        reconciliation_tool = tool_policy.get("reconciliation_tool")
        if not reconciliation_tool:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tool '{op_record.tool_name}' has no reconciliation tool configured. Manual resolution required.",
            )

        allow_private = bool(conn.configuration.get("allow_private_hosts", False) and self._is_dev_or_test_env())
        auth_headers: dict[str, str] = {}
        secret = await self._resolve_credential_secret(conn, conn.endpoint_url)
        if secret:
            auth_headers["Authorization"] = f"Bearer {secret}"

        config = MCPClientConfig(
            endpoint_url=conn.endpoint_url,
            transport=MCPTransportType(conn.transport),
            auth_headers=auth_headers,
            timeout_seconds=min(float(conn.configuration.get("timeout_seconds", 10.0)), 30.0),
            read_timeout_seconds=min(float(conn.configuration.get("read_timeout_seconds", 30.0)), 60.0),
            allow_private_hosts=allow_private,
        )

        key_param = tool_policy.get("reconciliation_key_param", "id")
        lookup_val = (op_record.arguments or {}).get(key_param) or op_record.idempotency_key
        now = datetime.now(timezone.utc)
        op_record.reconciliation_attempts += 1
        op_record.last_reconciled_at = now

        try:
            async with GenericMCPClient(config, http_client_factory=self.http_client_factory) as client:
                recon_res = await client.call_tool(reconciliation_tool, {key_param: lookup_val})
                if not recon_res.is_error and recon_res.text and "not found" not in recon_res.text.lower():
                    op_record.status = "CONFIRMED"
                    op_record.result_payload = {
                        "text": recon_res.text,
                        "structured_content": recon_res.structured_content,
                    }
                    op_record.completed_at = now
                else:
                    op_record.status = "RECONCILIATION_REQUIRED"
        except Exception as exc:
            op_record.status = "RECONCILIATION_REQUIRED"
            op_record.error_message = f"Reconciliation attempt failed: {exc}"

        op_record.updated_at = now
        await self.session.commit()
        await self.session.refresh(op_record)
        return op_record

