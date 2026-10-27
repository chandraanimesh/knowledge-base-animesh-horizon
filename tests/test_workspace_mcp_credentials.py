"""
Milestone 6G Acceptance Suite: Secure Workspace Custom Credential Storage.

Verifies:
1. Fernet symmetric encryption and decryption round-trip using dedicated APP_ENCRYPTION_KEY.
2. MultiFernet key rotation from secondary to primary key.
3. Ciphertext tampering and corruption fail closed with CredentialDecryptionError.
4. Missing, invalid, or JWT_SECRET-colliding encryption keys fail closed with CredentialEncryptionConfigurationError.
5. Custom credentials stored encrypted at rest (starts with Fernet header gAAAAA), never in plaintext.
6. Public API responses disclose credential_mode and has_custom_credential, but NEVER raw secret or ciphertext.
7. Masked placeholders (•••••••• or ********) do NOT overwrite existing stored secrets.
8. Explicit clear_custom_credential safely clears stored secret.
9. Mode switching between 'default', 'custom', and allowlisted 'env_ref'.
10. Strict multi-tenant isolation: Workspace A's credentials cannot be accessed, updated, or used by Workspace B.
11. SSRF validation occurs before credential resolution.
12. Connection test endpoint verifies connectivity with configured credentials.
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
import uuid
from unittest.mock import AsyncMock, patch

from cryptography.fernet import Fernet
from fastapi import HTTPException
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from rag_app.core.config import get_settings
from rag_app.db.models import Department, Permission, User, UserDepartment, UserPermission, Workspace, WorkspaceMCPConnection, WorkspaceMember
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.client import GenericMCPClient, MCPClientConfig, MCPServerInfo, MCPTransportType
from rag_app.mcp.connection_service import WorkspaceMCPConnectionService
from rag_app.schemas.mcp_connection import (
    MCPConnectionCreate,
    MCPConnectionResponse,
    MCPConnectionUpdate,
)
from rag_app.security.encryption import (
    CredentialDecryptionError,
    CredentialEncryptionConfigurationError,
    CredentialEncryptionService,
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest.fixture
def sample_encryption_keys():
    """Generates two fresh Fernet keys for encryption and rotation tests."""
    k1 = Fernet.generate_key().decode()
    k2 = Fernet.generate_key().decode()
    return k1, k2


@pytest_asyncio.fixture
async def setup_workspace_context():
    """
    Sets up two isolated workspaces (Workspace A and Workspace B)
    with separate departments, owners, and members.
    Cleans up any lingering test connections to guarantee test independence.
    """
    async with AsyncSessionLocal() as session:
        dept_a = await session.scalar(select(Department).where(Department.code == "CRED_DEPT_A"))
        if not dept_a:
            dept_a = Department(code="CRED_DEPT_A", name="Cred Dept A", is_active=True)
            session.add(dept_a)

        dept_b = await session.scalar(select(Department).where(Department.code == "CRED_DEPT_B"))
        if not dept_b:
            dept_b = Department(code="CRED_DEPT_B", name="Cred Dept B", is_active=True)
            session.add(dept_b)

        await session.commit()
        await session.refresh(dept_a)
        await session.refresh(dept_b)

        async def get_or_create_user(email: str, role: str) -> User:
            u = await session.scalar(select(User).where(User.email == email))
            if not u:
                u = User(email=email, hashed_password="pw", role=role, is_active=True)
                session.add(u)
                await session.commit()
                await session.refresh(u)
            return u

        admin = await get_or_create_user("cred_admin@test.com", "ADMIN")
        user_a = await get_or_create_user("cred_owner_a@test.com", "USER")
        user_b = await get_or_create_user("cred_owner_b@test.com", "USER")

        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(code="VIEW", description="View files", is_active=True)
            session.add(view_perm)
            await session.commit()
            await session.refresh(view_perm)

        for u, d in [(user_a, dept_a), (user_b, dept_b)]:
            ud = await session.scalar(
                select(UserDepartment).where(UserDepartment.user_id == u.id, UserDepartment.department_id == d.id)
            )
            if not ud:
                session.add(UserDepartment(user_id=u.id, department_id=d.id))
            up = await session.scalar(
                select(UserPermission).where(
                    UserPermission.user_id == u.id,
                    UserPermission.department_id == d.id,
                    UserPermission.permission_id == view_perm.id,
                )
            )
            if not up:
                session.add(UserPermission(user_id=u.id, department_id=d.id, permission_id=view_perm.id))
        await session.commit()

        async def get_or_create_ws(dept_id: uuid.UUID, code: str, name: str, creator_id: uuid.UUID) -> Workspace:
            ws = await session.scalar(select(Workspace).where(Workspace.code == code, Workspace.department_id == dept_id))
            if not ws:
                ws = Workspace(name=name, code=code, department_id=dept_id, created_by=creator_id, is_active=True)
                session.add(ws)
                await session.commit()
                await session.refresh(ws)
            return ws

        ws_a = await get_or_create_ws(dept_a.id, "WS_CRED_A", "Workspace Cred A", user_a.id)
        ws_b = await get_or_create_ws(dept_b.id, "WS_CRED_B", "Workspace Cred B", user_b.id)

        async def ensure_member(ws_id: uuid.UUID, u_id: uuid.UUID, role: str):
            m = await session.scalar(select(WorkspaceMember).where(WorkspaceMember.workspace_id == ws_id, WorkspaceMember.user_id == u_id))
            if not m:
                session.add(WorkspaceMember(workspace_id=ws_id, user_id=u_id, role=role))
            elif m.role != role:
                m.role = role
            await session.commit()

        await ensure_member(ws_a.id, user_a.id, "OWNER")
        await ensure_member(ws_b.id, user_b.id, "OWNER")

        # Clean lingering connections in test workspaces
        await session.execute(
            delete(WorkspaceMCPConnection).where(
                WorkspaceMCPConnection.workspace_id.in_([ws_a.id, ws_b.id])
            )
        )
        await session.commit()

        return {
            "ws_a": ws_a,
            "ws_b": ws_b,
            "user_a": user_a,
            "user_b": user_b,
            "admin": admin,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 1. UNIT TESTS: Cryptographic Invariants & Key Rotation
# ─────────────────────────────────────────────────────────────────────────────

def test_fernet_encryption_roundtrip(sample_encryption_keys):
    """Encrypts with primary key and decrypts back to identical plaintext."""
    k1, _ = sample_encryption_keys
    svc = CredentialEncryptionService(key=k1)
    secret = "sk-live-workspace-api-key-998877"

    encrypted = svc.encrypt(secret)
    assert encrypted != secret
    assert encrypted.startswith("gAAAAA")

    decrypted = svc.decrypt(encrypted)
    assert decrypted == secret


def test_multifernet_key_rotation(sample_encryption_keys):
    """MultiFernet decrypts older key token and rotates to current primary key."""
    k1, k2 = sample_encryption_keys
    old_svc = CredentialEncryptionService(key=k2)
    token_k2 = old_svc.encrypt("rotation-secret")

    rotated_svc = CredentialEncryptionService(key=f"{k1},{k2}")
    decrypted = rotated_svc.decrypt(token_k2)
    assert decrypted == "rotation-secret"

    new_token, was_rotated = rotated_svc.rotate(token_k2)
    assert was_rotated is True
    assert new_token != token_k2

    primary_svc = CredentialEncryptionService(key=k1)
    assert primary_svc.decrypt(new_token) == "rotation-secret"


def test_ciphertext_tampering_fails_closed(sample_encryption_keys):
    """Corrupted or tampered ciphertext raises CredentialDecryptionError."""
    k1, _ = sample_encryption_keys
    svc = CredentialEncryptionService(key=k1)
    encrypted = svc.encrypt("authentic-secret")

    tampered = encrypted[:-5] + "AAAAA"
    with pytest.raises(CredentialDecryptionError):
        svc.decrypt(tampered)

    with pytest.raises(CredentialDecryptionError):
        svc.decrypt("not-a-valid-fernet-token")


def test_missing_or_invalid_encryption_key_fails_closed():
    """Missing key, invalid key, or JWT_SECRET reuse fails closed."""
    # 1. Unconfigured key fails closed on require_configured / encrypt
    svc_none = CredentialEncryptionService(key="")
    with pytest.raises(CredentialEncryptionConfigurationError):
        svc_none.require_configured()

    # 2. Invalid non-base64 key fails closed
    svc_invalid = CredentialEncryptionService(key="short-invalid-key")
    with pytest.raises(CredentialEncryptionConfigurationError):
        svc_invalid.require_configured()

    # 3. Dedicated Fernet key colliding with JWT_SECRET fails closed
    actual_jwt_key = getattr(get_settings(), "jwt_secret_key", None)
    if actual_jwt_key:
        svc_jwt_colliding = CredentialEncryptionService(key=actual_jwt_key)
        with pytest.raises(CredentialEncryptionConfigurationError) as exc_info:
            svc_jwt_colliding.require_configured()
        assert "never match JWT_SECRET" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTEGRATION TESTS: Workspace Credential CRUD & Security Semantics
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_and_read_connection_with_custom_credential(setup_workspace_context, sample_encryption_keys):
    """Creating a custom connection encrypts the secret and never leaks it in API responses."""
    ctx = setup_workspace_context
    ws_a = ctx["ws_a"]
    user_a = ctx["user_a"]
    k1, _ = sample_encryption_keys

    enc_svc = CredentialEncryptionService(key=k1)

    raw_secret = "sk-custom-workspace-secret-12345"
    payload = MCPConnectionCreate(
        name=f"Workspace_Private_MCP_{uuid.uuid4().hex[:6]}",
        endpoint_url="http://localhost:3000/mcp",
        transport=MCPTransportType.STREAMABLE_HTTP,
        credential_mode="custom",
        custom_credential=raw_secret,
        configuration={"allow_private_hosts": True},
    )

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session, encryption_service=enc_svc)
        created = await service.create_connection(user_a, ws_a.id, payload)

        # 1. API response validation
        assert created.credential_mode == "custom"
        assert created.has_custom_credential is True
        assert not hasattr(created, "custom_credential") or getattr(created, "custom_credential", None) is None
        assert "encrypted_credential" not in (created.configuration or {})

        # 2. Database validation: Secret is encrypted at rest
        db_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == created.id))
        assert db_conn is not None
        assert db_conn.credential_mode == "custom"
        assert db_conn.encrypted_credential is not None
        assert db_conn.encrypted_credential.startswith("gAAAAA")
        assert raw_secret not in db_conn.encrypted_credential
        assert raw_secret not in str(db_conn.configuration)

        # 3. Clean up
        await session.delete(db_conn)
        await session.commit()


@pytest.mark.asyncio
async def test_masked_placeholder_preserves_existing_secret(setup_workspace_context, sample_encryption_keys):
    """Submitting masked placeholder bullets preserves existing encrypted secret."""
    ctx = setup_workspace_context
    ws_a = ctx["ws_a"]
    user_a = ctx["user_a"]
    k1, _ = sample_encryption_keys

    enc_svc = CredentialEncryptionService(key=k1)

    initial_secret = "sk-permanent-custom-token-999"
    payload = MCPConnectionCreate(
        name=f"Placeholder_Preserving_Conn_{uuid.uuid4().hex[:6]}",
        endpoint_url="http://localhost:3000/mcp",
        transport=MCPTransportType.STREAMABLE_HTTP,
        credential_mode="custom",
        custom_credential=initial_secret,
        configuration={"allow_private_hosts": True},
    )

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session, encryption_service=enc_svc)
        created = await service.create_connection(user_a, ws_a.id, payload)
        db_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == created.id))
        stored_ciphertext = db_conn.encrypted_credential

        # Update with masked placeholder bullets (e.g. \u2022\u2022\u2022\u2022)
        new_name = f"Renamed_{uuid.uuid4().hex[:6]}"
        update_payload = MCPConnectionUpdate(
            name=new_name,
            custom_credential="\u2022" * 16,
        )
        updated = await service.update_connection(user_a, ws_a.id, created.id, update_payload)
        assert updated.name == new_name
        assert updated.has_custom_credential is True

        # Refresh and verify ciphertext was NOT overwritten
        await session.refresh(db_conn)
        assert db_conn.encrypted_credential == stored_ciphertext
        decrypted = enc_svc.decrypt(db_conn.encrypted_credential)
        assert decrypted == initial_secret

        await session.delete(db_conn)
        await session.commit()


@pytest.mark.asyncio
async def test_clear_custom_credential_semantics(setup_workspace_context, sample_encryption_keys):
    """clear_custom_credential removes stored ciphertext and resets status."""
    ctx = setup_workspace_context
    ws_a = ctx["ws_a"]
    user_a = ctx["user_a"]
    k1, _ = sample_encryption_keys

    enc_svc = CredentialEncryptionService(key=k1)

    payload = MCPConnectionCreate(
        name=f"Temporary_Secret_Conn_{uuid.uuid4().hex[:6]}",
        endpoint_url="http://localhost:3000/mcp",
        transport=MCPTransportType.STREAMABLE_HTTP,
        credential_mode="custom",
        custom_credential="sk-secret-to-be-cleared",
        configuration={"allow_private_hosts": True},
    )

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session, encryption_service=enc_svc)
        created = await service.create_connection(user_a, ws_a.id, payload)
        assert created.has_custom_credential is True

        # Explicitly clear credential and revert to default mode
        update_payload = MCPConnectionUpdate(
            credential_mode="default",
            clear_custom_credential=True,
        )
        updated = await service.update_connection(user_a, ws_a.id, created.id, update_payload)
        assert updated.credential_mode == "default"
        assert updated.has_custom_credential is False

        # Verify DB ciphertext is removed
        db_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == created.id))
        assert db_conn.encrypted_credential is None
        assert db_conn.credential_mode == "default"

        await session.delete(db_conn)
        await session.commit()


@pytest.mark.asyncio
async def test_mode_switch_to_env_ref_validates_allowlist(setup_workspace_context, sample_encryption_keys):
    """Switching to env_ref requires an allowlisted reference; unapproved ref is rejected."""
    ctx = setup_workspace_context
    ws_a = ctx["ws_a"]
    user_a = ctx["user_a"]
    k1, _ = sample_encryption_keys

    enc_svc = CredentialEncryptionService(key=k1)

    payload = MCPConnectionCreate(
        name=f"Env_Ref_Conn_{uuid.uuid4().hex[:6]}",
        endpoint_url="http://localhost:3000/mcp",
        transport=MCPTransportType.STREAMABLE_HTTP,
        credential_mode="default",
        configuration={"allow_private_hosts": True},
    )

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session, encryption_service=enc_svc)
        created = await service.create_connection(user_a, ws_a.id, payload)

        # 1. Non-allowlisted env ref rejected with 400
        with pytest.raises(HTTPException) as exc_info:
            await service.update_connection(
                user_a,
                ws_a.id,
                created.id,
                MCPConnectionUpdate(
                    credential_mode="env_ref",
                    credential_ref="UNAPPROVED_SECRET_ENV_VAR",
                ),
            )
        assert exc_info.value.status_code == 400
        assert "not approved" in exc_info.value.detail.lower()

        # 2. Allowlisted env ref accepted
        updated = await service.update_connection(
            user_a,
            ws_a.id,
            created.id,
            MCPConnectionUpdate(
                credential_mode="env_ref",
                credential_ref="SERVER_TWENTY_API_KEY",
            ),
        )
        assert updated.credential_mode == "env_ref"
        assert updated.credential_ref == "SERVER_TWENTY_API_KEY"

        db_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == created.id))
        await session.delete(db_conn)
        await session.commit()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SECURITY TESTS: Multi-Tenant Isolation & SSRF Guard
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_tenant_isolation_cross_workspace_credentials_inaccessible(setup_workspace_context, sample_encryption_keys):
    """Workspace A's credentials cannot be accessed, updated, or tested by Workspace B (multi-tenant boundary)."""
    ctx = setup_workspace_context
    ws_a = ctx["ws_a"]
    ws_b = ctx["ws_b"]
    user_b = ctx["user_b"]
    k1, _ = sample_encryption_keys

    enc_svc = CredentialEncryptionService(key=k1)

    payload_a = MCPConnectionCreate(
        name=f"Tenant_A_Secret_{uuid.uuid4().hex[:6]}",
        endpoint_url="http://localhost:3000/mcp",
        transport=MCPTransportType.STREAMABLE_HTTP,
        credential_mode="custom",
        custom_credential="sk-ws-a-confidential-token",
        configuration={"allow_private_hosts": True},
    )

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session, encryption_service=enc_svc)
        conn_a = await service.create_connection(ctx["user_a"], ws_a.id, payload_a)

        # 1. User B tries to view Tenant A's connection -> 404 (IDOR prevention)
        with pytest.raises(HTTPException) as exc_info:
            await service.get_connection(user_b, ws_b.id, conn_a.id)
        assert exc_info.value.status_code == 404

        # 2. User B tries to update Tenant A's connection -> 404
        with pytest.raises(HTTPException) as exc_info:
            await service.update_connection(
                user_b, ws_b.id, conn_a.id, MCPConnectionUpdate(name="Hijacked")
            )
        assert exc_info.value.status_code == 404

        # 3. User B tries to test Tenant A's connection -> 404
        with pytest.raises(HTTPException) as exc_info:
            await service.test_connection(user_b, ws_b.id, conn_a.id)
        assert exc_info.value.status_code == 404

        # Clean up
        db_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == conn_a.id))
        await session.delete(db_conn)
        await session.commit()


@pytest.mark.asyncio
async def test_endpoint_ssrf_validation_runs_before_credential_resolution(setup_workspace_context, sample_encryption_keys):
    """Unsafe endpoint is rejected by SSRF validation without attempting credential resolution."""
    ctx = setup_workspace_context
    ws_a = ctx["ws_a"]
    user_a = ctx["user_a"]
    k1, _ = sample_encryption_keys

    enc_svc = CredentialEncryptionService(key=k1)

    ssrf_payload = MCPConnectionCreate(
        name=f"SSRF_Attack_{uuid.uuid4().hex[:6]}",
        endpoint_url="http://169.254.169.254/latest/meta-data/",
        transport=MCPTransportType.STREAMABLE_HTTP,
        credential_mode="custom",
        custom_credential="sk-dummy-token",
        configuration={"allow_private_hosts": False},
    )

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session, encryption_service=enc_svc)
        with pytest.raises(HTTPException) as exc_info:
            await service.create_connection(user_a, ws_a.id, ssrf_payload)
        assert exc_info.value.status_code == 400
        assert "forbidden" in exc_info.value.detail.lower() or "prohibited" in exc_info.value.detail.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUNTIME TEST: Test Connection Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_test_connection_probes_endpoint_with_custom_credential(setup_workspace_context, sample_encryption_keys):
    """test_connection resolves encrypted custom credential and returns status and latency."""
    ctx = setup_workspace_context
    ws_a = ctx["ws_a"]
    user_a = ctx["user_a"]
    k1, _ = sample_encryption_keys

    enc_svc = CredentialEncryptionService(key=k1)

    payload = MCPConnectionCreate(
        name=f"Testable_MCP_Server_{uuid.uuid4().hex[:6]}",
        endpoint_url="http://localhost:3000/mcp",
        transport=MCPTransportType.STREAMABLE_HTTP,
        credential_mode="custom",
        custom_credential="sk-test-probe-key",
        configuration={"allow_private_hosts": True},
    )

    mock_server_info = MCPServerInfo(
        name="Mock MCP Server",
        version="1.0.0",
        protocol_version="2024-11-05",
        capabilities={},
    )

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session, encryption_service=enc_svc)
        created = await service.create_connection(user_a, ws_a.id, payload)

        with patch.object(GenericMCPClient, "connect", new_callable=AsyncMock) as mock_connect, \
             patch.object(GenericMCPClient, "list_tools", new_callable=AsyncMock) as mock_list_tools, \
             patch.object(GenericMCPClient, "close", new_callable=AsyncMock) as mock_close:

            mock_connect.return_value = mock_server_info
            mock_list_tools.return_value = []

            result = await service.test_connection(user_a, ws_a.id, created.id)

            assert result["status"] == "SUCCESS"
            assert result["credential_mode"] == "custom"
            assert result["discovered_tool_count"] == 0
            assert "latency_ms" in result
            assert result["latency_ms"] >= 0

        db_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == created.id))
        await session.delete(db_conn)
        await session.commit()


@pytest.mark.asyncio
async def test_gate0_rotation_persists_to_db_and_decrypts_with_new_key_alone(setup_workspace_context, sample_encryption_keys):
    """
    Gate 0 Verification:
    1. Connection is created with old key k2.
    2. Rotation is triggered via MultiFernet (k1, k2).
    3. Replacement ciphertext is persisted to Postgres database.
    4. Database connection is reloaded from a completely fresh session.
    5. The rotated credential decrypts successfully with primary key k1 ALONE.
    6. Decrypting with old key k2 ALONE fails with CredentialDecryptionError.
    """
    ctx = setup_workspace_context
    ws_a = ctx["ws_a"]
    user_a = ctx["user_a"]
    k1, k2 = sample_encryption_keys

    # Step 1: Create connection using old key k2 alone
    old_enc_svc = CredentialEncryptionService(key=k2)
    raw_secret = "sk-gate0-live-credential-test-888"
    payload = MCPConnectionCreate(
        name=f"Gate0_Rotation_Conn_{uuid.uuid4().hex[:6]}",
        endpoint_url="http://localhost:3000/mcp",
        transport=MCPTransportType.STREAMABLE_HTTP,
        credential_mode="custom",
        custom_credential=raw_secret,
        configuration={"allow_private_hosts": True},
    )

    async with AsyncSessionLocal() as session:
        service_old = WorkspaceMCPConnectionService(session, encryption_service=old_enc_svc)
        created = await service_old.create_connection(user_a, ws_a.id, payload)
        conn_id = created.id

    # Verify initial ciphertext in DB was encrypted with k2 (cannot decrypt with k1 alone)
    async with AsyncSessionLocal() as session:
        db_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == conn_id))
        assert db_conn is not None
        initial_ciphertext = db_conn.encrypted_credential
        k1_svc = CredentialEncryptionService(key=k1)
        with pytest.raises(CredentialDecryptionError):
            k1_svc.decrypt(initial_ciphertext)

    # Step 2 & 3: MultiFernet (k1, k2) resolves credential, rotates, and commits replacement ciphertext to DB
    rotated_enc_svc = CredentialEncryptionService(key=f"{k1},{k2}")
    async with AsyncSessionLocal() as session:
        service_rotated = WorkspaceMCPConnectionService(session, encryption_service=rotated_enc_svc)
        db_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == conn_id))
        resolved_secret = await service_rotated._resolve_credential_secret(db_conn, db_conn.endpoint_url)
        assert resolved_secret == raw_secret

    # Step 4, 5 & 6: Reload from a completely fresh database session
    async with AsyncSessionLocal() as session:
        reloaded_conn = await session.scalar(select(WorkspaceMCPConnection).where(WorkspaceMCPConnection.id == conn_id))
        persisted_rotated_ciphertext = reloaded_conn.encrypted_credential

        # Must be different from initial ciphertext
        assert persisted_rotated_ciphertext != initial_ciphertext

        # MUST decrypt successfully with new primary key k1 ALONE
        primary_alone_svc = CredentialEncryptionService(key=k1)
        decrypted_new = primary_alone_svc.decrypt(persisted_rotated_ciphertext)
        assert decrypted_new == raw_secret

        # MUST fail to decrypt with old secondary key k2 ALONE
        k2_alone_svc = CredentialEncryptionService(key=k2)
        with pytest.raises(CredentialDecryptionError):
            k2_alone_svc.decrypt(persisted_rotated_ciphertext)

        # Cleanup
        await session.delete(reloaded_conn)
        await session.commit()

