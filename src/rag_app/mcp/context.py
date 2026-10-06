"""
Request Context and Authentication Resolution for MCP Gateway.
Ensures caller identity is cryptographically resolved from Bearer tokens
and never accepted from client-controlled tool arguments.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.core.config import get_settings
from rag_app.core.logging import new_request_id, set_request_id
from rag_app.db.models import User

logger = logging.getLogger(__name__)
settings = get_settings()

ALL_ADMIN_SCOPES = frozenset([
    "identity.read",
    "rag.read",
    "rag.write",
    "conversation.read",
    "conversation.write",
    "workspace.read",
    "workspace.write",
    "access.read",
    "access.write",
    "access.admin",
    "admin.users",
    "admin.permissions",
    "admin.documents",
    "admin.departments",
    "diagnostics",
])

STANDARD_USER_SCOPES = frozenset([
    "identity.read",
    "rag.read",
    "rag.write",
    "conversation.read",
    "conversation.write",
    "workspace.read",
    "workspace.write",
    "access.read",
    "access.write",
    "diagnostics",
])


class MCPAuthenticationError(Exception):
    """Raised when authentication credentials are missing, expired, or invalid."""
    pass


class MCPForbiddenError(Exception):
    """Raised when caller lacks required MCP scope or administrative privilege."""
    pass


@dataclass(frozen=True)
class MCPRequestContext:
    request_id: str
    user: User
    scopes: frozenset[str]
    client_id: Optional[str] = None


async def resolve_mcp_request_context(
    session: AsyncSession,
    auth_token: Optional[str] = None,
    request_id: Optional[str] = None,
) -> MCPRequestContext:
    """
    Deterministically resolves caller identity from incoming JWT token.
    Enforces that unauthenticated or invalid requests strictly fail closed with MCPAuthenticationError.
    """
    req_id = request_id or new_request_id()
    set_request_id(req_id)

    if not auth_token or not auth_token.strip():
        # Strict Fail-Closed: Missing token
        raise MCPAuthenticationError("Authentication required. Please provide a valid Bearer auth_token.")

    try:
        token_str = auth_token.strip()
        if token_str.lower().startswith("bearer "):
            token_str = token_str[7:].strip()

        if not token_str:
            raise MCPAuthenticationError("Bearer token is empty.")

        payload = jwt.decode(
            token_str,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise MCPAuthenticationError("JWT token missing 'sub' claim.")

        user = await session.get(User, uuid.UUID(user_id_str))
        if not user:
            raise MCPAuthenticationError("User account does not exist.")
        if not user.is_active:
            raise MCPAuthenticationError("User account is inactive.")

    except (jwt.ExpiredSignatureError,) as exp_exc:
        logger.warning(f"MCP authentication failure (token expired): {exp_exc}")
        raise MCPAuthenticationError("Authentication token has expired.")
    except (jwt.PyJWTError, ValueError, TypeError) as exc:
        logger.warning(f"MCP authentication failure: {exc}")
        raise MCPAuthenticationError(f"Authentication failed: {exc}")

    # Compute authorized scopes based on verified database role
    scopes = ALL_ADMIN_SCOPES if user.role == "ADMIN" else STANDARD_USER_SCOPES

    return MCPRequestContext(
        request_id=req_id,
        user=user,
        scopes=scopes,
    )
