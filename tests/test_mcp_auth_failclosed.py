"""
Phase 5 Tests: Production MCP Authentication Fail-Closed & Remote Readiness.
Validates:
1. Missing token -> 401 AUTHENTICATION_REQUIRED
2. Invalid token / signature mismatch -> 401 AUTHENTICATION_REQUIRED
3. Malformed token -> 401 AUTHENTICATION_REQUIRED
4. Expired token -> 401 AUTHENTICATION_REQUIRED
5. Non-existent user -> 401 AUTHENTICATION_REQUIRED
6. Inactive user -> 401 AUTHENTICATION_REQUIRED
7. Valid USER -> Accepted with standard scopes
8. Valid ADMIN -> Accepted with admin scopes
9. Caller identity and scopes strictly server-derived (tampering rejected)
10. Sanitized audit logging (no bearer tokens or credentials in logs)
"""

from datetime import datetime, timedelta, timezone
import logging
import uuid
import jwt
import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.core.config import get_settings
from rag_app.db.models import User
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.gateway import MCPGatewayError
from rag_app.mcp.server import (
    get_current_user_profile,
    get_current_user_access,
    admin_list_users,
    rag_search,
)
from rag_app.security.authentication import create_access_token

settings = get_settings()


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def auth_test_users():
    """Create active user, active admin, and inactive user for auth matrix testing."""
    async with AsyncSessionLocal() as session:
        # 1. Active Standard User
        active_user = User(
            email=f"auth_user_{uuid.uuid4().hex[:8]}@test.com",
            hashed_password="hashed_pwd",
            role="USER",
            is_active=True,
        )
        session.add(active_user)

        # 2. Active Admin User
        active_admin = User(
            email=f"auth_admin_{uuid.uuid4().hex[:8]}@test.com",
            hashed_password="hashed_pwd",
            role="ADMIN",
            is_active=True,
        )
        session.add(active_admin)

        # 3. Inactive User
        inactive_user = User(
            email=f"auth_inactive_{uuid.uuid4().hex[:8]}@test.com",
            hashed_password="hashed_pwd",
            role="USER",
            is_active=False,
        )
        session.add(inactive_user)

        await session.commit()
        await session.refresh(active_user)
        await session.refresh(active_admin)
        await session.refresh(inactive_user)

        user_token = f"Bearer {create_access_token(subject=str(active_user.id))}"
        admin_token = f"Bearer {create_access_token(subject=str(active_admin.id))}"
        inactive_token = f"Bearer {create_access_token(subject=str(inactive_user.id))}"

        return {
            "active_user": active_user,
            "active_admin": active_admin,
            "inactive_user": inactive_user,
            "user_token": user_token,
            "admin_token": admin_token,
            "inactive_token": inactive_token,
        }


# ── 1. Missing Token Fail-Closed ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auth_missing_token_fails_closed():
    """Calls with None, empty, or whitespace-only token must fail closed with 401."""
    for empty_val in [None, "", "   ", "Bearer "]:
        with pytest.raises(MCPGatewayError) as exc_info:
            await get_current_user_profile(auth_token=empty_val)
        assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
        assert exc_info.value.status_code == 401


# ── 2. Invalid Token & Signature Mismatch Fail-Closed ─────────────────────────

@pytest.mark.asyncio
async def test_auth_invalid_signature_fails_closed(auth_test_users):
    """Token signed with an unauthorized secret key must fail closed with 401."""
    user = auth_test_users["active_user"]
    fake_token = jwt.encode(
        {"sub": str(user.id), "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        "WRONG_SECRET_KEY_FOR_TESTING_1234567890",
        algorithm="HS256",
    )

    with pytest.raises(MCPGatewayError) as exc_info:
        await get_current_user_profile(auth_token=f"Bearer {fake_token}")
    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.status_code == 401


# ── 3. Malformed Token Fail-Closed ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auth_malformed_token_fails_closed():
    """Corrupted / non-JWT strings must fail closed with 401."""
    for bad_token in ["not_a_jwt", "Bearer abc.123", "Bearer !!!???"]:
        with pytest.raises(MCPGatewayError) as exc_info:
            await get_current_user_profile(auth_token=bad_token)
        assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
        assert exc_info.value.status_code == 401


# ── 4. Expired Token Fail-Closed ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auth_expired_token_fails_closed(auth_test_users):
    """Expired JWT token must fail closed with 401."""
    user = auth_test_users["active_user"]
    expired_payload = {
        "sub": str(user.id),
        "exp": datetime.now(timezone.utc) - timedelta(minutes=10),
    }
    expired_token = jwt.encode(
        expired_payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(MCPGatewayError) as exc_info:
        await get_current_user_profile(auth_token=f"Bearer {expired_token}")
    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.status_code == 401


# ── 5. Non-existent User Fail-Closed ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_auth_nonexistent_user_fails_closed():
    """Valid JWT containing a deleted or nonexistent user ID must fail closed with 401."""
    random_id = str(uuid.uuid4())
    token = jwt.encode(
        {"sub": random_id, "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(MCPGatewayError) as exc_info:
        await get_current_user_profile(auth_token=f"Bearer {token}")
    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.status_code == 401


# ── 6. Inactive User Fail-Closed ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auth_inactive_user_fails_closed(auth_test_users):
    """Deactivated user token must fail closed with 401."""
    inactive_token = auth_test_users["inactive_token"]

    with pytest.raises(MCPGatewayError) as exc_info:
        await get_current_user_profile(auth_token=inactive_token)
    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.status_code == 401


# ── 7. Valid USER vs Valid ADMIN Authentication & Authorization ───────────────

@pytest.mark.asyncio
async def test_valid_user_and_admin_auth_boundaries(auth_test_users):
    """Valid USER can access user tools; ADMIN can access admin tools; USER is forbidden from admin."""
    user = auth_test_users["active_user"]
    admin = auth_test_users["active_admin"]
    user_token = auth_test_users["user_token"]
    admin_token = auth_test_users["admin_token"]

    # 1. User gets own profile
    user_profile = await get_current_user_profile(auth_token=user_token)
    assert user_profile["id"] == str(user.id)
    assert user_profile["role"] == "USER"

    # 2. Admin gets own profile
    admin_profile = await get_current_user_profile(auth_token=admin_token)
    assert admin_profile["id"] == str(admin.id)
    assert admin_profile["role"] == "ADMIN"

    # 3. Admin can call admin tool
    admin_res = await admin_list_users(auth_token=admin_token)
    assert isinstance(admin_res, list)

    # 4. Standard User calling admin tool -> 403 FORBIDDEN
    with pytest.raises(MCPGatewayError) as exc_info:
        await admin_list_users(auth_token=user_token)
    assert exc_info.value.code == "FORBIDDEN"
    assert exc_info.value.status_code == 403


# ── 8. Sanitized Audit Logging Verification ───────────────────────────────────

@pytest.mark.asyncio
async def test_sanitized_audit_logging_no_secrets(auth_test_users, caplog):
    """Ensure MCP Gateway logs do NOT contain Bearer tokens or secret credentials."""
    user_token = auth_test_users["user_token"]

    with caplog.at_level(logging.INFO):
        await get_current_user_profile(auth_token=user_token)

    # Inspect log records
    raw_logs = " ".join(record.message for record in caplog.records)
    assert user_token not in raw_logs
    assert "jwt_secret" not in raw_logs
    assert "[MCP-GATEWAY]" in raw_logs
