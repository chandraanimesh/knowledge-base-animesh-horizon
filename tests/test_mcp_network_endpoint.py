"""
Automated tests for network-mounted MCP endpoint and header-based authentication.
Verifies:
1. GET /mcp returns 200 discovery response with SSE URL.
2. Lifespan properly initializes without TaskGroup RuntimeError.
3. Header-based authorization works seamlessly for external clients (Twenty CRM).
"""

import pytest
import uuid
from fastapi.testclient import TestClient
from sqlalchemy import select

from rag_app.db.models import User
from rag_app.db.session import AsyncSessionLocal
from rag_app.main import app
from rag_app.mcp.gateway import mcp_gateway
from rag_app.security.authentication import create_access_token


from rag_app.db.session import AsyncSessionLocal, engine

@pytest.fixture(autouse=True)
def cleanup_engine():
    yield
    engine.sync_engine.dispose()

@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client


def test_mcp_discovery_endpoints(test_client):
    """Verify both /mcp and /mcp/ return 200 discovery metadata for local MCP clients."""
    for path in ["/mcp", "/mcp/"]:
        resp = test_client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["transport"] == "sse"
        assert data["sse_endpoint"] == "/mcp/sse"
        assert "/mcp/sse" in data["local_base_url"]
        assert data["tools_count"] >= 50


@pytest.mark.asyncio
async def test_header_based_auth_fallback():
    """Verify MCP Gateway resolves identity directly from incoming token."""
    async with AsyncSessionLocal() as session:
        user = await session.scalar(
            select(User).where(User.role == "ADMIN", User.is_active == True).limit(1)
        )
        if not user:
            user = User(
                id=uuid.uuid4(),
                email=f"mcp_net_{uuid.uuid4().hex[:6]}@example.com",
                hashed_password="hash",
                role="ADMIN",
                is_active=True,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

        token = create_access_token(subject=str(user.id))
        bearer_token = f"Bearer {token}"

    # Handler expecting auth_token receives the resolved token
    async def sample_handler(auth_token=None):
        return {"caller_token_present": bool(auth_token)}

    res = await mcp_gateway.execute(
        tool_name="list_documents",
        handler=sample_handler,
        auth_token=bearer_token,
    )
    assert res["caller_token_present"] is True
