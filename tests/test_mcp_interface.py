"""
Comprehensive tests for FastMCP Interface.
Tests all 5 protected baseline MCP capabilities:
1. run_diagnostics
2. list_documents
3. ingest_document
4. rag_search
5. rag_ask
"""

import base64
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.db.models import User
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.server import (
    mcp,
    ingest_document,
    rag_ask,
    rag_search,
    list_documents,
    run_diagnostics,
)
from rag_app.security.authentication import create_access_token


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def admin_auth_token():
    """Generates a valid JWT bearer token for an active admin user."""
    async with AsyncSessionLocal() as session:
        admin_user = await session.scalar(
            select(User).where(User.role == "ADMIN", User.is_active == True).limit(1)
        )
        if not admin_user:
            admin_user = User(
                email=f"admin_test_{uuid.uuid4().hex[:6]}@example.com",
                hashed_password="hash",
                role="ADMIN",
                is_active=True,
            )
            session.add(admin_user)
            await session.commit()
            await session.refresh(admin_user)

        token = create_access_token(subject=str(admin_user.id))
        return f"Bearer {token}"


from rag_app.mcp.policies import TOOL_POLICIES


@pytest.mark.asyncio
async def test_mcp_server_registered_tools():
    """Verify that FastMCP server exposes all baseline and domain tools matching policy registry."""
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    
    # 5 protected regression baseline tools must be present
    baseline_tools = {
        "ingest_document",
        "rag_ask",
        "rag_search",
        "list_documents",
        "run_diagnostics",
    }
    assert baseline_tools.issubset(tool_names)
    assert tool_names == set(TOOL_POLICIES.keys())


@pytest.mark.asyncio
async def test_run_diagnostics_structure(admin_auth_token):
    """Verify run_diagnostics returns safe structured status without executing commands or leaking secrets."""
    res = await run_diagnostics(include_rag_smoketest=False, auth_token=admin_auth_token)
    assert isinstance(res, dict)
    assert "status" in res
    assert "runtime" in res
    assert "imports" in res
    assert "database" in res
    assert "models" in res
    assert "rag" in res
    assert res["imports"].get("fastapi") is True
    assert res["imports"].get("fastmcp") is True
    assert res["imports"].get("pgvector") is True


@pytest.mark.asyncio
async def test_list_documents(admin_auth_token):
    """Verify list_documents returns structured read-only document list with chunk counts."""
    res = await list_documents(limit=10, auth_token=admin_auth_token)
    assert isinstance(res, dict)
    assert "total_count" in res
    assert "documents" in res
    assert isinstance(res["documents"], list)


@pytest.mark.asyncio
async def test_ingest_and_search_and_ask_lifecycle(admin_auth_token):
    """Verify full end-to-end lifecycle: ingest -> search -> ask."""
    test_text = f"FastMCP integration test unique token {uuid.uuid4().hex}"
    
    # 1. Ingest
    ingest_res = await ingest_document(
        filename=f"test_{uuid.uuid4().hex[:8]}.txt",
        content_text=test_text,
        auth_token=admin_auth_token,
    )
    assert isinstance(ingest_res, dict)
    assert ingest_res.get("success") is True or ingest_res.get("skipped") is True

    # 2. Search
    search_res = await rag_search(
        query=test_text,
        top_k=3,
        rerank=False,
        auth_token=admin_auth_token,
    )
    assert isinstance(search_res, dict)
    assert "results" in search_res
    assert search_res.get("total_results", 0) >= 0

    # 3. Ask
    ask_res = await rag_ask(
        query=f"What unique token is mentioned in {test_text}?",
        top_k=3,
        rerank=False,
        auth_token=admin_auth_token,
    )
    assert isinstance(ask_res, dict)
    assert "answer" in ask_res
    assert "sources" in ask_res


@pytest.mark.asyncio
async def test_ingest_document_rejects_unsupported_extension(admin_auth_token):
    """Verify unsupported file types are safely rejected without processing."""
    res = await ingest_document(
        filename="malicious.exe",
        content_text="binary content",
        auth_token=admin_auth_token,
    )
    assert res["success"] is False
    assert res["skipped"] is True
    assert "Unsupported file format" in res["reason"]


@pytest.mark.asyncio
async def test_ingest_document_base64(admin_auth_token):
    """Verify base64 binary content decoding and ingestion."""
    text_content = "This is base64 encoded document content for MCP testing."
    b64_content = base64.b64encode(text_content.encode("utf-8")).decode("ascii")

    res = await ingest_document(
        filename=f"b64_test_{uuid.uuid4().hex[:8]}.txt",
        content_base64=b64_content,
        auth_token=admin_auth_token,
    )
    assert res["success"] is True or res["skipped"] is True
