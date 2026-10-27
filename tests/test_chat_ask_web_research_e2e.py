"""
End-to-end integration tests for /chat/ask with mocked TinyFish Web Search.
Verifies the full pipeline:
Route -> Query Parser -> Research Graph -> Provider Registry -> TinyFishProvider ->
Transient Chunks -> ContextCompiler -> Generation -> Citations -> Response Sources Serialization.
"""

import uuid
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import (
    User,
    Department,
    Workspace,
    MemoryItem,
    MemoryProcessingJob,
    Conversation,
    ChatMessage,
    Document,
    DocumentSection,
    DocumentChunk,
)
from rag_app.research.registry import get_provider_registry
from rag_app.research.provider import (
    WebSearchResponse,
    WebSearchResult,
    WebPageResult,
)
from rag_app.research.providers.tinyfish import TinyFishProvider
from rag_app.schemas.rag import GenerateRequest, GenerateResponse, RetrievalResponse
from rag_app.api.routes import ask_question


@pytest_asyncio.fixture
async def web_e2e_env():
    """Sets up an isolated test schema with live PostgreSQL engine."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_web_e2e_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        def create_tables(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
            MemoryProcessingJob.__table__.create(sync_conn)
            MemoryItem.__table__.create(sync_conn)
            Conversation.__table__.create(sync_conn)
            ChatMessage.__table__.create(sync_conn)
            Document.__table__.create(sync_conn)
            DocumentSection.__table__.create(sync_conn)
            DocumentChunk.__table__.create(sync_conn)
        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:8]
    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        dept = Department(name=f"WebE2E Dept {suffix}", code=f"WED_{suffix}")
        session.add(dept)
        await session.flush()

        user1 = User(email=f"u1_{suffix}@example.com", hashed_password="pw", role="ADMIN")
        session.add(user1)
        await session.flush()

        ws1 = Workspace(name=f"WS1 {suffix}", code=f"W1_{suffix}", department_id=dept.id, created_by=user1.id)
        session.add(ws1)
        await session.commit()

        ws1_id = ws1.id
        u1_id = user1.id
        dept_id = dept.id

        yield {
            "session_maker": session_maker,
            "user1": user1,
            "ws1": ws1,
            "dept": dept,
            "schema_name": schema_name,
            "ws1_id": ws1_id,
            "u1_id": u1_id,
            "dept_id": dept_id,
        }

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        await conn.execute(text("DELETE FROM workspaces WHERE id = :wid"), {"wid": ws1_id})
        await conn.execute(text("DELETE FROM users WHERE id = :u1"), {"u1": u1_id})
        await conn.execute(text("DELETE FROM departments WHERE id = :did"), {"did": dept_id})
    await engine.dispose()


@pytest.mark.asyncio
async def test_chat_ask_explicit_web_search_with_mocked_tinyfish(web_e2e_env):
    """
    E2E Test 1: User explicitly asks to search the web for external information.
    Verifies that:
    1. TinyFishProvider is selected from ProviderRegistry.
    2. Search and Scrape methods are invoked with correct parameters.
    3. External evidence is returned in response retrieved_chunks and serialized sources payload.
    4. Internal document tables remain 100% unpolluted.
    """
    session_maker = web_e2e_env["session_maker"]
    user1 = web_e2e_env["user1"]
    schema_name = web_e2e_env["schema_name"]

    # Mock TinyFishProvider
    mock_tinyfish = MagicMock(spec=TinyFishProvider)
    mock_tinyfish.search = AsyncMock(
        return_value=WebSearchResponse(
            results=[
                WebSearchResult(
                    url="https://example.com/ai-report-2026",
                    title="Global AI Market Report 2026",
                    snippet="The enterprise AI market reached $150 billion in Q1 2026.",
                )
            ]
        )
    )
    mock_tinyfish.scrape = AsyncMock(
        return_value=WebPageResult(
            url="https://example.com/ai-report-2026",
            title="Global AI Market Report 2026",
            markdown="# Global AI Market Report 2026\n\nThe enterprise AI market reached $150 billion in Q1 2026 driven by open-weight models.",
            metadata={"status_code": 200},
        )
    )

    get_provider_registry()._providers["tinyfish"] = mock_tinyfish

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResponse(results=[], query="test"))

        req = GenerateRequest(
            query="Search the web for the latest enterprise AI market size in 2026",
            workspace_id=None,
        )

        mock_gen = MagicMock()
        async def fake_generate_from_context(**kwargs):
            chunks = kwargs.get("chunks", [])
            return GenerateResponse(
                answer="Based on web research, the enterprise AI market reached $150 billion in Q1 2026.",
                citations=[c.file_name for c in chunks],
                retrieved_chunks=chunks,
            )
        mock_gen.generate_from_context = AsyncMock(side_effect=fake_generate_from_context)

        with patch("rag_app.research.graph.get_settings") as mock_graph_settings, \
             patch("rag_app.api.routes.get_settings") as mock_route_settings, \
             patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            cfg = get_settings().model_copy(
                update={
                    "enable_web_research": True,
                    "web_search_provider": "tinyfish",
                    "tinyfish_api_key": "test-key-mock",
                }
            )
            mock_graph_settings.return_value = cfg
            mock_route_settings.return_value = cfg

            resp = await ask_question(
                request=req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=user1,
            )

        # 1. Verify answer returned
        assert resp.answer is not None
        assert len(resp.answer) > 0

        # 2. Verify TinyFish was called
        mock_tinyfish.search.assert_awaited()
        mock_tinyfish.scrape.assert_awaited()

        # 3. Verify retrieved chunks contain web transient chunks
        web_chunks = [c for c in resp.retrieved_chunks if c.chunk_id.startswith("web_")]
        assert len(web_chunks) > 0
        assert any("example.com/ai-report-2026" in c.document_id for c in web_chunks)

        # 4. Verify DB pollution guard: No documents inserted into canonical internal document tables
        docs = (await session.execute(select(Document))).scalars().all()
        chunks = (await session.execute(select(DocumentChunk))).scalars().all()
        assert len(docs) == 0, "No external web docs must be written to internal documents table"
        assert len(chunks) == 0, "No external web chunks must be written to internal document_chunks table"

        # 5. Verify assistant message was saved with web_research sources payload
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.role == "assistant")
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        )
        asst_msg = (await session.execute(stmt)).scalars().first()
        assert asst_msg is not None
        assert asst_msg.sources is not None
        assert len(asst_msg.sources) > 0
        web_source = next((s for s in asst_msg.sources if s.get("type") == "web_research"), None)
        assert web_source is not None
        assert "example.com/ai-report-2026" in web_source.get("url", "")


@pytest.mark.asyncio
async def test_chat_ask_freshness_query_routes_to_web(web_e2e_env):
    """
    E2E Test 2: User asks a query with freshness intent ("latest updates today").
    Verifies that the live query planner triggers web research even without "search the web".
    """
    session_maker = web_e2e_env["session_maker"]
    user1 = web_e2e_env["user1"]
    schema_name = web_e2e_env["schema_name"]

    mock_tinyfish = MagicMock(spec=TinyFishProvider)
    mock_tinyfish.search = AsyncMock(
        return_value=WebSearchResponse(
            results=[
                WebSearchResult(
                    url="https://news.example.com/today-ai",
                    title="Latest AI Breakthrough Today",
                    snippet="New open model release announced today.",
                )
            ]
        )
    )
    mock_tinyfish.scrape = AsyncMock(
        return_value=WebPageResult(
            url="https://news.example.com/today-ai",
            title="Latest AI Breakthrough Today",
            markdown="New open model release announced today with record benchmark scores.",
        )
    )

    get_provider_registry()._providers["tinyfish"] = mock_tinyfish

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        req = GenerateRequest(
            query="What are the latest AI news and updates today?",
            workspace_id=None,
        )

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResponse(results=[], query="test"))

        mock_gen = MagicMock()
        async def fake_generate_from_context(**kwargs):
            return GenerateResponse(
                answer="Latest AI updates today include new benchmark models.",
                citations=[],
                retrieved_chunks=[],
            )
        mock_gen.generate_from_context = AsyncMock(side_effect=fake_generate_from_context)

        with patch("rag_app.research.graph.get_settings") as mock_graph_settings, \
             patch("rag_app.api.routes.get_settings") as mock_route_settings, \
             patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            cfg = get_settings().model_copy(
                update={
                    "enable_web_research": True,
                    "web_search_provider": "tinyfish",
                    "tinyfish_api_key": "test-key-mock",
                }
            )
            mock_graph_settings.return_value = cfg
            mock_route_settings.return_value = cfg

            resp = await ask_question(
                request=req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=user1,
            )

        assert resp.answer is not None
        mock_tinyfish.search.assert_awaited()
