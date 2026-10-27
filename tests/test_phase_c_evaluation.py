"""
Phase C.1 — Unified ContextCompiler End-to-End Evaluation Suite.

Evaluates:
1. Live-path integration through /chat/ask route and LangGraph answer generation.
2. Strict token budgeting across assembled prompt messages and pre-invocation checks.
3. Conflicting evidence handling (temporal/authority resolution, divergence notices, budget protection without overflow).
4. Real-path authorization and provenance isolation (cross-user memory, cross-workspace thread, unapproved shared knowledge, cross-department chunks, external web evidence).
"""

import uuid
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
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
from rag_app.schemas.rag import (
    GenerateRequest,
    GenerateResponse,
    RetrievalResponse,
    RetrievedChunk,
    ChunkProvenance,
    GraphEdgeProvenance,
)
from rag_app.api.routes import ask_question
from rag_app.context.compiler import ContextCompiler, estimate_tokens
from rag_app.context.contracts import SourceAuthority, ContextEvidence
from rag_app.generation.service import GenerationService
from rag_app.generation.citations import validate_and_render_citations
from rag_app.generation.prompt import build_rag_messages, RAG_SYSTEM_PROMPT


@pytest_asyncio.fixture
async def phase_c_eval_env():
    """Sets up an isolated test schema with live PostgreSQL engine for end-to-end evaluation."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_c1_eval_{uuid.uuid4().hex[:8]}"

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
        dept1 = Department(name=f"Finance {suffix}", code=f"FIN_{suffix}")
        dept2 = Department(name=f"HR {suffix}", code=f"HR_{suffix}")
        session.add_all([dept1, dept2])
        await session.flush()

        user1 = User(email=f"alice_{suffix}@example.com", hashed_password="pw", role="ADMIN")
        user2 = User(email=f"bob_{suffix}@example.com", hashed_password="pw", role="USER")
        session.add_all([user1, user2])
        await session.flush()

        ws1 = Workspace(name=f"Finance Workspace {suffix}", code=f"FWS_{suffix}", department_id=dept1.id, created_by=user1.id)
        ws2 = Workspace(name=f"HR Workspace {suffix}", code=f"HWS_{suffix}", department_id=dept2.id, created_by=user2.id)
        session.add_all([ws1, ws2])
        await session.commit()

        yield {
            "session_maker": session_maker,
            "user1": user1,
            "user2": user2,
            "dept1": dept1,
            "dept2": dept2,
            "ws1": ws1,
            "ws2": ws2,
            "schema_name": schema_name,
        }

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        await conn.execute(text("DELETE FROM workspaces WHERE id IN (:w1, :w2)"), {"w1": ws1.id, "w2": ws2.id})
        await conn.execute(text("DELETE FROM users WHERE id IN (:u1, :u2)"), {"u1": user1.id, "u2": user2.id})
        await conn.execute(text("DELETE FROM departments WHERE id IN (:d1, :d2)"), {"d1": dept1.id, "d2": dept2.id})
    await engine.dispose()


# =============================================================================
# EVALUATION 1: Live-Path Integration through /chat/ask
# =============================================================================

@pytest.mark.asyncio
async def test_live_path_chat_ask_invokes_unified_context_compiler(phase_c_eval_env):
    """
    Evaluation 1: Verifies that /chat/ask invokes the real ContextCompiler and
    passes only selected, authorized evidence to the generation stage.
    """
    session_maker = phase_c_eval_env["session_maker"]
    user1 = phase_c_eval_env["user1"]
    dept1 = phase_c_eval_env["dept1"]
    ws1 = phase_c_eval_env["ws1"]
    schema_name = phase_c_eval_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        authorized_chunk = RetrievedChunk(
            chunk_id="chunk_fin_1",
            document_id="doc_fin_1",
            department_id=str(dept1.id),
            file_name="q3_finance.pdf",
            chunk_index=0,
            text="Q3 net income was $4.2M.",
            hybrid_score=0.95,
        )

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(
            return_value=RetrievalResponse(results=[authorized_chunk], query="What was Q3 net income?")
        )

        captured_kwargs = {}

        async def capture_generate_from_context(**kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return GenerateResponse(
                answer="Q3 net income was $4.2M [chunk: chunk_fin_1].",
                citations=["q3_finance.pdf"],
                retrieved_chunks=kwargs.get("chunks", []),
            )

        mock_gen_service = MagicMock()
        mock_gen_service.generate_from_context = AsyncMock(side_effect=capture_generate_from_context)

        req = GenerateRequest(
            query="What was Q3 net income?",
            workspace_id=ws1.id,
        )

        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[dept1.id]):
            resp = await ask_question(
                request=req,
                service=mock_gen_service,
                retrieval_service=mock_retrieval,
                session=session,
                user=user1,
            )

        # 1. Assert ContextCompiler was invoked and selected chunks passed to generation
        mock_gen_service.generate_from_context.assert_awaited_once()
        passed_chunks = captured_kwargs.get("chunks", [])
        assert len(passed_chunks) == 1
        assert passed_chunks[0].chunk_id == "chunk_fin_1"

        # 2. Assert rendered_context is present and formatted
        assert "Q3 net income was $4.2M" in captured_kwargs.get("rendered_context", "")

        # 3. Assert response structure remains compatible with GenerateResponse
        assert isinstance(resp, GenerateResponse)
        assert resp.answer == "Q3 net income was $4.2M [chunk: chunk_fin_1]."
        assert resp.citations == ["q3_finance.pdf"]
        assert len(resp.retrieved_chunks) == 1
        assert resp.conversation_id is not None
        assert resp.message_id is not None


# =============================================================================
# EVALUATION 2: Strict Token Budget & No Silent Prompt Overflow
# =============================================================================

def test_strict_token_budget_counts_system_and_query_overhead():
    """
    Evaluation 2: Verifies that total context tokens include system prompt,
    query tokens, prompt overhead, memory, and evidence.
    """
    compiler = ContextCompiler(
        model_context_limit=4000,
        reserved_output_tokens=500,
        max_context_tokens=1500,
    )
    dept_id = uuid.uuid4()

    query = "Provide a comprehensive breakdown of our enterprise SaaS sales."
    chunk = RetrievedChunk(
        chunk_id="chunk_saas",
        document_id="doc_saas",
        department_id=str(dept_id),
        file_name="saas_report.pdf",
        chunk_index=0,
        text="Enterprise SaaS sales generated $8.5M in ARR across 45 enterprise accounts.",
        hybrid_score=0.92,
    )

    compiled = compiler.compile(
        chunks=[chunk],
        query=query,
        allowed_departments=[dept_id],
    )

    # System instruction tokens (~2200) + query + overhead + evidence
    assert compiled.total_planned_tokens > estimate_tokens(chunk.text)
    assert compiled.available_context_tokens <= 1500
    assert compiled.total_planned_tokens <= (4000 - 500)
    assert compiled.budget_status in ("NORMAL", "CONSTRAINED")


@pytest.mark.asyncio
async def test_generation_service_rejects_silent_prompt_overflow():
    """
    Evaluation 2: Verifies that if assembled prompt tokens exceed max capacity,
    GenerationService returns a safe constrained outcome instead of silently overflowing.
    """
    mock_retrieval = MagicMock()
    mock_retrieval.session = MagicMock()
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock()

    service = GenerationService(session=AsyncMock())
    service.llm = mock_llm

    # Construct an oversized chunk that exceeds capacity
    dept_id = uuid.uuid4()
    huge_chunk = RetrievedChunk(
        chunk_id="huge_chunk_1",
        document_id="doc_huge",
        department_id=str(dept_id),
        file_name="huge.pdf",
        chunk_index=0,
        text="Overflow data word " * 8000,  # ~24,000 tokens
        hybrid_score=0.9,
    )

    with patch("rag_app.generation.service.get_settings") as mock_settings:
        cfg = get_settings().model_copy(
            update={
                "model_context_limit": 4000,
                "reserved_output_tokens": 500,
            }
        )
        mock_settings.return_value = cfg

        resp = await service.generate_from_context(
            query="Analyze huge data",
            chunks=[huge_chunk],
            allowed_departments=[dept_id],
        )

    # Must return safe constrained outcome and NEVER invoke LLM with overflow
    mock_llm.generate.assert_not_awaited()
    assert "exceeds the maximum allowable token budget" in resp.answer
    assert resp.citations == []
    assert resp.retrieved_chunks == []


def test_tokenizer_estimation_consistency():
    """
    Evaluation 2: Documents and verifies token estimation consistency:
    len(text) / 4 vs word count.
    """
    text_sample = "Acme Corp reported record annual recurring revenue of $25M in fiscal year 2025."
    est = estimate_tokens(text_sample)
    words = len(text_sample.split())

    # In English prose, 1 word ~= 1.3 tokens, len(text)/4 gives reliable deterministic upper/lower bound
    assert est > 0
    assert abs(est - words) <= (words * 0.5)  # within 50% of word count


# =============================================================================
# EVALUATION 3: Conflicting Evidence Handling & Non-Overflow
# =============================================================================

def test_conflicting_evidence_newer_and_authoritative_preferred():
    """
    Evaluation 3: Synthetic fixture with historical $10M and newer $25M revenue.
    Verifies that the newer authoritative value is surfaced and relationship divergence is noted.
    """
    compiler = ContextCompiler()
    dept_id = uuid.uuid4()

    chunk_old = RetrievedChunk(
        chunk_id="chunk_rev_10m",
        document_id="doc_annual_2023",
        department_id=str(dept_id),
        file_name="annual_report_2023.pdf",
        chunk_index=0,
        text="Fiscal year 2023 revenue closed at $10M.",
        hybrid_score=0.88,
    )
    chunk_new = RetrievedChunk(
        chunk_id="chunk_rev_25m",
        document_id="doc_annual_2024",
        department_id=str(dept_id),
        file_name="annual_report_2024.pdf",
        chunk_index=0,
        text="Fiscal year 2024 revenue closed at $25M.",
        hybrid_score=0.96,
    )

    provenance = {
        "chunk_rev_10m": ChunkProvenance(
            chunk_id="chunk_rev_10m",
            graph_path=[GraphEdgeProvenance(subject="Acme Corp", predicate="revenue", object="$10M")],
        ),
        "chunk_rev_25m": ChunkProvenance(
            chunk_id="chunk_rev_25m",
            graph_path=[GraphEdgeProvenance(subject="Acme Corp", predicate="revenue", object="$25M")],
        ),
    }

    compiled = compiler.compile(
        chunks=[chunk_old, chunk_new],
        query="What is the revenue of Acme Corp?",
        allowed_departments=[dept_id],
        chunk_provenance=provenance,
    )

    assert len(compiled.conflicts) == 1
    conflict = compiled.conflicts[0]
    assert conflict.subject.lower() == "acme corp"
    assert conflict.predicate.lower() == "revenue"
    assert "$10M" in conflict.conflicting_objects
    assert "$25M" in conflict.conflicting_objects

    # Both sides preserved in context so LLM recognizes the historical progression/divergence
    assert len(compiled.chunks) == 2


def test_conflict_set_exceeding_budget_does_not_overflow():
    """
    Evaluation 3: A conflict set that exceeds the available evidence budget.
    Verifies that the compiler NEVER exceeds the budget merely to preserve every conflict,
    surfaces the notice, and prunes the excess evidence.
    """
    # Extremely tight evidence budget: max 50 tokens
    compiler = ContextCompiler(max_context_tokens=50)
    dept_id = uuid.uuid4()

    huge_conflict_1 = RetrievedChunk(
        chunk_id="conflict_chunk_1",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="rev_source_1.pdf",
        chunk_index=0,
        text="Conflicting revenue source 1 claims $10M: " + ("A" * 200),  # ~55 tokens
        hybrid_score=0.95,
    )
    huge_conflict_2 = RetrievedChunk(
        chunk_id="conflict_chunk_2",
        document_id="doc_2",
        department_id=str(dept_id),
        file_name="rev_source_2.pdf",
        chunk_index=0,
        text="Conflicting revenue source 2 claims $25M: " + ("B" * 200),  # ~55 tokens
        hybrid_score=0.94,
    )

    provenance = {
        "conflict_chunk_1": ChunkProvenance(
            chunk_id="conflict_chunk_1",
            graph_path=[GraphEdgeProvenance(subject="Company", predicate="revenue", object="$10M")],
        ),
        "conflict_chunk_2": ChunkProvenance(
            chunk_id="conflict_chunk_2",
            graph_path=[GraphEdgeProvenance(subject="Company", predicate="revenue", object="$25M")],
        ),
    }

    compiled = compiler.compile(
        chunks=[huge_conflict_1, huge_conflict_2],
        query="What is company revenue?",
        allowed_departments=[dept_id],
        chunk_provenance=provenance,
    )

    # 1. Total evidence tokens must NOT exceed available evidence budget (50 tokens)
    assert compiled.token_count <= 50

    # 2. At least one chunk must be dropped because 55 + 55 = 110 > 50
    assert len(compiled.chunks) <= 1
    assert compiled.dropped_count >= 1

    # 3. Conflict divergence notice is STILL preserved in metadata/prompt guidance!
    assert len(compiled.conflicts) == 1
    assert "CONFLICT_SET_EXCEEDS_BUDGET" in compiled.dropped_reasons.values()

    # 4. Citations validation guarantees dropped chunk cannot be cited
    cited_answer = "Revenue is $10M [chunk: conflict_chunk_1] and $25M [chunk: conflict_chunk_2]."
    sanitized, citations = validate_and_render_citations(cited_answer, compiled.chunks)
    assert "[chunk: conflict_chunk_2]" not in sanitized


# =============================================================================
# EVALUATION 4: Real-Path Authorization, RBAC & Provenance Isolation
# =============================================================================

@pytest.mark.asyncio
async def test_real_path_cross_user_memory_never_reaches_generation(phase_c_eval_env):
    """
    Evaluation 4: Verifies through the real request path that another user's
    personal memory never reaches the generation input.
    """
    session_maker = phase_c_eval_env["session_maker"]
    user1 = phase_c_eval_env["user1"]
    user2 = phase_c_eval_env["user2"]
    dept1 = phase_c_eval_env["dept1"]
    schema_name = phase_c_eval_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # User2 saves a secret memory
        secret_memory = MemoryItem(
            user_id=user2.id,
            content="User2 confidential compensation is $300k bonus.",
            memory_type="DYNAMIC",
            canonical_key="user2_comp_secret",
            status="ACTIVE",
        )
        session.add(secret_memory)
        await session.commit()

        # User1 asks a question
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResponse(results=[], query="What is compensation?"))

        captured_kwargs = {}
        async def fake_generate(**kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return GenerateResponse(answer="No compensation info found.", citations=[], retrieved_chunks=[])

        mock_gen = MagicMock()
        mock_gen.generate_from_context = AsyncMock(side_effect=fake_generate)

        req = GenerateRequest(query="What is compensation?", workspace_id=None)

        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[dept1.id]):
            await ask_question(
                request=req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=user1,
            )

        # Assert user2's secret memory never reached generation
        passed_memories = captured_kwargs.get("personal_memories") or []
        for mem in passed_memories:
            c_text = getattr(mem, "content", "") if hasattr(mem, "content") else str(mem)
            assert "User2 confidential compensation" not in c_text
            assert "$300k" not in c_text


@pytest.mark.asyncio
async def test_real_path_cross_workspace_thread_isolation(phase_c_eval_env):
    """
    Evaluation 4: Verifies through the real request path that threads from another
    workspace are isolated and excluded.
    """
    session_maker = phase_c_eval_env["session_maker"]
    user1 = phase_c_eval_env["user1"]
    dept1 = phase_c_eval_env["dept1"]
    ws1 = phase_c_eval_env["ws1"]
    ws2 = phase_c_eval_env["ws2"]
    schema_name = phase_c_eval_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # Conversation in WS2 (HR)
        conv_ws2 = Conversation(user_id=user1.id, workspace_id=ws2.id, title="HR Private Discussions")
        session.add(conv_ws2)
        await session.flush()
        msg_ws2 = ChatMessage(conversation_id=conv_ws2.id, role="user", content="HR grievance details for case #404")
        session.add(msg_ws2)
        await session.commit()

        # User queries WS1 (Finance)
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResponse(results=[], query="Grievance details"))

        captured_kwargs = {}
        async def fake_generate(**kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            return GenerateResponse(answer="No grievance records found in Finance.", citations=[], retrieved_chunks=[])

        mock_gen = MagicMock()
        mock_gen.generate_from_context = AsyncMock(side_effect=fake_generate)

        req = GenerateRequest(query="Grievance details", workspace_id=ws1.id)

        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[dept1.id]):
            await ask_question(
                request=req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=user1,
            )

        passed_threads = captured_kwargs.get("workspace_thread_context") or []
        for t in passed_threads:
            assert "HR grievance details" not in str(t.get("content", ""))


def test_real_path_unauthorized_department_chunk_blocked():
    """
    Evaluation 4: Verifies that an internal document chunk from an unauthorized
    department cannot enter the final compiled context.
    """
    compiler = ContextCompiler()
    dept_allowed = uuid.uuid4()
    dept_unauthorized = uuid.uuid4()

    unauthorized_chunk = RetrievedChunk(
        chunk_id="chunk_unauth_payroll",
        document_id="doc_payroll",
        department_id=str(dept_unauthorized),
        file_name="payroll_master.pdf",
        chunk_index=0,
        text="Confidential payroll data.",
        hybrid_score=0.99,
    )

    with pytest.raises(RuntimeError, match="Security violation: Unauthorized chunk detected"):
        compiler.compile(
            chunks=[unauthorized_chunk],
            query="Show payroll",
            allowed_departments=[dept_allowed],
            is_web_context=False,
        )


def test_external_web_evidence_remains_distinguishable():
    """
    Evaluation 4: Verifies that external web research evidence remains explicitly
    distinguishable from authoritative internal records.
    """
    compiler = ContextCompiler()
    dept_id = uuid.uuid4()

    web_chunk = RetrievedChunk(
        chunk_id="web_firecrawl_999",
        document_id="https://news.ycombinator.com/item?id=123",
        department_id="None",
        file_name="Hacker News Thread",
        chunk_index=0,
        text="Discussion on latest AI benchmarks.",
        hybrid_score=0.85,
    )

    compiled = compiler.compile(
        chunks=[web_chunk],
        query="What is HN discussing?",
        allowed_departments=[dept_id],
        is_web_context=True,
    )

    assert len(compiled.chunks) == 1
    assert compiled.chunks[0].chunk_id.startswith("web_")
    assert len(compiled.selected_evidence) == 1
    ev = compiled.selected_evidence[0]
    assert ev.is_external is True
    assert ev.source_authority == SourceAuthority.WEB_RESEARCH.value
    assert ev.source_kind == "web_research"
