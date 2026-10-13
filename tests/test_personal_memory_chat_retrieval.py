import uuid
import warnings
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from sqlalchemy import text, select
from sqlalchemy.exc import SAWarning
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import User, Department, Workspace, MemoryItem, Conversation, ChatMessage, Document, DocumentSection, DocumentChunk
from rag_app.memory.schemas import MemoryCandidate
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.retrieval.sufficiency import EvidenceSufficiencyEvaluator, is_personal_memory_relevant
from rag_app.schemas.research import SufficiencyStatus
from rag_app.schemas.rag import CompanyFilters, GenerateRequest, GenerateResponse, RetrievalResponse
from rag_app.api.routes import ask_question
from rag_app.retrieval.service import RetrievalService


# ==============================================================================
# UNIT REGRESSION TESTS: Evidence Sufficiency Evaluation
# ==============================================================================

def test_sufficiency_personal_memory_satisfies_non_doc_query():
    """
    Personal memory context alone satisfies non-document questions when relevant,
    answering as previously saved personal context.
    """
    evaluator = EvidenceSufficiencyEvaluator()
    query = "what do u know about nexora solutions?"
    chunks = []
    conv_history = [
        {
            "role": "system",
            "content": "[User Personal Preferences & Context]\n- [DYNAMIC] Nexora Solutions is our prospective enterprise client we met last week",
            "name": "personal_memory",
        }
    ]

    res = evaluator.evaluate_deterministic(
        query=query,
        chunks=chunks,
        conversation_history=conv_history,
    )
    assert res.status == SufficiencyStatus.SUFFICIENT
    assert res.is_sufficient is True
    assert res.evidence_found is True
    assert res.answerable_from_context is True
    assert "personal_memory" in res.evidence_types
    assert "previously saved personal context" in res.reason.lower()


def test_sufficiency_unrelated_personal_memory_rejected():
    """
    If saved personal memory does not cover the query topic, sufficiency rejects it.
    """
    evaluator = EvidenceSufficiencyEvaluator()
    query = "what do u know about acme corp?"
    chunks = []
    conv_history = [
        {
            "role": "system",
            "content": "[User Personal Preferences & Context]\n- [DYNAMIC] Nexora Solutions is our prospective client",
            "name": "personal_memory",
        }
    ]

    res = evaluator.evaluate_deterministic(
        query=query,
        chunks=chunks,
        conversation_history=conv_history,
    )
    assert res.status == SufficiencyStatus.INSUFFICIENT
    assert res.is_sufficient is False
    assert res.evidence_found is False


def test_sufficiency_preserves_strict_document_facts_requirement():
    """
    Questions genuinely requiring authoritative documents (e.g. contracts, policies)
    cannot be satisfied by personal memory alone when document evidence is missing.
    """
    evaluator = EvidenceSufficiencyEvaluator()
    chunks = []
    conv_history = [
        {
            "role": "system",
            "content": "[User Personal Preferences & Context]\n- [DYNAMIC] Nexora Solutions is our prospective client",
            "name": "personal_memory",
        }
    ]

    # 1. Asking for contract document
    res_contract = evaluator.evaluate_deterministic(
        query="what does the contract document say about Nexora Solutions?",
        chunks=chunks,
        conversation_history=conv_history,
    )
    assert res_contract.status == SufficiencyStatus.INSUFFICIENT
    assert "document query" in res_contract.reason
    assert "cannot satisfy document facts" in res_contract.reason

    # 2. Asking for company policy
    res_policy = evaluator.evaluate_deterministic(
        query="show me the official travel reimbursement policy file",
        chunks=chunks,
        conversation_history=conv_history,
    )
    assert res_policy.status == SufficiencyStatus.INSUFFICIENT
    assert "document query" in res_policy.reason


def test_is_personal_memory_relevant_keyword_and_profile_matching():
    """Verifies precision of memory relevance detection."""
    mem_text = "[User Personal Preferences & Context]\n- [DYNAMIC] Nexora Solutions is an AI startup in Berlin\n- [STATIC] I prefer concise bullet points"

    # Positive matches
    assert is_personal_memory_relevant("what do u know about nexora solutions?", mem_text) is True
    assert is_personal_memory_relevant("tell me about nexora", mem_text) is True
    assert is_personal_memory_relevant("what are my preferences?", mem_text) is True
    assert is_personal_memory_relevant("how do i prefer answers?", mem_text) is True

    # Negative matches
    assert is_personal_memory_relevant("what is the weather in Tokyo?", mem_text) is False
    assert is_personal_memory_relevant("what do you know about acme corp?", mem_text) is False
    assert is_personal_memory_relevant("", mem_text) is False
    assert is_personal_memory_relevant("anything", "") is False


# ==============================================================================
# DATABASE & END-TO-END FIXTURES
# ==============================================================================

import pytest_asyncio

@pytest_asyncio.fixture
async def chat_mem_env():
    """Sets up an isolated test schema with live PostgreSQL engine."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_chat_mem_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        def create_tables(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
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
        dept = Department(name=f"ChatMem Dept {suffix}", code=f"CMD_{suffix}")
        session.add(dept)
        await session.flush()

        user1 = User(email=f"u1_{suffix}@example.com", hashed_password="pw", role="USER")
        user2 = User(email=f"u2_{suffix}@example.com", hashed_password="pw", role="USER")
        session.add_all([user1, user2])
        await session.flush()

        ws1 = Workspace(name=f"WS1 {suffix}", code=f"W1_{suffix}", department_id=dept.id, created_by=user1.id)
        session.add(ws1)
        await session.commit()

        ws1_id = ws1.id
        u1_id = user1.id
        u2_id = user2.id
        dept_id = dept.id

        yield {
            "session_maker": session_maker,
            "user1": user1,
            "user2": user2,
            "ws1": ws1,
            "dept": dept,
            "schema_name": schema_name,
            "ws1_id": ws1_id,
            "u1_id": u1_id,
            "u2_id": u2_id,
            "dept_id": dept_id,
        }

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        await conn.execute(
            text("DELETE FROM workspaces WHERE id = :wid"),
            {"wid": ws1_id},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:u1, :u2)"),
            {"u1": u1_id, "u2": u2_id},
        )
        await conn.execute(
            text("DELETE FROM departments WHERE id = :did"),
            {"did": dept_id},
        )
    await engine.dispose()


# ==============================================================================
# INTEGRATION & FLOW REGRESSION TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_chat_ask_personal_memory_lifecycle_and_cross_user_isolation(chat_mem_env: dict):
    """
    Regression test covering:
    1. User 1 saves memory via /rem: 'Nexora Solutions is our prospective client'
    2. User 1 asks 'what do u know about nexora solutions?'
       -> Graph evaluates sufficiency from personal memory, generates answer labelling personal context.
    3. User 1 asks for authoritative document facts 'show me the contract document for Nexora Solutions'
       -> Insufficient document evidence, rejects cleanly.
    4. User 1 asks about unrelated topic 'what do u know about acme corp?'
       -> Insufficient document evidence, rejects cleanly.
    5. User 2 asks 'what do u know about nexora solutions?'
       -> Cross-user isolation: User 2 has zero memories, rejects cleanly without leaking User 1's memory.
    """
    session_maker = chat_mem_env["session_maker"]
    user1: User = chat_mem_env["user1"]
    user2: User = chat_mem_env["user2"]
    user1.role = "ADMIN"
    user2.role = "ADMIN"
    schema_name = chat_mem_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        mock_retrieval_service = MagicMock()
        mock_retrieval_response = RetrievalResponse(
            results=[],
            query="test",
            strategy_trace={},
            chunk_provenance={},
            chunk_hierarchy={},
        )
        mock_retrieval_service.retrieve = AsyncMock(return_value=mock_retrieval_response)

        mock_gen_service = MagicMock()
        async def fake_generate_from_context(**kwargs):
            return GenerateResponse(
                answer="Based on your previously saved personal context, Nexora Solutions is our prospective enterprise client we met last week. (Note: this is from personal memory, not verified company documents).",
                citations=[],
                retrieved_chunks=[],
            )
        mock_gen_service.generate_from_context = AsyncMock(side_effect=fake_generate_from_context)

        # ----------------------------------------------------------------------
        # Step 1: User 1 saves memory via /rem
        # ----------------------------------------------------------------------
        save_req = GenerateRequest(
            query="/rem Nexora Solutions is our prospective enterprise client we met last week",
            workspace_id=None,
        )
        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            save_resp = await ask_question(
                request=save_req,
                service=mock_gen_service,
                retrieval_service=mock_retrieval_service,
                session=session,
                user=user1,
            )
        assert "Memory saved" in save_resp.answer
        assert "Nexora Solutions" in save_resp.answer
        conv_id = save_resp.conversation_id

        # ----------------------------------------------------------------------
        # Step 2: User 1 asks "what do u know about nexora solutions?"
        # ----------------------------------------------------------------------
        query_req = GenerateRequest(
            query="what do u know about nexora solutions?",
            conversation_id=uuid.UUID(conv_id),
            workspace_id=None,
        )
        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            query_resp = await ask_question(
                request=query_req,
                service=mock_gen_service,
                retrieval_service=mock_retrieval_service,
                session=session,
                user=user1,
            )
        # Successfully answered from personal memory
        assert "Nexora Solutions" in query_resp.answer
        assert "previously saved personal context" in query_resp.answer
        assert len(query_resp.citations) == 0
        assert len(query_resp.retrieved_chunks) == 0

        # ----------------------------------------------------------------------
        # Step 3: User 1 asks for authoritative document facts
        # ----------------------------------------------------------------------
        doc_req = GenerateRequest(
            query="show me the contract document for Nexora Solutions",
            conversation_id=uuid.UUID(conv_id),
            workspace_id=None,
        )
        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            doc_resp = await ask_question(
                request=doc_req,
                service=mock_gen_service,
                retrieval_service=mock_retrieval_service,
                session=session,
                user=user1,
            )
        # Must fail with insufficient document evidence
        assert "could not verify" in doc_resp.answer.lower()

        # ----------------------------------------------------------------------
        # Step 4: User 1 asks about unrelated topic (not in memory or documents)
        # ----------------------------------------------------------------------
        unrelated_req = GenerateRequest(
            query="what do u know about acme corp?",
            conversation_id=uuid.UUID(conv_id),
            workspace_id=None,
        )
        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            unrelated_resp = await ask_question(
                request=unrelated_req,
                service=mock_gen_service,
                retrieval_service=mock_retrieval_service,
                session=session,
                user=user1,
            )
        assert "could not verify" in unrelated_resp.answer.lower()

        # ----------------------------------------------------------------------
        # Step 5: User 2 asks about Nexora Solutions (Cross-user isolation)
        # ----------------------------------------------------------------------
        u2_req = GenerateRequest(
            query="what do u know about nexora solutions?",
            workspace_id=None,
        )
        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            u2_resp = await ask_question(
                request=u2_req,
                service=mock_gen_service,
                retrieval_service=mock_retrieval_service,
                session=session,
                user=user2,
            )
        # User 2 has zero access to User 1's memory -> cannot verify
        assert "could not verify" in u2_resp.answer.lower()
        assert "Nexora" not in u2_resp.answer


@pytest.mark.asyncio
async def test_hierarchical_search_cartesian_product_regression(chat_mem_env: dict):
    """
    Verifies that _hierarchical_search with CompanyFilters executes cleanly
    without triggering SQLAlchemy SAWarning for Cartesian products.
    """
    session_maker = chat_mem_env["session_maker"]
    schema_name = chat_mem_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        service = RetrievalService(session)

        # Catch all SQLAlchemy warnings as errors
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=SAWarning)
            filters = CompanyFilters(company_id="COMP_TEST_123")
            # Should not raise SAWarning about Cartesian product
            results = await service._hierarchical_search(
                query="quarterly roadmap",
                candidate_k=5,
                filters=filters,
            )
            assert isinstance(results, list)
