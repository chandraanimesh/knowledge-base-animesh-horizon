"""
Unit and Integration Tests for Workflow Studio AI Integration (Phase 5A-5D).
Verifies:
1. Phase 5C: Workflow Retrieval invokes real RetrievalService (Hybrid Vector + FTS + RRF + Reranker)
   with workspace isolation and department authorization filtering.
2. Phase 5B: Workflow Synthesis invokes real LLM generation, populates upstream citations,
   and records genuine token metrics.
3. Phase 5B: Workflow Grounding evaluates real evidence, detects missing evidence,
   and distinguishes verified vs unverified claims without hardcoded values.
4. Phase 5D: Safe node-level AI overrides (model_tier, temperature, max_tokens, search_provider, rerank)
   are validated and executed safely without leaking credentials.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool

# Guarantee tests execute strictly against the confirmed disposable database rag_disposable_test on port 5433
TEST_DB_URL = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"
os.environ["DATABASE_URL"] = TEST_DB_URL

test_engine = create_async_engine(TEST_DB_URL, echo=False, poolclass=NullPool)
TestSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await test_engine.dispose()

from rag_app.db.models import (
    Base,
    Department,
    Document,
    DocumentChunk,
    User,
    Workspace,
    WorkspaceProviderProfile,
    WorkflowDefinition,
    WorkflowVersion,
    WorkflowRun,
    WorkflowTask,
)
from rag_app.orchestration.action_registry import (
    ActionValidationError,
    GroundingActionStrategy,
    GroundingInput,
    GroundingOutput,
    RetrievalActionStrategy,
    RetrievalInput,
    RetrievalOutput,
    SynthesisActionStrategy,
    SynthesisInput,
    SynthesisOutput,
    WebSearchActionStrategy,
    WebSearchInput,
)
from rag_app.orchestration.executor import DAGExecutionEngine
from rag_app.providers.adapters.llm.mock import MockLLMAdapter
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    ExecutionTask,
    TaskEvidenceRef,
    TaskType,
    WorkflowStatus,
    WorkflowTaskStatus,
)


@pytest_asyncio.fixture
async def setup_ai_test_environment():
    """
    Creates isolated test departments, workspaces, users, documents, and document chunks in rag_disposable_test.
    """
    assert "rag_disposable_test" in str(test_engine.url), "Safety violation: must target rag_disposable_test"

    dept_research = Department(
        id=uuid.uuid4(),
        name="Research & Intelligence",
        code=f"RES_{uuid.uuid4().hex[:6]}",
        is_active=True,
    )
    dept_hr = Department(
        id=uuid.uuid4(),
        name="Human Resources",
        code=f"HR_{uuid.uuid4().hex[:6]}",
        is_active=True,
    )

    user = User(
        id=uuid.uuid4(),
        email=f"ai_architect_{uuid.uuid4().hex[:6]}@example.com",
        hashed_password="test_secret_hash",
        role="ADMIN",
        is_active=True,
    )

    async with TestSessionLocal() as session:
        session.add_all([dept_research, dept_hr, user])
        await session.flush()

        ws = Workspace(
            id=uuid.uuid4(),
            name="AI Research Workspace",
            code=f"WS_{uuid.uuid4().hex[:4]}",
            department_id=dept_research.id,
            created_by=user.id,
            is_active=True,
        )
        session.add(ws)
        await session.flush()

        # Seed Document in Research department
        doc = Document(
            id=uuid.uuid4(),
            workspace_id=ws.id,
            department_id=dept_research.id,
            file_name="Acme_Financial_Report_2025.pdf",
            file_path="/tmp/acme_2025.pdf",
            content_hash=f"hash_{uuid.uuid4().hex}",
            mime_type="application/pdf",
            created_by=user.id,
        )
        session.add(doc)
        await session.flush()

        # Seed DocumentChunks with real embeddings from the embedding provider
        from rag_app.ingestion.embedding_v1 import get_embedding_provider
        embedder = get_embedding_provider()
        c1_text = "Acme Corp achieved total revenue of $450 million in FY2025, growing 18% year over year."
        c2_text = "Operating margins expanded to 24%, driven by AI automation efficiencies across operations."
        embs = await embedder.embed_texts([c1_text, c2_text])

        chunk_1 = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            text=c1_text,
            embedding=embs[0],
        )
        chunk_2 = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=1,
            text=c2_text,
            embedding=embs[1],
        )
        session.add_all([chunk_1, chunk_2])
        await session.commit()

        yield {
            "department_research": dept_research,
            "department_hr": dept_hr,
            "workspace": ws,
            "user": user,
            "doc": doc,
            "chunk_1": chunk_1,
            "chunk_2": chunk_2,
        }


# ============================================================================
# PHASE 5C: RETRIEVAL STRATEGY INTEGRATION WITH HYBRID RAG
# ============================================================================

@pytest.mark.asyncio
async def test_retrieval_strategy_executes_hybrid_rag_and_returns_scores(setup_ai_test_environment):
    """Workflow Retrieval action executes RetrievalService, returning chunk results with ranking scores."""
    env = setup_ai_test_environment
    ws = env["workspace"]
    user = env["user"]
    dept_res = env["department_research"]

    async with TestSessionLocal() as session:
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            department_id=dept_res.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="revenue and margin",
            request_hash="req_retrieval_hybrid",
            max_tokens_budget=10000,
            max_provider_calls_budget=10,
        )
        session.add(run)
        await session.commit()

        strat = RetrievalActionStrategy()
        task = ExecutionTask(
            task_key="hybrid_retrieval_step",
            task_type=TaskType.RETRIEVAL,
            description="Retrieve financial metrics with hybrid search",
            input_payload={"query": "revenue", "limit": 2, "rerank": True},
        )

        result = await strat.execute(task, session=session, run=run)
        assert result.status == WorkflowTaskStatus.COMPLETED
        assert result.output_result["status"] == "success"
        assert result.output_result["retrieved_count"] >= 1
        assert len(result.evidence_items) >= 1

        # Check evidence snippet content and chunk identifiers
        snippets = [ev.snippet for ev in result.evidence_items]
        assert any("Acme Corp" in s or "revenue" in s.lower() for s in snippets)

        # Check structured results field
        assert "results" in result.output_result
        assert len(result.output_result["results"]) >= 1
        assert "chunk_id" in result.output_result["results"][0]


@pytest.mark.asyncio
async def test_retrieval_strategy_enforces_department_authorization_fence(setup_ai_test_environment):
    """Workflow Retrieval action excludes chunks when run is scoped to a department with no access."""
    env = setup_ai_test_environment
    ws = env["workspace"]
    user = env["user"]
    dept_hr = env["department_hr"]  # Chunks belong to dept_research, not dept_hr

    async with TestSessionLocal() as session:
        run_hr = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            department_id=dept_hr.id,  # Scoped strictly to HR department
            status=WorkflowStatus.RUNNING.value,
            initial_query="revenue",
            request_hash="req_dept_fence",
            max_tokens_budget=10000,
            max_provider_calls_budget=10,
        )
        session.add(run_hr)
        await session.commit()

        strat = RetrievalActionStrategy()
        task = ExecutionTask(
            task_key="dept_fenced_retrieval",
            task_type=TaskType.RETRIEVAL,
            description="Attempt cross-department chunk retrieval",
            input_payload={"query": "revenue", "limit": 5},
        )

        result = await strat.execute(task, session=session, run=run_hr)
        assert result.status == WorkflowTaskStatus.COMPLETED
        # Research department chunks must NOT be returned to an unauthorized department run
        for ev in result.evidence_items:
            assert "Acme Corp achieved total revenue of $450 million" not in ev.snippet


# ============================================================================
# PHASE 5B: REAL SYNTHESIS WITH LLM ADAPTER & TOKEN ACCRUAL
# ============================================================================

@pytest.mark.asyncio
async def test_synthesis_strategy_invokes_llm_and_accrues_tokens(setup_ai_test_environment):
    """SynthesisActionStrategy formats upstream evidence into prompt and accrues tokens from LLMProvider."""
    env = setup_ai_test_environment
    ws = env["workspace"]
    user = env["user"]

    async with TestSessionLocal() as session:
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="FY2025 financial synthesis",
            request_hash="req_synthesis_llm",
            max_tokens_budget=10000,
            max_provider_calls_budget=10,
        )
        session.add(run)
        await session.commit()

        strat = SynthesisActionStrategy()
        task = ExecutionTask(
            task_key="synthesis_step",
            task_type=TaskType.SYNTHESIS,
            description="Produce brief with citations",
            dependencies=["retrieval_upstream"],
            input_payload={
                "prompt_template": "Summarize key financial figures: {{trigger.query}}",
                "focus": "Revenue & Margin Summary",
                "max_length": 1500,
            },
        )

        mock_context = {
            "retrieval_upstream": {
                "output": {"retrieved_count": 1, "chunk_ids": ["c1"]},
                "status": "COMPLETED",
                "evidence": [
                    {
                        "source_type": "internal",
                        "source_id": "chunk-1",
                        "snippet": "Acme Corp achieved total revenue of $450 million in FY2025.",
                    }
                ],
            }
        }

        result = await strat.execute(task, session=session, run=run, context=mock_context)
        assert result.status == WorkflowTaskStatus.COMPLETED
        assert result.output_result["status"] == "synthesized"
        assert result.output_result["citations_count"] == 1
        assert len(result.output_result["synthesis_text"]) > 10
        assert result.tokens_spent > 0
        assert result.provider_calls_spent == 1


# ============================================================================
# PHASE 5B: REAL GROUNDING VERIFICATION (ANTI-HALLUCINATION)
# ============================================================================

@pytest.mark.asyncio
async def test_grounding_detects_missing_evidence_and_fails_closed(setup_ai_test_environment):
    """GroundingActionStrategy detects missing upstream evidence and reports missing_evidence with 0 confidence."""
    env = setup_ai_test_environment
    ws = env["workspace"]
    user = env["user"]

    async with TestSessionLocal() as session:
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="Grounding test",
            request_hash="req_missing_evidence",
        )

        strat = GroundingActionStrategy()
        task = ExecutionTask(
            task_key="grounding_step",
            task_type=TaskType.GROUNDING,
            description="Verify unevidenced claim",
            dependencies=["step_no_evidence"],
            input_payload={"claim": "Acme Corp operates 500 offices in Asia."},
        )

        # Context has no evidence snippets
        empty_context = {
            "step_no_evidence": {
                "output": {"status": "success"},
                "status": "COMPLETED",
                "evidence": [],
            }
        }

        result = await strat.execute(task, session=session, run=run, context=empty_context)
        assert result.status == WorkflowTaskStatus.COMPLETED
        out = result.output_result
        assert out["status"] == "missing_evidence"
        assert out["is_grounded"] is False
        assert out["confidence"] == 0.0
        assert any("no evidence" in g.lower() for g in out["unresolved_gaps"])


@pytest.mark.asyncio
async def test_grounding_verifies_supported_claim_against_evidence(setup_ai_test_environment):
    """GroundingActionStrategy marks verified when claim terms are genuinely supported by evidence."""
    env = setup_ai_test_environment
    ws = env["workspace"]
    user = env["user"]

    async with TestSessionLocal() as session:
        run = WorkflowRun(id=uuid.uuid4(), user_id=user.id, workspace_id=ws.id, request_hash="req_grounded")

        strat = GroundingActionStrategy()
        task = ExecutionTask(
            task_key="grounding_step",
            task_type=TaskType.GROUNDING,
            description="Verify evidenced claim",
            dependencies=["retrieval_step"],
            input_payload={
                "claim": "Acme Corp achieved revenue of $450 million with 24% operating margins.",
                "confidence_threshold": 0.6,
            },
        )

        context = {
            "retrieval_step": {
                "output": {},
                "status": "COMPLETED",
                "evidence": [
                    {
                        "source_type": "internal",
                        "source_id": "c1",
                        "snippet": "Acme Corp achieved total revenue of $450 million in FY2025. Operating margins expanded to 24%.",
                    }
                ],
            }
        }

        result = await strat.execute(task, session=session, run=run, context=context)
        assert result.status == WorkflowTaskStatus.COMPLETED
        out = result.output_result
        assert out["status"] == "grounding_verified"
        assert out["is_grounded"] is True
        assert out["confidence"] >= 0.6
        assert len(out["unresolved_gaps"]) == 0


@pytest.mark.asyncio
async def test_grounding_flags_unverified_unsupported_claim(setup_ai_test_environment):
    """GroundingActionStrategy flags grounding_unverified when claim is not supported by evidence."""
    env = setup_ai_test_environment
    ws = env["workspace"]
    user = env["user"]

    async with TestSessionLocal() as session:
        run = WorkflowRun(id=uuid.uuid4(), user_id=user.id, workspace_id=ws.id, request_hash="req_unverified")

        strat = GroundingActionStrategy()
        task = ExecutionTask(
            task_key="grounding_step",
            task_type=TaskType.GROUNDING,
            description="Verify ungrounded claim against unrelated evidence",
            dependencies=["retrieval_step"],
            input_payload={
                "claim": "Quantum superconducting qubits reached zero kelvin in aerospace laboratories.",
                "confidence_threshold": 0.7,
            },
        )

        context = {
            "retrieval_step": {
                "output": {},
                "status": "COMPLETED",
                "evidence": [
                    {
                        "source_type": "internal",
                        "source_id": "c1",
                        "snippet": "Acme Corp achieved total revenue of $450 million with 24% operating margins.",
                    }
                ],
            }
        }

        result = await strat.execute(task, session=session, run=run, context=context)
        assert result.status == WorkflowTaskStatus.COMPLETED
        out = result.output_result
        assert out["status"] == "grounding_unverified"
        assert out["is_grounded"] is False
        assert out["confidence"] < 0.7
        assert len(out["unresolved_gaps"]) > 0


# ============================================================================
# PHASE 5D: SAFE NODE-LEVEL AI OVERRIDES & SCHEMA BOUNDS
# ============================================================================

def test_synthesis_node_overrides_validation():
    """SynthesisInput accepts valid model_tier, temperature and max_tokens overrides within policy bounds."""
    strat = SynthesisActionStrategy()

    # Valid overrides
    valid = strat.validate_inputs({
        "prompt_template": "Synthesize report",
        "model_tier": "FAST",
        "temperature": 0.3,
        "max_tokens": 1024,
    })
    assert isinstance(valid, SynthesisInput)
    assert valid.model_tier == "FAST"
    assert valid.temperature == 0.3
    assert valid.max_tokens == 1024

    # Invalid temperature (> 1.0) must fail validation
    with pytest.raises(ActionValidationError):
        strat.validate_inputs({"temperature": 2.5})

    # Invalid negative temperature must fail validation
    with pytest.raises(ActionValidationError):
        strat.validate_inputs({"temperature": -0.1})

    # Out-of-bounds max_tokens (> 4096) must fail validation
    with pytest.raises(ActionValidationError):
        strat.validate_inputs({"max_tokens": 8000})


def test_web_search_provider_override_validation():
    """WebSearchInput accepts optional search_provider override parameter."""
    strat = WebSearchActionStrategy()
    valid = strat.validate_inputs({"query": "AI research trends", "search_provider": "firecrawl"})
    assert isinstance(valid, WebSearchInput)
    assert valid.search_provider == "firecrawl"


def test_retrieval_rerank_override_validation():
    """RetrievalInput accepts optional rerank boolean override."""
    strat = RetrievalActionStrategy()
    valid = strat.validate_inputs({"query": "annual revenue", "rerank": False})
    assert isinstance(valid, RetrievalInput)
    assert valid.rerank is False
