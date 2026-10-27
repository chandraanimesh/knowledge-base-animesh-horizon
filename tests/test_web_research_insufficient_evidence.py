"""
Regression test suite for Web Research Triggering After Insufficient Internal Evidence.

Covers:
1. A query explicitly requesting web search when web research is enabled.
2. A generic query with insufficient internal evidence and web fallback permitted (generic regression case).
3. A query sufficiently answered by internal evidence, where unnecessary web research should not run.
4. Web research disabled, where the system must not claim that it searched the web.
5. Web provider failure, with safe fallback behavior.
6. State propagation from sufficiency evaluation to graph routing.
7. External evidence provenance and citation validation.
8. Existing RBAC and context-budget protections remaining intact.
"""

import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from rag_app.core.config import Settings
from rag_app.db.models import User, Document, DocumentChunk
from rag_app.schemas.rag import (
    QueryPlan,
    RetrievedChunk,
    GenerateResponse,
    ChunkHierarchyMetadata,
)
from rag_app.schemas.research import (
    SufficiencyEvaluation,
    SufficiencyStatus,
)
from rag_app.research.provider import (
    WebSearchResponse,
    WebSearchResult,
    WebPageResult,
    ProviderUnavailableError,
)
from rag_app.research.providers.tinyfish import TinyFishProvider
from rag_app.research.registry import ProviderRegistry
from rag_app.research.graph import (
    ResearchState,
    classify_query,
    decide_next_after_eval,
    init_research_run,
    search_web,
    ingest_into_rag_node,
    generate_final_answer,
    no_context,
)
from rag_app.retrieval.sufficiency import EvidenceSufficiencyEvaluator
from rag_app.generation.citations import validate_and_render_citations
from rag_app.context.compiler import ContextCompiler


def make_dummy_user() -> User:
    user = User()
    user.id = uuid.uuid4()
    user.email = "test_user@example.com"
    user.role = "USER"
    return user


def build_base_state(
    query: str = "top 10 restaurant in ahemadabad",
    plan: QueryPlan | None = None,
    chunks: list[RetrievedChunk] | None = None,
    is_sufficient: bool = False,
    sufficiency_status: str = "insufficient",
    suff_eval: SufficiencyEvaluation | None = None,
) -> ResearchState:
    user = make_dummy_user()
    dept_id = uuid.uuid4()
    ws_id = uuid.uuid4()

    if plan is None:
        plan = QueryPlan(
            semantic_query=query,
            web_search_needed=False,
            web_search_reason=None,
            internal_retrieval_needed=True,
        )

    return {
        "query": query,
        "raw_query": query,
        "top_k": 5,
        "rerank": False,
        "user": user,
        "allowed_departments": [dept_id],
        "workspace_id": ws_id,
        "query_type": "knowledge",
        "query_plan": plan,
        "retrieved_chunks": chunks or [],
        "strategy_trace": {"semantic": len(chunks or [])},
        "chunk_provenance": {},
        "chunk_hierarchy": {},
        "is_sufficient": is_sufficient,
        "sufficiency_status": sufficiency_status,
        "sufficiency_evaluation": suff_eval,
        "research_run_id": None,
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": None,
        "web_research_error": None,
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": uuid.uuid4(),
    }


# ============================================================================
# 1. Query explicitly requesting web search when web research is enabled
# ============================================================================

@pytest.mark.asyncio
async def test_explicit_web_search_when_enabled():
    query = "Search the web for top restaurants in Ahmedabad"
    state: ResearchState = {
        "query": query,
        "raw_query": query,
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": None,
        "query_plan": None,
        "retrieved_chunks": [],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": None,
        "sufficiency_status": None,
        "sufficiency_evaluation": None,
        "research_run_id": None,
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": None,
        "web_research_error": None,
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
    }

    # Classification extracts explicit web intent
    classified = await classify_query(state)
    assert classified["query_type"] == "knowledge"
    plan: QueryPlan = classified["query_plan"]
    assert plan.web_search_needed is True
    assert plan.web_search_reason == "explicit_request"

    state["query_plan"] = plan
    state["is_sufficient"] = True  # Even if some internal docs exist
    state["sufficiency_status"] = "sufficient"

    with patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=True, web_search_provider="tinyfish")
        next_step = decide_next_after_eval(state)
        assert next_step == "init_research_run", "Explicit web search intent must route to research when enabled"


# ============================================================================
# 2. Generic query with insufficient internal evidence triggers web fallback
# ============================================================================

@pytest.mark.asyncio
async def test_generic_query_insufficient_evidence_triggers_web_fallback():
    # Observed query: "top 10 restaurant in ahemadabad"
    query = "top 10 restaurant in ahemadabad"

    # Retrieved internal evidence contained generic company records, not restaurants
    company_chunk = RetrievedChunk(
        chunk_id="comp_101",
        document_id=str(uuid.uuid4()),
        department_id=str(uuid.uuid4()),
        file_name="companies_1000_synthetic.csv",
        text="Company XYZ, Ahmedabad. Sector: Industrial Manufacturing. Revenue: $50M.",
        chunk_index=0,
        hybrid_score=0.3,
    )

    suff_eval = SufficiencyEvaluation(
        status=SufficiencyStatus.INSUFFICIENT,
        is_sufficient=False,
        evidence_found=False,
        confidence=0.1,
        coverage=0.0,
        reason="Retrieved evidence contained generic company records, not restaurants.",
        missing_requirements=["restaurant listings", "dining recommendations in Ahmedabad"],
    )

    state = build_base_state(
        query=query,
        chunks=[company_chunk],
        is_sufficient=False,
        sufficiency_status="insufficient",
        suff_eval=suff_eval,
    )

    with patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=True, web_search_provider="tinyfish")
        next_step = decide_next_after_eval(state)
        assert next_step == "init_research_run", "Insufficient internal evidence must trigger web research when enabled"


# ============================================================================
# 3. Query sufficiently answered by internal evidence does NOT trigger web search
# ============================================================================

def test_sufficient_internal_evidence_does_not_trigger_web_research():
    query = "What is the company remote work policy?"

    suff_eval = SufficiencyEvaluation(
        status=SufficiencyStatus.SUFFICIENT,
        is_sufficient=True,
        evidence_found=True,
        confidence=0.95,
        coverage=1.0,
        reason="The remote work policy document provides complete guidelines.",
        missing_requirements=[],
    )

    state = build_base_state(
        query=query,
        chunks=[
            RetrievedChunk(
                chunk_id="chunk_rw_1",
                document_id=str(uuid.uuid4()),
                department_id=str(uuid.uuid4()),
                file_name="hr_remote_policy.pdf",
                text="Employees are permitted up to 2 days of remote work per week.",
                chunk_index=0,
                hybrid_score=0.9,
            )
        ],
        is_sufficient=True,
        sufficiency_status="sufficient",
        suff_eval=suff_eval,
    )

    with patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=True, web_search_provider="tinyfish")
        next_step = decide_next_after_eval(state)
        assert next_step == "generate_final_answer", "Sufficient internal evidence must route directly to answer generation without web calls"


# ============================================================================
# 4. Web research disabled: system must NOT claim that it searched the web
# ============================================================================

@pytest.mark.asyncio
async def test_web_research_disabled_does_not_claim_web_search():
    query = "top 10 restaurant in ahemadabad"

    suff_eval = SufficiencyEvaluation(
        status=SufficiencyStatus.INSUFFICIENT,
        is_sufficient=False,
        evidence_found=False,
        confidence=0.1,
        coverage=0.0,
        reason="Retrieved evidence contained generic company records, not restaurants.",
        missing_requirements=["restaurants in Ahmedabad"],
    )

    state = build_base_state(
        query=query,
        chunks=[],
        is_sufficient=False,
        sufficiency_status="insufficient",
        suff_eval=suff_eval,
    )

    with patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=False, web_search_provider="tinyfish")
        next_step = decide_next_after_eval(state)
        assert next_step == "no_context", "When web research is disabled, insufficient internal evidence must route to no_context"

    # Verify no_context output does not claim external web search happened
    no_ctx_out = await no_context(state)
    resp: GenerateResponse = no_ctx_out["final_response"]
    assert "available documents" in resp.answer
    assert "web sources" not in resp.answer
    assert len(resp.retrieved_chunks) == 0


# ============================================================================
# 5. Web provider failure: safe fallback behavior without crashing
# ============================================================================

@pytest.mark.asyncio
async def test_web_provider_failure_safe_fallback():
    mock_registry = ProviderRegistry()
    mock_provider = MagicMock(spec=TinyFishProvider)
    mock_provider.search = AsyncMock(side_effect=ProviderUnavailableError("TinyFish search connection timeout"))
    mock_registry.register("tinyfish", mock_provider)

    state = build_base_state(
        query="top 10 restaurant in ahemadabad",
        is_sufficient=False,
        sufficiency_status="insufficient",
    )

    with patch("rag_app.research.graph.get_provider_registry", return_value=mock_registry), \
         patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=True, web_search_provider="tinyfish")

        # 1. Search web node handles provider failure safely and returns empty search_results
        search_out = await search_web(state)
        assert search_out == {"search_results": []}

        # Update state with search failure (no web chunks retrieved)
        state["search_results"] = search_out["search_results"]

        # 2. Final answer generation provides safe, transparent fallback without crashing
        mock_gen_service = MagicMock()
        mock_compiler = MagicMock()
        config = {
            "configurable": {
                "generation_service": mock_gen_service,
                "context_compiler": mock_compiler,
            }
        }

        final_out = await generate_final_answer(state, config)
        final_resp: GenerateResponse = final_out["final_response"]
        assert final_resp is not None
        assert "available documents or web sources" in final_resp.answer
        assert len(final_resp.citations) == 0
        assert len(final_resp.retrieved_chunks) == 0


# ============================================================================
# 6. State propagation from sufficiency evaluation to graph routing
# ============================================================================

@pytest.mark.asyncio
async def test_state_propagation_from_sufficiency_to_routing():
    query = "top 10 restaurant in ahemadabad"
    state = build_base_state(query=query)

    # Initial query plan has web_search_needed=False
    assert state["query_plan"].web_search_needed is False
    assert state["query_plan"].web_search_reason is None

    # After sufficiency determines insufficient internal evidence:
    state["is_sufficient"] = False
    state["sufficiency_status"] = "insufficient"
    state["sufficiency_evaluation"] = SufficiencyEvaluation(
        status=SufficiencyStatus.INSUFFICIENT,
        is_sufficient=False,
        confidence=0.1,
        coverage=0.0,
        missing_requirements=["restaurants"],
    )

    # decide_next_after_eval routes to init_research_run
    with patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=True, web_search_provider="tinyfish")
        route = decide_next_after_eval(state)
        assert route == "init_research_run"

    # init_research_run updates query_plan to propagate research need
    config = {"configurable": {"db_session": None}}
    run_init = await init_research_run(state, config)
    assert run_init["research_run_id"] is not None
    updated_plan: QueryPlan = run_init["query_plan"]
    assert updated_plan.web_search_needed is True
    assert updated_plan.web_search_reason == "insufficient_internal_evidence"


# ============================================================================
# 7. External evidence provenance and citation validation
# ============================================================================

@pytest.mark.asyncio
async def test_external_evidence_provenance_and_citations():
    from rag_app.schemas.research import StructuredKnowledgeCandidate, CleanWebPage, ExtractedFact

    candidate = StructuredKnowledgeCandidate(
        facts=[
            ExtractedFact(
                entity_name="Agashiye",
                entity_type="FACILITY",
                attributes={"cuisine": "Traditional Gujarati Thali"},
                department_code="GLOBAL",
                evidence_text="Agashiye is a renowned restaurant in Ahmedabad.",
                source_url="https://restaurants.example.com/agashiye",
            )
        ]
    )

    clean_pages = [
        CleanWebPage(
            url="https://restaurants.example.com/agashiye",
            title="Agashiye - Ahmedabad Heritage Dining",
            content="Agashiye is an iconic rooftop restaurant in Ahmedabad serving authentic Gujarati thalis.",
        )
    ]

    mock_session = MagicMock()
    state = build_base_state(query="top 10 restaurant in ahemadabad")
    state["research_run_id"] = str(uuid.uuid4())
    state["extracted_knowledge"] = candidate
    state["clean_pages"] = clean_pages

    config = {"configurable": {"db_session": mock_session}}
    ingest_out = await ingest_into_rag_node(state, config)
    web_chunks: list[RetrievedChunk] = ingest_out["web_chunks"]

    # 1. Verify chunk provenance format
    assert len(web_chunks) > 0
    for wc in web_chunks:
        assert wc.chunk_id.startswith("web_"), f"Web chunk {wc.chunk_id} must have web_ prefix"
        assert wc.document_id.startswith("https://"), "Web chunk document_id must be the source URL"

    # 2. Verify DB guard: No canonical documents or chunks inserted into PostgreSQL
    mock_session.add.assert_not_called()

    # 3. Citation validation
    fact_chunk_id = web_chunks[0].chunk_id
    answer_text = f"Agashiye is a renowned restaurant in Ahmedabad [chunk: {fact_chunk_id}]."
    validated_ans, citations = validate_and_render_citations(
        raw_answer=answer_text,
        chunks=web_chunks,
    )
    assert f"[chunk: {fact_chunk_id}]" in validated_ans
    assert len(citations) > 0


# ============================================================================
# 8. RBAC and context-budget protections remaining intact
# ============================================================================

def test_rbac_and_context_budget_protection():
    allowed_dept = uuid.uuid4()
    unauthorized_dept = uuid.uuid4()

    compiler = ContextCompiler(max_context_tokens=500)

    # Chunks from authorized and unauthorized departments
    auth_chunk = RetrievedChunk(
        chunk_id="doc_auth_1",
        document_id=str(uuid.uuid4()),
        department_id=str(allowed_dept),
        file_name="auth_guide.txt",
        text="Authorized department context.",
        chunk_index=0,
        hybrid_score=0.9,
    )

    unauth_chunk = RetrievedChunk(
        chunk_id="doc_unauth_2",
        document_id=str(uuid.uuid4()),
        department_id=str(unauthorized_dept),
        file_name="restricted.txt",
        text="Restricted department secret information.",
        chunk_index=0,
        hybrid_score=0.99,
    )

    # Transient web chunk
    web_chunk = RetrievedChunk(
        chunk_id="web_1",
        document_id="https://example.com/guide",
        department_id="None",
        file_name="Public Web Guide",
        text="Public restaurant guide in Ahmedabad.",
        chunk_index=0,
        hybrid_score=0.85,
    )

    # 1. Security guard: Unauthorized internal chunks strictly rejected with RuntimeError
    with pytest.raises(RuntimeError, match="Security violation: Unauthorized chunk detected"):
        compiler.compile(
            chunks=[auth_chunk, unauth_chunk, web_chunk],
            query="top 10 restaurant in ahemadabad",
            allowed_departments=[allowed_dept],
            is_web_context=True,
        )

    # 2. When only authorized chunks + transient web chunks are passed, compile succeeds and respects budget
    compiled = compiler.compile(
        chunks=[auth_chunk, web_chunk],
        query="top 10 restaurant in ahemadabad",
        allowed_departments=[allowed_dept],
        is_web_context=True,
    )

    compiled_ids = [c.chunk_id for c in compiled.chunks]
    assert "doc_auth_1" in compiled_ids
    assert "web_1" in compiled_ids
    assert compiled.token_count <= 500
