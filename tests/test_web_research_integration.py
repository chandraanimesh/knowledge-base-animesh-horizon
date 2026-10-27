import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_app.core.config import Settings
from rag_app.db.models import Document, DocumentChunk, KnowledgeItem, User
from rag_app.generation.citations import validate_and_render_citations
from rag_app.generation.prompt import build_rag_messages
from rag_app.research.graph import (
    ResearchState,
    classify_query,
    crawl_web,
    decide_next_after_eval,
    generate_final_answer,
    ingest_into_rag_node,
    init_research_run,
    is_safe_web_url,
    search_web,
)
from rag_app.research.persistence import (
    create_transient_web_chunks,
    persist_knowledge,
)
from rag_app.research.provider import (
    WebPageResult,
    WebSearchResult,
    WebSearchResponse,
)
from rag_app.schemas.rag import (
    GenerateResponse,
    RetrievedChunk,
)
from rag_app.schemas.research import (
    CleanWebPage,
    ExtractedFact,
    StructuredKnowledgeCandidate,
    SufficiencyEvaluation,
    SufficiencyStatus,
)


def make_dummy_user() -> User:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    return user


def make_chunk(chunk_id: str, text: str, file_name: str = "doc.md") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=str(uuid.uuid4()),
        department_id=str(uuid.uuid4()),
        text=text,
        file_name=file_name,
        chunk_index=1,
        hybrid_score=0.9,
    )


# ============================================================================
# 1. Internal Evidence Sufficient -> Research NOT triggered
# ============================================================================

def test_internal_sufficient_skips_research():
    state: ResearchState = {
        "query": "What is our refund policy?",
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "retrieved_chunks": [make_chunk("c1", "Refunds are processed within 14 days.")],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": True,
        "sufficiency_status": "sufficient",
        "sufficiency_evaluation": SufficiencyEvaluation(
            status=SufficiencyStatus.SUFFICIENT,
            is_sufficient=True,
            confidence=0.95,
            coverage=0.9,
            evidence_types=["fact"],
            missing_requirements=[],
        ),
        "research_run_id": None,
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": None,
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "raw_query": None,
    }

    with patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=True)
        next_step = decide_next_after_eval(state)
        assert next_step == "generate_final_answer", "Sufficient internal evidence must route to generate_final_answer, not research"


# ============================================================================
# 2. Internal Evidence Insufficient + Research Disabled -> Safe Fallback
# ============================================================================

def test_internal_insufficient_research_disabled():
    state: ResearchState = {
        "query": "What are ACME Corp 2026 specs?",
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "retrieved_chunks": [],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": False,
        "sufficiency_status": "insufficient",
        "sufficiency_evaluation": SufficiencyEvaluation(
            status=SufficiencyStatus.INSUFFICIENT,
            is_sufficient=False,
            confidence=0.1,
            coverage=0.0,
            evidence_types=[],
            missing_requirements=["ACME Corp specs"],
        ),
        "research_run_id": None,
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": None,
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "raw_query": None,
    }

    with patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=False)
        next_step = decide_next_after_eval(state)
        assert next_step == "no_context", "When research is disabled, insufficient internal evidence must route to no_context"


# ============================================================================
# 3. Internal Evidence Insufficient + Research Enabled -> Init Research Run
# ============================================================================

def test_internal_insufficient_research_enabled():
    state: ResearchState = {
        "query": "What are ACME Corp 2026 specs?",
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "retrieved_chunks": [],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": False,
        "sufficiency_status": "insufficient",
        "sufficiency_evaluation": SufficiencyEvaluation(
            status=SufficiencyStatus.INSUFFICIENT,
            is_sufficient=False,
            confidence=0.1,
            coverage=0.0,
            evidence_types=[],
            missing_requirements=["ACME Corp specs"],
        ),
        "research_run_id": None,
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": None,
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "raw_query": None,
    }

    with patch("rag_app.research.graph.get_settings") as mock_settings:
        mock_settings.return_value = Settings(enable_web_research=True)
        next_step = decide_next_after_eval(state)
        assert next_step == "init_research_run", "When research is enabled, insufficient internal evidence must route to init_research_run"


# ============================================================================
# 4. Research Returns Relevant Evidence -> Answer Generation Receives Transient Evidence
# ============================================================================

@pytest.mark.asyncio
async def test_research_returns_relevant_evidence_transient_contract():
    candidate = StructuredKnowledgeCandidate(
        facts=[
            ExtractedFact(
                entity_name="ACME Corp",
                entity_type="ORGANIZATION",
                attributes={"product": "Widget X", "launch": "March 2026"},
                department_code="SALES",
                source_url="https://example.com/acme-news",
                evidence_text="ACME Corp announced Widget X in March 2026.",
            )
        ]
    )
    clean_pages = [
        CleanWebPage(
            url="https://example.com/acme-news",
            title="ACME Widget X Announcement",
            content="ACME Corp announced Widget X in March 2026.",
        )
    ]
    run_id = str(uuid.uuid4())

    state: ResearchState = {
        "query": "When did ACME announce Widget X?",
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "retrieved_chunks": [],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": False,
        "sufficiency_status": "insufficient",
        "sufficiency_evaluation": None,
        "research_run_id": run_id,
        "search_results": [],
        "selected_urls": ["https://example.com/acme-news"],
        "crawled_pages": [],
        "clean_pages": clean_pages,
        "extracted_knowledge": candidate,
        "web_chunks": None,
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "raw_query": None,
    }

    config = {"configurable": {}}
    ingest_result = await ingest_into_rag_node(state, config)

    web_chunks = ingest_result["web_chunks"]
    assert len(web_chunks) >= 1
    assert all(c.chunk_id.startswith("web_") for c in web_chunks)
    assert any(c.file_name == "ACME Widget X Announcement" for c in web_chunks)
    assert any("Widget X" in c.text for c in web_chunks)

    # Now verify generate_final_answer Path B consumes these chunks
    state["web_chunks"] = web_chunks
    state["retrieved_chunks"] = web_chunks

    mock_gen_service = MagicMock()
    mock_gen_service.generate_from_context = AsyncMock(
        return_value=GenerateResponse(
            answer="ACME Corp announced Widget X in March 2026 [1].",
            citations=["ACME Widget X Announcement"],
            retrieved_chunks=web_chunks,
        )
    )

    mock_context_compiler = MagicMock()
    mock_context_compiler.compile.return_value = MagicMock(
        chunks=web_chunks,
        conflicts=[],
        chunk_hierarchy=None,
        rendered_context="[1] ACME Widget X Announcement: ACME Corp announced Widget X",
    )

    gen_config = {
        "configurable": {
            "generation_service": mock_gen_service,
            "context_compiler": mock_context_compiler,
        }
    }

    gen_result = await generate_final_answer(state, gen_config)
    resp = gen_result["final_response"]
    assert resp is not None
    assert "Widget X" in resp.answer
    assert mock_context_compiler.compile.call_args.kwargs.get("is_web_context") is True
    assert mock_gen_service.generate_from_context.call_args.kwargs.get("is_web_context") is True


# ============================================================================
# 5. Research Returns No Results -> Safe Abstention
# ============================================================================

@pytest.mark.asyncio
async def test_research_returns_no_results_safe_fallback():
    state: ResearchState = {
        "query": "Non-existent obscure topic?",
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "retrieved_chunks": [],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": False,
        "sufficiency_status": "insufficient",
        "sufficiency_evaluation": None,
        "research_run_id": str(uuid.uuid4()),
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": [],
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "raw_query": None,
    }

    mock_gen_service = MagicMock()
    mock_context_compiler = MagicMock()

    gen_config = {
        "configurable": {
            "generation_service": mock_gen_service,
            "context_compiler": mock_context_compiler,
        }
    }

    gen_result = await generate_final_answer(state, gen_config)
    resp = gen_result["final_response"]
    assert resp is not None
    assert "could not verify" in resp.answer.lower()
    assert resp.citations == []
    assert resp.retrieved_chunks == []
    # Ensure LLM generate_from_context was NOT called pointlessly with empty context
    mock_gen_service.generate_from_context.assert_not_called()


# ============================================================================
# 6. Provider Error or Timeout -> Graceful Handling, No Crash, No Infinite Loop
# ============================================================================

@pytest.mark.asyncio
async def test_provider_error_graceful_handling():
    mock_provider = MagicMock()
    mock_provider.search = AsyncMock(side_effect=TimeoutError("Firecrawl gateway timed out"))
    mock_provider.scrape = AsyncMock(side_effect=Exception("Scrape rate limit exceeded"))

    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_provider

    with patch("rag_app.research.graph.get_provider_registry", return_value=mock_registry):
        state: ResearchState = {
            "query": "ACME news",
            "top_k": 5,
            "rerank": False,
            "user": make_dummy_user(),
            "allowed_departments": [uuid.uuid4()],
            "workspace_id": uuid.uuid4(),
            "query_type": "knowledge",
            "retrieved_chunks": [],
            "strategy_trace": None,
            "chunk_provenance": None,
            "chunk_hierarchy": None,
            "is_sufficient": False,
            "sufficiency_status": "insufficient",
            "sufficiency_evaluation": None,
            "research_run_id": str(uuid.uuid4()),
            "search_results": [],
            "selected_urls": ["https://example.com/item1"],
            "crawled_pages": [],
            "clean_pages": [],
            "extracted_knowledge": None,
            "web_chunks": None,
            "conversation_history": None,
            "workspace_thread_context": None,
            "final_response": None,
            "output_preferences": None,
            "conversation_id": None,
            "raw_query": None,
        }

        # search_web should catch the exception and return empty search_results
        search_res = await search_web(state)
        assert search_res == {"search_results": []}

        # crawl_web should catch scrape errors and return empty crawled_pages
        crawl_res = await crawl_web(state)
        assert crawl_res == {"crawled_pages": []}


# ============================================================================
# 7. Zero DB Pollution: Web Content NEVER Inserted Into Documents or Chunks
# ============================================================================

@pytest.mark.asyncio
async def test_no_database_pollution_documents_and_document_chunks():
    # Create mock session to record any ORM calls
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    candidate = StructuredKnowledgeCandidate(
        facts=[
            ExtractedFact(
                entity_name="External Entity",
                entity_type="ORGANIZATION",
                attributes={"software": "New Software", "year": "2026"},
                department_code="ALL",
                source_url="https://external-news.com/release",
                evidence_text="External Entity released New Software in 2026.",
            )
        ]
    )
    clean_pages = [
        CleanWebPage(
            url="https://external-news.com/release",
            title="External Software Release",
            content="External Entity released New Software in 2026.",
        )
    ]
    run_id = str(uuid.uuid4())

    # Test 1: persist_knowledge only touches ResearchRun / ResearchKnowledge, NEVER Document or DocumentChunk or KnowledgeItem
    await persist_knowledge(
        session=mock_session,
        run_id_str=run_id,
        candidate=candidate,
    )

    for call_arg in mock_session.add.call_args_list:
        added_obj = call_arg[0][0]
        assert not isinstance(added_obj, Document), "Must NOT add instances of Document"
        assert not isinstance(added_obj, DocumentChunk), "Must NOT add instances of DocumentChunk"
        assert not isinstance(added_obj, KnowledgeItem), "Must NOT adapt web facts into authoritative KnowledgeItem"

    # Test 2: ingest_into_rag_node creates transient chunks and performs zero session additions
    mock_session.reset_mock()
    state: ResearchState = {
        "query": "External entity news?",
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "retrieved_chunks": [],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": False,
        "sufficiency_status": "insufficient",
        "sufficiency_evaluation": None,
        "research_run_id": run_id,
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": clean_pages,
        "extracted_knowledge": candidate,
        "web_chunks": None,
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "raw_query": None,
    }
    config = {"configurable": {"db_session": mock_session}}
    ingest_result = await ingest_into_rag_node(state, config)
    assert len(ingest_result["web_chunks"]) > 0
    mock_session.add.assert_not_called()


# ============================================================================
# 8. External Source Citations Validated Against Actual Retrieved Web Evidence
# ============================================================================

def test_citation_validation_against_web_evidence():
    web_chunk = RetrievedChunk(
        chunk_id="web_1",
        document_id="https://example.com/press",
        department_id="None",
        text="Beta announced product launch in April 2026.",
        file_name="Beta Press Release",
        chunk_index=1,
        hybrid_score=1.0,
    )

    answer_with_valid_web_citation = (
        "Beta announced the launch in April 2026 [chunk: web_1]."
    )
    validated_answer, rendered_citations = validate_and_render_citations(
        raw_answer=answer_with_valid_web_citation,
        chunks=[web_chunk],
    )
    assert "[chunk: web_1]" in validated_answer
    assert "Beta Press Release" in rendered_citations

    # Fabricated / ungrounded chunk ID
    answer_with_fake_citation = (
        "Beta announced the launch in April 2026 [chunk: fake_web_999]."
    )
    validated_fake_answer, fake_citations = validate_and_render_citations(
        raw_answer=answer_with_fake_citation,
        chunks=[web_chunk],
    )
    assert "fake_web_999" not in validated_fake_answer


# ============================================================================
# 9. SSRF Guard Blocks Malicious / Private IP URLs
# ============================================================================

def test_ssrf_guard_blocks_unsafe_urls():
    assert is_safe_web_url("https://www.google.com") is True
    assert is_safe_web_url("http://example.com/article/1") is True

    # Blocked schemes
    assert is_safe_web_url("file:///etc/passwd") is False
    assert is_safe_web_url("ftp://ftp.server.com") is False

    # Blocked localhost and loopback
    assert is_safe_web_url("http://localhost:8000/api") is False
    assert is_safe_web_url("http://127.0.0.1:5432") is False
    assert is_safe_web_url("http://[::1]:8080") is False

    # Blocked private networks (RFC 1918)
    assert is_safe_web_url("http://192.168.1.1/admin") is False
    assert is_safe_web_url("http://10.0.0.1/dashboard") is False
    assert is_safe_web_url("http://172.16.0.5/internal") is False

    # Blocked cloud metadata
    assert is_safe_web_url("http://169.254.169.254/latest/meta-data") is False

    # Blocked local domain suffixes
    assert is_safe_web_url("http://service.internal/secret") is False
    assert is_safe_web_url("http://router.local/config") is False


# ============================================================================
# 10. Web Research Prompt Guidance Injected When Web Chunks Present
# ============================================================================

def test_web_research_prompt_guidance_present():
    web_chunk = RetrievedChunk(
        chunk_id="web_fact_0",
        document_id="https://example.com/info",
        department_id="None",
        text="Sample external fact",
        file_name="External Source",
        chunk_index=1,
        hybrid_score=1.0,
    )
    messages = build_rag_messages(
        context="[1] External Source: Sample external fact",
        question="What is the external fact?",
        chunks=[web_chunk],
    )
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "Web Research Evidence Notice" in user_msg["content"]
    assert "External web evidence is unverified external research" in user_msg["content"]
