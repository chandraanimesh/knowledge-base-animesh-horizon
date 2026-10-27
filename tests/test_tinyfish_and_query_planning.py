import uuid
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from rag_app.core.config import Settings
from rag_app.db.models import User
from rag_app.generation.citations import validate_and_render_citations
from rag_app.research.graph import (
    ResearchState,
    decide_next_after_eval,
    generate_final_answer,
    search_web,
)
from rag_app.research.provider import (
    ProviderRequestError,
    ProviderUnavailableError,
    WebSearchResponse,
    WebSearchResult,
    WebPageResult,
)
from rag_app.research.providers.tinyfish import TinyFishProvider
from rag_app.research.registry import ProviderRegistry
from rag_app.retrieval.query_parser import detect_web_search_intent, QueryParser
from rag_app.schemas.rag import (
    GenerateResponse,
    QueryPlan,
    RetrievedChunk,
)
from rag_app.schemas.research import (
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
# 1. TinyFish Search Response Normalization
# ============================================================================

@pytest.mark.asyncio
async def test_tinyfish_search_response_normalization():
    mock_payload = {
        "query": "deepseek v3 benchmarks",
        "total_results": 2,
        "page": 0,
        "results": [
            {
                "position": 1,
                "site_name": "example.com",
                "title": "DeepSeek V3 Benchmark Analysis",
                "snippet": "DeepSeek V3 demonstrates state of the art reasoning performance.",
                "url": "https://example.com/deepseek-v3",
                "date": "2026-03-01",
            },
            {
                "position": 2,
                "site_name": "techblog.org",
                "title": "LLM Comparisons 2026",
                "snippet": "Comprehensive analysis of open weight models.",
                "url": "https://techblog.org/llm-2026",
            },
        ],
    }

    provider = TinyFishProvider(
        api_key="sk-tinyfish-mock",
        search_url="https://api.search.tinyfish.ai",
    )

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        res = await provider.search("deepseek v3 benchmarks", limit=5)

        assert isinstance(res, WebSearchResponse)
        assert len(res.results) == 2
        assert res.results[0].title == "DeepSeek V3 Benchmark Analysis"
        assert res.results[0].url == "https://example.com/deepseek-v3"
        assert "reasoning performance" in res.results[0].snippet


# ============================================================================
# 2. TinyFish Fetch Markdown Response Normalization
# ============================================================================

@pytest.mark.asyncio
async def test_tinyfish_fetch_markdown_normalization():
    mock_payload = {
        "results": [
            {
                "url": "https://example.com/deepseek-v3",
                "final_url": "https://example.com/deepseek-v3",
                "title": "DeepSeek V3 Benchmark Analysis",
                "text": "# DeepSeek V3\n\nDeepSeek V3 achieved benchmark parity at 1/5th the training cost.",
                "published_date": "2026-03-01",
                "language": "en",
            }
        ],
        "errors": [],
    }

    provider = TinyFishProvider(
        api_key="sk-tinyfish-mock",
        fetch_url="https://api.fetch.tinyfish.ai",
    )

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        res = await provider.scrape("https://example.com/deepseek-v3")

        assert isinstance(res, WebPageResult)
        assert res.url == "https://example.com/deepseek-v3"
        assert res.title == "DeepSeek V3 Benchmark Analysis"
        assert "# DeepSeek V3" in res.markdown
        assert res.metadata.get("language") == "en"


# ============================================================================
# 3. Missing API Key Guard
# ============================================================================

def test_tinyfish_missing_api_key_guard():
    with patch("rag_app.research.providers.tinyfish.get_settings") as mock_settings:
        mock_settings.return_value = Settings(tinyfish_api_key=None)
        with pytest.raises(ProviderRequestError) as exc_info:
            TinyFishProvider(api_key=None)
        assert "TINYFISH_API_KEY is not configured" in str(exc_info.value)


# ============================================================================
# 4. Timeouts and Rate-Limit / Provider Failures
# ============================================================================

@pytest.mark.asyncio
async def test_tinyfish_timeout_and_error_handling():
    provider = TinyFishProvider(api_key="sk-tinyfish-mock")

    # Timeout
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ReadTimeout("Timeout exceeded")
        with pytest.raises(ProviderUnavailableError):
            await provider.search("query")

    # Rate limit (429)
    req = httpx.Request("GET", "https://api.search.tinyfish.ai")
    resp_429 = httpx.Response(status_code=429, request=req)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError("Rate limit", request=req, response=resp_429)
        with pytest.raises(ProviderRequestError) as exc_info:
            await provider.search("query")
        assert "rate limit exceeded" in str(exc_info.value).lower()

    # Server error (502)
    resp_502 = httpx.Response(status_code=502, request=req)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError("Bad Gateway", request=req, response=resp_502)
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await provider.search("query")
        assert "server error" in str(exc_info.value).lower()


# ============================================================================
# 5. Provider Registry Selection
# ============================================================================

def test_provider_registry_selection():
    registry = ProviderRegistry()
    mock_tf = MagicMock(spec=TinyFishProvider)
    mock_fc = MagicMock()

    registry.register("tinyfish", mock_tf)
    registry.register("firecrawl", mock_fc)

    assert registry.get("tinyfish") is mock_tf
    assert registry.get("firecrawl") is mock_fc


# ============================================================================
# 6. Explicit Web-Search Request Triggers Research
# ============================================================================

def test_explicit_web_search_intent_routing():
    query = "Please search the web for latest Python 3.14 features"
    needed, reason, internal_needed = detect_web_search_intent(query)
    assert needed is True
    assert reason == "explicit_request"

    plan = QueryPlan(
        semantic_query="Python 3.14 features",
        web_search_needed=needed,
        web_search_reason=reason,
        internal_retrieval_needed=internal_needed,
    )

    state: ResearchState = {
        "query": query,
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "query_plan": plan,
        "retrieved_chunks": [make_chunk("c1", "Internal document about python")],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": True,
        "sufficiency_status": "sufficient",
        "sufficiency_evaluation": SufficiencyEvaluation(
            status=SufficiencyStatus.SUFFICIENT,
            is_sufficient=True,
            confidence=0.9,
            coverage=0.9,
            evidence_types=["prose"],
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
        mock_settings.return_value = Settings(enable_web_research=True, web_search_provider="tinyfish")
        next_step = decide_next_after_eval(state)
        # Explicit request must trigger research even though internal evidence was found
        assert next_step == "init_research_run"


# ============================================================================
# 7. Current-Information / Freshness Query Triggers Research
# ============================================================================

def test_freshness_intent_routing():
    query = "What are the latest updates on NVIDIA RTX 5090 specs today?"
    needed, reason, internal_needed = detect_web_search_intent(query)
    assert needed is True
    assert reason == "freshness"

    plan = QueryPlan(
        semantic_query="NVIDIA RTX 5090 specs",
        web_search_needed=needed,
        web_search_reason=reason,
        internal_retrieval_needed=internal_needed,
    )

    state: ResearchState = {
        "query": query,
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "query_plan": plan,
        "retrieved_chunks": [make_chunk("c1", "Internal specs draft from 2024")],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": True,
        "sufficiency_status": "sufficient",
        "sufficiency_evaluation": SufficiencyEvaluation(
            status=SufficiencyStatus.SUFFICIENT,
            is_sufficient=True,
            confidence=0.9,
            coverage=0.9,
            evidence_types=["prose"],
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
        mock_settings.return_value = Settings(enable_web_research=True, web_search_provider="tinyfish")
        next_step = decide_next_after_eval(state)
        assert next_step == "init_research_run"


# ============================================================================
# 8. Normal Internal Question Avoids Unnecessary External Search
# ============================================================================

def test_normal_internal_query_avoids_web_search():
    query = "What is our company employee leave policy?"
    needed, reason, _ = detect_web_search_intent(query)
    assert needed is False

    plan = QueryPlan(
        semantic_query="company employee leave policy",
        web_search_needed=False,
    )

    state: ResearchState = {
        "query": query,
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "query_plan": plan,
        "retrieved_chunks": [make_chunk("c1", "Annual leave is 20 days.")],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": True,
        "sufficiency_status": "sufficient",
        "sufficiency_evaluation": SufficiencyEvaluation(
            status=SufficiencyStatus.SUFFICIENT,
            is_sufficient=True,
            confidence=0.95,
            coverage=0.95,
            evidence_types=["prose"],
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
        mock_settings.return_value = Settings(enable_web_research=True, web_search_provider="tinyfish")
        next_step = decide_next_after_eval(state)
        # Normal query with sufficient internal evidence must directly generate answer
        assert next_step == "generate_final_answer"


# ============================================================================
# 9. Hybrid Internal + Web Evidence Combined With Separate Provenance Intact
# ============================================================================

@pytest.mark.asyncio
async def test_hybrid_internal_and_web_evidence_provenance():
    internal_chunk = RetrievedChunk(
        chunk_id="doc_internal_1",
        document_id="dept-handbook.pdf",
        department_id=str(uuid.uuid4()),
        text="Internal handbook discusses product roadmap v1.",
        file_name="dept-handbook.pdf",
        chunk_index=1,
        hybrid_score=0.85,
    )
    web_chunk = RetrievedChunk(
        chunk_id="web_tinyfish_1",
        document_id="https://news.example.com/roadmap-v2",
        department_id="None",
        text="External press announces product roadmap v2 release.",
        file_name="Tech News: Roadmap v2",
        chunk_index=1,
        hybrid_score=1.0,
    )

    state: ResearchState = {
        "query": "What is the status of the product roadmap?",
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "query_plan": QueryPlan(web_search_needed=True, web_search_reason="freshness"),
        "retrieved_chunks": [internal_chunk],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": False,
        "sufficiency_status": "insufficient",
        "sufficiency_evaluation": None,
        "research_run_id": str(uuid.uuid4()),
        "search_results": [],
        "selected_urls": ["https://news.example.com/roadmap-v2"],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": [web_chunk],
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "raw_query": None,
    }

    mock_gen_service = MagicMock()
    mock_gen_service.generate_from_context = AsyncMock(
        return_value=GenerateResponse(
            answer="Internal roadmap v1 [chunk: doc_internal_1] was updated to v2 [chunk: web_tinyfish_1].",
            citations=["dept-handbook.pdf", "Tech News: Roadmap v2"],
            retrieved_chunks=[internal_chunk, web_chunk],
        )
    )

    mock_compiler = MagicMock()
    mock_compiler.compile.return_value = MagicMock(
        chunks=[internal_chunk, web_chunk],
        conflicts=[],
        chunk_hierarchy=None,
        rendered_context="Internal & Web Context",
    )

    config = {
        "configurable": {
            "generation_service": mock_gen_service,
            "context_compiler": mock_compiler,
        }
    }

    result = await generate_final_answer(state, config)
    compiled_chunks = mock_compiler.compile.call_args.kwargs["chunks"]

    # Verify both internal and web chunks are present in compiled context
    assert any(c.chunk_id == "doc_internal_1" for c in compiled_chunks)
    assert any(c.chunk_id == "web_tinyfish_1" for c in compiled_chunks)

    # Verify separate provenance
    int_c = next(c for c in compiled_chunks if c.chunk_id == "doc_internal_1")
    web_c = next(c for c in compiled_chunks if c.chunk_id == "web_tinyfish_1")
    assert int_c.file_name == "dept-handbook.pdf"
    assert web_c.document_id == "https://news.example.com/roadmap-v2"


# ============================================================================
# 10. Disabled Web Research Preserves Existing Safe Fallback
# ============================================================================

def test_disabled_web_research_safe_fallback():
    query = "Search the web for competitor pricing"
    needed, reason, _ = detect_web_search_intent(query)

    plan = QueryPlan(
        semantic_query="competitor pricing",
        web_search_needed=needed,
        web_search_reason=reason,
    )

    state: ResearchState = {
        "query": query,
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": [uuid.uuid4()],
        "workspace_id": uuid.uuid4(),
        "query_type": "knowledge",
        "query_plan": plan,
        "retrieved_chunks": [],
        "strategy_trace": None,
        "chunk_provenance": None,
        "chunk_hierarchy": None,
        "is_sufficient": False,
        "sufficiency_status": "insufficient",
        "sufficiency_evaluation": None,
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
        mock_settings.return_value = Settings(enable_web_research=False, web_search_provider="tinyfish")
        next_step = decide_next_after_eval(state)
        # Even with explicit web search intent, if research is disabled, route to no_context
        assert next_step == "no_context"
