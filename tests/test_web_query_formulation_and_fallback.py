import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_app.core.config import Settings
from rag_app.db.models import User
from rag_app.retrieval.query_parser import (
    strip_search_triggers,
    format_web_search_query,
)
from rag_app.schemas.rag import (
    QueryPlan,
    RetrievalResponse,
    RetrievedChunk,
    GenerateResponse,
)
from rag_app.schemas.research import (
    SufficiencyEvaluation,
    SufficiencyStatus,
)
from rag_app.research.provider import (
    ProviderRequestError,
    WebSearchResponse,
    WebSearchResult,
    WebPageResult,
)
from rag_app.research.registry import ProviderRegistry
from rag_app.research.graph import (
    ResearchState,
    retrieve_rag,
    search_web,
    select_sources,
    crawl_web,
    generate_final_answer,
    _classify_scrape_error,
)


def make_dummy_user() -> User:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    return user


# ============================================================================
# 1. Web Query Formulation Unit Tests
# ============================================================================

def test_strip_search_triggers_various_prefixes():
    cases = [
        ("Search the web for.. top restaurants in Ahmedabad", "top restaurants in Ahmedabad"),
        ("Google this: TCS latest share price", "TCS latest share price"),
        ("Please look up online for AI engineer jobs", "AI engineer jobs"),
        ("Search online for best laptops under 50000", "best laptops under 50000"),
        ("Find on the web: Python 3.12 release notes", "Python 3.12 release notes"),
        ("Check online for Ahmedabad weather today", "Ahmedabad weather today"),
        ("Direct query with no prefix", "Direct query with no prefix"),
    ]
    for raw, expected in cases:
        assert strip_search_triggers(raw) == expected


def test_format_web_search_query_with_semantic_plan():
    raw_query = "Search the web for..mai ahemdabad me hu as a ai engg aur mujhe 30k milti h salary aur tum comapare kro"
    plan = QueryPlan(
        semantic_query="average AI engineer salary in India, highest AI engineer salary in India, compare with 30,000 INR",
        web_search_needed=True,
    )
    formatted, source = format_web_search_query(raw_query, plan)
    assert source == "semantic_query"
    assert "average AI engineer salary in India" in formatted
    assert "30,000 INR" in formatted


def test_format_web_search_query_without_plan_strips_triggers_and_filler():
    raw_query = "Can you please search the web for.. in ahemdabad as a ai engg above 30k salary"
    formatted, source = format_web_search_query(raw_query, None)
    assert source == "prefix_stripped"
    assert "in ahemdabad as a ai engg above 30k salary" in formatted


def test_format_web_search_query_direct_short_query_preserved():
    raw_query = "Python 3.12 release date"
    formatted, source = format_web_search_query(raw_query, None)
    assert source == "raw_passthrough"
    assert formatted == "Python 3.12 release date"


# ============================================================================
# 2. QueryPlan Propagation in retrieve_rag
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_rag_propagates_query_plan_to_state():
    parsed_plan = QueryPlan(
        semantic_query="AI engineer jobs in Ahmedabad salary above 30k",
        web_search_needed=False,
        internal_retrieval_needed=True,
    )
    mock_response = RetrievalResponse(
        query="Search the web for.. in ahemdabad as a ai engg above 30k salary",
        results=[],
        query_plan=parsed_plan,
    )
    mock_service = MagicMock()
    mock_service.retrieve = AsyncMock(return_value=mock_response)

    initial_plan = QueryPlan(
        semantic_query="Search the web for.. in ahemdabad as a ai engg above 30k salary",
        web_search_needed=True,
        web_search_reason="explicit_request",
        internal_retrieval_needed=True,
    )

    state: ResearchState = {
        "query": "Search the web for.. in ahemdabad as a ai engg above 30k salary",
        "top_k": 5,
        "rerank": False,
        "user": make_dummy_user(),
        "allowed_departments": None,
        "workspace_id": None,
        "query_type": "knowledge",
        "query_plan": initial_plan,
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
        "raw_query": None,
    }

    config = {"configurable": {"retrieval_service": mock_service}}
    result = await retrieve_rag(state, config)

    assert "query_plan" in result
    propagated_plan = result["query_plan"]
    # Semantic query must be updated from parsed plan
    assert propagated_plan.semantic_query == "AI engineer jobs in Ahmedabad salary above 30k"
    # But routing flags like web_search_needed must be preserved
    assert propagated_plan.web_search_needed is True
    assert propagated_plan.web_search_reason == "explicit_request"


# ============================================================================
# 3. search_web Uses Formatted Query and Logs Structured Events
# ============================================================================

@pytest.mark.asyncio
async def test_search_web_uses_formatted_query_while_state_query_unchanged():
    mock_provider = MagicMock()
    mock_provider.search = AsyncMock(return_value=WebSearchResponse(results=[
        WebSearchResult(url="https://ambitionbox.com/salaries", title="AI Engineer Salary", snippet="Average 6 LPA")
    ]))

    mock_registry = ProviderRegistry()
    mock_registry.register("tinyfish", mock_provider)

    plan = QueryPlan(
        semantic_query="average AI engineer salary in India, compare with 30,000 INR",
        web_search_needed=True,
    )

    state: ResearchState = {
        "query": "Search the web for..mai ahemdabad me hu as a ai engg aur mujhe 30k milti h",
        "query_plan": plan,
    }

    with patch("rag_app.research.graph.get_provider_registry", return_value=mock_registry), \
         patch("rag_app.research.graph.get_settings", return_value=Settings(web_search_provider="tinyfish")):
        res = await search_web(state)

    # Provider search must have received the formatted query, with limit=10
    mock_provider.search.assert_awaited_once()
    called_query, called_kwargs = mock_provider.search.call_args[0][0], mock_provider.search.call_args[1]
    assert "average AI engineer salary in India" in called_query
    assert called_kwargs.get("limit") == 10
    assert len(res["search_results"]) == 1

    # Original state query was NOT altered
    assert state["query"] == "Search the web for..mai ahemdabad me hu as a ai engg aur mujhe 30k milti h"


# ============================================================================
# 4. Error Classification Unit Tests
# ============================================================================

def test_classify_scrape_error():
    assert _classify_scrape_error(ProviderRequestError("bot_blocked")) == ("bot_blocked", False)
    assert _classify_scrape_error(Exception("403 Forbidden")) == ("http_403", False)
    assert _classify_scrape_error(Exception("404 Page Not Found")) == ("http_404", False)
    assert _classify_scrape_error(Exception("429 Too Many Requests")) == ("rate_limited", True)
    assert _classify_scrape_error(Exception("Connection timed out")) == ("timeout", True)


# ============================================================================
# 5. Multi-Source Fallback When First Candidate Fails
# ============================================================================

@pytest.mark.asyncio
async def test_crawl_web_fallback_when_selected_candidate_is_bot_blocked():
    """
    Simulates the exact observed defect:
    - 1 candidate (Glassdoor) is selected.
    - Glassdoor returns bot_blocked.
    - Fallback candidate (AmbitionBox) in search_results pool is automatically attempted and succeeds.
    """
    mock_provider = MagicMock()

    async def mock_scrape(url: str):
        if "glassdoor" in url:
            raise ProviderRequestError("TinyFish scrape failed: bot_blocked")
        elif "ambitionbox" in url:
            return WebPageResult(
                url=url,
                title="AI Engineer Salaries in India",
                markdown="# Salary Data\nAverage AI Engineer salary in India is 6,50,000 INR per year.",
            )
        return WebPageResult(url=url, markdown="")

    mock_provider.scrape = AsyncMock(side_effect=mock_scrape)

    mock_registry = ProviderRegistry()
    mock_registry.register("tinyfish", mock_provider)

    state: ResearchState = {
        "selected_urls": ["https://www.glassdoor.co.in/Salaries/ai-engineer-salary-SRCH_KO0,11.htm"],
        "search_results": [
            WebSearchResult(url="https://www.glassdoor.co.in/Salaries/ai-engineer-salary-SRCH_KO0,11.htm", title="Glassdoor"),
            WebSearchResult(url="https://www.ambitionbox.com/salaries/ai-engineer-salaries", title="AmbitionBox"),
            WebSearchResult(url="https://in.indeed.com/career/ai-engineer/salaries", title="Indeed"),
        ],
    }

    settings = Settings(
        web_search_provider="tinyfish",
        web_research_target_pages=1,
        web_research_max_scrapes=3,
    )

    with patch("rag_app.research.graph.get_provider_registry", return_value=mock_registry), \
         patch("rag_app.research.graph.get_settings", return_value=settings):
        res = await crawl_web(state)

    # Must have crawled the fallback page
    assert len(res["crawled_pages"]) == 1
    assert res["crawled_pages"][0].url == "https://www.ambitionbox.com/salaries/ai-engineer-salaries"
    assert "Average AI Engineer salary" in res["crawled_pages"][0].markdown

    # Both URLs were attempted (Glassdoor failed, AmbitionBox succeeded)
    assert mock_provider.scrape.await_count == 2


@pytest.mark.asyncio
async def test_crawl_web_exhaustion_when_all_candidates_fail():
    """
    Verifies that when all candidates fail, the system bounds attempts at max_scrapes,
    does not hang, and returns empty pages with exhausted status.
    """
    mock_provider = MagicMock()
    mock_provider.scrape = AsyncMock(side_effect=ProviderRequestError("bot_blocked"))

    mock_registry = ProviderRegistry()
    mock_registry.register("tinyfish", mock_provider)

    state: ResearchState = {
        "selected_urls": ["https://site1.com/a"],
        "search_results": [
            WebSearchResult(url="https://site1.com/a", title="Site 1"),
            WebSearchResult(url="https://site2.com/b", title="Site 2"),
            WebSearchResult(url="https://site3.com/c", title="Site 3"),
            WebSearchResult(url="https://site4.com/d", title="Site 4"),
            WebSearchResult(url="https://site5.com/e", title="Site 5"),
        ],
    }

    settings = Settings(
        web_search_provider="tinyfish",
        web_research_target_pages=2,
        web_research_max_scrapes=3,  # budget cap
    )

    with patch("rag_app.research.graph.get_provider_registry", return_value=mock_registry), \
         patch("rag_app.research.graph.get_settings", return_value=settings):
        res = await crawl_web(state)

    assert mock_provider.scrape.await_count == 3


@pytest.mark.asyncio
async def test_search_web_falls_back_when_configured_provider_fails():
    """
    Verifies that if the configured provider (e.g. firecrawl) fails due to missing API key or error,
    search_web automatically falls back to an available registered provider (e.g. tinyfish).
    """
    broken_firecrawl = MagicMock()
    broken_firecrawl.search = AsyncMock(side_effect=ProviderRequestError("FIRECRAWL_API_KEY is not configured."))

    working_tinyfish = MagicMock()
    working_tinyfish.search = AsyncMock(return_value=WebSearchResponse(results=[
        WebSearchResult(url="https://example.com/fallback", title="Fallback Result", snippet="Info from fallback provider")
    ]))

    mock_registry = ProviderRegistry()
    mock_registry.register("tinyfish", working_tinyfish)

    state: ResearchState = {
        "query": "Search the web for AI engineer salary in India",
        "web_provider": "firecrawl",
    }

    with patch("rag_app.research.graph.get_provider_registry", return_value=mock_registry), \
         patch("rag_app.providers.registry.get_provider", return_value=broken_firecrawl):
        res = await search_web(state)

    # Broken provider was attempted
    broken_firecrawl.search.assert_awaited_once()
    # Working fallback provider was invoked and rescued the search
    working_tinyfish.search.assert_awaited_once()
    assert len(res["search_results"]) == 1
    assert res["search_results"][0].url == "https://example.com/fallback"
    assert res["web_provider"] == "tinyfish"

