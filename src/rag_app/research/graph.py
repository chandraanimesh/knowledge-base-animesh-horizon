import logging
import time
import uuid
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables.config import RunnableConfig
from sqlalchemy import select
from rag_app.core.config import get_settings
from rag_app.core.observability import (
    record_llm_call,
    record_stage_end,
    record_stage_start,
)
from rag_app.db.models import ResearchKnowledge, ResearchRun, User
from rag_app.generation.llm_factory import get_llm
from rag_app.generation.service import GenerationService
from rag_app.retrieval.query_parser import is_greeting
from rag_app.retrieval.service import RetrievalService
from rag_app.context.compiler import ContextCompiler
from rag_app.research.cleaner import clean_and_deduplicate
from rag_app.research.extractor import (
    extract_knowledge as extractor_func,
)
from rag_app.research.persistence import (
    ingest_knowledge_to_rag,
    persist_knowledge,
)
from rag_app.research.provider import (
    WebPageResult,
    WebSearchResult,
)
from rag_app.research.registry import get_provider_registry
from rag_app.schemas.rag import ChunkProvenance, GenerateResponse, RetrievedChunk
from rag_app.schemas.research import (
    CleanWebPage,
    SourceSelection,
    StructuredKnowledgeCandidate,
    SufficiencyEvaluation,
)

logger = logging.getLogger(__name__)


# ============================================================
# STATE
# ============================================================

class ResearchState(TypedDict):
    query: str
    top_k: int
    rerank: bool
    user: User
    allowed_departments: list[uuid.UUID] | None
    workspace_id: uuid.UUID | None

    # Query routing
    query_type: Literal["conversation", "knowledge"] | None

    # Internal RAG
    retrieved_chunks: list[RetrievedChunk]
    strategy_trace: dict[str, int] | None
    chunk_provenance: dict[str, ChunkProvenance] | None
    is_sufficient: bool | None

    # Research
    research_run_id: str | None
    search_results: list[WebSearchResult]
    selected_urls: list[str]
    crawled_pages: list[WebPageResult]
    clean_pages: list[CleanWebPage]
    extracted_knowledge: StructuredKnowledgeCandidate | None

    # Conversational context
    conversation_history: list[dict[str, str]] | None
    workspace_thread_context: list[dict] | None

    # Final response
    final_response: GenerateResponse | None


# ============================================================
# 0. QUERY TYPE ROUTER
# ============================================================

async def classify_query(
    state: ResearchState,
) -> dict:
    """
    Lightweight deterministic routing before RAG.

    Conversation:
        hi
        hello
        good morning
        how are you

    Knowledge:
        everything that actually needs RAG/research
    """

    query = state["query"]

    if is_greeting(query):
        logger.info(
            "conversation_query_detected_at_graph_entry",
            extra={
                "event": "conversation_query_detected_at_graph_entry",
                "query": query,
            },
        )

        return {
            "query_type": "conversation",
        }

    return {
        "query_type": "knowledge",
    }


def route_after_classification(
    state: ResearchState,
) -> Literal["conversation", "rag"]:
    if state.get("query_type") == "conversation":
        return "conversation"

    return "rag"


# ============================================================
# 1. DIRECT CONVERSATION RESPONSE
# ============================================================

async def handle_conversation(
    state: ResearchState,
) -> dict:
    """
    Handles simple conversational queries without:

    - query parser
    - embeddings
    - BM25
    - vector search
    - reranker
    - research engine
    - generation LLM
    """

    logger.info(
        "conversation_response_generated",
        extra={
            "event": "conversation_response_generated",
            "query": state["query"],
        },
    )

    response = GenerateResponse(
        answer=(
            "Hello! I'm Doot — a research assistant "
            "grounded in your document library. "
            "How can I help you?"
        ),
        citations=[],
        retrieved_chunks=[],
    )

    return {
        "final_response": response,
        "retrieved_chunks": [],
    }


# ============================================================
# 2. EXISTING RAG
# ============================================================
async def route_after_sufficiency(state:ResearchState):

    if state.get("query_type") == "conversation":
        return "final"

    if state.get("rag_sufficient"):
        return "final"

    # Settings dependency removed as research fallback is disabled
    # if not settings.enable_web_research:
    #     return "no_context"

    return "no_context"
    
async def no_context(state:ResearchState):
    return {
        "final_response": GenerateResponse(
            answer=(
                "I could not verify the requested information "
                "from the available documents."
            ),
            citations=[],
            retrieved_chunks=[],
        )
    }
async def retrieve_rag(
    state: ResearchState,
    config: RunnableConfig,
) -> dict:

    retrieval_service: RetrievalService = (
        config["configurable"]["retrieval_service"]
    )

    logger.info(
        "Graph Node: retrieve_rag"
    )

    response = await retrieval_service.retrieve(
        query=state["query"],
        top_k=state["top_k"],
        rerank=state["rerank"],
        allowed_departments=state["allowed_departments"],
        workspace_id=state.get("workspace_id"),
    )

    return {
        "retrieved_chunks": response.results,
        "strategy_trace": response.strategy_trace,
        "chunk_provenance": response.chunk_provenance,
    }


# ============================================================
# 3. RAG SUFFICIENCY
# ============================================================

async def evaluate_sufficiency(
    state: ResearchState,
) -> dict:

    logger.info(
        "Graph Node: evaluate_sufficiency"
    )

    chunks = state.get(
        "retrieved_chunks",
        [],
    )

    # No retrieved evidence.
    # For a KNOWLEDGE query this means:
    # RAG is insufficient -> Research Engine.
    if not chunks:
        from rag_app.retrieval.query_parser import is_history_query
        if is_history_query(state["query"]) or bool(state.get("workspace_thread_context")):
            logger.info("Empty chunks but detected history or workspace thread context; routing to LLM.")
            return {
                "is_sufficient": True,
            }
        return {
            "is_sufficient": False,
        }

    record_stage_start("research")
    try:
        llm = get_llm()

        context_text = "\n\n".join(
            f"Chunk {i + 1}:\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )

        logger.info(
            "evaluate_sufficiency_input",
            extra={
                "event": "evaluate_sufficiency_input",
                "query": state["query"],
                "candidate_count": len(chunks),
                "top_candidate_preview": chunks[0].text[:200] if chunks else None,
            },
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a sufficiency evaluator. "
                    "Determine whether the provided context contains relevant, factual information "
                    "or guidelines to answer or address the user's question. "
                    "Return True if the context contains relevant facts, SOP guidelines, policies, or substantive information addressing the query. "
                    "Return False only if the context is entirely irrelevant, ungrounded, or lacks information related to the user's question."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question:\n{state['query']}"
                ),
            },
        ]

        try:
            structured_llm = (
                llm.llm.with_structured_output(
                    SufficiencyEvaluation,
                    include_raw=True,
                )
            )

            _eval_start = time.perf_counter()
            inv_res = await structured_llm.ainvoke(messages)
            _eval_latency = (time.perf_counter() - _eval_start) * 1000

            raw_msg = inv_res.get("raw") if isinstance(inv_res, dict) else None
            usage_meta = getattr(raw_msg, "usage_metadata", None) or {}
            record_llm_call(
                stage_name="research",
                usage=usage_meta,
                latency_ms=_eval_latency,
                model_name=getattr(llm.llm, "model_name", None),
            )

            result: SufficiencyEvaluation = (
                inv_res.get("parsed") if isinstance(inv_res, dict) else inv_res
            )

            is_sufficient = result.is_sufficient if result is not None else True

            logger.info(
                "sufficiency_evaluation",
                extra={
                    "event": "sufficiency_evaluation",
                    "query": state["query"],
                    "candidate_count": len(chunks),
                    "is_sufficient": is_sufficient,
                    "reason": getattr(result, "reason", None) if result else None,
                },
            )

        except Exception as exc:
            logger.error(
                "sufficiency_evaluation_failed",
                extra={
                    "event": "sufficiency_evaluation_failed",
                    "error": str(exc),
                },
            )

            # Preserve your existing behavior:
            # if evaluator fails but RAG returned chunks,
            # use standard RAG rather than unnecessarily
            # launching research.
            is_sufficient = True

        return {
            "is_sufficient": is_sufficient,
        }
    finally:
        record_stage_end("research")


def decide_next_after_eval(
    state: ResearchState,
) -> Literal[
    "generate_final_answer",
    "no_context",
]:

    if state.get("is_sufficient"):
        return "generate_final_answer"

    return "no_context"


# ============================================================
# 4. INITIALIZE RESEARCH RUN
# ============================================================

async def init_research_run(
    state: ResearchState,
    config: RunnableConfig,
) -> dict:

    logger.info(
        "Graph Node: init_research_run"
    )

    session = config["configurable"]["db_session"]

    allowed_departments = state.get(
        "allowed_departments"
    )

    dept_id = (
        allowed_departments[0]
        if allowed_departments
        else None
    )

    run = ResearchRun(
        created_by=state["user"].id,
        department_id=dept_id,
        status="PENDING",
        query=state["query"],
    )

    session.add(run)
    await session.flush()

    return {
        "research_run_id": str(run.id),
    }


# ============================================================
# 5G - WEB SEARCH
# ============================================================

async def search_web(
    state: ResearchState,
) -> dict:

    logger.info(
        "Graph Node: search_web"
    )

    registry = get_provider_registry()

    try:
        provider = registry.get("firecrawl")
    except Exception as exc:
        logger.warning(
            "firecrawl_provider_not_found",
            extra={
                "event": "firecrawl_provider_not_found",
                "error": str(exc),
            },
        )

        return {
            "search_results": [],
        }

    try:
        response = await provider.search(
            state["query"],
            limit=5,
        )

        return {
            "search_results": response.results,
        }

    except Exception as exc:
        logger.error(
            "web_search_failed",
            extra={
                "event": "web_search_failed",
                "error": str(exc),
            },
        )

        return {
            "search_results": [],
        }


# ============================================================
# 5G - SOURCE SELECTION
# ============================================================

async def select_sources(
    state: ResearchState,
) -> dict:

    logger.info(
        "Graph Node: select_sources"
    )

    search_results = state.get(
        "search_results",
        [],
    )

    if not search_results:
        return {
            "selected_urls": [],
        }

    llm = get_llm()

    results_text = "\n".join(
        (
            f"[{i + 1}] {result.url}\n"
            f"Title: {result.title}\n"
            f"Snippet: {result.snippet}"
        )
        for i, result in enumerate(search_results)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Select the most relevant URLs likely to "
                "answer the user's question. "
                "Select at most 3 URLs. "
                "If none are relevant, return an empty list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {state['query']}\n\n"
                f"Search Results:\n{results_text}"
            ),
        },
    ]

    try:
        structured_llm = (
            llm.llm.with_structured_output(
                SourceSelection,
                include_raw=True,
            )
        )

        _sel_start = time.perf_counter()
        inv_res = await structured_llm.ainvoke(messages)
        _sel_latency = (time.perf_counter() - _sel_start) * 1000

        raw_msg = inv_res.get("raw") if isinstance(inv_res, dict) else None
        usage_meta = getattr(raw_msg, "usage_metadata", None) or {}
        record_llm_call(
            stage_name="research",
            usage=usage_meta,
            latency_ms=_sel_latency,
            model_name=getattr(llm.llm, "model_name", None),
        )

        result: SourceSelection = (
            inv_res.get("parsed") if isinstance(inv_res, dict) else inv_res
        )

        valid_urls = {
            item.url
            for item in search_results
        }

        selected = [
            url
            for url in (result.selected_urls if result else [])
            if url in valid_urls
        ][:3]

        logger.info(
            "source_selection_completed",
            extra={
                "event": "source_selection_completed",
                "selected_count": len(selected),
            },
        )

        return {
            "selected_urls": selected,
        }

    except Exception as exc:
        logger.error(
            "source_selection_failed",
            extra={
                "event": "source_selection_failed",
                "error": str(exc),
            },
        )

        fallback_urls = [
            item.url
            for item in search_results[:2]
        ]

        return {
            "selected_urls": fallback_urls,
        }


# ============================================================
# 5G - CRAWL / SCRAPE
# ============================================================

async def crawl_web(
    state: ResearchState,
) -> dict:

    logger.info(
        "Graph Node: crawl_web"
    )

    selected_urls = state.get(
        "selected_urls",
        [],
    )

    if not selected_urls:
        return {
            "crawled_pages": [],
        }

    registry = get_provider_registry()

    try:
        provider = registry.get("firecrawl")
    except Exception as exc:
        logger.warning(
            "firecrawl_provider_not_found",
            extra={
                "event": "firecrawl_provider_not_found",
                "error": str(exc),
            },
        )

        return {
            "crawled_pages": [],
        }

    crawled_pages = []

    for url in selected_urls:
        try:
            page = await provider.scrape(url)
            crawled_pages.append(page)

        except Exception as exc:
            logger.error(
                "web_scrape_failed",
                extra={
                    "event": "web_scrape_failed",
                    "url": url,
                    "error": str(exc),
                },
            )

    return {
        "crawled_pages": crawled_pages,
    }


# ============================================================
# 5H - CLEAN
# ============================================================

async def clean_web(
    state: ResearchState,
) -> dict:

    logger.info(
        "Graph Node: clean_web"
    )

    crawled_pages = state.get(
        "crawled_pages",
        [],
    )

    if not crawled_pages:
        return {
            "clean_pages": [],
        }

    clean_pages = clean_and_deduplicate(
        crawled_pages
    )

    logger.info(
        "web_cleaning_completed",
        extra={
            "event": "web_cleaning_completed",
            "raw_pages": len(crawled_pages),
            "clean_pages": len(clean_pages),
        },
    )

    return {
        "clean_pages": clean_pages,
    }


# ============================================================
# 5I - EXTRACT + CLASSIFY
# ============================================================

async def extract_knowledge_node(
    state: ResearchState,
) -> dict:

    logger.info(
        "Graph Node: extract_knowledge"
    )

    clean_pages = state.get(
        "clean_pages",
        [],
    )

    if not clean_pages:
        return {
            "extracted_knowledge": None,
        }

    all_facts = []

    for page in clean_pages:

        candidate = await extractor_func(
            page
        )

        if candidate and candidate.facts:
            all_facts.extend(
                candidate.facts
            )

    if not all_facts:
        return {
            "extracted_knowledge": None,
        }

    return {
        "extracted_knowledge": (
            StructuredKnowledgeCandidate(
                facts=all_facts
            )
        ),
    }


# ============================================================
# 5J - PERSIST KNOWLEDGE
# ============================================================

async def persist_knowledge_node(
    state: ResearchState,
    config: RunnableConfig,
) -> dict:

    logger.info(
        "Graph Node: persist_knowledge"
    )

    session = config["configurable"]["db_session"]

    candidate = state.get(
        "extracted_knowledge"
    )

    run_id_str = state.get(
        "research_run_id"
    )

    if (
        candidate
        and candidate.facts
        and run_id_str
    ):
        await persist_knowledge(
            session,
            run_id_str,
            candidate,
        )

    return {}


# ============================================================
# 5K - INGEST INTO RAG
# ============================================================

async def ingest_into_rag_node(
    state: ResearchState,
    config: RunnableConfig,
) -> dict:

    logger.info(
        "Graph Node: ingest_into_rag"
    )

    session = config["configurable"]["db_session"]

    run_id_str = state.get(
        "research_run_id"
    )

    candidate = state.get(
        "extracted_knowledge"
    )

    if not (
        candidate
        and candidate.facts
        and run_id_str
    ):
        return {}

    result = await session.execute(
        select(ResearchKnowledge).where(
            ResearchKnowledge.research_run_id
            == uuid.UUID(run_id_str)
        )
    )

    records = list(
        result.scalars().all()
    )

    if records:
        await ingest_knowledge_to_rag(
            session,
            run_id_str,
            records,
        )

    return {}


# ============================================================
# FINAL GENERATION
# ============================================================

async def generate_final_answer(
    state: ResearchState,
    config: RunnableConfig,
) -> dict:

    logger.info(
        "Graph Node: generate_final_answer"
    )

    generation_service: GenerationService = (
        config["configurable"]["generation_service"]
    )

    context_compiler: ContextCompiler = (
        config.get("configurable", {}).get("context_compiler")
        or ContextCompiler()
    )

    # Always initialize this.
    # This avoids any accidental reference-before-assignment.
    web_chunks: list[RetrievedChunk] = []

    # --------------------------------------------------------
    # PATH A — INTERNAL RAG WAS SUFFICIENT
    # --------------------------------------------------------

    if state.get("is_sufficient"):

        raw_chunks = state.get(
            "retrieved_chunks",
            [],
        )
        compiled_context = context_compiler.compile(
            chunks=raw_chunks,
            query=state["query"],
            allowed_departments=state["allowed_departments"],
            is_web_context=False,
            strategy_trace=state.get("strategy_trace"),
            chunk_provenance=state.get("chunk_provenance"),
        )

        response = (
            await generation_service.generate_from_context(
                query=state["query"],
                chunks=compiled_context.chunks,
                allowed_departments=(
                    state["allowed_departments"]
                ),
                is_web_context=False,
                conflicts=compiled_context.conflicts,
                chunk_hierarchy=compiled_context.chunk_hierarchy,
                rendered_context=compiled_context.rendered_context,
                conversation_history=state.get("conversation_history"),
                workspace_thread_context=state.get("workspace_thread_context"),
                workspace_id=state.get("workspace_id"),
            )
        )

    # --------------------------------------------------------
    # PATH B — RESEARCH WAS REQUIRED
    # --------------------------------------------------------

    else:

        candidate = state.get(
            "extracted_knowledge"
        )

        if candidate and candidate.facts:

            retrieval_service: RetrievalService = (
                config["configurable"]["retrieval_service"]
            )

            retrieval_response = (
                await retrieval_service.retrieve(
                    query=state["query"],
                    top_k=state["top_k"],
                    rerank=state["rerank"],
                    allowed_departments=(
                        state["allowed_departments"]
                    ),
                    workspace_id=state.get("workspace_id"),
                )
            )

            web_chunks = (
                retrieval_response.results
            )

        if web_chunks:

            compiled_context = context_compiler.compile(
                chunks=web_chunks,
                query=state["query"],
                allowed_departments=state["allowed_departments"],
                is_web_context=True,
                strategy_trace=(
                    retrieval_response.strategy_trace
                    if "retrieval_response" in locals()
                    else None
                ),
                chunk_provenance=(
                    retrieval_response.chunk_provenance
                    if "retrieval_response" in locals()
                    else None
                ),
                chunk_hierarchy=(
                    retrieval_response.chunk_hierarchy
                    if "retrieval_response" in locals()
                    else None
                ),
            )

            response = (
                await generation_service.generate_from_context(
                    query=state["query"],
                    chunks=compiled_context.chunks,
                    allowed_departments=(
                        state["allowed_departments"]
                    ),
                    is_web_context=True,
                    conflicts=compiled_context.conflicts,
                    chunk_hierarchy=compiled_context.chunk_hierarchy,
                    rendered_context=compiled_context.rendered_context,
                    conversation_history=state.get("conversation_history"),
                )
            )

        # ----------------------------------------------------
        # Research completely failed
        # ----------------------------------------------------

        else:

            response = GenerateResponse(
                answer=(
                    "I could not verify the information "
                    "to provide a reliable answer."
                ),
                citations=[],
                retrieved_chunks=[],
            )

    # ========================================================
    # RESEARCH RUN STATUS
    # ========================================================

    run_id_str = state.get(
        "research_run_id"
    )

    if run_id_str:

        session = config[
            "configurable"
        ]["db_session"]

        run = await session.get(
            ResearchRun,
            uuid.UUID(run_id_str),
        )

        if run:

            if state.get("is_sufficient"):
                run.status = "COMPLETED"

            elif web_chunks:
                run.status = "COMPLETED"

            else:
                run.status = "FAILED"

            await session.flush()

    return {
        "final_response": response,
    }


# ============================================================
# GRAPH
# ============================================================

builder = StateGraph(
    ResearchState
)

# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

builder.add_node(
    "classify_query",
    classify_query,
)

builder.add_node(
    "handle_conversation",
    handle_conversation,
)

builder.add_node(
    "retrieve_rag",
    retrieve_rag,
)

builder.add_node(
    "evaluate_sufficiency",
    evaluate_sufficiency,
)

builder.add_node(
    "init_research_run",
    init_research_run,
)

builder.add_node(
    "search_web",
    search_web,
)

builder.add_node(
    "select_sources",
    select_sources,
)

builder.add_node(
    "crawl_web",
    crawl_web,
)

builder.add_node(
    "clean_web",
    clean_web,
)

builder.add_node(
    "extract_knowledge",
    extract_knowledge_node,
)

builder.add_node(
    "persist_knowledge",
    persist_knowledge_node,
)

builder.add_node(
    "ingest_into_rag",
    ingest_into_rag_node,
)

builder.add_node(
    "generate_final_answer",
    generate_final_answer,
)

builder.add_node(
    "no_context",
    no_context,
)

# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

builder.add_edge(
    START,
    "classify_query",
)

builder.add_conditional_edges(
    "classify_query",
    route_after_classification,
    {
        "conversation": "handle_conversation",
        "rag": "retrieve_rag",
    },
)

# ------------------------------------------------------------
# CONVERSATION
# ------------------------------------------------------------

builder.add_edge(
    "handle_conversation",
    END,
)

# ------------------------------------------------------------
# RAG
# ------------------------------------------------------------

builder.add_edge(
    "retrieve_rag",
    "evaluate_sufficiency",
)

builder.add_conditional_edges(
    "evaluate_sufficiency",
    decide_next_after_eval,
    {
        "generate_final_answer":
            "generate_final_answer",

        "no_context":
            "no_context",
    },
)

builder.add_edge(
    "no_context",
    END,
)

# ------------------------------------------------------------
# RESEARCH ENGINE
# ------------------------------------------------------------

builder.add_edge(
    "init_research_run",
    "search_web",
)

builder.add_edge(
    "search_web",
    "select_sources",
)

builder.add_edge(
    "select_sources",
    "crawl_web",
)

builder.add_edge(
    "crawl_web",
    "clean_web",
)

builder.add_edge(
    "clean_web",
    "extract_knowledge",
)

builder.add_edge(
    "extract_knowledge",
    "persist_knowledge",
)

builder.add_edge(
    "persist_knowledge",
    "ingest_into_rag",
)

builder.add_edge(
    "ingest_into_rag",
    "generate_final_answer",
)

# ------------------------------------------------------------
# END
# ------------------------------------------------------------

builder.add_edge(
    "generate_final_answer",
    END,
)

research_graph = builder.compile()