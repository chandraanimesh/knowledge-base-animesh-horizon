import logging
import os
import time
import uuid
from typing import Any, Literal, TypedDict

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
from rag_app.schemas.users import UserContext
from rag_app.generation.llm_factory import get_llm
from rag_app.generation.service import GenerationService
from rag_app.retrieval.query_parser import is_greeting, detect_web_search_intent, format_web_search_query
from rag_app.retrieval.service import RetrievalService
from rag_app.context.compiler import ContextCompiler
import ipaddress
import urllib.parse
from rag_app.research.cleaner import clean_and_deduplicate
from rag_app.research.extractor import (
    extract_knowledge as extractor_func,
)
from rag_app.research.persistence import (
    create_transient_web_chunks,
    persist_knowledge,
)
from rag_app.research.provider import (
    WebPageResult,
    WebSearchResult,
)
from rag_app.research.registry import get_provider_registry
from rag_app.schemas.rag import ChunkHierarchyMetadata, ChunkProvenance, GenerateResponse, QueryPlan, RetrievedChunk
from rag_app.schemas.research import (
    CleanWebPage,
    SourceSelection,
    StructuredKnowledgeCandidate,
    SufficiencyEvaluation,
    SufficiencyStatus,
)

from rag_app.schemas.artifacts import OutputPreferences

logger = logging.getLogger(__name__)


def is_safe_web_url(url: str) -> bool:
    """SSRF guard: reject non-http(s), localhost, private/reserved IP addresses."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        hostname_lower = hostname.lower()
        if hostname_lower in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False
        try:
            ip = ipaddress.ip_address(hostname_lower)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            if hostname_lower.endswith(".local") or hostname_lower.endswith(".internal"):
                return False
        return True
    except Exception:
        return False


# ============================================================
# STATE
# ============================================================

class ResearchState(TypedDict):
    query: str
    top_k: int
    rerank: bool
    user: User | UserContext | Any
    allowed_departments: list[uuid.UUID] | None
    workspace_id: uuid.UUID | None

    # Query routing
    query_type: Literal["conversation", "knowledge"] | None
    query_plan: QueryPlan | None

    # Internal RAG
    retrieved_chunks: list[RetrievedChunk]
    strategy_trace: dict[str, int] | None
    chunk_provenance: dict[str, ChunkProvenance] | None
    chunk_hierarchy: dict[str, ChunkHierarchyMetadata] | None
    is_sufficient: bool | None
    sufficiency_status: str | None
    sufficiency_evaluation: SufficiencyEvaluation | None

    # Research
    research_run_id: str | None
    search_results: list[WebSearchResult]
    selected_urls: list[str]
    crawled_pages: list[WebPageResult]
    clean_pages: list[CleanWebPage]
    extracted_knowledge: StructuredKnowledgeCandidate | None
    web_chunks: list[RetrievedChunk] | None
    web_research_error: str | None

    # Conversational context
    conversation_history: list[dict[str, str]] | None
    workspace_thread_context: list[dict] | None

    # Final response
    final_response: GenerateResponse | None
    output_preferences: OutputPreferences | None
    conversation_id: uuid.UUID | None
    raw_query: str | None

    # Workspace provider configuration
    web_provider: str | None
    web_credential_ref: str | None
    web_enabled: bool | None

    # Workspace MCP read-only tools
    mcp_tools_called: list[str] | None
    mcp_chunks: list[RetrievedChunk] | None


# ============================================================
# 0. QUERY TYPE ROUTER
# ============================================================

async def classify_query(
    state: ResearchState,
) -> dict:
    """
    Lightweight deterministic routing before RAG.
    Also extracts query-planning intent (e.g. explicit web search or freshness).
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
            "query_plan": None,
        }

    # Evaluate query-planning web intent deterministically
    web_needed, web_reason, int_needed = detect_web_search_intent(query)
    current_plan = state.get("query_plan")
    if current_plan is None:
        current_plan = QueryPlan(
            semantic_query=query,
            web_search_needed=web_needed,
            web_search_reason=web_reason,
            internal_retrieval_needed=int_needed,
        )
    elif web_needed and not current_plan.web_search_needed:
        current_plan = current_plan.model_copy(update={
            "web_search_needed": web_needed,
            "web_search_reason": web_reason,
            "internal_retrieval_needed": int_needed,
        })

    return {
        "query_type": "knowledge",
        "query_plan": current_plan,
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

    results = list(response.results)
    strategy_trace = dict(response.strategy_trace) if response.strategy_trace else {}
    chunk_provenance = dict(response.chunk_provenance) if response.chunk_provenance else {}
    chunk_hierarchy = dict(response.chunk_hierarchy) if response.chunk_hierarchy else {}

    # Stage D: Integrate approved shared knowledge into Global Chat only
    if state.get("workspace_id") is None:
        db_session = config.get("configurable", {}).get("db_session")
        if db_session:
            try:
                from rag_app.knowledge.candidate_service import KnowledgeCandidateService
                kn_service = KnowledgeCandidateService(db_session)
                shared_chunks = await kn_service.retrieve_shared_knowledge_as_chunks(
                    query=state["query"],
                    allowed_departments=state.get("allowed_departments"),
                    limit=5,
                )
                if shared_chunks:
                    for sc in shared_chunks:
                        results.append(sc)
                        if sc.chunk_id:
                            chunk_provenance[sc.chunk_id] = ChunkProvenance(
                                chunk_id=sc.chunk_id,
                                strategies=["SHARED_KNOWLEDGE"],
                                knowledge_item_id=sc.chunk_id.replace("shared_kn_", ""),
                                confidence=sc.hybrid_score,
                            )
                    strategy_trace["shared_knowledge"] = len(shared_chunks)
            except Exception as e:
                logger.warning(
                    f"Approved shared knowledge retrieval failed: {e}",
                    extra={"event": "shared_knowledge_retrieval_exception", "error": str(e)},
                    exc_info=True,
                )

    ret_dict = {
        "retrieved_chunks": results,
        "strategy_trace": strategy_trace,
        "chunk_provenance": chunk_provenance,
        "chunk_hierarchy": chunk_hierarchy,
    }

    # Resolve workspace provider profile web settings
    db_session = config.get("configurable", {}).get("db_session")
    if state.get("workspace_id") and db_session:
        from rag_app.db.models import WorkspaceProviderProfile
        try:
            ws_prof = await db_session.scalar(
                select(WorkspaceProviderProfile).where(
                    WorkspaceProviderProfile.workspace_id == state.get("workspace_id"),
                    WorkspaceProviderProfile.is_active.is_(True),
                )
            )
            if ws_prof:
                if ws_prof.web_provider:
                    ret_dict["web_provider"] = ws_prof.web_provider
                if ws_prof.web_credential_ref:
                    ret_dict["web_credential_ref"] = ws_prof.web_credential_ref
                if ws_prof.configuration and "web_enabled" in ws_prof.configuration:
                    ret_dict["web_enabled"] = ws_prof.configuration["web_enabled"]
        except Exception as e:
            logger.warning("failed_to_resolve_workspace_web_settings", extra={"error": str(e)})

    # Propagate parsed QueryPlan back to ResearchState if present
    if getattr(response, "query_plan", None) is not None:
        curr_plan = state.get("query_plan")
        if curr_plan is not None:
            ret_dict["query_plan"] = response.query_plan.model_copy(update={
                "web_search_needed": curr_plan.web_search_needed or response.query_plan.web_search_needed,
                "web_search_reason": curr_plan.web_search_reason or response.query_plan.web_search_reason,
                "internal_retrieval_needed": curr_plan.internal_retrieval_needed,
            })
        else:
            ret_dict["query_plan"] = response.query_plan

    return ret_dict


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

    record_stage_start("research")
    try:
        from rag_app.retrieval.sufficiency import EvidenceSufficiencyEvaluator
        evaluator = EvidenceSufficiencyEvaluator()

        result = await evaluator.evaluate(
            query=state["query"],
            chunks=chunks,
            query_plan=state.get("query_plan"),
            chunk_hierarchy=state.get("chunk_hierarchy"),
            conversation_history=state.get("conversation_history"),
            workspace_thread_context=state.get("workspace_thread_context"),
        )

        from rag_app.core.observability import record_sufficiency_metrics, record_failure_category
        from rag_app.schemas.rag import FailureCategory
        from rag_app.schemas.research import SufficiencyStatus

        status_str = result.status.value if hasattr(result.status, "value") else str(result.status)
        record_sufficiency_metrics(
            sufficiency=result.is_sufficient,
            status=status_str,
            evidence_found=result.evidence_found,
            confidence=result.confidence,
            coverage=result.coverage,
            reason=result.reason,
            missing_requirements_count=len(result.missing_requirements),
            contradiction_count=len(result.contradictions),
            evidence_types=result.evidence_types,
        )
        if result.status == SufficiencyStatus.INSUFFICIENT:
            record_failure_category(FailureCategory.SUFFICIENCY)

        logger.info(
            "sufficiency_evaluation",
            extra={
                "event": "sufficiency_evaluation",
                "query": state["query"],
                "candidate_count": len(chunks),
                "status": status_str,
                "is_sufficient": result.is_sufficient,
                "evidence_found": result.evidence_found,
                "required_count": len(result.required_information),
                "supported_count": len(result.supported_information),
                "missing_count": len(result.missing_requirements),
                "contradiction_count": len(result.contradictions),
                "answerable_from_context": result.answerable_from_context,
                "reason": result.reason,
                "confidence": result.confidence,
                "coverage": result.coverage,
                "evidence_types": result.evidence_types,
            },
        )

        return {
            "is_sufficient": result.is_sufficient,
            "sufficiency_status": status_str,
            "sufficiency_evaluation": result,
        }
    finally:
        record_stage_end("research")


def decide_next_after_eval(
    state: ResearchState,
) -> Literal[
    "generate_final_answer",
    "query_workspace_mcp",
    "init_research_run",
    "no_context",
]:
    from rag_app.schemas.research import SufficiencyStatus, SufficiencyEvaluation
    settings = get_settings()
    suff_eval = state.get("sufficiency_evaluation")
    query_plan = state.get("query_plan")

    is_web_enabled = (
        state.get("web_enabled")
        if state.get("web_enabled") is not None
        else getattr(settings, "enable_web_research", False)
    )

    # 1. Explicit Web Search Intent: if web research is enabled and explicitly requested in query plan,
    # route directly to web research run (even if internal chunks exist).
    if is_web_enabled and query_plan and getattr(query_plan, "web_search_needed", False):
        return "init_research_run"

    # Evaluate internal evidence sufficiency
    is_internal_sufficient = False
    suff_status = state.get("sufficiency_status")
    if suff_status in ("sufficient", "partial"):
        is_internal_sufficient = True
    elif isinstance(suff_eval, SufficiencyEvaluation):
        if suff_eval.status in (SufficiencyStatus.SUFFICIENT, SufficiencyStatus.PARTIAL):
            is_internal_sufficient = True
    elif isinstance(suff_eval, dict):
        status_val = suff_eval.get("status")
        if status_val in (SufficiencyStatus.SUFFICIENT, SufficiencyStatus.PARTIAL, "sufficient", "partial"):
            is_internal_sufficient = True
    elif state.get("is_sufficient"):
        is_internal_sufficient = True

    # Fast-Path Preservation: If internal RAG is sufficient, immediately generate answer (MCP bypassed)
    if is_internal_sufficient:
        return "generate_final_answer"

    # 2. If internal evidence is insufficient and workspace_id is present with CRM/MCP intent:
    query_str = (state.get("query") or "").lower()
    has_mcp_intent = any(term in query_str for term in ("crm", "twenty", "workspace mcp", "mcp tool", "in our crm", "crm record"))
    if state.get("workspace_id") and has_mcp_intent:
        return "query_workspace_mcp"

    # Fallback to web research if enabled
    if is_web_enabled:
        return "init_research_run"

    return "no_context"


async def query_workspace_mcp_node(
    state: ResearchState,
    config: RunnableConfig,
) -> dict:
    """
    LangGraph node: executes explicitly enabled and server-verified read-only MCP tools
    bound to the current workspace.
    """
    workspace_id = state.get("workspace_id")
    user = state.get("user")
    db_session = config.get("configurable", {}).get("db_session")

    if not workspace_id or not user or not db_session:
        return {"mcp_chunks": []}

    from rag_app.mcp.langgraph_binding import WorkspaceReadOnlyMCPToolBinding
    from rag_app.db.models import User as UserModel
    from rag_app.schemas.users import UserContext

    user_obj = user
    if isinstance(user, UserContext) or not hasattr(user, "role"):
        user_id = getattr(user, "id", None)
        if user_id:
            user_obj = await db_session.get(UserModel, user_id)

    if not user_obj:
        return {"mcp_chunks": []}

    binding = WorkspaceReadOnlyMCPToolBinding(session=db_session)
    tools = await binding.get_workspace_readonly_tools(
        user=user_obj,
        workspace_id=workspace_id,
    )

    if not tools:
        return {"mcp_chunks": []}

    chunks = list(state.get("retrieved_chunks") or [])
    mcp_chunks = []
    tools_called = []

    # Select best matching tool based on query tokens or fallback to first
    query_lower = state["query"].lower()
    selected_tool = tools[0]
    for t in tools:
        clean_name = t.name.replace("mcp_", "").lower()
        if any(term in clean_name for term in query_lower.split()):
            selected_tool = t
            break

    try:
        output = await selected_tool.ainvoke({"query": state["query"]})
        tools_called.append(selected_tool.name)

        if output and not output.startswith("Tool Execution Error:") and not output.startswith("Tool Execution Rejected"):
            dept_id_str = str(state.get("allowed_departments")[0]) if state.get("allowed_departments") else "None"
            chunk = RetrievedChunk(
                chunk_id=f"mcp_{uuid.uuid4().hex[:8]}",
                document_id="doc_workspace_mcp",
                department_id=dept_id_str,
                file_name=f"Workspace MCP: {selected_tool.name}",
                chunk_index=0,
                text=output,
                page_number=1,
            )
            chunks.append(chunk)
            mcp_chunks.append(chunk)
    except Exception as exc:
        logger.warning("Error querying workspace MCP tool in LangGraph: %s", exc)

    return {
        "retrieved_chunks": chunks,
        "mcp_chunks": mcp_chunks,
        "mcp_tools_called": tools_called,
    }


def decide_after_mcp(
    state: ResearchState,
) -> Literal[
    "generate_final_answer",
    "init_research_run",
    "no_context",
]:
    # If MCP tools provided grounded evidence, generate final answer directly
    if state.get("mcp_chunks"):
        return "generate_final_answer"

    settings = get_settings()
    is_web_enabled = (
        state.get("web_enabled")
        if state.get("web_enabled") is not None
        else getattr(settings, "enable_web_research", False)
    )
    if is_web_enabled:
        return "init_research_run"

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

    session = config.get("configurable", {}).get("db_session")

    allowed_departments = state.get(
        "allowed_departments"
    )

    dept_id = (
        allowed_departments[0]
        if allowed_departments
        else None
    )

    run_id = uuid.uuid4()
    if session is not None:
        try:
            user_val = state.get("user")
            created_by = getattr(user_val, "id", user_val) if user_val else None
            if not isinstance(created_by, uuid.UUID) and created_by is not None:
                try:
                    created_by = uuid.UUID(str(created_by))
                except (ValueError, TypeError):
                    created_by = None

            run = ResearchRun(
                id=run_id,
                created_by=created_by,
                department_id=dept_id,
                status="PENDING",
                query=state["query"],
            )
            session.add(run)
            await session.flush()
        except Exception as exc:
            logger.warning(
                "init_research_run_failed_to_persist",
                extra={"event": "init_research_run_failed_to_persist", "error": str(exc)},
            )

    current_plan = state.get("query_plan")
    updated_plan = current_plan
    if current_plan is not None and not current_plan.web_search_needed:
        updated_plan = current_plan.model_copy(update={
            "web_search_needed": True,
            "web_search_reason": current_plan.web_search_reason or "insufficient_internal_evidence",
        })

    return {
        "research_run_id": str(run_id),
        "query_plan": updated_plan,
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
    settings = get_settings()
    provider_name = (
        state.get("web_provider")
        or getattr(settings, "web_search_provider", "tinyfish")
    )

    cred_ref = state.get("web_credential_ref")
    api_key_override = None
    if cred_ref:
        # Resolve from environment variable reference
        api_key_override = os.getenv(cred_ref) or (cred_ref if not cred_ref.isupper() and len(cred_ref) > 10 else None)

    provider = None
    if registry.has(provider_name) and not api_key_override:
        try:
            provider = registry.get(provider_name)
        except Exception:
            provider = None

    if provider is None:
        try:
            from rag_app.providers.registry import get_provider, provider_registry
            if provider_registry.has_provider("web", provider_name):
                kwargs = {}
                if api_key_override:
                    kwargs["api_key"] = api_key_override
                provider = get_provider("web", provider_name, **kwargs)
        except Exception:
            provider = None

    if provider is None:
        try:
            alt_name = "tinyfish" if provider_name != "tinyfish" else "firecrawl"
            provider = registry.get(alt_name)
        except Exception as exc:
            logger.warning(
                "web_provider_not_found",
                extra={
                    "event": "web_provider_not_found",
                    "provider": provider_name,
                    "error": str(exc),
                },
            )
            return {
                "search_results": [],
            }

    raw_query = state["query"]
    query_plan = state.get("query_plan")
    formatted_query, formulation_source = format_web_search_query(raw_query, query_plan)

    logger.info(
        "web_query_formulated",
        extra={
            "event": "web_query_formulated",
            "raw_query": raw_query,
            "formatted_query": formatted_query,
            "formulation_source": formulation_source,
        },
    )

    logger.info(
        "web_search_attempted",
        extra={
            "event": "web_search_attempted",
            "query": formatted_query,
            "provider": provider_name,
            "limit": 10,
        },
    )

    _search_start = time.perf_counter()
    try:
        response = await provider.search(
            formatted_query,
            limit=10,
        )
        _search_latency = (time.perf_counter() - _search_start) * 1000

        logger.info(
            "web_search_completed",
            extra={
                "event": "web_search_completed",
                "provider": provider_name,
                "result_count": len(response.results),
                "latency_ms": _search_latency,
            },
        )

        return {
            "search_results": response.results,
            "web_provider": provider_name,
        }

    except Exception as exc:
        _search_latency = (time.perf_counter() - _search_start) * 1000
        logger.error(
            "web_search_failed",
            extra={
                "event": "web_search_failed",
                "provider": provider_name,
                "error": str(exc),
                "latency_ms": _search_latency,
            },
        )

        # Fallback to available alternative registered provider
        alt_name = "tinyfish" if provider_name != "tinyfish" else "firecrawl"
        if registry.has(alt_name):
            try:
                alt_provider = registry.get(alt_name)
                logger.info(
                    "web_search_fallback_attempted",
                    extra={
                        "event": "web_search_fallback_attempted",
                        "failed_provider": provider_name,
                        "fallback_provider": alt_name,
                        "error": str(exc),
                    },
                )
                _fb_start = time.perf_counter()
                fb_response = await alt_provider.search(
                    formatted_query,
                    limit=10,
                )
                _fb_latency = (time.perf_counter() - _fb_start) * 1000
                logger.info(
                    "web_search_completed",
                    extra={
                        "event": "web_search_completed",
                        "provider": alt_name,
                        "result_count": len(fb_response.results),
                        "latency_ms": _fb_latency,
                        "fallback": True,
                    },
                )
                return {
                    "search_results": fb_response.results,
                    "web_provider": alt_name,
                }
            except Exception as fb_exc:
                logger.error(
                    "web_search_fallback_failed",
                    extra={
                        "event": "web_search_fallback_failed",
                        "provider": alt_name,
                        "error": str(fb_exc),
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
            "web_source_selected",
            extra={
                "event": "web_source_selected",
                "selected_count": len(selected),
                "selected_urls": selected,
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

def _extract_domain(url: str) -> str:
    try:
        return (urllib.parse.urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _classify_scrape_error(exc: Exception) -> tuple[str, bool]:
    """Returns (error_category: str, is_retryable: bool)"""
    err_str = str(exc).lower()
    if "bot_blocked" in err_str or "cloudflare" in err_str or "captcha" in err_str:
        return "bot_blocked", False
    if "403" in err_str or "forbidden" in err_str:
        return "http_403", False
    if "404" in err_str or "not found" in err_str:
        return "http_404", False
    if "429" in err_str or "rate limit" in err_str:
        return "rate_limited", True
    if "timeout" in err_str or "timed out" in err_str:
        return "timeout", True
    return "provider_error", False


async def crawl_web(
    state: ResearchState,
) -> dict:

    logger.info(
        "Graph Node: crawl_web"
    )

    selected_urls = list(state.get("selected_urls", []))
    search_results = state.get("search_results", [])

    if not selected_urls and not search_results:
        return {
            "crawled_pages": [],
        }

    registry = get_provider_registry()
    settings = get_settings()
    provider_name = (
        state.get("web_provider")
        or getattr(settings, "web_search_provider", "tinyfish")
    )

    provider = None
    if registry.has(provider_name):
        try:
            provider = registry.get(provider_name)
        except Exception:
            provider = None

    if provider is None:
        try:
            from rag_app.providers.registry import get_provider, provider_registry
            if provider_registry.has_provider("web", provider_name):
                provider = get_provider("web", provider_name)
        except Exception:
            provider = None

    if provider is None:
        try:
            alt_name = "tinyfish" if provider_name != "tinyfish" else "firecrawl"
            provider = registry.get(alt_name)
        except Exception as exc:
            logger.warning(
                "web_provider_not_found",
                extra={
                    "event": "web_provider_not_found",
                    "provider": provider_name,
                    "error": str(exc),
                },
            )
            return {
                "crawled_pages": [],
            }

    target_usable_pages = getattr(settings, "web_research_target_pages", 2)
    max_scrapes = getattr(settings, "web_research_max_scrapes", 4)
    crawled_pages: list[WebPageResult] = []
    attempted_urls: set[str] = set()
    attempted_domains: set[str] = set()
    total_scrapes = 0

    async def _try_scrape_candidate(target_url: str, is_fallback: bool = False) -> bool:
        nonlocal total_scrapes
        domain = _extract_domain(target_url)
        attempted_urls.add(target_url)
        attempted_domains.add(domain)
        total_scrapes += 1

        if not is_safe_web_url(target_url):
            logger.warning(
                "unsafe_web_url_skipped",
                extra={"event": "unsafe_web_url_skipped", "url": target_url},
            )
            return False

        if is_fallback:
            logger.info(
                "web_fallback_attempted",
                extra={
                    "event": "web_fallback_attempted",
                    "url": target_url,
                    "domain": domain,
                    "attempt_index": total_scrapes,
                },
            )

        logger.info(
            "web_source_fetch_attempted",
            extra={
                "event": "web_source_fetch_attempted",
                "url": target_url,
                "domain": domain,
                "attempt_index": total_scrapes,
            },
        )

        try:
            page = await provider.scrape(target_url)
            if page and page.markdown and page.markdown.strip():
                crawled_pages.append(page)
                logger.info(
                    "web_source_fetch_completed",
                    extra={
                        "event": "web_source_fetch_completed",
                        "url": target_url,
                        "domain": domain,
                        "content_length": len(page.markdown),
                    },
                )
                return True
            else:
                logger.warning(
                    "web_source_fetch_empty",
                    extra={"event": "web_source_fetch_empty", "url": target_url, "domain": domain},
                )
                return False
        except Exception as exc:
            alt_name = "tinyfish" if provider_name != "tinyfish" else "firecrawl"
            if registry.has(alt_name):
                try:
                    alt_provider = registry.get(alt_name)
                    logger.info(
                        "web_source_fetch_provider_fallback",
                        extra={
                            "event": "web_source_fetch_provider_fallback",
                            "url": target_url,
                            "failed_provider": provider_name,
                            "fallback_provider": alt_name,
                            "error": str(exc),
                        },
                    )
                    page = await alt_provider.scrape(target_url)
                    if page and page.markdown and page.markdown.strip():
                        crawled_pages.append(page)
                        logger.info(
                            "web_source_fetch_completed",
                            extra={
                                "event": "web_source_fetch_completed",
                                "url": target_url,
                                "domain": domain,
                                "content_length": len(page.markdown),
                                "provider": alt_name,
                            },
                        )
                        return True
                except Exception:
                    pass

            err_cat, is_retryable = _classify_scrape_error(exc)
            logger.error(
                "web_source_fetch_failed",
                extra={
                    "event": "web_source_fetch_failed",
                    "url": target_url,
                    "domain": domain,
                    "error_category": err_cat,
                    "is_retryable": is_retryable,
                    "error": str(exc),
                },
            )
            return False

    # 1. Scrape initial selected candidates
    for url in selected_urls:
        if len(crawled_pages) >= target_usable_pages or total_scrapes >= max_scrapes:
            break
        await _try_scrape_candidate(url, is_fallback=False)

    # 2. Bounded Candidate-Pool Fallback: If target usable pages not met, try remaining search candidates
    if len(crawled_pages) < target_usable_pages and total_scrapes < max_scrapes and search_results:
        # Prioritize candidates from unattempted domains for diversity
        unattempted_candidates = [
            item.url for item in search_results
            if item.url not in attempted_urls and is_safe_web_url(item.url)
        ]
        diverse_candidates = [
            u for u in unattempted_candidates if _extract_domain(u) not in attempted_domains
        ]
        remaining_candidates = [
            u for u in unattempted_candidates if u not in diverse_candidates
        ]
        fallback_queue = diverse_candidates + remaining_candidates

        for fb_url in fallback_queue:
            if len(crawled_pages) >= target_usable_pages or total_scrapes >= max_scrapes:
                break
            await _try_scrape_candidate(fb_url, is_fallback=True)

    if not crawled_pages:
        logger.warning(
            "web_research_exhausted",
            extra={
                "event": "web_research_exhausted",
                "attempted_count": total_scrapes,
                "successful_count": 0,
                "reason": "all_scraped_candidates_failed_or_empty",
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
        _ext_start = time.perf_counter()
        candidate = await extractor_func(
            page
        )
        _ext_latency = (time.perf_counter() - _ext_start) * 1000
        fact_count = len(candidate.facts) if candidate and candidate.facts else 0
        logger.info(
            "web_extraction_completed",
            extra={
                "event": "web_extraction_completed",
                "url": page.url,
                "fact_count": fact_count,
                "latency_ms": _ext_latency,
            },
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

    session = config.get("configurable", {}).get("db_session")

    candidate = state.get(
        "extracted_knowledge"
    )

    run_id_str = state.get(
        "research_run_id"
    )

    if (
        session is not None
        and candidate
        and candidate.facts
        and run_id_str
    ):
        try:
            await persist_knowledge(
                session,
                run_id_str,
                candidate,
            )
        except Exception as exc:
            logger.warning(
                "persist_knowledge_failed",
                extra={"event": "persist_knowledge_failed", "error": str(exc)},
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

    candidate = state.get(
        "extracted_knowledge"
    )
    clean_pages = state.get(
        "clean_pages",
        [],
    )
    run_id_str = state.get(
        "research_run_id"
    )

    # Convert web facts and pages into transient in-memory RetrievedChunks
    # External research results are strictly ephemeral and NOT inserted into documents or document_chunks!
    web_chunks = create_transient_web_chunks(
        candidate=candidate,
        clean_pages=clean_pages,
        run_id_str=run_id_str,
    )

    logger.info(
        "web_research_completed",
        extra={
            "event": "web_research_completed",
            "run_id": run_id_str,
            "usable_pages_count": len(clean_pages),
            "chunk_count": len(web_chunks),
        },
    )

    return {
        "web_chunks": web_chunks,
        "retrieved_chunks": web_chunks,
    }


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

    # Always initialize this to avoid reference-before-assignment
    web_chunks: list[RetrievedChunk] = []

    # --------------------------------------------------------
    # PATH A — INTERNAL RAG WAS SUFFICIENT
    # --------------------------------------------------------

    suff_eval = state.get("sufficiency_evaluation")
    suff_status = state.get("sufficiency_status")
    if not suff_status and suff_eval is not None:
        suff_status = (
            suff_eval.status.value
            if hasattr(suff_eval.status, "value")
            else str(suff_eval.status)
        )

    is_internal_sufficient = (
        bool(state.get("is_sufficient"))
        or suff_status in ("sufficient", "partial")
        or (
            suff_eval is not None
            and getattr(suff_eval, "status", None)
            in (
                SufficiencyStatus.SUFFICIENT,
                SufficiencyStatus.PARTIAL,
                "sufficient",
                "partial",
            )
        )
    )

    if is_internal_sufficient:

        raw_chunks = state.get(
            "retrieved_chunks",
            [],
        )
        compiled_context = context_compiler.compile(
            chunks=raw_chunks,
            query=state["query"],
            allowed_departments=state.get("allowed_departments"),
            is_web_context=False,
            strategy_trace=state.get("strategy_trace"),
            chunk_provenance=state.get("chunk_provenance"),
            chunk_hierarchy=state.get("chunk_hierarchy"),
            conversation_history=state.get("conversation_history"),
            workspace_thread_context=state.get("workspace_thread_context"),
            user=state.get("user"),
            workspace_id=state.get("workspace_id"),
        )

        response = (
            await generation_service.generate_from_context(
                query=state["query"],
                chunks=compiled_context.chunks,
                allowed_departments=(
                    state.get("allowed_departments")
                ),
                is_web_context=False,
                conflicts=compiled_context.conflicts,
                chunk_hierarchy=compiled_context.chunk_hierarchy,
                rendered_context=compiled_context.rendered_context,
                conversation_history=compiled_context.compiled_conversation_history if compiled_context.compiled_conversation_history else state.get("conversation_history"),
                workspace_thread_context=compiled_context.compiled_workspace_threads if compiled_context.compiled_workspace_threads else state.get("workspace_thread_context"),
                personal_memories=compiled_context.compiled_memories if compiled_context.compiled_memories else None,
                workspace_id=state.get("workspace_id"),
                sufficiency_evaluation=state.get("sufficiency_evaluation"),
                output_preferences=state.get("output_preferences"),
                user=state.get("user"),
                conversation_id=state.get("conversation_id"),
            )
        )

    # --------------------------------------------------------
    # PATH B — RESEARCH WAS REQUIRED
    # --------------------------------------------------------

    else:

        web_chunks = state.get("web_chunks") or []
        actual_web_chunks = [c for c in web_chunks if str(c.chunk_id).startswith("web_")]
        if not actual_web_chunks:
            actual_web_chunks = [c for c in (state.get("retrieved_chunks") or []) if str(c.chunk_id).startswith("web_")]

        # Preserve internal chunks for hybrid internal + web evidence synthesis
        internal_chunks = [c for c in (state.get("retrieved_chunks") or []) if not str(c.chunk_id).startswith("web_")]

        # Combine with separate provenance intact
        combined_chunks = internal_chunks + actual_web_chunks if actual_web_chunks else internal_chunks

        if actual_web_chunks:

            compiled_context = context_compiler.compile(
                chunks=combined_chunks,
                query=state["query"],
                allowed_departments=state.get("allowed_departments"),
                is_web_context=True,
                strategy_trace=state.get("strategy_trace"),
                chunk_provenance=state.get("chunk_provenance"),
                chunk_hierarchy=state.get("chunk_hierarchy"),
                conversation_history=state.get("conversation_history"),
                workspace_thread_context=state.get("workspace_thread_context"),
                user=state.get("user"),
                workspace_id=state.get("workspace_id"),
            )

            response = (
                await generation_service.generate_from_context(
                    query=state["query"],
                    chunks=compiled_context.chunks,
                    allowed_departments=(
                        state.get("allowed_departments")
                    ),
                    is_web_context=True,
                    conflicts=compiled_context.conflicts,
                    chunk_hierarchy=compiled_context.chunk_hierarchy,
                    rendered_context=compiled_context.rendered_context,
                    conversation_history=compiled_context.compiled_conversation_history if compiled_context.compiled_conversation_history else state.get("conversation_history"),
                    workspace_thread_context=compiled_context.compiled_workspace_threads if compiled_context.compiled_workspace_threads else state.get("workspace_thread_context"),
                    personal_memories=compiled_context.compiled_memories if compiled_context.compiled_memories else None,
                    workspace_id=state.get("workspace_id"),
                    sufficiency_evaluation=state.get("sufficiency_evaluation"),
                    output_preferences=state.get("output_preferences"),
                    user=state.get("user"),
                    conversation_id=state.get("conversation_id"),
                )
            )

        # ----------------------------------------------------
        # Research returned no evidence / failed
        # ----------------------------------------------------

        else:
            web_err = state.get("web_research_error")
            if web_err:
                response = GenerateResponse(
                    answer=(
                        "I could not verify the requested information from available documents, "
                        "and external web research could not be completed at this time."
                    ),
                    citations=[],
                    retrieved_chunks=[],
                )
            else:
                response = GenerateResponse(
                    answer=(
                        "I could not verify the requested information "
                        "from available documents or web sources."
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

        session = config.get(
            "configurable", {}
        ).get("db_session")

        if session is not None:
            try:
                run = await session.get(
                    ResearchRun,
                    uuid.UUID(run_id_str),
                )

                if run:
                    if is_internal_sufficient:
                        run.status = "COMPLETED"
                    elif actual_web_chunks if not is_internal_sufficient else False:
                        run.status = "COMPLETED"
                    else:
                        run.status = "FAILED"

                    await session.flush()
            except Exception as exc:
                logger.warning(
                    "failed_to_update_research_run_status",
                    extra={"event": "failed_to_update_research_run_status", "error": str(exc)},
                )

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

builder.add_node(
    "query_workspace_mcp",
    query_workspace_mcp_node,
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

        "query_workspace_mcp":
            "query_workspace_mcp",

        "init_research_run":
            "init_research_run",

        "no_context":
            "no_context",
    },
)

builder.add_conditional_edges(
    "query_workspace_mcp",
    decide_after_mcp,
    {
        "generate_final_answer":
            "generate_final_answer",

        "init_research_run":
            "init_research_run",

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