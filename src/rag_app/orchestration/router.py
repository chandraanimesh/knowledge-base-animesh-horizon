import logging
import re
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from rag_app.core.config import get_settings
from rag_app.generation.llm_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    GREETING = "greeting"
    COMMAND = "command"
    PERSONAL_MEMORY = "memory_recall"
    DOCUMENT_KNOWLEDGE = "general_knowledge_base"
    TASK_MANAGEMENT = "task_management"
    HISTORICAL_DISCUSSION = "interaction_history"
    COMPOUND_MIXED = "mixed"


class RoutingDecision(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    intent: QueryIntent
    sources: list[str] = Field(default_factory=list)  # ["rag", "memory", "threads", "tasks"]
    temporal_scope: Optional[str] = "current"          # "current" | "historical"
    confidence: float = 1.0
    routing_reason: str = ""


ROUTER_SYSTEM_PROMPT = """You are a strict, conservative query routing classifier for an enterprise RAG system.
Classify the user's information need and select ONLY the necessary internal sources.
Sources available:
- 'rag': corporate document library, policies, contracts, filings, handbooks
- 'memory': user's personal preferences, profile, saved memories, private context
- 'threads': prior workspace discussions and past chat threads

Rules:
1. 'general_knowledge_base' (sources=['rag']): questions about corporate policies, products, companies, documents, metrics, guidelines.
2. 'memory_recall' (sources=['memory']): questions about user's own identity, saved preferences, persona, what did I tell you.
3. 'mixed' (sources=['rag', 'memory']): compound questions comparing personal facts/preferences with company documents.
4. 'interaction_history' (sources=['threads']): questions asking what was discussed earlier in threads or conversations.
Output must adhere strictly to the schema.
"""


class HybridQueryRouter:
    """
    Hybrid Query Router (PRP Section 5.1).
    - Deterministic rules for clear command syntax, greetings, and unambiguous patterns.
    - LLM classifier for uncertain or compound questions.
    - Conservative deterministic fallback on LLM failure.
    - Never invokes web search.
    """

    def __init__(self, provider=None):
        self.provider = provider

    def route_deterministic(self, query: str) -> Optional[RoutingDecision]:
        clean = (query or "").strip()
        q_lower = clean.lower()

        # 1. Slash commands
        if clean.startswith("/"):
            cmd = clean.split()[0].lower()
            if cmd in ("/rem", "/remember", "/memory", "/memories", "/forget", "/forget_memory"):
                return RoutingDecision(
                    intent=QueryIntent.COMMAND,
                    sources=["memory"],
                    confidence=1.0,
                    routing_reason="Deterministic slash command for personal memory",
                )
            if cmd in ("/task", "/tasks"):
                return RoutingDecision(
                    intent=QueryIntent.COMMAND,
                    sources=["tasks"],
                    confidence=1.0,
                    routing_reason="Deterministic slash command for structured tasks",
                )

        # 2. Greetings
        if re.match(r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening|night)|how\s+are\s+you)\b", q_lower):
            return RoutingDecision(
                intent=QueryIntent.GREETING,
                sources=[],
                confidence=1.0,
                routing_reason="Deterministic conversational greeting",
            )

        # 3. Explicit Personal Memory / User Preference Queries
        pref_queries = [
            r"\b(?:my\s+(?:preferences?|profile|role|job|title|format|style|persona|background|notes?))\b",
            r"\b(?:what\s+do\s+you\s+(?:remember|know|have)\s+about\s+(?:me|my\s+role|my\s+preferences?))\b",
            r"\b(?:how\s+do\s+i\s+(?:prefer|like)\b)",
            r"\b(?:what\s+(?:did\s+i\s+tell\s+you|is\s+saved\s+in\s+my\s+memory|do\s+you\s+recall))\b",
        ]
        if any(re.search(pat, q_lower) for pat in pref_queries):
            is_mixed = any(doc_kw in q_lower for doc_kw in ["policy", "contract", "company", "document", "handbook"])
            if is_mixed:
                return RoutingDecision(
                    intent=QueryIntent.COMPOUND_MIXED,
                    sources=["rag", "memory"],
                    confidence=0.95,
                    routing_reason="Deterministic compound query matching personal preference and company documentation",
                )
            return RoutingDecision(
                intent=QueryIntent.PERSONAL_MEMORY,
                sources=["memory"],
                confidence=0.95,
                routing_reason="Deterministic personal memory and user preference query",
            )

        # 4. Pure Document Knowledge Keywords
        doc_keywords = ["policy", "contract", "handbook", "guideline", "document", "report", "compliance", "file", "pdf"]
        if any(re.search(r"\b" + kw + r"\b", q_lower) for kw in doc_keywords):
            return RoutingDecision(
                intent=QueryIntent.DOCUMENT_KNOWLEDGE,
                sources=["rag"],
                confidence=0.9,
                routing_reason="Deterministic document evidence query",
            )

        # 5. Prior discussion / thread history
        thread_keywords = ["prior discussion", "earlier thread", "workspace discussion", "what did we discuss"]
        if any(kw in q_lower for kw in thread_keywords):
            temporal = "historical"
            return RoutingDecision(
                intent=QueryIntent.HISTORICAL_DISCUSSION,
                sources=["threads"],
                temporal_scope=temporal,
                confidence=0.9,
                routing_reason="Deterministic workspace thread history query",
            )

        return None

    async def route(self, query: str) -> RoutingDecision:
        """
        Routes the query using rule-first checks and LLM fallback.
        """
        det = self.route_deterministic(query)
        if det is not None:
            return det

        # Fallback to LLM classifier for compound/uncertain questions
        try:
            llm_inst = self.provider or get_llm()
            structured = llm_inst.llm.with_structured_output(RoutingDecision)
            res = await structured.ainvoke([
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f"Query: {query}"),
            ])
            if res and isinstance(res, RoutingDecision):
                return res
        except Exception as exc:
            logger.warning(
                "llm_query_routing_failed_using_fallback",
                extra={"query": query, "error": str(exc)},
            )

        # Conservative Fallback: RAG + Memory (safe default, no web search)
        return RoutingDecision(
            intent=QueryIntent.DOCUMENT_KNOWLEDGE,
            sources=["rag", "memory"],
            confidence=0.6,
            routing_reason="Conservative deterministic fallback on ambiguous query",
        )


# ============================================================================
# Milestone 10A: Adaptive Agent Runtime Router
# ============================================================================

class AdaptiveRouteMode(str, Enum):
    """Execution route modes for Adaptive Agent Runtime (Milestone 10A)."""
    FAST_RAG = "FAST_RAG"               # Mode A: Simple knowledge query; bypasses planner
    CONTEXTUAL = "CONTEXTUAL"           # Mode B: Knowledge + user interaction/preferences
    ADAPTIVE_PLAN = "ADAPTIVE_PLAN"     # Mode C: Complex multi-step task DAG


class AdaptiveRequestClassification(BaseModel):
    """Typed classification result for an incoming adaptive request."""
    model_config = ConfigDict(from_attributes=True)

    intent: str = Field(..., description="High-level intent descriptor")
    route: AdaptiveRouteMode = Field(..., description="Selected execution route")
    required_capabilities: list[str] = Field(default_factory=list, description="List of required capability IDs")
    requires_planning: bool = Field(default=False, description="True if DAG decomposition is needed")
    requires_external_action: bool = Field(default=False, description="True if side-effecting external write is requested")
    estimated_complexity: str = Field(default="simple", description="'simple', 'moderate', or 'complex'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="Explanation of routing decision")
    runtime_budgets: dict[str, Any] = Field(default_factory=dict, description="Resolved runtime execution limits")


ADAPTIVE_ROUTER_SYSTEM_PROMPT = """You are the central Adaptive Request Router for an enterprise AI platform.
Classify incoming user requests into exactly one of three execution modes:
1. 'FAST_RAG': Single-shot internal corporate knowledge queries (policies, handbook, expenses, definitions, guidelines).
2. 'CONTEXTUAL': Questions requiring user memory, personal preferences, past discussion, or persona context combined with internal documents.
3. 'ADAPTIVE_PLAN': Complex multi-step research, architectural comparisons, workflow execution, multi-tool analysis, or side-effecting CRM/MCP tasks.

Output strictly according to the AdaptiveRequestClassification schema.
"""


class AdaptiveRequestRouter:
    """
    Central Adaptive Runtime Entry Router (Milestone 10A).
    Understands incoming requests, extracts intent, and selects optimal execution path:
    - FAST_RAG: Pure retrieval and grounded synthesis; zero planner overhead.
    - CONTEXTUAL: Memory + RAG fused via ContextCompiler; zero task DAG overhead.
    - ADAPTIVE_PLAN: Typed task DAG planning for complex multi-step and side-effect workflows.
    """

    def __init__(self, provider=None):
        self.provider = provider
        self.hybrid_router = HybridQueryRouter(provider=provider)

    def _resolve_budgets(self) -> dict[str, Any]:
        settings = get_settings()
        return {
            "max_planner_calls": getattr(settings, "adaptive_max_planner_calls", 3),
            "max_plan_iterations": getattr(settings, "adaptive_max_plan_iterations", 3),
            "max_tasks": getattr(settings, "adaptive_max_tasks", 15),
            "max_parallel_tasks": getattr(settings, "adaptive_max_parallel_tasks", 5),
            "max_execution_time_seconds": getattr(settings, "adaptive_max_execution_time_seconds", 180),
            "max_tool_calls": getattr(settings, "adaptive_max_tool_calls", 20),
            "max_tokens": getattr(settings, "adaptive_max_tokens", 25000),
            "max_retries": getattr(settings, "adaptive_max_retries", 3),
        }

    def classify_deterministic(self, query: str) -> Optional[AdaptiveRequestClassification]:
        clean = (query or "").strip()
        q_lower = clean.lower()
        budgets = self._resolve_budgets()

        # 1. Obvious multi-step planning / CRM writeback patterns -> ADAPTIVE_PLAN
        adaptive_patterns = [
            r"\b(?:compare|comparison|versus|vs\.?)\b.*?\b(?:and\s+recommend|recommendation|gap\s+analysis|roadmap)\b",
            r"\b(?:research|investigate)\b.*?\b(?:and\s+compare|and\s+prepare|and\s+create|and\s+summarize)\b",
            r"\b(?:sync|write|update|create|draft|add|log)\b.*?\b(?:to|in|into)\s+(?:crm|twenty)\b",
            r"\b(?:twenty\s+crm|crm\s+note|crm\s+update|crm\s+task|crm\s+lead)\b",
            r"\b(?:workflow\s+run|adaptive\s+workflow|dag\s+workflow)\b",
            r"\b(?:deep\s+dive|multi-step|multi-stage|systematic\s+review)\b",
        ]
        if any(re.search(pat, q_lower) for pat in adaptive_patterns):
            caps = ["RAG", "RESEARCH", "ANALYSIS", "SYNTHESIS"]
            has_crm = any(k in q_lower for k in ["crm", "twenty", "contact", "lead", "company"])
            if has_crm:
                caps.append("CRM")
            return AdaptiveRequestClassification(
                intent="COMPLEX_RESEARCH_AND_PLANNING",
                route=AdaptiveRouteMode.ADAPTIVE_PLAN,
                required_capabilities=caps,
                requires_planning=True,
                requires_external_action=any(k in q_lower for k in ["update", "create", "write", "sync", "log"]),
                estimated_complexity="complex",
                confidence=0.95,
                reason="Deterministic match for multi-step research, analysis, or CRM workflow",
                runtime_budgets=budgets,
            )

        # 2. Contextual requests (Personal memory + past decisions + preferences) -> CONTEXTUAL
        contextual_patterns = [
            r"\b(?:as\s+we\s+discussed|earlier\s+discussion|discussed\s+earlier|our\s+previous\s+decision)\b",
            r"\b(?:in\s+my\s+preferred\s+style|using\s+my\s+preferences?|my\s+tech\s+stack|my\s+current\s+project)\b",
            r"\b(?:how\s+should\s+you\s+explain|remember\s+how\s+i|what\s+did\s+we\s+agree)\b",
            r"\b(?:my\s+preferences?|my\s+profile|my\s+role|my\s+saved\s+memory)\b",
        ]
        if any(re.search(pat, q_lower) for pat in contextual_patterns):
            return AdaptiveRequestClassification(
                intent="CONTEXTUAL_RETRIEVAL",
                route=AdaptiveRouteMode.CONTEXTUAL,
                required_capabilities=["MEMORY", "RAG"],
                requires_planning=False,
                requires_external_action=False,
                estimated_complexity="moderate",
                confidence=0.95,
                reason="Deterministic match for user preference, personal memory, or interaction context",
                runtime_budgets=budgets,
            )

        # 3. Simple internal knowledge / policy / greetings -> FAST_RAG
        simple_rag_patterns = [
            r"\b(?:what\s+is|who\s+is|where\s+is|when\s+is)\b",
            r"\b(?:leave|expense|travel|holiday|remote\s+work|hr|handbook|security)\s+policy\b",
            r"\b(?:company|office|benefits|insurance|reimbursement|code\s+of\s+conduct|handbook)\b",
            r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))\b",
        ]
        if any(re.search(pat, q_lower) for pat in simple_rag_patterns):
            return AdaptiveRequestClassification(
                intent="KNOWLEDGE_QUERY",
                route=AdaptiveRouteMode.FAST_RAG,
                required_capabilities=["RAG"],
                requires_planning=False,
                requires_external_action=False,
                estimated_complexity="simple",
                confidence=0.95,
                reason="Deterministic match for simple company policy or direct factual question",
                runtime_budgets=budgets,
            )

        return None

    async def classify(self, query: str) -> AdaptiveRequestClassification:
        """
        Classifies incoming request using rule-first fast path, falling back to LLM.
        """
        det = self.classify_deterministic(query)
        if det is not None:
            return det

        # Fallback to LLM structured output
        try:
            llm_inst = self.provider or get_llm()
            structured = llm_inst.llm.with_structured_output(AdaptiveRequestClassification)
            res = await structured.ainvoke([
                SystemMessage(content=ADAPTIVE_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f"User request: {query}"),
            ])
            if res and isinstance(res, AdaptiveRequestClassification):
                if not res.runtime_budgets:
                    res.runtime_budgets = self._resolve_budgets()
                return res
        except Exception as exc:
            logger.warning(
                "llm_adaptive_request_routing_failed_using_safe_fallback",
                extra={"query": query, "error": str(exc)},
            )

        # Safe conservative fallback: FAST_RAG
        return AdaptiveRequestClassification(
            intent="GENERAL_QUERY",
            route=AdaptiveRouteMode.FAST_RAG,
            required_capabilities=["RAG"],
            requires_planning=False,
            requires_external_action=False,
            estimated_complexity="simple",
            confidence=0.6,
            reason="Conservative safe fallback to FAST_RAG on ambiguous classification",
            runtime_budgets=self._resolve_budgets(),
        )

