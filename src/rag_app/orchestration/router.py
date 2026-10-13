import logging
import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

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
