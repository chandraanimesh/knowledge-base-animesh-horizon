import logging
import uuid
from typing import Optional, List, Union
from datetime import datetime, timezone, timedelta

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from rag_app.generation.llm_factory import get_llm
from rag_app.memory.schemas import (
    MemoryCandidate,
    MemoryStatus,
    MemoryType,
)
from rag_app.memory.materializer import generate_memory_canonical_key

logger = logging.getLogger(__name__)

MEMORY_EXTRACTION_SYSTEM_PROMPT = """You are a strict, deterministic personal memory extraction system.
Your job is to extract explicit user facts, durable preferences, persona details, or active task contexts from the user conversation.

# Classification Rules:
1. STATIC memory:
   - Durable, long-term personal facts, user preferences, identity traits, job roles, communication style, contact preferences, or permanent constraints.
   - Examples:
     * "I prefer dark mode" -> content="User prefers dark mode interface", memory_type="STATIC", topic_key="ui_theme"
     * "I am a Senior Financial Auditor in Chicago" -> content="User is a Senior Financial Auditor based in Chicago", memory_type="STATIC", topic_key="user_role"
     * "Always format monetary amounts in EUR" -> content="User prefers monetary amounts formatted in EUR", memory_type="STATIC", topic_key="currency_preference"

2. DYNAMIC memory:
   - Transient, evolving context, current project/task, immediate goals, temporary constraints, or active document review.
   - Examples:
     * "I am currently reviewing the Q3 tax compliance documentation" -> content="User is currently reviewing Q3 tax compliance documentation", memory_type="DYNAMIC", topic_key="current_task"
     * "I am traveling to Berlin next week" -> content="User is traveling to Berlin next week", memory_type="DYNAMIC", topic_key="travel_plan"
     * "Working on bug ticket 402 today" -> content="User is working on bug ticket 402 today", memory_type="DYNAMIC", topic_key="current_task"

3. Irrelevant / No-op Content (Produce NO memories):
   - Greetings ("hi", "hello", "good morning", "how are you"), casual chit-chat, jokes, weather commentary.
   - General knowledge queries ("What is EBITDA?", "Explain neural networks", "Summarize this PDF").
   - Document-specific inquiries where the user reveals nothing about their own facts, preferences, or tasks.
   - If no explicit personal facts, preferences, or tasks are stated by the user, return an EMPTY memories list.

4. Explicit Update / Supersession:
   - If the user explicitly states they changed, moved, or updated a prior preference or fact (e.g. "I moved to New York", "I now prefer light mode", "I changed my role to VP", "I finished that and now I'm working on project Y"), set is_explicit_update=True.
"""


class ExtractedMemoryFact(BaseModel):
    """A single factual memory extracted from conversation."""
    content: str = Field(..., description="A clear, self-contained statement of the extracted fact or preference.")
    memory_type: str = Field(..., description="Classification: 'STATIC' for durable long-term facts/preferences, 'DYNAMIC' for current task, 'SEMANTIC' for project knowledge, 'EPISODIC' for events.")
    topic_key: str = Field(..., description="A concise, normalized topic identifier (e.g., 'ui_theme', 'user_role', 'location', 'current_task', 'currency_preference').")
    stability_tier: Optional[str] = Field(default="EVOLVING", description="Stability tier: 'ETERNAL', 'STABLE', 'EVOLVING', 'EXPERIMENTAL'.")
    source_quote: Optional[str] = Field(default=None, description="Verbatim supporting span from conversation text.")
    structured_data: dict = Field(default_factory=dict, description="Key-value structured attributes if explicitly present.")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0.")
    valid_duration_days: Optional[int] = Field(default=None, description="Optional validity duration in days for dynamic memory.")
    is_explicit_update: bool = Field(default=False, description="True if the user is explicitly updating, correcting, or replacing an existing fact or preference.")


class ConversationMemoryExtractionResult(BaseModel):
    """Collection of extracted memory items from conversation."""
    memories: list[ExtractedMemoryFact] = Field(default_factory=list, description="List of extracted memories. Empty if conversation contains no personal user facts.")


class MemoryExtractor:
    """
    Extracts structured MemoryCandidates from conversational messages.
    Applies deterministic candidate rules first for high accuracy and efficiency,
    falling back to the configured LLM only for uncertain or compound statements.
    Validates grounded quotes to eliminate hallucinations.
    """

    def __init__(self, provider=None):
        self.provider = provider or get_llm()
        self.structured_llm = self.provider.llm.with_structured_output(ConversationMemoryExtractionResult)

    async def extract_from_conversation(
        self,
        text: str,
        user_id: uuid.UUID,
        workspace_id: uuid.UUID,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        event_id: Optional[str] = None,
        source_type: str = "CONVERSATION",
        existing_memories: Optional[list] = None,
    ) -> list[MemoryCandidate]:
        """
        Extracts MemoryCandidates from conversation text using rule-first candidate detection
        with LLM fallback for ambiguous cases.
        """
        if not user_id:
            raise ValueError("user_id is required for memory extraction")
        if not workspace_id:
            raise ValueError("workspace_id is required for memory extraction")

        text_clean = (text or "").strip()
        if not text_clean or len(text_clean) < 3:
            return []

        # 1. Deterministic Rule-First Detection
        from rag_app.memory.command_parser import detect_memory_candidate_rules, StabilityTier
        is_candidate, is_transient, hints = detect_memory_candidate_rules(text_clean)
        now = datetime.now(timezone.utc)

        if is_transient:
            logger.info("memory_extraction_skipped_transient", extra={"text": text_clean})
            return []

        if is_candidate:
            mem_type = hints.get("memory_type", MemoryType.DYNAMIC)
            tier = hints.get("stability_tier", StabilityTier.EVOLVING)
            topic_key = hints.get("topic_key") or text_clean
            c_key = generate_memory_canonical_key(
                mem_type.value if hasattr(mem_type, "value") else str(mem_type),
                topic_key,
            )

            supersedes_id = None
            if existing_memories:
                for ex in existing_memories:
                    ex_key = getattr(ex, "canonical_key", None)
                    ex_status = getattr(ex, "status", None)
                    if ex_key == c_key and (ex_status == MemoryStatus.ACTIVE or ex_status == "ACTIVE"):
                        supersedes_id = getattr(ex, "id", None)
                        break

            return [
                MemoryCandidate(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    content=text_clean,
                    canonical_key=c_key,
                    memory_type=mem_type,
                    status=MemoryStatus.ACTIVE,
                    stability_tier=tier,
                    importance_signals=hints.get("importance_signals", []),
                    source_quote=text_clean,
                    confidence=1.0,
                    valid_from=now,
                    supersedes_id=supersedes_id,
                    source_type=source_type,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    event_id=event_id,
                    extraction_method="rule_deterministic",
                )
            ]

        # 2. LLM Fallback for Uncertain / Compound Cases
        messages = [
            SystemMessage(content=MEMORY_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"CONVERSATION TEXT:\n{text_clean}"),
        ]

        try:
            result: ConversationMemoryExtractionResult = await self.structured_llm.ainvoke(messages)
            if not result or not result.memories:
                return []

            candidates: list[MemoryCandidate] = []
            text_lower = text_clean.lower()

            for fact in result.memories:
                content = (fact.content or "").strip()
                if not content:
                    continue

                # Grounding Validation: quote must be supported by original text
                source_quote = (fact.source_quote or "").strip()
                if source_quote and source_quote.lower() not in text_lower:
                    logger.warning(
                        "memory_candidate_rejected_unsupported_quote",
                        extra={"quote": source_quote, "text": text_clean},
                    )
                    # If quote is not found in text, reject candidate to prevent hallucination
                    continue

                mem_type_str = (fact.memory_type or "DYNAMIC").strip().upper()
                if mem_type_str in ("STATIC", "SEMANTIC"):
                    mem_type = MemoryType.STATIC
                elif mem_type_str == "PROCEDURAL":
                    mem_type = MemoryType.PROCEDURAL
                elif mem_type_str == "EPISODIC":
                    mem_type = MemoryType.EPISODIC
                else:
                    mem_type = MemoryType.DYNAMIC

                topic_key = (fact.topic_key or "general").strip().lower()
                canonical_key = generate_memory_canonical_key(
                    mem_type.value if hasattr(mem_type, "value") else str(mem_type),
                    topic_key,
                )

                valid_until = None
                if fact.valid_duration_days and fact.valid_duration_days > 0:
                    valid_until = now + timedelta(days=fact.valid_duration_days)

                tier_str = (fact.stability_tier or "EVOLVING").strip().upper()
                tier = (
                    StabilityTier[tier_str]
                    if tier_str in StabilityTier.__members__
                    else StabilityTier.EVOLVING
                )

                supersedes_id = None
                if existing_memories:
                    for ex in existing_memories:
                        ex_status = getattr(ex, "status", None)
                        ex_key = getattr(ex, "canonical_key", None)
                        if (
                            (ex_status == MemoryStatus.ACTIVE or ex_status == "ACTIVE")
                            and ex_key == canonical_key
                            and getattr(ex, "id", None) is not None
                        ):
                            if fact.is_explicit_update or getattr(ex, "content", "") != content:
                                supersedes_id = ex.id
                                break

                candidate = MemoryCandidate(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    content=content,
                    canonical_key=canonical_key,
                    memory_type=mem_type,
                    status=MemoryStatus.ACTIVE,
                    stability_tier=tier,
                    importance_signals=["llm_extracted"],
                    source_quote=source_quote or text_clean,
                    structured_data=fact.structured_data or {},
                    confidence=fact.confidence,
                    valid_from=now,
                    valid_until=valid_until,
                    supersedes_id=supersedes_id,
                    source_type=source_type,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    event_id=event_id,
                    extraction_method="memory_llm_fallback",
                )
                candidates.append(candidate)

            return candidates

        except Exception as exc:
            logger.error(
                "memory_extraction_failed",
                extra={
                    "event": "memory_extraction_failed",
                    "user_id": str(user_id),
                    "workspace_id": str(workspace_id),
                    "error": str(exc),
                },
            )
            return []
