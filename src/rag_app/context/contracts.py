import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class SourceAuthority(str, Enum):
    AUTHORITATIVE_RECORD = "AUTHORITATIVE_RECORD"  # Approved shared knowledge, CRM, verified company notes (Rank 1)
    DOCUMENT_EVIDENCE = "DOCUMENT_EVIDENCE"        # Authorized uploaded documents/chunks (Rank 2)
    WEB_RESEARCH = "WEB_RESEARCH"                  # External transient web research evidence (Rank 2.5)
    PERSONAL_MEMORY = "PERSONAL_MEMORY"            # User personal preferences/context (Rank 3)
    CONVERSATION_THREAD = "CONVERSATION_THREAD"    # Workspace discussion threads (Rank 4)
    CONVERSATION_TURN = "CONVERSATION_TURN"        # Conversation turns/history in active thread (Rank 5)


AUTHORITY_WEIGHTS = {
    SourceAuthority.AUTHORITATIVE_RECORD.value: 1.0,
    SourceAuthority.DOCUMENT_EVIDENCE.value: 0.85,
    SourceAuthority.WEB_RESEARCH.value: 0.75,
    SourceAuthority.PERSONAL_MEMORY.value: 0.65,
    SourceAuthority.CONVERSATION_THREAD.value: 0.45,
    SourceAuthority.CONVERSATION_TURN.value: 0.35,
}


class ContextEvidence(BaseModel):
    """
    Common evidence schema for all retrieved sources (PRP Section 5.3).
    Decouples the context assembler from inner retriever implementations.
    """
    model_config = ConfigDict(from_attributes=True)

    evidence_id: str
    source_kind: str  # "rag_chunk" | "shared_knowledge" | "memory_fact" | "memory_episode" | "thread" | "web_research" | "conversation_turn"
    source_id: str
    claim_or_text: str
    entity_id: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    recorded_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    source_authority: str = SourceAuthority.DOCUMENT_EVIDENCE.value
    relevance_score: Optional[float] = 1.0
    importance_signals: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    access_scope: dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0
    is_external: bool = False
    status: Optional[str] = None


class ContextBudget(BaseModel):
    """
    Context token budget configuration and accounting (PRP Section 5.4).
    Enforces a single shared hard context budget across all sources:
    available_context_tokens = model_context_limit - reserved_output_tokens - other_prompt_tokens
    """
    model_config = ConfigDict(from_attributes=True)

    model_context_limit: int = 8192
    input_limit: int = 4000
    reserved_output_tokens: int = 500
    system_tokens: int = 0
    query_tokens: int = 0
    instructions_tokens: int = 0
    history_tokens: int = 0
    memory_tokens: int = 0
    thread_tokens: int = 0
    evidence_tokens: int = 0
    available_context_tokens: int = 0
    total_planned_input_tokens: int = 0
    budget_status: str = "NORMAL"  # "NORMAL" | "CONSTRAINED" | "OVERFLOW"

    @property
    def max_allowed_input(self) -> int:
        return max(0, self.model_context_limit - self.reserved_output_tokens)


class ContextPacket(BaseModel):
    """
    Validated context packet delivered to generation (PRP Section 5.4).
    Makes provenance, citations, and unresolved ambiguity visible to generation.
    """
    model_config = ConfigDict(from_attributes=True)

    query: str
    answer_instructions: list[str] = Field(default_factory=list)
    selected_evidence: list[ContextEvidence] = Field(default_factory=list)
    unresolved_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    omissions: list[str] = Field(default_factory=list)
    budget: ContextBudget
    rendered_context: str = ""
