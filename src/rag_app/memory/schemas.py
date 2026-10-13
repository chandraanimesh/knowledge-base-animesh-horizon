import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryType(str, Enum):
    # Core categories (PRP & Session 6)
    EPISODIC = "EPISODIC"      # What happened: events, conversations, interactions
    SEMANTIC = "SEMANTIC"      # What is known: facts, preferences, domain entities
    PROCEDURAL = "PROCEDURAL"  # How to do things: instructions, workflows, conventions
    
    # Existing legacy tiers (preserved for backward compatibility)
    STATIC = "STATIC"          # Permanent user profile, preferences, persona facts
    DYNAMIC = "DYNAMIC"        # Evolving user context, tasks, observations


class StabilityTier(str, Enum):
    ETERNAL = "ETERNAL"          # Core user identity / stable preferences; no auto-decay
    STABLE = "STABLE"            # Durable project conventions and architectural decisions
    EVOLVING = "EVOLVING"        # Current task, active commitment, mutable status
    EXPERIMENTAL = "EXPERIMENTAL"# Short-lived hypothesis, temporary debugging note


class MemoryStatus(str, Enum):
    ACTIVE = "ACTIVE"          # Currently active memory
    SUPERSEDED = "SUPERSEDED"  # Replaced by newer memory without deleting history
    ARCHIVED = "ARCHIVED"      # Manually/policy archived memory
    EXPIRED = "EXPIRED"        # Passed validity window


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    RETRY_PENDING = "RETRY_PENDING"
    DEAD_LETTER = "DEAD_LETTER"
    SKIPPED_NOT_MEMORABLE = "SKIPPED_NOT_MEMORABLE"


class MemoryCandidate(BaseModel):
    """
    Contract for a memory candidate extracted from conversations or events,
    ready for materialization into persistent PostgreSQL storage.
    """
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    workspace_id: Optional[uuid.UUID] = None
    content: str
    canonical_key: Optional[str] = None
    memory_type: Union[MemoryType, str] = MemoryType.DYNAMIC
    status: Union[MemoryStatus, str] = MemoryStatus.ACTIVE
    stability_tier: Union[StabilityTier, str] = StabilityTier.EVOLVING
    importance_signals: list[str] = Field(default_factory=list)
    subject: Optional[str] = None
    predicate: Optional[str] = None
    value: Optional[Any] = None
    source_quote: Optional[str] = None
    structured_data: Optional[dict] = Field(default_factory=dict)
    confidence: Optional[float] = 1.0
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    retention_expires_at: Optional[datetime] = None
    supersedes_id: Optional[uuid.UUID] = None
    source_type: Optional[str] = "CONVERSATION"
    source_job_id: Optional[uuid.UUID] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    event_id: Optional[str] = None
    extraction_method: Optional[str] = "conversation_extraction"
    embedding: Optional[list[float]] = None


class MemoryRecord(BaseModel):
    """
    Read model for a persisted memory record retrieved from PostgreSQL.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    workspace_id: Optional[uuid.UUID] = None
    canonical_key: str
    memory_type: str
    status: str
    content: str
    stability_tier: Optional[str] = None
    importance_signals: list[str] = Field(default_factory=list)
    subject: Optional[str] = None
    predicate: Optional[str] = None
    source_quote: Optional[str] = None
    structured_data: Optional[dict] = Field(default_factory=dict)
    confidence: Optional[float] = 1.0
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    retention_expires_at: Optional[datetime] = None
    superseded_by_id: Optional[uuid.UUID] = None
    source_type: Optional[str] = None
    source_job_id: Optional[uuid.UUID] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    event_id: Optional[str] = None
    extraction_method: Optional[str] = None
    embedding: Optional[list[float]] = None
    score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def extract_metadata(cls, data: Any) -> Any:
        if data is None:
            return data
        if hasattr(data, "__dict__") or not isinstance(data, dict):
            sd = getattr(data, "structured_data", None) or {}
            res = {
                "id": getattr(data, "id", None),
                "user_id": getattr(data, "user_id", None),
                "workspace_id": getattr(data, "workspace_id", None),
                "canonical_key": getattr(data, "canonical_key", ""),
                "memory_type": getattr(data, "memory_type", "DYNAMIC"),
                "status": getattr(data, "status", "ACTIVE"),
                "content": getattr(data, "content", ""),
                "stability_tier": sd.get("stability_tier"),
                "importance_signals": sd.get("importance_signals") or [],
                "subject": sd.get("subject"),
                "predicate": sd.get("predicate"),
                "source_quote": sd.get("source_quote"),
                "structured_data": sd,
                "confidence": getattr(data, "confidence", 1.0),
                "valid_from": getattr(data, "valid_from", None),
                "valid_until": getattr(data, "valid_until", None),
                "retention_expires_at": getattr(data, "retention_expires_at", None),
                "superseded_by_id": getattr(data, "superseded_by_id", None),
                "source_type": getattr(data, "source_type", None),
                "source_job_id": getattr(data, "source_job_id", None),
                "conversation_id": getattr(data, "conversation_id", None),
                "message_id": getattr(data, "message_id", None),
                "event_id": getattr(data, "event_id", None),
                "extraction_method": getattr(data, "extraction_method", None),
                "embedding": getattr(data, "embedding", None),
                "score": getattr(data, "score", None),
                "created_at": getattr(data, "created_at", None),
                "updated_at": getattr(data, "updated_at", None),
            }
            return res
        return data


class MemoryProcessingJobRecord(BaseModel):
    """
    Read model for a durable memory processing job outbox record.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_message_id: Optional[uuid.UUID] = None
    conversation_id: Optional[uuid.UUID] = None
    owner_user_id: uuid.UUID
    workspace_id: Optional[uuid.UUID] = None
    source_text_snapshot: str
    trigger_kind: str
    status: str
    attempt_count: int = 0
    max_attempts: int = 3
    next_attempt_at: Optional[datetime] = None
    last_error_code: Optional[str] = None
    error_message: Optional[str] = None
    locked_at: Optional[datetime] = None
    locked_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None


class MemoryRetrievalResponse(BaseModel):
    """
    Response model for memory retrieval queries, partitioning static profile facts,
    dynamic contextual memories, and semantically relevant memories.
    """
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    workspace_id: Optional[uuid.UUID] = None
    static_memories: list[MemoryRecord] = Field(default_factory=list)
    dynamic_memories: list[MemoryRecord] = Field(default_factory=list)
    relevant_memories: list[MemoryRecord] = Field(default_factory=list)
    historical_memories: list[MemoryRecord] = Field(default_factory=list)
    total_count: int = 0
