import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class MemoryType(str, Enum):
    STATIC = "STATIC"      # Permanent user profile, preferences, persona facts
    DYNAMIC = "DYNAMIC"    # Evolving user context, tasks, observations


class MemoryStatus(str, Enum):
    ACTIVE = "ACTIVE"          # Currently active memory
    SUPERSEDED = "SUPERSEDED"  # Replaced by newer memory without deleting history
    ARCHIVED = "ARCHIVED"      # Manually/policy archived memory
    EXPIRED = "EXPIRED"        # Passed validity window


class MemoryCandidate(BaseModel):
    """
    Contract for a memory candidate extracted from conversations or events,
    ready for materialization into persistent PostgreSQL storage.
    """
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    content: str
    canonical_key: Optional[str] = None
    memory_type: Union[MemoryType, str] = MemoryType.DYNAMIC
    status: Union[MemoryStatus, str] = MemoryStatus.ACTIVE
    structured_data: Optional[dict] = Field(default_factory=dict)
    confidence: Optional[float] = 1.0
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    supersedes_id: Optional[uuid.UUID] = None
    source_type: Optional[str] = "CONVERSATION"
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
    canonical_key: str
    memory_type: str
    status: str
    content: str
    structured_data: Optional[dict] = Field(default_factory=dict)
    confidence: Optional[float] = 1.0
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    superseded_by_id: Optional[uuid.UUID] = None
    source_type: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    event_id: Optional[str] = None
    extraction_method: Optional[str] = None
    embedding: Optional[list[float]] = None
    score: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class MemoryRetrievalResponse(BaseModel):
    """
    Response model for memory retrieval queries, partitioning static profile facts,
    dynamic contextual memories, and semantically relevant memories.
    """
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    static_memories: list[MemoryRecord] = Field(default_factory=list)
    dynamic_memories: list[MemoryRecord] = Field(default_factory=list)
    relevant_memories: list[MemoryRecord] = Field(default_factory=list)
    total_count: int = 0



