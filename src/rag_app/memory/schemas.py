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
    DELETION_REQUESTED = "DELETION_REQUESTED" # Queued for purge
    DELETION_PROCESSING = "DELETION_PROCESSING" # Purge in progress
    PURGED = "PURGED"          # Sensitive content removed, embedding zeroed


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    RETRY_PENDING = "RETRY_PENDING"
    DEAD_LETTER = "DEAD_LETTER"
    SKIPPED_NOT_MEMORABLE = "SKIPPED_NOT_MEMORABLE"
    SKIPPED_POLICY_PAUSED = "SKIPPED_POLICY_PAUSED"


class ScopeType(str, Enum):
    TENANT = "TENANT"
    WORKSPACE = "WORKSPACE"
    USER = "USER"


class PrivacyPolicyPayload(BaseModel):
    scope_type: ScopeType = ScopeType.USER
    scope_id: Optional[uuid.UUID] = None
    memory_collection_enabled: Optional[bool] = None
    memory_extraction_enabled: Optional[bool] = None
    memory_dreaming_enabled: Optional[bool] = None
    memory_retrieval_enabled: Optional[bool] = None
    retention_days: Optional[int] = Field(None, ge=1, le=3650)
    auto_expiration_enabled: Optional[bool] = None
    user_deletion_enabled: Optional[bool] = None
    profile_scrubbing_enabled: Optional[bool] = None
    memory_export_enabled: Optional[bool] = None
    user_can_pause: Optional[bool] = None


class PrivacyPolicyRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[uuid.UUID] = None
    scope_type: ScopeType
    scope_id: uuid.UUID
    memory_collection_enabled: bool = True
    memory_extraction_enabled: bool = True
    memory_dreaming_enabled: bool = True
    memory_retrieval_enabled: bool = True
    retention_days: Optional[int] = None
    auto_expiration_enabled: bool = True
    user_deletion_enabled: bool = True
    profile_scrubbing_enabled: bool = True
    memory_export_enabled: bool = True
    user_can_pause: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProfileScrubPayload(BaseModel):
    section: str = Field(..., description="Profile section to scrub (e.g. work_context, user_preferences, stable_technical_preferences)")
    key: str = Field(..., description="Attribute key to scrub (e.g. role, communication_style, database)")


class MemoryExportItem(BaseModel):
    id: uuid.UUID
    canonical_key: str
    memory_type: str
    status: str
    content: str
    structured_data: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
    source_type: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    retention_expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class MemoryExportResponse(BaseModel):
    user_id: uuid.UUID
    workspace_id: Optional[uuid.UUID] = None
    exported_at: datetime
    total_memories: int
    memories: list[MemoryExportItem]


class MemoryPrivacyStatusResponse(BaseModel):
    user_id: uuid.UUID
    collection_state: str  # "ENABLED", "PAUSED", "DISABLED"
    effective_policy: PrivacyPolicyRecord
    total_active_memories: int
    total_expired_memories: int
    total_purged_memories: int
    living_profile_summary: dict[str, Any]
    last_dreaming_run: Optional[str] = None


class PrivacyAuditEventRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    workspace_id: Optional[uuid.UUID] = None
    department_id: Optional[uuid.UUID] = None
    action: str
    status: str
    metadata_: Optional[dict[str, Any]] = None
    created_at: datetime


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

    @property
    def is_episodic(self) -> bool:
        val = self.memory_type.value if hasattr(self.memory_type, "value") else str(self.memory_type)
        return val.upper() == "EPISODIC"

    @property
    def is_semantic(self) -> bool:
        val = self.memory_type.value if hasattr(self.memory_type, "value") else str(self.memory_type)
        return val.upper() in ("SEMANTIC", "STATIC")

    @property
    def is_procedural(self) -> bool:
        val = self.memory_type.value if hasattr(self.memory_type, "value") else str(self.memory_type)
        return val.upper() == "PROCEDURAL"


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

    @property
    def is_episodic(self) -> bool:
        return (self.memory_type or "").upper() == "EPISODIC"

    @property
    def is_semantic(self) -> bool:
        return (self.memory_type or "").upper() in ("SEMANTIC", "STATIC")

    @property
    def is_procedural(self) -> bool:
        return (self.memory_type or "").upper() == "PROCEDURAL"


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


class RelationshipType(str, Enum):
    DUPLICATE = "DUPLICATE"
    COMPLEMENTS = "COMPLEMENTS"
    SUPERSEDES = "SUPERSEDES"
    CONTRADICTS = "CONTRADICTS"
    RELATED = "RELATED"
    UNRESOLVED = "UNRESOLVED"


class MemoryRelationshipPayload(BaseModel):
    source_memory_id: uuid.UUID
    target_memory_id: uuid.UUID
    relationship_type: RelationshipType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: Optional[str] = None
    source_evidence: Optional[dict[str, Any]] = None


class MemoryRelationshipRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_memory_id: uuid.UUID
    target_memory_id: uuid.UUID
    relationship_type: str
    confidence: float = 1.0
    reason: Optional[str] = None
    source_evidence: Optional[dict[str, Any]] = None
    created_at: datetime


class MemoryDreamingRunRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: str
    user_id: uuid.UUID
    workspace_id: Optional[uuid.UUID] = None
    trigger_type: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    memories_inspected: int = 0
    memories_skipped: int = 0
    duplicates_detected: int = 0
    conflicts_detected: int = 0
    new_memories_created: int = 0
    memories_superseded: int = 0
    profile_updated: bool = False
    error_message: Optional[str] = None
    token_usage: Optional[dict[str, Any]] = None
    checkpoint_data: Optional[dict[str, Any]] = None
    duration_seconds: Optional[float] = None


class MemoryOverviewResponse(BaseModel):
    user_id: uuid.UUID
    total_memories: int
    active_memories: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    total_relationships: int
    total_dreaming_runs: int
    last_dreaming_run: Optional[str] = None
    living_profile_summary: dict[str, Any]


class MemoryEvalMetricRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    eval_run_id: str
    scenario_name: str
    metric_category: str
    metrics_json: dict[str, Any]
    passed: bool
    details: Optional[str] = None
    created_at: datetime


class DreamingRunRequest(BaseModel):
    workspace_id: Optional[uuid.UUID] = None
    threshold: Optional[int] = Field(default=None, ge=1, le=100)
    max_batch_size: Optional[int] = Field(default=50, ge=1, le=500)


class DreamingRunResponse(BaseModel):
    run_id: str
    status: str
    user_id: uuid.UUID
    clusters_consolidated: int = 0
    memories_consolidated: int = 0
    new_memory_ids: list[str] = Field(default_factory=list)
    duration_seconds: Optional[float] = None
    message: Optional[str] = None


class EvaluationRunRequest(BaseModel):
    eval_run_id: Optional[str] = None


class EvaluationRunResponse(BaseModel):
    eval_run_id: str
    total_scenarios: int
    passed_scenarios: int
    pass_rate: float
    duration_seconds: float
    scenarios: list[dict[str, Any]]

