import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


class ConnectionDiscoverRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source_item_id: uuid.UUID = Field(..., description="ID of source KnowledgeItem or MemoryItem")
    source_item_type: str = Field("NOTE", description="Source type: NOTE, MEMORY, or KNOWLEDGE")
    max_candidates: int = Field(10, ge=1, le=50, description="Max candidate suggestions to discover")


class ConnectionReviewRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    decision: str = Field(..., description="Review outcome: CONFIRMED, REJECTED, or CONTRADICTED")
    rationale: str = Field(..., min_length=3, max_length=2000, description="Review justification / notes")
    source_version_hash: str = Field(..., description="Source version hash that was reviewed")
    target_version_hash: str = Field(..., description="Target version hash that was reviewed")
    policy_version: str = Field("v1.0", description="Policy version under which review was performed")


class ConnectionEvidenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    candidate_id: uuid.UUID
    document_chunk_id: Optional[uuid.UUID] = None
    chat_message_id: Optional[uuid.UUID] = None
    research_evidence_id: Optional[uuid.UUID] = None
    claim_excerpt: str
    content_hash: str
    verification_status: str
    created_at: datetime


class ConnectionCandidateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_item_id: uuid.UUID
    source_item_type: str
    source_title: Optional[str] = None
    target_item_id: uuid.UUID
    target_item_type: str
    target_title: Optional[str] = None
    connection_type: str
    predicate: str
    lifecycle_state: str
    is_directed: bool
    ranking_score: float
    source_version_hash: str
    target_version_hash: str
    department_id: Optional[uuid.UUID] = None
    workspace_id: Optional[uuid.UUID] = None
    author_user_id: Optional[uuid.UUID] = None
    evidence: List[ConnectionEvidenceResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ConnectionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_item_id: uuid.UUID
    source_item_type: str
    target_source_hash: str
    requesting_user_id: uuid.UUID
    status: str
    attempt_count: int
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
