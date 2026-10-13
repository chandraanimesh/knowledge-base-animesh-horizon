import uuid
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class CandidateCreateRequest(BaseModel):
    conversation_id: uuid.UUID = Field(..., description="Source conversation ID")
    chat_message_id: uuid.UUID = Field(..., description="Source chat message ID containing the fact")
    statement: str = Field(..., min_length=3, description="Extracted business fact, policy, or decision")
    topic: str = Field(default="general", description="Topic category (e.g. pricing, policy, client_decision)")
    kind: str = Field(default="FACT", description="Knowledge kind ('FACT', 'ENTITY', 'RELATIONSHIP')")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Extraction confidence score")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Structured attributes")


class CandidateRejectRequest(BaseModel):
    reason: Optional[str] = Field(default=None, description="Optional reason for rejecting the candidate")


class KnowledgeEvidenceSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_type: str
    conversation_id: Optional[uuid.UUID] = None
    chat_message_id: Optional[uuid.UUID] = None
    document_chunk_id: Optional[uuid.UUID] = None
    research_evidence_id: Optional[uuid.UUID] = None
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None
    created_at: datetime


class KnowledgeCandidateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    canonical_key: str
    statement: str
    kind: str
    status: str
    visibility: str
    approval_status: str
    source_type: str
    workspace_id: Optional[uuid.UUID] = None
    workspace_name: Optional[str] = None
    department_id: Optional[uuid.UUID] = None
    department_code: Optional[str] = None
    author_id: Optional[uuid.UUID] = None
    author_email: Optional[str] = None
    approved_by_id: Optional[uuid.UUID] = None
    approved_by_email: Optional[str] = None
    approved_at: Optional[datetime] = None
    confidence: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    evidence_count: int = 0
    evidence: list[KnowledgeEvidenceSummary] = Field(default_factory=list)


class CompanyNoteWebsiteSourceInput(BaseModel):
    url: str = Field(..., min_length=4, description="Website URL (HTTP/HTTPS)")
    title: Optional[str] = Field(default=None, description="Optional website title or label")
    user_excerpt: Optional[str] = Field(default=None, description="Optional user-provided supporting passage")


class CompanyNoteEvidenceItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_type: str = "DOCUMENT"  # "DOCUMENT" or "WEBSITE"
    document_chunk_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    file_name: Optional[str] = None
    original_file_name: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_ordinal: Optional[str] = None
    excerpt: str = ""
    extraction_method: Optional[str] = None
    confidence: Optional[float] = None
    created_at: datetime

    # Website source fields
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    user_excerpt: Optional[str] = None
    verification_status: Optional[str] = "NOT_VERIFIED"
    verified_at: Optional[datetime] = None


class CompanyNoteCreateRequest(BaseModel):
    company_name: str = Field(..., min_length=1, description="Company or potential client name")
    statement: str = Field(..., min_length=3, description="Observation or note statement")
    observation_type: str = Field(
        default="SUPPORTED",
        description="Confidence marker: 'SUPPORTED' (directly backed), 'INFERRED' (deduced), or 'UNVERIFIED' (assertion)",
    )
    workspace_id: Optional[uuid.UUID] = Field(default=None, description="Optional workspace scope")
    department_id: Optional[uuid.UUID] = Field(default=None, description="Optional department scope")
    visibility: str = Field(
        default="PRIVATE",
        description="Initial visibility: 'PRIVATE' (author-only draft) or 'WORKSPACE'",
    )
    document_chunk_ids: list[uuid.UUID] = Field(
        default_factory=list,
        description="Optional source document chunk IDs to attach as evidence",
    )
    website_sources: list[CompanyNoteWebsiteSourceInput] = Field(
        default_factory=list,
        description="Optional website sources to attach as evidence",
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    attributes: dict[str, Any] = Field(default_factory=dict, description="Additional structured attributes")


class CompanyNoteUpdateRequest(BaseModel):
    company_name: Optional[str] = Field(default=None, min_length=1)
    statement: Optional[str] = Field(default=None, min_length=3)
    observation_type: Optional[str] = Field(
        default=None,
        description="'SUPPORTED', 'INFERRED', or 'UNVERIFIED'",
    )
    workspace_id: Optional[uuid.UUID] = None
    department_id: Optional[uuid.UUID] = None
    visibility: Optional[str] = None
    attributes: Optional[dict[str, Any]] = None


class CompanyNoteEvidenceAttachRequest(BaseModel):
    document_chunk_id: Optional[uuid.UUID] = Field(default=None, description="Target document chunk ID for document evidence")
    source_url: Optional[str] = Field(default=None, description="Target website URL for website evidence")
    source_title: Optional[str] = Field(default=None, description="Optional website title or label")
    extraction_method: Optional[str] = Field(default="MANUAL_EVIDENCE", description="Evidence attachment method")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    custom_excerpt: Optional[str] = Field(default=None, description="Optional user excerpt or note")


class CompanyNoteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    canonical_key: str
    company_name: str
    statement: str
    observation_type: str
    kind: str
    status: str
    visibility: str
    approval_status: str
    source_type: str
    workspace_id: Optional[uuid.UUID] = None
    workspace_name: Optional[str] = None
    department_id: Optional[uuid.UUID] = None
    department_code: Optional[str] = None
    author_id: Optional[uuid.UUID] = None
    author_email: Optional[str] = None
    approved_by_id: Optional[uuid.UUID] = None
    approved_by_email: Optional[str] = None
    approved_at: Optional[datetime] = None
    confidence: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    evidence_count: int = 0
    evidence: list[CompanyNoteEvidenceItem] = Field(default_factory=list)


class CompanyNoteSubmitRequest(BaseModel):
    target_audience: str = Field(default="SHARED", description="'WORKSPACE' or 'SHARED'")
    target_workspace_id: Optional[uuid.UUID] = Field(default=None, description="Target workspace if audience is WORKSPACE")
    target_department_id: Optional[uuid.UUID] = Field(default=None, description="Target department if audience is SHARED")


class CompanyNoteReviewRequest(BaseModel):
    action: str = Field(..., description="'APPROVE', 'REJECT', 'REVISE', 'WITHDRAW', or 'ARCHIVE'")
    reason: Optional[str] = Field(default=None, description="Optional reviewer reason")


class CompanyNoteArchiveRequest(BaseModel):
    reason: Optional[str] = Field(default=None, description="Optional withdrawal or archive reason")


class CompanyNoteSupersedeRequest(BaseModel):
    replacement_note_id: uuid.UUID = Field(..., description="ID of the new note superseding this note")
    reason: Optional[str] = Field(default=None, description="Optional supersession reason")


class BatchPublicationRequest(BaseModel):
    note_ids: list[uuid.UUID] = Field(..., min_length=1, description="List of note IDs to publish")
    target_audience: str = Field(default="SHARED", description="'SHARED' or 'WORKSPACE'")
    target_department_id: Optional[uuid.UUID] = Field(default=None, description="Target department for shared publication")


class BatchPublicationItemOutcome(BaseModel):
    note_id: uuid.UUID
    canonical_key: Optional[str] = None
    company_name: Optional[str] = None
    success: bool
    status: str
    error: Optional[str] = None


class BatchPublicationResponse(BaseModel):
    batch_id: uuid.UUID
    total_items: int
    published_count: int
    failed_count: int
    is_idempotent_retry: bool
    outcomes: list[BatchPublicationItemOutcome]
