"""
Pydantic schemas for Milestone 7A.1: Chunk Curation & Rollback Lifecycle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class ChunkEditRequest(BaseModel):
    new_text: str = Field(..., min_length=1, description="Updated text content for the chunk")
    base_version: int = Field(..., ge=1, description="Expected current version number for optimistic locking")
    edit_reason: Optional[str] = Field(None, max_length=500, description="Audit reason for the curation change")


class ChunkRollbackRequest(BaseModel):
    target_version: int = Field(..., ge=1, description="Historical version number to restore")
    rollback_reason: Optional[str] = Field(None, max_length=500, description="Audit explanation for the rollback action")


class ChunkVersionResponse(BaseModel):
    id: uuid.UUID
    chunk_id: uuid.UUID
    version_number: int
    text_before: str
    text_after: str
    diff_patch: Optional[str] = None
    embedding_model: str
    embedding_dim: int
    chunk_hash: str
    edited_by: Optional[uuid.UUID] = None
    created_at: datetime
    edit_reason: Optional[str] = None
    is_rollback: bool = False
    reverted_from_version: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)



class ChunkDiffResponse(BaseModel):
    chunk_id: uuid.UUID
    v1: int
    v2: int
    text_v1: str
    text_v2: str
    diff_patch: str
    lines_added: int
    lines_removed: int
    characters_changed: int


class ChunkDetailResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    text: str
    chunk_hash: str
    current_version: int
    created_at: datetime
    updated_at: datetime
    page_number: Optional[int] = None
    node_id: Optional[uuid.UUID] = None
    is_managed_external: bool = False

    model_config = ConfigDict(from_attributes=True)



class ChunkEditMutationResponse(BaseModel):
    chunk: ChunkDetailResponse
    version: ChunkVersionResponse


class QualityCheckFlag(BaseModel):
    flag_id: str
    severity: str = Field(..., description="INFO, WARNING, or ERROR")
    message: str
    suggested_remediation: str = Field(..., description="MANUAL_CURATE, REINGEST_DOCUMENT, or SOURCE_CORRECTION")


class QualityReportItem(BaseModel):
    report_id: str
    reported_by: str
    reported_at: datetime
    issue_type: str
    description: str
    suggested_remediation: str
    status: str = "OPEN"


class ChunkIssueReportRequest(BaseModel):
    issue_type: str = Field(..., min_length=2, max_length=50, description="Category of the reported quality problem")
    description: str = Field(..., min_length=3, max_length=1000, description="User explanation of the defect")
    suggested_remediation: str = Field("MANUAL_CURATE", description="MANUAL_CURATE, REINGEST_DOCUMENT, or SOURCE_CORRECTION")


class ChunkEvidenceResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    file_name: str
    file_path: str
    mime_type: str
    source_available: bool
    source_availability_reason: Optional[str] = None
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    node_id: Optional[uuid.UUID] = None
    section_title: Optional[str] = None
    section_ordinal: Optional[str] = None
    section_abstract: Optional[str] = None
    current_text: str
    original_text: str
    is_curated: bool
    current_version: int
    created_at: datetime
    updated_at: datetime
    chunk_hash: str
    is_managed_external: bool
    quality_flags: list[QualityCheckFlag] = Field(default_factory=list)
    quality_reports: list[QualityReportItem] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

