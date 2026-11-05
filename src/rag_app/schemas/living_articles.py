"""
Milestone 7B: Living Articles Pydantic Schemas.
Request and response models for version history, shadow comparison, staging, publishing, and rollback.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentVersionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    version_number: int
    parent_version_id: uuid.UUID | None = None
    source_content_hash: str
    extraction_config_version: str = "v1"
    chunking_config_version: str = "v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    status: str
    is_active: bool
    change_summary: dict[str, Any] | None = None
    created_at: datetime
    published_at: datetime | None = None
    rolled_back_at: datetime | None = None


class DocumentVersionListResponse(BaseModel):
    document_id: uuid.UUID
    current_version_id: uuid.UUID | None = None
    versions: list[DocumentVersionItem]


class ShadowReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_version_id: uuid.UUID
    comparison_status: str
    can_publish: bool
    reasons: list[str] | None = None
    page_diff_summary: dict[str, Any] | None = None
    chunk_diff_summary: dict[str, Any] | None = None
    embedding_diff_summary: dict[str, Any] | None = None
    knowledge_diff_summary: dict[str, Any] | None = None
    retrieval_diff_summary: dict[str, Any] | None = None
    created_at: datetime


class PublishVersionResponse(BaseModel):
    success: bool
    version_id: uuid.UUID
    document_id: uuid.UUID
    status: str
    published_at: datetime | None = None


class RollbackVersionResponse(BaseModel):
    success: bool
    document_id: uuid.UUID
    rolled_back_version_id: uuid.UUID
    active_version_id: uuid.UUID
    status: str

