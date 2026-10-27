"""
Pydantic schemas for CRM Knowledge Base synchronization and write-back endpoints.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional
from pydantic import BaseModel, Field


class CRMBatchSyncRequest(BaseModel):
    department_id: str = Field(..., description="Target department UUID or code")
    limit: int = Field(default=20, ge=1, le=100, description="Max records per batch")
    offset: int = Field(default=0, ge=0, description="Pagination offset / checkpoint cursor")
    force: bool = Field(default=False, description="Force re-sync even if fingerprint matches")


class CRMBatchSyncResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    total_records: int
    processed_records: int
    succeeded_records: int
    skipped_records: int
    failed_records: int
    checkpoint_cursor: Optional[str] = None
    records: list[dict[str, Any]] = Field(default_factory=list)


class CRMRetryRequest(BaseModel):
    department_id: Optional[str] = Field(default=None, description="Optional target department")
    limit: int = Field(default=10, ge=1, le=50, description="Max retry attempts")


class CRMReconcileRequest(BaseModel):
    department_id: str = Field(..., description="Target department UUID or code")
    max_records: int = Field(default=50, ge=1, le=100, description="Max records to reconcile")


class CRMWriteBackRequest(BaseModel):
    company_id: str = Field(..., description="Target Twenty CRM Company UUID")
    note_title: str = Field(..., min_length=1, max_length=255, description="Note title")
    summary_markdown: str = Field(..., min_length=1, description="Meeting summary content in Markdown")
    idempotency_key: str = Field(..., min_length=8, max_length=255, description="Client idempotency key")
    approved: bool = Field(default=True, description="Client acknowledgement flag")
    approval_id: Optional[uuid.UUID] = Field(default=None, description="Server-side trusted approval record ID")


class CRMWriteBackResponse(BaseModel):
    status: str
    idempotency_key: str
    crm_note_id: Optional[str] = None
    crm_note_target_id: Optional[str] = None
    note_title: str
    is_duplicate_replay: bool
    approval_id: Optional[str] = None


class CRMCreateApprovalRequest(BaseModel):
    company_id: str = Field(..., description="Target Twenty CRM Company UUID")
    department_id: str = Field(..., description="Target department UUID or code")
    operation_type: str = Field(default="NOTE_CREATE", description="Operation type")
    valid_hours: int = Field(default=24, ge=1, le=168, description="Validity period in hours")
    approved_draft_hash: Optional[str] = Field(default=None, description="SHA-256 hash of approved draft content")
    intended_action: str = Field(default="CRM_CREATE_NOTE", description="Intended writeback action")
    workflow_run_id: Optional[uuid.UUID] = Field(default=None, description="Associated WorkflowRun ID")


class CRMApprovalResponse(BaseModel):
    id: uuid.UUID
    operation_type: str
    company_id: str
    department_id: uuid.UUID
    status: str
    approved_by: uuid.UUID
    expires_at: str
    approved_draft_hash: Optional[str] = None
    intended_action: Optional[str] = "CRM_CREATE_NOTE"
    workflow_run_id: Optional[uuid.UUID] = None


class CRMConnectionSyncRequest(BaseModel):
    department_id: Optional[str] = Field(default=None, description="Optional target department UUID or code")
    limit: int = Field(default=20, ge=1, le=100, description="Max records per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset / checkpoint cursor")
    force: bool = Field(default=False, description="Force re-sync even if fingerprint matches")
    max_pages: int = Field(default=1, ge=1, le=10, description="Max pages to sync in this run")


class CRMConnectionReconcileRequest(BaseModel):
    department_id: Optional[str] = Field(default=None, description="Optional target department UUID or code")
    max_records: int = Field(default=50, ge=1, le=100, description="Max records to audit")


