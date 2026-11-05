"""
FastAPI Routes for Twenty CRM Knowledge Base Synchronization and Controlled Write-Back.

Endpoints:
- POST /crm/sync/batch (Admin only): Trigger bounded batch sync across Twenty CRM records.
- POST /crm/sync/retry (Admin only): Retry eligible failed sync records with backoff.
- POST /crm/sync/reconcile (Admin only): Trigger drift reconciliation sweep.
- GET  /crm/sync/jobs/{job_id} (Authenticated): Get durable job metrics and progress.
- POST /crm/webhooks/twenty (Public / Signed): Inbound webhook receiver for Twenty CRM events.
- POST /crm/writeback/meeting-note (Authenticated + Approved): Controlled meeting summary note write-back.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user, require_admin
from rag_app.core.config import settings
from rag_app.crm.sync_service import CRMSyncService
from rag_app.db.models import CRMSyncJob, User
from rag_app.db.session import get_db_session
from rag_app.schemas.crm import (
    CRMApprovalResponse,
    CRMBatchSyncRequest,
    CRMBatchSyncResponse,
    CRMCreateApprovalRequest,
    CRMReconcileRequest,
    CRMRetryRequest,
    CRMWriteBackRequest,
    CRMWriteBackResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crm", tags=["crm-sync"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
AdminUser = Annotated[User, Depends(require_admin)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post(
    "/sync/batch",
    response_model=CRMBatchSyncResponse,
    summary="Trigger bounded batch synchronization from Twenty CRM to Knowledge Base",
)
async def trigger_batch_sync(
    payload: CRMBatchSyncRequest,
    session: DbSession,
    admin: AdminUser,
) -> CRMBatchSyncResponse:
    service = CRMSyncService(session)
    try:
        summary = await service.sync_batch(
            department_id=payload.department_id,
            limit=payload.limit,
            offset=payload.offset,
            created_by=admin.id,
            force=payload.force,
        )
        return CRMBatchSyncResponse(
            job_id=summary.job_id,
            status=summary.status,
            total_records=summary.total_records,
            processed_records=summary.processed_records,
            succeeded_records=summary.succeeded_records,
            skipped_records=summary.skipped_records,
            failed_records=summary.failed_records,
            checkpoint_cursor=summary.checkpoint_cursor,
            records=[
                {
                    "record_id": r.record_id,
                    "company_name": r.company_name,
                    "status": r.status,
                    "document_id": str(r.document_id) if r.document_id else None,
                    "error_message": r.error_message,
                    "skipped": r.skipped,
                }
                for r in summary.records
            ],
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as ex:
        logger.error("Batch sync failed: %s", ex, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch sync error: {ex}")


@router.post(
    "/sync/retry",
    summary="Retry eligible transient failed CRM sync records",
)
async def retry_failed_syncs(
    payload: CRMRetryRequest,
    session: DbSession,
    _: AdminUser,
) -> dict[str, Any]:
    service = CRMSyncService(session)
    try:
        retried = await service.retry_eligible_failed_records(
            department_id=payload.department_id,
            limit=payload.limit,
        )
        return {
            "retried_count": len(retried),
            "results": [
                {
                    "record_id": r.record_id,
                    "status": r.status,
                    "document_id": str(r.document_id) if r.document_id else None,
                    "error_message": r.error_message,
                }
                for r in retried
            ],
        }
    except Exception as ex:
        logger.error("Retry failed: %s", ex, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))


@router.post(
    "/sync/reconcile",
    summary="Run scheduled reconciliation between Twenty CRM and Knowledge Base ledger",
)
async def reconcile_crm_records(
    payload: CRMReconcileRequest,
    session: DbSession,
    _: AdminUser,
) -> dict[str, Any]:
    service = CRMSyncService(session)
    try:
        return await service.reconcile_scheduled(
            department_id=payload.department_id,
            max_records=payload.max_records,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as ex:
        logger.error("Reconciliation failed: %s", ex, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))


@router.get(
    "/sync/jobs/{job_id}",
    summary="Retrieve details and metrics of a CRM sync job",
)
async def get_sync_job_status(
    job_id: uuid.UUID,
    session: DbSession,
    _: CurrentUser,
) -> dict[str, Any]:
    job = await session.get(CRMSyncJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Sync job {job_id} not found.")

    return {
        "id": str(job.id),
        "job_type": job.job_type,
        "status": job.status,
        "department_id": str(job.department_id) if job.department_id else None,
        "total_records": job.total_records,
        "processed_records": job.processed_records,
        "succeeded_records": job.succeeded_records,
        "skipped_records": job.skipped_records,
        "failed_records": job.failed_records,
        "checkpoint_cursor": job.checkpoint_cursor,
        "error_message": job.error_message,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


@router.post(
    "/webhooks/twenty",
    summary="Inbound webhook endpoint for Twenty CRM event notifications",
)
async def handle_twenty_webhook(
    request: Request,
    session: DbSession,
    x_twenty_webhook_signature: Optional[str] = Header(None, alias="X-Twenty-Webhook-Signature"),
    x_twenty_signature: Optional[str] = Header(None, alias="X-Twenty-Signature"),
    x_signature: Optional[str] = Header(None, alias="X-Signature"),
    x_twenty_webhook_timestamp: Optional[str] = Header(None, alias="X-Twenty-Webhook-Timestamp"),
    x_twenty_webhook_nonce: Optional[str] = Header(None, alias="X-Twenty-Webhook-Nonce"),
    department_id: Optional[str] = Query(None, description="Target department UUID or code for ingestion"),
) -> dict[str, Any]:
    # 1. Enforce payload size limit (SEC-05)
    max_bytes = settings.twenty_webhook_max_bytes
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Webhook payload exceeds maximum size limit of {max_bytes} bytes.",
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Content-Length header.",
            )

    body_bytes = bytearray()
    async for chunk in request.stream():
        body_bytes.extend(chunk)
        if len(body_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Webhook payload exceeds maximum size limit of {max_bytes} bytes.",
            )
    raw_bytes = bytes(body_bytes)

    # 2. Enforce fail-closed secret configuration (SEC-01)
    secret = settings.twenty_webhook_secret
    if not secret or not secret.strip():
        logger.error("Twenty CRM webhook rejected: twenty_webhook_secret is not configured on server.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Twenty CRM webhook secret is not configured on the server; rejecting fail-closed.",
        )

    # 3. Enforce presence of signature header
    signature = x_twenty_webhook_signature or x_twenty_signature or x_signature
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing webhook signature header (X-Twenty-Signature or X-Twenty-Webhook-Signature).",
        )

    service = CRMSyncService(session)

    try:
        result = await service.handle_webhook_event(
            payload=raw_bytes,
            signature_header=signature,
            secret=secret,
            department_id=department_id,
            timestamp=x_twenty_webhook_timestamp,
            event_id=x_twenty_webhook_nonce,
        )
        return result
    except ValueError as ve:
        logger.warning("Twenty CRM webhook ValueError: %s", ve, exc_info=True)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(ve))
    except Exception as ex:
        logger.error("Webhook processing error: %s", ex, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Webhook processing error: {ex}")


@router.post(
    "/writeback/approvals",
    response_model=CRMApprovalResponse,
    summary="Create a server-side approval ticket for CRM write-back (Admin only)",
)
async def create_writeback_approval_ticket(
    payload: CRMCreateApprovalRequest,
    session: DbSession,
    admin: AdminUser,
) -> CRMApprovalResponse:
    service = CRMSyncService(session)
    try:
        approval = await service.create_writeback_approval(
            company_id=payload.company_id,
            department_id=payload.department_id,
            approved_by=admin.id,
            valid_hours=payload.valid_hours,
            operation_type=payload.operation_type,
            approved_draft_hash=payload.approved_draft_hash,
            intended_action=payload.intended_action,
            workflow_run_id=payload.workflow_run_id,
        )
        return CRMApprovalResponse(
            id=approval.id,
            operation_type=approval.operation_type,
            company_id=approval.external_record_id,
            department_id=approval.department_id,
            status=approval.status,
            approved_by=approval.approved_by,
            expires_at=approval.expires_at.isoformat(),
            approved_draft_hash=approval.approved_draft_hash,
            intended_action=approval.intended_action,
            workflow_run_id=approval.workflow_run_id,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as ex:
        logger.error("Failed to create write-back approval: %s", ex, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))


@router.post(
    "/writeback/meeting-note",
    response_model=CRMWriteBackResponse,
    summary="Create a meeting summary as a linked CRM Note in Twenty CRM",
)
async def writeback_meeting_note(
    payload: CRMWriteBackRequest,
    session: DbSession,
    user: CurrentUser,
) -> CRMWriteBackResponse:
    service = CRMSyncService(session)
    try:
        res = await service.create_meeting_note_writeback(
            company_id=payload.company_id,
            note_title=payload.note_title,
            summary_markdown=payload.summary_markdown,
            idempotency_key=payload.idempotency_key,
            user_id=user.id,
            user_role=user.role,
            approval_id=payload.approval_id,
            approved=payload.approved,
        )
        return CRMWriteBackResponse(
            status=res["status"],
            idempotency_key=res["idempotency_key"],
            crm_note_id=res.get("crm_note_id"),
            crm_note_target_id=res.get("crm_note_target_id"),
            note_title=res["note_title"],
            is_duplicate_replay=res.get("is_duplicate_replay", False),
            approval_id=res.get("approval_id"),
        )
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as ex:
        logger.error("Meeting note write-back failed: %s", ex, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Write-back error: {ex}")
