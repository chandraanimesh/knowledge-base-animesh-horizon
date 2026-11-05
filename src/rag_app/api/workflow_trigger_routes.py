"""
FastAPI Router for Workflow Triggers, Signed Webhooks, Outbox Processing, and Schedules.
Enforces workspace isolation, RBAC, HMAC verification, replay protection, and version pinning.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Optional
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import Department, User, WorkflowScheduleState, Workspace, WorkspaceMember
from rag_app.db.session import get_db_session
from rag_app.orchestration.trigger_service import WorkflowTriggerService
from rag_app.schemas.workflow_trigger import (
    WorkflowScheduleStateResponse,
    WorkflowTriggerRequest,
    WorkflowTriggerResponse,
    WorkflowWebhookResponse,
)
from rag_app.security.authorization import authorize

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workflows", tags=["workflow-triggers"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


async def _verify_workspace_trigger_access(
    session: AsyncSession,
    user: User,
    workspace_id: uuid.UUID,
) -> Workspace:
    """Enforces workspace membership and department authorization to trigger workflows."""
    ws = await session.scalar(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.deleted_at.is_(None),
            Workspace.is_active == True,
        )
    )
    if not ws:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' not found or inactive.",
        )

    if user.role == "ADMIN":
        return ws

    dept = await session.get(Department, ws.department_id)
    if not dept or not (await authorize(session, user, "VIEW", dept.code)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: You do not have access to department '{dept.code if dept else ''}'.",
        )

    member = await session.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == ws.id,
            WorkspaceMember.user_id == user.id,
        )
    )
    if not member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: You are not a member of workspace '{ws.name}'.",
        )

    # Viewers cannot trigger execution tasks
    if member.role not in ("OWNER", "EDITOR"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Triggering workflow execution requires 'EDITOR' or 'OWNER' role.",
        )

    return ws


# ============================================================================
# 1. Manual / API Trigger Endpoint
# ============================================================================

@router.post(
    "/definitions/{workflow_id}/trigger",
    response_model=WorkflowTriggerResponse,
    status_code=status.HTTP_200_OK,
)
async def trigger_workflow(
    workflow_id: uuid.UUID,
    payload: WorkflowTriggerRequest,
    user: CurrentUser,
    session: DbSession,
    workspace_id: uuid.UUID = Query(..., description="Target workspace ID"),
) -> WorkflowTriggerResponse:
    """
    Authenticated Manual Trigger API:
    - Resolves the active immutable WorkflowVersion.
    - Creates a durable WorkflowRun linked to the exact version.
    - Supports idempotency keys.
    - Enforces workspace permissions.
    - Dispatches execution through DAGExecutionEngine.
    """
    await _verify_workspace_trigger_access(session, user, workspace_id)
    service = WorkflowTriggerService(session)
    return await service.trigger_manual(
        workflow_id=workflow_id,
        workspace_id=workspace_id,
        user=user,
        request=payload,
    )


# ============================================================================
# 2. Signed Inbound Webhook Trigger Endpoint
# ============================================================================

@router.post(
    "/triggers/webhook",
    response_model=WorkflowWebhookResponse,
    status_code=status.HTTP_200_OK,
)
async def handle_inbound_webhook(
    request: Request,
    session: DbSession,
    x_twenty_webhook_signature: Optional[str] = Header(None, alias="X-Twenty-Webhook-Signature"),
    x_twenty_signature: Optional[str] = Header(None, alias="X-Twenty-Signature"),
    x_twenty_webhook_timestamp: Optional[str] = Header(None, alias="X-Twenty-Webhook-Timestamp"),
    x_twenty_webhook_nonce: Optional[str] = Header(None, alias="X-Twenty-Webhook-Nonce"),
) -> WorkflowWebhookResponse:
    """
    Signed Inbound Webhook Endpoint for external events (e.g. Twenty CRM):
    - Validates HMAC-SHA256 signature and timestamp drift.
    - Maps server-verified identity to target workflow and workspace.
    - Persists acceptance and version pinning in Transactional Outbox BEFORE responding.
    - Replay deduplication prevents duplicate event runs.
    """
    raw_body = await request.body()
    sig = x_twenty_webhook_signature or x_twenty_signature
    service = WorkflowTriggerService(session)
    return await service.handle_inbound_webhook(
        raw_bytes=raw_body,
        signature_header=sig,
        timestamp_header=x_twenty_webhook_timestamp,
        nonce_header=x_twenty_webhook_nonce,
    )


# ============================================================================
# 3. Schedule State & Outbox Sweep Endpoints
# ============================================================================

@router.get(
    "/definitions/{workflow_id}/schedule",
    response_model=Optional[WorkflowScheduleStateResponse],
    status_code=status.HTTP_200_OK,
)
async def get_schedule_state(
    workflow_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    workspace_id: uuid.UUID = Query(...),
) -> Optional[WorkflowScheduleStateResponse]:
    """Retrieves schedule window and state for a scheduled workflow."""
    await _verify_workspace_trigger_access(session, user, workspace_id)
    state = await session.scalar(
        select(WorkflowScheduleState).where(WorkflowScheduleState.workflow_id == workflow_id)
    )
    if not state:
        return None
    return WorkflowScheduleStateResponse.model_validate(state)


@router.post(
    "/triggers/outbox/recover",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def recover_outbox(
    user: CurrentUser,
    session: DbSession,
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Admin endpoint to manually trigger outbox sweep of pending events."""
    if user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only administrators can trigger manual outbox recovery.",
        )
    service = WorkflowTriggerService(session)
    recovered = await service.recover_pending_outbox_events(limit=limit, system_user_id=user.id)
    return {"recovered_count": recovered}
