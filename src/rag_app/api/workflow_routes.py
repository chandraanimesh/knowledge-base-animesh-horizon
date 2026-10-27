"""
FastAPI Router for Adaptive Workflow Orchestration.
Exposes durable workflow execution, scoped idempotency, and status inspection.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import Department, User, UserDepartment, WorkflowRun, WorkflowTask, Workspace, WorkspaceMember
from rag_app.db.session import get_db_session
from rag_app.orchestration.service import WorkflowService
from rag_app.schemas.workflow import (
    WorkflowApprovalRequest,
    WorkflowApprovalResponse,
    WorkflowRunCreate,
    WorkflowRunResponse,
    WorkflowStatus,
    WorkflowTaskResponse,
    WorkflowTaskStatus,
)
from rag_app.security.authorization import authorize

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


async def _verify_workspace_authorization(session: AsyncSession, user: User, workspace_id: uuid.UUID) -> Workspace:
    """Enforces workspace existence and user department/membership access."""
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

    if user.role != "ADMIN":
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
    return ws


async def _get_workflow_draft_and_approval(session: AsyncSession, run: WorkflowRun) -> dict[str, Any]:
    """Retrieves authoritative draft text, approval record, and evidence items for a run."""
    from rag_app.db.models import CRMWriteBackApproval
    result: dict[str, Any] = {
        "draft_text": None,
        "evidence_items": [],
        "approval_id": None,
        "intended_action": None,
        "target_record_id": None,
    }

    # Fetch synthesis task output
    synth_task = await session.scalar(
        select(WorkflowTask).where(
            and_(
                WorkflowTask.workflow_run_id == run.id,
                WorkflowTask.task_type == "SYNTHESIS",
            )
        )
    )
    if synth_task and synth_task.output_result:
        result["draft_text"] = synth_task.output_result.get("draft_text")

    # Fetch associated active approval
    approval = await session.scalar(
        select(CRMWriteBackApproval).where(
            CRMWriteBackApproval.workflow_run_id == run.id
        ).order_by(CRMWriteBackApproval.created_at.desc())
    )
    if approval:
        result["approval_id"] = approval.id
        result["intended_action"] = getattr(approval, "intended_action", "CRM_CREATE_NOTE")
        result["target_record_id"] = approval.external_record_id

    # Fetch evidence items from tasks
    all_tasks = (await session.scalars(
        select(WorkflowTask).where(WorkflowTask.workflow_run_id == run.id)
    )).all()
    evidence = []
    for t in all_tasks:
        if t.output_result and isinstance(t.output_result, dict):
            for ev in t.output_result.get("evidence", []):
                evidence.append(ev)
    result["evidence_items"] = evidence
    return result


def _build_workflow_run_response(
    run: WorkflowRun,
    tasks: list[WorkflowTask],
    extra_details: Optional[dict[str, Any]] = None,
) -> WorkflowRunResponse:
    extra = extra_details or {}
    task_responses = [
        WorkflowTaskResponse(
            id=t.id,
            workflow_run_id=t.workflow_run_id,
            task_key=t.task_key,
            task_type=t.task_type,
            status=WorkflowTaskStatus(t.status) if hasattr(WorkflowTaskStatus, t.status) else t.status,
            dependencies=t.dependencies if isinstance(t.dependencies, list) else [],
            input_payload=t.input_payload if isinstance(t.input_payload, dict) else {},
            output_result=t.output_result if isinstance(t.output_result, dict) else None,
            error_message=t.error_message,
            tokens_spent=t.tokens_spent,
            provider_calls=t.provider_calls,
            retry_count=t.retry_count,
            max_retries=t.max_retries,
            locked_by=t.locked_by,
            fencing_token=t.fencing_token,
            created_at=t.created_at,
            updated_at=t.updated_at,
            completed_at=t.completed_at,
        )
        for t in tasks
    ]
    return WorkflowRunResponse(
        id=run.id,
        user_id=run.user_id,
        workspace_id=run.workspace_id,
        department_id=run.department_id,
        conversation_id=run.conversation_id,
        scope_type=run.scope_type,
        thread_id=run.thread_id,
        status=WorkflowStatus(run.status) if hasattr(WorkflowStatus, run.status) else run.status,
        idempotency_key=run.idempotency_key,
        request_hash=run.request_hash,
        draft_hash=run.draft_hash,
        max_tokens_budget=run.max_tokens_budget,
        tokens_spent=run.tokens_spent,
        tokens_reserved=run.tokens_reserved,
        max_provider_calls_budget=run.max_provider_calls_budget,
        provider_calls_spent=run.provider_calls_spent,
        provider_calls_reserved=run.provider_calls_reserved,
        wall_clock_timeout_seconds=run.wall_clock_timeout_seconds,
        failure_category=run.failure_category,
        error_message=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at,
        awaiting_approval_at=run.awaiting_approval_at,
        approved_at=run.approved_at,
        executing_at=run.executing_at,
        completed_at=run.completed_at,
        cancelled_at=run.cancelled_at,
        failed_at=run.failed_at,
        tasks=task_responses,
        draft_text=extra.get("draft_text"),
        evidence_items=extra.get("evidence_items", []),
        approval_id=extra.get("approval_id"),
        intended_action=extra.get("intended_action"),
        target_record_id=extra.get("target_record_id"),
    )


@router.post("/runs", response_model=WorkflowRunResponse, status_code=status.HTTP_201_CREATED)
async def create_and_execute_workflow(
    payload: WorkflowRunCreate,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowRunResponse:
    """
    Creates a new WorkflowRun with scoped idempotency and executes it through WorkflowService.
    Returns the persisted run details and truthfully recorded lifecycle state.
    """
    if payload.workspace_id is not None:
        ws = await _verify_workspace_authorization(session, user, payload.workspace_id)
        if payload.department_id is None:
            payload.department_id = ws.department_id

    service = WorkflowService(session=session)
    run, is_new = await service.get_or_create_workflow_run(user=user, request=payload)

    if is_new:
        # Execute workflow synchronously to determine initial phase outcome (AWAITING_APPROVAL, COMPLETED, or FAILED)
        await service.execute_workflow(run)
        await session.refresh(run)

    tasks_stmt = select(WorkflowTask).where(WorkflowTask.workflow_run_id == run.id).order_by(WorkflowTask.created_at)
    tasks = (await session.scalars(tasks_stmt)).all()
    extra_details = await _get_workflow_draft_and_approval(session, run)

    return _build_workflow_run_response(run, list(tasks), extra_details)


@router.get("/runs/{run_id}", response_model=WorkflowRunResponse, status_code=status.HTTP_200_OK)
async def get_workflow_run(
    run_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowRunResponse:
    """
    Retrieves the persisted WorkflowRun status and its associated tasks.
    Enforces user ownership or workspace authorization.
    """
    run = await session.get(WorkflowRun, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow run '{run_id}' not found.",
        )

    # Authorization Check
    if user.role != "ADMIN":
        if run.workspace_id is not None:
            await _verify_workspace_authorization(session, user, run.workspace_id)
        elif run.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: You do not have access to this workflow run.",
            )

    tasks_stmt = select(WorkflowTask).where(WorkflowTask.workflow_run_id == run.id).order_by(WorkflowTask.created_at)
    tasks = (await session.scalars(tasks_stmt)).all()
    extra_details = await _get_workflow_draft_and_approval(session, run)

    return _build_workflow_run_response(run, list(tasks), extra_details)


@router.post("/runs/{run_id}/approval", response_model=WorkflowApprovalResponse, status_code=status.HTTP_200_OK)
async def submit_workflow_approval(
    run_id: uuid.UUID,
    payload: WorkflowApprovalRequest,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowApprovalResponse:
    """
    Processes a human approval decision (APPROVED, REJECTED, or CANCELLED) for an AWAITING_APPROVAL workflow.
    Validates canonical draft hash, checks approver authorization, records audit trail, and resumes execution safely.
    """
    from rag_app.db.models import CRMWriteBackApproval

    run = await session.get(WorkflowRun, run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow run '{run_id}' not found.",
        )

    if run.workspace_id is not None:
        await _verify_workspace_authorization(session, user, run.workspace_id)
    elif user.role != "ADMIN" and run.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have access to this workflow run.",
        )

    # Find the active approval record
    approval = await session.scalar(
        select(CRMWriteBackApproval).where(
            CRMWriteBackApproval.workflow_run_id == run.id
        ).order_by(CRMWriteBackApproval.created_at.desc())
    )
    if not approval:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No approval record found for this workflow run.",
        )

    # Cryptographic Draft Hash Pre-Validation (detect client tampering)
    if approval.approved_draft_hash and payload.draft_hash != approval.approved_draft_hash:
        logger.warning(
            f"Draft hash mismatch for run {run_id}: expected {approval.approved_draft_hash}, provided {payload.draft_hash}."
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Draft hash does not match server-persisted draft. The draft may have changed or been tampered with.",
        )

    service = WorkflowService(session=session)
    try:
        outcome = await service.resume_workflow_approval(
            run_id=run.id,
            approval_id=approval.id,
            user=user,
            user_role=user.role,
            decision=payload.decision,
            rejection_reason=payload.rejection_reason,
            proposed_action=payload.proposed_action,
            proposed_target=payload.proposed_target or approval.external_record_id,
        )
        return WorkflowApprovalResponse(
            workflow_run_id=run.id,
            approval_id=approval.id,
            status=WorkflowStatus(outcome["status"]),
            side_effect_executed=outcome.get("side_effect_executed", False),
            writeback_result=outcome.get("writeback_result"),
        )
    except PermissionError as pe:
        logger.warning(f"Workflow approval authorization failed: {pe}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error resuming workflow approval for run {run_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow approval execution failed: {str(exc)}",
        )

