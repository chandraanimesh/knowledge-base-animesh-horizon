"""
API routes for Workspace-Scoped MCP Connection Manager (Milestone 6C).
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import User
from rag_app.db.session import get_db_session
from rag_app.mcp.connection_service import WorkspaceMCPConnectionService
from rag_app.schemas.mcp_connection import (
    MCPApprovalDecisionRequest,
    MCPApprovalResponse,
    MCPConnectionCreate,
    MCPConnectionResponse,
    MCPConnectionUpdate,
    MCPProposalCreateRequest,
    MCPToolAllowlistUpdate,
    MCPToolInvokeRequest,
    MCPToolInvokeResponse,
    MCPToolToggleRequest,
)
from rag_app.schemas.crm import CRMConnectionSyncRequest, CRMConnectionReconcileRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces/{workspace_id}/mcp-connections", tags=["workspace-mcp-connections"])

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]



@router.post(
    "",
    response_model=MCPConnectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new external MCP connection in the workspace",
)
async def create_connection(
    workspace_id: uuid.UUID,
    payload: MCPConnectionCreate,
    user: CurrentUser,
    session: DbSession,
) -> MCPConnectionResponse:
    service = WorkspaceMCPConnectionService(session)
    return await service.create_connection(user, workspace_id, payload)


@router.get(
    "",
    response_model=list[MCPConnectionResponse],
    summary="List all MCP connections visible to the authorized workspace member",
)
async def list_connections(
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> list[MCPConnectionResponse]:
    service = WorkspaceMCPConnectionService(session)
    return await service.list_connections(user, workspace_id)


@router.get(
    "/{connection_id}",
    response_model=MCPConnectionResponse,
    summary="Get details and discovered tools for a workspace MCP connection",
)
async def get_connection(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> MCPConnectionResponse:
    service = WorkspaceMCPConnectionService(session)
    return await service.get_connection(user, workspace_id, connection_id)


@router.patch(
    "/{connection_id}",
    response_model=MCPConnectionResponse,
    summary="Update MCP connection configuration or enable/disable connection",
)
async def update_connection(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    payload: MCPConnectionUpdate,
    user: CurrentUser,
    session: DbSession,
) -> MCPConnectionResponse:
    service = WorkspaceMCPConnectionService(session)
    return await service.update_connection(user, workspace_id, connection_id, payload)


@router.delete(
    "/{connection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an MCP connection from the workspace",
)
async def delete_connection(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> None:
    service = WorkspaceMCPConnectionService(session)
    await service.delete_connection(user, workspace_id, connection_id)


@router.post(
    "/{connection_id}/sync",
    response_model=MCPConnectionResponse,
    summary="Probe endpoint, capture server capabilities, and refresh cached tool metadata",
)
async def sync_tools(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> MCPConnectionResponse:
    service = WorkspaceMCPConnectionService(session)
    return await service.sync_tools(user, workspace_id, connection_id)


@router.post(
    "/{connection_id}/test",
    summary="Probe connectivity and credential authentication using the connection's configured credential mode",
)
async def test_connection(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> dict[str, Any]:
    service = WorkspaceMCPConnectionService(session)
    return await service.test_connection(user, workspace_id, connection_id)


@router.post(
    "/{connection_id}/tools/toggle",
    response_model=MCPConnectionResponse,
    summary="Enable or disable an individual discovered tool in the connection allowlist",
)
async def toggle_tool(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    payload: MCPToolToggleRequest,
    user: CurrentUser,
    session: DbSession,
) -> MCPConnectionResponse:
    service = WorkspaceMCPConnectionService(session)
    return await service.toggle_tool(user, workspace_id, connection_id, payload.tool_name, payload.enabled)


@router.put(
    "/{connection_id}/tools/allowlist",
    response_model=MCPConnectionResponse,
    summary="Replace active enabled tool allowlist with approved tool names",
)
async def update_tool_allowlist(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    payload: MCPToolAllowlistUpdate,
    user: CurrentUser,
    session: DbSession,
) -> MCPConnectionResponse:
    service = WorkspaceMCPConnectionService(session)
    return await service.update_tool_allowlist(user, workspace_id, connection_id, payload.enabled_tools)


@router.post(
    "/{connection_id}/tools/call",
    response_model=MCPToolInvokeResponse,
    summary="Execute an approved tool on the remote MCP server (Default-Deny enforced)",
)
async def invoke_workspace_tool(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    payload: MCPToolInvokeRequest,
    user: CurrentUser,
    session: DbSession,
) -> MCPToolInvokeResponse:
    service = WorkspaceMCPConnectionService(session)
    return await service.invoke_workspace_tool(
        user,
        workspace_id,
        connection_id,
        payload.tool_name,
        payload.arguments,
    )


@router.post(
    "/{connection_id}/tools/{tool_name}/invoke",
    response_model=MCPToolInvokeResponse,
    summary="Execute an approved tool on the remote MCP server by tool_name path",
)
async def invoke_workspace_tool_by_name(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    tool_name: str,
    payload: dict[str, Any],
    user: CurrentUser,
    session: DbSession,
) -> MCPToolInvokeResponse:
    service = WorkspaceMCPConnectionService(session)
    arguments = payload.get("arguments", payload) if isinstance(payload, dict) else {}
    return await service.invoke_workspace_tool(
        user,
        workspace_id,
        connection_id,
        tool_name,
        arguments,
    )


@router.post(
    "/{connection_id}/tools/{tool_name}/propose",
    response_model=MCPApprovalResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Propose an external MCP tool mutation for human-in-the-loop review (Milestone 6E)",
)
async def propose_workspace_tool_mutation(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    tool_name: str,
    payload: MCPProposalCreateRequest,
    user: CurrentUser,
    session: DbSession,
) -> MCPApprovalResponse:
    service = WorkspaceMCPConnectionService(session)
    approval = await service.propose_tool_mutation(
        user=user,
        workspace_id=workspace_id,
        connection_id=connection_id,
        tool_name=tool_name,
        arguments=payload.arguments,
        run_id=payload.run_id,
        task_id=payload.task_id,
    )
    return MCPApprovalResponse(
        id=approval.id,
        workspace_id=workspace_id,
        connection_id=connection_id,
        tool_name=tool_name,
        target_record_id=approval.external_record_id,
        status=approval.status,
        approved_draft_hash=approval.approved_draft_hash,
        expires_at=approval.expires_at,
        created_at=approval.created_at,
    )


@router.get(
    "/approvals/{approval_id}",
    response_model=MCPApprovalResponse,
    summary="Retrieve details of a pending or processed MCP human approval ticket",
)
async def get_mcp_approval_details(
    workspace_id: uuid.UUID,
    approval_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> MCPApprovalResponse:
    from rag_app.db.models import CRMWriteBackApproval, MCPToolOperationRecord
    from fastapi import HTTPException

    service = WorkspaceMCPConnectionService(session)
    await service._check_workspace_access(user, workspace_id, require_write=False)

    approval = await session.get(CRMWriteBackApproval, approval_id)
    if not approval:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval ticket not found.")

    op_record = await session.scalar(
        select(MCPToolOperationRecord).where(MCPToolOperationRecord.approval_ticket_id == approval_id)
    )

    return MCPApprovalResponse(
        id=approval.id,
        workspace_id=workspace_id,
        connection_id=op_record.connection_id if op_record else None,
        tool_name=approval.intended_action,
        target_record_id=approval.external_record_id,
        status=approval.status,
        approved_draft_hash=approval.approved_draft_hash,
        expires_at=approval.expires_at,
        created_at=approval.created_at,
    )


@router.post(
    "/approvals/{approval_id}/decision",
    summary="Process human approval decision (APPROVE or REJECT) for proposed MCP mutation",
)
async def submit_mcp_approval_decision(
    workspace_id: uuid.UUID,
    approval_id: uuid.UUID,
    payload: MCPApprovalDecisionRequest,
    user: CurrentUser,
    session: DbSession,
) -> dict[str, Any]:
    from rag_app.db.models import CRMWriteBackApproval, MCPToolOperationRecord
    from fastapi import HTTPException

    service = WorkspaceMCPConnectionService(session)
    ws, role = await service._check_workspace_access(user, workspace_id, require_write=True)

    approval = await session.get(CRMWriteBackApproval, approval_id)
    if not approval:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approval ticket not found.")

    op_record = await session.scalar(
        select(MCPToolOperationRecord).where(MCPToolOperationRecord.approval_ticket_id == approval_id)
    )
    if not op_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Operation record for approval ticket not found.")

    # Process Rejection
    if payload.decision == "REJECTED":
        now = datetime.now(timezone.utc)
        approval.status = "REJECTED"
        approval.revoked_at = now
        op_record.status = "FAILED"
        op_record.error_message = payload.rejection_reason or "Rejected by human reviewer"
        op_record.updated_at = now
        await session.commit()
        return {
            "status": "REJECTED",
            "side_effect_executed": False,
            "approval_id": str(approval.id),
            "rejection_reason": op_record.error_message,
        }

    if payload.decision != "APPROVED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid decision '{payload.decision}'.")

    # Process Approval: Execute mutation through WorkspaceMCPConnectionService
    effective_arguments = payload.arguments if payload.arguments is not None else op_record.arguments
    res = await service.invoke_workspace_tool(
        user=user,
        workspace_id=workspace_id,
        connection_id=op_record.connection_id,
        tool_name=op_record.tool_name,
        arguments=effective_arguments,
        idempotency_key=op_record.idempotency_key,
        approval_ticket_id=approval.id,
        run_id=op_record.run_id,
        task_id=op_record.task_id,
    )

    return {
        "status": "CONFIRMED" if not res.is_error else "FAILED",
        "side_effect_executed": not res.is_error,
        "approval_id": str(approval.id),
        "result": res.model_dump(),
    }


@router.post(
    "/{connection_id}/crm-sync",
    summary="Trigger durable bounded batch/incremental CRM sync for this workspace connection",
)
async def trigger_connection_crm_sync(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    payload: CRMConnectionSyncRequest,
    user: CurrentUser,
    session: DbSession,
) -> dict[str, Any]:
    from fastapi import HTTPException
    from rag_app.crm.sync_service import CRMSyncService, ConcurrencyConflictError
    from rag_app.db.models import Workspace

    ws = await session.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace {workspace_id} not found.")

    target_dept = payload.department_id or ws.department_id
    if not target_dept:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No department specified or configured for workspace.")

    crm_svc = CRMSyncService(session)
    try:
        summary = await crm_svc.sync_connection_batch(
            workspace_id=workspace_id,
            connection_id=connection_id,
            department_id=target_dept,
            limit=payload.limit,
            offset=payload.offset,
            created_by=user.id,
            force=payload.force,
            max_pages=payload.max_pages,
        )
        return {
            "job_id": str(summary.job_id),
            "status": summary.status,
            "total_records": summary.total_records,
            "processed_records": summary.processed_records,
            "succeeded_records": summary.succeeded_records,
            "skipped_records": summary.skipped_records,
            "failed_records": summary.failed_records,
            "checkpoint_cursor": summary.checkpoint_cursor,
            "records": [
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
        }
    except ConcurrencyConflictError as cce:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(cce))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as ex:
        logger.error("Connection CRM sync error: %s", ex, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))


@router.post(
    "/{connection_id}/crm-reconcile",
    summary="Trigger drift reconciliation and conservative archival for this workspace connection",
)
async def trigger_connection_crm_reconcile(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    payload: CRMConnectionReconcileRequest,
    user: CurrentUser,
    session: DbSession,
) -> dict[str, Any]:
    from fastapi import HTTPException
    from rag_app.crm.sync_service import CRMSyncService
    from rag_app.db.models import Workspace

    ws = await session.get(Workspace, workspace_id)
    if not ws:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace {workspace_id} not found.")

    target_dept = payload.department_id or ws.department_id
    if not target_dept:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No department specified or configured for workspace.")

    crm_svc = CRMSyncService(session)
    try:
        return await crm_svc.reconcile_connection(
            workspace_id=workspace_id,
            connection_id=connection_id,
            department_id=target_dept,
            max_records=payload.max_records,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as ex:
        logger.error("Connection CRM reconciliation error: %s", ex, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))


@router.get(
    "/{connection_id}/crm-sync-jobs",
    summary="List sync jobs for this workspace connection",
)
async def list_connection_crm_sync_jobs(
    workspace_id: uuid.UUID,
    connection_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    limit: int = 20,
) -> list[dict[str, Any]]:
    from rag_app.db.models import CRMSyncJob

    stmt = (
        select(CRMSyncJob)
        .where(
            CRMSyncJob.workspace_id == workspace_id,
            CRMSyncJob.job_type == "CONNECTION_SYNC",
        )
        .order_by(CRMSyncJob.created_at.desc())
        .limit(limit)
    )
    jobs = (await session.scalars(stmt)).all()
    results = []
    for j in jobs:
        if j.checkpoint_cursor and str(connection_id) in j.checkpoint_cursor:
            results.append({
                "id": str(j.id),
                "job_type": j.job_type,
                "status": j.status,
                "total_records": j.total_records,
                "processed_records": j.processed_records,
                "succeeded_records": j.succeeded_records,
                "skipped_records": j.skipped_records,
                "failed_records": j.failed_records,
                "checkpoint_cursor": j.checkpoint_cursor,
                "error_message": j.error_message,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            })
    return results


