"""
FastAPI Router for Persistent Workflow Definitions and Versioning.
Enforces workspace isolation, RBAC (VIEWER vs EDITOR/OWNER/ADMIN), immutable published versioning,
and transactional publication with row-level locking.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import Department, User, Workspace, WorkspaceMember
from rag_app.db.session import get_db_session
from rag_app.orchestration.workflow_definition_service import WorkflowDefinitionService
from rag_app.schemas.workflow import WorkflowRunResponse
from rag_app.schemas.workflow_definition import (
    WorkflowDAGValidationResponse,
    WorkflowDefinitionCreate,
    WorkflowDefinitionResponse,
    WorkflowDefinitionUpdate,
    WorkflowTestRunRequest,
    WorkflowVersionCreateDraft,
    WorkflowVersionResponse,
    WorkflowVersionStatus,
    WorkflowVersionUpdate,
)
from rag_app.security.authorization import authorize

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workflows/definitions", tags=["workflow-definitions"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


async def _verify_workspace_access(
    session: AsyncSession,
    user: User,
    workspace_id: uuid.UUID,
    require_write: bool = False,
) -> Workspace:
    """
    Enforces workspace existence, department authorization, and membership role checks.
    """
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

    if require_write and member.role not in ("OWNER", "EDITOR"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Modifying workflow definitions requires 'EDITOR' or 'OWNER' role.",
        )

    return ws


# ---------------------------------------------------------------------------
# Workflow Definition Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=WorkflowDefinitionResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    payload: WorkflowDefinitionCreate,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowDefinitionResponse:
    """
    Creates a new WorkflowDefinition and initializes Version 1 (DRAFT).
    Requires EDITOR, OWNER, or ADMIN role.
    """
    await _verify_workspace_access(session, user, payload.workspace_id, require_write=True)
    service = WorkflowDefinitionService(session)
    wf, v1 = await service.create_workflow(payload.workspace_id, user, payload)
    return WorkflowDefinitionResponse.model_validate(wf)


@router.get("", response_model=list[WorkflowDefinitionResponse], status_code=status.HTTP_200_OK)
async def list_workflows(
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
) -> list[WorkflowDefinitionResponse]:
    """
    Lists all active workflow definitions in a workspace.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=False)
    service = WorkflowDefinitionService(session)
    workflows = await service.list_workflows(workspace_id, skip=skip, limit=limit)
    return [WorkflowDefinitionResponse.model_validate(w) for w in workflows]


@router.get("/{workflow_id}", response_model=WorkflowDefinitionResponse, status_code=status.HTTP_200_OK)
async def get_workflow(
    workflow_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowDefinitionResponse:
    """
    Retrieves a single workflow definition by ID.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=False)
    service = WorkflowDefinitionService(session)
    wf = await service.get_workflow(workspace_id, workflow_id)
    return WorkflowDefinitionResponse.model_validate(wf)


@router.put("/{workflow_id}", response_model=WorkflowDefinitionResponse, status_code=status.HTTP_200_OK)
async def update_workflow(
    workflow_id: uuid.UUID,
    workspace_id: uuid.UUID,
    payload: WorkflowDefinitionUpdate,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowDefinitionResponse:
    """
    Updates workflow metadata (name, description, is_active).
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=True)
    service = WorkflowDefinitionService(session)
    wf = await service.update_workflow(workspace_id, workflow_id, payload)
    return WorkflowDefinitionResponse.model_validate(wf)


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> None:
    """
    Soft-deletes a workflow definition.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=True)
    service = WorkflowDefinitionService(session)
    await service.delete_workflow(workspace_id, workflow_id)


# ---------------------------------------------------------------------------
# Workflow Version Endpoints
# ---------------------------------------------------------------------------

@router.post("/{workflow_id}/versions", response_model=WorkflowVersionResponse, status_code=status.HTTP_201_CREATED)
async def create_draft_version(
    workflow_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    payload: Optional[WorkflowVersionCreateDraft] = None,
) -> WorkflowVersionResponse:
    """
    Creates a new draft version for a workflow, optionally forked from a specified version.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=True)
    service = WorkflowDefinitionService(session)
    version = await service.create_draft_version(workspace_id, workflow_id, user, payload)
    return WorkflowVersionResponse.model_validate(version)


@router.get("/{workflow_id}/versions", response_model=list[WorkflowVersionResponse], status_code=status.HTTP_200_OK)
async def list_versions(
    workflow_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> list[WorkflowVersionResponse]:
    """
    Lists version history for a workflow.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=False)
    service = WorkflowDefinitionService(session)
    versions = await service.list_versions(workspace_id, workflow_id)
    return [WorkflowVersionResponse.model_validate(v) for v in versions]


@router.get("/{workflow_id}/versions/{version_id}", response_model=WorkflowVersionResponse, status_code=status.HTTP_200_OK)
async def get_version(
    workflow_id: uuid.UUID,
    version_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowVersionResponse:
    """
    Retrieves exact details for a specific workflow version.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=False)
    service = WorkflowDefinitionService(session)
    version = await service.get_version(workspace_id, workflow_id, version_id)
    return WorkflowVersionResponse.model_validate(version)


@router.put("/{workflow_id}/versions/{version_id}", response_model=WorkflowVersionResponse, status_code=status.HTTP_200_OK)
async def update_draft_version(
    workflow_id: uuid.UUID,
    version_id: uuid.UUID,
    workspace_id: uuid.UUID,
    payload: WorkflowVersionUpdate,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowVersionResponse:
    """
    Updates steps, triggers, or canvas layout of a DRAFT version.
    Fails with 409 Conflict if the version is already ACTIVE or ARCHIVED.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=True)
    service = WorkflowDefinitionService(session)
    version = await service.update_draft_version(workspace_id, workflow_id, version_id, payload)
    return WorkflowVersionResponse.model_validate(version)


@router.post("/{workflow_id}/versions/{version_id}/validate", response_model=WorkflowDAGValidationResponse, status_code=status.HTTP_200_OK)
async def validate_version_dag(
    workflow_id: uuid.UUID,
    version_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowDAGValidationResponse:
    """
    Deterministically validates the DAG structure of a workflow version without publishing it.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=False)
    service = WorkflowDefinitionService(session)
    return await service.validate_version_dag(workspace_id, workflow_id, version_id)


@router.post("/{workflow_id}/versions/{version_id}/publish", response_model=WorkflowVersionResponse, status_code=status.HTTP_200_OK)
async def publish_version(
    workflow_id: uuid.UUID,
    version_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowVersionResponse:
    """
    Atomically publishes a version to ACTIVE state:
    - Enforces valid DAG (fails with 422 Unprocessable Entity if cycles or invalid dependencies).
    - Archives previous ACTIVE version.
    - Sets active_version_id pointer on WorkflowDefinition.
    - Uses row-level locking to prevent concurrent publication conflicts.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=True)
    service = WorkflowDefinitionService(session)
    published = await service.publish_version(workspace_id, workflow_id, version_id, user)
    return WorkflowVersionResponse.model_validate(published)


@router.post(
    "/{workflow_id}/versions/{version_id}/test",
    response_model=WorkflowRunResponse,
    status_code=status.HTTP_200_OK,
)
async def test_workflow_version(
    workflow_id: uuid.UUID,
    version_id: uuid.UUID,
    workspace_id: uuid.UUID,
    payload: WorkflowTestRunRequest,
    user: CurrentUser,
    session: DbSession,
) -> WorkflowRunResponse:
    """
    Executes a draft or active workflow version in server-controlled safe TEST mode:
    - Binds execution strictly to the tested version ID.
    - Real CRM mutations, email dispatch, and external side-effects are disabled with dry-run protection.
    - Idempotency keys are isolated under 'test:' namespace.
    - Does not publish the draft or mutate production workflow state.
    - Requires 'EDITOR' or 'OWNER' role in the workspace.
    """
    await _verify_workspace_access(session, user, workspace_id, require_write=True)
    service = WorkflowDefinitionService(session)
    run, tasks = await service.execute_test_run(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        version_id=version_id,
        user=user,
        payload=payload,
    )
    from rag_app.api.workflow_routes import _build_workflow_run_response, _get_workflow_draft_and_approval

    extra = await _get_workflow_draft_and_approval(session, run)
    extra["is_test"] = True
    return _build_workflow_run_response(run, tasks, extra_details=extra, user_role=user.role)


