import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import User
from rag_app.db.session import get_db_session
from rag_app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceResponse,
    WorkspaceUpdate,
    WorkspaceMemberCreate,
    WorkspaceMemberResponse,
    WorkspaceMemberUpdate,
    WorkspaceProviderProfileUpdate,
    WorkspaceProviderProfileResponse,
    ProviderRegistrySummary,
)
from rag_app.workspace.service import WorkspaceService

router = APIRouter(prefix="/workspaces", tags=["workspaces"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    payload: WorkspaceCreate,
    user: CurrentUser,
    session: DbSession,
) -> WorkspaceResponse:
    service = WorkspaceService(session)
    return await service.create_workspace(user, payload)


@router.get("", response_model=List[WorkspaceResponse], status_code=status.HTTP_200_OK)
async def list_workspaces(
    user: CurrentUser,
    session: DbSession,
    department_id: Optional[uuid.UUID] = Query(None, description="Filter by department UUID"),
    include_inactive: bool = Query(False, description="Include inactive workspaces (default false)"),
) -> List[WorkspaceResponse]:
    service = WorkspaceService(session)
    return await service.list_workspaces(
        user=user,
        department_id=department_id,
        include_inactive=include_inactive,
    )


@router.get("/{workspace_id}", response_model=WorkspaceResponse, status_code=status.HTTP_200_OK)
async def get_workspace(
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> WorkspaceResponse:
    service = WorkspaceService(session)
    return await service.get_workspace(user, workspace_id)


@router.patch("/{workspace_id}", response_model=WorkspaceResponse, status_code=status.HTTP_200_OK)
async def update_workspace(
    workspace_id: uuid.UUID,
    payload: WorkspaceUpdate,
    user: CurrentUser,
    session: DbSession,
) -> WorkspaceResponse:
    service = WorkspaceService(session)
    return await service.update_workspace(user, workspace_id, payload)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> None:
    service = WorkspaceService(session)
    await service.delete_workspace(user, workspace_id)


# ------------------------------------------------------------------
# Workspace Membership Endpoints
# ------------------------------------------------------------------


@router.get(
    "/{workspace_id}/members",
    response_model=List[WorkspaceMemberResponse],
    status_code=status.HTTP_200_OK,
)
async def list_workspace_members(
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> List[WorkspaceMemberResponse]:
    service = WorkspaceService(session)
    return await service.list_members(user, workspace_id)


@router.post(
    "/{workspace_id}/members",
    response_model=WorkspaceMemberResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_workspace_member(
    workspace_id: uuid.UUID,
    payload: WorkspaceMemberCreate,
    user: CurrentUser,
    session: DbSession,
) -> WorkspaceMemberResponse:
    service = WorkspaceService(session)
    return await service.add_member(user, workspace_id, payload)


@router.patch(
    "/{workspace_id}/members/{user_id}",
    response_model=WorkspaceMemberResponse,
    status_code=status.HTTP_200_OK,
)
async def update_workspace_member_role(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: WorkspaceMemberUpdate,
    user: CurrentUser,
    session: DbSession,
) -> WorkspaceMemberResponse:
    service = WorkspaceService(session)
    return await service.update_member_role(user, workspace_id, user_id, payload)


@router.delete(
    "/{workspace_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_workspace_member(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> None:
    service = WorkspaceService(session)
    await service.remove_member(user, workspace_id, user_id)


# ------------------------------------------------------------------
# Workspace Provider Profile Endpoints
# ------------------------------------------------------------------


@router.get(
    "/{workspace_id}/provider-profile",
    response_model=Optional[WorkspaceProviderProfileResponse],
    status_code=status.HTTP_200_OK,
    summary="Get provider profile for a workspace",
    description=(
        "Returns the active provider profile for the workspace, or null if none has been set. "
        "ADMIN, OWNER, EDITOR, and VIEWER can read the profile."
    ),
)
async def get_workspace_provider_profile(
    workspace_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> Optional[WorkspaceProviderProfileResponse]:
    service = WorkspaceService(session)
    return await service.get_provider_profile(user, workspace_id)


@router.put(
    "/{workspace_id}/provider-profile",
    response_model=WorkspaceProviderProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Create or update provider profile for a workspace",
    description=(
        "Creates or updates the provider profile for the workspace. "
        "Only ADMIN and workspace OWNER can modify. "
        "All provider names must already be registered in the platform provider registry. "
        "IMPORTANT: Never send real API keys in this request body. "
        "Use credential_ref as a human-readable label for the external secret."
    ),
)
async def upsert_workspace_provider_profile(
    workspace_id: uuid.UUID,
    payload: WorkspaceProviderProfileUpdate,
    user: CurrentUser,
    session: DbSession,
) -> WorkspaceProviderProfileResponse:
    service = WorkspaceService(session)
    return await service.upsert_provider_profile(user, workspace_id, payload)


# ------------------------------------------------------------------
# Provider Registry Summary (for frontend dropdowns)
# ------------------------------------------------------------------


@router.get(
    "/registry/capabilities",
    response_model=ProviderRegistrySummary,
    status_code=status.HTTP_200_OK,
    summary="List all registered providers by capability",
    description=(
        "Returns all provider names registered in the platform registry, "
        "grouped by capability. Used by the frontend to populate provider selection dropdowns. "
        "Any authenticated user can call this endpoint."
    ),
)
async def list_provider_capabilities(
    user: CurrentUser,
) -> ProviderRegistrySummary:
    from rag_app.providers.resolver import get_provider_registry_summary
    capabilities = await get_provider_registry_summary()
    return ProviderRegistrySummary(capabilities=capabilities)
