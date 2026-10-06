"""
Workspace domain tools for FastMCP.
Exposes:
- create_workspace
- list_workspaces
- get_workspace
- update_workspace
- delete_workspace
- list_workspace_members
- add_workspace_member
- update_workspace_member_role
- remove_workspace_member
- get_workspace_provider_profile
- upsert_workspace_provider_profile
- list_provider_capabilities

Reuses existing WorkspaceService and Provider resolver.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import User
from rag_app.mcp.gateway import mcp_gateway
from rag_app.providers.resolver import get_provider_registry_summary
from rag_app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceMemberCreate,
    WorkspaceMemberUpdate,
    WorkspaceProviderProfileUpdate,
    WorkspaceUpdate,
)
from rag_app.workspace.service import WorkspaceService


# ── Underlying Handlers ───────────────────────────────────────────────────────

async def handle_create_workspace(
    session: AsyncSession,
    user: User,
    name: str,
    code: str,
    department_id: str,
    description: Optional[str] = None,
    is_active: bool = True,
) -> dict[str, Any]:
    try:
        dept_uuid = uuid.UUID(department_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid department UUID: {department_id}")

    payload = WorkspaceCreate(
        name=name,
        code=code,
        department_id=dept_uuid,
        description=description,
        is_active=is_active,
    )
    service = WorkspaceService(session)
    res = await service.create_workspace(user, payload)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def handle_list_workspaces(
    session: AsyncSession,
    user: User,
    department_id: Optional[str] = None,
    include_inactive: bool = False,
) -> list[dict[str, Any]]:
    dept_uuid = uuid.UUID(department_id) if department_id else None
    service = WorkspaceService(session)
    res = await service.list_workspaces(
        user=user,
        department_id=dept_uuid,
        include_inactive=include_inactive,
    )
    return [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in res]


async def handle_get_workspace(
    workspace_id: str,
    session: AsyncSession,
    user: User,
) -> dict[str, Any]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    service = WorkspaceService(session)
    res = await service.get_workspace(user, ws_uuid)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def handle_update_workspace(
    workspace_id: str,
    session: AsyncSession,
    user: User,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> dict[str, Any]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    payload = WorkspaceUpdate(
        name=name,
        description=description,
        is_active=is_active,
    )
    service = WorkspaceService(session)
    res = await service.update_workspace(user, ws_uuid, payload)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def handle_delete_workspace(
    workspace_id: str,
    session: AsyncSession,
    user: User,
) -> dict[str, str]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    service = WorkspaceService(session)
    await service.delete_workspace(user, ws_uuid)
    return {"status": "success", "message": f"Workspace {workspace_id} deleted."}


async def handle_list_workspace_members(
    workspace_id: str,
    session: AsyncSession,
    user: User,
) -> list[dict[str, Any]]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    service = WorkspaceService(session)
    res = await service.list_members(user, ws_uuid)
    return [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in res]


async def handle_add_workspace_member(
    workspace_id: str,
    user_id: str,
    role: str,
    session: AsyncSession,
    user: User,
) -> dict[str, Any]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
        target_uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID for workspace_id or user_id.")

    payload = WorkspaceMemberCreate(user_id=target_uid, role=role)
    service = WorkspaceService(session)
    res = await service.add_member(user, ws_uuid, payload)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def handle_update_workspace_member_role(
    workspace_id: str,
    user_id: str,
    role: str,
    session: AsyncSession,
    user: User,
) -> dict[str, Any]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
        target_uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID for workspace_id or user_id.")

    payload = WorkspaceMemberUpdate(role=role)
    service = WorkspaceService(session)
    res = await service.update_member_role(user, ws_uuid, target_uid, payload)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def handle_remove_workspace_member(
    workspace_id: str,
    user_id: str,
    session: AsyncSession,
    user: User,
) -> dict[str, str]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
        target_uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID for workspace_id or user_id.")

    service = WorkspaceService(session)
    await service.remove_member(user, ws_uuid, target_uid)
    return {"status": "success", "message": f"User {user_id} removed from workspace {workspace_id}."}


async def handle_get_workspace_provider_profile(
    workspace_id: str,
    session: AsyncSession,
    user: User,
) -> Optional[dict[str, Any]]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    service = WorkspaceService(session)
    res = await service.get_provider_profile(user, ws_uuid)
    return res.model_dump() if (res and hasattr(res, "model_dump")) else None


async def handle_upsert_workspace_provider_profile(
    workspace_id: str,
    session: AsyncSession,
    user: User,
    chat_provider: Optional[str] = None,
    chat_model: Optional[str] = None,
    chat_credential_ref: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_credential_ref: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_model: Optional[str] = None,
    rerank_credential_ref: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    payload = WorkspaceProviderProfileUpdate(
        chat_provider=chat_provider,
        chat_model=chat_model,
        chat_credential_ref=chat_credential_ref,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_credential_ref=embedding_credential_ref,
        rerank_provider=rerank_provider,
        rerank_model=rerank_model,
        rerank_credential_ref=rerank_credential_ref,
        notes=notes,
    )
    service = WorkspaceService(session)
    res = await service.upsert_provider_profile(user, ws_uuid, payload)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def handle_list_provider_capabilities() -> dict[str, Any]:
    caps = await get_provider_registry_summary()
    return {"capabilities": caps}


# ── MCP Tool Callers ─────────────────────────────────────────────────────────

async def create_workspace(
    name: str,
    code: str,
    department_id: str,
    description: Optional[str] = None,
    is_active: bool = True,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "create_workspace",
        handle_create_workspace,
        name=name,
        code=code,
        department_id=department_id,
        description=description,
        is_active=is_active,
        auth_token=auth_token,
    )


async def list_workspaces(
    department_id: Optional[str] = None,
    include_inactive: bool = False,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "list_workspaces",
        handle_list_workspaces,
        department_id=department_id,
        include_inactive=include_inactive,
        auth_token=auth_token,
    )


async def get_workspace(
    workspace_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "get_workspace",
        handle_get_workspace,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


async def update_workspace(
    workspace_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_active: Optional[bool] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "update_workspace",
        handle_update_workspace,
        workspace_id=workspace_id,
        name=name,
        description=description,
        is_active=is_active,
        auth_token=auth_token,
    )


async def delete_workspace(
    workspace_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "delete_workspace",
        handle_delete_workspace,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


async def list_workspace_members(
    workspace_id: str,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "list_workspace_members",
        handle_list_workspace_members,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


async def add_workspace_member(
    workspace_id: str,
    user_id: str,
    role: str = "MEMBER",
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "add_workspace_member",
        handle_add_workspace_member,
        workspace_id=workspace_id,
        user_id=user_id,
        role=role,
        auth_token=auth_token,
    )


async def update_workspace_member_role(
    workspace_id: str,
    user_id: str,
    role: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "update_workspace_member_role",
        handle_update_workspace_member_role,
        workspace_id=workspace_id,
        user_id=user_id,
        role=role,
        auth_token=auth_token,
    )


async def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "remove_workspace_member",
        handle_remove_workspace_member,
        workspace_id=workspace_id,
        user_id=user_id,
        auth_token=auth_token,
    )


async def get_workspace_provider_profile(
    workspace_id: str,
    auth_token: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    return await mcp_gateway.execute(
        "get_workspace_provider_profile",
        handle_get_workspace_provider_profile,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


async def upsert_workspace_provider_profile(
    workspace_id: str,
    chat_provider: Optional[str] = None,
    chat_model: Optional[str] = None,
    chat_credential_ref: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_credential_ref: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_model: Optional[str] = None,
    rerank_credential_ref: Optional[str] = None,
    notes: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "upsert_workspace_provider_profile",
        handle_upsert_workspace_provider_profile,
        workspace_id=workspace_id,
        chat_provider=chat_provider,
        chat_model=chat_model,
        chat_credential_ref=chat_credential_ref,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_credential_ref=embedding_credential_ref,
        rerank_provider=rerank_provider,
        rerank_model=rerank_model,
        rerank_credential_ref=rerank_credential_ref,
        notes=notes,
        auth_token=auth_token,
    )


async def list_provider_capabilities(
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "list_provider_capabilities",
        handle_list_provider_capabilities,
        auth_token=auth_token,
    )
