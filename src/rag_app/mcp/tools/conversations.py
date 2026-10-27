"""
Conversation domain tools for FastMCP.
Exposes:
- list_conversations
- create_conversation
- get_conversation
- delete_conversation

Reuses existing ConversationService. Caller identity is strictly resolved from context.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.conversation.service import ConversationService
from rag_app.db.models import User
from rag_app.mcp.gateway import mcp_gateway


# ── Underlying Handlers ───────────────────────────────────────────────────────

async def _resolve_workspace_identifier(
    session: AsyncSession,
    user: User,
    workspace_id: Optional[str],
) -> Optional[str]:
    """
    Resolves omitted workspace_id adhering to application policy:
    1. If explicitly provided, returns it as-is.
    2. If user is ADMIN, returns None (legitimate Global Chat).
    3. If user is USER and workspace_id is omitted:
       - Checks allowed workspaces. If exactly 1 allowed workspace exists, returns it.
       - If no allowed workspaces exist, checks allowed departments (with VIEW permission).
         If exactly 1 allowed department exists, returns department code so ConversationService
         can auto-provision/link the established department workspace.
       - If multiple workspaces/departments exist, raises HTTPException(400) requiring explicit workspace_id.
       - If 0 workspaces/departments exist, raises HTTPException(403) requiring valid workspace membership.
    """
    if workspace_id and str(workspace_id).strip():
        return str(workspace_id).strip()
    if user.role == "ADMIN":
        return None

    from rag_app.security.authorization import get_allowed_workspaces, get_allowed_departments
    from rag_app.db.models import Department

    allowed_ws = await get_allowed_workspaces(session, user)
    if allowed_ws and len(allowed_ws) == 1:
        return str(allowed_ws[0])
    elif allowed_ws and len(allowed_ws) > 1:
        raise HTTPException(
            status_code=400,
            detail="Multiple accessible workspaces found. Please specify 'workspace_id'.",
        )

    # If user has no existing workspace memberships, check their allowed departments
    allowed_depts = await get_allowed_departments(session, user)
    if not allowed_depts:
        raise HTTPException(
            status_code=403,
            detail="Standard users must belong to an active department/workspace to use conversations. Please specify 'workspace_id'.",
        )
    if len(allowed_depts) == 1:
        dept = await session.get(Department, allowed_depts[0])
        if dept:
            return dept.code

    raise HTTPException(
        status_code=400,
        detail="Multiple accessible departments found. Please specify 'workspace_id'.",
    )


async def handle_list_conversations(
    session: AsyncSession,
    user: User,
    workspace_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List conversations for current user."""
    resolved_ws = await _resolve_workspace_identifier(session, user, workspace_id)
    service = ConversationService(session)
    res = await service.list_conversations(user=user, workspace_id=resolved_ws, limit=min(max(1, limit), 100))
    return [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in res]


async def handle_create_conversation(
    session: AsyncSession,
    user: User,
    title: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new conversation for current user."""
    resolved_ws = await _resolve_workspace_identifier(session, user, workspace_id)
    service = ConversationService(session)
    res = await service.create_conversation(user=user, title=title, workspace_id=resolved_ws)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def handle_get_conversation(
    conversation_id: str,
    session: AsyncSession,
    user: User,
) -> dict[str, Any]:
    """Get full conversation detail."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid conversation UUID: {conversation_id}")

    service = ConversationService(session)
    res = await service.get_conversation_detail(conversation_id=conv_uuid, user=user)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def handle_delete_conversation(
    conversation_id: str,
    session: AsyncSession,
    user: User,
) -> dict[str, str]:
    """Delete conversation and messages."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid conversation UUID: {conversation_id}")

    service = ConversationService(session)
    return await service.delete_conversation(conversation_id=conv_uuid, user=user)


# ── MCP Tool Callers ─────────────────────────────────────────────────────────

async def list_conversations(
    workspace_id: Optional[str] = None,
    limit: int = 50,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List conversations for current user."""
    return await mcp_gateway.execute(
        "list_conversations",
        handle_list_conversations,
        workspace_id=workspace_id,
        limit=limit,
        auth_token=auth_token,
    )


async def create_conversation(
    title: Optional[str] = None,
    workspace_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new conversation for current user."""
    return await mcp_gateway.execute(
        "create_conversation",
        handle_create_conversation,
        title=title,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


async def get_conversation(
    conversation_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Get complete conversation message history."""
    return await mcp_gateway.execute(
        "get_conversation",
        handle_get_conversation,
        conversation_id=conversation_id,
        auth_token=auth_token,
    )


async def delete_conversation(
    conversation_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    """Delete a conversation."""
    return await mcp_gateway.execute(
        "delete_conversation",
        handle_delete_conversation,
        conversation_id=conversation_id,
        auth_token=auth_token,
    )
