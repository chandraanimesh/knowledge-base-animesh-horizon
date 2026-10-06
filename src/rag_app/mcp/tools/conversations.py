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

async def handle_list_conversations(
    session: AsyncSession,
    user: User,
    workspace_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List conversations for current user."""
    ws_uuid = uuid.UUID(workspace_id) if workspace_id else None
    service = ConversationService(session)
    res = await service.list_conversations(user=user, workspace_id=ws_uuid, limit=min(max(1, limit), 100))
    return [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in res]


async def handle_create_conversation(
    session: AsyncSession,
    user: User,
    title: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new conversation for current user."""
    ws_uuid = uuid.UUID(workspace_id) if workspace_id else None
    service = ConversationService(session)
    res = await service.create_conversation(user=user, title=title, workspace_id=ws_uuid)
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
