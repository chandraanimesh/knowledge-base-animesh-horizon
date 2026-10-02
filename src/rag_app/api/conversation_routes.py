import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.conversation.service import ConversationService
from rag_app.db.models import User
from rag_app.db.session import get_db_session
from rag_app.schemas.rag import (
    ConversationDetailResponse,
    ConversationResponse,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    session: DbSession,
    user: User = Depends(get_current_user),
    workspace_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[ConversationResponse]:
    """
    List conversations for the current authenticated user.
    Optionally scoped by workspace_id if specified.
    """
    service = ConversationService(session)
    return await service.list_conversations(user=user, workspace_id=workspace_id, limit=limit)


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    session: DbSession,
    user: User = Depends(get_current_user),
    title: str | None = None,
    workspace_id: uuid.UUID | None = Query(default=None),
) -> ConversationResponse:
    """
    Create a new empty conversation for the current user in the given workspace.
    """
    service = ConversationService(session)
    return await service.create_conversation(user=user, title=title, workspace_id=workspace_id)



@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> ConversationDetailResponse:
    """
    Get full conversation message history.
    """
    service = ConversationService(session)
    return await service.get_conversation_detail(conversation_id=conversation_id, user=user)


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    """
    Delete a conversation and all its messages.
    """
    service = ConversationService(session)
    return await service.delete_conversation(conversation_id=conversation_id, user=user)
