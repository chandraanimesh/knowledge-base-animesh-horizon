import logging
from typing import Annotated, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.session import get_db_session
from rag_app.db.models import User
from rag_app.api.dependencies.auth import get_current_user
from rag_app.conversation.service import ConversationService
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.retrieval import MemoryRetrievalService
from rag_app.memory.schemas import (
    MemoryCandidate,
    MemoryRecord,
    MemoryRetrievalResponse,
    MemoryType,
    MemoryStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


class CreateMemoryPayload(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    memory_type: MemoryType = MemoryType.DYNAMIC
    canonical_key: Optional[str] = None
    workspace_id: Optional[UUID] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    structured_data: Optional[dict[str, Any]] = None


class UpdateMemoryPayload(BaseModel):
    content: Optional[str] = Field(None, min_length=1, max_length=2000)
    structured_data: Optional[dict[str, Any]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class SearchMemoryPayload(BaseModel):
    query: str = Field(..., min_length=1)
    workspace_id: Optional[UUID] = None
    limit: int = Field(10, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


@router.get("", response_model=list[MemoryRecord])
async def list_memories(
    session: DbSession,
    user: CurrentUser,
    workspace_id: Optional[UUID] = Query(None, description="Workspace ID filter"),
    include_global: bool = Query(True, description="Whether to include global memories"),
    limit: int = Query(50, ge=1, le=200),
) -> list[MemoryRecord]:
    """
    List active memories for the authenticated user.
    If workspace_id is provided, verifies user membership/access before returning.
    """
    if workspace_id is not None:
        conv_service = ConversationService(session)
        await conv_service._validate_workspace_access(user, workspace_id)

    records = await MemoryMaterializer.get_user_memories(
        session=session,
        user_id=user.id,
        workspace_id=workspace_id,
        limit=limit,
        include_global=include_global,
    )
    return records


@router.post("", response_model=MemoryRecord, status_code=status.HTTP_201_CREATED)
async def create_memory(
    session: DbSession,
    user: CurrentUser,
    payload: CreateMemoryPayload,
) -> MemoryRecord:
    """
    Create a new memory from explicit user input.
    Enforces user ownership and workspace access.
    """
    if payload.workspace_id is not None:
        conv_service = ConversationService(session)
        await conv_service._validate_workspace_access(user, payload.workspace_id)

    candidate = MemoryCandidate(
        user_id=user.id,
        workspace_id=payload.workspace_id,
        memory_type=payload.memory_type,
        content=payload.content.strip(),
        canonical_key=payload.canonical_key,
        confidence=payload.confidence,
        structured_data=payload.structured_data,
        source_type="EXPLICIT_USER_STATEMENT",
    )

    created = await MemoryMaterializer.materialize_memory(session, candidate)
    return created


@router.patch("/{memory_id}", response_model=MemoryRecord)
async def edit_memory(
    memory_id: UUID,
    payload: UpdateMemoryPayload,
    session: DbSession,
    user: CurrentUser,
) -> MemoryRecord:
    """
    Edit a selected active memory owned by the authenticated user.
    """
    updated = await MemoryMaterializer.update_memory(
        session=session,
        memory_id=memory_id,
        user_id=user.id,
        content=payload.content,
        structured_data=payload.structured_data,
        confidence=payload.confidence,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or unauthorized.",
        )
    return updated


@router.delete("/{memory_id}", status_code=status.HTTP_200_OK)
async def archive_memory(
    memory_id: UUID,
    session: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """
    Archive a selected memory owned by the authenticated user through the existing lifecycle.
    """
    success = await MemoryMaterializer.archive_memory(
        session=session,
        memory_id=memory_id,
        user_id=user.id,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or unauthorized.",
        )
    return {"status": "ARCHIVED", "memory_id": str(memory_id)}


@router.post("/search", response_model=MemoryRetrievalResponse)
async def search_memories(
    payload: SearchMemoryPayload,
    session: DbSession,
    user: CurrentUser,
) -> MemoryRetrievalResponse:
    """
    Retrieve relevant memories within the authorized scope.
    """
    if payload.workspace_id is not None:
        conv_service = ConversationService(session)
        await conv_service._validate_workspace_access(user, payload.workspace_id)

    response = await MemoryRetrievalService.retrieve(
        session=session,
        user_id=user.id,
        workspace_id=payload.workspace_id,
        query=payload.query,
        limit_relevant=payload.limit,
        similarity_threshold=payload.similarity_threshold,
    )
    return response
