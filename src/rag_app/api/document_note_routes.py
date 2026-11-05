import uuid
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import User
from rag_app.db.session import get_db_session
from rag_app.knowledge.errata_service import ErrataService
from rag_app.schemas.document_notes import DocumentNoteCreate, DocumentNoteResponse

router = APIRouter(prefix="/documents", tags=["document-notes"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post("/{document_id}/notes", response_model=DocumentNoteResponse, status_code=status.HTTP_201_CREATED)
async def create_document_note(
    document_id: uuid.UUID,
    request: DocumentNoteCreate,
    session: DbSession,
    user: CurrentUser,
) -> DocumentNoteResponse:
    """Create a verified correction/errata note for a document."""
    service = ErrataService(session)
    note = await service.create_note(user=user, document_id=document_id, request=request)
    await session.commit()
    return DocumentNoteResponse.model_validate(note)


@router.get("/{document_id}/notes", response_model=List[DocumentNoteResponse])
async def list_document_notes(
    document_id: uuid.UUID,
    session: DbSession,
    user: CurrentUser,
    include_pending: bool = False,
) -> List[DocumentNoteResponse]:
    """List errata notes for a document."""
    service = ErrataService(session)
    notes = await service.list_notes(user=user, document_id=document_id, include_pending=include_pending)
    return [DocumentNoteResponse.model_validate(n) for n in notes]


@router.patch("/notes/{note_id}/approve", response_model=DocumentNoteResponse)
async def approve_document_note(
    note_id: uuid.UUID,
    session: DbSession,
    user: CurrentUser,
) -> DocumentNoteResponse:
    """Admin approves a pending errata note."""
    service = ErrataService(session)
    note = await service.approve_note(user=user, note_id=note_id)
    await session.commit()
    return DocumentNoteResponse.model_validate(note)


@router.delete("/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_document_note(
    note_id: uuid.UUID,
    session: DbSession,
    user: CurrentUser,
) -> None:
    """Admin revokes an errata note."""
    service = ErrataService(session)
    await service.revoke_note(user=user, note_id=note_id)
    await session.commit()

