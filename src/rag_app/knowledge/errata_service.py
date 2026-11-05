from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Sequence

from fastapi import HTTPException, status
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import Document, DocumentNote, User, WorkspaceMember
from rag_app.schemas.document_notes import DocumentNoteCreate
from rag_app.security.authorization import get_allowed_departments

logger = logging.getLogger(__name__)


class ErrataService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _validate_document_access(self, user: User, document_id: uuid.UUID) -> Document:
        doc = await self.session.get(Document, document_id)
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        # Admin bypasses workspace/dept check
        if user.role == "ADMIN":
            return doc

        # Workspace check
        if doc.workspace_id:
            member = await self.session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == doc.workspace_id,
                    WorkspaceMember.user_id == user.id,
                )
            )
            if not member:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this document's workspace")
            return doc

        # Department check
        if doc.department_id:
            allowed = await get_allowed_departments(self.session, user)
            if allowed is not None and str(doc.department_id) not in {str(d) for d in allowed}:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this document's department")

        return doc

    async def create_note(
        self,
        user: User,
        document_id: uuid.UUID,
        request: DocumentNoteCreate,
        is_agent_call: bool = False,
    ) -> DocumentNote:
        doc = await self._validate_document_access(user, document_id)

        # Safety rule: Agent-generated corrections or non-admin users default to PENDING approval
        initial_status = "APPROVED" if (user.role == "ADMIN" and not is_agent_call) else "PENDING"
        if request.approval_status and user.role == "ADMIN" and not is_agent_call:
            initial_status = request.approval_status

        note = DocumentNote(
            document_id=doc.id,
            author_user_id=user.id,
            correction_text=request.correction_text.strip(),
            evidence_reference=request.evidence_reference.strip() if request.evidence_reference else None,
            approval_status=initial_status,
            is_active=True,
        )
        self.session.add(note)
        await self.session.flush()

        logger.info(
            "document_note_created",
            extra={
                "event": "document_note_created",
                "note_id": str(note.id),
                "document_id": str(doc.id),
                "author_id": str(user.id),
                "approval_status": initial_status,
            },
        )
        return note

    async def list_notes(
        self,
        user: User,
        document_id: uuid.UUID,
        include_pending: bool = False,
    ) -> Sequence[DocumentNote]:
        doc = await self._validate_document_access(user, document_id)

        stmt = select(DocumentNote).where(DocumentNote.document_id == doc.id)
        if not include_pending:
            stmt = stmt.where(DocumentNote.approval_status == "APPROVED", DocumentNote.is_active == True)
        stmt = stmt.order_by(DocumentNote.created_at.desc())

        result = await self.session.scalars(stmt)
        return result.all()

    async def approve_note(self, user: User, note_id: uuid.UUID) -> DocumentNote:
        if user.role != "ADMIN":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can approve document errata notes")

        note = await self.session.get(DocumentNote, note_id)
        if not note:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document note not found")

        note.approval_status = "APPROVED"
        note.is_active = True
        await self.session.flush()
        return note

    async def revoke_note(self, user: User, note_id: uuid.UUID) -> DocumentNote:
        if user.role != "ADMIN":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can revoke document errata notes")

        note = await self.session.get(DocumentNote, note_id)
        if not note:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document note not found")

        note.is_active = False
        await self.session.flush()
        return note

    async def get_active_errata_for_documents(
        self,
        document_ids: Sequence[uuid.UUID],
    ) -> dict[uuid.UUID, list[DocumentNote]]:
        """
        Retrieves active, approved, unexpired errata notes for a batch of documents in a single IN query.
        Guarantees zero N+1 database queries.
        """
        if not document_ids:
            return {}

        now = datetime.now(timezone.utc)
        stmt = (
            select(DocumentNote)
            .where(
                DocumentNote.document_id.in_(document_ids),
                DocumentNote.is_active == True,
                DocumentNote.approval_status == "APPROVED",
                DocumentNote.effective_from <= now,
                or_(DocumentNote.effective_until.is_(None), DocumentNote.effective_until >= now),
            )
            .order_by(DocumentNote.created_at.desc())
        )

        notes = (await self.session.scalars(stmt)).all()
        mapping: dict[uuid.UUID, list[DocumentNote]] = {}
        for n in notes:
            mapping.setdefault(n.document_id, []).append(n)
        return mapping

