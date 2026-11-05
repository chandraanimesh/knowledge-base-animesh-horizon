"""LangGraph tool: record_document_note

Allows agents and authorized users to record verified corrections/errata notes to documents.
Never allows untrusted LLM-generated corrections to automatically become authoritative
without human/admin approval (defaults to PENDING approval).
"""

from __future__ import annotations

import uuid
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import User
from rag_app.knowledge.errata_service import ErrataService
from rag_app.schemas.document_notes import DocumentNoteCreate


class RecordDocumentNoteInput(BaseModel):
    document_id: str = Field(..., description="UUID of the document being corrected")
    correction_text: str = Field(..., description="The verified correction, updated policy, or errata text")
    evidence_reference: str | None = Field(None, description="Supporting evidence, citation, ticket URL, or source quote")


class RecordDocumentNoteTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "record_document_note"
    description: str = (
        "Records a verified errata or correction note on an existing company document. "
        "Use this tool when you discover outdated, conflicting, or corrected information. "
        "The correction will be submitted for verification and linked to future retrieval queries."
    )
    args_schema: Type[BaseModel] = RecordDocumentNoteInput

    session: AsyncSession
    user: User

    async def _arun(
        self,
        document_id: str,
        correction_text: str,
        evidence_reference: str | None = None,
    ) -> str:
        try:
            doc_uuid = uuid.UUID(str(document_id).strip())
        except (ValueError, TypeError):
            return f"Error: Invalid document UUID '{document_id}'."

        service = ErrataService(self.session)
        try:
            req = DocumentNoteCreate(
                correction_text=correction_text,
                evidence_reference=evidence_reference,
            )
            note = await service.create_note(
                user=self.user,
                document_id=doc_uuid,
                request=req,
                is_agent_call=True,
            )
            await self.session.commit()
            return (
                f"Successfully recorded document correction [Note ID: {note.id}]. "
                f"Approval status: {note.approval_status}. "
                f"Correction: \"{note.correction_text}\""
            )
        except Exception as exc:
            return f"Error recording document note: {exc}"

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use async _arun.")

