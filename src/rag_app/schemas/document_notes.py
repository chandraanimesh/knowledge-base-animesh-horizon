from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class DocumentNoteCreate(BaseModel):
    correction_text: str = Field(..., min_length=3, description="The verified correction or errata text")
    evidence_reference: str | None = Field(None, description="Supporting evidence, citation, ticket URL, or reference")
    approval_status: str | None = Field("PENDING", description="Initial approval status (defaults to PENDING for safety)")


class DocumentNoteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    author_user_id: UUID | None
    correction_text: str
    evidence_reference: str | None
    approval_status: str
    is_active: bool
    effective_from: datetime
    effective_until: datetime | None
    created_at: datetime
    updated_at: datetime

