import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    OPEN = "OPEN"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class TaskProvenanceType(str, Enum):
    EXPLICIT_USER_ACTION = "EXPLICIT_USER_ACTION"
    EXTRACTED_FROM_INTERACTION = "EXTRACTED_FROM_INTERACTION"
    EXTRACTED_FROM_DOCUMENT = "EXTRACTED_FROM_DOCUMENT"


class TaskFilterType(str, Enum):
    ALL = "ALL"
    DUE_TODAY = "DUE_TODAY"
    OVERDUE = "OVERDUE"
    UPCOMING = "UPCOMING"
    COMPANY = "COMPANY"


class TaskCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Clear description of commitment or action")
    description: Optional[str] = Field(default=None, description="Detailed context or notes")
    due_date: Optional[datetime] = Field(default=None, description="Due date/time with timezone awareness")
    timezone: Optional[str] = Field(default="UTC", description="IANA Timezone (e.g. Asia/Kolkata, America/New_York)")
    company_name: Optional[str] = Field(default=None, max_length=255, description="Associated company if any")
    contact_name: Optional[str] = Field(default=None, max_length=255, description="Associated contact if any")
    workspace_id: Optional[uuid.UUID] = Field(default=None, description="Optional workspace scope")
    department_id: Optional[uuid.UUID] = Field(default=None, description="Optional department scope")
    provenance_type: Optional[str] = Field(default="EXPLICIT_USER_ACTION", description="Origin of task")
    conversation_id: Optional[uuid.UUID] = Field(default=None, description="Originating conversation")
    message_id: Optional[uuid.UUID] = Field(default=None, description="Originating message")
    document_chunk_id: Optional[uuid.UUID] = Field(default=None, description="Originating document chunk")
    attributes: Optional[dict] = Field(default_factory=dict, description="Custom attributes e.g. participants, meeting link")


class TaskUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    timezone: Optional[str] = None
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    status: Optional[str] = Field(default=None, description="OPEN, COMPLETED, or CANCELLED")
    attributes: Optional[dict] = None


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    workspace_id: Optional[uuid.UUID] = None
    department_id: Optional[uuid.UUID] = None
    title: str
    description: Optional[str] = None
    status: str
    due_date: Optional[datetime] = None
    timezone: str = "UTC"
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    provenance_type: str
    conversation_id: Optional[uuid.UUID] = None
    message_id: Optional[uuid.UUID] = None
    document_chunk_id: Optional[uuid.UUID] = None
    attributes: Optional[dict] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    is_overdue: bool = False


class TaskListResponse(BaseModel):
    total: int
    tasks: list[TaskResponse]
    filter_type: str = "ALL"
    user_timezone: str = "UTC"
