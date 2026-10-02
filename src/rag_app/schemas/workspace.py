import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Workspace display name")
    code: str = Field(..., min_length=1, max_length=100, description="Workspace code identifier")
    department_id: uuid.UUID = Field(..., description="Department this workspace belongs to")
    description: Optional[str] = Field(None, description="Optional description of the workspace")


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class WorkspaceResponse(BaseModel):
    id: uuid.UUID
    name: str
    code: str
    description: Optional[str] = None
    department_id: uuid.UUID
    department_name: Optional[str] = None
    department_code: Optional[str] = None
    is_active: bool
    created_by: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class WorkspaceMemberCreate(BaseModel):
    user_id: uuid.UUID = Field(..., description="User ID to add as workspace member")
    role: str = Field("VIEWER", pattern="^(OWNER|EDITOR|VIEWER)$", description="Workspace role")


class WorkspaceMemberUpdate(BaseModel):
    role: str = Field(..., pattern="^(OWNER|EDITOR|VIEWER)$", description="Workspace role")


class WorkspaceMemberResponse(BaseModel):
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    user_email: Optional[str] = None
    role: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Workspace Provider Profile schemas
# ---------------------------------------------------------------------------

class WorkspaceProviderProfileUpdate(BaseModel):
    """
    Request body for PUT /workspaces/{id}/provider-profile.

    All fields are optional — only set the ones you want to override.
    Set any field to null/None to revert that capability to the platform default.

    IMPORTANT: Never send actual API keys here.  Use credential_ref as a
    human-readable label that maps to an environment variable or secret store
    entry managed outside the database.
    """
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    is_active: Optional[bool] = None

    llm_provider: Optional[str] = Field(None, max_length=100)
    llm_model: Optional[str] = Field(None, max_length=255)
    llm_credential_ref: Optional[str] = Field(None, max_length=255)

    embedding_provider: Optional[str] = Field(None, max_length=100)
    embedding_model: Optional[str] = Field(None, max_length=255)

    document_reader_provider: Optional[str] = Field(None, max_length=100)

    reranker_provider: Optional[str] = Field(None, max_length=100)
    reranker_model: Optional[str] = Field(None, max_length=255)

    ocr_provider: Optional[str] = Field(None, max_length=100)

    web_provider: Optional[str] = Field(None, max_length=100)
    web_credential_ref: Optional[str] = Field(None, max_length=255)

    configuration: Optional[Dict[str, Any]] = None


class WorkspaceProviderProfileResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    is_active: bool

    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_credential_ref: Optional[str] = None

    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None

    document_reader_provider: Optional[str] = None

    reranker_provider: Optional[str] = None
    reranker_model: Optional[str] = None

    ocr_provider: Optional[str] = None

    web_provider: Optional[str] = None
    web_credential_ref: Optional[str] = None

    configuration: Dict[str, Any] = {}

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProviderCapabilitySummary(BaseModel):
    """
    Describes an individual registered provider for a specific capability.
    Returned by the GET /providers/capabilities endpoint.
    """
    capability: str
    provider_name: str


class ProviderRegistrySummary(BaseModel):
    """
    Summary of all registered providers in the platform registry.
    Used by the frontend to populate provider selection dropdowns.
    """
    capabilities: Dict[str, list[str]]
