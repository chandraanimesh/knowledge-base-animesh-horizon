from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Literal, Optional
from enum import Enum

class PermissionPreset(str, Enum):
    VIEWER = "VIEWER"
    EDITOR = "EDITOR"

PRESET_PERMISSIONS = {
    PermissionPreset.VIEWER: ["VIEW"],
    PermissionPreset.EDITOR: ["VIEW", "UPLOAD", "EDIT"]
}

class PresetRequest(BaseModel):
    preset: PermissionPreset

class UserCreate(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password")
    role: Literal["USER", "ADMIN"] = Field(default="USER", description="Role for the new user")
    department_code: Optional[str] = Field(default=None, description="Optional initial department code for USER")
    permissions: Optional[list[Literal["VIEW", "UPLOAD", "EDIT", "DELETE"]]] = Field(
        default=None, description="Optional initial permissions for the initial department"
    )

class UserStatusUpdate(BaseModel):
    is_active: bool = Field(..., description="Whether the user is active")

class UserResponse(BaseModel):
    id: UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class DepartmentResponse(BaseModel):
    id: UUID
    code: str
    name: str
    is_active: bool
    created_at: datetime
    deleted_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class PermissionResponse(BaseModel):
    id: UUID
    code: str
    description: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class DeptAccessSummary(BaseModel):
    code: str
    name: str
    permissions: list[str]


class UserAccessSummary(BaseModel):
    id: UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime
    departments: list[DeptAccessSummary] = []

    class Config:
        from_attributes = True


# ── Department Management ────────────────────────────────────────────────────

class DepartmentCreate(BaseModel):
    code: str = Field(..., min_length=1, max_length=32, description="Uppercase department code e.g. SALES")
    name: str = Field(..., min_length=1, max_length=128, description="Human-readable department name")


class DepartmentStatusUpdate(BaseModel):
    is_active: bool = Field(..., description="Whether the department is active")


# ── Access Requests ──────────────────────────────────────────────────────────

class AccessRequestCreate(BaseModel):
    """USER submits this. Grants ZERO authorization until APPROVED."""
    department_code: str = Field(..., description="Department code to request access for")
    requested_permissions: list[Literal["VIEW", "UPLOAD", "EDIT", "DELETE"]] = Field(
        ..., min_length=1, description="Permissions the user is requesting"
    )
    reason: Optional[str] = Field(None, max_length=512, description="Why access is needed")


class AccessRequestApprove(BaseModel):
    """ADMIN approval payload with optional adjusted permissions and review note."""
    approved_permissions: Optional[list[Literal["VIEW", "UPLOAD", "EDIT", "DELETE"]]] = Field(
        None, description="Permissions to grant (defaults to requested permissions if omitted)"
    )
    review_note: Optional[str] = Field(None, max_length=512, description="Review note for approval")


class AccessRequestReject(BaseModel):
    """ADMIN rejection payload with optional review note."""
    review_note: Optional[str] = Field(None, max_length=512, description="Reason / note for rejection")


class AccessRequestResponse(BaseModel):
    id: UUID
    user_id: UUID
    user_email: Optional[str] = None
    department_id: UUID
    department_code: str
    department_name: str
    requested_permissions: list[str]
    status: str
    reason: Optional[str] = None
    requested_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[UUID] = None
    reviewer_email: Optional[str] = None
    review_note: Optional[str] = None

    class Config:
        from_attributes = True


class AccessRequestReview(BaseModel):
    """Legacy review schema if needed."""
    action: Literal["APPROVE", "REJECT"] = Field(..., description="Approve or reject the request")
    approved_permissions: Optional[list[Literal["VIEW", "UPLOAD", "EDIT", "DELETE"]]] = Field(
        None, description="Permissions to actually grant"
    )
    reason: Optional[str] = Field(None, max_length=512, description="Reason for rejection or note for approval")


# ── Document Scope Administration ───────────────────────────────────────────

class DocumentDepartmentUpdate(BaseModel):
    department_code: str = Field(..., description="Target active department code to assign document to")


class DocumentAdminResponse(BaseModel):
    id: UUID
    file_name: str
    department_id: Optional[UUID] = None
    department_code: Optional[str] = None
    department_name: Optional[str] = None
    workspace_id: Optional[UUID] = None
    workspace_code: Optional[str] = None
    workspace_name: Optional[str] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    id: UUID
    file_name: str
    department_id: Optional[UUID] = None
    department_code: Optional[str] = None
    department_name: Optional[str] = None
    workspace_id: Optional[UUID] = None
    workspace_code: Optional[str] = None
    workspace_name: Optional[str] = None
    uploaded_by_id: Optional[UUID] = None
    uploaded_by_email: Optional[str] = None
    uploaded_by_name: Optional[str] = None
    mime_type: Optional[str] = None
    status: str
    created_at: datetime
    preview_url: Optional[str] = None

    class Config:
        from_attributes = True



# ── Document Permissions ───────────────────────────────────────────────────

class DocPermissionGrantRequest(BaseModel):
    user_id: UUID
    permission_code: str = Field(..., description="VIEW, UPLOAD, EDIT, or DELETE")


class DocPermissionRevokeRequest(BaseModel):
    user_id: UUID
    permission_code: str = Field(..., description="VIEW, UPLOAD, EDIT, or DELETE")


class DocPermissionSetRequest(BaseModel):
    user_id: UUID
    permissions: list[str] = Field(default_factory=list, description="List of permission codes to grant for this document")


class DocUserAccessItem(BaseModel):
    user_id: UUID
    email: str
    full_name: Optional[str] = None
    department_permissions: list[str] = Field(default_factory=list)
    document_permissions: list[str] = Field(default_factory=list)
    effective_permissions: list[str] = Field(default_factory=list)


class DocumentAccessListResponse(BaseModel):
    document_id: UUID
    file_name: str
    department_id: UUID
    department_code: str
    department_name: str
    users: list[DocUserAccessItem]


