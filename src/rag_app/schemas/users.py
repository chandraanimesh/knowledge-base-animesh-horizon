from pydantic import BaseModel, ConfigDict, Field
from uuid import UUID
from datetime import datetime
from typing import Any, Literal, Optional
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

    model_config = ConfigDict(from_attributes=True)

class DepartmentResponse(BaseModel):
    id: UUID
    code: str
    name: str
    is_active: bool
    created_at: datetime
    deleted_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class PermissionResponse(BaseModel):
    id: UUID
    code: str
    description: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


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

class UserContext(BaseModel):
    id: UUID
    email: Optional[str] = None
    role: str = "USER"
    is_active: bool = True

    model_config = {"from_attributes": True}

    @classmethod
    def from_user(cls, user: Any) -> "UserContext":
        """
        Creates a detached, immutable UserContext snapshot from an ORM User, dict, or mock.

        Fail-closed security guarantees:
        - When called on an ORM User, required authorization attributes ('is_active' and 'role')
          must be already loaded in memory.
        - If attributes are expired or unloaded, this method fails closed with ValueError rather
          than triggering implicit database I/O (which causes MissingGreenlet in async contexts)
          or silently manufacturing permissive default authorization values (such as is_active=True).
        - Call sites must snapshot UserContext at route/request entry while the authenticated
          user is fully loaded, before database commits expire ORM session state.
        """
        if isinstance(user, cls):
            return user
        if user is None:
            raise ValueError("user cannot be None when creating UserContext")

        user_id = None
        user_email = None
        user_role = None
        user_is_active = None

        sa_state = getattr(user, "_sa_instance_state", None)
        if sa_state is not None and type(sa_state).__name__ == "InstanceState":
            is_expired = getattr(sa_state, "expired", False)
            unloaded = getattr(sa_state, "unloaded", set()) or set()
            loaded_dict = getattr(sa_state, "dict", {})

            # 1. Identity extraction without triggering lazy-load I/O
            if "id" in loaded_dict:
                user_id = loaded_dict["id"]
            elif sa_state.key and len(sa_state.key) > 1 and sa_state.key[1]:
                user_id = sa_state.key[1][0]
            elif "id" not in unloaded and not is_expired:
                user_id = getattr(user, "id", None)

            if user_id is None:
                raise ValueError("Cannot establish UserContext: user identity is unavailable or could not be determined")

            # 2. Authorization attributes extraction (fail-closed if expired or unloaded)
            missing_attrs: list[str] = []

            # Check is_active
            if "is_active" in loaded_dict:
                user_is_active = loaded_dict["is_active"]
            elif "is_active" not in unloaded and not is_expired:
                user_is_active = getattr(user, "is_active", None)

            if user_is_active is None:
                missing_attrs.append("is_active")

            # Check role
            if "role" in loaded_dict:
                user_role = loaded_dict["role"]
            elif "role" not in unloaded and not is_expired:
                user_role = getattr(user, "role", None)

            if user_role is None:
                missing_attrs.append("role")

            if missing_attrs:
                raise ValueError(
                    f"Cannot establish UserContext for user {user_id}: "
                    f"required authorization attribute(s) {missing_attrs} are expired or unloaded. "
                    "UserContext must be captured before session commit expires ORM state."
                )

            # 3. Non-authorization attributes (email)
            if "email" in loaded_dict:
                user_email = loaded_dict["email"]
            elif "email" not in unloaded and not is_expired:
                user_email = getattr(user, "email", None)
            else:
                user_email = None

        else:
            if isinstance(user, dict):
                user_id = user.get("id")
                user_email = user.get("email")
                user_role = user.get("role")
                user_is_active = user.get("is_active")
            else:
                user_id = getattr(user, "id", None)
                user_email = getattr(user, "email", None)
                user_role = getattr(user, "role", None)
                user_is_active = getattr(user, "is_active", None)

            if user_id is None:
                raise ValueError("Cannot establish UserContext: missing user id")
            if user_role is None:
                raise ValueError("Cannot establish UserContext: missing user role")
            if user_is_active is None:
                raise ValueError("Cannot establish UserContext: missing user is_active")

        if isinstance(user_id, str):
            user_id = UUID(user_id)

        return cls(
            id=user_id,
            email=user_email,
            role=str(user_role),
            is_active=bool(user_is_active),
        )


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

    model_config = ConfigDict(from_attributes=True)


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
    original_file_name: Optional[str] = None
    department_id: Optional[UUID] = None
    department_code: Optional[str] = None
    department_name: Optional[str] = None
    workspace_id: Optional[UUID] = None
    workspace_code: Optional[str] = None
    workspace_name: Optional[str] = None
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: UUID
    file_name: str
    original_file_name: Optional[str] = None
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

    model_config = ConfigDict(from_attributes=True)



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


