"""Admin User Registration Approval Workflow

Endpoints for administrators to list pending self-registrations, approve users
with department assignments and permission presets, or decline registrations.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import require_admin
from rag_app.db.models import Department, Permission, User, UserDepartment, UserPermission
from rag_app.db.session import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/registrations", tags=["admin-registrations"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
AdminUser = Annotated[User, Depends(require_admin)]


class ApproveRegistrationRequest(BaseModel):
    department_codes: list[str] = Field(default_factory=list)
    role: str = "USER"
    assigned_permissions: list[str] = Field(default_factory=lambda: ["VIEW"])


class RejectRegistrationRequest(BaseModel):
    reason: str | None = None


class PendingRegistrationResponse(BaseModel):
    id: str
    email: str
    role: str
    registration_status: str
    created_at: str
    profile_json: dict[str, Any] | None = None


@router.get("/pending", response_model=list[PendingRegistrationResponse])
async def list_pending_registrations(
    session: DbSession,
    _: AdminUser,
) -> list[PendingRegistrationResponse]:
    """Lists all user accounts awaiting administrator approval."""
    stmt = (
        select(User)
        .where(User.registration_status == "PENDING")
        .order_by(User.created_at.asc())
    )
    users = (await session.execute(stmt)).scalars().all()
    return [
        PendingRegistrationResponse(
            id=str(u.id),
            email=u.email,
            role=u.role,
            registration_status=u.registration_status,
            created_at=u.created_at.isoformat() if u.created_at else "",
            profile_json=u.profile_json or {},
        )
        for u in users
    ]


@router.post("/{user_id}/approve", status_code=status.HTTP_200_OK)
async def approve_registration(
    user_id: UUID,
    payload: ApproveRegistrationRequest,
    session: DbSession,
    admin: AdminUser,
) -> dict[str, Any]:
    """
    Approves a pending user registration, activates the account, and
    assigns requested or specified department access.
    """
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.registration_status = "APPROVED"
    user.is_active = True
    user.role = payload.role

    # Assign departments and default permissions
    assigned_depts = []
    for dept_code in payload.department_codes:
        dept = await session.scalar(select(Department).where(Department.code == dept_code.upper().strip()))
        if dept:
            ud = UserDepartment(user_id=user.id, department_id=dept.id)
            session.add(ud)
            assigned_depts.append(dept.code)

            # Assign default VIEW permission
            for perm_code in payload.assigned_permissions:
                perm = await session.scalar(select(Permission).where(Permission.code == perm_code.upper().strip()))
                if perm:
                    up = UserPermission(user_id=user.id, department_id=dept.id, permission_id=perm.id)
                    session.add(up)

    await session.commit()
    logger.info(
        "user_registration_approved",
        extra={
            "event": "user_registration_approved",
            "user_id": str(user.id),
            "approved_by": str(admin.id),
            "assigned_departments": assigned_depts,
        },
    )

    return {
        "status": "APPROVED",
        "user_id": str(user.id),
        "email": user.email,
        "is_active": user.is_active,
        "assigned_departments": assigned_depts,
    }


@router.post("/{user_id}/reject", status_code=status.HTTP_200_OK)
async def reject_registration(
    user_id: UUID,
    payload: RejectRegistrationRequest,
    session: DbSession,
    admin: AdminUser,
) -> dict[str, Any]:
    """Rejects a pending user registration, leaving the account inactive."""
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.registration_status = "REJECTED"
    user.is_active = False

    profile = dict(user.profile_json or {})
    if payload.reason:
        profile["rejection_reason"] = payload.reason
    user.profile_json = profile

    await session.commit()
    logger.info(
        "user_registration_rejected",
        extra={
            "event": "user_registration_rejected",
            "user_id": str(user.id),
            "rejected_by": str(admin.id),
            "reason": payload.reason,
        },
    )

    return {
        "status": "REJECTED",
        "user_id": str(user.id),
        "email": user.email,
        "is_active": False,
        "rejection_reason": payload.reason,
    }

