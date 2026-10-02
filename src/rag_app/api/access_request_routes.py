from typing import Annotated, List, Optional, Literal
from uuid import UUID
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.session import get_db_session
from rag_app.db.models import (
    User,
    Department,
    Permission,
    UserDepartment,
    UserPermission,
    AccessRequest,
    AccessRequestPermission,
)
from rag_app.api.dependencies.auth import get_current_user, require_admin
from rag_app.schemas.users import (
    AccessRequestCreate,
    AccessRequestApprove,
    AccessRequestReject,
    AccessRequestResponse,
)

router = APIRouter(prefix="/access-requests", tags=["access-requests"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


async def _build_response(req: AccessRequest, session: AsyncSession) -> AccessRequestResponse:
    user = await session.get(User, req.user_id)
    dept = await session.get(Department, req.department_id)
    reviewer = await session.get(User, req.reviewed_by) if req.reviewed_by else None

    # Query requested permissions
    perms = await session.scalars(
        select(Permission.code)
        .join(AccessRequestPermission, Permission.id == AccessRequestPermission.permission_id)
        .where(AccessRequestPermission.access_request_id == req.id)
    )
    perm_codes = list(perms.all())

    return AccessRequestResponse(
        id=req.id,
        user_id=req.user_id,
        user_email=user.email if user else "",
        department_id=req.department_id,
        department_code=dept.code if dept else "",
        department_name=dept.name if dept else "",
        requested_permissions=perm_codes,
        status=req.status,
        reason=req.reason,
        requested_at=req.requested_at,
        reviewed_at=req.reviewed_at,
        reviewed_by=req.reviewed_by,
        reviewer_email=reviewer.email if reviewer else None,
        review_note=req.review_note,
    )


# ── USER: Submit Access Request ───────────────────────────────────────────────
@router.post("", response_model=AccessRequestResponse, status_code=status.HTTP_201_CREATED)
async def create_access_request(
    payload: AccessRequestCreate,
    session: DbSession,
    current_user: User = Depends(get_current_user),
):
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user cannot submit access requests.",
        )

    dept_code = payload.department_code.strip().upper()
    dept = await session.scalar(
        select(Department).where(Department.code == dept_code)
    )
    if not dept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department '{dept_code}' not found.",
        )
    if not dept.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Department '{dept_code}' is inactive.",
        )

    requested_codes = list(dict.fromkeys(payload.requested_permissions))
    if not requested_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one requested permission is required.",
        )

    # Verify all requested permissions exist and are active
    perms = await session.scalars(
        select(Permission).where(
            Permission.code.in_(requested_codes),
            Permission.is_active == True,
        )
    )
    perm_map = {p.code: p for p in perms.all()}
    missing_perms = set(requested_codes) - set(perm_map.keys())
    if missing_perms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested permissions {missing_perms} are invalid or inactive.",
        )

    # Check if user already fully possesses all requested permissions for this department
    existing_user_perms = await session.scalars(
        select(Permission.code)
        .join(UserPermission, Permission.id == UserPermission.permission_id)
        .where(
            UserPermission.user_id == current_user.id,
            UserPermission.department_id == dept.id,
        )
    )
    already_possessed = set(existing_user_perms.all())
    if set(requested_codes).issubset(already_possessed):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already possesses all requested permissions for this department.",
        )

    # Check if a pending request already exists for (user_id, department_id)
    existing_pending = await session.scalar(
        select(AccessRequest).where(
            AccessRequest.user_id == current_user.id,
            AccessRequest.department_id == dept.id,
            AccessRequest.status == "PENDING",
        )
    )
    if existing_pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A pending access request already exists for department '{dept_code}'.",
        )

    new_req = AccessRequest(
        user_id=current_user.id,
        department_id=dept.id,
        status="PENDING",
        reason=payload.reason.strip() if payload.reason else None,
    )
    session.add(new_req)
    await session.flush()

    for code in requested_codes:
        session.add(
            AccessRequestPermission(
                access_request_id=new_req.id,
                permission_id=perm_map[code].id,
            )
        )

    try:
        await session.commit()
        await session.refresh(new_req)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A pending access request already exists for department '{dept_code}'.",
        )

    return await _build_response(new_req, session)


# ── USER: View Own Requests ───────────────────────────────────────────────────
@router.get("/my", response_model=List[AccessRequestResponse])
async def list_my_access_requests(
    session: DbSession,
    current_user: User = Depends(get_current_user),
):
    stmt = (
        select(AccessRequest)
        .where(AccessRequest.user_id == current_user.id)
        .order_by(AccessRequest.requested_at.desc())
    )
    requests = (await session.scalars(stmt)).all()
    return [await _build_response(r, session) for r in requests]


# ── ADMIN: List All Requests ──────────────────────────────────────────────────
@router.get("", response_model=List[AccessRequestResponse])
async def list_all_access_requests(
    session: DbSession,
    status_filter: Optional[Literal["PENDING", "APPROVED", "REJECTED", "CANCELLED"]] = None,
    _: User = Depends(require_admin),
):
    stmt = select(AccessRequest).order_by(AccessRequest.requested_at.desc())
    if status_filter:
        stmt = stmt.where(AccessRequest.status == status_filter)

    requests = (await session.scalars(stmt)).all()
    return [await _build_response(r, session) for r in requests]


# ── ADMIN: Approve Request ────────────────────────────────────────────────────
@router.post("/{id}/approve", response_model=AccessRequestResponse)
async def approve_access_request(
    id: UUID,
    payload: AccessRequestApprove,
    session: DbSession,
    current_admin: User = Depends(require_admin),
):
    req = await session.get(AccessRequest, id)
    if not req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access request not found.",
        )

    if req.status != "PENDING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot approve a request with status '{req.status}'.",
        )

    requester = await session.get(User, req.user_id)
    if not requester or not requester.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Requester is inactive or no longer exists.",
        )

    dept = await session.get(Department, req.department_id)
    if not dept or not dept.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Department is inactive or no longer exists.",
        )

    # Determine approved permission codes
    if payload.approved_permissions is not None:
        approved_codes = list(dict.fromkeys(payload.approved_permissions))
    else:
        # Load from requested permissions
        child_perms = await session.scalars(
            select(Permission.code)
            .join(AccessRequestPermission, Permission.id == AccessRequestPermission.permission_id)
            .where(AccessRequestPermission.access_request_id == req.id)
        )
        approved_codes = list(child_perms.all())

    if not approved_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one approved permission is required.",
        )

    # Verify all approved permissions exist and are active
    perms = await session.scalars(
        select(Permission).where(
            Permission.code.in_(approved_codes),
            Permission.is_active == True,
        )
    )
    perm_map = {p.code: p for p in perms.all()}
    missing = set(approved_codes) - set(perm_map.keys())
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Approved permissions {missing} are invalid or inactive.",
        )

    # 1. Ensure user_departments membership exists
    ud = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == req.user_id,
            UserDepartment.department_id == req.department_id,
        )
    )
    if not ud:
        session.add(
            UserDepartment(
                user_id=req.user_id,
                department_id=req.department_id,
            )
        )
        await session.flush()

    # 2. Grant approved user_permissions
    existing_user_perms = await session.scalars(
        select(UserPermission.permission_id).where(
            UserPermission.user_id == req.user_id,
            UserPermission.department_id == req.department_id,
        )
    )
    existing_pids = set(existing_user_perms.all())

    for code in approved_codes:
        perm_obj = perm_map[code]
        if perm_obj.id not in existing_pids:
            session.add(
                UserPermission(
                    user_id=req.user_id,
                    department_id=req.department_id,
                    permission_id=perm_obj.id,
                )
            )

    # 3. Mark request APPROVED
    req.status = "APPROVED"
    req.reviewed_by = current_admin.id
    req.reviewed_at = datetime.now(timezone.utc)
    if payload.review_note is not None:
        req.review_note = payload.review_note.strip() if payload.review_note else None

    await session.commit()
    await session.refresh(req)

    return await _build_response(req, session)


# ── ADMIN: Reject Request ────────────────────────────────────────────────────
@router.post("/{id}/reject", response_model=AccessRequestResponse)
async def reject_access_request(
    id: UUID,
    payload: AccessRequestReject,
    session: DbSession,
    current_admin: User = Depends(require_admin),
):
    req = await session.get(AccessRequest, id)
    if not req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access request not found.",
        )

    if req.status != "PENDING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot reject a request with status '{req.status}'.",
        )

    req.status = "REJECTED"
    req.reviewed_by = current_admin.id
    req.reviewed_at = datetime.now(timezone.utc)
    if payload.review_note is not None:
        req.review_note = payload.review_note.strip() if payload.review_note else None

    await session.commit()
    await session.refresh(req)

    return await _build_response(req, session)
