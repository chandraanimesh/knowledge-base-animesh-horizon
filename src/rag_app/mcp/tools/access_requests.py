"""
Access Requests domain tools for FastMCP.
Exposes:
- create_access_request
- list_my_access_requests
- list_all_access_requests
- approve_access_request
- reject_access_request

Caller and admin reviewer identity are strictly resolved from context.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import (
    AccessRequest,
    AccessRequestPermission,
    Department,
    Permission,
    User,
    UserDepartment,
    UserPermission,
)
from rag_app.mcp.gateway import mcp_gateway


async def _build_access_request_response(req: AccessRequest, session: AsyncSession) -> dict[str, Any]:
    user = await session.get(User, req.user_id)
    dept = await session.get(Department, req.department_id)
    reviewer = await session.get(User, req.reviewed_by) if req.reviewed_by else None

    perms = await session.scalars(
        select(Permission.code)
        .join(AccessRequestPermission, Permission.id == AccessRequestPermission.permission_id)
        .where(AccessRequestPermission.access_request_id == req.id)
    )
    perm_codes = list(perms.all())

    return {
        "id": str(req.id),
        "user_id": str(req.user_id),
        "user_email": user.email if user else "",
        "department_id": str(req.department_id),
        "department_code": dept.code if dept else "",
        "department_name": dept.name if dept else "",
        "requested_permissions": perm_codes,
        "status": req.status,
        "reason": req.reason,
        "requested_at": req.requested_at.isoformat() if req.requested_at else None,
        "reviewed_at": req.reviewed_at.isoformat() if req.reviewed_at else None,
        "reviewed_by": str(req.reviewed_by) if req.reviewed_by else None,
        "reviewer_email": reviewer.email if reviewer else None,
        "review_note": req.review_note,
    }


# ── Underlying Handlers ───────────────────────────────────────────────────────

async def handle_create_access_request(
    department_code: str,
    requested_permissions: list[str],
    session: AsyncSession,
    user: User,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Inactive user cannot submit access requests.")

    dept_code = department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == dept_code))
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{dept_code}' not found.")
    if not dept.is_active:
        raise HTTPException(status_code=400, detail=f"Department '{dept_code}' is inactive.")

    requested_codes = list(dict.fromkeys(requested_permissions))
    if not requested_codes:
        raise HTTPException(status_code=400, detail="At least one requested permission is required.")

    perms = await session.scalars(
        select(Permission).where(Permission.code.in_(requested_codes), Permission.is_active == True)
    )
    perm_map = {p.code: p for p in perms.all()}
    missing_perms = set(requested_codes) - set(perm_map.keys())
    if missing_perms:
        raise HTTPException(status_code=400, detail=f"Requested permissions {missing_perms} are invalid or inactive.")

    # Check if user already fully possesses permissions
    existing_user_perms = await session.scalars(
        select(Permission.code)
        .join(UserPermission, Permission.id == UserPermission.permission_id)
        .where(UserPermission.user_id == user.id, UserPermission.department_id == dept.id)
    )
    if set(requested_codes).issubset(set(existing_user_perms.all())):
        raise HTTPException(status_code=400, detail="User already possesses all requested permissions for this department.")

    existing_pending = await session.scalar(
        select(AccessRequest).where(
            AccessRequest.user_id == user.id,
            AccessRequest.department_id == dept.id,
            AccessRequest.status == "PENDING",
        )
    )
    if existing_pending:
        raise HTTPException(status_code=409, detail=f"A pending access request already exists for department '{dept_code}'.")

    new_req = AccessRequest(
        user_id=user.id,
        department_id=dept.id,
        status="PENDING",
        reason=reason.strip() if reason else None,
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
        raise HTTPException(status_code=409, detail=f"A pending access request already exists for department '{dept_code}'.")

    return await _build_access_request_response(new_req, session)


async def handle_list_my_access_requests(
    session: AsyncSession,
    user: User,
) -> list[dict[str, Any]]:
    stmt = (
        select(AccessRequest)
        .where(AccessRequest.user_id == user.id)
        .order_by(AccessRequest.requested_at.desc())
    )
    requests = (await session.scalars(stmt)).all()
    return [await _build_access_request_response(r, session) for r in requests]


async def handle_list_all_access_requests(
    session: AsyncSession,
    user: User,
    status_filter: Optional[str] = None,
) -> list[dict[str, Any]]:
    stmt = select(AccessRequest).order_by(AccessRequest.requested_at.desc())
    if status_filter:
        stmt = stmt.where(AccessRequest.status == status_filter.upper())
    requests = (await session.scalars(stmt)).all()
    return [await _build_access_request_response(r, session) for r in requests]


async def handle_approve_access_request(
    request_id: str,
    session: AsyncSession,
    user: User,
    approved_permissions: Optional[list[str]] = None,
    review_note: Optional[str] = None,
) -> dict[str, Any]:
    try:
        req_uuid = uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid access request UUID: {request_id}")

    req = await session.get(AccessRequest, req_uuid)
    if not req:
        raise HTTPException(status_code=404, detail="Access request not found.")
    if req.status != "PENDING":
        raise HTTPException(status_code=409, detail=f"Cannot approve a request with status '{req.status}'.")

    requester = await session.get(User, req.user_id)
    if not requester or not requester.is_active:
        raise HTTPException(status_code=400, detail="Requester is inactive or no longer exists.")

    dept = await session.get(Department, req.department_id)
    if not dept or not dept.is_active:
        raise HTTPException(status_code=400, detail="Department is inactive or no longer exists.")

    if approved_permissions is not None:
        approved_codes = list(dict.fromkeys(approved_permissions))
    else:
        child_perms = await session.scalars(
            select(Permission.code)
            .join(AccessRequestPermission, Permission.id == AccessRequestPermission.permission_id)
            .where(AccessRequestPermission.access_request_id == req.id)
        )
        approved_codes = list(child_perms.all())

    if not approved_codes:
        raise HTTPException(status_code=400, detail="At least one approved permission is required.")

    perms = await session.scalars(
        select(Permission).where(Permission.code.in_(approved_codes), Permission.is_active == True)
    )
    perm_map = {p.code: p for p in perms.all()}
    missing = set(approved_codes) - set(perm_map.keys())
    if missing:
        raise HTTPException(status_code=400, detail=f"Approved permissions {missing} are invalid or inactive.")

    # 1. Ensure user_department exists
    ud = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == req.user_id,
            UserDepartment.department_id == req.department_id,
        )
    )
    if not ud:
        session.add(UserDepartment(user_id=req.user_id, department_id=req.department_id))
        await session.flush()

    # 2. Grant permissions
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

    # 3. Mark approved
    req.status = "APPROVED"
    req.reviewed_by = user.id
    req.reviewed_at = datetime.now(timezone.utc)
    if review_note is not None:
        req.review_note = review_note.strip() if review_note else None

    await session.commit()
    await session.refresh(req)
    return await _build_access_request_response(req, session)


async def handle_reject_access_request(
    request_id: str,
    session: AsyncSession,
    user: User,
    review_note: Optional[str] = None,
) -> dict[str, Any]:
    try:
        req_uuid = uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid access request UUID: {request_id}")

    req = await session.get(AccessRequest, req_uuid)
    if not req:
        raise HTTPException(status_code=404, detail="Access request not found.")
    if req.status != "PENDING":
        raise HTTPException(status_code=409, detail=f"Cannot reject a request with status '{req.status}'.")

    req.status = "REJECTED"
    req.reviewed_by = user.id
    req.reviewed_at = datetime.now(timezone.utc)
    if review_note is not None:
        req.review_note = review_note.strip() if review_note else None

    await session.commit()
    await session.refresh(req)
    return await _build_access_request_response(req, session)


# ── MCP Tool Callers ─────────────────────────────────────────────────────────

async def create_access_request(
    department_code: str,
    requested_permissions: list[str],
    reason: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "create_access_request",
        handle_create_access_request,
        department_code=department_code,
        requested_permissions=requested_permissions,
        reason=reason,
        auth_token=auth_token,
    )


async def list_my_access_requests(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "list_my_access_requests",
        handle_list_my_access_requests,
        auth_token=auth_token,
    )


async def list_all_access_requests(
    status_filter: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "list_all_access_requests",
        handle_list_all_access_requests,
        status_filter=status_filter,
        auth_token=auth_token,
    )


async def approve_access_request(
    request_id: str,
    approved_permissions: Optional[list[str]] = None,
    review_note: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "approve_access_request",
        handle_approve_access_request,
        request_id=request_id,
        approved_permissions=approved_permissions,
        review_note=review_note,
        auth_token=auth_token,
    )


async def reject_access_request(
    request_id: str,
    review_note: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "reject_access_request",
        handle_reject_access_request,
        request_id=request_id,
        review_note=review_note,
        auth_token=auth_token,
    )
