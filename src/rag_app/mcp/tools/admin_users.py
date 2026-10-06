"""
Admin User Management domain tools for FastMCP.
Exposes:
- admin_create_user
- admin_list_users
- admin_list_users_with_access
- admin_list_all_departments_lookup
- admin_get_user
- admin_update_user_status
- admin_list_user_departments
- admin_assign_user_department
- admin_remove_user_department

All tools require MCP scope 'admin.users' and administrative privilege.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional
from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import Department, Permission, User, UserDepartment, UserPermission
from rag_app.mcp.gateway import mcp_gateway
from rag_app.security.authentication import get_password_hash


# ── Underlying Handlers ───────────────────────────────────────────────────────

async def handle_admin_create_user(
    email: str,
    password: str,
    role: Literal["ADMIN", "USER"],
    session: AsyncSession,
    department_code: Optional[str] = None,
    permissions: Optional[list[str]] = None,
) -> dict[str, Any]:
    normalized_email = email.lower().strip()
    existing = await session.scalar(select(User).where(User.email == normalized_email))
    if existing:
        raise HTTPException(status_code=409, detail="User with this email already exists.")

    dept = None
    perms_to_grant = []
    if role == "USER" and department_code:
        dept_code = department_code.strip().upper()
        dept = await session.scalar(select(Department).where(Department.code == dept_code))
        if not dept:
            raise HTTPException(status_code=404, detail=f"Department '{dept_code}' not found.")
        if not dept.is_active:
            raise HTTPException(status_code=400, detail=f"Department '{dept_code}' is inactive.")

        if permissions:
            req_codes = list(dict.fromkeys(permissions))
            perm_records = (
                await session.scalars(
                    select(Permission).where(
                        Permission.code.in_(req_codes),
                        Permission.is_active == True,
                    )
                )
            ).all()
            if len(perm_records) != len(req_codes):
                found = {p.code for p in perm_records}
                missing = [c for c in req_codes if c not in found]
                raise HTTPException(status_code=400, detail=f"Invalid or inactive permissions: {missing}")
            perms_to_grant = perm_records

    new_user = User(
        email=normalized_email,
        hashed_password=get_password_hash(password),
        role=role,
        is_active=True,
    )
    session.add(new_user)
    try:
        await session.flush()
        if dept:
            session.add(UserDepartment(user_id=new_user.id, department_id=dept.id))
            for perm in perms_to_grant:
                session.add(
                    UserPermission(
                        user_id=new_user.id,
                        department_id=dept.id,
                        permission_id=perm.id,
                    )
                )
        await session.commit()
        await session.refresh(new_user)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="User with this email already exists.")

    return {
        "id": str(new_user.id),
        "email": new_user.email,
        "role": new_user.role,
        "is_active": new_user.is_active,
        "created_at": new_user.created_at.isoformat() if hasattr(new_user, "created_at") and new_user.created_at else None,
    }


async def handle_admin_list_users(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.scalars(select(User).order_by(User.email.asc()))
    users = result.all()
    return [
        {
            "id": str(u.id),
            "email": u.email,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if hasattr(u, "created_at") and u.created_at else None,
        }
        for u in users
    ]


async def handle_admin_list_users_with_access(session: AsyncSession) -> list[dict[str, Any]]:
    users = (await session.scalars(select(User).order_by(User.email.asc()))).all()
    user_ids = [u.id for u in users]
    if not user_ids:
        return []

    dept_rows = (
        await session.execute(
            select(UserDepartment.user_id, Department.code, Department.name)
            .join(Department, Department.id == UserDepartment.department_id)
            .where(UserDepartment.user_id.in_(user_ids), Department.is_active == True)
        )
    ).all()

    perm_rows = (
        await session.execute(
            select(UserPermission.user_id, Department.code, Permission.code)
            .join(Department, Department.id == UserPermission.department_id)
            .join(Permission, Permission.id == UserPermission.permission_id)
            .where(
                UserPermission.user_id.in_(user_ids),
                Department.is_active == True,
                Permission.is_active == True,
            )
        )
    ).all()

    user_dept_map: dict[uuid.UUID, dict[str, str]] = {}
    for uid, code, name in dept_rows:
        if uid not in user_dept_map:
            user_dept_map[uid] = {}
        user_dept_map[uid][code] = name

    user_perm_map: dict[uuid.UUID, dict[str, list[str]]] = {}
    for uid, dcode, pcode in perm_rows:
        if uid not in user_perm_map:
            user_perm_map[uid] = {}
        if dcode not in user_perm_map[uid]:
            user_perm_map[uid][dcode] = []
        user_perm_map[uid][dcode].append(pcode)

    summaries = []
    for u in users:
        dept_summaries = []
        if u.role == "ADMIN":
            dept_summaries.append(
                {"department_code": "*", "department_name": "All Departments (ADMIN)", "permissions": ["ALL"]}
            )
        else:
            u_depts = user_dept_map.get(u.id, {})
            u_perms = user_perm_map.get(u.id, {})
            for dcode, dname in u_depts.items():
                dept_summaries.append(
                    {
                        "department_code": dcode,
                        "department_name": dname,
                        "permissions": u_perms.get(dcode, []),
                    }
                )
        summaries.append(
            {
                "id": str(u.id),
                "email": u.email,
                "role": u.role,
                "is_active": u.is_active,
                "departments": dept_summaries,
            }
        )
    return summaries


async def handle_admin_list_all_departments_lookup(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.scalars(select(Department).order_by(Department.code.asc()))
    depts = result.all()
    return [
        {
            "id": str(d.id),
            "code": d.code,
            "name": d.name,
            "is_active": d.is_active,
        }
        for d in depts
    ]


async def handle_admin_get_user(user_id: str, session: AsyncSession) -> dict[str, Any]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if hasattr(user, "created_at") and user.created_at else None,
    }


async def handle_admin_update_user_status(
    user_id: str,
    is_active: bool,
    session: AsyncSession,
) -> dict[str, Any]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = is_active
    await session.commit()
    await session.refresh(user)

    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
    }


async def handle_admin_list_user_departments(user_id: str, session: AsyncSession) -> list[dict[str, Any]]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    result = await session.scalars(
        select(Department)
        .join(UserDepartment, UserDepartment.department_id == Department.id)
        .where(UserDepartment.user_id == uid)
        .order_by(Department.code.asc())
    )
    depts = result.all()
    return [
        {
            "id": str(d.id),
            "code": d.code,
            "name": d.name,
            "is_active": d.is_active,
        }
        for d in depts
    ]


async def handle_admin_assign_user_department(
    user_id: str,
    department_code: str,
    session: AsyncSession,
) -> dict[str, Any]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    code = department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == code))
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{code}' not found")
    if not dept.is_active:
        raise HTTPException(status_code=400, detail=f"Department '{code}' is inactive")

    existing = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == uid,
            UserDepartment.department_id == dept.id,
        )
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"User already assigned to department '{code}'")

    session.add(UserDepartment(user_id=uid, department_id=dept.id))
    await session.commit()

    return {
        "id": str(dept.id),
        "code": dept.code,
        "name": dept.name,
        "is_active": dept.is_active,
    }


async def handle_admin_remove_user_department(
    user_id: str,
    department_code: str,
    session: AsyncSession,
) -> dict[str, str]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    code = department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == code))
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{code}' not found")

    ud = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == uid,
            UserDepartment.department_id == dept.id,
        )
    )
    if not ud:
        raise HTTPException(status_code=404, detail=f"User is not assigned to department '{code}'")

    await session.delete(ud)
    await session.execute(
        delete(UserPermission).where(
            UserPermission.user_id == uid,
            UserPermission.department_id == dept.id,
        )
    )
    await session.commit()
    return {"status": "success", "message": f"User removed from department '{code}'"}


# ── MCP Tool Callers ─────────────────────────────────────────────────────────

async def admin_create_user(
    email: str,
    password: str,
    role: Literal["ADMIN", "USER"] = "USER",
    department_code: Optional[str] = None,
    permissions: Optional[list[str]] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "admin_create_user",
        handle_admin_create_user,
        email=email,
        password=password,
        role=role,
        department_code=department_code,
        permissions=permissions,
        auth_token=auth_token,
    )


async def admin_list_users(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "admin_list_users",
        handle_admin_list_users,
        auth_token=auth_token,
    )


async def admin_list_users_with_access(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "admin_list_users_with_access",
        handle_admin_list_users_with_access,
        auth_token=auth_token,
    )


async def admin_list_all_departments_lookup(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "admin_list_all_departments_lookup",
        handle_admin_list_all_departments_lookup,
        auth_token=auth_token,
    )


async def admin_get_user(
    user_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "admin_get_user",
        handle_admin_get_user,
        user_id=user_id,
        auth_token=auth_token,
    )


async def admin_update_user_status(
    user_id: str,
    is_active: bool,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "admin_update_user_status",
        handle_admin_update_user_status,
        user_id=user_id,
        is_active=is_active,
        auth_token=auth_token,
    )


async def admin_list_user_departments(
    user_id: str,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "admin_list_user_departments",
        handle_admin_list_user_departments,
        user_id=user_id,
        auth_token=auth_token,
    )


async def admin_assign_user_department(
    user_id: str,
    department_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "admin_assign_user_department",
        handle_admin_assign_user_department,
        user_id=user_id,
        department_code=department_code,
        auth_token=auth_token,
    )


async def admin_remove_user_department(
    user_id: str,
    department_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "admin_remove_user_department",
        handle_admin_remove_user_department,
        user_id=user_id,
        department_code=department_code,
        auth_token=auth_token,
    )
