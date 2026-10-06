"""
Admin Permissions domain tools for FastMCP.
Exposes:
- admin_list_user_permissions
- admin_grant_user_permission
- admin_revoke_user_permission
- admin_apply_permission_preset

All tools require MCP scope 'admin.permissions' and administrative privilege.
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
from rag_app.schemas.users import PRESET_PERMISSIONS, PermissionPreset


# ── Underlying Handlers ───────────────────────────────────────────────────────

async def handle_admin_list_user_permissions(
    user_id: str,
    department_code: str,
    session: AsyncSession,
) -> list[dict[str, Any]]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    code = department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == code))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found.")

    is_member = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == uid,
            UserDepartment.department_id == dept.id,
        )
    )
    if not is_member:
        raise HTTPException(status_code=400, detail="User is not a member of this department.")

    stmt = (
        select(Permission)
        .join(UserPermission, Permission.id == UserPermission.permission_id)
        .where(
            UserPermission.user_id == uid,
            UserPermission.department_id == dept.id,
        )
    )
    result = await session.scalars(stmt)
    perms = result.all()
    return [
        {
            "id": str(p.id),
            "code": p.code,
            "description": p.description,
            "is_active": p.is_active,
        }
        for p in perms
    ]


async def handle_admin_grant_user_permission(
    user_id: str,
    department_code: str,
    permission_code: str,
    session: AsyncSession,
) -> dict[str, str]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    code = department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == code))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found.")
    if not dept.is_active:
        raise HTTPException(status_code=400, detail="Cannot assign permission in an inactive department.")

    pcode = permission_code.strip().upper()
    perm = await session.scalar(select(Permission).where(Permission.code == pcode))
    if not perm:
        raise HTTPException(status_code=404, detail="Permission not found.")
    if not perm.is_active:
        raise HTTPException(status_code=400, detail="Cannot assign an inactive permission.")

    is_member = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == uid,
            UserDepartment.department_id == dept.id,
        )
    )
    if not is_member:
        raise HTTPException(status_code=400, detail="User must belong to the department before permissions can be granted.")

    user_perm = UserPermission(user_id=uid, department_id=dept.id, permission_id=perm.id)
    session.add(user_perm)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Permission already granted for this department.")

    return {"status": "success", "message": f"Permission '{pcode}' granted successfully to user {user_id}."}


async def handle_admin_revoke_user_permission(
    user_id: str,
    department_code: str,
    permission_code: str,
    session: AsyncSession,
) -> dict[str, str]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    code = department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == code))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found.")

    pcode = permission_code.strip().upper()
    perm = await session.scalar(select(Permission).where(Permission.code == pcode))
    if not perm:
        raise HTTPException(status_code=404, detail="Permission not found.")

    stmt = delete(UserPermission).where(
        UserPermission.user_id == uid,
        UserPermission.department_id == dept.id,
        UserPermission.permission_id == perm.id,
    )
    result = await session.execute(stmt)
    await session.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Permission is not granted for this user in this department.")

    return {"status": "success", "message": f"Permission '{pcode}' revoked successfully from user {user_id}."}


async def handle_admin_apply_permission_preset(
    user_id: str,
    department_code: str,
    preset: PermissionPreset,
    session: AsyncSession,
) -> dict[str, str]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid user UUID: {user_id}")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    code = department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == code))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found.")
    if not dept.is_active:
        raise HTTPException(status_code=400, detail="Cannot assign permissions in an inactive department.")

    is_member = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == uid,
            UserDepartment.department_id == dept.id,
        )
    )
    if not is_member:
        raise HTTPException(status_code=400, detail="User must belong to the department before presets can be applied.")

    required_codes = PRESET_PERMISSIONS[preset]
    perms = await session.scalars(
        select(Permission).where(
            Permission.code.in_(required_codes),
            Permission.is_active == True,
        )
    )
    perm_map = {p.code: p.id for p in perms.all()}
    missing = set(required_codes) - set(perm_map.keys())
    if missing:
        raise HTTPException(status_code=500, detail=f"Required permissions {missing} for preset are missing or inactive.")

    existing_user_perms = await session.scalars(
        select(UserPermission.permission_id).where(
            UserPermission.user_id == uid,
            UserPermission.department_id == dept.id,
        )
    )
    existing_perm_ids = set(existing_user_perms.all())

    new_perms = []
    for pcode in required_codes:
        pid = perm_map[pcode]
        if pid not in existing_perm_ids:
            new_perms.append(UserPermission(user_id=uid, department_id=dept.id, permission_id=pid))

    if new_perms:
        session.add_all(new_perms)
        await session.commit()

    return {"status": "success", "message": f"Preset {preset} applied successfully"}


# ── MCP Tool Callers ─────────────────────────────────────────────────────────

async def admin_list_user_permissions(
    user_id: str,
    department_code: str,
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "admin_list_user_permissions",
        handle_admin_list_user_permissions,
        user_id=user_id,
        department_code=department_code,
        auth_token=auth_token,
    )


async def admin_grant_user_permission(
    user_id: str,
    department_code: str,
    permission_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "admin_grant_user_permission",
        handle_admin_grant_user_permission,
        user_id=user_id,
        department_code=department_code,
        permission_code=permission_code,
        auth_token=auth_token,
    )


async def admin_revoke_user_permission(
    user_id: str,
    department_code: str,
    permission_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "admin_revoke_user_permission",
        handle_admin_revoke_user_permission,
        user_id=user_id,
        department_code=department_code,
        permission_code=permission_code,
        auth_token=auth_token,
    )


async def admin_apply_permission_preset(
    user_id: str,
    department_code: str,
    preset: PermissionPreset,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "admin_apply_permission_preset",
        handle_admin_apply_permission_preset,
        user_id=user_id,
        department_code=department_code,
        preset=preset,
        auth_token=auth_token,
    )
