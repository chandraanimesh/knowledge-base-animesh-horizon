"""
Identity domain tools for FastMCP.
Exposes authenticated caller profile and active department/permission access.
Caller identity is strictly resolved from request context.
"""

from __future__ import annotations

from typing import Any, Optional
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import Department, Permission, User, UserDepartment, UserPermission
from rag_app.mcp.context import MCPRequestContext
from rag_app.mcp.gateway import mcp_gateway


async def handle_get_current_user_profile(
    user: User,
    session: AsyncSession,
) -> dict[str, Any]:
    """Return caller's user profile."""
    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if hasattr(user, "created_at") and user.created_at else None,
    }


async def handle_get_current_user_access(
    user: User,
    session: AsyncSession,
) -> dict[str, list[str]]:
    """Return dictionary of department codes -> list of active permission codes."""
    if user.role == "ADMIN":
        stmt = select(Department.code).where(Department.is_active == True)
        result = await session.execute(stmt)
        dept_codes = result.scalars().all()
        return {code: ["ALL"] for code in dept_codes}

    stmt = (
        select(Department.code, Permission.code)
        .join(UserDepartment, UserDepartment.department_id == Department.id)
        .join(
            UserPermission,
            and_(
                UserPermission.user_id == user.id,
                UserPermission.department_id == Department.id,
            ),
        )
        .join(Permission, Permission.id == UserPermission.permission_id)
        .where(
            UserDepartment.user_id == user.id,
            Department.is_active == True,
            Permission.is_active == True,
        )
    )
    result = await session.execute(stmt)
    access: dict[str, list[str]] = {}
    for dept_code, perm_code in result.all():
        if dept_code not in access:
            access[dept_code] = []
        access[dept_code].append(perm_code)

    return access


# MCP Tool Callers (Invoked via Gateway)
async def get_current_user_profile(auth_token: Optional[str] = None) -> dict[str, Any]:
    """Get the profile of the current authenticated caller."""
    return await mcp_gateway.execute(
        "get_current_user_profile",
        handle_get_current_user_profile,
        auth_token=auth_token,
    )


async def get_current_user_access(auth_token: Optional[str] = None) -> dict[str, list[str]]:
    """Inspect active department memberships and permissions for the current authenticated caller."""
    return await mcp_gateway.execute(
        "get_current_user_access",
        handle_get_current_user_access,
        auth_token=auth_token,
    )
