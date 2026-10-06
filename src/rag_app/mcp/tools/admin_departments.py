"""
Admin Department Management domain tools for FastMCP.
Exposes:
- admin_list_departments
- admin_create_department
- admin_update_department_status

All tools require MCP scope 'admin.departments' and administrative privilege.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import Department
from rag_app.mcp.gateway import mcp_gateway


# ── Underlying Handlers ───────────────────────────────────────────────────────

async def handle_admin_list_departments(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.scalars(select(Department).order_by(Department.code.asc()))
    depts = result.all()
    return [
        {
            "id": str(d.id),
            "code": d.code,
            "name": d.name,
            "is_active": d.is_active,
            "created_at": d.created_at.isoformat() if hasattr(d, "created_at") and d.created_at else None,
        }
        for d in depts
    ]


async def handle_admin_create_department(
    code: str,
    name: str,
    session: AsyncSession,
) -> dict[str, Any]:
    clean_code = code.strip().upper()
    clean_name = name.strip()

    if not clean_code:
        raise HTTPException(status_code=400, detail="Department code must not be empty.")
    if not clean_name:
        raise HTTPException(status_code=400, detail="Department name must not be empty.")

    existing = await session.scalar(select(Department).where(Department.code == clean_code))
    if existing:
        raise HTTPException(status_code=409, detail=f"Department with code '{clean_code}' already exists.")

    dept = Department(code=clean_code, name=clean_name, is_active=True)
    session.add(dept)
    try:
        await session.commit()
        await session.refresh(dept)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail=f"Department with code '{clean_code}' already exists.")

    return {
        "id": str(dept.id),
        "code": dept.code,
        "name": dept.name,
        "is_active": dept.is_active,
        "created_at": dept.created_at.isoformat() if hasattr(dept, "created_at") and dept.created_at else None,
    }


async def handle_admin_update_department_status(
    code: str,
    is_active: bool,
    session: AsyncSession,
) -> dict[str, Any]:
    clean_code = code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == clean_code))
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{clean_code}' not found.")

    dept.is_active = is_active
    if is_active:
        dept.deleted_at = None
    else:
        dept.deleted_at = datetime.now(timezone.utc)

    await session.commit()
    await session.refresh(dept)

    return {
        "id": str(dept.id),
        "code": dept.code,
        "name": dept.name,
        "is_active": dept.is_active,
    }


# ── MCP Tool Callers ─────────────────────────────────────────────────────────

async def admin_list_departments(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    return await mcp_gateway.execute(
        "admin_list_departments",
        handle_admin_list_departments,
        auth_token=auth_token,
    )


async def admin_create_department(
    code: str,
    name: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "admin_create_department",
        handle_admin_create_department,
        code=code,
        name=name,
        auth_token=auth_token,
    )


async def admin_update_department_status(
    code: str,
    is_active: bool,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "admin_update_department_status",
        handle_admin_update_department_status,
        code=code,
        is_active=is_active,
        auth_token=auth_token,
    )
