from typing import Annotated, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.session import get_db_session
from rag_app.db.models import Department, User
from rag_app.api.dependencies.auth import require_admin
from rag_app.schemas.users import (
    DepartmentCreate,
    DepartmentResponse,
    DepartmentStatusUpdate,
)

router = APIRouter(prefix="/admin/departments", tags=["admin-departments"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


@router.post("", response_model=DepartmentResponse, status_code=status.HTTP_201_CREATED)
async def create_department(
    payload: DepartmentCreate,
    session: DbSession,
    _: User = Depends(require_admin),
):
    code = payload.code.strip().upper()
    name = payload.name.strip()

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Department code must not be empty.",
        )
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Department name must not be empty.",
        )

    # Check for existing department with same code (active or inactive)
    existing = await session.scalar(
        select(Department).where(Department.code == code)
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Department with code '{code}' already exists.",
        )

    dept = Department(
        code=code,
        name=name,
        is_active=True,
    )
    session.add(dept)
    try:
        await session.commit()
        await session.refresh(dept)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Department with code '{code}' already exists.",
        )

    return dept


@router.get("", response_model=List[DepartmentResponse])
async def list_departments(
    session: DbSession,
    _: User = Depends(require_admin),
):
    """Return ALL departments, including active and inactive."""
    result = await session.scalars(
        select(Department).order_by(Department.code.asc())
    )
    return result.all()


@router.patch("/{code}/status", response_model=DepartmentResponse)
async def update_department_status(
    code: str,
    payload: DepartmentStatusUpdate,
    session: DbSession,
    _: User = Depends(require_admin),
):
    normalized_code = code.strip().upper()
    dept = await session.scalar(
        select(Department).where(Department.code == normalized_code)
    )
    if not dept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department '{normalized_code}' not found.",
        )

    dept.is_active = payload.is_active
    if payload.is_active:
        dept.deleted_at = None
    else:
        dept.deleted_at = datetime.now(timezone.utc)

    await session.commit()
    await session.refresh(dept)
    return dept
