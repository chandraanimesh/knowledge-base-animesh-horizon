import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import User
from rag_app.db.session import get_db_session
from rag_app.schemas.tasks import (
    TaskCreateRequest,
    TaskListResponse,
    TaskResponse,
    TaskUpdateRequest,
)
from rag_app.tasks.service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


@router.post(
    "",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an actionable structured task, commitment, or follow-up",
)
async def create_task(
    request: TaskCreateRequest,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> TaskResponse:
    service = TaskService(session)
    task = await service.create_task(user=user, request=request)
    return service.to_response(task)


@router.get(
    "",
    response_model=TaskListResponse,
    summary="Query structured tasks (DUE_TODAY, OVERDUE, UPCOMING, COMPANY, ALL) with timezone resolution",
)
async def query_tasks(
    session: DbSession,
    user: User = Depends(get_current_user),
    filter_type: str = Query(default="ALL", description="ALL, DUE_TODAY, OVERDUE, UPCOMING, or COMPANY"),
    company_name: Optional[str] = Query(default=None, description="Company name filter"),
    status: Optional[str] = Query(default=None, description="Status filter: OPEN, COMPLETED, or CANCELLED"),
    workspace_id: Optional[uuid.UUID] = Query(default=None, description="Optional workspace scope"),
    user_timezone: Optional[str] = Query(default=None, description="User IANA timezone, e.g. Asia/Kolkata"),
    x_user_timezone: Optional[str] = Header(default=None, alias="X-User-Timezone"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> TaskListResponse:
    service = TaskService(session)
    effective_tz = user_timezone or x_user_timezone or "UTC"
    tasks, total = await service.query_tasks(
        user=user,
        filter_type=filter_type,
        company_name=company_name,
        status_filter=status,
        workspace_id=workspace_id,
        user_timezone=effective_tz,
        limit=limit,
        offset=offset,
    )
    return TaskListResponse(
        total=total,
        tasks=[service.to_response(t) for t in tasks],
        filter_type=filter_type.upper(),
        user_timezone=effective_tz,
    )


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Get single task by ID with authorization verification",
)
async def get_task(
    task_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> TaskResponse:
    service = TaskService(session)
    task = await service.get_task(user=user, task_id=task_id)
    return service.to_response(task)


@router.patch(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Update task details or status",
)
async def update_task(
    task_id: uuid.UUID,
    request: TaskUpdateRequest,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> TaskResponse:
    service = TaskService(session)
    task = await service.update_task(user=user, task_id=task_id, request=request)
    return service.to_response(task)


@router.post(
    "/{task_id}/complete",
    response_model=TaskResponse,
    summary="Complete a task (sets status=COMPLETED and records completed_at timestamp)",
)
async def complete_task(
    task_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> TaskResponse:
    service = TaskService(session)
    task = await service.complete_task(user=user, task_id=task_id)
    return service.to_response(task)


@router.post(
    "/{task_id}/cancel",
    response_model=TaskResponse,
    summary="Cancel a task (sets status=CANCELLED)",
)
async def cancel_task(
    task_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> TaskResponse:
    service = TaskService(session)
    task = await service.cancel_task(user=user, task_id=task_id)
    return service.to_response(task)


@router.post(
    "/{task_id}/reopen",
    response_model=TaskResponse,
    summary="Reopen a completed or cancelled task back to OPEN",
)
async def reopen_task(
    task_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> TaskResponse:
    service = TaskService(session)
    task = await service.reopen_task(user=user, task_id=task_id)
    return service.to_response(task)
