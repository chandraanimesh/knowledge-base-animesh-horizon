import logging
import uuid
import zoneinfo
from datetime import datetime, time, timezone
from typing import Optional, Sequence, Tuple

from fastapi import HTTPException, status
from sqlalchemy import func, select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import Department, TaskItem, User, Workspace, WorkspaceMember
from rag_app.schemas.tasks import (
    TaskCreateRequest,
    TaskFilterType,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
    TaskUpdateRequest,
)
from rag_app.security.authorization import authorize

logger = logging.getLogger(__name__)


def resolve_timezone(tz_name: Optional[str]) -> zoneinfo.ZoneInfo:
    """Safely resolves an IANA timezone string, falling back to UTC if invalid."""
    if not tz_name or not tz_name.strip():
        return zoneinfo.ZoneInfo("UTC")
    try:
        return zoneinfo.ZoneInfo(tz_name.strip())
    except Exception:
        logger.warning(f"Invalid timezone '{tz_name}', falling back to UTC")
        return zoneinfo.ZoneInfo("UTC")


def get_user_day_bounds(user_tz: zoneinfo.ZoneInfo) -> Tuple[datetime, datetime, datetime]:
    """
    Computes exact start and end of day in the user's local timezone
    and converts them to timezone-aware UTC datetimes.
    """
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(user_tz)
    local_day_start = datetime.combine(now_local.date(), time.min, tzinfo=user_tz)
    local_day_end = datetime.combine(now_local.date(), time.max, tzinfo=user_tz)

    start_utc = local_day_start.astimezone(timezone.utc)
    end_utc = local_day_end.astimezone(timezone.utc)
    return start_utc, end_utc, now_utc


class TaskService:
    """
    Service for managing structured actionable tasks, commitments, and follow-ups.
    Enforces user isolation, workspace/department authorization, and deterministic date filtering.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_task(self, user: User, request: TaskCreateRequest) -> TaskItem:
        """
        Creates an actionable follow-up task.
        Validates authorization for workspace and department if provided.
        Ensures due date carries proper timezone information.
        """
        title_clean = request.title.strip()
        if not title_clean:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task title must not be empty",
            )

        # Validate workspace authorization if scoped
        if request.workspace_id:
            ws = await self.session.get(Workspace, request.workspace_id)
            if not ws or not ws.is_active:
                raise HTTPException(status_code=404, detail="Target workspace not found or inactive")
            if user.role != "ADMIN":
                is_mem = await self.session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == request.workspace_id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not is_mem:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You are not a member of the target workspace",
                    )

        # Validate department authorization if scoped
        if request.department_id:
            dept = await self.session.get(Department, request.department_id)
            if not dept:
                raise HTTPException(status_code=404, detail="Target department not found")
            if user.role != "ADMIN":
                is_auth = await authorize(self.session, user, "VIEW", dept.code)
                if not is_auth:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Forbidden: You do not have access to department '{dept.code}'",
                    )

        # Normalize due date timezone
        tz_obj = resolve_timezone(request.timezone)
        normalized_due_date = None
        if request.due_date is not None:
            if request.due_date.tzinfo is None:
                normalized_due_date = request.due_date.replace(tzinfo=tz_obj).astimezone(timezone.utc)
            else:
                normalized_due_date = request.due_date.astimezone(timezone.utc)

        task = TaskItem(
            user_id=user.id,
            workspace_id=request.workspace_id,
            department_id=request.department_id,
            title=title_clean,
            description=request.description.strip() if request.description else None,
            status=TaskStatus.OPEN.value,
            due_date=normalized_due_date,
            timezone=str(tz_obj),
            company_name=request.company_name.strip() if request.company_name else None,
            contact_name=request.contact_name.strip() if request.contact_name else None,
            provenance_type=request.provenance_type or "EXPLICIT_USER_ACTION",
            conversation_id=request.conversation_id,
            message_id=request.message_id,
            document_chunk_id=request.document_chunk_id,
            attributes=request.attributes or {},
        )
        self.session.add(task)
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def get_task(self, user: User, task_id: uuid.UUID) -> TaskItem:
        """Fetches a task ensuring caller is the owner or authorized administrator."""
        task = await self.session.get(TaskItem, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if user.role != "ADMIN" and task.user_id != user.id:
            # Check if user is a member of the task's workspace
            if task.workspace_id:
                is_mem = await self.session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == task.workspace_id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not is_mem:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You are not authorized to view this task",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: You are not authorized to view this task",
                )
        return task

    async def update_task(self, user: User, task_id: uuid.UUID, request: TaskUpdateRequest) -> TaskItem:
        """Updates task fields with ownership verification and status transition handling."""
        task = await self.get_task(user, task_id)

        if user.role != "ADMIN" and task.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Only the task owner or an administrator can modify this task",
            )

        now = datetime.now(timezone.utc)
        if request.title is not None:
            clean_title = request.title.strip()
            if not clean_title:
                raise HTTPException(status_code=400, detail="Task title must not be empty")
            task.title = clean_title

        if request.description is not None:
            task.description = request.description.strip() if request.description else None

        if request.timezone is not None:
            task.timezone = str(resolve_timezone(request.timezone))

        if request.due_date is not None:
            tz_obj = resolve_timezone(task.timezone)
            if request.due_date.tzinfo is None:
                task.due_date = request.due_date.replace(tzinfo=tz_obj).astimezone(timezone.utc)
            else:
                task.due_date = request.due_date.astimezone(timezone.utc)

        if request.company_name is not None:
            task.company_name = request.company_name.strip() if request.company_name else None

        if request.contact_name is not None:
            task.contact_name = request.contact_name.strip() if request.contact_name else None

        if request.attributes is not None:
            existing = dict(task.attributes or {})
            existing.update(request.attributes)
            task.attributes = existing

        if request.status is not None:
            st = request.status.upper()
            if st not in (TaskStatus.OPEN.value, TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value):
                raise HTTPException(status_code=400, detail=f"Invalid task status '{request.status}'")
            task.status = st
            if st == TaskStatus.COMPLETED.value:
                task.completed_at = now
            elif st == TaskStatus.OPEN.value:
                task.completed_at = None

        task.updated_at = now
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def complete_task(self, user: User, task_id: uuid.UUID) -> TaskItem:
        """Idempotently transitions a task to COMPLETED."""
        return await self.update_task(user, task_id, TaskUpdateRequest(status=TaskStatus.COMPLETED.value))

    async def cancel_task(self, user: User, task_id: uuid.UUID) -> TaskItem:
        """Idempotently transitions a task to CANCELLED."""
        return await self.update_task(user, task_id, TaskUpdateRequest(status=TaskStatus.CANCELLED.value))

    async def reopen_task(self, user: User, task_id: uuid.UUID) -> TaskItem:
        """Reopens a completed or cancelled task back to OPEN."""
        return await self.update_task(user, task_id, TaskUpdateRequest(status=TaskStatus.OPEN.value))

    async def query_tasks(
        self,
        user: User,
        filter_type: str = "ALL",
        company_name: Optional[str] = None,
        status_filter: Optional[str] = None,
        workspace_id: Optional[uuid.UUID] = None,
        user_timezone: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[list[TaskItem], int]:
        """
        Queries tasks deterministically based on temporal criteria:
        - DUE_TODAY: resolves today using user's configured timezone.
        - OVERDUE: open tasks where due_date < now_utc.
        - UPCOMING: open tasks where due_date > now_utc.
        - COMPANY: open and historical tasks matching company_name.
        - ALL: all tasks matching ownership and optional status.
        """
        # Determine base ownership filter
        if user.role == "ADMIN" and workspace_id:
            base_filters = [TaskItem.workspace_id == workspace_id]
        elif workspace_id:
            # Must be member
            is_mem = await self.session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == user.id,
                )
            )
            if not is_mem:
                raise HTTPException(status_code=403, detail="Forbidden: Not a member of requested workspace")
            base_filters = [
                or_(
                    TaskItem.user_id == user.id,
                    TaskItem.workspace_id == workspace_id,
                )
            ]
        else:
            base_filters = [TaskItem.user_id == user.id]

        if status_filter:
            base_filters.append(TaskItem.status == status_filter.upper())

        ft = filter_type.upper()
        user_tz = resolve_timezone(user_timezone)
        start_utc, end_utc, now_utc = get_user_day_bounds(user_tz)

        if ft == TaskFilterType.DUE_TODAY.value:
            base_filters.append(TaskItem.status == TaskStatus.OPEN.value)
            base_filters.append(TaskItem.due_date.is_not(None))
            base_filters.append(TaskItem.due_date >= start_utc)
            base_filters.append(TaskItem.due_date <= end_utc)

        elif ft == TaskFilterType.OVERDUE.value:
            base_filters.append(TaskItem.status == TaskStatus.OPEN.value)
            base_filters.append(TaskItem.due_date.is_not(None))
            base_filters.append(TaskItem.due_date < now_utc)

        elif ft == TaskFilterType.UPCOMING.value:
            base_filters.append(TaskItem.status == TaskStatus.OPEN.value)
            base_filters.append(TaskItem.due_date.is_not(None))
            base_filters.append(TaskItem.due_date > now_utc)

        elif ft == TaskFilterType.COMPANY.value:
            if not company_name:
                raise HTTPException(status_code=400, detail="company_name is required when filter_type=COMPANY")
            clean_comp = company_name.strip()
            base_filters.append(func.lower(TaskItem.company_name) == clean_comp.lower())

        elif ft != TaskFilterType.ALL.value:
            raise HTTPException(status_code=400, detail=f"Invalid filter_type '{filter_type}'")

        # Additional company name filter if provided with any filter_type
        if company_name and ft != TaskFilterType.COMPANY.value:
            clean_comp = company_name.strip()
            base_filters.append(func.lower(TaskItem.company_name) == clean_comp.lower())

        # Count total
        count_stmt = select(func.count(TaskItem.id)).where(and_(*base_filters))
        total = await self.session.scalar(count_stmt) or 0

        # Fetch records ordered by due_date ascending (soonest first)
        query_stmt = (
            select(TaskItem)
            .where(and_(*base_filters))
            .order_by(TaskItem.due_date.asc().nulls_last(), TaskItem.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        tasks = (await self.session.scalars(query_stmt)).all()
        return list(tasks), total

    @staticmethod
    def to_response(task: TaskItem, reference_now: Optional[datetime] = None) -> TaskResponse:
        """Converts ORM model to TaskResponse Pydantic schema with is_overdue flag."""
        now = reference_now or datetime.now(timezone.utc)
        is_overdue = bool(
            task.status == TaskStatus.OPEN.value
            and task.due_date is not None
            and task.due_date < now
        )
        return TaskResponse(
            id=task.id,
            user_id=task.user_id,
            workspace_id=task.workspace_id,
            department_id=task.department_id,
            title=task.title,
            description=task.description,
            status=task.status,
            due_date=task.due_date,
            timezone=task.timezone,
            company_name=task.company_name,
            contact_name=task.contact_name,
            provenance_type=task.provenance_type,
            conversation_id=task.conversation_id,
            message_id=task.message_id,
            document_chunk_id=task.document_chunk_id,
            attributes=task.attributes or {},
            created_at=task.created_at,
            updated_at=task.updated_at,
            completed_at=task.completed_at,
            is_overdue=is_overdue,
        )
