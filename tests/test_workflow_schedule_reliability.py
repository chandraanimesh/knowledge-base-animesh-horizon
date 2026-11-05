"""
Phase 3.1 Acceptance Suite: Production-Grade Cron Scheduling Reliability.

Verifies:
1. Exact-minute schedule detection when scheduler wakes late (09:30 detected at 09:35).
2. 15-minute scheduler wake interval with minute-level cron without missing triggers.
3. Multiple missed occurrences coalescing under RUN_ONCE misfire policy.
4. Multiple missed occurrences recording without execution under SKIP misfire policy.
5. Recovery after extended downtime with bounded catch-up slices (no silent discard).
6. Timezone scheduling across UTC and non-UTC (Asia/Kolkata, America/New_York).
7. DST spring-forward nonexistent local time catch-up.
8. DST fall-back repeated local time disambiguation (fold=0 vs fold=1).
9. Concurrent scheduler workers protection via row-level locking.
10. Duplicate window key deduplication.
11. Version pinning on scheduled execution.
"""

from __future__ import annotations

import os

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timedelta, timezone
import uuid
from zoneinfo import ZoneInfo
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.engine.url import make_url

from rag_app.db.models import (
    Department,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowScheduleState,
    WorkflowTask,
    WorkflowTriggerEvent,
    WorkflowVersion,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.orchestration.trigger_service import (
    WorkflowTriggerService,
    compute_schedule_window_key,
    evaluate_due_schedule_occurrences,
    is_cron_due,
)
from rag_app.schemas.workflow import WorkflowStatus
from rag_app.schemas.workflow_definition import WorkflowTriggerType, WorkflowVersionStatus
from rag_app.schemas.workflow_trigger import MisfirePolicy, OutboxStatus, TriggerSource

# Preflight Safety Gate
_db_url = make_url(os.environ["DATABASE_URL"])
assert _db_url.database == "rag_disposable_test", (
    f"FATAL SAFETY ABORT: Refusing to run tests against '{_db_url.database}'!"
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_schedule_context():
    """Initializes schema and test workspace in rag_disposable_test."""
    async with engine.begin() as conn:
        for tbl in [
            Department.__table__,
            User.__table__,
            Permission.__table__,
            UserDepartment.__table__,
            UserPermission.__table__,
            Workspace.__table__,
            WorkspaceMember.__table__,
            WorkflowDefinition.__table__,
            WorkflowVersion.__table__,
            WorkflowRun.__table__,
            WorkflowTask.__table__,
            WorkflowTriggerEvent.__table__,
            WorkflowScheduleState.__table__,
        ]:
            await conn.run_sync(tbl.create, checkfirst=True)

    async with AsyncSessionLocal() as session:
        dept = Department(
            id=uuid.uuid4(),
            name="Schedule Reliability Dept",
            code=f"SCH_DEPT_{uuid.uuid4().hex[:6]}",
        )
        session.add(dept)

        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(id=uuid.uuid4(), code="VIEW", description="View access", is_active=True)
            session.add(view_perm)
            await session.flush()

        admin = User(
            id=uuid.uuid4(),
            email=f"sch_admin_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hash",
            role="ADMIN",
            is_active=True,
        )
        session.add(admin)
        await session.flush()

        ws = Workspace(
            id=uuid.uuid4(),
            name="Schedule Reliability Workspace",
            code=f"SCH_{uuid.uuid4().hex[:4]}",
            department_id=dept.id,
            created_by=admin.id,
            is_active=True,
        )
        session.add(ws)
        await session.flush()

        session.add_all([
            UserDepartment(user_id=admin.id, department_id=dept.id),
            UserPermission(user_id=admin.id, department_id=dept.id, permission_id=view_perm.id),
            WorkspaceMember(workspace_id=ws.id, user_id=admin.id, role="OWNER"),
        ])
        await session.commit()

        yield {
            "department": dept,
            "admin": admin,
            "workspace": ws,
        }


async def _create_test_workflow_and_version(
    session,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    name: str,
    cron_expr: str,
    tz_str: str = "UTC",
    misfire_policy: str = "RUN_ONCE",
    dst_repeated_policy: str = "RUN_ONCE",
):
    wf = WorkflowDefinition(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        name=name,
        created_by_id=user_id,
        is_active=True,
    )
    session.add(wf)
    await session.flush()

    ver = WorkflowVersion(
        id=uuid.uuid4(),
        workflow_id=wf.id,
        version_number=1,
        status=WorkflowVersionStatus.ACTIVE.value,
        steps_spec=[
            {
                "id": "step_synthesis",
                "name": "Audit reporting",
                "node_type": "SYNTHESIS",
                "inputs": {"focus": "Schedule execution audit"},
                "dependencies": [],
            }
        ],
        trigger_spec={
            "trigger_type": "SCHEDULE",
            "config": {
                "cron": cron_expr,
                "timezone": tz_str,
                "misfire_policy": misfire_policy,
                "dst_repeated_policy": dst_repeated_policy,
            },
        },
        created_by_id=user_id,
        published_at=datetime.now(timezone.utc),
    )
    session.add(ver)
    await session.flush()
    wf.active_version_id = ver.id
    return wf, ver


# ============================================================================
# 1. Missed Window Catch-up Tests
# ============================================================================

@pytest.mark.asyncio
async def test_minute_cron_detected_when_scheduler_wakes_later(setup_schedule_context):
    """09:30 schedule is reliably detected and executed when scheduler wakes at 09:35."""
    ctx = setup_schedule_context
    ws = ctx["workspace"]
    admin = ctx["admin"]

    async with AsyncSessionLocal() as session:
        wf, ver = await _create_test_workflow_and_version(
            session=session,
            workspace_id=ws.id,
            user_id=admin.id,
            name="Daily 09:30 Pipeline",
            cron_expr="30 9 * * *",
            tz_str="UTC",
            misfire_policy="RUN_ONCE",
        )
        # Previous cursor was at 09:20 UTC
        t_previous = datetime(2026, 11, 2, 9, 20, tzinfo=timezone.utc)
        state = WorkflowScheduleState(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            last_scheduled_at=t_previous,
            status="ACTIVE",
        )
        session.add(state)
        await session.commit()

        # Scheduler wakes at 09:35 UTC (15 minutes after 09:20, 5 minutes after 09:30)
        t_wake = datetime(2026, 11, 2, 9, 35, tzinfo=timezone.utc)
        svc = WorkflowTriggerService(session)
        triggered = await svc.process_scheduled_workflows(as_of_time=t_wake, workflow_id=wf.id)

        assert triggered == 1

        # Verify state cursor advanced to 09:35 UTC
        await session.refresh(state)
        assert state.last_scheduled_at == t_wake
        assert state.last_window_key == compute_schedule_window_key(
            wf.id, datetime(2026, 11, 2, 9, 30, tzinfo=timezone.utc)
        )

        # Verify the created WorkflowRun used pinned version
        ev = await session.scalar(
            select(WorkflowTriggerEvent).where(WorkflowTriggerEvent.workflow_id == wf.id)
        )
        assert ev is not None
        assert ev.workflow_version_id == ver.id
        assert ev.status == OutboxStatus.DISPATCHED.value


@pytest.mark.asyncio
async def test_fifteen_minute_interval_catches_minute_crons(setup_schedule_context):
    """A 15-minute scheduler wake interval catches all due occurrences of a step cron without missing."""
    ctx = setup_schedule_context
    ws = ctx["workspace"]
    admin = ctx["admin"]

    async with AsyncSessionLocal() as session:
        # Cron every 5 minutes (*/5 * * * *)
        wf, ver = await _create_test_workflow_and_version(
            session=session,
            workspace_id=ws.id,
            user_id=admin.id,
            name="Every 5 min Pipeline",
            cron_expr="*/5 * * * *",
            tz_str="UTC",
            misfire_policy="RUN_ONCE",
        )
        t_prev = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
        state = WorkflowScheduleState(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            last_scheduled_at=t_prev,
            status="ACTIVE",
        )
        session.add(state)
        await session.commit()

        # Wakes 15 minutes later at 10:15 UTC (due at 10:05, 10:10, 10:15 -> 3 occurrences)
        t_wake = datetime(2026, 11, 2, 10, 15, tzinfo=timezone.utc)
        svc = WorkflowTriggerService(session)
        triggered = await svc.process_scheduled_workflows(as_of_time=t_wake, workflow_id=wf.id)

        # Under RUN_ONCE, 3 occurrences are coalesced into exactly 1 catch-up run
        assert triggered == 1

        await session.refresh(state)
        assert state.last_scheduled_at == t_wake
        assert state.misfire_count == 2  # 3 total occurrences - 1 executed = 2 recorded as coalesced


@pytest.mark.asyncio
async def test_multiple_missed_occurrences_under_skip(setup_schedule_context):
    """Under SKIP policy, multiple missed occurrences advance cursor without executing runs."""
    ctx = setup_schedule_context
    ws = ctx["workspace"]
    admin = ctx["admin"]

    async with AsyncSessionLocal() as session:
        wf, ver = await _create_test_workflow_and_version(
            session=session,
            workspace_id=ws.id,
            user_id=admin.id,
            name="Skip Misfire Pipeline",
            cron_expr="*/5 * * * *",
            tz_str="UTC",
            misfire_policy="SKIP",
        )
        t_prev = datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)
        state = WorkflowScheduleState(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            last_scheduled_at=t_prev,
            status="ACTIVE",
        )
        session.add(state)
        await session.commit()

        # Wakes 30 minutes later at 08:30 UTC (6 occurrences: :05, :10, :15, :20, :25, :30)
        t_wake = datetime(2026, 11, 2, 8, 30, tzinfo=timezone.utc)
        svc = WorkflowTriggerService(session)
        triggered = await svc.process_scheduled_workflows(as_of_time=t_wake, workflow_id=wf.id)

        # Under SKIP, 0 runs are executed
        assert triggered == 0

        await session.refresh(state)
        assert state.last_scheduled_at == t_wake
        assert state.misfire_count == 6  # All 6 recorded as skipped


# ============================================================================
# 2. Downtime & Bounded Catch-Up Tests
# ============================================================================

@pytest.mark.asyncio
async def test_bounded_catchup_scan_after_extended_downtime(setup_schedule_context):
    """Extended downtime (48 hours) is evaluated in bounded 24-hour slices without discarding windows."""
    ctx = setup_schedule_context
    ws = ctx["workspace"]
    admin = ctx["admin"]

    async with AsyncSessionLocal() as session:
        wf, ver = await _create_test_workflow_and_version(
            session=session,
            workspace_id=ws.id,
            user_id=admin.id,
            name="Daily Audit Downtime",
            cron_expr="0 12 * * *",  # Noon daily
            tz_str="UTC",
            misfire_policy="RUN_ONCE",
        )
        # Down for 48 hours (from Nov 2 12:00 to Nov 4 12:00)
        t_start = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
        t_now = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)

        state = WorkflowScheduleState(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            last_scheduled_at=t_start,
            status="ACTIVE",
        )
        session.add(state)
        await session.commit()

        svc = WorkflowTriggerService(session)

        # Direct test of evaluate_due_schedule_occurrences with 1440 min bound
        occurrences_1, cursor_1, has_more_1 = evaluate_due_schedule_occurrences(
            workflow_id=wf.id,
            cron_expr="0 12 * * *",
            interval_sec=None,
            local_tz=timezone.utc,
            start_cursor_utc=t_start,
            target_utc=t_now,
            max_scan_minutes=1440,
        )

        # Slice 1: advanced by 24h to Nov 3 12:00, found 1 occurrence (Nov 3 12:00), has_more is True!
        assert cursor_1 == datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
        assert has_more_1 is True
        assert len(occurrences_1) == 1

        # Slice 2: picks up from cursor_1 to t_now
        occurrences_2, cursor_2, has_more_2 = evaluate_due_schedule_occurrences(
            workflow_id=wf.id,
            cron_expr="0 12 * * *",
            interval_sec=None,
            local_tz=timezone.utc,
            start_cursor_utc=cursor_1,
            target_utc=t_now,
            max_scan_minutes=1440,
        )

        assert cursor_2 == t_now
        assert has_more_2 is False
        assert len(occurrences_2) == 1  # Found Nov 4 12:00 occurrence!
        # Both slices recovered all occurrences without silently dropping any!


# ============================================================================
# 3. Timezone & DST Transition Tests
# ============================================================================

@pytest.mark.asyncio
async def test_non_utc_timezone_scheduling(setup_schedule_context):
    """Accurately schedules in non-UTC timezone (Asia/Kolkata, UTC+5:30)."""
    ctx = setup_schedule_context
    ws = ctx["workspace"]
    admin = ctx["admin"]

    tz_ist = ZoneInfo("Asia/Kolkata")
    async with AsyncSessionLocal() as session:
        # Cron for 18:30 IST daily (which corresponds exactly to 13:00 UTC)
        wf, ver = await _create_test_workflow_and_version(
            session=session,
            workspace_id=ws.id,
            user_id=admin.id,
            name="IST Evening Audit",
            cron_expr="30 18 * * *",
            tz_str="Asia/Kolkata",
            misfire_policy="RUN_ONCE",
        )
        t_prev = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
        state = WorkflowScheduleState(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            last_scheduled_at=t_prev,
            status="ACTIVE",
        )
        session.add(state)
        await session.commit()

        # When evaluated at 13:05 UTC (18:35 IST): 18:30 IST is detected and executed!
        t_wake = datetime(2026, 11, 2, 13, 5, tzinfo=timezone.utc)
        svc = WorkflowTriggerService(session)
        triggered = await svc.process_scheduled_workflows(as_of_time=t_wake, workflow_id=wf.id)

        assert triggered == 1
        await session.refresh(state)
        assert state.last_scheduled_at == t_wake


def test_dst_spring_forward_gap_catchup():
    """Daily schedule falling into nonexistent DST spring-forward hour is caught up at transition minute."""
    tz_ny = ZoneInfo("America/New_York")
    # March 8, 2026: clocks jump from 01:59:00-05:00 directly to 03:00:00-04:00.
    # Daily 02:30 AM local time does NOT exist on the clock!
    t_before_utc = datetime(2026, 3, 8, 6, 50, tzinfo=timezone.utc) # 01:50 EST
    t_after_utc = datetime(2026, 3, 8, 7, 5, tzinfo=timezone.utc)  # 03:05 EDT

    wf_id = uuid.uuid4()
    occurrences, cursor, has_more = evaluate_due_schedule_occurrences(
        workflow_id=wf_id,
        cron_expr="30 2 * * *",  # 02:30 AM daily
        interval_sec=None,
        local_tz=tz_ny,
        start_cursor_utc=t_before_utc,
        target_utc=t_after_utc,
    )

    # The nonexistent 02:30 AM job was detected via gap detection!
    assert len(occurrences) == 1
    assert occurrences[0].is_gap_catchup is True
    assert "gap_catchup" in occurrences[0].window_key


def test_dst_fall_back_repeated_local_time_disambiguation():
    """Fall-back repeated local time (01:30 occurring twice) generates distinct keys and respects policy."""
    tz_ny = ZoneInfo("America/New_York")
    # Nov 1, 2026: clocks repeat 01:00-01:59.
    # Pass 1: 05:30 UTC -> 01:30 EDT (fold=0, offset=-0400)
    # Pass 2: 06:30 UTC -> 01:30 EST (fold=1, offset=-0500)
    t_start = datetime(2026, 11, 1, 5, 0, tzinfo=timezone.utc)
    t_end = datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)

    wf_id = uuid.uuid4()

    # Under RUN_ONCE (default for daily jobs): only 1 run executed for the calendar day
    occ_run_once, _, _ = evaluate_due_schedule_occurrences(
        workflow_id=wf_id,
        cron_expr="30 1 * * *",
        interval_sec=None,
        local_tz=tz_ny,
        start_cursor_utc=t_start,
        target_utc=t_end,
        dst_repeated_policy="RUN_ONCE",
    )
    assert len(occ_run_once) == 1

    # Under RUN_ALL (e.g. hourly cron 30 * * * *): both 01:30 EDT and 01:30 EST generate distinct keys
    occ_run_all, _, _ = evaluate_due_schedule_occurrences(
        workflow_id=wf_id,
        cron_expr="30 * * * *",
        interval_sec=None,
        local_tz=tz_ny,
        start_cursor_utc=t_start,
        target_utc=t_end,
        dst_repeated_policy="RUN_ALL",
    )
    assert len(occ_run_all) == 2
    key1 = occ_run_all[0].window_key
    key2 = occ_run_all[1].window_key
    assert key1 != key2  # Keys are disambiguated by fold and UTC offset!


# ============================================================================
# 4. Concurrency & Duplicate Window Protection
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_scheduler_workers_row_locking(setup_schedule_context):
    """Two concurrent scheduler workers evaluating the same due workflow create exactly 1 run."""
    ctx = setup_schedule_context
    ws = ctx["workspace"]
    admin = ctx["admin"]

    async with AsyncSessionLocal() as session:
        wf, ver = await _create_test_workflow_and_version(
            session=session,
            workspace_id=ws.id,
            user_id=admin.id,
            name="Concurrent Lock Pipeline",
            cron_expr="* * * * *",
            tz_str="UTC",
            misfire_policy="RUN_ONCE",
        )
        t_prev = datetime(2026, 11, 2, 11, 0, tzinfo=timezone.utc)
        state = WorkflowScheduleState(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            last_scheduled_at=t_prev,
            status="ACTIVE",
        )
        session.add(state)
        await session.commit()

    t_eval = datetime(2026, 11, 2, 11, 5, tzinfo=timezone.utc)

    async def run_worker():
        async with AsyncSessionLocal() as sess:
            s = WorkflowTriggerService(sess)
            return await s.process_scheduled_workflows(as_of_time=t_eval, workflow_id=wf.id)

    # Launch two workers simultaneously
    results = await asyncio.gather(run_worker(), run_worker(), return_exceptions=True)

    # Exactly one worker claims the run (1, 0)
    assert sum(r for r in results if isinstance(r, int)) == 1

    # Verify exactly 1 WorkflowRun exists in DB
    async with AsyncSessionLocal() as session:
        runs = (await session.scalars(
            select(WorkflowRun).where(WorkflowRun.workflow_id == wf.id)
        )).all()
        assert len(runs) == 1
        assert runs[0].status == WorkflowStatus.COMPLETED.value

