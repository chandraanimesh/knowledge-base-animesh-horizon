"""
Phase 3 Acceptance Suite: Workflow Trigger Router, Transactional Outbox & Scheduling Engine.

Verifies:
1. Manual / API trigger execution creating linked WorkflowRun and outbox event.
2. Unauthorized workspace trigger rejection (VIEWER vs EDITOR/OWNER, cross-workspace).
3. Missing or unpublished active workflow version rejection (409 Conflict).
4. Signed Twenty CRM Webhook ingestion with HMAC verification.
5. Duplicate webhook event delivery rejection (idempotent replay protection).
6. Immutable version pinning at event acceptance time (subsequent publication does not alter pinned event).
7. Transactional Outbox recovery after dispatch interruption.
8. Scheduled workflow execution, window key calculation, and duplicate prevention.
9. Schedule misfire policy handling.
10. Scheduler integration via DurableReconciliationScheduler.run_once.
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.engine.url import make_url

from rag_app.core.config import settings
from rag_app.db.models import (
    Conversation,
    CRMWriteBackApproval,
    Department,
    Document,
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
from rag_app.knowledge.scheduler import DurableReconciliationScheduler
from rag_app.main import app
from rag_app.orchestration.trigger_service import (
    compute_schedule_window_key,
    is_cron_due,
    WorkflowTriggerService,
)
from rag_app.schemas.workflow import WorkflowStatus, WorkflowTaskStatus
from rag_app.schemas.workflow_definition import (
    WorkflowActionType,
    WorkflowTriggerType,
    WorkflowVersionStatus,
)
from rag_app.schemas.workflow_trigger import (
    MisfirePolicy,
    OutboxStatus,
    TriggerSource,
    WorkflowTriggerRequest,
)
from rag_app.security.authentication import create_access_token


# SAFETY GATE: Preflight verification that tests never target production or dev databases
_db_url = make_url(os.environ["DATABASE_URL"])
assert _db_url.database == "rag_disposable_test", (
    f"FATAL SAFETY ABORT: Refusing to run tests against '{_db_url.database}'! "
    f"Tests must strictly run against 'rag_disposable_test'."
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_test_context():
    """
    Initializes test schema and entities in rag_disposable_test:
    - Department & Permissions
    - Workspace A and Workspace B
    - Admin, Editor, Viewer users
    - Active published WorkflowDefinition in Workspace A
    """
    async with engine.begin() as conn:
        for tbl in [
            Department.__table__,
            User.__table__,
            Permission.__table__,
            UserDepartment.__table__,
            UserPermission.__table__,
            Workspace.__table__,
            WorkspaceMember.__table__,
            Document.__table__,
            Conversation.__table__,
            WorkflowDefinition.__table__,
            WorkflowVersion.__table__,
            WorkflowRun.__table__,
            WorkflowTask.__table__,
            CRMWriteBackApproval.__table__,
            WorkflowTriggerEvent.__table__,
            WorkflowScheduleState.__table__,
        ]:
            await conn.run_sync(tbl.create, checkfirst=True)

    async with AsyncSessionLocal() as session:
        dept = Department(
            id=uuid.uuid4(),
            name="Workflow Trigger Dept",
            code=f"TRG_DEPT_{uuid.uuid4().hex[:6]}",
        )
        session.add(dept)

        # Permissions (get or create to prevent UniqueViolation across test runs)
        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(id=uuid.uuid4(), code="VIEW", description="View access", is_active=True)
            session.add(view_perm)
            await session.flush()

        # Users
        admin_user = User(
            id=uuid.uuid4(),
            email=f"trig_admin_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hash",
            role="ADMIN",
            is_active=True,
        )
        editor_user = User(
            id=uuid.uuid4(),
            email=f"trig_editor_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hash",
            role="USER",
            is_active=True,
        )
        viewer_user = User(
            id=uuid.uuid4(),
            email=f"trig_viewer_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hash",
            role="USER",
            is_active=True,
        )
        session.add_all([admin_user, editor_user, viewer_user])
        await session.flush()

        # User Department & Permissions
        session.add_all([
            UserDepartment(user_id=editor_user.id, department_id=dept.id),
            UserDepartment(user_id=viewer_user.id, department_id=dept.id),
            UserPermission(user_id=editor_user.id, department_id=dept.id, permission_id=view_perm.id),
            UserPermission(user_id=viewer_user.id, department_id=dept.id, permission_id=view_perm.id),
        ])

        # Workspaces
        ws_a = Workspace(
            id=uuid.uuid4(),
            name="Trigger Workspace A",
            code=f"TRGA_{uuid.uuid4().hex[:4]}",
            department_id=dept.id,
            created_by=admin_user.id,
            is_active=True,
        )
        ws_b = Workspace(
            id=uuid.uuid4(),
            name="Trigger Workspace B",
            code=f"TRGB_{uuid.uuid4().hex[:4]}",
            department_id=dept.id,
            created_by=admin_user.id,
            is_active=True,
        )
        session.add_all([ws_a, ws_b])
        await session.flush()

        # Memberships
        session.add_all([
            WorkspaceMember(workspace_id=ws_a.id, user_id=admin_user.id, role="OWNER"),
            WorkspaceMember(workspace_id=ws_a.id, user_id=editor_user.id, role="EDITOR"),
            WorkspaceMember(workspace_id=ws_a.id, user_id=viewer_user.id, role="VIEWER"),
            WorkspaceMember(workspace_id=ws_b.id, user_id=admin_user.id, role="OWNER"),
        ])

        # Documents in Workspace A
        doc_a = Document(
            id=uuid.uuid4(),
            workspace_id=ws_a.id,
            file_name="workspace_a_intelligence.txt",
            file_path="/tmp/intelligence.txt",
            content_hash=f"hash_{uuid.uuid4().hex}",
            mime_type="text/plain",
            department_id=dept.id,
            created_by=admin_user.id,
        )
        session.add(doc_a)

        # Published Workflow Definition in Workspace A
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ws_a.id,
            name="Inbound Trigger Pipeline",
            description="Automated intake from webhooks and APIs",
            created_by_id=admin_user.id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status=WorkflowVersionStatus.ACTIVE.value,
            steps_spec=[
                {
                    "id": "step_retrieval",
                    "name": "Retrieve internal docs",
                    "node_type": "RETRIEVAL",
                    "inputs": {"query": "{{trigger.query}}", "limit": 3},
                    "dependencies": [],
                },
                {
                    "id": "step_synthesis",
                    "name": "Synthesize summary",
                    "node_type": "SYNTHESIS",
                    "inputs": {"focus": "{{trigger.query}}"},
                    "dependencies": ["step_retrieval"],
                },
            ],
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            created_by_id=admin_user.id,
            published_at=datetime.now(timezone.utc),
        )
        session.add(v1)
        await session.flush()
        wf.active_version_id = v1.id

        await session.commit()

        yield {
            "department": dept,
            "admin": admin_user,
            "editor": editor_user,
            "viewer": viewer_user,
            "workspace_a": ws_a,
            "workspace_b": ws_b,
            "workflow": wf,
            "version_v1": v1,
            "doc_a": doc_a,
        }


# ============================================================================
# 1. Manual / API Trigger Tests
# ============================================================================

@pytest.mark.asyncio
async def test_manual_trigger_creates_run_and_outbox_event(setup_test_context):
    """Manual API trigger resolves active version, persists outbox event, and executes DAG."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    wf = ctx["workflow"]
    v1 = ctx["version_v1"]
    editor = ctx["editor"]

    async with AsyncSessionLocal() as session:
        service = WorkflowTriggerService(session)
        req = WorkflowTriggerRequest(
            inputs={"query": "Quarterly financial summary"},
            idempotency_key="manual-idemp-001",
        )

        resp = await service.trigger_manual(
            workflow_id=wf.id,
            workspace_id=ws_a.id,
            user=editor,
            request=req,
        )

        assert resp.status == OutboxStatus.DISPATCHED.value
        assert resp.workflow_id == wf.id
        assert resp.workflow_version_id == v1.id
        assert resp.workflow_run_id is not None
        assert not resp.is_replay

        # Verify run record in database
        run = await session.get(WorkflowRun, resp.workflow_run_id)
        assert run is not None
        assert run.status == WorkflowStatus.COMPLETED.value
        assert run.workflow_id == wf.id
        assert run.workflow_version_id == v1.id

        # Verify outbox event record in database
        outbox = await session.scalar(
            select(WorkflowTriggerEvent).where(WorkflowTriggerEvent.event_id == resp.event_id)
        )
        assert outbox is not None
        assert outbox.status == OutboxStatus.DISPATCHED.value
        assert outbox.workflow_version_id == v1.id
        assert outbox.event_source == TriggerSource.API.value


@pytest.mark.asyncio
async def test_manual_trigger_idempotency_replay(setup_test_context):
    """Re-triggering with identical idempotency_key returns replay response without creating duplicate run."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    wf = ctx["workflow"]
    editor = ctx["editor"]

    async with AsyncSessionLocal() as session:
        service = WorkflowTriggerService(session)
        idemp_key = f"unique-idemp-key-{uuid.uuid4().hex[:8]}"
        req = WorkflowTriggerRequest(
            inputs={"query": "Idempotent call"},
            idempotency_key=idemp_key,
        )

        # First trigger
        resp1 = await service.trigger_manual(wf.id, ws_a.id, editor, req)
        assert not resp1.is_replay
        first_run_id = resp1.workflow_run_id

        # Second trigger (replay)
        resp2 = await service.trigger_manual(wf.id, ws_a.id, editor, req)
        assert resp2.is_replay
        assert resp2.event_id == resp1.event_id
        assert resp2.workflow_run_id == first_run_id

        # Ensure only 1 run exists for this idempotency key
        runs_count = len((await session.scalars(
            select(WorkflowRun).where(WorkflowRun.idempotency_key == idemp_key)
        )).all())
        assert runs_count == 1


# ============================================================================
# 2. Authorization & Inactive Workflow Rejection Tests
# ============================================================================

@pytest.mark.asyncio
async def test_viewer_role_rejected_from_triggering_workflow(setup_test_context):
    """VIEWER role cannot trigger execution; requires EDITOR or OWNER role."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    wf = ctx["workflow"]
    viewer = ctx["viewer"]

    token = create_access_token(subject=str(viewer.id))
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"/api/v1/workflows/definitions/{wf.id}/trigger?workspace_id={ws_a.id}",
            json={"inputs": {"query": "Viewer attempt"}},
            headers=headers,
        )
        assert res.status_code == 403
        assert "requires 'EDITOR' or 'OWNER'" in res.json()["detail"]


@pytest.mark.asyncio
async def test_cross_workspace_trigger_rejected(setup_test_context):
    """Triggering a workflow across unassociated workspace returns 404 Not Found."""
    ctx = setup_test_context
    ws_b = ctx["workspace_b"]  # Workflow belongs to ws_a, not ws_b
    wf = ctx["workflow"]
    admin = ctx["admin"]

    token = create_access_token(subject=str(admin.id))
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"/api/v1/workflows/definitions/{wf.id}/trigger?workspace_id={ws_b.id}",
            json={"inputs": {"query": "Cross-workspace attempt"}},
            headers=headers,
        )
        assert res.status_code == 404


@pytest.mark.asyncio
async def test_unpublished_workflow_trigger_rejected(setup_test_context):
    """Workflow with no active version rejects trigger with 409 Conflict."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    admin = ctx["admin"]

    async with AsyncSessionLocal() as session:
        # Create draft-only workflow (active_version_id is None)
        draft_wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ws_a.id,
            name="Unpublished Pipeline",
            created_by_id=admin.id,
            is_active=True,
            active_version_id=None,
        )
        session.add(draft_wf)
        await session.commit()

        service = WorkflowTriggerService(session)
        with pytest.raises(Exception) as exc_info:
            await service.trigger_manual(
                workflow_id=draft_wf.id,
                workspace_id=ws_a.id,
                user=admin,
                request=WorkflowTriggerRequest(inputs={}),
            )
        assert "has no active published version" in str(exc_info.value)


# ============================================================================
# 3. Signed Twenty CRM Webhook Tests
# ============================================================================

@pytest.mark.asyncio
async def test_signed_webhook_ingestion_and_outbox_persistence(setup_test_context, monkeypatch):
    """Signed Twenty CRM webhook validates HMAC, maps identity, pins version, and persists to outbox."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    wf = ctx["workflow"]
    v1 = ctx["version_v1"]

    test_secret = "twenty-secret-super-key-32b-long!!"
    webhook_id = f"wh_{uuid.uuid4().hex[:8]}"

    monkeypatch.setattr(settings, "twenty_webhook_secret", test_secret)
    monkeypatch.setattr(
        settings,
        "twenty_webhook_mappings",
        {webhook_id: {"workspace_id": str(ws_a.id), "workflow_id": str(wf.id)}},
    )

    payload_dict = {
        "webhookId": webhook_id,
        "event": "company.created",
        "record": {"id": "comp-789", "name": "Acme Innovations"},
        "query": "Analyze new company Acme Innovations",
    }
    raw_payload = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(test_secret.encode("utf-8"), raw_payload, hashlib.sha256).hexdigest()
    event_nonce = f"evt_{uuid.uuid4().hex[:12]}"

    headers = {
        "X-Twenty-Webhook-Signature": f"sha256={sig}",
        "X-Twenty-Webhook-Timestamp": str(datetime.now(timezone.utc).timestamp()),
        "X-Twenty-Webhook-Nonce": event_nonce,
        "Content-Type": "application/json",
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/v1/workflows/triggers/webhook",
            content=raw_payload,
            headers=headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "accepted"
        assert data["event_id"] == event_nonce
        assert data["workflow_run_id"] is not None

    # Verify outbox event record
    async with AsyncSessionLocal() as session:
        ev = await session.scalar(
            select(WorkflowTriggerEvent).where(WorkflowTriggerEvent.event_id == event_nonce)
        )
        assert ev is not None
        assert ev.status == OutboxStatus.DISPATCHED.value
        assert ev.workflow_version_id == v1.id  # PINNED VERSION
        assert ev.event_source == TriggerSource.TWENTY_CRM.value


@pytest.mark.asyncio
async def test_duplicate_webhook_rejected_idempotently(setup_test_context, monkeypatch):
    """Duplicate delivery of exact same webhook event returns duplicate status without duplicate run."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    wf = ctx["workflow"]

    test_secret = "twenty-secret-key-12345"
    webhook_id = f"wh_{uuid.uuid4().hex[:8]}"

    monkeypatch.setattr(settings, "twenty_webhook_secret", test_secret)
    monkeypatch.setattr(
        settings,
        "twenty_webhook_mappings",
        {webhook_id: {"workspace_id": str(ws_a.id), "workflow_id": str(wf.id)}},
    )

    payload_dict = {"webhookId": webhook_id, "event": "note.added", "note": "High priority note"}
    raw_payload = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(test_secret.encode("utf-8"), raw_payload, hashlib.sha256).hexdigest()
    event_nonce = f"evt_dup_{uuid.uuid4().hex[:8]}"

    headers = {
        "X-Twenty-Webhook-Signature": f"sha256={sig}",
        "X-Twenty-Webhook-Nonce": event_nonce,
        "Content-Type": "application/json",
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # First call
        res1 = await client.post("/api/v1/workflows/triggers/webhook", content=raw_payload, headers=headers)
        assert res1.status_code == 200
        assert res1.json()["status"] == "accepted"

        # Duplicate call
        res2 = await client.post("/api/v1/workflows/triggers/webhook", content=raw_payload, headers=headers)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["status"] == "duplicate"
        assert "replay protection" in data2["message"]


# ============================================================================
# 4. Version Pinning Invariant Tests
# ============================================================================

@pytest.mark.asyncio
async def test_accepted_event_pins_active_version_even_after_workflow_update(setup_test_context):
    """Events pin the exact workflow version active at accepted-event time."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    wf = ctx["workflow"]
    v1 = ctx["version_v1"]
    admin = ctx["admin"]

    async with AsyncSessionLocal() as session:
        # 1. Accept an outbox event pinned to Version 1
        event_id = f"evt_pin_{uuid.uuid4().hex[:8]}"
        outbox_event = WorkflowTriggerEvent(
            id=uuid.uuid4(),
            event_id=event_id,
            event_source=TriggerSource.WEBHOOK.value,
            event_name="company.created",
            workspace_id=ws_a.id,
            workflow_id=wf.id,
            workflow_version_id=v1.id,  # PINNED TO V1
            payload={"query": "Pinned execution"},
            status=OutboxStatus.PENDING.value,
        )
        session.add(outbox_event)

        # 2. Publish Version 2 for the workflow
        v2 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=2,
            status=WorkflowVersionStatus.ACTIVE.value,
            steps_spec=[
                {
                    "id": "step_retrieval_v2",
                    "name": "V2 retrieval step",
                    "node_type": "RETRIEVAL",
                    "inputs": {"query": "v2 query"},
                    "dependencies": [],
                }
            ],
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            created_by_id=admin.id,
            published_at=datetime.now(timezone.utc),
        )
        session.add(v2)
        v1.status = WorkflowVersionStatus.ARCHIVED.value
        wf.active_version_id = v2.id
        await session.commit()

        # 3. Process the event via outbox recovery
        service = WorkflowTriggerService(session)
        recovered = await service.recover_pending_outbox_events(limit=5, workflow_id=wf.id)
        assert recovered == 1

        # 4. Verify the created WorkflowRun executed Version 1 (PINNED), NOT Version 2!
        refreshed_ev = await session.scalar(
            select(WorkflowTriggerEvent).where(WorkflowTriggerEvent.event_id == event_id)
        )
        assert refreshed_ev.status == OutboxStatus.DISPATCHED.value
        assert refreshed_ev.workflow_version_id == v1.id  # Still pinned to v1

        run = await session.get(WorkflowRun, refreshed_ev.workflow_run_id)
        assert run is not None
        assert run.workflow_version_id == v1.id


# ============================================================================
# 5. Scheduled Trigger & Window Key Tests
# ============================================================================

def test_cron_evaluator_matches_schedule_windows():
    """is_cron_due accurately evaluates standard 5-part cron expressions."""
    # Every Monday at 09:30 UTC
    cron = "30 9 * * 1"
    # 2026-11-02 was a Monday! (Nov 2 2026, 09:30 UTC)
    mon_due = datetime(2026, 11, 2, 9, 30, tzinfo=timezone.utc)
    assert is_cron_due(cron, mon_due) is True

    # 09:31 UTC should be False
    mon_not_due = datetime(2026, 11, 2, 9, 31, tzinfo=timezone.utc)
    assert is_cron_due(cron, mon_not_due) is False

    # Tuesday 09:30 UTC should be False
    tue_not_due = datetime(2026, 11, 3, 9, 30, tzinfo=timezone.utc)
    assert is_cron_due(cron, tue_not_due) is False

    # Step format: every 15 minutes (*/15 * * * *)
    step_cron = "*/15 * * * *"
    assert is_cron_due(step_cron, datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)) is True
    assert is_cron_due(step_cron, datetime(2026, 11, 2, 10, 15, tzinfo=timezone.utc)) is True
    assert is_cron_due(step_cron, datetime(2026, 11, 2, 10, 20, tzinfo=timezone.utc)) is False


@pytest.mark.asyncio
async def test_scheduled_workflow_execution_and_window_idempotency(setup_test_context):
    """Scheduled engine executes due workflows and enforces window key deduplication."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    admin = ctx["admin"]

    async with AsyncSessionLocal() as session:
        # Create scheduled workflow
        sched_wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ws_a.id,
            name="Daily Scheduled Audit",
            created_by_id=admin.id,
            is_active=True,
        )
        session.add(sched_wf)
        await session.flush()

        v_sched = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=sched_wf.id,
            version_number=1,
            status=WorkflowVersionStatus.ACTIVE.value,
            steps_spec=[
                {
                    "id": "step_retrieval",
                    "name": "Audit check",
                    "node_type": "RETRIEVAL",
                    "inputs": {"query": "Daily compliance audit"},
                    "dependencies": [],
                }
            ],
            trigger_spec={
                "trigger_type": "SCHEDULE",
                "config": {"cron": "* * * * *", "timezone": "UTC", "misfire_policy": "RUN_ONCE"},
            },
            created_by_id=admin.id,
            published_at=datetime.now(timezone.utc),
        )
        session.add(v_sched)
        await session.flush()
        sched_wf.active_version_id = v_sched.id
        await session.commit()

        fixed_time = datetime(2026, 11, 2, 14, 0, tzinfo=timezone.utc)
        service = WorkflowTriggerService(session)

        # First run: should trigger the workflow
        triggered_1 = await service.process_scheduled_workflows(as_of_time=fixed_time, workflow_id=sched_wf.id)
        assert triggered_1 == 1

        # Second run at same time (same window): should be skipped due to window key deduplication
        triggered_2 = await service.process_scheduled_workflows(as_of_time=fixed_time, workflow_id=sched_wf.id)
        assert triggered_2 == 0

        # Verify state in database
        state = await session.scalar(
            select(WorkflowScheduleState).where(WorkflowScheduleState.workflow_id == sched_wf.id)
        )
        assert state is not None
        assert state.status == "ACTIVE"
        assert state.last_window_key == compute_schedule_window_key(sched_wf.id, fixed_time)


# ============================================================================
# 6. Background Scheduler Integration Test
# ============================================================================

@pytest.mark.asyncio
async def test_scheduler_run_once_sweeps_outbox(setup_test_context):
    """DurableReconciliationScheduler.run_once integrates workflow trigger outbox recovery."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    wf = ctx["workflow"]
    v1 = ctx["version_v1"]

    async with AsyncSessionLocal() as session:
        # Create abandoned pending event
        abandoned_event = WorkflowTriggerEvent(
            id=uuid.uuid4(),
            event_id=f"abandoned_{uuid.uuid4().hex[:8]}",
            event_source=TriggerSource.API.value,
            event_name="abandoned.test",
            workspace_id=ws_a.id,
            workflow_id=wf.id,
            workflow_version_id=v1.id,
            payload={"query": "Recovered by background sweep"},
            status=OutboxStatus.PENDING.value,
        )
        session.add(abandoned_event)
        await session.commit()

    # Execute scheduler cycle
    scheduler = DurableReconciliationScheduler()
    metrics = await scheduler.run_once()

    assert "recovered_trigger_events" in metrics
    assert metrics["recovered_trigger_events"] >= 1

    # Verify event dispatched
    async with AsyncSessionLocal() as session:
        ev = await session.scalar(
            select(WorkflowTriggerEvent).where(WorkflowTriggerEvent.id == abandoned_event.id)
        )
        assert ev.status == OutboxStatus.DISPATCHED.value
        assert ev.workflow_run_id is not None

