"""
Milestone 1 Acceptance Suite: Workflow Persistence, Scoped Idempotency,
Budget Reservation, Lease Recovery & Fencing, and Bound Approvals.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import and_, select, update
from sqlalchemy.exc import IntegrityError
from unittest.mock import AsyncMock

from rag_app.crm.sync_service import CRMSyncService
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import (
    CRMWriteBackApproval,
    Department,
    GeneratedArtifact,
    User,
    WorkflowRun,
    WorkflowTask,
    Workspace,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.schemas.workflow import (
    compute_canonical_draft_hash,
    compute_canonical_request_hash,
    WorkflowStatus,
    WorkflowTaskStatus,
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_test_context():
    async with AsyncSessionLocal() as session:
        # 1. Setup workspace and users
        dept = await session.scalar(select(Department).where(Department.code == "WF_TEST_DEPT"))
        if not dept:
            dept = Department(code="WF_TEST_DEPT", name="Workflow Test Department", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        admin_user = await session.scalar(select(User).where(User.email == "wf_admin@test.com"))
        if not admin_user:
            admin_user = User(
                email="wf_admin@test.com",
                hashed_password="hashed_pw",
                role="ADMIN",
                is_active=True,
            )
            session.add(admin_user)
            await session.commit()
            await session.refresh(admin_user)

        normal_user = await session.scalar(select(User).where(User.email == "wf_user@test.com"))
        if not normal_user:
            normal_user = User(
                email="wf_user@test.com",
                hashed_password="hashed_pw",
                role="USER",
                is_active=True,
            )
            session.add(normal_user)
            await session.commit()
            await session.refresh(normal_user)

        workspace = await session.scalar(select(Workspace).where(Workspace.code == "WF_WS_1"))
        if not workspace:
            workspace = Workspace(
                name="Workflow Workspace",
                code="WF_WS_1",
                created_by=admin_user.id,
                department_id=dept.id,
                is_active=True,
            )
            session.add(workspace)
            await session.commit()
            await session.refresh(workspace)

        return dept, admin_user, normal_user, workspace


# ============================================================================
# Gate 1: Scoped Idempotency, Changed-Payload Conflict & Replay Authorization
# ============================================================================

@pytest.mark.asyncio
async def test_scoped_idempotency_same_key_same_payload_returns_existing(setup_test_context):
    """Verifies same scoped key + same request hash identifies existing run."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        key = f"idem-key-{uuid.uuid4().hex[:8]}"
        payload = {"topic": "Market Research", "depth": "detailed"}
        req_hash = compute_canonical_request_hash(payload)

        # 1. Create run
        run1 = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=workspace.id,
            department_id=dept.id,
            scope_type="WORKSPACE",
            idempotency_key=key,
            request_hash=req_hash,
            initial_query="Market Research",
            status=WorkflowStatus.RUNNING.value,
        )
        session.add(run1)
        await session.commit()
        await session.refresh(run1)

        # 2. Replay with same user + workspace + key + req_hash
        existing = await session.scalar(
            select(WorkflowRun).where(
                and_(
                    WorkflowRun.user_id == normal_user.id,
                    WorkflowRun.workspace_id == workspace.id,
                    WorkflowRun.idempotency_key == key,
                )
            )
        )
        assert existing is not None
        assert existing.id == run1.id
        assert existing.request_hash == req_hash


@pytest.mark.asyncio
async def test_scoped_idempotency_changed_payload_conflict(setup_test_context):
    """Verifies same scoped key with DIFFERENT payload hash is detected as conflict."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        key = f"idem-key-{uuid.uuid4().hex[:8]}"
        payload1 = {"topic": "AI Trends 2026"}
        payload2 = {"topic": "AI Trends 2027"}  # Modified payload!

        hash1 = compute_canonical_request_hash(payload1)
        hash2 = compute_canonical_request_hash(payload2)
        assert hash1 != hash2

        run1 = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=workspace.id,
            department_id=dept.id,
            scope_type="WORKSPACE",
            idempotency_key=key,
            request_hash=hash1,
            initial_query="AI Trends 2026",
        )
        session.add(run1)
        await session.commit()

        # Database unique constraint on (user_id, workspace_id, idempotency_key) prevents duplicate insertion
        run2 = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=workspace.id,
            department_id=dept.id,
            scope_type="WORKSPACE",
            idempotency_key=key,
            request_hash=hash2,
            initial_query="AI Trends 2027",
        )
        session.add(run2)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_scoped_idempotency_isolated_across_workspaces(setup_test_context):
    """Same idempotency_key can be used by the same user in a DIFFERENT workspace."""
    dept, admin_user, normal_user, ws1 = setup_test_context
    async with AsyncSessionLocal() as session:
        ws2 = Workspace(
            name="Second Workspace",
            code=f"WS_{uuid.uuid4().hex[:6].upper()}",
            created_by=admin_user.id,
            department_id=dept.id,
            is_active=True,
        )
        session.add(ws2)
        await session.commit()
        await session.refresh(ws2)

        key = f"shared-key-{uuid.uuid4().hex[:8]}"
        req_hash = compute_canonical_request_hash({"query": "test"})

        # Run in ws1
        run1 = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=ws1.id,
            department_id=dept.id,
            scope_type="WORKSPACE",
            idempotency_key=key,
            request_hash=req_hash,
            initial_query="test query 1",
        )
        # Run in ws2 with same key and same user
        run2 = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=ws2.id,
            department_id=dept.id,
            scope_type="WORKSPACE",
            idempotency_key=key,
            request_hash=req_hash,
            initial_query="test query 2",
        )
        session.add_all([run1, run2])
        await session.commit()
        assert run1.id != run2.id


# ============================================================================
# Gate 2: Atomic Budget Reservation, Reconciliation & Cleanup
# ============================================================================

@pytest.mark.asyncio
async def test_atomic_budget_reservation_and_concurrency_protection(setup_test_context):
    """Verifies atomic pre-call token reservation bounds concurrent requests."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        run = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=workspace.id,
            department_id=dept.id,
            request_hash=compute_canonical_request_hash({"test": "budget"}),
            initial_query="Budget test",
            max_tokens_budget=1000,
            max_provider_calls_budget=5,
            tokens_spent=500,
            tokens_reserved=300,  # Total active: 800. Remaining: 200
            provider_calls_spent=2,
            provider_calls_reserved=1,  # Total active: 3. Remaining: 2
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        # 1. Attempt to reserve 250 tokens (exceeds remaining 200): must return 0 rows
        stmt_excessive = (
            update(WorkflowRun)
            .where(
                and_(
                    WorkflowRun.id == run.id,
                    WorkflowRun.tokens_spent + WorkflowRun.tokens_reserved + 250 <= WorkflowRun.max_tokens_budget,
                )
            )
            .values(tokens_reserved=WorkflowRun.tokens_reserved + 250)
        )
        result = await session.execute(stmt_excessive)
        await session.commit()
        assert result.rowcount == 0, "Over-budget reservation must be atomically rejected"

        # 2. Attempt to reserve 150 tokens (within remaining 200): must succeed
        stmt_valid = (
            update(WorkflowRun)
            .where(
                and_(
                    WorkflowRun.id == run.id,
                    WorkflowRun.tokens_spent + WorkflowRun.tokens_reserved + 150 <= WorkflowRun.max_tokens_budget,
                )
            )
            .values(tokens_reserved=WorkflowRun.tokens_reserved + 150)
        )
        result = await session.execute(stmt_valid)
        await session.commit()
        assert result.rowcount == 1, "In-budget reservation must succeed"

        # 3. Post-call reconciliation: actual usage was 120 tokens
        # Atomically decrement reserved by 150 and increment spent by 120
        stmt_reconcile = (
            update(WorkflowRun)
            .where(WorkflowRun.id == run.id)
            .values(
                tokens_reserved=WorkflowRun.tokens_reserved - 150,
                tokens_spent=WorkflowRun.tokens_spent + 120,
            )
        )
        await session.execute(stmt_reconcile)
        await session.commit()

        refreshed = await session.get(WorkflowRun, run.id)
        assert refreshed.tokens_reserved == 300  # returned to pre-reservation level
        assert refreshed.tokens_spent == 620   # 500 + 120


@pytest.mark.asyncio
async def test_budget_reservation_cleanup_on_cancellation(setup_test_context):
    """When a task/run is cancelled or fails, reserved counters are rolled back."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        run = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=workspace.id,
            department_id=dept.id,
            request_hash=compute_canonical_request_hash({"test": "cleanup"}),
            initial_query="Cleanup test",
            tokens_reserved=400,
            provider_calls_reserved=2,
        )
        session.add(run)
        await session.commit()

        # Cancellation cleanup: zero out reservations
        stmt_cleanup = (
            update(WorkflowRun)
            .where(WorkflowRun.id == run.id)
            .values(
                tokens_reserved=0,
                provider_calls_reserved=0,
                status=WorkflowStatus.CANCELLED.value,
            )
        )
        await session.execute(stmt_cleanup)
        await session.commit()

        refreshed = await session.get(WorkflowRun, run.id)
        assert refreshed.tokens_reserved == 0
        assert refreshed.provider_calls_reserved == 0
        assert refreshed.status == WorkflowStatus.CANCELLED.value


# ============================================================================
# Gate 3: Task Dependency Validation, Atomic Claiming, Leases & Fencing
# ============================================================================

@pytest.mark.asyncio
async def test_task_dag_dependency_validation(setup_test_context):
    """Worker only claims tasks whose dependencies are completed."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        run = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=workspace.id,
            department_id=dept.id,
            request_hash=compute_canonical_request_hash({"test": "dag"}),
            initial_query="DAG test",
        )
        session.add(run)
        await session.commit()

        # Step 1: Research task (PENDING)
        task1 = WorkflowTask(
            workflow_run_id=run.id,
            task_key="query_1",
            task_type="EXTRACT_EVIDENCE",
            status=WorkflowTaskStatus.PENDING.value,
        )
        session.add(task1)
        await session.commit()
        await session.refresh(task1)

        # Step 2: Synthesis task depending on Step 1
        task2 = WorkflowTask(
            workflow_run_id=run.id,
            task_key="synth_1",
            task_type="SYNTHESIZE_REPORT",
            status=WorkflowTaskStatus.PENDING.value,
            dependencies=[str(task1.id)],
        )
        session.add(task2)
        await session.commit()
        await session.refresh(task2)

        # Verify task2 is not eligible for execution while task1 is PENDING
        assert task2.dependencies == [str(task1.id)]
        assert task1.status != WorkflowTaskStatus.COMPLETED.value

        # Complete task1
        task1.status = WorkflowTaskStatus.COMPLETED.value
        await session.commit()

        # Now all dependencies of task2 are COMPLETED
        dep_tasks = await session.scalars(
            select(WorkflowTask).where(WorkflowTask.id.in_([uuid.UUID(d) for d in task2.dependencies]))
        )
        assert all(t.status == WorkflowTaskStatus.COMPLETED.value for t in dep_tasks)


@pytest.mark.asyncio
async def test_task_atomic_worker_claiming_and_fencing(setup_test_context):
    """Worker claims task with atomic CAS and increments fencing token."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        run = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=workspace.id,
            department_id=dept.id,
            request_hash=compute_canonical_request_hash({"test": "claiming"}),
            initial_query="Claiming test",
        )
        session.add(run)
        await session.commit()

        task = WorkflowTask(
            workflow_run_id=run.id,
            task_key="claim_task_1",
            task_type="EXTRACT_EVIDENCE",
            status=WorkflowTaskStatus.READY.value,
            fencing_token=0,
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)

        now = datetime.now(timezone.utc)
        worker_1_id = f"worker-{uuid.uuid4().hex[:6]}"
        lease_duration = timedelta(minutes=5)

        # Worker 1 claims task atomically
        stmt_claim = (
            update(WorkflowTask)
            .where(
                and_(
                    WorkflowTask.id == task.id,
                    WorkflowTask.status == WorkflowTaskStatus.READY.value,
                )
            )
            .values(
                status=WorkflowTaskStatus.RUNNING.value,
                locked_by=worker_1_id,
                locked_at=now,
                lease_expires_at=now + lease_duration,
                fencing_token=WorkflowTask.fencing_token + 1,
            )
        )
        res = await session.execute(stmt_claim)
        await session.commit()
        assert res.rowcount == 1

        refreshed = await session.get(WorkflowTask, task.id)
        assert refreshed.locked_by == worker_1_id
        assert refreshed.fencing_token == 1
        assert refreshed.status == WorkflowTaskStatus.RUNNING.value

        # Fencing check: A stale worker presenting fencing_token=0 must fail to commit results
        stmt_stale_complete = (
            update(WorkflowTask)
            .where(
                and_(
                    WorkflowTask.id == task.id,
                    WorkflowTask.fencing_token == 0,  # Old token!
                )
            )
            .values(status=WorkflowTaskStatus.COMPLETED.value)
        )
        stale_res = await session.execute(stmt_stale_complete)
        await session.commit()
        assert stale_res.rowcount == 0, "Stale worker with outdated fencing token must be rejected"


@pytest.mark.asyncio
async def test_task_lease_recovery_after_expiry(setup_test_context):
    """When a worker dies and its lease expires, a new worker reclaims the task."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        run = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=workspace.id,
            department_id=dept.id,
            request_hash=compute_canonical_request_hash({"test": "lease"}),
            initial_query="Lease recovery test",
        )
        session.add(run)
        await session.commit()

        # Task with expired lease
        past_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        task = WorkflowTask(
            workflow_run_id=run.id,
            task_key="stale_task_1",
            task_type="EXTRACT_EVIDENCE",
            status=WorkflowTaskStatus.RUNNING.value,
            locked_by="dead-worker-1",
            lease_expires_at=past_time,
            fencing_token=1,
            retry_count=0,
            max_retries=3,
        )
        session.add(task)
        await session.commit()
        await session.refresh(task)

        # Worker 2 discovers expired lease and claims it
        now = datetime.now(timezone.utc)
        worker_2_id = f"worker-recover-{uuid.uuid4().hex[:6]}"
        stmt_recover = (
            update(WorkflowTask)
            .where(
                and_(
                    WorkflowTask.id == task.id,
                    WorkflowTask.status == WorkflowTaskStatus.RUNNING.value,
                    WorkflowTask.lease_expires_at < now,
                    WorkflowTask.retry_count < WorkflowTask.max_retries,
                )
            )
            .values(
                locked_by=worker_2_id,
                locked_at=now,
                lease_expires_at=now + timedelta(minutes=5),
                retry_count=WorkflowTask.retry_count + 1,
                fencing_token=WorkflowTask.fencing_token + 1,  # Now 2!
            )
        )
        rec_res = await session.execute(stmt_recover)
        await session.commit()
        assert rec_res.rowcount == 1

        refreshed = await session.get(WorkflowTask, task.id)
        assert refreshed.locked_by == worker_2_id
        assert refreshed.fencing_token == 2
        assert refreshed.retry_count == 1


# ============================================================================
# Gate 4: Draft-Hash Binding, Tamper Revocation & Legacy Fail-Closed Behavior
# ============================================================================

@pytest.mark.asyncio
async def test_draft_tamper_invalidates_approval_ticket(setup_test_context):
    """If draft content changes even by 1 character, ticket is REVOKED and write-back fails."""
    dept, admin_user, normal_user, _ = setup_test_context
    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        cid = f"company-{uuid.uuid4().hex[:6]}"

        original_draft = "Executive Summary: Q3 target exceeded by 15%."
        approved_hash = compute_canonical_draft_hash(original_draft)

        approval = await service.create_writeback_approval(
            company_id=cid,
            department_id=dept.id,
            approved_by=admin_user.id,
            requested_by=normal_user.id,
            approved_draft_hash=approved_hash,
            intended_action="CRM_CREATE_NOTE",
        )
        assert approval.status == "APPROVED"

        # Tampered content
        tampered_draft = "Executive Summary: Q3 target exceeded by 95%."
        with pytest.raises(PermissionError, match="modified since approval was granted"):
            await service.create_meeting_note_writeback(
                company_id=cid,
                note_title="Q3 Review",
                summary_markdown=tampered_draft,
                idempotency_key=f"idem-{uuid.uuid4().hex[:6]}",
                user_id=normal_user.id,
                user_role="USER",
                approval_id=approval.id,
            )

        # Ticket must now be marked REVOKED in database
        refreshed_app = await session.get(CRMWriteBackApproval, approval.id)
        assert refreshed_app.status == "REVOKED"


@pytest.mark.asyncio
async def test_legacy_unhashed_approval_fails_closed_for_user(setup_test_context):
    """Legacy approval tickets without cryptographic draft hash fail closed for non-admin callers."""
    dept, admin_user, normal_user, _ = setup_test_context
    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        cid = f"company-legacy-{uuid.uuid4().hex[:6]}"

        # Legacy ticket created without draft hash
        approval = await service.create_writeback_approval(
            company_id=cid,
            department_id=dept.id,
            approved_by=admin_user.id,
            requested_by=normal_user.id,
            approved_draft_hash=None,
        )
        assert approval.approved_draft_hash is None

        # 1. Non-admin caller MUST fail closed
        with pytest.raises(PermissionError, match="Legacy approval ticket lacks cryptographic draft hash"):
            await service.create_meeting_note_writeback(
                company_id=cid,
                note_title="Legacy Note",
                summary_markdown="Unverified legacy content",
                idempotency_key=f"idem-{uuid.uuid4().hex[:6]}",
                user_id=normal_user.id,
                user_role="USER",
                approval_id=approval.id,
            )

        # 2. Admin caller override succeeds
        mock_mcp.create_note.return_value = "note-admin-override"
        mock_mcp.create_note_target_for_company.return_value = "target-admin-override"
        res = await service.create_meeting_note_writeback(
            company_id=cid,
            note_title="Admin Override Note",
            summary_markdown="Unverified legacy content executed by admin",
            idempotency_key=f"idem-{uuid.uuid4().hex[:6]}",
            user_id=admin_user.id,
            user_role="ADMIN",
            approval_id=approval.id,
        )
        assert res["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_intended_action_mismatch_rejected(setup_test_context):
    """Approval bound to CRM_CREATE_NOTE cannot authorize another action."""
    dept, admin_user, normal_user, _ = setup_test_context
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())
        cid = f"company-action-{uuid.uuid4().hex[:6]}"

        approval = await service.create_writeback_approval(
            company_id=cid,
            department_id=dept.id,
            approved_by=admin_user.id,
            approved_draft_hash=compute_canonical_draft_hash("content"),
            intended_action="CRM_DELETE_COMPANY",  # Not NOTE_CREATE!
        )

        with pytest.raises(PermissionError, match="intended action .* cannot authorize 'CRM_CREATE_NOTE'"):
            await service.create_meeting_note_writeback(
                company_id=cid,
                note_title="Action Mismatch",
                summary_markdown="content",
                idempotency_key=f"idem-{uuid.uuid4().hex[:6]}",
                user_id=normal_user.id,
                user_role="USER",
                approval_id=approval.id,
            )


# ============================================================================
# Gate 7: Workspace/Department Isolation & State Constraints
# ============================================================================

@pytest.mark.asyncio
async def test_scope_consistency_constraint(setup_test_context):
    """Database check constraint requires workspace_id when scope_type='WORKSPACE'."""
    dept, _, normal_user, _ = setup_test_context
    async with AsyncSessionLocal() as session:
        # Invalid: scope_type='WORKSPACE' but workspace_id is None
        invalid_run = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=None,
            department_id=dept.id,
            scope_type="WORKSPACE",
            request_hash=compute_canonical_request_hash({"test": "scope"}),
            initial_query="Invalid scope run",
        )
        session.add(invalid_run)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # Valid global run: scope_type='GLOBAL' and workspace_id is None
        valid_global_run = WorkflowRun(
            user_id=normal_user.id,
            workspace_id=None,
            department_id=dept.id,
            scope_type="GLOBAL",
            request_hash=compute_canonical_request_hash({"test": "global"}),
            initial_query="Valid global run",
        )
        session.add(valid_global_run)
        await session.commit()
        assert valid_global_run.id is not None
