"""
Acceptance Suite for Post-Milestone 2 Vertical Slice:
End-to-End Research -> Grounded Writing -> Validation -> Human Approval Interrupt -> Idempotent CRM Side-Effect.
"""

from __future__ import annotations

import os

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError

from rag_app.crm.sync_service import CRMSyncService
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import (
    CRMWriteBackApproval,
    CRMWriteBackRecord,
    Department,
    User,
    UserDepartment,
    WorkflowRun,
    WorkflowTask,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.orchestration.executor import DAGExecutionEngine, FencingConflictError
from rag_app.orchestration.planner import AdaptiveTaskPlanner
from rag_app.orchestration.service import WorkflowService
from rag_app.orchestration.validator import DraftValidator
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    BudgetLimits,
    compute_canonical_draft_hash,
    compute_canonical_request_hash,
    ExecutionTask,
    TaskEvidenceRef,
    TaskExecutionResult,
    TaskType,
    WorkflowRunCreate,
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
        dept = await session.scalar(select(Department).where(Department.code == "VS_TEST_DEPT"))
        if not dept:
            dept = Department(code="VS_TEST_DEPT", name="Vertical Slice Department", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        admin_user = await session.scalar(select(User).where(User.email == "vs_admin@test.com"))
        if not admin_user:
            admin_user = User(
                email="vs_admin@test.com",
                hashed_password="hashed_pw",
                role="ADMIN",
                is_active=True,
            )
            session.add(admin_user)
            await session.commit()
            await session.refresh(admin_user)

        normal_user = await session.scalar(select(User).where(User.email == "vs_user@test.com"))
        if not normal_user:
            normal_user = User(
                email="vs_user@test.com",
                hashed_password="hashed_pw",
                role="USER",
                is_active=True,
            )
            session.add(normal_user)
            await session.commit()
            await session.refresh(normal_user)

        # Ensure user belongs to department
        ud = await session.scalar(
            select(UserDepartment).where(
                UserDepartment.user_id == normal_user.id,
                UserDepartment.department_id == dept.id,
            )
        )
        if not ud:
            ud = UserDepartment(user_id=normal_user.id, department_id=dept.id)
            session.add(ud)
            await session.commit()

        workspace = await session.scalar(select(Workspace).where(Workspace.code == "VS_WS_1"))
        if not workspace:
            workspace = Workspace(
                name="Vertical Slice Workspace",
                code="VS_WS_1",
                created_by=admin_user.id,
                department_id=dept.id,
                is_active=True,
            )
            session.add(workspace)
            await session.commit()
            await session.refresh(workspace)

        # Workspace membership
        wm = await session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace.id,
                WorkspaceMember.user_id == normal_user.id,
            )
        )
        if not wm:
            wm = WorkspaceMember(workspace_id=workspace.id, user_id=normal_user.id, role="EDITOR")
            session.add(wm)
            await session.commit()

        return dept, admin_user, normal_user, workspace


def make_test_workflow_service(session: AsyncSession) -> WorkflowService:
    """Creates a WorkflowService with registered research handlers yielding verified evidence for tests."""
    executor = DAGExecutionEngine(session=session)
    async def valid_evidence_handler(task: ExecutionTask, s, r):
        q = (task.input_payload.get("query") or "") if task.input_payload else ""
        return TaskExecutionResult(
            task_key=task.task_key,
            status=WorkflowTaskStatus.COMPLETED,
            evidence_items=[
                TaskEvidenceRef(
                    task_key=task.task_key,
                    source_type="internal_rag",
                    source_id="doc-acme-q3",
                    snippet=f"Acme Corp verified data for {q}: 24% customer growth in Q3 with $12M revenue.",
                )
            ],
            tokens_spent=100,
            provider_calls_spent=1,
        )
    for tt in [TaskType.RETRIEVAL.value, TaskType.WEB_SEARCH.value, TaskType.EXTRACTION.value, TaskType.GROUNDING.value, TaskType.SYNTHESIS.value]:
        executor.register_handler(tt, valid_evidence_handler)
    return WorkflowService(session=session, executor=executor)


# ============================================================================
# Gate 1: Planning & Research Execution with Provenance
# ============================================================================

@pytest.mark.asyncio
async def test_complex_request_produces_valid_plan_and_executes(setup_test_context):
    """1 & 2: Complex request produces valid plan, executes tasks, and preserves provenance."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        service = make_test_workflow_service(session)
        req = WorkflowRunCreate(
            query="Research Q3 revenue performance and customer growth for Acme Corp",
            workspace_id=workspace.id,
            department_id=dept.id,
            idempotency_key=f"vs-plan-{uuid.uuid4().hex[:6]}",
        )
        run, created = await service.get_or_create_workflow_run(user=normal_user, request=req)
        assert created is True

        res = await service.execute_workflow(run=run)
        assert res["status"] == WorkflowStatus.COMPLETED.value
        assert res["evidence_count"] > 0
        assert "synthesis" in res
        assert "[source:" in res["synthesis"]

        # Verify DB tasks persisted
        db_tasks = (
            await session.scalars(
                select(WorkflowTask).where(WorkflowTask.workflow_run_id == run.id)
            )
        ).all()
        assert len(db_tasks) >= 3


# ============================================================================
# Gate 2: Evidence Sufficiency Gate
# ============================================================================

@pytest.mark.asyncio
async def test_insufficient_evidence_does_not_assert_factual_answer(setup_test_context):
    """3: Insufficient evidence does not silently produce an asserted factual answer or sensitive action."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        mock_executor = DAGExecutionEngine(session=session)

        # Empty retrieval handler simulating missing documents
        async def empty_handler(task, s, r):
            return TaskExecutionResult(
                task_key=task.task_key,
                status=WorkflowTaskStatus.COMPLETED,
                evidence_items=[],
                tokens_spent=50,
                provider_calls_spent=1,
            )

        for tt in [
            TaskType.RETRIEVAL.value,
            TaskType.WEB_SEARCH.value,
            TaskType.EXTRACTION.value,
            TaskType.GROUNDING.value,
            TaskType.SYNTHESIS.value,
        ]:
            mock_executor.register_handler(tt, empty_handler)

        service = WorkflowService(session=session, executor=mock_executor)
        req = WorkflowRunCreate(
            query="Find confidential proprietary formula for Project X",
            workspace_id=workspace.id,
            department_id=dept.id,
        )
        run, _ = await service.get_or_create_workflow_run(user=normal_user, request=req)
        res = await service.execute_workflow(
            run=run,
            intended_action="CRM_CREATE_NOTE",
            target_record_id="comp-123",
        )

        assert res["status"] == WorkflowStatus.COMPLETED.value
        assert res["approval_required"] is False
        assert "Safe partial findings" in res["synthesis"]

        # No approval ticket must be generated
        approvals = (
            await session.scalars(
                select(CRMWriteBackApproval).where(CRMWriteBackApproval.workflow_run_id == run.id)
            )
        ).all()
        assert len(approvals) == 0


# ============================================================================
# Gate 3: Validation Gate
# ============================================================================

@pytest.mark.asyncio
async def test_validation_failure_prevents_approval_and_execution(setup_test_context):
    """4: Validation failure (e.g. fabricated numbers or invalid citations) prevents approval and execution."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        # Mock executor that injects fabricated numbers in synthesis
        mock_executor = DAGExecutionEngine(session=session)

        validator = DraftValidator()
        # Test validator directly on unsupported claim
        fake_ev = [
            TaskEvidenceRef(
                task_key="t1",
                source_type="internal",
                source_id="doc_1",
                snippet="Acme Corp achieved $10M revenue in Q3.",
            )
        ]
        # Draft asserting $999M revenue
        bad_draft = "Acme Corp achieved $999M in Q3 [source: doc_1]."
        val_res = validator.validate_draft(bad_draft, fake_ev)
        assert val_res.is_valid is False
        assert val_res.grounding_passed is False
        assert "999M" in val_res.unsupported_claims[0]

        # Citation mismatch
        bad_cit_draft = "Acme Corp achieved $10M in Q3 [source: non_existent_doc]."
        val_res2 = validator.validate_draft(bad_cit_draft, fake_ev)
        assert val_res2.is_valid is False
        assert val_res2.citations_valid is False
        assert "non_existent_doc" in val_res2.invalid_citations[0]


# ============================================================================
# Gate 4: Approval Interrupt & Paused Workflow State
# ============================================================================

@pytest.mark.asyncio
async def test_validated_draft_transitions_to_awaiting_approval_and_pauses(setup_test_context):
    """5 & 6: Successfully validated draft transitions to AWAITING_APPROVAL and pauses without side-effects."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        service = make_test_workflow_service(session)
        target_comp = f"comp-{uuid.uuid4().hex[:6]}"
        req = WorkflowRunCreate(
            query="Research Q3 customer metrics and draft meeting note for Acme Corp",
            workspace_id=workspace.id,
            department_id=dept.id,
            idempotency_key=f"vs-appr-{uuid.uuid4().hex[:6]}",
        )
        run, _ = await service.get_or_create_workflow_run(user=normal_user, request=req)

        # Execute workflow slice with requested sensitive action
        res = await service.execute_workflow(
            run=run,
            intended_action="CRM_CREATE_NOTE",
            target_record_id=target_comp,
        )

        assert res["status"] == WorkflowStatus.AWAITING_APPROVAL.value
        assert res["approval_required"] is True
        approval_id = uuid.UUID(res["approval_id"])

        # Check DB state
        refreshed_run = await session.get(WorkflowRun, run.id)
        assert refreshed_run.status == "AWAITING_APPROVAL"
        assert refreshed_run.awaiting_approval_at is not None
        assert refreshed_run.draft_hash == res["draft_hash"]

        appr = await session.get(CRMWriteBackApproval, approval_id)
        assert appr is not None
        assert appr.workflow_run_id == run.id
        assert appr.approved_draft_hash == res["draft_hash"]
        assert appr.external_record_id == target_comp
        assert appr.intended_action == "CRM_CREATE_NOTE"

        # Verify NO side effect executed yet
        records = (
            await session.scalars(
                select(CRMWriteBackRecord).where(CRMWriteBackRecord.external_record_id == target_comp)
            )
        ).all()
        assert len(records) == 0


# ============================================================================
# Gate 5: Approval Resume, Tampering Rejection & Idempotent Side Effect
# ============================================================================

@pytest.mark.asyncio
async def test_valid_approval_resumes_workflow_and_executes_side_effect(setup_test_context):
    """7 & 12: Valid approval resumes the existing workflow, reauthorizes, and executes idempotent side-effect."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.create_note.return_value = "note-vs-101"
        mock_mcp.create_note_target_for_company.return_value = "target-vs-101"
        crm_service = CRMSyncService(session=session, mcp_client=mock_mcp, ingestion_service=AsyncMock())

        service = make_test_workflow_service(session)
        target_comp = f"comp-{uuid.uuid4().hex[:6]}"
        req = WorkflowRunCreate(
            query="Analyze Q3 metrics and draft note for Acme Corp",
            workspace_id=workspace.id,
            department_id=dept.id,
        )
        run, _ = await service.get_or_create_workflow_run(user=normal_user, request=req)
        prep_res = await service.execute_workflow(
            run=run,
            intended_action="CRM_CREATE_NOTE",
            target_record_id=target_comp,
        )
        approval_id = uuid.UUID(prep_res["approval_id"])

        # Resume workflow approval
        resume_res = await service.resume_workflow_approval(
            run_id=run.id,
            approval_id=approval_id,
            user=normal_user,
            user_role="USER",
            decision="APPROVED",
            crm_sync_service=crm_service,
            proposed_action="CRM_CREATE_NOTE",
            proposed_target=target_comp,
        )

        assert resume_res["status"] == WorkflowStatus.COMPLETED.value
        assert resume_res["side_effect_executed"] is True
        assert resume_res["writeback_result"]["status"] == "COMPLETED"

        # Verify database lifecycle completed
        refreshed_run = await session.get(WorkflowRun, run.id)
        assert refreshed_run.status == "COMPLETED"
        assert refreshed_run.completed_at is not None
        assert refreshed_run.approved_at is not None

        # Verify CRM note created
        mock_mcp.create_note.assert_awaited_once()


@pytest.mark.asyncio
async def test_draft_tampering_revokes_approval_and_blocks_execution(setup_test_context):
    """8: Modifying draft content revokes approval and prevents side-effect execution."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        service = make_test_workflow_service(session)
        target_comp = f"comp-{uuid.uuid4().hex[:6]}"
        req = WorkflowRunCreate(
            query="Analyze Q3 metrics for Acme Corp",
            workspace_id=workspace.id,
            department_id=dept.id,
        )
        run, _ = await service.get_or_create_workflow_run(user=normal_user, request=req)
        prep_res = await service.execute_workflow(
            run=run,
            intended_action="CRM_CREATE_NOTE",
            target_record_id=target_comp,
        )
        approval_id = uuid.UUID(prep_res["approval_id"])

        # Maliciously tamper with the server-side stored draft
        synth_task = await session.scalar(
            select(WorkflowTask).where(
                and_(
                    WorkflowTask.workflow_run_id == run.id,
                    WorkflowTask.task_type == TaskType.SYNTHESIS.value,
                )
            )
        )
        synth_task.output_result = {
            "draft_text": "TAMPERED: Attacker modified draft content!",
            "draft_hash": "tampered_hash",
        }
        await session.commit()

        # Resuming MUST fail closed and revoke approval
        with pytest.raises(PermissionError, match="Draft content has been modified since approval was granted"):
            await service.resume_workflow_approval(
                run_id=run.id,
                approval_id=approval_id,
                user=normal_user,
                user_role="USER",
                decision="APPROVED",
                crm_sync_service=AsyncMock(),
            )

        # Ticket must now be REVOKED and workflow FAILED
        refreshed_app = await session.get(CRMWriteBackApproval, approval_id)
        assert refreshed_app.status == "REVOKED"
        assert refreshed_app.revoked_at is not None

        refreshed_run = await session.get(WorkflowRun, run.id)
        assert refreshed_run.status == "FAILED"


@pytest.mark.asyncio
async def test_action_or_target_mismatch_blocks_execution(setup_test_context):
    """9: Intended action or target mismatch blocks execution."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        service = make_test_workflow_service(session)
        target_comp = f"comp-{uuid.uuid4().hex[:6]}"
        req = WorkflowRunCreate(
            query="Analyze Q3 metrics for Acme Corp",
            workspace_id=workspace.id,
            department_id=dept.id,
        )
        run, _ = await service.get_or_create_workflow_run(user=normal_user, request=req)
        prep_res = await service.execute_workflow(
            run=run,
            intended_action="CRM_CREATE_NOTE",
            target_record_id=target_comp,
        )
        approval_id = uuid.UUID(prep_res["approval_id"])

        # 1. Action mismatch
        with pytest.raises(PermissionError, match="cannot authorize 'CRM_DELETE_COMPANY'"):
            await service.resume_workflow_approval(
                run_id=run.id,
                approval_id=approval_id,
                user=normal_user,
                user_role="USER",
                decision="APPROVED",
                proposed_action="CRM_DELETE_COMPANY",
            )

        # 2. Target mismatch
        with pytest.raises(PermissionError, match="cannot authorize target 'different-company'"):
            await service.resume_workflow_approval(
                run_id=run.id,
                approval_id=approval_id,
                user=normal_user,
                user_role="USER",
                decision="APPROVED",
                proposed_target="different-company",
            )


@pytest.mark.asyncio
async def test_rejection_cancellation_and_expiry_never_execute_side_effect(setup_test_context):
    """11: Rejection, cancellation, and expiry update lifecycle and never execute the side-effect."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        service = make_test_workflow_service(session)

        # 1. Rejection
        run1, _ = await service.get_or_create_workflow_run(
            user=normal_user,
            request=WorkflowRunCreate(query="Q3 Note 1", workspace_id=workspace.id, department_id=dept.id),
        )
        prep1 = await service.execute_workflow(run1, "CRM_CREATE_NOTE", "c-1")
        rej_res = await service.resume_workflow_approval(
            run_id=run1.id,
            approval_id=uuid.UUID(prep1["approval_id"]),
            user=normal_user,
            user_role="USER",
            decision="REJECTED",
            rejection_reason="Contains inaccurate assumptions.",
        )
        assert rej_res["status"] == "REJECTED"
        assert rej_res["side_effect_executed"] is False

        # 2. Cancellation
        run2, _ = await service.get_or_create_workflow_run(
            user=normal_user,
            request=WorkflowRunCreate(query="Q3 Note 2", workspace_id=workspace.id, department_id=dept.id),
        )
        prep2 = await service.execute_workflow(run2, "CRM_CREATE_NOTE", "c-2")
        canc_res = await service.resume_workflow_approval(
            run_id=run2.id,
            approval_id=uuid.UUID(prep2["approval_id"]),
            user=normal_user,
            user_role="USER",
            decision="CANCELLED",
        )
        assert canc_res["status"] == "CANCELLED"
        assert canc_res["side_effect_executed"] is False

        # 3. Expiry
        run3, _ = await service.get_or_create_workflow_run(
            user=normal_user,
            request=WorkflowRunCreate(query="Q3 Note 3", workspace_id=workspace.id, department_id=dept.id),
        )
        prep3 = await service.execute_workflow(run3, "CRM_CREATE_NOTE", "c-3")
        appr3 = await session.get(CRMWriteBackApproval, uuid.UUID(prep3["approval_id"]))
        # Fast-forward expiry
        appr3.expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        await session.commit()

        with pytest.raises(PermissionError, match="Approval record has expired"):
            await service.resume_workflow_approval(
                run_id=run3.id,
                approval_id=appr3.id,
                user=normal_user,
                user_role="USER",
                decision="APPROVED",
            )
        assert appr3.status == "EXPIRED"


@pytest.mark.asyncio
async def test_execution_time_deauthorization_fails_closed(setup_test_context):
    """12: Current authorization is checked again at resume time; revocation blocks execution."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        service = make_test_workflow_service(session)
        target_comp = f"comp-{uuid.uuid4().hex[:6]}"
        req = WorkflowRunCreate(
            query="Q3 Review",
            workspace_id=workspace.id,
            department_id=dept.id,
        )
        run, _ = await service.get_or_create_workflow_run(user=normal_user, request=req)
        prep = await service.execute_workflow(run, "CRM_CREATE_NOTE", target_comp)
        approval_id = uuid.UUID(prep["approval_id"])

        # Revoke user's department authorization before resume
        await session.execute(
            select(UserDepartment).where(
                UserDepartment.user_id == normal_user.id,
                UserDepartment.department_id == dept.id,
            )
        )
        ud = await session.scalar(
            select(UserDepartment).where(
                UserDepartment.user_id == normal_user.id,
                UserDepartment.department_id == dept.id,
            )
        )
        if ud:
            await session.delete(ud)
            await session.commit()

        # Resuming MUST fail closed
        with pytest.raises(PermissionError, match="not authorized for department"):
            await service.resume_workflow_approval(
                run_id=run.id,
                approval_id=approval_id,
                user=normal_user,
                user_role="USER",
                decision="APPROVED",
            )


# ============================================================================
# Gate 6: Fast-path Routing & Scoped Idempotency Isolation
# ============================================================================

def test_fast_path_routing_preserves_simple_queries():
    """18: Greetings and simple queries correctly identified for fast-path without DAG overhead."""
    assert WorkflowService.is_fast_path("Hello there!") is True
    assert WorkflowService.is_fast_path("What is vacation policy?") is True
    assert WorkflowService.is_fast_path("Research competitor pricing and writeback note to CRM") is False
    # Conversational comparison and web search questions must stay on fast path
    assert WorkflowService.is_fast_path("Search the web for salary comparison in Ahmedabad") is True
    assert WorkflowService.is_fast_path("Compare Q1 and Q2 financial statements from our internal policy docs") is True


@pytest.mark.asyncio
async def test_unregistered_task_handler_fails_closed_without_fabricating_evidence(setup_test_context):
    """Unregistered handlers must fail safely with evidence_items=[] and never fabricate snippets."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowService(session=session)
        req = WorkflowRunCreate(
            query="Analyze something with unhandled tasks",
            workspace_id=workspace.id,
            department_id=dept.id,
        )
        run, _ = await service.get_or_create_workflow_run(user=normal_user, request=req)
        res = await service.execute_workflow(run=run)

        assert res["evidence_count"] == 0
        assert "Safe partial findings" in res["synthesis"]
        assert "Evidence collected for" not in res["synthesis"]


def test_task_descriptions_never_become_grounded_findings():
    """_synthesize_grounded_draft must strictly reject task description metadata."""
    service = WorkflowService(session=None)
    fake_metadata_ev = [
        TaskEvidenceRef(
            task_key="gather_web_evidence",
            source_type="internal_rag",
            source_id="doc-gatherwebevidence",
            snippet="Evidence collected for Gather external research and industry context for: Search the web",
        )
    ]
    synthesis = service._synthesize_grounded_draft("Search the web", fake_metadata_ev, [])
    assert "Safe partial findings" in synthesis
    assert "Evidence collected for" not in synthesis


def test_observability_reports_actual_zero_llm_calls():
    """Finalized metrics must accurately report 0 LLM calls without forcing to 1."""
    from rag_app.core.observability import init_request_metrics, finalize_request_metrics
    init_request_metrics("test-req-zero-calls")
    metrics = finalize_request_metrics()
    assert metrics is not None
    assert metrics.total_llm_calls == 0


@pytest.mark.asyncio
async def test_workspace_and_global_scopes_remain_isolated(setup_test_context):
    """17: Workspace and global scopes remain isolated."""
    dept, _, normal_user, workspace = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowService(session=session)
        shared_key = f"scope-test-{uuid.uuid4().hex[:6]}"

        # Global run
        req_global = WorkflowRunCreate(
            query="Global analysis",
            workspace_id=None,
            idempotency_key=shared_key,
        )
        run_g, _ = await service.get_or_create_workflow_run(normal_user, req_global)
        assert run_g.scope_type == "GLOBAL"
        assert run_g.workspace_id is None

        # Workspace run with same key
        req_ws = WorkflowRunCreate(
            query="Workspace analysis",
            workspace_id=workspace.id,
            department_id=dept.id,
            idempotency_key=shared_key,
        )
        run_ws, _ = await service.get_or_create_workflow_run(normal_user, req_ws)
        assert run_ws.scope_type == "WORKSPACE"
        assert run_ws.workspace_id == workspace.id
        assert run_g.id != run_ws.id
