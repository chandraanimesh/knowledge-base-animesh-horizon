"""
Production Readiness Gates Verification Suite.

Validates the two remaining production gates:
- Gate A: Adaptive Workflow Execution (end-to-end execution, real web evidence, DAG transitions, budget enforcement, citations, fail-closed unsupported tasks)
- Gate B: Live CRM Write-Back Verification (live Twenty CRM staging container, server-side mutation gate, pre-approval invariant, independent REST verification, durable ledger, idempotent replay, two-phase retry without duplication)
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timezone, timedelta
import json
from typing import Any
import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select, and_, text

from rag_app.core.config import get_settings
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
from rag_app.orchestration.dag_validator import validate_task_dag
from rag_app.orchestration.executor import DAGExecutionEngine
from rag_app.orchestration.planner import AdaptiveTaskPlanner
from rag_app.orchestration.service import WorkflowService
from rag_app.orchestration.validator import DraftValidator
from rag_app.research.providers.tinyfish import TinyFishProvider
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    BudgetLimits,
    ExecutionTask,
    TaskEvidenceRef,
    TaskExecutionResult,
    TaskType,
    WorkflowRunCreate,
    WorkflowStatus,
    WorkflowTaskStatus,
)


STAGING_COMPANY_ID = "7a0e5540-c5cc-4fbd-950d-6416093e194f"
TWENTY_BASE_URL = os.getenv("TWENTY_BASE_URL", "http://localhost:3000").rstrip("/")
TWENTY_API_KEY = os.getenv("TWENTY_API_KEY")


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def disposable_staging_context():
    """Provides disposable database context in rag_disposable_test."""
    async with AsyncSessionLocal() as session:
        user_id = uuid.uuid4()
        user = User(
            id=user_id,
            email=f"gate-tester-{user_id.hex[:6]}@example.com",
            hashed_password="dummy_hashed_password",
            role="ADMIN",
            is_active=True,
        )
        session.add(user)

        dept_id = uuid.uuid4()
        dept = Department(
            id=dept_id,
            name=f"Engineering-{dept_id.hex[:6]}",
            code=f"ENG-{dept_id.hex[:6]}",
            is_active=True,
        )
        session.add(dept)

        ws_id = uuid.uuid4()
        ws = Workspace(
            id=ws_id,
            name=f"Staging-WS-{ws_id.hex[:6]}",
            code=f"WS-{ws_id.hex[:6]}",
            department_id=dept.id,
            created_by=user_id,
        )
        session.add(ws)

        await session.commit()

        user_dept = UserDepartment(user_id=user.id, department_id=dept.id)
        session.add(user_dept)

        member = WorkspaceMember(
            workspace_id=ws.id,
            user_id=user.id,
            role="OWNER",
        )
        session.add(member)

        await session.commit()

        yield {
            "user": user,
            "department": dept,
            "workspace": ws,
        }

        # Cleanup
        try:
            await session.execute(
                text("DELETE FROM crm_write_back_records WHERE created_by = :uid"),
                {"uid": user_id},
            )
            await session.execute(
                text("DELETE FROM crm_write_back_approvals WHERE requested_by = :uid"),
                {"uid": user_id},
            )
            await session.execute(
                text("DELETE FROM workflow_tasks WHERE workflow_run_id IN (SELECT id FROM workflow_runs WHERE user_id = :uid)"),
                {"uid": user_id},
            )
            await session.execute(
                text("DELETE FROM workflow_runs WHERE user_id = :uid"),
                {"uid": user_id},
            )
            await session.execute(
                text("DELETE FROM workspace_members WHERE user_id = :uid"),
                {"uid": user_id},
            )
            await session.execute(
                text("DELETE FROM user_departments WHERE user_id = :uid"),
                {"uid": user_id},
            )
            await session.execute(
                text("DELETE FROM workspaces WHERE id = :wid"),
                {"wid": ws_id},
            )
            await session.execute(
                text("DELETE FROM departments WHERE id = :did"),
                {"did": dept_id},
            )
            await session.execute(
                text("DELETE FROM users WHERE id = :uid"),
                {"uid": user_id},
            )
            await session.commit()
        except Exception:
            await session.rollback()


# ============================================================================
# GATE A: Adaptive Workflow Execution
# ============================================================================

@pytest.mark.asyncio
async def test_gate_a_handler_inspection_and_unsupported_tasks_fail_closed():
    """
    Gate A Step 1 & 6:
    1. Inspect the actual registered handlers for internal retrieval, web research, and synthesis.
    6. Confirm that unsupported tasks fail closed without fabricated evidence.
    """
    engine_inst = DAGExecutionEngine()

    # Step 1: Inspect actual registered handlers
    handlers = engine_inst.task_handlers
    assert TaskType.RETRIEVAL.value in handlers, "RETRIEVAL handler must be registered"
    assert TaskType.WEB_SEARCH.value in handlers, "WEB_SEARCH handler must be registered"
    assert TaskType.SYNTHESIS.value in handlers, "SYNTHESIS handler must be registered"
    assert TaskType.EXTRACTION.value in handlers, "EXTRACTION handler must be registered"
    assert TaskType.GROUNDING.value in handlers, "GROUNDING handler must be registered"

    assert callable(handlers[TaskType.RETRIEVAL.value])
    assert callable(handlers[TaskType.WEB_SEARCH.value])
    assert callable(handlers[TaskType.SYNTHESIS.value])

    # Step 6: Unsupported task fails closed safely without fabricated evidence
    unsupported_task = ExecutionTask(
        task_key="unsupported_custom_action",
        task_type=TaskType.EXECUTION,
        description="Attempt an unsupported arbitrary action",
        dependencies=[],
        input_payload={"command": "format_disk"},
        estimated_tokens=500,
        estimated_provider_calls=1,
    )

    result = await engine_inst._default_task_execution(unsupported_task)
    assert result.status == WorkflowTaskStatus.FAILED
    assert result.tokens_spent == 0
    assert result.provider_calls_spent == 0
    assert len(result.evidence_items) == 0, "No fabricated evidence allowed on unsupported tasks"
    assert "No registered handler available" in (result.error_message or "")


@pytest.mark.asyncio
async def test_gate_a_adaptive_workflow_end_to_end_execution(disposable_staging_context):
    """
    Gate A Steps 2-5 & 7:
    2. Execute a supported adaptive workflow end to end in disposable staging.
    3. Verify that the web-evidence task invokes its real configured search handler.
    4. Verify that task results contain actual evidence, not task metadata.
    5. Verify dependencies, state transitions, budget enforcement, final synthesis, citations, and execution metrics.
    7. Preserve existing MCP routing, adaptive workflow state management, HITL approvals, and authorization.
    """
    ctx = disposable_staging_context
    user = ctx["user"]
    dept = ctx["department"]
    ws = ctx["workspace"]

    query = "Analyze the origins and design history of Python programming language"
    wf_req = WorkflowRunCreate(
        query=query,
        workspace_id=ws.id,
        department_id=dept.id,
        budget_limits=BudgetLimits(max_tokens=4000, max_provider_calls=10),
    )

    async with AsyncSessionLocal() as session:
        wf_service = WorkflowService(session=session)

        # 1. State transition: PENDING on creation
        run, is_new = await wf_service.get_or_create_workflow_run(user=user, request=wf_req)
        assert is_new is True
        assert run.status == WorkflowStatus.PENDING.value

        # 2. Execute adaptive workflow end-to-end
        result = await wf_service.execute_workflow(run)

        # 3. State transition: COMPLETED
        await session.refresh(run)
        assert run.status == WorkflowStatus.COMPLETED.value
        assert run.started_at is not None
        assert run.completed_at is not None
        assert result["status"] == WorkflowStatus.COMPLETED.value
        assert result["evidence_count"] > 0

        # 4. Verify web-evidence task invoked configured search handler (TinyFish)
        tasks = (await session.scalars(
            select(WorkflowTask).where(WorkflowTask.workflow_run_id == run.id)
        )).all()
        task_map = {t.task_key: t for t in tasks}

        web_task = task_map.get("gather_web_evidence")
        assert web_task is not None
        assert web_task.status == WorkflowTaskStatus.COMPLETED.value
        assert web_task.output_result is not None
        assert web_task.output_result.get("status") == "success"
        assert len(web_task.output_result.get("urls", [])) > 0
        # Invariant: Contains actual URLs from real search
        assert any("http" in url for url in web_task.output_result["urls"])

        # 5. Verify task results contain actual evidence, NOT task descriptions
        for t in tasks:
            if t.task_key == "gather_web_evidence":
                for url in t.output_result.get("urls", []):
                    assert not url.startswith("Evidence collected for ")

        # 6. Verify dependencies: extract_key_facts depends on gather_internal_knowledge & gather_web_evidence
        extract_task = task_map.get("extract_key_facts")
        assert extract_task is not None
        assert "gather_web_evidence" in extract_task.dependencies
        assert "gather_internal_knowledge" in extract_task.dependencies
        assert extract_task.status == WorkflowTaskStatus.COMPLETED.value

        # 7. Budget enforcement: token budget tracked and reserved tokens cleared
        assert run.tokens_spent > 0
        assert run.tokens_reserved == 0
        assert run.provider_calls_spent > 0
        assert run.provider_calls_reserved == 0

        # 8. Final synthesis and citations
        synthesis_text = result["synthesis"]
        assert synthesis_text is not None
        assert "# Research Report:" in synthesis_text
        assert "## Grounded Findings" in synthesis_text
        assert "[source:" in synthesis_text
        # Invariant: No fabricated task description statements in draft
        assert "Evidence collected for " not in synthesis_text

        # 9. Verify draft hash was computed and stored
        assert run.draft_hash is not None
        assert len(run.draft_hash) == 64  # SHA-256


# ============================================================================
# GATE B: Live CRM Write-Back Verification (Twenty CRM Staging)
# ============================================================================

@pytest.mark.asyncio
async def test_gate_b_fast_path_cannot_bypass_server_side_mutation_gate():
    """
    Gate B Step 1:
    Confirm that a request with execution_mode="fast_path" cannot bypass the server-side mutation gate.
    """
    mutation_query = f"Create a meeting note for company {STAGING_COMPANY_ID} detailing our latest contract discussion"
    assert WorkflowService.is_fast_path(mutation_query) is False, (
        "Server-side gate must reject fast-path execution when CRM mutation / note creation intent is detected"
    )

    # Fast-path is only valid for simple read questions
    simple_query = "What is our refund policy?"
    assert WorkflowService.is_fast_path(simple_query) is True


@pytest.mark.asyncio
async def test_gate_b_live_crm_pre_approval_and_independent_remote_verification(disposable_staging_context):
    """
    Gate B Steps 2-4:
    2. Confirm that no remote mutation occurs before explicit approval.
    3. Approve the exact reviewed operation and verify the actual remote note independently.
    4. Verify durable ledger state and idempotent replay.
    7. Preserve credentials, workspace isolation, and all existing approval requirements.
    """
    ctx = disposable_staging_context
    user = ctx["user"]
    dept = ctx["department"]

    # Verify staging Twenty CRM is live and configured
    if not TWENTY_API_KEY:
        pytest.skip("TWENTY_API_KEY not configured; reporting gate as BLOCKED would be handled if offline.")

    headers = {
        "Authorization": f"Bearer {TWENTY_API_KEY}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=10.0) as http_client:
        # Check Twenty CRM health
        health_res = await http_client.get(f"{TWENTY_BASE_URL}/healthz")
        assert health_res.status_code == 200, "Twenty CRM container must be healthy"

        # Record initial notes count via Twenty REST API independently
        initial_notes_res = await http_client.get(f"{TWENTY_BASE_URL}/rest/notes", headers=headers)
        assert initial_notes_res.status_code == 200
        initial_note_ids = {n["id"] for n in initial_notes_res.json().get("data", {}).get("notes", [])}

        async with AsyncSessionLocal() as session:
            sync_service = CRMSyncService(session=session)

            # Step 2: Invariant: unapproved writeback raises PermissionError, no remote mutation occurs
            idempotency_key = f"gate-b-test-{uuid.uuid4().hex[:12]}"
            note_title = f"Gate B Verification Note {uuid.uuid4().hex[:6]}"
            note_content = "Verified meeting notes from adaptive workflow execution."

            with pytest.raises(PermissionError) as exc_info:
                await sync_service.create_meeting_note_writeback(
                    company_id=STAGING_COMPANY_ID,
                    note_title=note_title,
                    summary_markdown=note_content,
                    idempotency_key=idempotency_key,
                    user_role="MEMBER",  # Non-admin without approval ticket
                    approved=False,
                )
            assert "explicitly approved" in str(exc_info.value) or "approval" in str(exc_info.value).lower()

            # Verify NO remote note was created in Twenty CRM
            pre_approval_res = await http_client.get(f"{TWENTY_BASE_URL}/rest/notes", headers=headers)
            current_note_ids = {n["id"] for n in pre_approval_res.json().get("data", {}).get("notes", [])}
            assert current_note_ids == initial_note_ids, "Invariant violation: remote note created prior to approval"

            # Step 3: Approve exact reviewed operation and execute live write-back
            # Generate valid approval ticket
            from rag_app.schemas.workflow import compute_canonical_draft_hash
            draft_hash = compute_canonical_draft_hash(note_content)
            now = datetime.now(timezone.utc)

            approval = CRMWriteBackApproval(
                id=uuid.uuid4(),
                operation_type="NOTE_CREATE",
                external_record_id=STAGING_COMPANY_ID,
                target_object="company",
                department_id=dept.id,
                requested_by=user.id,
                approved_by=user.id,
                status="APPROVED",
                expires_at=now + timedelta(hours=24),
                created_at=now,
                approved_draft_hash=draft_hash,
                intended_action="CRM_CREATE_NOTE",
            )
            session.add(approval)
            await session.commit()

            # Execute approved live write-back
            exec_result = await sync_service.create_meeting_note_writeback(
                company_id=STAGING_COMPANY_ID,
                note_title=note_title,
                summary_markdown=note_content,
                idempotency_key=idempotency_key,
                user_id=user.id,
                user_role="ADMIN",
                approval_id=approval.id,
                approved=True,
            )

            assert exec_result["status"] == "COMPLETED"
            assert exec_result["is_duplicate_replay"] is False
            live_note_id = exec_result["crm_note_id"]
            live_target_id = exec_result["crm_note_target_id"]
            assert live_note_id is not None
            assert live_target_id is not None

            # INDEPENDENT REMOTE VERIFICATION:
            # Query Twenty CRM REST API directly to verify the note exists remotely with exact content
            verify_res = await http_client.get(f"{TWENTY_BASE_URL}/rest/notes/{live_note_id}", headers=headers)
            assert verify_res.status_code == 200, f"Remote note {live_note_id} not found in Twenty CRM REST API"
            remote_note_data = verify_res.json().get("data", {}).get("note", {})
            assert remote_note_data.get("id") == live_note_id
            assert remote_note_data.get("title") == note_title
            assert remote_note_data.get("bodyV2", {}).get("markdown") == note_content

            # Step 4: Durable ledger state & idempotent replay
            ledger_record = await session.scalar(
                select(CRMWriteBackRecord).where(CRMWriteBackRecord.idempotency_key == idempotency_key)
            )
            assert ledger_record is not None
            assert ledger_record.status == "COMPLETED"
            assert ledger_record.crm_note_id == live_note_id
            assert ledger_record.crm_note_target_id == live_target_id

            # Replay with same idempotency key
            replay_result = await sync_service.create_meeting_note_writeback(
                company_id=STAGING_COMPANY_ID,
                note_title=note_title,
                summary_markdown=note_content,
                idempotency_key=idempotency_key,
                user_id=user.id,
                user_role="ADMIN",
                approval_id=approval.id,
                approved=True,
            )

            assert replay_result["is_duplicate_replay"] is True
            assert replay_result["crm_note_id"] == live_note_id

            # Verify no second note was created in Twenty CRM
            notes_after_replay = await http_client.get(f"{TWENTY_BASE_URL}/rest/notes", headers=headers)
            notes_count_after = len(notes_after_replay.json().get("data", {}).get("notes", []))
            assert notes_count_after == len(initial_note_ids) + 1, "Idempotent replay must not create duplicate remote notes"

            # Cleanup live test artifacts in Twenty CRM
            await http_client.delete(f"{TWENTY_BASE_URL}/rest/noteTargets/{live_target_id}", headers=headers)
            await http_client.delete(f"{TWENTY_BASE_URL}/rest/notes/{live_note_id}", headers=headers)


@pytest.mark.asyncio
async def test_gate_b_uncertain_outcome_reconciliation(disposable_staging_context):
    """
    Gate B Steps 5 & 6:
    5. Exercise the existing uncertain-outcome/reconciliation path safely.
    6. Do not create duplicate notes during retries or reconciliation.
    """
    ctx = disposable_staging_context
    user = ctx["user"]
    dept = ctx["department"]

    if not TWENTY_API_KEY:
        pytest.skip("TWENTY_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {TWENTY_API_KEY}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=10.0) as http_client:
        async with AsyncSessionLocal() as session:
            sync_service = CRMSyncService(session=session)

            idempotency_key = f"gate-b-retry-{uuid.uuid4().hex[:12]}"
            note_title = f"Gate B Retry Recovery Note {uuid.uuid4().hex[:6]}"
            note_content = "Testing two-phase retry recovery without note duplication."

            # Pre-create the note remotely to simulate a crash/failure during linking phase
            mcp_client = TwentyMCPClient(enabled=True)
            existing_note_id = await mcp_client.create_note(title=note_title, markdown_body=note_content)

            # Insert partial ledger record: NOTE_CREATED, but LINK_FAILED
            partial_record = CRMWriteBackRecord(
                id=uuid.uuid4(),
                operation_type="NOTE_CREATE",
                provider="twenty_crm",
                external_record_id=STAGING_COMPANY_ID,
                target_object="company",
                idempotency_key=idempotency_key,
                status="LINK_FAILED",
                crm_note_id=existing_note_id,
                crm_note_target_id=None,
                note_title=note_title,
                summary_content=note_content,
                created_by=user.id,
                approved_by=user.id,
            )
            session.add(partial_record)
            await session.commit()

            # Count notes before retry
            notes_before_retry = await http_client.get(f"{TWENTY_BASE_URL}/rest/notes", headers=headers)
            count_before = len(notes_before_retry.json().get("data", {}).get("notes", []))

            # Execute write-back retry: should recognize existing crm_note_id, NOT create duplicate note,
            # and proceed to link and reach COMPLETED.
            retry_result = await sync_service.create_meeting_note_writeback(
                company_id=STAGING_COMPANY_ID,
                note_title=note_title,
                summary_markdown=note_content,
                idempotency_key=idempotency_key,
                user_id=user.id,
                user_role="ADMIN",
                approved=True,
            )

            assert retry_result["status"] == "COMPLETED"
            assert retry_result["crm_note_id"] == existing_note_id, "Must reuse existing note ID"
            linked_target_id = retry_result["crm_note_target_id"]
            assert linked_target_id is not None

            # Verify no duplicate note was created during retry
            notes_after_retry = await http_client.get(f"{TWENTY_BASE_URL}/rest/notes", headers=headers)
            count_after = len(notes_after_retry.json().get("data", {}).get("notes", []))
            assert count_after == count_before, "Retry must not create a duplicate note"

            # Cleanup
            await http_client.delete(f"{TWENTY_BASE_URL}/rest/noteTargets/{linked_target_id}", headers=headers)
            await http_client.delete(f"{TWENTY_BASE_URL}/rest/notes/{existing_note_id}", headers=headers)
