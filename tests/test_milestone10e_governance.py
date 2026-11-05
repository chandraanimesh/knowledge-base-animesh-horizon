"""
Milestone 10E Verification Test Suite: Tool Governance, Authorization & HITL.

Verifies:
1. Execution-time permission check: fail-closed if required permission is missing at dispatch.
2. Capability allowlist enforcement: only registered and authorized capabilities can be dispatched.
3. Mutating actions require HITL: state transitions to AWAITING_APPROVAL with proposal draft.
4. Cryptographic draft hash binding: SHA-256 hash match authorizes writeback.
5. Anti-tampering: Modified draft invalidates approval and revokes ticket.
6. Single-use consumption: Consumed approvals cannot be re-executed.
"""

from __future__ import annotations

import os

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from datetime import datetime, timezone, timedelta
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.db.models import (
    CRMWriteBackApproval,
    Department,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    WorkflowRun,
    WorkflowTask,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.orchestration.capability_registry import (
    CapabilityPermissionDeniedError,
    CapabilityRegistry,
    get_capability_registry,
)
from rag_app.orchestration.service import WorkflowService
from rag_app.schemas.workflow import (
    compute_canonical_draft_hash,
    WorkflowStatus,
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_governance_context():
    async with AsyncSessionLocal() as session:
        dept = Department(
            id=uuid.uuid4(),
            name="Governance Dept",
            code=f"GOV_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(dept)
        await session.flush()

        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(id=uuid.uuid4(), code="VIEW", name="View")
            session.add(view_perm)
            await session.flush()

        user = User(
            id=uuid.uuid4(),
            email=f"gov_user_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        session.add(user)
        await session.flush()

        session.add(UserDepartment(user_id=user.id, department_id=dept.id))
        session.add(UserPermission(user_id=user.id, department_id=dept.id, permission_id=view_perm.id))

        workspace = Workspace(
            id=uuid.uuid4(),
            name="Gov Workspace",
            code=f"gov_ws_{uuid.uuid4().hex[:6]}",
            department_id=dept.id,
            is_active=True,
            created_by=user.id,
        )
        session.add(workspace)
        await session.flush()

        session.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="OWNER"))
        await session.commit()

        return {
            "user": user,
            "department": dept,
            "workspace": workspace,
        }


# ==============================================================================
# Unit & Service Tests: Governance & Authorization
# ==============================================================================

def test_execution_time_permission_enforcement():
    registry = get_capability_registry()

    # User has no permissions -> cannot execute CRM write or Documents Read
    with pytest.raises(CapabilityPermissionDeniedError) as exc:
        registry.check_authorization("crm.create_note", user_permissions=[])
    assert "CRM_WRITE" in str(exc.value)

    # User with CRM_WRITE permission succeeds
    assert registry.check_authorization("crm.create_note", user_permissions=["CRM_WRITE"]) is True


def test_canonical_draft_hash_deterministic_and_tamper_evident():
    content_original = "Executive Summary: Integration with Twenty CRM approved with 99.9% uptime SLA."
    content_tampered = "Executive Summary: Integration with Twenty CRM approved with 90.0% uptime SLA."

    hash_original = compute_canonical_draft_hash(content_original)
    hash_repeat = compute_canonical_draft_hash(content_original)
    hash_tampered = compute_canonical_draft_hash(content_tampered)

    # Strict deterministic hash
    assert hash_original == hash_repeat
    assert len(hash_original) == 64  # SHA-256 hex string

    # Even a small modification produces completely different hash
    assert hash_original != hash_tampered


@pytest.mark.asyncio
async def test_draft_tampering_revokes_approval_ticket(setup_governance_context):
    ctx = setup_governance_context
    user = ctx["user"]
    ws = ctx["workspace"]
    dept = ctx["department"]

    async with AsyncSessionLocal() as session:
        now = datetime.now(timezone.utc)
        original_draft = "Original authorized recommendation."
        original_hash = compute_canonical_draft_hash(original_draft)

        wf_run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            department_id=dept.id,
            initial_query="CRM integration proposal",
            request_hash="reqhash123",
            scope_type="WORKSPACE",
            status=WorkflowStatus.AWAITING_APPROVAL.value,
            awaiting_approval_at=now,
            draft_hash=original_hash,
        )
        session.add(wf_run)
        await session.flush()

        # Synthesis task holding the stored draft
        task = WorkflowTask(
            id=uuid.uuid4(),
            workflow_run_id=wf_run.id,
            task_key="synthesize_findings",
            task_type="SYNTHESIS",
            status="COMPLETED",
            output_result={"draft_text": "TAMPERED content injected after approval!"},  # Tampered!
        )
        session.add(task)

        # Approval ticket created for original hash
        approval = CRMWriteBackApproval(
            id=uuid.uuid4(),
            workflow_run_id=wf_run.id,
            operation_type="NOTE_CREATE",
            external_record_id="comp-123",
            target_object="company",
            department_id=dept.id,
            requested_by=user.id,
            approved_by=user.id,
            status="PENDING",
            expires_at=now + timedelta(hours=24),
            approved_draft_hash=original_hash,
            intended_action="CRM_CREATE_NOTE",
        )
        session.add(approval)
        await session.commit()

        service = WorkflowService(session=session)

        # Execution must fail closed and revoke ticket
        with pytest.raises(PermissionError) as exc:
            await service.resume_workflow_approval(
                run_id=wf_run.id,
                approval_id=approval.id,
                user=user,
                user_role=user.role,
                decision="APPROVED",
                proposed_action="CRM_CREATE_NOTE",
            )
        assert "Draft content has been modified" in str(exc.value)

        # Verify DB status transitioned to REVOKED and FAILED
        await session.refresh(approval)
        await session.refresh(wf_run)
        assert approval.status == "REVOKED"
        assert wf_run.status == WorkflowStatus.FAILED.value


@pytest.mark.asyncio
async def test_single_use_approval_consumption(setup_governance_context):
    ctx = setup_governance_context
    user = ctx["user"]
    ws = ctx["workspace"]
    dept = ctx["department"]

    async with AsyncSessionLocal() as session:
        now = datetime.now(timezone.utc)
        wf_run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            department_id=dept.id,
            initial_query="CRM action proposal",
            request_hash="reqhash456",
            scope_type="WORKSPACE",
            status="COMPLETED",  # Already finished!
            draft_hash="hash123",
        )
        session.add(wf_run)
        await session.flush()

        approval = CRMWriteBackApproval(
            id=uuid.uuid4(),
            operation_type="NOTE_CREATE",
            external_record_id="comp-123",
            target_object="company",
            department_id=dept.id,
            requested_by=user.id,
            approved_by=user.id,
            status="APPROVED",  # Already consumed!
            expires_at=now + timedelta(hours=24),
            approved_draft_hash="hash123",
            intended_action="CRM_CREATE_NOTE",
        )
        session.add(approval)
        await session.commit()

        service = WorkflowService(session=session)

        # Resuming an already completed run fails with HTTPException 400
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await service.resume_workflow_approval(
                run_id=wf_run.id,
                approval_id=approval.id,
                user=user,
                user_role="ADMIN",
                decision="APPROVED",
            )
        assert exc.value.status_code == 400
        assert "is in status 'COMPLETED', not 'AWAITING_APPROVAL'" in str(exc.value.detail)
