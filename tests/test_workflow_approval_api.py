"""
API Integration & Regression Test Suite for Workflow Run Progress & Approval API.
Milestone 4D verification:
1. GET /api/v1/workflows/runs/{run_id} returns enriched draft, canonical hash, evidence, and task details.
2. POST /api/v1/workflows/runs/{run_id}/approval with valid APPROVED decision resumes workflow, verifies hash, and completes execution.
3. POST /api/v1/workflows/runs/{run_id}/approval with REJECTED decision records audit trail, transitions to REJECTED, and does not execute side effects.
4. Tampered or stale draft_hash is rejected with HTTP 409 Conflict.
5. Non-authorized / cross-workspace user cannot view or approve the run (HTTP 403 Forbidden).
6. Duplicate approval attempts handle retries safely without duplicate side effects.
"""

from __future__ import annotations

import os

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

from rag_app.db.models import (
    CRMWriteBackApproval,
    CRMWriteBackRecord,
    Department,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
    WorkflowRun,
    WorkflowTask,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.main import app
from rag_app.schemas.workflow import (
    WorkflowStatus,
    WorkflowTaskStatus,
    TaskType,
    compute_canonical_draft_hash,
)
from rag_app.security.authentication import create_access_token


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_approval_context():
    """
    Sets up a department, workspace, authorized user + token, and an unauthorized user.
    """
    async with AsyncSessionLocal() as session:
        dept = Department(
            id=uuid.uuid4(),
            name="Approval Test Dept",
            code=f"APPR_DEPT_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(dept)
        await session.flush()

        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(id=uuid.uuid4(), code="VIEW", name="View Permission")
            session.add(view_perm)
            await session.flush()

        user = User(
            id=uuid.uuid4(),
            email=f"appr_user_{uuid.uuid4().hex[:6]}@example.com",
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
            name="Approval Workspace",
            code=f"appr_ws_{uuid.uuid4().hex[:6]}",
            department_id=dept.id,
            is_active=True,
            created_by=user.id,
        )
        session.add(workspace)
        await session.flush()

        session.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="OWNER"))

        # Unauthorized user in a different department and workspace
        other_dept = Department(
            id=uuid.uuid4(),
            name="Other Department",
            code=f"OTHER_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(other_dept)
        await session.flush()

        other_user = User(
            id=uuid.uuid4(),
            email=f"other_user_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        session.add(other_user)
        await session.flush()

        other_ws = Workspace(
            id=uuid.uuid4(),
            name="Other Workspace",
            code=f"other_ws_{uuid.uuid4().hex[:6]}",
            department_id=other_dept.id,
            is_active=True,
            created_by=other_user.id,
        )
        session.add(other_ws)
        await session.flush()
        session.add(WorkspaceMember(workspace_id=other_ws.id, user_id=other_user.id, role="OWNER"))

        await session.commit()
        await session.refresh(user)
        await session.refresh(other_user)
        await session.refresh(workspace)
        await session.refresh(other_ws)

        token = create_access_token(subject=str(user.id))
        other_token = create_access_token(subject=str(other_user.id))

        return {
            "dept": dept,
            "user": user,
            "workspace": workspace,
            "token": token,
            "headers": {"Authorization": f"Bearer {token}"},
            "other_user": other_user,
            "other_workspace": other_ws,
            "other_headers": {"Authorization": f"Bearer {other_token}"},
        }


async def _create_awaiting_approval_run(session, ctx, draft_text: str = "Quarterly strategic review grounded draft.") -> tuple[WorkflowRun, CRMWriteBackApproval]:
    canonical_hash = compute_canonical_draft_hash(draft_text)
    now = datetime.now(timezone.utc)

    run = WorkflowRun(
        id=uuid.uuid4(),
        user_id=ctx["user"].id,
        workspace_id=ctx["workspace"].id,
        department_id=ctx["dept"].id,
        initial_query="Generate strategic briefing note for CRM writeback",
        status=WorkflowStatus.AWAITING_APPROVAL.value,
        request_hash="test-req-hash",
        draft_hash=canonical_hash,
        awaiting_approval_at=now,
        max_tokens_budget=10000,
        tokens_spent=250,
        tokens_reserved=0,
        max_provider_calls_budget=20,
        provider_calls_spent=3,
        provider_calls_reserved=0,
    )
    session.add(run)
    await session.flush()

    # Research Task with evidence
    res_task = WorkflowTask(
        id=uuid.uuid4(),
        workflow_run_id=run.id,
        task_key="task-research-1",
        task_type=TaskType.RETRIEVAL.value,
        status=WorkflowTaskStatus.COMPLETED.value,
        output_result={
            "evidence": [
                {
                    "content": "Acme Corp revenue grew 25% year over year.",
                    "source": "10-Q Filing Q3",
                    "source_id": "doc-sec-10q",
                    "score": 0.94,
                }
            ]
        },
    )
    session.add(res_task)

    # Synthesis Task with authoritative draft
    synth_task = WorkflowTask(
        id=uuid.uuid4(),
        workflow_run_id=run.id,
        task_key="task-synthesis-1",
        task_type=TaskType.SYNTHESIS.value,
        status=WorkflowTaskStatus.COMPLETED.value,
        output_result={"draft_text": draft_text},
    )
    session.add(synth_task)

    # CRMWriteBackApproval ticket
    approval = CRMWriteBackApproval(
        id=uuid.uuid4(),
        workflow_run_id=run.id,
        operation_type="NOTE_CREATE",
        target_object="company",
        external_record_id="company-acme-123",
        intended_action="CRM_CREATE_NOTE",
        status="APPROVED",
        approved_draft_hash=canonical_hash,
        department_id=ctx["dept"].id,
        requested_by=ctx["user"].id,
        approved_by=ctx["user"].id,
        expires_at=now + timedelta(hours=24),
        created_at=now,
    )
    session.add(approval)
    await session.commit()
    await session.refresh(run)
    await session.refresh(approval)

    return run, approval


@pytest.mark.asyncio
async def test_get_workflow_run_awaiting_approval_details(setup_approval_context):
    """
    Phase 2: GET /api/v1/workflows/runs/{run_id} returns authoritative draft, canonical hash,
    and evidence citations when status is AWAITING_APPROVAL.
    """
    ctx = setup_approval_context
    draft_content = "Strategic analysis: Target company exhibits sustained cash flow growth."

    async with AsyncSessionLocal() as session:
        run, approval = await _create_awaiting_approval_run(session, ctx, draft_content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(f"/api/v1/workflows/runs/{run.id}", headers=ctx["headers"])
        assert res.status_code == 200, res.text
        data = res.json()

        assert data["id"] == str(run.id)
        assert data["status"] == "AWAITING_APPROVAL"
        assert data["draft_text"] == draft_content
        assert data["draft_hash"] == compute_canonical_draft_hash(draft_content)
        assert data["approval_id"] == str(approval.id)
        assert data["intended_action"] == "CRM_CREATE_NOTE"
        assert data["target_record_id"] == "company-acme-123"
        assert len(data["tasks"]) == 2
        assert len(data["evidence_items"]) >= 1
        item = data["evidence_items"][0]
        text_val = item.get("content") or item.get("snippet", "")
        assert "Acme Corp revenue grew 25%" in text_val


@pytest.mark.asyncio
async def test_workflow_approval_valid_approved(setup_approval_context):
    """
    Phase 3: Valid human approval verifies canonical draft hash, executes idempotent side effect,
    and transitions workflow to COMPLETED.
    """
    ctx = setup_approval_context
    draft_content = "Authoritative strategic memo for CRM update."

    async with AsyncSessionLocal() as session:
        run, approval = await _create_awaiting_approval_run(session, ctx, draft_content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Mock Twenty CRM MCP note creation
        with (
            patch("rag_app.crm.twenty_mcp_client.TwentyMCPClient.create_note", new_callable=AsyncMock) as mock_crm,
            patch("rag_app.crm.twenty_mcp_client.TwentyMCPClient.create_note_target_for_company", new_callable=AsyncMock) as mock_target,
        ):
            mock_crm.return_value = "note-created-999"
            mock_target.return_value = "target-created-999"

            approval_payload = {
                "decision": "APPROVED",
                "draft_hash": compute_canonical_draft_hash(draft_content),
                "proposed_action": "CRM_CREATE_NOTE",
                "proposed_target": "company-acme-123",
            }
            res = await client.post(
                f"/api/v1/workflows/runs/{run.id}/approval",
                json=approval_payload,
                headers=ctx["headers"],
            )
            assert res.status_code == 200, res.text
            data = res.json()
            assert data["workflow_run_id"] == str(run.id)
            assert data["status"] == "COMPLETED"
            assert data["side_effect_executed"] is True
            assert mock_crm.await_count == 1

    # Verify DB state
    async with AsyncSessionLocal() as session:
        db_run = await session.get(WorkflowRun, run.id)
        assert db_run.status == "COMPLETED"
        assert db_run.approved_at is not None
        assert db_run.completed_at is not None

        db_appr = await session.get(CRMWriteBackApproval, approval.id)
        assert db_appr.status == "CONSUMED"


@pytest.mark.asyncio
async def test_workflow_approval_tampered_draft_hash_conflict(setup_approval_context):
    """
    Phase 3: Tampered or stale draft hash is rejected server-side with HTTP 409 Conflict.
    """
    ctx = setup_approval_context
    draft_content = "Legitimate grounded draft text."

    async with AsyncSessionLocal() as session:
        run, approval = await _create_awaiting_approval_run(session, ctx, draft_content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        tampered_payload = {
            "decision": "APPROVED",
            "draft_hash": compute_canonical_draft_hash("tampered different content"),
            "proposed_action": "CRM_CREATE_NOTE",
            "proposed_target": "company-acme-123",
        }
        res = await client.post(
            f"/api/v1/workflows/runs/{run.id}/approval",
            json=tampered_payload,
            headers=ctx["headers"],
        )
        assert res.status_code == 409, res.text
        assert "Draft hash does not match" in res.json()["detail"]

    # Verify DB state was NOT transitioned
    async with AsyncSessionLocal() as session:
        db_run = await session.get(WorkflowRun, run.id)
        assert db_run.status == "AWAITING_APPROVAL"


@pytest.mark.asyncio
async def test_workflow_approval_rejection_with_reason(setup_approval_context):
    """
    Phase 3: Rejection records human reviewer reason, transitions to REJECTED,
    and executes ZERO side effects.
    """
    ctx = setup_approval_context
    draft_content = "Draft requiring revision."

    async with AsyncSessionLocal() as session:
        run, approval = await _create_awaiting_approval_run(session, ctx, draft_content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("rag_app.crm.twenty_mcp_client.TwentyMCPClient.create_note", new_callable=AsyncMock) as mock_crm:
            reject_payload = {
                "decision": "REJECTED",
                "draft_hash": compute_canonical_draft_hash(draft_content),
                "rejection_reason": "Financial ratios are misaligned with newest 10-Q filing.",
            }
            res = await client.post(
                f"/api/v1/workflows/runs/{run.id}/approval",
                json=reject_payload,
                headers=ctx["headers"],
            )
            assert res.status_code == 200, res.text
            data = res.json()
            assert data["workflow_run_id"] == str(run.id)
            assert data["status"] == "REJECTED"
            assert data["side_effect_executed"] is False
            assert mock_crm.await_count == 0

    # Verify DB state
    async with AsyncSessionLocal() as session:
        db_run = await session.get(WorkflowRun, run.id)
        assert db_run.status == "REJECTED"
        assert "Financial ratios are misaligned" in (db_run.error_message or "")

        db_appr = await session.get(CRMWriteBackApproval, approval.id)
        assert db_appr.status == "REJECTED"


@pytest.mark.asyncio
async def test_workflow_approval_cross_workspace_forbidden(setup_approval_context):
    """
    Phase 3: Cross-workspace access restriction: User from another workspace receives 403 Forbidden.
    """
    ctx = setup_approval_context
    draft_content = "Confidential Workspace Memo."

    async with AsyncSessionLocal() as session:
        run, approval = await _create_awaiting_approval_run(session, ctx, draft_content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET access forbidden
        get_res = await client.get(
            f"/api/v1/workflows/runs/{run.id}",
            headers=ctx["other_headers"],
        )
        assert get_res.status_code == 403, get_res.text

        # POST approval forbidden
        post_res = await client.post(
            f"/api/v1/workflows/runs/{run.id}/approval",
            json={"decision": "APPROVED", "draft_hash": compute_canonical_draft_hash(draft_content)},
            headers=ctx["other_headers"],
        )
        assert post_res.status_code == 403, post_res.text


@pytest.mark.asyncio
async def test_workflow_approval_expired_ticket_rejected(setup_approval_context):
    """
    Phase 4C Security: Expired approval ticket is rejected and transitioned to EXPIRED.
    """
    ctx = setup_approval_context
    draft_content = "Draft with expired approval ticket."

    async with AsyncSessionLocal() as session:
        run, approval = await _create_awaiting_approval_run(session, ctx, draft_content)
        # Force expiration timestamp into the past
        approval.expires_at = datetime.now(timezone.utc) - timedelta(minutes=15)
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            f"/api/v1/workflows/runs/{run.id}/approval",
            json={
                "decision": "APPROVED",
                "draft_hash": compute_canonical_draft_hash(draft_content),
            },
            headers=ctx["headers"],
        )
        assert res.status_code == 403, res.text
        assert "expired" in res.json()["detail"].lower()

    async with AsyncSessionLocal() as session:
        db_run = await session.get(WorkflowRun, run.id)
        assert db_run.status == "EXPIRED"


@pytest.mark.asyncio
async def test_workflow_approval_already_consumed_rejected(setup_approval_context):
    """
    Phase 4C Security: Single-use consumption safeguard - already completed or consumed tickets cannot be re-approved.
    """
    ctx = setup_approval_context
    draft_content = "Memo for single-use test."

    async with AsyncSessionLocal() as session:
        run, approval = await _create_awaiting_approval_run(session, ctx, draft_content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with (
            patch("rag_app.crm.twenty_mcp_client.TwentyMCPClient.create_note", new_callable=AsyncMock) as mock_crm,
            patch("rag_app.crm.twenty_mcp_client.TwentyMCPClient.create_note_target_for_company", new_callable=AsyncMock) as mock_target,
        ):
            mock_crm.return_value = "note-first-pass"
            mock_target.return_value = "target-first-pass"

            # 1. First approval succeeds
            first_res = await client.post(
                f"/api/v1/workflows/runs/{run.id}/approval",
                json={
                    "decision": "APPROVED",
                    "draft_hash": compute_canonical_draft_hash(draft_content),
                },
                headers=ctx["headers"],
            )
            assert first_res.status_code == 200

            # 2. Duplicate submission must be rejected (cannot consume twice)
            second_res = await client.post(
                f"/api/v1/workflows/runs/{run.id}/approval",
                json={
                    "decision": "APPROVED",
                    "draft_hash": compute_canonical_draft_hash(draft_content),
                },
                headers=ctx["headers"],
            )
            assert second_res.status_code == 400
            assert "AWAITING_APPROVAL" in second_res.json()["detail"]
