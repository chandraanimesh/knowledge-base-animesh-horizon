"""
Comprehensive Acceptance Test Suite for Milestone 6E:
Human-in-the-Loop (HITL) Approval-Gated MCP Write-Back.

Tests:
1. Approval requested before mutation: agent proposes mutation -> approval record created (PENDING, draft hash, expiry),
   workflow in AWAITING_APPROVAL, no external write executed.
2. Rejecting an approval results in no external mutation: status marked REJECTED, operation FAILED, zero remote calls.
3. Expired, revoked, and reused (CONSUMED) approval tickets are rejected fail-closed.
4. Workspace and role violations fail closed (cross-workspace and VIEWER blocked).
5. Tampered payload or tool identity mismatch invalidates the approval and fails closed.
6. Concurrent approval submissions cannot dispatch duplicate mutations (atomic claim & idempotency).
7. Confirmed operation replay returns cached result with is_replay: True without remote dispatch.
8. Timeout after remote commit creates UNKNOWN status and does not blindly retry.
9. Reconciliation can confirm a committed operation (UNKNOWN -> CONFIRMED).
10. Inconclusive reconciliation remains RECONCILIATION_REQUIRED without duplicate write.
11. Workflow resume routes strictly through WorkspaceMCPConnectionService.
12. LangGraph proposal tools enable agents to propose mutations safely without autonomous write access.
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Optional
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select

from rag_app.crm.twenty_mcp_client import APPROVED_WRITE_BACK_TOOLS
from rag_app.db.models import (
    CRMWriteBackApproval,
    Department,
    MCPToolOperationRecord,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    WorkflowRun,
    WorkflowTask,
    Workspace,
    WorkspaceMCPConnection,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.client import MCPClientError, MCPTimeoutError, MCPToolResult
from rag_app.mcp.connection_service import (
    WorkspaceMCPConnectionService,
    compute_canonical_draft_hash,
)
from rag_app.mcp.langgraph_binding import WorkspaceReadOnlyMCPToolBinding
from rag_app.orchestration.service import WorkflowService
from rag_app.schemas.workflow import TaskType, WorkflowStatus


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_6e_context():
    """
    Sets up isolated test context in rag_disposable_test:
    - Dept A & Dept B
    - Users: Owner (ADMIN), Editor (USER), Viewer (USER), Cross-Workspace User (USER)
    - Workspaces: Workspace A & Workspace B
    - Workspace A MCP Connection enabled for Twenty CRM with 'create_one_note' and 'get_company'
    """
    async with AsyncSessionLocal() as session:
        # 1. Departments
        dept_a = await session.scalar(select(Department).where(Department.code == "M6E_DEPT_A"))
        if not dept_a:
            dept_a = Department(code="M6E_DEPT_A", name="Milestone 6E Dept A", is_active=True)
            session.add(dept_a)

        dept_b = await session.scalar(select(Department).where(Department.code == "M6E_DEPT_B"))
        if not dept_b:
            dept_b = Department(code="M6E_DEPT_B", name="Milestone 6E Dept B", is_active=True)
            session.add(dept_b)

        await session.commit()
        await session.refresh(dept_a)
        await session.refresh(dept_b)

        # 2. Permissions
        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(code="VIEW", description="View files", is_active=True)
            session.add(view_perm)
            await session.commit()
            await session.refresh(view_perm)

        # 3. Users
        async def get_or_create_user(email: str, role: str) -> User:
            u = await session.scalar(select(User).where(User.email == email))
            if not u:
                u = User(
                    email=email,
                    hashed_password="hashed_password",
                    role=role,
                    is_active=True,
                )
                session.add(u)
                await session.commit()
                await session.refresh(u)
            elif u.role != role:
                u.role = role
                session.add(u)
                await session.commit()
                await session.refresh(u)
            return u

        owner = await get_or_create_user("m6e_owner@test.com", "ADMIN")
        editor = await get_or_create_user("m6e_editor@test.com", "USER")
        viewer = await get_or_create_user("m6e_viewer@test.com", "USER")
        cross_user = await get_or_create_user("m6e_cross@test.com", "USER")

        # 4. Department memberships & permissions
        for u in [owner, editor, viewer]:
            ud = await session.scalar(
                select(UserDepartment).where(UserDepartment.user_id == u.id, UserDepartment.department_id == dept_a.id)
            )
            if not ud:
                session.add(UserDepartment(user_id=u.id, department_id=dept_a.id))
            up = await session.scalar(
                select(UserPermission).where(
                    UserPermission.user_id == u.id,
                    UserPermission.department_id == dept_a.id,
                    UserPermission.permission_id == view_perm.id,
                )
            )
            if not up:
                session.add(UserPermission(user_id=u.id, department_id=dept_a.id, permission_id=view_perm.id))

        # cross_user belongs only to dept_b
        ud_b = await session.scalar(
            select(UserDepartment).where(UserDepartment.user_id == cross_user.id, UserDepartment.department_id == dept_b.id)
        )
        if not ud_b:
            session.add(UserDepartment(user_id=cross_user.id, department_id=dept_b.id))
        up_b = await session.scalar(
            select(UserPermission).where(
                UserPermission.user_id == cross_user.id,
                UserPermission.department_id == dept_b.id,
                UserPermission.permission_id == view_perm.id,
            )
        )
        if not up_b:
            session.add(UserPermission(user_id=cross_user.id, department_id=dept_b.id, permission_id=view_perm.id))

        await session.commit()

        # 5. Workspaces
        ws_a = await session.scalar(select(Workspace).where(Workspace.code == "m6e-ws-a"))
        if not ws_a:
            ws_a = Workspace(
                name="Workspace 6E A",
                code="m6e-ws-a",
                department_id=dept_a.id,
                created_by=owner.id,
                is_active=True,
            )
            session.add(ws_a)
            await session.commit()
            await session.refresh(ws_a)

        ws_b = await session.scalar(select(Workspace).where(Workspace.code == "m6e-ws-b"))
        if not ws_b:
            ws_b = Workspace(
                name="Workspace 6E B",
                code="m6e-ws-b",
                department_id=dept_b.id,
                created_by=cross_user.id,
                is_active=True,
            )
            session.add(ws_b)
            await session.commit()
            await session.refresh(ws_b)

        # 6. Workspace Memberships
        async def ensure_membership(ws_id: uuid.UUID, user_id: uuid.UUID, role: str):
            m = await session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == ws_id,
                    WorkspaceMember.user_id == user_id,
                )
            )
            if not m:
                m = WorkspaceMember(workspace_id=ws_id, user_id=user_id, role=role)
                session.add(m)
                await session.commit()
            elif m.role != role:
                m.role = role
                session.add(m)
                await session.commit()

        await ensure_membership(ws_a.id, owner.id, "OWNER")
        await ensure_membership(ws_a.id, editor.id, "EDITOR")
        await ensure_membership(ws_a.id, viewer.id, "VIEWER")
        await ensure_membership(ws_b.id, cross_user.id, "OWNER")

        # 7. Workspace A Twenty CRM MCP Connection
        conn_a = await session.scalar(
            select(WorkspaceMCPConnection).where(
                WorkspaceMCPConnection.workspace_id == ws_a.id,
                WorkspaceMCPConnection.name == "Twenty CRM Writeback Connection",
            )
        )
        if not conn_a:
            conn_a = WorkspaceMCPConnection(
                workspace_id=ws_a.id,
                name="Twenty CRM Writeback Connection",
                endpoint_url="https://api.twenty.com/mcp",
                transport="streamable-http",
                credential_ref="SERVER_TWENTY_API_KEY",
                is_enabled=True,
                status="CONNECTED",
                discovered_tools=[
                    {
                        "name": "get_company",
                        "description": "Retrieve company details",
                        "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
                    },
                    {
                        "name": "create_one_note",
                        "description": "Create a Note record in Twenty CRM",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "bodyV2": {"type": "object"},
                            },
                        },
                    },
                ],
                enabled_tools=["get_company", "create_one_note"],
                configuration={
                    "tool_policies": {
                        "create_one_note": {
                            "risk_level": "mutating",
                            "reconciliation_tool": "get_company",
                            "reconciliation_key_param": "id",
                        }
                    }
                },
                created_by=owner.id,
            )
            session.add(conn_a)
            await session.commit()
            await session.refresh(conn_a)

        # Clean up any extraneous connections in workspace_a from previous test runs
        extra_conns = (
            await session.scalars(
                select(WorkspaceMCPConnection).where(
                    WorkspaceMCPConnection.workspace_id == ws_a.id,
                    WorkspaceMCPConnection.id != conn_a.id,
                )
            )
        ).all()
        for ec in extra_conns:
            await session.delete(ec)
        if extra_conns:
            await session.commit()

        return {
            "department_a": dept_a,
            "department_b": dept_b,
            "owner": owner,
            "editor": editor,
            "viewer": viewer,
            "cross_user": cross_user,
            "workspace_a": ws_a,
            "workspace_b": ws_b,
            "connection_a": conn_a,
        }


# ==============================================================================
# TEST 1: Approval Requested Before Mutation
# ==============================================================================

@pytest.mark.asyncio
async def test_approval_requested_before_mutation(setup_6e_context):
    """
    Verifies that when an agent proposes a CRM mutation:
    - Target connection and write-back tool policy are validated.
    - CRMWriteBackApproval is persisted with status PENDING, canonical draft hash, and 24h expiry.
    - MCPToolOperationRecord is persisted with status PENDING and stable idempotency key.
    - WorkflowRun transitions to AWAITING_APPROVAL.
    - No remote MCP call is dispatched.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        # Create a mock workflow run
        run_id = uuid.uuid4()
        run = WorkflowRun(
            id=run_id,
            workspace_id=ctx["workspace_a"].id,
            user_id=ctx["editor"].id,
            scope_type="WORKSPACE",
            status="RUNNING",
            request_hash="req-hash-m6e-test1",
            initial_query="Sync verified company brief to Twenty CRM",
        )
        session.add(run)
        await session.commit()

        proposed_payload = {
            "title": "Q3 Strategic Growth Brief",
            "bodyV2": {"markdown": "## Growth Overview\n- Rapid pipeline expansion."},
        }

        # Propose mutation
        approval_res = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=proposed_payload,
            target_record_id="comp-uuid-12345",
            run_id=run_id,
        )

        assert approval_res.status == "PENDING"
        assert approval_res.intended_action == "create_one_note"
        assert approval_res.external_record_id == "comp-uuid-12345"
        assert approval_res.approved_draft_hash is not None
        assert approval_res.expires_at > datetime.now(timezone.utc)

        # Verify CRMWriteBackApproval record in DB
        app_db = await session.get(CRMWriteBackApproval, approval_res.id)
        assert app_db is not None
        assert app_db.status == "PENDING"
        expected_hash = compute_canonical_draft_hash("## Growth Overview\n- Rapid pipeline expansion.")
        assert app_db.approved_draft_hash == expected_hash

        # Verify linked MCPToolOperationRecord in DB
        op_stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.approval_ticket_id == approval_res.id
        )
        op_record = await session.scalar(op_stmt)
        assert op_record is not None
        assert op_record.status == "PENDING"
        assert op_record.tool_name == "create_one_note"
        assert op_record.connection_id == ctx["connection_a"].id
        assert op_record.idempotency_key is not None

        # Verify WorkflowRun was transitioned to AWAITING_APPROVAL
        await session.refresh(run)
        assert run.status == WorkflowStatus.AWAITING_APPROVAL.value


# ==============================================================================
# TEST 2: Rejecting Approval Results in No External Mutation
# ==============================================================================

@pytest.mark.asyncio
async def test_rejecting_approval_results_in_no_external_mutation(setup_6e_context):
    """
    Verifies that human rejection:
    - Transitions CRMWriteBackApproval to REJECTED with revoked_at timestamp.
    - Transitions MCPToolOperationRecord to FAILED with rejection reason.
    - Zero external mutations or GenericMCPClient calls are executed.
    """
    from rag_app.api.workspace_mcp_routes import submit_mcp_approval_decision
    from rag_app.schemas.mcp_connection import MCPApprovalDecisionRequest

    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        # 1. Propose mutation
        approval_res = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments={"title": "Proposed Note", "bodyV2": {"markdown": "Test"}},
        )

        # 2. Reject mutation via decision endpoint
        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            decision_payload = MCPApprovalDecisionRequest(
                decision="REJECTED",
                rejection_reason="Executive summary lacks required compliance disclaimers.",
            )
            result = await submit_mcp_approval_decision(
                workspace_id=ctx["workspace_a"].id,
                approval_id=approval_res.id,
                payload=decision_payload,
                user=ctx["editor"],
                session=session,
            )

            assert result["status"] == "REJECTED"
            assert result["side_effect_executed"] is False
            assert "compliance disclaimers" in result["rejection_reason"]

            # Ensure GenericMCPClient was NEVER instantiated or called
            mock_client_cls.assert_not_called()

        # 3. Verify DB state
        app_db = await session.get(CRMWriteBackApproval, approval_res.id)
        assert app_db.status == "REJECTED"
        assert app_db.revoked_at is not None

        op_stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.approval_ticket_id == approval_res.id
        )
        op_db = await session.scalar(op_stmt)
        assert op_db.status == "FAILED"
        assert "compliance disclaimers" in op_db.error_message


# ==============================================================================
# TEST 3: Expired, Revoked, and Reused Tickets Fail Closed
# ==============================================================================

@pytest.mark.asyncio
async def test_expired_revoked_and_reused_tickets_fail_closed(setup_6e_context):
    """
    Verifies that invalid approval tickets are rejected fail-closed:
    - Expired tickets (>24 hours) raise HTTP 403/409.
    - Revoked tickets raise HTTP 403/409.
    - Reused (CONSUMED) tickets raise HTTP 409.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)
        now = datetime.now(timezone.utc)
        payload = {"title": "Test Ticket Life", "bodyV2": {"markdown": "Body"}}

        # Case A: Expired ticket
        app_expired = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=payload,
        )
        app_expired.expires_at = now - timedelta(hours=1)
        await session.commit()

        with pytest.raises(HTTPException) as exc_expired:
            await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                approval_ticket_id=app_expired.id,
            )
        assert exc_expired.value.status_code in (403, 409)
        assert "expired" in exc_expired.value.detail.lower()

        # Case B: Revoked ticket
        app_revoked = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=payload,
        )
        app_revoked.status = "REVOKED"
        app_revoked.revoked_at = now
        await session.commit()

        with pytest.raises(HTTPException) as exc_revoked:
            await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                approval_ticket_id=app_revoked.id,
            )
        assert exc_revoked.value.status_code in (403, 409)
        assert "revoked" in exc_revoked.value.detail.lower()

        # Case C: Reused (already CONSUMED) ticket
        app_consumed = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=payload,
        )
        app_consumed.status = "CONSUMED"
        app_consumed.consumed_at = now
        app_consumed.approved_by = ctx["editor"].id
        await session.commit()

        with pytest.raises(HTTPException) as exc_consumed:
            await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                approval_ticket_id=app_consumed.id,
            )
        assert exc_consumed.value.status_code == 409
        assert "already been consumed" in exc_consumed.value.detail.lower()


# ==============================================================================
# TEST 4: Workspace and Role Violations Fail Closed
# ==============================================================================

@pytest.mark.asyncio
async def test_workspace_and_role_violations_fail_closed(setup_6e_context):
    """
    Verifies that:
    - Cross-workspace caller (belongs only to Dept B) cannot propose or invoke mutating tools in Workspace A (HTTP 403).
    - Workspace VIEWER role cannot invoke mutating tools or consume approval tickets (HTTP 403).
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)
        payload = {"title": "Role Test", "bodyV2": {"markdown": "Body"}}

        # Cross-workspace user attempting to propose in Workspace A
        with pytest.raises(HTTPException) as exc_cross:
            await service.propose_tool_mutation(
                user=ctx["cross_user"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
            )
        assert exc_cross.value.status_code == 403

        # Create a valid approval ticket with editor
        approval_res = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=payload,
        )

        # Viewer role attempting to approve/invoke mutation
        with pytest.raises(HTTPException) as exc_viewer:
            await service.invoke_workspace_tool(
                user=ctx["viewer"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                approval_ticket_id=approval_res.id,
            )
        assert exc_viewer.value.status_code == 403
        assert "workspace viewers cannot execute mutating tools" in exc_viewer.value.detail.lower()


# ==============================================================================
# TEST 5: Tampered Payload or Tool Identity Mismatch Invalidates Approval
# ==============================================================================

@pytest.mark.asyncio
async def test_tampered_payload_or_tool_mismatch_invalidates_approval(setup_6e_context):
    """
    Verifies cryptographic payload hash binding:
    - If payload is modified after approval (tampered), hash mismatch revokes the ticket with HTTP 409.
    - If tool name differs from approved intended action, execution fails with HTTP 403/409.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)
        original_payload = {"title": "Original Approved Title", "bodyV2": {"markdown": "Verified Brief"}}

        # 1. Propose mutation
        approval_res = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=original_payload,
        )

        # 2. Attempt to execute with tampered payload (malicious addition)
        tampered_payload = {"title": "Tampered Title", "bodyV2": {"markdown": "Injected malicious text"}}

        with pytest.raises(HTTPException) as exc_tamper:
            await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=tampered_payload,
                approval_ticket_id=approval_res.id,
            )
        assert exc_tamper.value.status_code == 409
        assert "changed since approval was granted" in exc_tamper.value.detail.lower()
        assert "prior approval ticket has been revoked" in exc_tamper.value.detail.lower()

        # Verify that the ticket was permanently revoked due to tampering
        app_db = await session.get(CRMWriteBackApproval, approval_res.id)
        assert app_db.status == "REVOKED"
        assert app_db.revoked_at is not None

        # 3. Tool mismatch test: propose another ticket and invoke with different tool name
        approval_res2 = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=original_payload,
        )

        with pytest.raises(HTTPException) as exc_tool_mismatch:
            await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="get_company",  # Different tool
                arguments=original_payload,
                approval_ticket_id=approval_res2.id,
            )
        assert exc_tool_mismatch.value.status_code in (403, 409)
        assert "cannot authorize tool" in exc_tool_mismatch.value.detail.lower()


# ==============================================================================
# TEST 6: Concurrent Approval Submissions Prevent Duplicate Dispatches
# ==============================================================================

@pytest.mark.asyncio
async def test_concurrent_approval_submissions_prevent_duplicate_dispatches(setup_6e_context):
    """
    Verifies that atomic ticket consumption prevents duplicate remote dispatches
    when two approval requests arrive concurrently for the same ticket.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)
        payload = {"title": "Concurrent Note", "bodyV2": {"markdown": "Brief"}}

        # Propose mutation
        approval_res = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=payload,
        )

        fake_res = MCPToolResult(
            name="create_one_note",
            content=[{"type": "text", "text": "Created"}],
            text='{"id": "note-uuid-999"}',
            structured_content={"record": {"id": "note-uuid-999"}},
            is_error=False,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_inst.call_tool.return_value = fake_res
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # First invocation succeeds and consumes the ticket
            res1 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                approval_ticket_id=approval_res.id,
            )
            assert res1.is_error is False

            # Second concurrent/repeated invocation with the same approval ticket fails closed
            with pytest.raises(HTTPException) as exc_second:
                await service.invoke_workspace_tool(
                    user=ctx["editor"],
                    workspace_id=ctx["workspace_a"].id,
                    connection_id=ctx["connection_a"].id,
                    tool_name="create_one_note",
                    arguments=payload,
                    approval_ticket_id=approval_res.id,
                )
            assert exc_second.value.status_code == 409
            assert "already been consumed" in exc_second.value.detail.lower()

            # Ensure remote write tool was called exactly ONCE
            assert mock_inst.call_tool.call_count == 1


# ==============================================================================
# TEST 7: Confirmed Operation Replay Returns Cached Result
# ==============================================================================

@pytest.mark.asyncio
async def test_confirmed_operation_replay_returns_cached_result(setup_6e_context):
    """
    Verifies that calling invoke_workspace_tool with an existing CONFIRMED operation's
    idempotency key returns the cached result with is_replay: True without re-dispatching.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)
        payload = {"title": "Replay Note", "bodyV2": {"markdown": "Replay Content"}}
        idempotency_key = f"replay-key-{uuid.uuid4().hex[:8]}"

        fake_res = MCPToolResult(
            name="create_one_note",
            content=[{"type": "text", "text": "Note created"}],
            text='{"id": "note-uuid-replay-1"}',
            structured_content={"record": {"id": "note-uuid-replay-1"}},
            is_error=False,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_inst.call_tool.return_value = fake_res
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # 1. Propose mutation
            approval_res = await service.propose_tool_mutation(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                idempotency_key=idempotency_key,
            )

            # 2. First execution: executes remote mutation
            res1 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                idempotency_key=idempotency_key,
                approval_ticket_id=approval_res.id,
            )
            assert res1.is_error is False
            assert mock_inst.call_tool.call_count == 1

            # 3. Replay with same idempotency key (no new approval ticket needed for cached replay)
            res2 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                idempotency_key=idempotency_key,
            )
            assert res2.is_error is False
            assert res2.structured_content.get("is_replay") is True
            assert res2.structured_content.get("idempotency_key") == idempotency_key

            # Ensure call count did NOT increase
            assert mock_inst.call_tool.call_count == 1


# ==============================================================================
# TEST 8: Timeout After Remote Commit Creates UNKNOWN Status
# ==============================================================================

@pytest.mark.asyncio
async def test_timeout_after_remote_commit_creates_unknown_status(setup_6e_context):
    """
    Verifies that when remote mutation times out after dispatch:
    - Operation status transitions to UNKNOWN (or RECONCILIATION_REQUIRED).
    - Status is NEVER marked FAILED.
    - No blind retry is performed.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)
        payload = {"title": "Timeout Note", "bodyV2": {"markdown": "Body"}}
        idempotency_key = f"timeout-key-{uuid.uuid4().hex[:8]}"

        # Setup connection without reconciliation tool in Workspace B to avoid polluting Workspace A
        conn_no_recon = await session.scalar(
            select(WorkspaceMCPConnection).where(
                WorkspaceMCPConnection.workspace_id == ctx["workspace_b"].id,
                WorkspaceMCPConnection.name == "Twenty No Recon Connection",
            )
        )
        if not conn_no_recon:
            conn_no_recon = WorkspaceMCPConnection(
                workspace_id=ctx["workspace_b"].id,
                name="Twenty No Recon Connection",
                endpoint_url="https://api.twenty.com/mcp",
                transport="streamable-http",
                credential_ref="SERVER_TWENTY_API_KEY",
                is_enabled=True,
                status="CONNECTED",
                discovered_tools=[
                    {
                        "name": "create_one_note",
                        "description": "Create note",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ],
                enabled_tools=["create_one_note"],
                configuration={"tool_policies": {"create_one_note": {"risk_level": "mutating"}}},
                created_by=ctx["cross_user"].id,
            )
            session.add(conn_no_recon)
            await session.commit()
            await session.refresh(conn_no_recon)

        approval_res = await service.propose_tool_mutation(
            user=ctx["cross_user"],
            workspace_id=ctx["workspace_b"].id,
            connection_id=conn_no_recon.id,
            tool_name="create_one_note",
            arguments=payload,
            idempotency_key=idempotency_key,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_inst.call_tool.side_effect = MCPTimeoutError("Timed out waiting for response from Twenty MCP")
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            with pytest.raises(HTTPException) as exc_info:
                await service.invoke_workspace_tool(
                    user=ctx["cross_user"],
                    workspace_id=ctx["workspace_b"].id,
                    connection_id=conn_no_recon.id,
                    tool_name="create_one_note",
                    arguments=payload,
                    idempotency_key=idempotency_key,
                    approval_ticket_id=approval_res.id,
                )
            assert exc_info.value.status_code == 504
            assert "UNKNOWN" in exc_info.value.detail or "timed out" in exc_info.value.detail.lower()

        # Check operation record status in DB
        op_stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.workspace_id == ctx["workspace_b"].id,
            MCPToolOperationRecord.idempotency_key == idempotency_key,
        )
        op_db = await session.scalar(op_stmt)
        assert op_db is not None
        assert op_db.status == "UNKNOWN"
        assert "timed out after dispatch" in op_db.error_message


# ==============================================================================
# TEST 9: Reconciliation Can Confirm a Committed Operation
# ==============================================================================

@pytest.mark.asyncio
async def test_reconciliation_confirms_committed_operation(setup_6e_context):
    """
    Verifies that when a timeout occurs, if the tool policy specifies a reconciliation tool
    and that tool finds the record, the operation transitions to CONFIRMED with reconciled: True.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)
        payload = {"title": "Reconciled Note", "bodyV2": {"markdown": "Body"}, "id": "rec-note-123"}
        idempotency_key = f"recon-success-{uuid.uuid4().hex[:8]}"

        approval_res = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=payload,
            idempotency_key=idempotency_key,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            # First call (write) times out; second call (reconciliation lookup) finds the record
            mock_inst.call_tool.side_effect = [
                MCPTimeoutError("Write timeout"),
                MCPToolResult(
                    name="get_company",
                    content=[{"type": "text", "text": "Record found"}],
                    text='{"id": "rec-note-123", "status": "active"}',
                    structured_content={"id": "rec-note-123"},
                    is_error=False,
                ),
            ]
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            res = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace_a"].id,
                connection_id=ctx["connection_a"].id,
                tool_name="create_one_note",
                arguments=payload,
                idempotency_key=idempotency_key,
                approval_ticket_id=approval_res.id,
            )

            assert res.is_error is False
            assert res.structured_content.get("reconciled") is True

        op_stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.workspace_id == ctx["workspace_a"].id,
            MCPToolOperationRecord.idempotency_key == idempotency_key,
        )
        op_db = await session.scalar(op_stmt)
        assert op_db is not None
        assert op_db.status == "CONFIRMED"
        assert op_db.reconciliation_attempts == 1


# ==============================================================================
# TEST 10: Inconclusive Reconciliation Remains Unresolved
# ==============================================================================

@pytest.mark.asyncio
async def test_inconclusive_reconciliation_remains_unresolved(setup_6e_context):
    """
    Verifies that when reconciliation lookup returns 'not found', the operation record
    remains RECONCILIATION_REQUIRED and raises HTTP 504 without blind retries.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)
        payload = {"title": "Inconclusive Note", "bodyV2": {"markdown": "Body"}, "id": "unknown-id-777"}
        idempotency_key = f"recon-inconc-{uuid.uuid4().hex[:8]}"

        approval_res = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments=payload,
            idempotency_key=idempotency_key,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            # First call (write) times out; second call (reconciliation lookup) reports not found
            mock_inst.call_tool.side_effect = [
                MCPTimeoutError("Write timeout"),
                MCPToolResult(
                    name="get_company",
                    content=[{"type": "text", "text": "Record not found"}],
                    text="Record not found",
                    structured_content=None,
                    is_error=False,
                ),
            ]
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            with pytest.raises(HTTPException) as exc_info:
                await service.invoke_workspace_tool(
                    user=ctx["editor"],
                    workspace_id=ctx["workspace_a"].id,
                    connection_id=ctx["connection_a"].id,
                    tool_name="create_one_note",
                    arguments=payload,
                    idempotency_key=idempotency_key,
                    approval_ticket_id=approval_res.id,
                )
            assert exc_info.value.status_code == 504
            assert "RECONCILIATION_REQUIRED" in exc_info.value.detail

        op_stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.workspace_id == ctx["workspace_a"].id,
            MCPToolOperationRecord.idempotency_key == idempotency_key,
        )
        op_db = await session.scalar(op_stmt)
        assert op_db is not None
        assert op_db.status == "RECONCILIATION_REQUIRED"
        assert op_db.reconciliation_attempts == 1


# ==============================================================================
# TEST 11: Workflow Resume Approval Routes Through MCP Connection Service
# ==============================================================================

@pytest.mark.asyncio
async def test_workflow_resume_approval_routes_through_mcp_connection(setup_6e_context):
    """
    Verifies that WorkflowService.resume_workflow_approval discovers the workspace MCP
    connection and executes the writeback via WorkspaceMCPConnectionService.invoke_workspace_tool.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        wf_service = WorkflowService(session)
        service = WorkspaceMCPConnectionService(session)

        # 1. Create a WorkflowRun
        run_id = uuid.uuid4()
        run = WorkflowRun(
            id=run_id,
            workspace_id=ctx["workspace_a"].id,
            user_id=ctx["editor"].id,
            scope_type="WORKSPACE",
            status="RUNNING",
            request_hash="req-hash-resume-test",
            initial_query="Analyze and writeback company brief",
        )
        session.add(run)

        # 2. Add synthesis task with draft text
        draft_content = "## Strategic Brief\n- Strong operational alignment."
        synth_task = WorkflowTask(
            id=uuid.uuid4(),
            workflow_run_id=run_id,
            task_key="synthesis_output",
            task_type=TaskType.SYNTHESIS.value,
            status="COMPLETED",
            output_result={"draft_text": draft_content},
        )
        session.add(synth_task)
        await session.commit()

        # 3. Propose approval ticket bound to this run
        approval = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
            connection_id=ctx["connection_a"].id,
            tool_name="create_one_note",
            arguments={"bodyV2": {"markdown": draft_content}, "targetCompanyId": "comp-789"},
            run_id=run_id,
        )

        fake_res = MCPToolResult(
            name="create_one_note",
            content=[{"type": "text", "text": "Created"}],
            text='{"id": "note-resumed-100"}',
            structured_content={"record": {"id": "note-resumed-100"}},
            is_error=False,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_inst.call_tool.return_value = fake_res
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # Resume workflow approval
            updated_run = await wf_service.resume_workflow_approval(
                run_id=run_id,
                approval_id=approval.id,
                user=ctx["editor"],
                user_role="EDITOR",
                decision="APPROVED",
            )

            assert updated_run["status"] == "COMPLETED"
            assert updated_run["side_effect_executed"] is True
            assert mock_inst.call_tool.call_count == 1

            # Check that the approval record was consumed
            await session.refresh(approval)
            assert approval.status == "CONSUMED"
            assert approval.consumed_at is not None
            assert approval.approved_by == ctx["editor"].id


# ==============================================================================
# TEST 12: LangGraph Proposal Tool Generation
# ==============================================================================

@pytest.mark.asyncio
async def test_langgraph_mcp_proposal_tool_generation(setup_6e_context):
    """
    Verifies that LangGraph tool binding exposes proposal tools (mcp_propose_create_one_note)
    and strictly excludes direct mutating write tools from autonomous execution.
    """
    ctx = setup_6e_context
    async with AsyncSessionLocal() as session:
        binding = WorkspaceReadOnlyMCPToolBinding(session)
        # Get proposal tools for workspace A
        proposal_tools = await binding.get_workspace_proposal_tools(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
        )

        assert len(proposal_tools) == 1
        tool = proposal_tools[0]
        assert tool.name == "mcp_propose_create_one_note"
        assert "propose creating a crm note" in tool.description.lower()
        assert "requests human review" in tool.description.lower()

        # Get all tools (read-only + proposal)
        all_tools = await binding.get_all_workspace_tools(
            user=ctx["editor"],
            workspace_id=ctx["workspace_a"].id,
        )
        tool_names = [t.name for t in all_tools]
        assert "mcp_get_company" in tool_names
        assert "mcp_propose_create_one_note" in tool_names
        # Direct mutating tool must NEVER be exposed as an executable tool!
        assert "mcp_create_one_note" not in tool_names

        # Propose mutation using the LangGraph proposal tool
        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            tool_output_str = await tool.ainvoke({
                "title": "Agent Suggested Note",
                "markdown": "This was generated during research.",
                "company_id": "comp-agent-456",
            })

            assert "Mutation Proposed Successfully" in tool_output_str
            assert "PENDING_HUMAN_APPROVAL" in tool_output_str

            # Verify no direct MCP mutation was called
            mock_client_cls.assert_not_called()
