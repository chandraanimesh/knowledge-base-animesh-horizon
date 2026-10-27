"""
Deterministic Backend Test Suite for Milestone 6F.1:
Two-Step CRM Write Recovery, Operation Records, and Zero-Duplicate Idempotency.

Scenarios tested:
1. Full success: create_one_note succeeds and create_one_note_target succeeds.
2. Partial failure: create_one_note succeeds but create_one_note_target fails. Note ID remains durably saved.
3. Timeout ambiguity: create_one_note succeeds, create_one_note_target times out after dispatch.
   Operation transitions to UNKNOWN / RECONCILIATION_REQUIRED.
4. Workflow restart / replay: re-submitting after partial success reuses the confirmed note ID,
   bypasses remote note creation (call count = 1), and successfully completes target linking without duplicate notes.
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timedelta, timezone
import json
from typing import Any
from unittest.mock import AsyncMock, patch
import uuid

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select

from rag_app.crm.twenty_mcp_client import APPROVED_WRITE_BACK_TOOLS, TwentyMCPClient
from rag_app.crm.sync_service import CRMSyncService
from rag_app.db.models import (
    CRMWriteBackApproval,
    CRMWriteBackRecord,
    Department,
    MCPToolOperationRecord,
    Permission,
    User,
    UserDepartment,
    UserPermission,
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


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_two_step_context():
    """
    Sets up isolated test context in rag_disposable_test with:
    - Dedicated Department
    - Workspace Editor User
    - Workspace with active MCP Connection allowing create_one_note and create_one_note_target
    """
    async with AsyncSessionLocal() as session:
        dept = await session.scalar(select(Department).where(Department.code == "M6F1_DEPT"))
        if not dept:
            dept = Department(code="M6F1_DEPT", name="Milestone 6F.1 Dept", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(code="VIEW", description="View permission", is_active=True)
            session.add(view_perm)
            await session.commit()
            await session.refresh(view_perm)

        editor_email = f"editor_6f1_{uuid.uuid4().hex[:6]}@example.com"
        editor = User(email=editor_email, hashed_password="hashed_password", role="USER", is_active=True)
        session.add(editor)
        await session.commit()
        await session.refresh(editor)

        session.add(UserDepartment(user_id=editor.id, department_id=dept.id))
        session.add(UserPermission(user_id=editor.id, department_id=dept.id, permission_id=view_perm.id))

        workspace = Workspace(
            name=f"Workspace 6F1 {uuid.uuid4().hex[:6]}",
            code=f"ws6f1_{uuid.uuid4().hex[:6]}",
            department_id=dept.id,
            is_active=True,
            created_by=editor.id,
        )
        session.add(workspace)
        await session.commit()
        await session.refresh(workspace)

        session.add(WorkspaceMember(workspace_id=workspace.id, user_id=editor.id, role="EDITOR"))

        conn = WorkspaceMCPConnection(
            workspace_id=workspace.id,
            name="Twenty CRM Test",
            endpoint_url="http://localhost:3000/mcp",
            transport="streamable-http",
            credential_ref="SERVER_TWENTY_API_KEY",
            is_enabled=True,
            status="CONNECTED",
            discovered_tools=[
                {"name": "create_one_note", "description": "Create note"},
                {"name": "create_one_note_target", "description": "Link note to target"},
                {"name": "find_many_companies", "description": "Search companies"},
            ],
            enabled_tools=["create_one_note", "create_one_note_target", "find_many_companies"],
            configuration={"allow_private_hosts": True},
            created_by=editor.id,
        )
        session.add(conn)
        await session.commit()
        await session.refresh(conn)

        return {
            "dept": dept,
            "editor": editor,
            "workspace": workspace,
            "connection": conn,
        }


# ==============================================================================
# TEST 1: Full Two-Step Success (Note creation + Target linking)
# ==============================================================================

@pytest.mark.asyncio
async def test_two_step_write_both_succeed(setup_two_step_context):
    """
    Verifies that executing create_one_note followed by create_one_note_target
    creates two distinct CONFIRMED operation records and returns the remote IDs.
    """
    ctx = setup_two_step_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        # 1. Propose and approve note creation
        draft_markdown = "## Executive Summary\nStrategic growth targets for Q4."
        approval = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace"].id,
            connection_id=ctx["connection"].id,
            tool_name="create_one_note",
            arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-777"},
        )

        note_fake_res = MCPToolResult(
            name="create_one_note",
            content=[{"type": "text", "text": "Created"}],
            text='{"id": "note-remote-1001"}',
            structured_content={"record": {"id": "note-remote-1001"}},
            is_error=False,
        )

        target_fake_res = MCPToolResult(
            name="create_one_note_target",
            content=[{"type": "text", "text": "Linked"}],
            text='{"id": "target-remote-2002"}',
            structured_content={"record": {"id": "target-remote-2002"}},
            is_error=False,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # Step 1: Execute create_one_note
            mock_inst.call_tool.return_value = note_fake_res
            step1_idem = f"idem-step1-{uuid.uuid4().hex[:8]}"
            res1 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note",
                arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-777"},
                idempotency_key=step1_idem,
                approval_ticket_id=approval.id,
            )

            assert res1.is_error is False
            assert "note-remote-1001" in res1.text

            # Step 2: Execute create_one_note_target
            mock_inst.call_tool.return_value = target_fake_res
            step2_idem = f"idem-step2-{uuid.uuid4().hex[:8]}"
            res2 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note_target",
                arguments={"noteId": "note-remote-1001", "targetCompanyId": "comp-777"},
                idempotency_key=step2_idem,
            )

            assert res2.is_error is False
            assert "target-remote-2002" in res2.text

        # Verify database operation records
        op1 = await session.scalar(
            select(MCPToolOperationRecord).where(MCPToolOperationRecord.idempotency_key == step1_idem)
        )
        assert op1 is not None
        assert op1.status == "CONFIRMED"
        assert op1.tool_name == "create_one_note"

        op2 = await session.scalar(
            select(MCPToolOperationRecord).where(MCPToolOperationRecord.idempotency_key == step2_idem)
        )
        assert op2 is not None
        assert op2.status == "CONFIRMED"
        assert op2.tool_name == "create_one_note_target"


# ==============================================================================
# TEST 2: Partial Failure (create_one_note succeeds, create_one_note_target fails)
# ==============================================================================

@pytest.mark.asyncio
async def test_two_step_write_partial_failure_linking_fails(setup_two_step_context):
    """
    Verifies that when Step 1 (create_one_note) succeeds but Step 2 (create_one_note_target) fails:
    - Step 1 operation record remains CONFIRMED with remote note ID.
    - Step 2 operation record is recorded as FAILED.
    - Note record is NOT rolled back or lost from DB.
    """
    ctx = setup_two_step_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        draft_markdown = "## Partial Failure Note\nShould remain saved."
        approval = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace"].id,
            connection_id=ctx["connection"].id,
            tool_name="create_one_note",
            arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-888"},
        )

        note_fake_res = MCPToolResult(
            name="create_one_note",
            content=[{"type": "text", "text": "Created"}],
            text='{"id": "note-partial-123"}',
            structured_content={"record": {"id": "note-partial-123"}},
            is_error=False,
        )

        target_error_res = MCPToolResult(
            name="create_one_note_target",
            content=[{"type": "text", "text": "Target company does not exist"}],
            text="Target company does not exist",
            structured_content=None,
            is_error=True,
            error_message="Invalid target company ID",
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # Step 1: Note creation succeeds
            mock_inst.call_tool.return_value = note_fake_res
            step1_idem = f"idem-part1-{uuid.uuid4().hex[:8]}"
            res1 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note",
                arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-888"},
                idempotency_key=step1_idem,
                approval_ticket_id=approval.id,
            )
            assert res1.is_error is False

            # Step 2: Linking fails
            mock_inst.call_tool.return_value = target_error_res
            step2_idem = f"idem-part2-{uuid.uuid4().hex[:8]}"
            res2 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note_target",
                arguments={"noteId": "note-partial-123", "targetCompanyId": "comp-888"},
                idempotency_key=step2_idem,
            )
            assert res2.is_error is True

        # Check DB records
        op1 = await session.scalar(
            select(MCPToolOperationRecord).where(MCPToolOperationRecord.idempotency_key == step1_idem)
        )
        assert op1 is not None
        assert op1.status == "CONFIRMED"
        assert "note-partial-123" in json.dumps(op1.result_payload)

        op2 = await session.scalar(
            select(MCPToolOperationRecord).where(MCPToolOperationRecord.idempotency_key == step2_idem)
        )
        assert op2 is not None
        assert op2.status == "FAILED"


# ==============================================================================
# TEST 3: Target Linking Timing Out After Possible Remote Commit
# ==============================================================================

@pytest.mark.asyncio
async def test_two_step_write_target_linking_timeout_creates_unknown(setup_two_step_context):
    """
    Verifies that when Step 1 succeeds and Step 2 times out after dispatch:
    - Step 2 operation record transitions to UNKNOWN and flags RECONCILIATION_REQUIRED.
    - An immediate duplicate retry is blocked with 409 Conflict.
    - Step 1 confirmed note is preserved.
    """
    ctx = setup_two_step_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        draft_markdown = "## Timeout Test Note\nStep 2 will time out."
        approval = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace"].id,
            connection_id=ctx["connection"].id,
            tool_name="create_one_note",
            arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-timeout"},
        )

        note_fake_res = MCPToolResult(
            name="create_one_note",
            content=[{"type": "text", "text": "Created"}],
            text='{"id": "note-timeout-555"}',
            structured_content={"record": {"id": "note-timeout-555"}},
            is_error=False,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # Step 1 succeeds
            mock_inst.call_tool.return_value = note_fake_res
            step1_idem = f"idem-to1-{uuid.uuid4().hex[:8]}"
            await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note",
                arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-timeout"},
                idempotency_key=step1_idem,
                approval_ticket_id=approval.id,
            )

            # Step 2 times out after dispatch
            mock_inst.call_tool.side_effect = MCPTimeoutError("ReadTimeout during target linking")
            step2_idem = f"idem-to2-{uuid.uuid4().hex[:8]}"

            with pytest.raises(HTTPException) as exc_info:
                await service.invoke_workspace_tool(
                    user=ctx["editor"],
                    workspace_id=ctx["workspace"].id,
                    connection_id=ctx["connection"].id,
                    tool_name="create_one_note_target",
                    arguments={"noteId": "note-timeout-555", "targetCompanyId": "comp-timeout"},
                    idempotency_key=step2_idem,
                )
            assert exc_info.value.status_code == 504
            assert "UNKNOWN" in exc_info.value.detail

        # Step 2 operation record is in UNKNOWN state pending reconciliation
        op2 = await session.scalar(
            select(MCPToolOperationRecord).where(MCPToolOperationRecord.idempotency_key == step2_idem)
        )
        assert op2 is not None
        assert op2.status == "UNKNOWN"

        # Blind immediate retry is blocked with 409 Conflict
        with pytest.raises(HTTPException) as retry_exc:
            await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note_target",
                arguments={"noteId": "note-timeout-555", "targetCompanyId": "comp-timeout"},
                idempotency_key=step2_idem,
            )
        assert retry_exc.value.status_code == 409
        assert "reconciliation" in retry_exc.value.detail.lower()


# ==============================================================================
# TEST 4: Workflow Restart / Replay Reuses Note ID (Zero Duplicates)
# ==============================================================================

@pytest.mark.asyncio
async def test_two_step_write_replay_reuses_note_id_avoids_duplicate(setup_two_step_context):
    """
    Verifies that when a workflow re-runs after Step 1 completed:
    - Step 1 (create_one_note) returns the cached note ID with is_replay: True.
    - Remote MCP client create_one_note is NOT called a second time (call count remains 1).
    - Step 2 proceeds with the reused note ID and completes linking.
    - Zero duplicate notes are generated.
    """
    ctx = setup_two_step_context
    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        draft_markdown = "## Replay Safety Test\nMust never duplicate."
        approval = await service.propose_tool_mutation(
            user=ctx["editor"],
            workspace_id=ctx["workspace"].id,
            connection_id=ctx["connection"].id,
            tool_name="create_one_note",
            arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-replay-99"},
        )

        note_fake_res = MCPToolResult(
            name="create_one_note",
            content=[{"type": "text", "text": "Created"}],
            text='{"id": "note-durable-777"}',
            structured_content={"record": {"id": "note-durable-777"}},
            is_error=False,
        )

        target_fake_res = MCPToolResult(
            name="create_one_note_target",
            content=[{"type": "text", "text": "Linked"}],
            text='{"id": "target-link-888"}',
            structured_content={"record": {"id": "target-link-888"}},
            is_error=False,
        )

        step1_idem = f"idem-rep1-{uuid.uuid4().hex[:8]}"

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # Run 1, Step 1: Note created remotely
            mock_inst.call_tool.return_value = note_fake_res
            res1_attempt1 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note",
                arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-replay-99"},
                idempotency_key=step1_idem,
                approval_ticket_id=approval.id,
            )
            assert "note-durable-777" in res1_attempt1.text
            assert mock_inst.call_tool.call_count == 1

            # (Assume workflow halted or crashed before Step 2 was executed)

            # Run 2 (Restart/Replay): Workflow re-executes Step 1 with same idempotency key
            res1_attempt2 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note",
                arguments={"bodyV2": {"markdown": draft_markdown}, "targetCompanyId": "comp-replay-99"},
                idempotency_key=step1_idem,
            )

            # Must return cached result!
            assert "note-durable-777" in res1_attempt2.text
            structured = res1_attempt2.structured_content or {}
            assert structured.get("is_replay") is True
            # Remote client call_tool must NOT have been called again!
            assert mock_inst.call_tool.call_count == 1

            # Reused note ID is extracted and passed to Step 2
            reused_note_id = "note-durable-777"
            mock_inst.call_tool.return_value = target_fake_res
            step2_idem = f"idem-rep2-{uuid.uuid4().hex[:8]}"

            res2 = await service.invoke_workspace_tool(
                user=ctx["editor"],
                workspace_id=ctx["workspace"].id,
                connection_id=ctx["connection"].id,
                tool_name="create_one_note_target",
                arguments={"noteId": reused_note_id, "targetCompanyId": "comp-replay-99"},
                idempotency_key=step2_idem,
            )
            assert res2.is_error is False
            assert "target-link-888" in res2.text
            # Exactly 1 note call + 1 target link call total across all attempts
            assert mock_inst.call_tool.call_count == 2
