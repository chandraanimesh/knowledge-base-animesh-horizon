"""
Milestone 6C.2 Verification Suite:
MCP Security Hardening, Reliable External Write Outcomes, Task Identity, and Bounded Recovery.

Covers:
- Part A: Tool Risk Classification and Authorization
- Part B: Credential Reference and Endpoint Security
- Part C: Reliable Handling of Ambiguous External Writes (Durable State Machine & Reconciliation)
- Part E: DAG Task Identity, Bounded Execution, Fencing & Recovery, Observability
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import uuid
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
import httpx2
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from rag_app.crm.twenty_mcp_client import READ_ONLY_ALLOWED_TOOLS
from rag_app.db.models import (
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
from rag_app.mcp.client import (
    GenericMCPClient,
    MCPClientConfig,
    MCPSSRFError,
    MCPTimeoutError,
    MCPToolResult,
    MCPTransportType,
)
from rag_app.mcp.connection_service import (
    APPROVED_MCP_CREDENTIAL_REFS,
    VERIFIED_READ_ONLY_MCP_TOOLS,
    WorkspaceMCPConnectionService,
)
from rag_app.orchestration.executor import (
    BudgetExceededError,
    DAGExecutionEngine,
    FencingConflictError,
)
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    ExecutionTask,
    TaskExecutionResult,
    TaskType,
    WorkflowTaskStatus,
)



@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_security_context():
    """
    Sets up a clean workspace with Owner, Editor, Viewer, and Non-Member users.
    """
    async with AsyncSessionLocal() as session:
        dept = await session.scalar(select(Department).where(Department.code == "MCP_SEC_DEPT"))
        if not dept:
            dept = Department(code="MCP_SEC_DEPT", name="MCP Security Dept", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        async def get_or_create_user(email: str, role: str) -> User:
            u = await session.scalar(select(User).where(User.email == email))
            if not u:
                u = User(email=email, hashed_password="pw", role=role, is_active=True)
                session.add(u)
                await session.commit()
                await session.refresh(u)
            return u

        owner = await get_or_create_user("owner_sec@example.com", "USER")
        editor = await get_or_create_user("editor_sec@example.com", "USER")
        viewer = await get_or_create_user("viewer_sec@example.com", "USER")
        outsider = await get_or_create_user("outsider_sec@example.com", "USER")
        admin = await get_or_create_user("admin_sec@example.com", "ADMIN")

        # 3. Department permissions
        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(code="VIEW", description="View files", is_active=True)
            session.add(view_perm)
            await session.commit()
            await session.refresh(view_perm)

        for u in [owner, editor, viewer, admin]:
            ud = await session.scalar(
                select(UserDepartment).where(UserDepartment.user_id == u.id, UserDepartment.department_id == dept.id)
            )
            if not ud:
                session.add(UserDepartment(user_id=u.id, department_id=dept.id))
            up = await session.scalar(
                select(UserPermission).where(
                    UserPermission.user_id == u.id,
                    UserPermission.department_id == dept.id,
                    UserPermission.permission_id == view_perm.id,
                )
            )
            if not up:
                session.add(UserPermission(user_id=u.id, department_id=dept.id, permission_id=view_perm.id))
        await session.commit()

        ws = await session.scalar(select(Workspace).where(Workspace.code == "WS_MCP_SEC"))
        if not ws:
            ws = Workspace(
                name="MCP Security WS",
                code="WS_MCP_SEC",
                department_id=dept.id,
                created_by=owner.id,
                is_active=True,
            )
            session.add(ws)
            await session.commit()
            await session.refresh(ws)

        # Clear existing members and connections for pristine state
        await session.execute(delete(WorkspaceMCPConnection).where(WorkspaceMCPConnection.workspace_id == ws.id))
        await session.execute(delete(MCPToolOperationRecord).where(MCPToolOperationRecord.workspace_id == ws.id))
        await session.execute(delete(WorkspaceMember).where(WorkspaceMember.workspace_id == ws.id))

        m_owner = WorkspaceMember(workspace_id=ws.id, user_id=owner.id, role="OWNER")
        m_editor = WorkspaceMember(workspace_id=ws.id, user_id=editor.id, role="EDITOR")
        m_viewer = WorkspaceMember(workspace_id=ws.id, user_id=viewer.id, role="VIEWER")
        session.add_all([m_owner, m_editor, m_viewer])
        await session.commit()

        yield {
            "workspace": ws,
            "owner": owner,
            "editor": editor,
            "viewer": viewer,
            "outsider": outsider,
            "admin": admin,
        }


# ==============================================================================
# PART A TESTS: Tool Risk Classification and Authorization
# ==============================================================================

@pytest.mark.asyncio
async def test_part_a_viewer_blocked_from_mutating_and_destructive_tools(setup_security_context):
    """
    Requirements:
    - VIEWER cannot execute a mutating or destructive tool merely because it is allowlisted.
    - Enabling a tool in the allowlist must not override caller authorization.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    viewer = ctx["viewer"]
    owner = ctx["owner"]

    async with AsyncSessionLocal() as session:
        svc = WorkspaceMCPConnectionService(session)

        # Create connection with mutating & destructive tools in tool_policies
        conn = WorkspaceMCPConnection(
            workspace_id=ws.id,
            name="Test Remote Tools",
            endpoint_url="https://api.twenty.com/mcp",
            transport="streamable-http",
            is_enabled=True,
            status="CONNECTED",
            discovered_tools=[
                {"name": "create_task", "description": "Mutating task create"},
                {"name": "delete_all", "description": "Destructive wipe"},
                {"name": "get_company", "description": "Read company"},
            ],
            enabled_tools=["create_task", "delete_all", "get_company"],
            configuration={
                "tool_policies": {
                    "create_task": {"risk_level": "mutating"},
                    "delete_all": {"risk_level": "destructive"},
                }
            },
            created_by=owner.id,
        )
        session.add(conn)
        await session.commit()
        await session.refresh(conn)

        # 1. VIEWER calling mutating tool 'create_task' must be rejected (403)
        with pytest.raises(HTTPException) as exc_info:
            await svc.invoke_workspace_tool(
                user=viewer,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="create_task",
                arguments={"name": "test"},
            )
        assert exc_info.value.status_code == 403
        assert "Workspace Viewers cannot execute mutating tools" in exc_info.value.detail

        # 2. VIEWER calling destructive tool 'delete_all' must be rejected (403)
        with pytest.raises(HTTPException) as exc_info:
            await svc.invoke_workspace_tool(
                user=viewer,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="delete_all",
                arguments={},
            )
        assert exc_info.value.status_code == 403
        assert "classified as destructive" in exc_info.value.detail


@pytest.mark.asyncio
async def test_part_a_unverified_tool_downgrade_to_read_only_is_rejected(setup_security_context):
    """
    Requirement 3:
    A workspace-configurable risk_level="read_only" must not automatically downgrade
    an unknown or potentially mutating tool to read-only.
    Unclassified tools fail closed.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    viewer = ctx["viewer"]
    owner = ctx["owner"]

    async with AsyncSessionLocal() as session:
        svc = WorkspaceMCPConnectionService(session)

        # Attempt to declare an arbitrary unknown mutating tool as 'read_only' in config
        conn = WorkspaceMCPConnection(
            workspace_id=ws.id,
            name="Untrusted Tool Risk Override",
            endpoint_url="https://api.twenty.com/mcp",
            transport="streamable-http",
            is_enabled=True,
            status="CONNECTED",
            discovered_tools=[
                {"name": "execute_shell", "description": "Arbitrary remote exec"},
                {"name": "unclassified_tool", "description": "No policy"},
            ],
            enabled_tools=["execute_shell", "unclassified_tool"],
            configuration={
                "tool_policies": {
                    "execute_shell": {"risk_level": "read_only"},  # Malicious downgrade attempt
                }
            },
            created_by=owner.id,
        )
        session.add(conn)
        await session.commit()
        await session.refresh(conn)

        # 1. Downgrade rejected: VIEWER calling 'execute_shell' must STILL be forbidden (403)
        with pytest.raises(HTTPException) as exc_info:
            await svc.invoke_workspace_tool(
                user=viewer,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="execute_shell",
                arguments={"cmd": "whoami"},
            )
        assert exc_info.value.status_code == 403
        assert "Workspace Viewers cannot execute mutating tools" in exc_info.value.detail

        # 2. Unclassified tool: fails closed for VIEWER!
        with pytest.raises(HTTPException) as exc_info:
            await svc.invoke_workspace_tool(
                user=viewer,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="unclassified_tool",
                arguments={},
            )
        assert exc_info.value.status_code == 403
        assert "Workspace Viewers cannot execute mutating tools" in exc_info.value.detail


@pytest.mark.asyncio
async def test_part_a_authorized_read_only_access_succeeds(setup_security_context):
    """
    Existing authorized read-only tools (in VERIFIED_READ_ONLY_MCP_TOOLS) are accessible to VIEWERS.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    viewer = ctx["viewer"]
    owner = ctx["owner"]

    async with AsyncSessionLocal() as session:
        svc = WorkspaceMCPConnectionService(session)

        conn = WorkspaceMCPConnection(
            workspace_id=ws.id,
            name="Read Only Connection",
            endpoint_url="https://api.twenty.com/mcp",
            transport="streamable-http",
            is_enabled=True,
            status="CONNECTED",
            discovered_tools=[{"name": "get_company", "description": "Read company"}],
            enabled_tools=["get_company"],
            configuration={},
            created_by=owner.id,
        )
        session.add(conn)
        await session.commit()
        await session.refresh(conn)

        fake_res = MCPToolResult(
            name="get_company",
            content=[{"type": "text", "text": "Company Acme"}],
            text="Company Acme",
            structured_content={"company_name": "Acme"},
            is_error=False,
        )

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_inst.call_tool.return_value = fake_res
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            res = await svc.invoke_workspace_tool(
                user=viewer,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="get_company",
                arguments={"id": "comp-123"},
            )

            assert res.tool_name == "get_company"
            assert res.is_error is False
            assert "Acme" in res.text


# ==============================================================================
# PART B TESTS: Credential Reference and Endpoint Security
# ==============================================================================

@pytest.mark.asyncio
async def test_part_b_credential_reference_host_binding_and_spoof_prevention():
    """
    Requirements:
    - Arbitrary environment variables are rejected.
    - An approved credential cannot be forwarded to an untrusted host (e.g. attacker-twenty.com).
    - Subdomain suffix checks must strictly enforce boundary (dot required).
    """
    # 1. Arbitrary environment variables rejected
    with pytest.raises(HTTPException) as exc_info:
        WorkspaceMCPConnectionService._validate_credential_ref("DATABASE_URL")
    assert exc_info.value.status_code == 400
    assert "not approved for MCP use" in exc_info.value.detail

    # 2. Approved reference SERVER_TWENTY_API_KEY
    assert WorkspaceMCPConnectionService._validate_credential_ref("SERVER_TWENTY_API_KEY") == "SERVER_TWENTY_API_KEY"

    # 3. Host binding verification
    # Allowed: twenty.com, api.twenty.com, localhost
    WorkspaceMCPConnectionService._verify_credential_endpoint_compatibility(
        "SERVER_TWENTY_API_KEY", "https://api.twenty.com/mcp"
    )
    WorkspaceMCPConnectionService._verify_credential_endpoint_compatibility(
        "SERVER_TWENTY_API_KEY", "https://twenty.com/mcp"
    )

    # Spoofed host: attacker-twenty.com must be REJECTED
    with pytest.raises(HTTPException) as exc_info:
        WorkspaceMCPConnectionService._verify_credential_endpoint_compatibility(
            "SERVER_TWENTY_API_KEY", "https://attacker-twenty.com/mcp"
        )
    assert exc_info.value.status_code == 400
    assert "cannot be bound to endpoint host 'attacker-twenty.com'" in exc_info.value.detail

    # Untrusted domain: evil.com must be REJECTED
    with pytest.raises(HTTPException) as exc_info:
        WorkspaceMCPConnectionService._verify_credential_endpoint_compatibility(
            "SERVER_TWENTY_API_KEY", "https://evil.com/mcp"
        )
    assert exc_info.value.status_code == 400


# ==============================================================================
# PART C TESTS: Ambiguous External Write Outcomes & Reconciliation
# ==============================================================================

@pytest.mark.asyncio
async def test_part_c_mutating_tool_timeout_persists_unknown_and_prevents_blind_retry(setup_security_context):
    """
    Requirements:
    - A remote mutating tool timeout must NEVER be treated as confirmed failure.
    - Must persist UNKNOWN durable state.
    - A subsequent call with the same idempotency key must not blindly retry.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    editor = ctx["editor"]

    async with AsyncSessionLocal() as session:
        svc = WorkspaceMCPConnectionService(session)

        conn = WorkspaceMCPConnection(
            workspace_id=ws.id,
            name="CRM Task Service",
            endpoint_url="https://api.twenty.com/mcp",
            transport="streamable-http",
            is_enabled=True,
            status="CONNECTED",
            discovered_tools=[{"name": "create_crm_task", "description": "Create CRM Task"}],
            enabled_tools=["create_crm_task"],
            configuration={
                "tool_policies": {
                    "create_crm_task": {"risk_level": "mutating"}
                }
            },
            created_by=editor.id,
        )
        session.add(conn)
        await session.commit()
        await session.refresh(conn)

        idempotency_key = f"task-write-{uuid.uuid4().hex[:8]}"

        # Mock timeout during remote execution
        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_inst.call_tool.side_effect = MCPTimeoutError("Timed out after 10s")
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # Execution times out
            with pytest.raises(HTTPException) as exc_info:
                await svc.invoke_workspace_tool(
                    user=editor,
                    workspace_id=ws.id,
                    connection_id=conn.id,
                    tool_name="create_crm_task",
                    arguments={"title": "Follow up with Acme"},
                    idempotency_key=idempotency_key,
                )
            assert exc_info.value.status_code == 504
            assert "Outcome is UNKNOWN" in exc_info.value.detail

        # Check durable database state: MUST be 'UNKNOWN', NOT 'FAILED'!
        op_stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.workspace_id == ws.id,
            MCPToolOperationRecord.idempotency_key == idempotency_key,
        )
        op_record = await session.scalar(op_stmt)
        assert op_record is not None
        assert op_record.status == "UNKNOWN"
        assert "timed out after dispatch" in op_record.error_message

        # Attempting second execution with same idempotency key: MUST FAIL with 409 Conflict (no blind retry!)
        with pytest.raises(HTTPException) as exc_info2:
            await svc.invoke_workspace_tool(
                user=editor,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="create_crm_task",
                arguments={"title": "Follow up with Acme"},
                idempotency_key=idempotency_key,
            )
        assert exc_info2.value.status_code == 409
        assert "Blind retry or duplicate execution prevented" in exc_info2.value.detail


@pytest.mark.asyncio
async def test_part_c_reconciliation_lookup_confirms_timed_out_operation(setup_security_context):
    """
    Requirements:
    - If a reconciliation tool is configured and lookup finds the created record,
      the operation transitions UNKNOWN -> CONFIRMED.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    editor = ctx["editor"]

    async with AsyncSessionLocal() as session:
        svc = WorkspaceMCPConnectionService(session)

        conn = WorkspaceMCPConnection(
            workspace_id=ws.id,
            name="CRM Reconciled Service",
            endpoint_url="https://api.twenty.com/mcp",
            transport="streamable-http",
            is_enabled=True,
            status="CONNECTED",
            discovered_tools=[
                {"name": "create_crm_task", "description": "Create task"},
                {"name": "get_crm_task", "description": "Lookup task"},
            ],
            enabled_tools=["create_crm_task", "get_crm_task"],
            configuration={
                "tool_policies": {
                    "create_crm_task": {
                        "risk_level": "mutating",
                        "reconciliation_tool": "get_crm_task",
                        "reconciliation_key_param": "id",
                    },
                    "get_crm_task": {"risk_level": "read_only"},
                }
            },
            created_by=editor.id,
        )
        session.add(conn)
        await session.commit()
        await session.refresh(conn)

        idempotency_key = f"task-recon-{uuid.uuid4().hex[:8]}"

        # Mock: first call times out; reconciliation call returns confirmed task
        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            # First call_tool (create) times out; second call_tool (reconciliation lookup) succeeds
            mock_inst.call_tool.side_effect = [
                MCPTimeoutError("Write call timed out"),
                MCPToolResult(
                    name="get_crm_task",
                    content=[{"type": "text", "text": "Task #999: Follow up with Acme"}],
                    text="Task #999: Follow up with Acme",
                    structured_content={"task_id": "999", "status": "OPEN"},
                    is_error=False,
                ),
            ]
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            res = await svc.invoke_workspace_tool(
                user=editor,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="create_crm_task",
                arguments={"id": "999", "title": "Follow up with Acme"},
                idempotency_key=idempotency_key,
            )

            assert res.is_error is False
            assert res.structured_content.get("reconciled") is True

        # Check durable database state: was transitioned to CONFIRMED by reconciliation!
        op_stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.workspace_id == ws.id,
            MCPToolOperationRecord.idempotency_key == idempotency_key,
        )
        op_record = await session.scalar(op_stmt)
        assert op_record is not None
        assert op_record.status == "CONFIRMED"
        assert op_record.reconciliation_attempts == 1


@pytest.mark.asyncio
async def test_part_c_reconciliation_lookup_empty_persists_reconciliation_required(setup_security_context):
    """
    Requirement 6:
    An immediate lookup returning no record must not automatically trigger a retry.
    Must persist RECONCILIATION_REQUIRED, never FAILED.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    editor = ctx["editor"]

    async with AsyncSessionLocal() as session:
        svc = WorkspaceMCPConnectionService(session)

        conn = WorkspaceMCPConnection(
            workspace_id=ws.id,
            name="CRM Eventual Consistency Service",
            endpoint_url="https://api.twenty.com/mcp",
            transport="streamable-http",
            is_enabled=True,
            status="CONNECTED",
            discovered_tools=[
                {"name": "create_crm_task", "description": "Create task"},
                {"name": "get_crm_task", "description": "Lookup task"},
            ],
            enabled_tools=["create_crm_task", "get_crm_task"],
            configuration={
                "tool_policies": {
                    "create_crm_task": {
                        "risk_level": "mutating",
                        "reconciliation_tool": "get_crm_task",
                        "reconciliation_key_param": "id",
                    },
                    "get_crm_task": {"risk_level": "read_only"},
                }
            },
            created_by=editor.id,
        )
        session.add(conn)
        await session.commit()
        await session.refresh(conn)

        idempotency_key = f"task-eventual-{uuid.uuid4().hex[:8]}"

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_inst.call_tool.side_effect = [
                MCPTimeoutError("Write call timed out"),
                MCPToolResult(
                    name="get_crm_task",
                    content=[],
                    text="Not found",
                    structured_content=None,
                    is_error=False,
                ),
            ]
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            with pytest.raises(HTTPException) as exc_info:
                await svc.invoke_workspace_tool(
                    user=editor,
                    workspace_id=ws.id,
                    connection_id=conn.id,
                    tool_name="create_crm_task",
                    arguments={"id": "not-found-yet"},
                    idempotency_key=idempotency_key,
                )
            assert exc_info.value.status_code == 504
            assert "RECONCILIATION_REQUIRED" in exc_info.value.detail

        op_stmt = select(MCPToolOperationRecord).where(
            MCPToolOperationRecord.workspace_id == ws.id,
            MCPToolOperationRecord.idempotency_key == idempotency_key,
        )
        op_record = await session.scalar(op_stmt)
        assert op_record is not None
        assert op_record.status == "RECONCILIATION_REQUIRED"
        assert op_record.reconciliation_attempts == 1


@pytest.mark.asyncio
async def test_part_c_idempotent_replay_returns_cached_confirmed_result(setup_security_context):
    """
    Requirement 9:
    Repeated requests correlated with the original operation record return
    the confirmed result without re-executing remote side effects.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    editor = ctx["editor"]

    async with AsyncSessionLocal() as session:
        svc = WorkspaceMCPConnectionService(session)

        conn = WorkspaceMCPConnection(
            workspace_id=ws.id,
            name="CRM Replay Service",
            endpoint_url="https://api.twenty.com/mcp",
            transport="streamable-http",
            is_enabled=True,
            status="CONNECTED",
            discovered_tools=[{"name": "create_crm_task", "description": "Create task"}],
            enabled_tools=["create_crm_task"],
            configuration={
                "tool_policies": {
                    "create_crm_task": {"risk_level": "mutating"}
                }
            },
            created_by=editor.id,
        )
        session.add(conn)
        await session.commit()
        await session.refresh(conn)

        idempotency_key = f"task-replay-{uuid.uuid4().hex[:8]}"

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_inst.call_tool.return_value = MCPToolResult(
                name="create_crm_task",
                content=[{"type": "text", "text": "Task #777 created"}],
                text="Task #777 created",
                structured_content={"task_id": "777"},
                is_error=False,
            )
            mock_client_cls.return_value.__aenter__.return_value = mock_inst

            # 1. First invocation
            res1 = await svc.invoke_workspace_tool(
                user=editor,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="create_crm_task",
                arguments={"title": "Original write"},
                idempotency_key=idempotency_key,
            )
            assert res1.is_error is False
            assert "Task #777 created" in res1.text
            assert mock_inst.call_tool.call_count == 1

            # 2. Second invocation with same idempotency key: MUST return cached result without calling remote tool again
            res2 = await svc.invoke_workspace_tool(
                user=editor,
                workspace_id=ws.id,
                connection_id=conn.id,
                tool_name="create_crm_task",
                arguments={"title": "Duplicate write"},
                idempotency_key=idempotency_key,
            )
            assert res2.is_error is False
            assert "Task #777 created" in res2.text
            assert res2.structured_content.get("is_replay") is True
            # call_count remains 1! No second remote call!
            assert mock_inst.call_tool.call_count == 1


# ==============================================================================
# PART E TESTS: DAG Task Identity, Bounded Execution, Recovery & Observability
# ==============================================================================

@pytest.mark.asyncio
async def test_part_e_dag_replay_reuses_completed_tasks_without_re_executing(setup_security_context):
    """
    Requirement E1:
    Every workflow task must have a stable persisted identity.
    Replaying or resuming the same workflow must determine whether work has already
    completed before re-executing it.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    owner = ctx["owner"]

    async with AsyncSessionLocal() as session:
        # Create a WorkflowRun
        run = WorkflowRun(
            user_id=owner.id,
            workspace_id=ws.id,
            scope_type="WORKSPACE",
            status="PENDING",
            request_hash="test-req-hash-1",
            initial_query="Test Task Identity Plan",
            max_tokens_budget=10000,
            max_provider_calls_budget=10,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        # Plan with 2 tasks: task_a -> task_b
        plan = AdaptiveTaskPlan(
            goal="Test Task Identity Plan",
            tasks=[
                ExecutionTask(
                    task_key="task_a",
                    task_type=TaskType.RETRIEVAL,
                    description="Initial retrieval",
                    estimated_tokens=50,
                    estimated_provider_calls=1,
                    dependencies=[],
                ),
                ExecutionTask(
                    task_key="task_b",
                    task_type=TaskType.SYNTHESIS,
                    description="Synthesize findings",
                    estimated_tokens=100,
                    estimated_provider_calls=1,
                    dependencies=["task_a"],
                ),
            ]
        )

        task_a_call_count = 0
        task_b_call_count = 0

        async def handler_task_a(task, sess, r):
            nonlocal task_a_call_count
            task_a_call_count += 1
            return TaskExecutionResult(
                task_key="task_a",
                status=WorkflowTaskStatus.COMPLETED,
                output_result={"found": 5},
                tokens_spent=50,
                provider_calls_spent=1,
            )

        async def handler_task_b(task, sess, r):
            nonlocal task_b_call_count
            task_b_call_count += 1
            return TaskExecutionResult(
                task_key="task_b",
                status=WorkflowTaskStatus.COMPLETED,
                output_result={"summary": "all good"},
                tokens_spent=100,
                provider_calls_spent=1,
            )

        engine1 = DAGExecutionEngine(
            session=session,
            max_concurrency=2,
            task_handlers={
                TaskType.RETRIEVAL.value: handler_task_a,
                TaskType.SYNTHESIS.value: handler_task_b,
            },
        )

        # First execution: runs both tasks
        results1 = await engine1.execute_plan(plan, run)
        assert results1["task_a"].status == WorkflowTaskStatus.COMPLETED
        assert results1["task_b"].status == WorkflowTaskStatus.COMPLETED
        assert task_a_call_count == 1
        assert task_b_call_count == 1

        # Second execution (Replay on same run):
        # Must detect that both tasks already completed and reuse their results!
        engine2 = DAGExecutionEngine(
            session=session,
            max_concurrency=2,
            task_handlers={
                TaskType.RETRIEVAL.value: handler_task_a,
                TaskType.SYNTHESIS.value: handler_task_b,
            },
        )
        results2 = await engine2.execute_plan(plan, run)
        assert results2["task_a"].status == WorkflowTaskStatus.COMPLETED
        assert results2["task_b"].status == WorkflowTaskStatus.COMPLETED

        # Handlers MUST NOT have been called again!
        assert task_a_call_count == 1
        assert task_b_call_count == 1


@pytest.mark.asyncio
async def test_part_e_stale_task_lease_recovery(setup_security_context):
    """
    Requirement E4:
    A stale RUNNING task whose lease has expired is recovered to READY and claimed
    with an incremented fencing token.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    owner = ctx["owner"]

    async with AsyncSessionLocal() as session:
        run = WorkflowRun(
            user_id=owner.id,
            workspace_id=ws.id,
            scope_type="WORKSPACE",
            status="PENDING",
            request_hash="test-req-hash-2",
            initial_query="Test Lease Recovery",
            max_tokens_budget=10000,
            max_provider_calls_budget=10,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        # Pre-seed a task in RUNNING state with EXPIRED lease (simulating a dead worker)
        expired_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        stale_task = WorkflowTask(
            workflow_run_id=run.id,
            task_key="stale_task_1",
            task_type="retrieval",
            status="RUNNING",
            dependencies=[],
            locked_by="dead-worker-1",
            locked_at=expired_time - timedelta(minutes=5),
            lease_expires_at=expired_time,
            fencing_token=1,
        )
        session.add(stale_task)
        await session.commit()

        plan = AdaptiveTaskPlan(
            goal="Test Stale Task Recovery",
            tasks=[
                ExecutionTask(
                    task_key="stale_task_1",
                    task_type=TaskType.RETRIEVAL,
                    description="Stale task needing recovery",
                    estimated_tokens=20,
                    estimated_provider_calls=1,
                    dependencies=[],
                )
            ]
        )

        recovered_ran = False

        async def handler(task, sess, r):
            nonlocal recovered_ran
            recovered_ran = True
            return TaskExecutionResult(
                task_key="stale_task_1",
                status=WorkflowTaskStatus.COMPLETED,
                output_result={"recovered": True},
            )

        dag_engine = DAGExecutionEngine(
            session=session,
            max_concurrency=2,
            task_handlers={TaskType.RETRIEVAL.value: handler},
        )

        res = await dag_engine.execute_plan(plan, run)
        assert recovered_ran is True
        assert res["stale_task_1"].status == WorkflowTaskStatus.COMPLETED

        # Verify DB record was updated with incremented fencing token (>= 2)
        await session.refresh(stale_task)
        assert stale_task.status == "COMPLETED"
        assert stale_task.fencing_token >= 2


@pytest.mark.asyncio
async def test_part_e_per_run_budget_enforcement(setup_security_context):
    """
    Requirement E2:
    Enforce explicit per-run limits on external calls/tokens.
    Exceeding limits must raise BudgetExceededError and reject execution.
    """
    ctx = setup_security_context
    ws = ctx["workspace"]
    owner = ctx["owner"]

    async with AsyncSessionLocal() as session:
        run = WorkflowRun(
            user_id=owner.id,
            workspace_id=ws.id,
            scope_type="WORKSPACE",
            status="PENDING",
            request_hash="test-req-hash-3",
            initial_query="Test Budget Enforcement",
            max_tokens_budget=100,  # Strict token budget
            max_provider_calls_budget=2,  # Strict call budget
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        # Plan that requests 500 tokens (exceeding 100 limit)
        greedy_plan = AdaptiveTaskPlan(
            goal="Test Budget Plan",
            tasks=[
                ExecutionTask(
                    task_key="greedy_task",
                    task_type=TaskType.SYNTHESIS,
                    description="Greedy task exceeding token budget",
                    estimated_tokens=500,  # Exceeds 100!
                    estimated_provider_calls=1,
                    dependencies=[],
                )
            ]
        )

        dag_engine = DAGExecutionEngine(session=session, max_concurrency=2)
        results = await dag_engine.execute_plan(greedy_plan, run)

        assert results["greedy_task"].status == WorkflowTaskStatus.FAILED
        assert "Budget reservation rejected" in results["greedy_task"].error_message or "exceed" in results["greedy_task"].error_message.lower()
