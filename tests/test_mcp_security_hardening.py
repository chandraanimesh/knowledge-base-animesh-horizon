"""
Milestone 6C.1 Security Hardening Test Suite:
1. Credential Reference Allowlist Enforcement (rejecting arbitrary process secrets like DATABASE_URL).
2. Endpoint-Credential Binding & Incompatibility Rejection.
3. Secret Confidentiality (secrets never in responses, logs, or exceptions).
4. Role-Based Invocation Authorization:
   - VIEWER can execute read-only tools but is blocked from mutating tools.
   - EDITOR/OWNER can execute mutating tools.
   - Destructive operations require OWNER/ADMIN.
5. Endpoint Revalidation & Cross-Origin Redirect Prevention.
"""

from __future__ import annotations

import os

# Guarantee execution strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from typing import Any, Optional
import uuid

from fastmcp import FastMCP
from fastapi import HTTPException
import httpx2
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from rag_app.db.models import Department, Permission, User, UserDepartment, UserPermission, Workspace, WorkspaceMCPConnection, WorkspaceMember
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.client import GenericMCPClient, MCPClientConfig, MCPServerInfo, MCPTransportType
from rag_app.mcp.connection_service import APPROVED_MCP_CREDENTIAL_REFS, WorkspaceMCPConnectionService
from rag_app.schemas.mcp_connection import (
    MCPConnectionCreate,
    MCPConnectionResponse,
    MCPConnectionUpdate,
    MCPToolAllowlistUpdate,
    MCPToolInvokeRequest,
    MCPToolInvokeResponse,
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_hardening_context():
    """
    Sets up isolated workspaces with OWNER, EDITOR, and VIEWER members.
    """
    async with AsyncSessionLocal() as session:
        # 1. Department
        dept = await session.scalar(select(Department).where(Department.code == "MCP_HARDEN_DEPT"))
        if not dept:
            dept = Department(code="MCP_HARDEN_DEPT", name="MCP Harden Dept", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        dept_b = await session.scalar(select(Department).where(Department.code == "MCP_HARDEN_DEPT_B"))
        if not dept_b:
            dept_b = Department(code="MCP_HARDEN_DEPT_B", name="MCP Harden Dept B", is_active=True)
            session.add(dept_b)
            await session.commit()
            await session.refresh(dept_b)

        # 2. Users
        async def get_or_create_user(email: str, role: str) -> User:
            u = await session.scalar(select(User).where(User.email == email))
            if not u:
                u = User(email=email, hashed_password="pw", role=role, is_active=True)
                session.add(u)
                await session.commit()
                await session.refresh(u)
            return u

        admin = await get_or_create_user("harden_admin@test.com", "ADMIN")
        owner = await get_or_create_user("harden_owner@test.com", "USER")
        editor = await get_or_create_user("harden_editor@test.com", "USER")
        viewer = await get_or_create_user("harden_viewer@test.com", "USER")
        user_b = await get_or_create_user("harden_user_b@test.com", "USER")

        # 3. Department permissions
        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(code="VIEW", description="View files", is_active=True)
            session.add(view_perm)
            await session.commit()
            await session.refresh(view_perm)

        for u in [owner, editor, viewer]:
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

        # User B in Dept B
        ud_b = await session.scalar(
            select(UserDepartment).where(UserDepartment.user_id == user_b.id, UserDepartment.department_id == dept_b.id)
        )
        if not ud_b:
            session.add(UserDepartment(user_id=user_b.id, department_id=dept_b.id))
        up_b = await session.scalar(
            select(UserPermission).where(
                UserPermission.user_id == user_b.id,
                UserPermission.department_id == dept_b.id,
                UserPermission.permission_id == view_perm.id,
            )
        )
        if not up_b:
            session.add(UserPermission(user_id=user_b.id, department_id=dept_b.id, permission_id=view_perm.id))
        await session.commit()

        # 4. Workspaces
        ws_a = await session.scalar(select(Workspace).where(Workspace.code == "WS_HARDEN_A"))
        if not ws_a:
            ws_a = Workspace(name="Workspace Harden A", code="WS_HARDEN_A", department_id=dept.id, created_by=owner.id, is_active=True)
            session.add(ws_a)
            await session.commit()
            await session.refresh(ws_a)

        ws_b = await session.scalar(select(Workspace).where(Workspace.code == "WS_HARDEN_B"))
        if not ws_b:
            ws_b = Workspace(name="Workspace Harden B", code="WS_HARDEN_B", department_id=dept_b.id, created_by=user_b.id, is_active=True)
            session.add(ws_b)
            await session.commit()
            await session.refresh(ws_b)

        # 5. Memberships
        async def ensure_member(ws_id: uuid.UUID, u_id: uuid.UUID, role: str):
            m = await session.scalar(select(WorkspaceMember).where(WorkspaceMember.workspace_id == ws_id, WorkspaceMember.user_id == u_id))
            if not m:
                session.add(WorkspaceMember(workspace_id=ws_id, user_id=u_id, role=role))
            elif m.role != role:
                m.role = role
            await session.commit()

        await ensure_member(ws_a.id, owner.id, "OWNER")
        await ensure_member(ws_a.id, editor.id, "EDITOR")
        await ensure_member(ws_a.id, viewer.id, "VIEWER")
        await ensure_member(ws_b.id, user_b.id, "OWNER")

        # Clear existing connections to prevent hitting workspace connection limit across test runs
        await session.execute(
            delete(WorkspaceMCPConnection).where(
                WorkspaceMCPConnection.workspace_id.in_([ws_a.id, ws_b.id])
            )
        )
        await session.commit()

        return {
            "admin": admin,
            "owner": owner,
            "editor": editor,
            "viewer": viewer,
            "user_b": user_b,
            "ws_a": ws_a,
            "ws_b": ws_b,
        }


# -----------------------------------------------------------------------------
# 1. Credential Reference Restrictions & Arbitrary Env-Var Rejection
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_arbitrary_environment_variables_strictly_rejected(setup_hardening_context):
    """
    Verify that arbitrary, process-wide, or server-secret environment variables
    (e.g., DATABASE_URL, SECRET_KEY, POSTGRES_PASSWORD) cannot be registered
    as credential_ref, preventing credential exfiltration attacks.
    """
    ctx = setup_hardening_context
    owner = ctx["owner"]
    ws_a = ctx["ws_a"]

    prohibited_refs = [
        "DATABASE_URL",
        "POSTGRES_PASSWORD",
        "SECRET_KEY",
        "JWT_SECRET",
        "ADMIN_PASSWORD",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "SOME_RANDOM_ENV_VAR",
    ]

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        for bad_ref in prohibited_refs:
            with pytest.raises(HTTPException) as exc:
                await service.create_connection(
                    owner,
                    ws_a.id,
                    MCPConnectionCreate(
                        name=f"BadRefConn_{uuid.uuid4().hex[:6]}",
                        endpoint_url="https://example.com/mcp",
                        credential_ref=bad_ref,
                    ),
                )
            assert exc.value.status_code == 400
            err_msg = str(exc.value.detail).lower()
            assert "not approved" in err_msg or "raw api keys" in err_msg


@pytest.mark.asyncio
async def test_endpoint_credential_compatibility_and_untrusted_forwarding_prevention(setup_hardening_context):
    """
    Verify that approved credentials cannot be forwarded to mismatched or
    untrusted endpoints (e.g. SERVER_TWENTY_API_KEY forwarded to an attacker's server).
    """
    ctx = setup_hardening_context
    owner = ctx["owner"]
    ws_a = ctx["ws_a"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        # 1. Forwarding SERVER_TWENTY_API_KEY to an attacker domain -> 400 Bad Request
        with pytest.raises(HTTPException) as exc:
            await service.create_connection(
                owner,
                ws_a.id,
                MCPConnectionCreate(
                    name=f"EvilForwardConn_{uuid.uuid4().hex[:6]}",
                    endpoint_url="https://attacker-domain.com/mcp",
                    credential_ref="SERVER_TWENTY_API_KEY",
                ),
            )
        assert exc.value.status_code == 400
        assert "cannot be bound to endpoint host" in str(exc.value.detail).lower()

        # 2. Binding SERVER_TWENTY_API_KEY to valid Twenty domain -> SUCCESS
        good_conn = await service.create_connection(
            owner,
            ws_a.id,
            MCPConnectionCreate(
                name=f"GoodTwentyConn_{uuid.uuid4().hex[:6]}",
                endpoint_url="https://api.twenty.com/mcp",
                credential_ref="SERVER_TWENTY_API_KEY",
            ),
        )
        assert good_conn.credential_ref == "SERVER_TWENTY_API_KEY"

        # 3. Setting a secret in environment and resolving it for authorized endpoint
        os.environ["TWENTY_API_KEY"] = "super_secret_twenty_token_xyz"
        try:
            resolved = service._resolve_credential_secret("SERVER_TWENTY_API_KEY", "https://api.twenty.com/mcp")
            assert resolved == "super_secret_twenty_token_xyz"

            # Attempting resolution against untrusted endpoint raises 400
            with pytest.raises(HTTPException) as res_exc:
                service._resolve_credential_secret("SERVER_TWENTY_API_KEY", "https://evil-server.org/mcp")
            assert res_exc.value.status_code == 400
            assert "cannot be bound" in str(res_exc.value.detail).lower()
        finally:
            os.environ.pop("TWENTY_API_KEY", None)


# -----------------------------------------------------------------------------
# 2. Role-Based Invocation Authorization & Risk Classification
# -----------------------------------------------------------------------------

@pytest.fixture
def multi_risk_mcp_app():
    """In-memory mock FastMCP application with read-only, mutating, and destructive tools."""
    server = FastMCP(name="MultiRiskServer", version="1.0.0")

    @server.tool(name="read_report", description="Returns read-only analytics data.")
    def read_report(report_id: str) -> str:
        return f"Report {report_id} data"

    @server.tool(name="mutate_record", description="Updates a record in external system.")
    def mutate_record(record_id: str, new_val: str) -> str:
        return f"Updated {record_id} to {new_val}"

    @server.tool(name="purge_records", description="Destructive operation that purges records.")
    def purge_records(confirm: bool) -> str:
        return "Records purged"

    return server.http_app(transport="streamable-http")


@pytest.mark.asyncio
async def test_tool_invocation_role_and_risk_authorization(setup_hardening_context, multi_risk_mcp_app):
    """
    Verify role-based authorization for tool execution:
    - VIEWER: Allowed to invoke tools classified as read_only; BLOCKED from mutating tools.
    - EDITOR: Allowed to invoke read_only and mutating tools; BLOCKED from destructive tools.
    - OWNER: Allowed to invoke read_only, mutating, and destructive tools.
    - Non-member (User B): BLOCKED from all tools (403).
    """
    ctx = setup_hardening_context
    owner = ctx["owner"]
    editor = ctx["editor"]
    viewer = ctx["viewer"]
    user_b = ctx["user_b"]
    ws_a = ctx["ws_a"]

    async with multi_risk_mcp_app.router.lifespan_context(multi_risk_mcp_app):
        transport = httpx2.ASGITransport(app=multi_risk_mcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://mockserver",
                headers=headers,
                timeout=timeout,
            )

        async with AsyncSessionLocal() as session:
            service = WorkspaceMCPConnectionService(session, http_client_factory=client_factory)

            # Owner registers connection with explicit risk policies
            conn_name = f"RiskConn_{uuid.uuid4().hex[:6]}"
            conn = await service.create_connection(
                owner,
                ws_a.id,
                MCPConnectionCreate(
                    name=conn_name,
                    endpoint_url="http://mockserver/mcp",
                    configuration={
                        "allow_private_hosts": True,
                        "read_only_tools": ["read_report"],
                        "tool_policies": {
                            "purge_records": {
                                "risk_level": "destructive",
                                "requires_approval": True,
                            }
                        },
                    },
                ),
            )

            # Sync tools and enable all three
            await service.sync_tools(owner, ws_a.id, conn.id)
            await service.update_tool_allowlist(owner, ws_a.id, conn.id, ["read_report", "mutate_record", "purge_records"])

            # ── 1. VIEWER TESTS ───────────────────────────────────────────────
            # VIEWER CAN invoke classified read-only tool
            res_v1 = await service.invoke_workspace_tool(viewer, ws_a.id, conn.id, "read_report", {"report_id": "rep_1"})
            assert "Report rep_1 data" in res_v1.text

            # VIEWER CANNOT invoke unclassified mutating tool -> 403 Forbidden
            with pytest.raises(HTTPException) as exc_v2:
                await service.invoke_workspace_tool(viewer, ws_a.id, conn.id, "mutate_record", {"record_id": "rec_1", "new_val": "abc"})
            assert exc_v2.value.status_code == 403
            assert "side-effecting/mutating" in str(exc_v2.value.detail).lower()

            # VIEWER CANNOT invoke destructive tool -> 403 Forbidden
            with pytest.raises(HTTPException) as exc_v3:
                await service.invoke_workspace_tool(viewer, ws_a.id, conn.id, "purge_records", {"confirm": True})
            assert exc_v3.value.status_code == 403

            # ── 2. EDITOR TESTS ───────────────────────────────────────────────
            # EDITOR CAN invoke read-only tool
            res_e1 = await service.invoke_workspace_tool(editor, ws_a.id, conn.id, "read_report", {"report_id": "rep_2"})
            assert "Report rep_2 data" in res_e1.text

            # EDITOR CAN invoke mutating tool
            res_e2 = await service.invoke_workspace_tool(editor, ws_a.id, conn.id, "mutate_record", {"record_id": "rec_2", "new_val": "xyz"})
            assert "Updated rec_2 to xyz" in res_e2.text

            # EDITOR CANNOT invoke destructive tool -> 403 Forbidden
            with pytest.raises(HTTPException) as exc_e3:
                await service.invoke_workspace_tool(editor, ws_a.id, conn.id, "purge_records", {"confirm": True})
            assert exc_e3.value.status_code == 403
            assert "destructive" in str(exc_e3.value.detail).lower()

            # ── 3. OWNER TESTS ────────────────────────────────────────────────
            # OWNER CAN invoke all three tools including destructive
            res_o1 = await service.invoke_workspace_tool(owner, ws_a.id, conn.id, "read_report", {"report_id": "rep_3"})
            assert "Report rep_3 data" in res_o1.text

            res_o2 = await service.invoke_workspace_tool(owner, ws_a.id, conn.id, "mutate_record", {"record_id": "rec_3", "new_val": "123"})
            assert "Updated rec_3 to 123" in res_o2.text

            res_o3 = await service.invoke_workspace_tool(owner, ws_a.id, conn.id, "purge_records", {"confirm": True})
            assert "Records purged" in res_o3.text

            # ── 4. CROSS-WORKSPACE USER B TESTS ──────────────────────────────
            with pytest.raises(HTTPException) as exc_b:
                await service.invoke_workspace_tool(user_b, ws_a.id, conn.id, "read_report", {"report_id": "rep_b"})
            assert exc_b.value.status_code == 403
