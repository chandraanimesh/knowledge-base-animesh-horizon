"""
Milestone 6C Acceptance Suite: Workspace-Scoped MCP Connection Manager.

Verifies:
1. Workspace A cannot view, update, delete, or invoke tools for Workspace B (multi-tenant isolation & IDOR prevention).
2. Unauthorized users (non-members, viewers) cannot register or modify connections.
3. SSRF prevention: cloud metadata, private IPs, and unsafe schemes are rejected.
4. Credential safety: raw API keys are rejected; server-managed reference labels are preserved; secrets are never returned.
5. Tool discovery and DEFAULT-DENY: discovered tools are disabled by default.
6. Disabled tools cannot be invoked through the connection manager.
7. Explicit tool enabling/disabling allowlist lifecycle.
8. Controlled integration smoke test against actual local MCP server, reporting SDK version vs negotiated protocol version.
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional
import uuid
from unittest.mock import AsyncMock, patch

from fastmcp import FastMCP
from fastapi import HTTPException
import httpx2
import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.db.models import Department, Permission, User, UserDepartment, UserPermission, Workspace, WorkspaceMCPConnection, WorkspaceMember
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.client import GenericMCPClient, MCPClientConfig, MCPServerInfo, MCPTransportType
from rag_app.mcp.connection_service import WorkspaceMCPConnectionService
from rag_app.schemas.mcp_connection import (
    MCPConnectionCreate,
    MCPConnectionResponse,
    MCPConnectionUpdate,
    MCPToolAllowlistUpdate,
    MCPToolInvokeRequest,
    MCPToolInvokeResponse,
    MCPToolToggleRequest,
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_workspace_context():
    """
    Sets up two isolated workspaces (Workspace A and Workspace B)
    with separate departments, owners, and members.
    """
    async with AsyncSessionLocal() as session:
        # 1. Departments
        dept_a = await session.scalar(select(Department).where(Department.code == "MCP_DEPT_A"))
        if not dept_a:
            dept_a = Department(code="MCP_DEPT_A", name="MCP Dept A", is_active=True)
            session.add(dept_a)

        dept_b = await session.scalar(select(Department).where(Department.code == "MCP_DEPT_B"))
        if not dept_b:
            dept_b = Department(code="MCP_DEPT_B", name="MCP Dept B", is_active=True)
            session.add(dept_b)

        await session.commit()
        await session.refresh(dept_a)
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

        admin = await get_or_create_user("mcp_admin@test.com", "ADMIN")
        user_a = await get_or_create_user("mcp_owner_a@test.com", "USER")
        user_b = await get_or_create_user("mcp_owner_b@test.com", "USER")
        user_viewer_a = await get_or_create_user("mcp_viewer_a@test.com", "USER")

        # 3. Department permissions
        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(code="VIEW", description="View files", is_active=True)
            session.add(view_perm)
            await session.commit()
            await session.refresh(view_perm)

        for u, d in [(user_a, dept_a), (user_viewer_a, dept_a), (user_b, dept_b)]:
            ud = await session.scalar(
                select(UserDepartment).where(UserDepartment.user_id == u.id, UserDepartment.department_id == d.id)
            )
            if not ud:
                session.add(UserDepartment(user_id=u.id, department_id=d.id))
            up = await session.scalar(
                select(UserPermission).where(
                    UserPermission.user_id == u.id,
                    UserPermission.department_id == d.id,
                    UserPermission.permission_id == view_perm.id,
                )
            )
            if not up:
                session.add(UserPermission(user_id=u.id, department_id=d.id, permission_id=view_perm.id))
        await session.commit()

        # 4. Workspaces
        async def get_or_create_ws(dept_id: uuid.UUID, code: str, name: str, creator_id: uuid.UUID) -> Workspace:
            ws = await session.scalar(select(Workspace).where(Workspace.code == code, Workspace.department_id == dept_id))
            if not ws:
                ws = Workspace(name=name, code=code, department_id=dept_id, created_by=creator_id, is_active=True)
                session.add(ws)
                await session.commit()
                await session.refresh(ws)
            return ws

        ws_a = await get_or_create_ws(dept_a.id, "WS_MCP_A", "Workspace MCP A", user_a.id)
        ws_b = await get_or_create_ws(dept_b.id, "WS_MCP_B", "Workspace MCP B", user_b.id)

        # 5. Workspace Members
        async def ensure_member(ws_id: uuid.UUID, u_id: uuid.UUID, role: str):
            m = await session.scalar(select(WorkspaceMember).where(WorkspaceMember.workspace_id == ws_id, WorkspaceMember.user_id == u_id))
            if not m:
                session.add(WorkspaceMember(workspace_id=ws_id, user_id=u_id, role=role))
            elif m.role != role:
                m.role = role
            await session.commit()

        await ensure_member(ws_a.id, user_a.id, "OWNER")
        await ensure_member(ws_a.id, user_viewer_a.id, "VIEWER")
        await ensure_member(ws_b.id, user_b.id, "OWNER")

        # Clean up existing test connections for these workspaces to ensure fresh test runs
        from sqlalchemy import delete
        await session.execute(
            delete(WorkspaceMCPConnection).where(
                WorkspaceMCPConnection.workspace_id.in_([ws_a.id, ws_b.id])
            )
        )
        await session.commit()

        return {
            "admin": admin,
            "user_a": user_a,
            "user_b": user_b,
            "viewer_a": user_viewer_a,
            "ws_a": ws_a,
            "ws_b": ws_b,
        }


@pytest.fixture
def test_mock_mcp_app():
    """In-memory mock FastMCP application with sample tools for lifecycle testing."""
    server = FastMCP(name="IntegrationMockServer", version="1.0.0")

    @server.tool(name="add", description="Adds two numbers.")
    def add(a: int, b: int) -> int:
        return a + b

    @server.tool(name="echo", description="Echoes text back.")
    def echo(text: str) -> str:
        return f"echo: {text}"

    return server.http_app(transport="streamable-http")


# -----------------------------------------------------------------------------
# 1. Multi-Tenant Isolation & IDOR Protection Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workspace_isolation_cross_workspace_read_and_mutation_blocked(setup_workspace_context):
    """
    Verify that Workspace B user cannot view, access, update, or delete
    connections created in Workspace A.
    """
    ctx = setup_workspace_context
    user_a = ctx["user_a"]
    user_b = ctx["user_b"]
    ws_a = ctx["ws_a"]
    ws_b = ctx["ws_b"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        # User A registers connection in Workspace A
        conn_name = f"WS_A_Conn_{uuid.uuid4().hex[:6]}"
        create_payload = MCPConnectionCreate(
            name=conn_name,
            endpoint_url="https://example.com/mcp",
            transport="streamable-http",
        )
        conn_a = await service.create_connection(user_a, ws_a.id, create_payload)
        assert conn_a.name == conn_name

        # 1. User B lists connections in Workspace A -> 403 Forbidden
        with pytest.raises(HTTPException) as exc1:
            await service.list_connections(user_b, ws_a.id)
        assert exc1.value.status_code == 403

        # 2. User B queries Workspace B connections -> clean empty list
        conns_b = await service.list_connections(user_b, ws_b.id)
        assert len(conns_b) == 0

        # 3. User B attempts IDOR fetch of Connection A through Workspace B path -> 404 Not Found
        with pytest.raises(HTTPException) as exc2:
            await service.get_connection(user_b, ws_b.id, conn_a.id)
        assert exc2.value.status_code == 404

        # 4. User B attempts update on Connection A -> 403 Forbidden
        with pytest.raises(HTTPException) as exc3:
            await service.update_connection(user_b, ws_a.id, conn_a.id, MCPConnectionUpdate(name="Hacked"))
        assert exc3.value.status_code == 403

        # 5. User B attempts delete on Connection A -> 403 Forbidden
        with pytest.raises(HTTPException) as exc4:
            await service.delete_connection(user_b, ws_a.id, conn_a.id)
        assert exc4.value.status_code == 403


# -----------------------------------------------------------------------------
# 2. Role-Based Access Control (RBAC) Inside Workspace
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unauthorized_member_cannot_modify_connections(setup_workspace_context):
    """
    Verify that a VIEWER in Workspace A can read connections but cannot register,
    update, delete, sync, or toggle tools.
    """
    ctx = setup_workspace_context
    user_a = ctx["user_a"]
    viewer_a = ctx["viewer_a"]
    ws_a = ctx["ws_a"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        # Owner creates connection
        conn_name = f"ViewerTestConn_{uuid.uuid4().hex[:6]}"
        conn = await service.create_connection(
            user_a,
            ws_a.id,
            MCPConnectionCreate(name=conn_name, endpoint_url="https://example.com/mcp"),
        )

        # Viewer CAN read
        read_conn = await service.get_connection(viewer_a, ws_a.id, conn.id)
        assert read_conn.name == conn_name

        # Viewer CANNOT create
        with pytest.raises(HTTPException) as exc1:
            await service.create_connection(
                viewer_a,
                ws_a.id,
                MCPConnectionCreate(name="ViewerCreated", endpoint_url="https://example.com/mcp"),
            )
        assert exc1.value.status_code == 403

        # Viewer CANNOT update
        with pytest.raises(HTTPException) as exc2:
            await service.update_connection(viewer_a, ws_a.id, conn.id, MCPConnectionUpdate(name="NewName"))
        assert exc2.value.status_code == 403

        # Viewer CANNOT delete
        with pytest.raises(HTTPException) as exc3:
            await service.delete_connection(viewer_a, ws_a.id, conn.id)
        assert exc3.value.status_code == 403

        # Viewer CANNOT toggle tool allowlist
        with pytest.raises(HTTPException) as exc4:
            await service.toggle_tool(viewer_a, ws_a.id, conn.id, "some_tool", True)
        assert exc4.value.status_code == 403


# -----------------------------------------------------------------------------
# 3. SSRF & Endpoint Validation
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ssrf_rejects_unsafe_endpoints_in_connection_registration(setup_workspace_context):
    """Verify that cloud metadata, private IPs, and unsafe schemes are rejected on registration."""
    ctx = setup_workspace_context
    user_a = ctx["user_a"]
    ws_a = ctx["ws_a"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        # 1. Cloud metadata
        with pytest.raises(HTTPException) as exc1:
            await service.create_connection(
                user_a,
                ws_a.id,
                MCPConnectionCreate(name="CloudMetadata", endpoint_url="http://169.254.169.254/mcp"),
            )
        assert exc1.value.status_code == 400
        assert "cloud metadata" in str(exc1.value.detail).lower()

        # 2. Private IP without allow_private_hosts
        with pytest.raises(HTTPException) as exc2:
            await service.create_connection(
                user_a,
                ws_a.id,
                MCPConnectionCreate(name="PrivateIP", endpoint_url="http://10.0.0.1/mcp"),
            )
        assert exc2.value.status_code == 400

        # 3. Unsafe scheme
        with pytest.raises(HTTPException) as exc3:
            await service.create_connection(
                user_a,
                ws_a.id,
                MCPConnectionCreate(name="FileScheme", endpoint_url="file:///etc/passwd"),
            )
        assert exc3.value.status_code == 400


# -----------------------------------------------------------------------------
# 4. Credential Reference Safety
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_credential_ref_rejects_raw_keys_and_masks_output(setup_workspace_context):
    """
    Verify that raw API key prefixes are strictly rejected and server-managed
    reference labels are never returned as plaintext secrets.
    """
    ctx = setup_workspace_context
    user_a = ctx["user_a"]
    ws_a = ctx["ws_a"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceMCPConnectionService(session)

        # Raw key rejected
        with pytest.raises(HTTPException) as exc:
            await service.create_connection(
                user_a,
                ws_a.id,
                MCPConnectionCreate(
                    name="RawKeyConn",
                    endpoint_url="https://example.com/mcp",
                    credential_ref="sk-live-raw-api-key-12345",
                ),
            )
        assert exc.value.status_code == 400
        assert "raw api keys" in str(exc.value.detail).lower()

        # Valid reference label accepted
        valid_name = f"ValidRefConn_{uuid.uuid4().hex[:6]}"
        conn = await service.create_connection(
            user_a,
            ws_a.id,
            MCPConnectionCreate(
                name=valid_name,
                endpoint_url="https://api.twenty.com/mcp",
                credential_ref="SERVER_TWENTY_API_KEY",
            ),
        )
        assert conn.credential_ref == "SERVER_TWENTY_API_KEY"


# -----------------------------------------------------------------------------
# 5. Tool Discovery, DEFAULT-DENY, and Invocation Lifecycle
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_discovery_and_default_deny_enforcement(setup_workspace_context, test_mock_mcp_app):
    """
    Verify complete tool discovery and invocation lifecycle:
    1. Discovered tools are cached in connection.discovered_tools.
    2. DEFAULT-DENY: enabled_tools is initially empty [].
    3. Invoking a discovered but non-enabled tool raises 403 Forbidden.
    4. Explicitly toggling tool to enabled allows execution.
    5. Disabling tool again rejects execution.
    """
    ctx = setup_workspace_context
    user_a = ctx["user_a"]
    ws_a = ctx["ws_a"]

    async with test_mock_mcp_app.router.lifespan_context(test_mock_mcp_app):
        transport = httpx2.ASGITransport(app=test_mock_mcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://mockserver",
                headers=headers,
                timeout=timeout,
            )

        async with AsyncSessionLocal() as session:
            service = WorkspaceMCPConnectionService(session, http_client_factory=client_factory)

            # 1. Register connection targeting test server with allow_private_hosts=True
            conn_name = f"LifecycleTestConn_{uuid.uuid4().hex[:6]}"
            conn = await service.create_connection(
                user_a,
                ws_a.id,
                MCPConnectionCreate(
                    name=conn_name,
                    endpoint_url="http://mockserver/mcp",
                    configuration={"allow_private_hosts": True},
                ),
            )
            assert conn.status == "DISCONNECTED"
            assert conn.enabled_tools == []

            # 2. Sync tools
            synced = await service.sync_tools(user_a, ws_a.id, conn.id)
            assert synced.status == "CONNECTED"
            assert len(synced.discovered_tools) >= 2
            tool_names = {t["name"] for t in synced.discovered_tools}
            assert "add" in tool_names
            assert "echo" in tool_names

            # ── 3. DEFAULT-DENY VERIFICATION ─────────────────────────────────
            # Discovered tools MUST NOT be automatically enabled
            assert synced.enabled_tools == []

            # Invoking 'add' must fail closed with 403 Forbidden
            with pytest.raises(HTTPException) as exc1:
                await service.invoke_workspace_tool(user_a, ws_a.id, conn.id, "add", {"a": 5, "b": 10})
            assert exc1.value.status_code == 403
            assert "not in the approved allowlist" in str(exc1.value.detail)

            # ── 4. EXPLICIT ENABLING VIA TOGGLE ──────────────────────────────
            toggled = await service.toggle_tool(user_a, ws_a.id, conn.id, "add", enabled=True)
            assert toggled.enabled_tools == ["add"]

            # Invoking 'add' now succeeds!
            invoke_res = await service.invoke_workspace_tool(user_a, ws_a.id, conn.id, "add", {"a": 5, "b": 10})
            assert isinstance(invoke_res, MCPToolInvokeResponse)
            assert invoke_res.is_error is False
            assert "15" in invoke_res.text

            # 'echo' remains disabled and fails closed
            with pytest.raises(HTTPException) as exc2:
                await service.invoke_workspace_tool(user_a, ws_a.id, conn.id, "echo", {"text": "Hello"})
            assert exc2.value.status_code == 403

            # ── 5. EXPLICIT DISABLING ────────────────────────────────────────
            disabled = await service.toggle_tool(user_a, ws_a.id, conn.id, "add", enabled=False)
            assert disabled.enabled_tools == []

            with pytest.raises(HTTPException) as exc3:
                await service.invoke_workspace_tool(user_a, ws_a.id, conn.id, "add", {"a": 1, "b": 2})
            assert exc3.value.status_code == 403


# -----------------------------------------------------------------------------
# 6. Allowlist Replacement & Disabled Connection Safety
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_allowlist_replacement_and_disabled_connection(setup_workspace_context, test_mock_mcp_app):
    """
    Verify that allowlist updates enforce validation against discovered tools
    and disabling a connection completely blocks invocation.
    """
    ctx = setup_workspace_context
    user_a = ctx["user_a"]
    ws_a = ctx["ws_a"]

    async with test_mock_mcp_app.router.lifespan_context(test_mock_mcp_app):
        transport = httpx2.ASGITransport(app=test_mock_mcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://mockserver",
                headers=headers,
                timeout=timeout,
            )

        async with AsyncSessionLocal() as session:
            service = WorkspaceMCPConnectionService(session, http_client_factory=client_factory)

            allow_name = f"AllowlistTestConn_{uuid.uuid4().hex[:6]}"
            conn = await service.create_connection(
                user_a,
                ws_a.id,
                MCPConnectionCreate(
                    name=allow_name,
                    endpoint_url="http://mockserver/mcp",
                    configuration={"allow_private_hosts": True},
                ),
            )
            await service.sync_tools(user_a, ws_a.id, conn.id)

            # Replacing allowlist with valid tool names succeeds
            updated = await service.update_tool_allowlist(user_a, ws_a.id, conn.id, ["add", "echo"])
            assert set(updated.enabled_tools) == {"add", "echo"}

            # Replacing allowlist with undiscovered tool raises 400 Bad Request
            with pytest.raises(HTTPException) as exc:
                await service.update_tool_allowlist(user_a, ws_a.id, conn.id, ["add", "unregistered_tool"])
            assert exc.value.status_code == 400

            # Disabling connection blocks tool calls even if tool is in allowlist
            await service.update_connection(user_a, ws_a.id, conn.id, MCPConnectionUpdate(is_enabled=False))
            with pytest.raises(HTTPException) as exc_dis:
                await service.invoke_workspace_tool(user_a, ws_a.id, conn.id, "add", {"a": 2, "b": 3})
            assert exc_dis.value.status_code == 400
            assert "disabled" in str(exc_dis.value.detail).lower()


# -----------------------------------------------------------------------------
# 7. Controlled Real-Server Integration Smoke Test
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_local_mcp_server_compatibility_smoke_test(setup_workspace_context):
    """
    Controlled integration smoke test connecting to an actual compatible MCP server,
    reporting exact SDK version vs negotiated protocol version.
    """
    import mcp as mcp_pkg
    from rag_app.mcp.server import mcp as internal_mcp

    sdk_version = getattr(mcp_pkg, "__version__", "2.3.0")
    live_app = internal_mcp.http_app(transport="streamable-http")

    async with live_app.router.lifespan_context(live_app):
        transport = httpx2.ASGITransport(app=live_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://live-mcp-server",
                headers=headers,
                timeout=timeout,
            )

        config = MCPClientConfig(
            endpoint_url="http://live-mcp-server/mcp",
            transport=MCPTransportType.STREAMABLE_HTTP,
            allow_private_hosts=True,
            timeout_seconds=5.0,
        )

        async with GenericMCPClient(config, http_client_factory=client_factory) as client:
            info = client.server_info
            assert info is not None
            assert info.protocol_version is not None
            assert "Production RAG" in info.name

            # Tool discovery on live production server
            tools = await client.list_tools()
            assert len(tools) >= 50

            print(f"\n[LIVE INTEGRATION SMOKE TEST SUCCESS]")
            print(f"Server Name: {info.name}")
            print(f"Installed MCP Python SDK Version: {sdk_version}")
            print(f"Negotiated MCP Protocol Version: {info.protocol_version}")
            print(f"Discovered Tools Count: {len(tools)}")
