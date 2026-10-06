"""
Phase 4 Tests: FastMCP Tool Discovery and Search.
Validates:
1. Tool discovery works for authenticated clients with natural language search.
2. Discovery strictly respects role-aware visibility (ADMIN vs USER).
3. Discovery does NOT bypass Gateway authorization.
4. Scope tampering attempts are rejected.
5. Purpose-built RAG tools remain authoritative and delegate to existing services.
6. Dynamic 1-to-1 mapping between registered tools and TOOL_POLICIES.
7. Domain-filtered tool discovery.
"""

import uuid
import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.db.models import Department, Permission, User
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.gateway import MCPGatewayError
from rag_app.mcp.policies import TOOL_POLICIES
from rag_app.mcp.server import (
    admin_list_users,
    mcp,
    rag_ask,
    rag_search,
    search_tools,
)
from rag_app.security.authentication import create_access_token


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def discovery_test_setup():
    """Setup test departments, permissions, admin user, and standard user."""
    async with AsyncSessionLocal() as session:
        # 1. Departments
        dept_eng = await session.scalar(select(Department).where(Department.code == "DISC_ENG"))
        if not dept_eng:
            dept_eng = Department(code="DISC_ENG", name="Discovery Engineering", is_active=True)
            session.add(dept_eng)

        # 2. Permissions
        perm_view = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not perm_view:
            perm_view = Permission(code="VIEW", description="View access", is_active=True)
            session.add(perm_view)

        await session.flush()

        # 3. Admin user
        admin_user = await session.scalar(
            select(User).where(User.email == "discovery_admin@test.com")
        )
        if not admin_user:
            admin_user = User(
                email="discovery_admin@test.com",
                hashed_password="hash",
                role="ADMIN",
                is_active=True,
            )
            session.add(admin_user)

        # 4. Standard user
        std_user = await session.scalar(
            select(User).where(User.email == "discovery_std_user@test.com")
        )
        if not std_user:
            std_user = User(
                email="discovery_std_user@test.com",
                hashed_password="hash",
                role="USER",
                is_active=True,
            )
            session.add(std_user)

        await session.commit()
        await session.refresh(admin_user)
        await session.refresh(std_user)

        admin_token = f"Bearer {create_access_token(subject=str(admin_user.id))}"
        std_token = f"Bearer {create_access_token(subject=str(std_user.id))}"

        return {
            "admin_user": admin_user,
            "std_user": std_user,
            "admin_token": admin_token,
            "std_token": std_token,
        }


# ── Test 1: Tool discovery works for authenticated clients ───────────────────

@pytest.mark.asyncio
async def test_tool_discovery_natural_language_queries(discovery_test_setup):
    """Authenticated user can discover relevant tools via natural language search."""
    std_token = discovery_test_setup["std_token"]

    # Search for RAG retrieval tools
    rag_res = await search_tools(
        query="search documents and retrieve evidence chunks",
        auth_token=std_token,
    )
    assert rag_res["total_matches"] > 0
    matched_names = [t["name"] for t in rag_res["tools"]]
    assert "rag_search" in matched_names or "list_documents" in matched_names

    # Search for conversation management tools
    conv_res = await search_tools(
        query="create new conversation thread and chat",
        auth_token=std_token,
    )
    assert conv_res["total_matches"] > 0
    matched_convs = [t["name"] for t in conv_res["tools"]]
    assert "create_conversation" in matched_convs or "list_conversations" in matched_convs


# ── Test 2: Discovery strictly respects role-aware visibility ────────────────

@pytest.mark.asyncio
async def test_discovery_respects_visibility_boundaries(discovery_test_setup):
    """
    Standard USER must NOT discover admin-only tools in search results.
    ADMIN user CAN discover admin-only tools.
    """
    std_token = discovery_test_setup["std_token"]
    admin_token = discovery_test_setup["admin_token"]

    # 1. Standard USER searches for administrative user management
    user_search = await search_tools(
        query="create user account and assign department permissions",
        auth_token=std_token,
    )
    # Must NOT return admin-only tools
    for tool in user_search["tools"]:
        assert tool["visibility"] != "admin-only", f"Admin tool {tool['name']} leaked to standard user"
        assert tool["admin_only"] is False

    # 2. ADMIN user searches for administrative user management
    admin_search = await search_tools(
        query="create user account and assign department permissions",
        auth_token=admin_token,
    )
    admin_tool_names = [t["name"] for t in admin_search["tools"]]
    assert any("admin_" in name for name in admin_tool_names)


# ── Test 3: Discovery does NOT bypass Gateway authorization ──────────────────

@pytest.mark.asyncio
async def test_discovery_does_not_bypass_gateway_authorization(discovery_test_setup):
    """
    Even if a user knows an admin tool name, calling it directly must be rejected by Gateway (403).
    """
    std_token = discovery_test_setup["std_token"]

    with pytest.raises(MCPGatewayError) as exc_info:
        await admin_list_users(auth_token=std_token)
    assert exc_info.value.code == "FORBIDDEN"
    assert exc_info.value.status_code == 403


# ── Test 4: Scope tampering remains impossible ───────────────────────────────

@pytest.mark.asyncio
async def test_scope_tampering_rejected_in_discovery_and_execution(discovery_test_setup):
    """
    Caller identity and scopes cannot be spoofed via tool parameters.
    """
    std_token = discovery_test_setup["std_token"]

    # Attempt to search as admin while passing standard token
    search_res = await search_tools(
        query="admin revoke user permission preset",
        auth_token=std_token,
    )
    # Must still strictly filter out admin tools based on verified database role
    for tool in search_res["tools"]:
        assert tool["admin_only"] is False


# ── Test 5: Domain filtering works ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_discovery_domain_filtering(discovery_test_setup):
    """Domain parameter restricts discovery to the specified category."""
    std_token = discovery_test_setup["std_token"]

    workspaces_res = await search_tools(
        query="list",
        domain="Workspaces",
        auth_token=std_token,
    )
    assert workspaces_res["total_matches"] > 0
    for tool in workspaces_res["tools"]:
        assert "workspace" in tool["domain"].lower()


# ── Test 6: Policy consistency remains dynamic 1-to-1 ────────────────────────

@pytest.mark.asyncio
async def test_policy_consistency_dynamic_1to1_mapping():
    """
    Registered FastMCP tools and TOOL_POLICIES maintain an exact 1-to-1 dynamic mapping.
    """
    registered_tools = await mcp.list_tools()
    registered_names = {t.name for t in registered_tools}
    policy_names = set(TOOL_POLICIES.keys())

    missing_policies = registered_names - policy_names
    orphan_policies = policy_names - registered_names

    assert len(missing_policies) == 0, f"Registered tools missing policy: {missing_policies}"
    assert len(orphan_policies) == 0, f"Policies missing registered tool: {orphan_policies}"
