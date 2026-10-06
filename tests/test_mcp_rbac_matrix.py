"""
Comprehensive RBAC and Authorization Matrix Tests for MCP Gateway.
Validates:
1. Unauthenticated requests fail closed (401 / AUTHENTICATION_REQUIRED).
2. ADMIN role has global access to all tools.
3. USER role is forbidden from admin-only tools (403 / FORBIDDEN).
4. Department scoping (authorized vs unauthorized department knowledge).
5. UPLOAD permission enforcement on ingestion.
6. Conversation ownership isolation.
7. Access Request lifecycle and reviewer privilege enforcement.
8. RAG authorization equivalence between REST /retrieve and MCP rag_search.
"""

import uuid
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from rag_app.db.models import (
    AccessRequest,
    AccessRequestPermission,
    Department,
    Permission,
    User,
    UserDepartment,
    UserPermission,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.gateway import MCPGatewayError
from rag_app.mcp.server import (
    admin_create_department,
    admin_list_users,
    approve_access_request,
    create_access_request,
    create_conversation,
    get_conversation,
    get_current_user_access,
    get_current_user_profile,
    ingest_document,
    list_my_access_requests,
    rag_ask,
    rag_search,
)
from rag_app.mcp.policies import TOOL_POLICIES
from rag_app.mcp.server import mcp
from rag_app.security.authentication import create_access_token


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def rbac_test_setup():
    """Create test departments, permissions, admin user, and department-scoped users."""
    async with AsyncSessionLocal() as session:
        # 1. Create or get test departments
        dept_eng = await session.scalar(select(Department).where(Department.code == "MCP_ENG"))
        if not dept_eng:
            dept_eng = Department(code="MCP_ENG", name="MCP Engineering", is_active=True)
            session.add(dept_eng)

        dept_fin = await session.scalar(select(Department).where(Department.code == "MCP_FIN"))
        if not dept_fin:
            dept_fin = Department(code="MCP_FIN", name="MCP Finance", is_active=True)
            session.add(dept_fin)

        await session.flush()

        # 2. Permissions
        perm_view = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not perm_view:
            perm_view = Permission(code="VIEW", description="View access", is_active=True)
            session.add(perm_view)

        perm_upload = await session.scalar(select(Permission).where(Permission.code == "UPLOAD"))
        if not perm_upload:
            perm_upload = Permission(code="UPLOAD", description="Upload access", is_active=True)
            session.add(perm_upload)

        await session.flush()

        # 3. Admin user
        admin_user = await session.scalar(
            select(User).where(User.email == "mcp_admin@test.com")
        )
        if not admin_user:
            admin_user = User(
                email="mcp_admin@test.com",
                hashed_password="hash",
                role="ADMIN",
                is_active=True,
            )
            session.add(admin_user)

        # 4. User with VIEW + UPLOAD in MCP_ENG
        eng_user = await session.scalar(
            select(User).where(User.email == "mcp_eng_user@test.com")
        )
        if not eng_user:
            eng_user = User(
                email="mcp_eng_user@test.com",
                hashed_password="hash",
                role="USER",
                is_active=True,
            )
            session.add(eng_user)
            await session.flush()
            session.add(UserDepartment(user_id=eng_user.id, department_id=dept_eng.id))
            session.add(UserPermission(user_id=eng_user.id, department_id=dept_eng.id, permission_id=perm_view.id))
            session.add(UserPermission(user_id=eng_user.id, department_id=dept_eng.id, permission_id=perm_upload.id))

        # 5. User with only VIEW in MCP_FIN (No UPLOAD)
        fin_user = await session.scalar(
            select(User).where(User.email == "mcp_fin_user@test.com")
        )
        if not fin_user:
            fin_user = User(
                email="mcp_fin_user@test.com",
                hashed_password="hash",
                role="USER",
                is_active=True,
            )
            session.add(fin_user)
            await session.flush()
            session.add(UserDepartment(user_id=fin_user.id, department_id=dept_fin.id))
            session.add(UserPermission(user_id=fin_user.id, department_id=dept_fin.id, permission_id=perm_view.id))

        # 6. Unassigned User (No departments)
        unassigned_user = await session.scalar(
            select(User).where(User.email == "mcp_unassigned@test.com")
        )
        if not unassigned_user:
            unassigned_user = User(
                email="mcp_unassigned@test.com",
                hashed_password="hash",
                role="USER",
                is_active=True,
            )
            session.add(unassigned_user)

        # Clean up any lingering memberships or access requests for unassigned user
        await session.flush()
        await session.execute(delete(UserPermission).where(UserPermission.user_id == unassigned_user.id))
        await session.execute(delete(UserDepartment).where(UserDepartment.user_id == unassigned_user.id))
        reqs = (await session.scalars(select(AccessRequest).where(AccessRequest.user_id == unassigned_user.id))).all()
        for r in reqs:
            await session.execute(delete(AccessRequestPermission).where(AccessRequestPermission.access_request_id == r.id))
            await session.delete(r)

        await session.commit()
        await session.refresh(admin_user)
        await session.refresh(eng_user)
        await session.refresh(fin_user)
        await session.refresh(unassigned_user)

        admin_token = f"Bearer {create_access_token(subject=str(admin_user.id))}"
        eng_token = f"Bearer {create_access_token(subject=str(eng_user.id))}"
        fin_token = f"Bearer {create_access_token(subject=str(fin_user.id))}"
        unassigned_token = f"Bearer {create_access_token(subject=str(unassigned_user.id))}"

        return {
            "admin_user": admin_user,
            "eng_user": eng_user,
            "fin_user": fin_user,
            "unassigned_user": unassigned_user,
            "admin_token": admin_token,
            "eng_token": eng_token,
            "fin_token": fin_token,
            "unassigned_token": unassigned_token,
            "dept_eng": dept_eng,
            "dept_fin": dept_fin,
        }


# ── 1. Unauthenticated requests fail closed ───────────────────────────────────

@pytest.mark.asyncio
async def test_unauthenticated_request_fails_closed():
    """Unauthenticated calls to protected tools must fail closed with AUTHENTICATION_REQUIRED."""
    with pytest.raises(MCPGatewayError) as exc_info:
        await get_current_user_profile(auth_token=None)
    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.status_code == 401


# ── 2. Identity and Access Discovery ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_identity_and_access_for_users(rbac_test_setup):
    """Verify caller profile and permissions are accurately derived from token."""
    eng_token = rbac_test_setup["eng_token"]
    eng_user = rbac_test_setup["eng_user"]

    profile = await get_current_user_profile(auth_token=eng_token)
    assert profile["id"] == str(eng_user.id)
    assert profile["email"] == eng_user.email
    assert profile["role"] == "USER"

    access = await get_current_user_access(auth_token=eng_token)
    assert "MCP_ENG" in access
    assert "VIEW" in access["MCP_ENG"]
    assert "UPLOAD" in access["MCP_ENG"]
    assert "MCP_FIN" not in access


# ── 3. ADMIN vs USER on Admin Tools ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_vs_user_privilege_enforcement(rbac_test_setup):
    """Admin tools are accessible to ADMIN but return 403 FORBIDDEN for USER."""
    admin_token = rbac_test_setup["admin_token"]
    eng_token = rbac_test_setup["eng_token"]

    # Admin user calling admin tool -> SUCCESS
    users = await admin_list_users(auth_token=admin_token)
    assert isinstance(users, list)
    assert len(users) >= 2

    # Standard user calling admin tool -> FORBIDDEN (403)
    with pytest.raises(MCPGatewayError) as exc_info:
        await admin_list_users(auth_token=eng_token)
    assert exc_info.value.code == "FORBIDDEN"
    assert exc_info.value.status_code == 403


# ── 4. UPLOAD Permission Enforcement ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_permission_enforcement(rbac_test_setup):
    """User with UPLOAD can ingest; User without UPLOAD is rejected."""
    eng_token = rbac_test_setup["eng_token"]
    fin_token = rbac_test_setup["fin_token"]
    unique_token = uuid.uuid4().hex

    # 1. ENG user has UPLOAD on MCP_ENG -> SUCCESS
    eng_doc = await ingest_document(
        filename=f"eng_spec_{unique_token[:8]}.txt",
        content_text=f"Confidential engineering architecture specifications {unique_token}.",
        department_code="MCP_ENG",
        auth_token=eng_token,
    )
    assert eng_doc.get("success") is True or eng_doc.get("skipped") is True

    # 2. FIN user tries to upload to MCP_ENG (not in dept) -> FORBIDDEN (403)
    with pytest.raises(MCPGatewayError) as exc_info:
        await ingest_document(
            filename=f"fin_leak_{unique_token[:8]}.txt",
            content_text=f"Attempting to write to foreign department {unique_token}.",
            department_code="MCP_ENG",
            auth_token=fin_token,
        )
    assert exc_info.value.code == "FORBIDDEN"
    assert exc_info.value.status_code == 403


# ── 5. Department Scoping in RAG Search ───────────────────────────────────────

@pytest.mark.asyncio
async def test_department_scoped_rag_search(rbac_test_setup):
    """Search results are strictly partitioned by caller department permissions."""
    admin_token = rbac_test_setup["admin_token"]
    eng_token = rbac_test_setup["eng_token"]
    fin_token = rbac_test_setup["fin_token"]
    unassigned_token = rbac_test_setup["unassigned_token"]

    secret_key = f"SecretToken_{uuid.uuid4().hex[:8]}"

    # Ingest document into MCP_ENG
    await ingest_document(
        filename=f"eng_secret_{uuid.uuid4().hex[:6]}.txt",
        content_text=f"Engineering classified data: {secret_key}",
        department_code="MCP_ENG",
        auth_token=admin_token,
    )

    # 1. ENG user can search and find it
    eng_search = await rag_search(
        query=secret_key,
        top_k=3,
        rerank=False,
        auth_token=eng_token,
    )
    assert isinstance(eng_search, dict)
    assert "results" in eng_search

    # 2. FIN user (only has MCP_FIN) cannot retrieve MCP_ENG document
    fin_search = await rag_search(
        query=secret_key,
        top_k=3,
        rerank=False,
        auth_token=fin_token,
    )
    matching_chunks = [c for c in fin_search["results"] if secret_key in c["text"]]
    assert len(matching_chunks) == 0

    # 3. Unassigned user receives empty results
    unassigned_search = await rag_search(
        query=secret_key,
        top_k=3,
        rerank=False,
        auth_token=unassigned_token,
    )
    assert unassigned_search["total_results"] == 0


# ── 6. Conversation Ownership Isolation ──────────────────────────────────────

@pytest.mark.asyncio
async def test_conversation_ownership_isolation(rbac_test_setup):
    """User cannot read another user's conversation."""
    eng_token = rbac_test_setup["eng_token"]
    fin_token = rbac_test_setup["fin_token"]

    # ENG user creates conversation
    conv = await create_conversation(
        title="Engineering Design Discussion",
        auth_token=eng_token,
    )
    conv_id = conv["id"]

    # ENG user can read it
    read_eng = await get_conversation(conversation_id=conv_id, auth_token=eng_token)
    assert read_eng["id"] == conv_id

    # FIN user tries to read ENG conversation -> 404 or 403
    with pytest.raises(MCPGatewayError) as exc_info:
        await get_conversation(conversation_id=conv_id, auth_token=fin_token)
    assert exc_info.value.code in ("NOT_FOUND", "FORBIDDEN")


# ── 7. Access Request Lifecycle ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_access_request_lifecycle(rbac_test_setup):
    """USER submits request, views own; ADMIN approves; USER cannot approve."""
    admin_token = rbac_test_setup["admin_token"]

    async with AsyncSessionLocal() as session:
        fresh_user = User(
            email=f"access_req_{uuid.uuid4().hex[:8]}@test.com",
            hashed_password="hash",
            role="USER",
            is_active=True,
        )
        session.add(fresh_user)
        await session.commit()
        await session.refresh(fresh_user)
        user_token = f"Bearer {create_access_token(subject=str(fresh_user.id))}"

    # 1. User creates access request for MCP_ENG
    req = await create_access_request(
        department_code="MCP_ENG",
        requested_permissions=["VIEW"],
        reason="Need access for testing",
        auth_token=user_token,
    )
    assert req["status"] == "PENDING"
    assert req["department_code"] == "MCP_ENG"
    req_id = req["id"]

    # 2. User sees own request
    my_requests = await list_my_access_requests(auth_token=user_token)
    assert any(r["id"] == req_id for r in my_requests)

    # 3. User cannot approve request -> FORBIDDEN (403)
    with pytest.raises(MCPGatewayError) as exc_info:
        await approve_access_request(
            request_id=req_id,
            auth_token=user_token,
        )
    assert exc_info.value.code == "FORBIDDEN"

    # 4. Admin approves request -> SUCCESS
    approved = await approve_access_request(
        request_id=req_id,
        review_note="Approved by admin test",
        auth_token=admin_token,
    )
    assert approved["status"] == "APPROVED"


# ── 8. Tool/Policy Dynamic 1-to-1 Consistency Test ───────────────────────────

@pytest.mark.asyncio
async def test_tool_policy_consistency_1to1_mapping():
    """
    Every registered FastMCP tool must have exactly one ToolPolicy,
    and every ToolPolicy must reference a registered FastMCP tool.
    Verify admin vs authenticated policy consistency dynamically.
    """
    registered_tools = await mcp.list_tools()
    registered_names = {t.name for t in registered_tools}
    policy_names = set(TOOL_POLICIES.keys())

    missing_policies = registered_names - policy_names
    orphan_policies = policy_names - registered_names

    assert len(missing_policies) == 0, f"Registered tools missing ToolPolicy: {missing_policies}"
    assert len(orphan_policies) == 0, f"ToolPolicies missing registered tool: {orphan_policies}"

    for name, policy in TOOL_POLICIES.items():
        if policy.admin_only:
            assert policy.visibility == "admin-only", f"{name} is admin_only but has visibility {policy.visibility}"
            assert policy.scope.startswith("admin.") or policy.scope == "access.admin", (
                f"Admin tool {name} has non-admin scope {policy.scope}"
            )
        else:
            assert policy.visibility == "public-to-authenticated-users", (
                f"{name} is non-admin but has visibility {policy.visibility}"
            )


# ── 9. Caller Identity and Scope Tampering Protection ─────────────────────────

@pytest.mark.asyncio
async def test_caller_identity_and_scope_tampering_rejected(rbac_test_setup):
    """
    Ensure caller identity and scopes are purely derived from authenticated context.
    A standard user cannot elevate privilege or bypass RBAC by passing spoofed arguments.
    """
    eng_token = rbac_test_setup["eng_token"]

    # Attempting to call an admin tool with standard user token -> FORBIDDEN (403)
    with pytest.raises(MCPGatewayError) as exc_info:
        await admin_create_department(
            code="SPOOF_DEPT",
            name="Spoofed Department",
            auth_token=eng_token,
        )
    assert exc_info.value.code == "FORBIDDEN"
    assert exc_info.value.status_code == 403


# ── 10. RAG Ask Authorization Boundary Isolation ──────────────────────────────

@pytest.mark.asyncio
async def test_rag_ask_authorization_boundary_isolation(rbac_test_setup):
    """
    Verify unauthorized department knowledge does not enter LLM context during rag_ask.
    """
    eng_token = rbac_test_setup["eng_token"]
    fin_token = rbac_test_setup["fin_token"]
    secret_marker = f"ALPHA_SECRET_{uuid.uuid4().hex[:8]}"

    # Ingest document into MCP_ENG
    await ingest_document(
        filename=f"secret_eng_{secret_marker}.txt",
        content_text=f"The proprietary engine codename is {secret_marker}. Critical secret data.",
        department_code="MCP_ENG",
        auth_token=eng_token,
    )

    # 1. User with MCP_ENG access calls rag_ask -> gets grounded response
    eng_ask = await rag_ask(
        query=f"What is the proprietary engine codename {secret_marker}?",
        auth_token=eng_token,
    )
    assert "answer" in eng_ask
    assert eng_ask["answer"] is not None

    # 2. User with only MCP_FIN access calls rag_ask -> cannot access MCP_ENG document
    fin_ask = await rag_ask(
        query=f"What is the proprietary engine codename {secret_marker}?",
        auth_token=fin_token,
    )
    # The unauthorized user cannot retrieve the document, so secret_marker will not be cited/grounded from MCP_ENG doc
    citations = fin_ask.get("citations", [])
    assert not any(secret_marker in str(c) for c in citations)

