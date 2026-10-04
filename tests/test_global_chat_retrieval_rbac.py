import asyncio
import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import (
    User,
    Department,
    Permission,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
    Conversation,
    ChatMessage,
)
from rag_app.security.authorization import (
    authorize,
    get_allowed_departments,
    get_allowed_workspaces,
    authorize_workspace,
)
from rag_app.retrieval.workspace_thread_retriever import WorkspaceThreadRetriever
from rag_app.conversation.service import ConversationService
from rag_app.generation.prompt import build_rag_messages


@pytest_asyncio.fixture
async def db_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_test_data(db_session: AsyncSession):
    session = db_session
    now = datetime.now(timezone.utc)

    # 1. Setup Permissions
    view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
    if not view_perm:
        view_perm = Permission(id=uuid.uuid4(), code="VIEW", description="View permission", is_active=True)
        session.add(view_perm)
        await session.flush()

    # 2. Setup Departments (Sales, Finance, HR)
    depts = {}
    for code, name in [
        ("SALES_TEST", "Sales Test Dept"),
        ("FINANCE_TEST", "Finance Test Dept"),
        ("HR_TEST", "HR Test Dept"),
    ]:
        d = await session.scalar(select(Department).where(Department.code == code))
        if not d:
            d = Department(id=uuid.uuid4(), code=code, name=name, is_active=True, created_at=now)
            session.add(d)
            await session.flush()
        depts[code] = d

    # 3. Setup Users
    users = {}
    users_data = [
        ("admin_test@example.com", "ADMIN", "ADMIN"),
        ("sales_test@example.com", "USER", "SALES"),
        ("finance_test@example.com", "USER", "FINANCE"),
        ("hr_test@example.com", "USER", "HR"),
        ("multi_test@example.com", "USER", "MULTI"),
        ("no_dept_test@example.com", "USER", "NO_DEPT"),
    ]
    for email, role, label in users_data:
        u = await session.scalar(select(User).where(User.email == email))
        if not u:
            u = User(id=uuid.uuid4(), email=email, hashed_password="hashedpassword", role=role, is_active=True, created_at=now)
            session.add(u)
            await session.flush()
        users[label] = u

    # 4. Department memberships & permissions (clean insert if missing)
    async def ensure_user_dept(user_id, dept_id):
        ud = await session.scalar(select(UserDepartment).where(UserDepartment.user_id == user_id, UserDepartment.department_id == dept_id))
        if not ud:
            session.add(UserDepartment(user_id=user_id, department_id=dept_id))
            await session.flush()
        up = await session.scalar(select(UserPermission).where(UserPermission.user_id == user_id, UserPermission.department_id == dept_id, UserPermission.permission_id == view_perm.id))
        if not up:
            session.add(UserPermission(user_id=user_id, department_id=dept_id, permission_id=view_perm.id))
            await session.flush()

    await ensure_user_dept(users["SALES"].id, depts["SALES_TEST"].id)
    await ensure_user_dept(users["FINANCE"].id, depts["FINANCE_TEST"].id)
    await ensure_user_dept(users["HR"].id, depts["HR_TEST"].id)
    await ensure_user_dept(users["MULTI"].id, depts["SALES_TEST"].id)
    await ensure_user_dept(users["MULTI"].id, depts["FINANCE_TEST"].id)

    # 5. Workspaces
    workspaces = {}
    for code, dept_code, name in [
        ("WS_SALES_TEST", "SALES_TEST", "Sales Test Workspace"),
        ("WS_FINANCE_TEST", "FINANCE_TEST", "Finance Test Workspace"),
        ("WS_HR_TEST", "HR_TEST", "HR Test Workspace"),
    ]:
        ws = await session.scalar(select(Workspace).where(Workspace.code == code, Workspace.department_id == depts[dept_code].id))
        if not ws:
            ws = Workspace(
                id=uuid.uuid4(),
                code=code,
                name=name,
                department_id=depts[dept_code].id,
                created_by=users["ADMIN"].id,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            session.add(ws)
            await session.flush()
        workspaces[code] = ws

    # 6. Workspace Memberships
    async def ensure_ws_member(ws_id, user_id):
        m = await session.scalar(select(WorkspaceMember).where(WorkspaceMember.workspace_id == ws_id, WorkspaceMember.user_id == user_id))
        if not m:
            session.add(WorkspaceMember(workspace_id=ws_id, user_id=user_id, role="EDITOR", created_at=now, updated_at=now))
            await session.flush()

    await ensure_ws_member(workspaces["WS_SALES_TEST"].id, users["SALES"].id)
    await ensure_ws_member(workspaces["WS_FINANCE_TEST"].id, users["FINANCE"].id)
    await ensure_ws_member(workspaces["WS_HR_TEST"].id, users["HR"].id)
    await ensure_ws_member(workspaces["WS_SALES_TEST"].id, users["MULTI"].id)
    await ensure_ws_member(workspaces["WS_FINANCE_TEST"].id, users["MULTI"].id)

    # 7. Create Conversations & Messages (unique title to avoid duplication)
    async def ensure_conv_with_msgs(user_id, ws_id, title, user_query, asst_answer):
        conv = await session.scalar(select(Conversation).where(Conversation.title == title, Conversation.workspace_id == ws_id))
        if not conv:
            conv = Conversation(
                id=uuid.uuid4(),
                user_id=user_id,
                workspace_id=ws_id,
                title=title,
                created_at=now,
                updated_at=now,
            )
            session.add(conv)
            await session.flush()
            m_user = ChatMessage(id=uuid.uuid4(), conversation_id=conv.id, role="user", content=user_query, created_at=now)
            m_asst = ChatMessage(id=uuid.uuid4(), conversation_id=conv.id, role="assistant", content=asst_answer, created_at=now)
            session.add_all([m_user, m_asst])
            await session.flush()
        return conv

    sales_conv = await ensure_conv_with_msgs(
        users["SALES"].id,
        workspaces["WS_SALES_TEST"].id,
        "Q2 Sales Thread ABC Discussion",
        "What did ABC Company discuss with Sales?",
        "ABC Company discussed pricing discount and annual enterprise tier with Sales.",
    )

    finance_conv = await ensure_conv_with_msgs(
        users["FINANCE"].id,
        workspaces["WS_FINANCE_TEST"].id,
        "Finance Thread ABC Confidential",
        "What did ABC Company discuss with Finance?",
        "ABC Company discussed confidential financial terms, payment milestones, and tax structures with Finance.",
    )

    hr_conv = await ensure_conv_with_msgs(
        users["HR"].id,
        workspaces["WS_HR_TEST"].id,
        "HR Thread ABC Requirements",
        "What did ABC Company discuss with HR?",
        "ABC Company discussed executive hiring requirements and compensation packages with HR.",
    )

    await session.commit()

    return {
        "users": users,
        "departments": depts,
        "workspaces": workspaces,
        "conversations": {
            "sales": sales_conv,
            "finance": finance_conv,
            "hr": hr_conv,
        },
    }


@pytest.mark.asyncio
async def test_admin_global_conversation_retrieval(db_session: AsyncSession, setup_test_data):
    """
    Test 1: ADMIN asks 'What did ABC Company discuss?' in Global Chat.
    Expected: ADMIN retrieves conversation evidence across Sales, Finance, and HR workspaces.
    """
    data = setup_test_data
    admin_user = data["users"]["ADMIN"]

    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=admin_user,
        workspace_id=None,  # Global Scope
        query="What did ABC Company discuss?",
        limit=10,
    )

    assert len(results) >= 3, "ADMIN Global Chat should retrieve messages from multiple workspaces"
    workspaces_found = {r["workspace_name"] for r in results if r.get("workspace_name")}
    assert "Sales Test Workspace" in workspaces_found
    assert "Finance Test Workspace" in workspaces_found
    assert "HR Test Workspace" in workspaces_found

    contents = " ".join(r["content"] for r in results)
    assert "pricing discount" in contents
    assert "confidential financial terms" in contents
    assert "executive hiring" in contents


@pytest.mark.asyncio
async def test_sales_user_global_conversation_retrieval_isolation(db_session: AsyncSession, setup_test_data):
    """
    Test 2 & 3: SALES_USER asks 'What did ABC Company discuss?' in Global Chat.
    Expected: Can use ONLY authorized Sales information.
    Must NOT retrieve Finance or HR messages.
    """
    data = setup_test_data
    sales_user = data["users"]["SALES"]

    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=sales_user,
        workspace_id=None,  # Global Scope
        query="What did ABC Company discuss?",
        limit=10,
    )

    assert len(results) > 0, "Sales user should retrieve authorized Sales messages"
    for r in results:
        assert r["workspace_name"] == "Sales Test Workspace", f"Unauthorized workspace data leaked: {r['workspace_name']}"
        assert "confidential financial terms" not in r["content"], "Finance content leaked to Sales user"
        assert "executive hiring" not in r["content"], "HR content leaked to Sales user"


@pytest.mark.asyncio
async def test_sales_user_explicit_unauthorized_finance_query(db_session: AsyncSession, setup_test_data):
    """
    Test 5 & 7: SALES_USER explicitly asks 'What did Finance discuss with ABC Company?'.
    Expected: Zero Finance messages/candidates returned. No snippets, no citations, no leaks.
    """
    data = setup_test_data
    sales_user = data["users"]["SALES"]

    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=sales_user,
        workspace_id=None,  # Global Scope
        query="What did Finance discuss with ABC Company?",
        limit=10,
    )

    # All returned results must strictly be from Sales only
    for r in results:
        assert r["workspace_name"] == "Sales Test Workspace"
        assert "Finance Test Workspace" != r["workspace_name"]
        assert "confidential financial terms" not in r["content"]
        assert "tax structures" not in r["content"]


@pytest.mark.asyncio
async def test_finance_user_global_conversation_retrieval(db_session: AsyncSession, setup_test_data):
    """
    Test 4: FINANCE_USER asks 'What did ABC Company discuss?'.
    Expected: Retrieves Finance information, but NOT Sales or HR.
    """
    data = setup_test_data
    fin_user = data["users"]["FINANCE"]

    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=fin_user,
        workspace_id=None,
        query="What did ABC Company discuss?",
        limit=10,
    )

    assert len(results) > 0
    for r in results:
        assert r["workspace_name"] == "Finance Test Workspace"
        assert "Sales Test Workspace" != r["workspace_name"]
        assert "HR Test Workspace" != r["workspace_name"]


@pytest.mark.asyncio
async def test_hr_user_global_conversation_retrieval(db_session: AsyncSession, setup_test_data):
    """
    Test 6: HR_USER asks 'What did ABC Company discuss?'.
    Expected: Retrieves HR information, but NOT Sales or Finance.
    """
    data = setup_test_data
    hr_user = data["users"]["HR"]

    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=hr_user,
        workspace_id=None,
        query="What did ABC Company discuss?",
        limit=10,
    )

    assert len(results) > 0
    for r in results:
        assert r["workspace_name"] == "HR Test Workspace"
        assert "Sales Test Workspace" != r["workspace_name"]
        assert "Finance Test Workspace" != r["workspace_name"]


@pytest.mark.asyncio
async def test_multi_department_user_retrieval(db_session: AsyncSession, setup_test_data):
    """
    Test 16: MULTI_DEPT_USER (Sales + Finance).
    Expected: Retrieves Sales + Finance messages, but zero HR messages.
    """
    data = setup_test_data
    multi_user = data["users"]["MULTI"]

    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=multi_user,
        workspace_id=None,
        query="What did ABC Company discuss?",
        limit=10,
    )

    assert len(results) >= 2
    ws_names = {r["workspace_name"] for r in results}
    assert "Sales Test Workspace" in ws_names
    assert "Finance Test Workspace" in ws_names
    assert "HR Test Workspace" not in ws_names, "HR workspace data must NOT leak to user authorized only for Sales + Finance"


@pytest.mark.asyncio
async def test_workspace_chat_scoping(db_session: AsyncSession, setup_test_data):
    """
    Test 7: Workspace Chat for Sales Workspace.
    Expected: Only retrieves from Sales workspace, does NOT search cross-workspace.
    """
    data = setup_test_data
    sales_user = data["users"]["SALES"]
    sales_ws = data["workspaces"]["WS_SALES_TEST"]

    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=sales_user,
        workspace_id=sales_ws.id,
        query="What did ABC Company discuss?",
        limit=10,
    )

    assert len(results) > 0
    for r in results:
        assert r["workspace_id"] == str(sales_ws.id)
        assert r["workspace_name"] == "Sales Test Workspace"


@pytest.mark.asyncio
async def test_unauthorized_workspace_direct_parameter_manipulation(db_session: AsyncSession, setup_test_data):
    """
    Test 15: SALES_USER directly sends Finance workspace_id to workspace retrieval.
    Expected: Server validates authorization independently and returns 0 candidates (fail-closed).
    """
    data = setup_test_data
    sales_user = data["users"]["SALES"]
    finance_ws = data["workspaces"]["WS_FINANCE_TEST"]

    # Attempt to retrieve from Finance workspace as Sales user
    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=sales_user,
        workspace_id=finance_ws.id,
        query="What did ABC Company discuss?",
        limit=10,
    )

    assert len(results) == 0, "Direct workspace_id parameter manipulation must return 0 candidates"


@pytest.mark.asyncio
async def test_user_with_no_department_access(db_session: AsyncSession, setup_test_data):
    """
    Edge Case 2: USER with no department access.
    Expected: Authorized workspace set = empty -> 0 candidates returned.
    """
    data = setup_test_data
    no_dept_user = data["users"]["NO_DEPT"]

    allowed_ws = await get_allowed_workspaces(db_session, no_dept_user)
    assert allowed_ws == [], "User with no department access must have empty allowed workspaces"

    results = await WorkspaceThreadRetriever.retrieve_workspace_threads(
        session=db_session,
        user=no_dept_user,
        workspace_id=None,
        query="What did ABC Company discuss?",
        limit=10,
    )
    assert len(results) == 0, "User with no access must receive 0 conversation candidates"


@pytest.mark.asyncio
async def test_five_turn_conversation_history_preserved(db_session: AsyncSession, setup_test_data):
    """
    Test 9 & 10: 5-turn (10 messages) conversation history mechanism remains intact.
    All DB messages preserved, while get_conversation_context_window returns latest 5 turns.
    """
    conv_service = ConversationService(db_session)
    user = setup_test_data["users"]["SALES"]
    now = datetime.now(timezone.utc)

    # Create a conversation with 8 turns (16 messages)
    test_conv = Conversation(
        id=uuid.uuid4(),
        user_id=user.id,
        workspace_id=setup_test_data["workspaces"]["WS_SALES_TEST"].id,
        title=f"History Test Conversation {uuid.uuid4()}",
        created_at=now,
        updated_at=now,
    )
    db_session.add(test_conv)

    for i in range(1, 9):
        t_user = now + timedelta(seconds=i * 2)
        t_asst = now + timedelta(seconds=i * 2 + 1)
        db_session.add(ChatMessage(id=uuid.uuid4(), conversation_id=test_conv.id, role="user", content=f"User turn {i}", created_at=t_user))
        db_session.add(ChatMessage(id=uuid.uuid4(), conversation_id=test_conv.id, role="assistant", content=f"Assistant turn {i}", created_at=t_asst))

    await db_session.commit()

    # Verify context window takes exactly latest 5 turns = 10 messages
    window = await conv_service.get_conversation_context_window(conversation_id=test_conv.id, max_turns=5)
    assert len(window) == 10, f"Expected 10 messages (5 turns), got {len(window)}"
    assert window[0]["content"] == "User turn 4"
    assert window[-1]["content"] == "Assistant turn 8"

    # Verify full DB conversation detail still preserves all 16 messages
    detail = await conv_service.get_conversation_detail(conversation_id=test_conv.id, user=user)
    assert len(detail.messages) == 16, f"Expected full 16 messages in DB, got {len(detail.messages)}"


@pytest.mark.asyncio
async def test_prompt_construction_contains_no_unauthorized_data(db_session: AsyncSession, setup_test_data):
    """
    Test 14: Verify that prompt construction with workspace_thread_context renders authorized data
    without leaking internal IDs or unauthorized workspace discussions.
    """
    ws_id = str(setup_test_data["workspaces"]["WS_SALES_TEST"].id)
    sales_thread_context = [
        {
            "source_type": "conversation",
            "workspace_id": ws_id,
            "workspace_name": "Sales Test Workspace",
            "thread_id": str(uuid.uuid4()),
            "thread_title": "Q2 Sales Discussion",
            "message_id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "ABC Company discussed 15% pricing discount with Sales.",
        }
    ]

    messages = build_rag_messages(
        context="Sales document context excerpt.",
        question="What did ABC discuss with Sales?",
        workspace_thread_context=sales_thread_context,
    )

    user_msg_content = messages[-1]["content"]
    assert "Relevant Prior Discussions in Authorized Workspaces:" in user_msg_content
    assert "[Sales Test Workspace] [Thread: \"Q2 Sales Discussion\"] (Assistant): ABC Company discussed 15% pricing discount with Sales." in user_msg_content
    assert "Finance" not in user_msg_content
    assert "HR" not in user_msg_content


@pytest.mark.asyncio
async def test_global_and_workspace_conversation_list_isolation(db_session: AsyncSession, setup_test_data):
    """
    Test 15: Global Chat (workspace_id=None) only lists global conversations (workspace_id IS NULL).
    Workspace Chat (workspace_id=UUID) only lists conversations belonging to that workspace.
    """
    conv_service = ConversationService(db_session)
    user = setup_test_data["users"]["ADMIN"]
    now = datetime.now(timezone.utc)
    sales_ws_id = setup_test_data["workspaces"]["WS_SALES_TEST"].id

    # Create a Global Chat conversation (workspace_id is None)
    global_conv = Conversation(
        id=uuid.uuid4(),
        user_id=user.id,
        workspace_id=None,
        title="Admin Global Chat Conversation",
        created_at=now,
        updated_at=now,
    )
    db_session.add(global_conv)

    # Create a Sales Workspace conversation
    sales_conv = Conversation(
        id=uuid.uuid4(),
        user_id=user.id,
        workspace_id=sales_ws_id,
        title="Admin Sales Workspace Conversation",
        created_at=now,
        updated_at=now,
    )
    db_session.add(sales_conv)
    await db_session.commit()

    # 1. List Global Chat conversations
    global_list = await conv_service.list_conversations(user=user, workspace_id=None)
    global_ids = {c.id for c in global_list}
    assert str(global_conv.id) in global_ids, "Global conversation must appear in Global Chat list"
    assert str(sales_conv.id) not in global_ids, "Sales conversation must NOT appear in Global Chat list"

    # 2. List Sales Workspace conversations
    sales_list = await conv_service.list_conversations(user=user, workspace_id=sales_ws_id)
    sales_ids = {c.id for c in sales_list}
    assert str(sales_conv.id) in sales_ids, "Sales conversation must appear in Sales Workspace list"
    assert str(global_conv.id) not in sales_ids, "Global conversation must NOT appear in Sales Workspace list"


@pytest.mark.asyncio
async def test_backend_scope_validation_in_get_or_create(db_session: AsyncSession, setup_test_data):
    """
    Test 16: Backend rejects requests where conversation_id scope does not match requested workspace_id.
    """
    from fastapi import HTTPException
    conv_service = ConversationService(db_session)
    user = setup_test_data["users"]["ADMIN"]
    now = datetime.now(timezone.utc)
    sales_ws_id = setup_test_data["workspaces"]["WS_SALES_TEST"].id

    # Create a Sales conversation
    sales_conv = Conversation(
        id=uuid.uuid4(),
        user_id=user.id,
        workspace_id=sales_ws_id,
        title="Sales Scoped Conversation",
        created_at=now,
        updated_at=now,
    )
    db_session.add(sales_conv)

    # Create a Global conversation
    global_conv = Conversation(
        id=uuid.uuid4(),
        user_id=user.id,
        workspace_id=None,
        title="Global Scoped Conversation",
        created_at=now,
        updated_at=now,
    )
    db_session.add(global_conv)
    await db_session.commit()

    # Attempt 1: Calling Global Chat (workspace_id=None) with a Sales conversation ID -> Must raise 400
    with pytest.raises(HTTPException) as exc_info:
        await conv_service.get_or_create_conversation(
            user=user,
            conversation_id=sales_conv.id,
            initial_query="Query",
            workspace_id=None,
        )
    assert exc_info.value.status_code == 400
    assert "Scope mismatch" in exc_info.value.detail

    # Attempt 2: Calling Sales Workspace (workspace_id=sales_ws_id) with a Global conversation ID -> Must raise 400
    with pytest.raises(HTTPException) as exc_info:
        await conv_service.get_or_create_conversation(
            user=user,
            conversation_id=global_conv.id,
            initial_query="Query",
            workspace_id=sales_ws_id,
        )
    assert exc_info.value.status_code == 400
    assert "Scope mismatch" in exc_info.value.detail

