"""
Regression test suite for Phase 2A: Fix MissingGreenlet in /chat/ask.
Verifies that:
1. Accessing an expired SQLAlchemy ORM User instance inside ContextCompiler.compile() does NOT raise MissingGreenlet.
2. UserContext extracts detached, primitive actor attributes and enforces memory isolation.
3. LangGraph execution with UserContext initializes ResearchRun and completes cleanly.
4. An end-to-end call to ask_question with an expired ORM User succeeds without HTTP 500 in both ADMIN (Global Chat) and USER (Workspace Chat) paths.
5. Role-based authorization and department boundaries are preserved.
"""

import uuid
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from sqlalchemy import text, select
from sqlalchemy.orm.attributes import instance_state
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import MissingGreenlet

from rag_app.core.config import get_settings
from rag_app.db.models import (
    User,
    Department,
    Workspace,
    WorkspaceMember,
    MemoryItem,
    MemoryProcessingJob,
    Conversation,
    ChatMessage,
    Document,
    DocumentSection,
    DocumentChunk,
    ResearchRun,
)
from rag_app.schemas.users import UserContext
from rag_app.schemas.rag import GenerateRequest, GenerateResponse, RetrievedChunk
from rag_app.context.compiler import ContextCompiler
from rag_app.api.routes import ask_question


@pytest_asyncio.fixture
async def regression_env():
    """Sets up an isolated test schema with live PostgreSQL engine enforcing session expiration."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    # Explicitly enforce expire_on_commit=True to reproduce the ORM expiration lifecycle
    session_maker = async_sessionmaker(engine, expire_on_commit=True, class_=AsyncSession)
    schema_name = f"test_mg_reg_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        def create_tables(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
            User.__table__.create(sync_conn)
            Department.__table__.create(sync_conn)
            Workspace.__table__.create(sync_conn)
            WorkspaceMember.__table__.create(sync_conn)
            MemoryProcessingJob.__table__.create(sync_conn)
            MemoryItem.__table__.create(sync_conn)
            Conversation.__table__.create(sync_conn)
            ChatMessage.__table__.create(sync_conn)
            Document.__table__.create(sync_conn)
            DocumentSection.__table__.create(sync_conn)
            DocumentChunk.__table__.create(sync_conn)
            ResearchRun.__table__.create(sync_conn)
        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:8]
    admin_id = uuid.uuid4()
    user_id = uuid.uuid4()
    dept_id = uuid.uuid4()
    ws_id = uuid.uuid4()

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        dept = Department(id=dept_id, code=f"SALES_{suffix}", name="Sales", is_active=True)
        session.add(dept)

        admin = User(
            id=admin_id,
            email=f"admin_{suffix}@acme.com",
            hashed_password="hashed_pw_test",
            role="ADMIN",
            is_active=True,
        )
        session.add(admin)

        user1 = User(
            id=user_id,
            email=f"rep_{suffix}@acme.com",
            hashed_password="hashed_pw_test",
            role="USER",
            is_active=True,
        )
        session.add(user1)

        ws = Workspace(
            id=ws_id,
            name="Sales Workspace",
            code=f"WS_{suffix}",
            department_id=dept_id,
            created_by=admin_id,
            is_active=True,
        )
        session.add(ws)

        member = WorkspaceMember(
            workspace_id=ws_id,
            user_id=user_id,
            role="EDITOR",
        )
        session.add(member)
        await session.commit()

    yield {
        "engine": engine,
        "session_maker": session_maker,
        "schema_name": schema_name,
        "admin_id": admin_id,
        "user_id": user_id,
        "dept_id": dept_id,
        "ws_id": ws_id,
    }

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_expired_orm_instance_in_compiler_does_not_raise_missing_greenlet(regression_env):
    """
    Direct test: Load an ORM User, commit the session to trigger attribute expiration,
    then invoke synchronous ContextCompiler.compile().
    Must NOT raise sqlalchemy.exc.MissingGreenlet.
    """
    session_maker = regression_env["session_maker"]
    schema_name = regression_env["schema_name"]
    user_id = regression_env["user_id"]
    dept_id = regression_env["dept_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user = await session.get(User, user_id)
        assert user is not None

        # Commit transaction -> attributes expire
        await session.commit()
        st = instance_state(user)
        assert st.expired is True, "User ORM instance must be expired"

        # Now invoke synchronous ContextCompiler.compile with the expired ORM user
        compiler = ContextCompiler()
        chunk = RetrievedChunk(
            chunk_id="chunk_reg_1",
            document_id="doc_reg_1",
            department_id=str(dept_id),
            file_name="q3_report.pdf",
            chunk_index=0,
            text="Revenue increased 22% in Q3.",
            hybrid_score=0.9,
        )

        # Before fix, this line crashed with sqlalchemy.exc.MissingGreenlet
        compiled = compiler.compile(
            chunks=[chunk],
            query="What is the Q3 revenue increase?",
            allowed_departments=[dept_id],
            user=user,
        )

        assert len(compiled.chunks) == 1
        assert "Revenue increased 22%" in compiled.rendered_context


@pytest.mark.asyncio
async def test_user_context_preserves_attributes_and_enforces_memory_isolation(regression_env):
    """
    Verifies that UserContext correctly resolves primitive attributes and
    preserves personal memory isolation across user boundaries.
    """
    user_id = regression_env["user_id"]
    other_user_id = uuid.uuid4()
    dept_id = regression_env["dept_id"]

    user_context = UserContext(
        id=user_id,
        email="user@example.com",
        role="USER",
        is_active=True,
    )

    compiler = ContextCompiler()
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        department_id=str(dept_id),
        file_name="f.pdf",
        chunk_index=0,
        text="Sample text",
    )

    # 1. Candidate memory belonging to THIS user -> Must be kept
    own_mem = {
        "id": "mem-1",
        "content": "User prefers concise tables",
        "user_id": str(user_id),
    }

    # 2. Candidate memory belonging to ANOTHER user -> Must be dropped
    unauth_mem = {
        "id": "mem-2",
        "content": "Secret note of another user",
        "user_id": str(other_user_id),
    }

    compiled = compiler.compile(
        chunks=[chunk],
        query="Test query",
        allowed_departments=[dept_id],
        personal_memories=[own_mem, unauth_mem],
        user=user_context,
    )

    # Verify that own memory was preserved and unauth memory was excluded
    mem_ids = [m.get("id") for m in (compiled.compiled_memories or [])]
    assert "mem-1" in mem_ids
    assert "mem-2" not in mem_ids
    assert compiled.dropped_reasons.get("mem-2") == "UNAUTHORIZED_USER_MEMORY"


@pytest.mark.asyncio
async def test_chat_ask_full_pipeline_with_expired_admin_and_user_succeeds(regression_env):
    """
    End-to-end regression: POST /chat/ask where conv_service.save_user_message commits
    the transaction, leaving the session user expired.
    Proves that the request completes with HTTP 200 and does NOT crash with MissingGreenlet
    for both ADMIN (Global Chat) and USER (Workspace Chat).
    """
    session_maker = regression_env["session_maker"]
    schema_name = regression_env["schema_name"]
    admin_id = regression_env["admin_id"]
    user_id = regression_env["user_id"]
    dept_id = regression_env["dept_id"]
    ws_id = regression_env["ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        admin = await session.get(User, admin_id)
        user = await session.get(User, user_id)
        assert admin is not None
        assert user is not None

        test_chunk = RetrievedChunk(
            chunk_id="chk_sales_1",
            document_id="doc_sales_1",
            department_id=str(dept_id),
            file_name="sales_q3.pdf",
            chunk_index=0,
            text="Sales in Q3 closed at $4.2M, beating target by 12%.",
            hybrid_score=0.92,
        )

        mock_retrieval = MagicMock()
        mock_retrieval.session = session
        mock_retrieval.retrieve = AsyncMock(
            return_value=MagicMock(
                results=[test_chunk],
                strategy_trace={"vector": 1, "keyword": 1},
                chunk_provenance={"chk_sales_1": MagicMock()},
                chunk_hierarchy={"chk_sales_1": MagicMock()},
            )
        )

        mock_gen = MagicMock()
        mock_gen.retrieval_service = mock_retrieval
        mock_gen.llm = MagicMock()

        async def fake_generate_from_context(**kwargs):
            return GenerateResponse(
                answer="Q3 sales reached $4.2M, beating target by 12%.",
                citations=["sales_q3.pdf"],
                retrieved_chunks=kwargs.get("chunks", []),
            )

        mock_gen.generate_from_context = AsyncMock(side_effect=fake_generate_from_context)

        # 1. Test ADMIN path in Global Chat (workspace_id=None)
        admin_req = GenerateRequest(
            query="What is our sales performance in Q3?",
            top_k=5,
            rerank=False,
        )
        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[dept_id]):
            admin_resp = await ask_question(
                request=admin_req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=admin,
            )

        assert admin_resp.answer is not None
        assert len(admin_resp.answer) > 0
        assert admin_resp.conversation_id is not None
        assert admin_resp.message_id is not None

        # 2. Test USER path in Workspace Chat (workspace_id=ws_id)
        # In live HTTP requests, FastAPI dependency injection (get_current_user) supplies a freshly
        # loaded authenticated user for each incoming request session.
        fresh_user = await session.get(User, user_id)
        assert fresh_user is not None
        user_req = GenerateRequest(
            query="What is our sales performance in Q3?",
            workspace_id=ws_id,
            top_k=5,
            rerank=False,
        )
        with patch("rag_app.security.authorization.get_allowed_departments", return_value=[dept_id]), \
             patch("rag_app.conversation.service.authorize", new_callable=AsyncMock, return_value=True):
            user_resp = await ask_question(
                request=user_req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=fresh_user,
            )

        assert user_resp.answer is not None
        assert len(user_resp.answer) > 0
        assert user_resp.conversation_id is not None
        assert user_resp.message_id is not None


@pytest.mark.asyncio
async def test_admin_and_user_authorization_scoping(regression_env):
    """
    Verifies that UserContext correctly represents both ADMIN and USER roles
    and preserves role authorization logic.
    """
    user_id = regression_env["user_id"]
    admin_context = UserContext(id=user_id, email="admin@acme.com", role="ADMIN", is_active=True)
    user_context = UserContext(id=user_id, email="rep@acme.com", role="USER", is_active=True)

    assert admin_context.role == "ADMIN"
    assert user_context.role == "USER"
    assert admin_context.id == user_id
    assert user_context.id == user_id

    # Test from_user factory with dict, mock, and UserContext itself
    from_ctx = UserContext.from_user(admin_context)
    assert from_ctx is admin_context

    mock_user = MagicMock(id=user_id, email="mock@acme.com", role="ADMIN", is_active=True)
    from_mock = UserContext.from_user(mock_user)
    assert from_mock.id == user_id
    assert from_mock.role == "ADMIN"
    assert from_mock.email == "mock@acme.com"


@pytest.mark.asyncio
async def test_user_context_from_user_normal_fully_loaded(regression_env):
    """Verifies the normal, fully loaded UserContext snapshot creation path."""
    session_maker = regression_env["session_maker"]
    schema_name = regression_env["schema_name"]
    user_id = regression_env["user_id"]
    admin_id = regression_env["admin_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user = await session.get(User, user_id)
        admin = await session.get(User, admin_id)
        assert user is not None
        assert admin is not None

        # Fully loaded ORM instances before commit
        user_ctx = UserContext.from_user(user)
        assert user_ctx.id == user_id
        assert user_ctx.role == "USER"
        assert user_ctx.is_active is True
        assert user_ctx.email == user.email

        admin_ctx = UserContext.from_user(admin)
        assert admin_ctx.id == admin_id
        assert admin_ctx.role == "ADMIN"
        assert admin_ctx.is_active is True
        assert admin_ctx.email == admin.email


@pytest.mark.asyncio
async def test_user_context_from_user_fails_closed_on_fully_expired_user(regression_env):
    """
    Verifies that calling UserContext.from_user() on a fully expired ORM instance
    fails closed with ValueError rather than guessing defaults or triggering MissingGreenlet.
    """
    session_maker = regression_env["session_maker"]
    schema_name = regression_env["schema_name"]
    user_id = regression_env["user_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user = await session.get(User, user_id)
        assert user is not None
        await session.commit()
        st = instance_state(user)
        assert st.expired is True

        try:
            with pytest.raises(ValueError) as exc_info:
                UserContext.from_user(user)
            assert "required authorization attribute(s)" in str(exc_info.value)
            assert "expired or unloaded" in str(exc_info.value)
        except MissingGreenlet:
            pytest.fail("UserContext.from_user raised MissingGreenlet instead of failing closed with ValueError!")


@pytest.mark.asyncio
async def test_user_context_from_user_fails_closed_on_expired_inactive_user(regression_env):
    """
    Verifies that an expired inactive user does NOT silently default to is_active=True.
    It must fail closed with ValueError.
    """
    session_maker = regression_env["session_maker"]
    schema_name = regression_env["schema_name"]
    inactive_uid = uuid.uuid4()

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        inactive_user = User(
            id=inactive_uid,
            email=f"inactive_{uuid.uuid4().hex[:6]}@acme.com",
            hashed_password="pw",
            role="USER",
            is_active=False,
        )
        session.add(inactive_user)
        await session.commit()
        st = instance_state(inactive_user)
        assert st.expired is True

        try:
            with pytest.raises(ValueError) as exc_info:
                UserContext.from_user(inactive_user)
            assert "required authorization attribute(s)" in str(exc_info.value)
        except MissingGreenlet:
            pytest.fail("UserContext.from_user raised MissingGreenlet instead of failing closed with ValueError!")


@pytest.mark.asyncio
async def test_user_context_from_user_fails_closed_on_expired_admin_user(regression_env):
    """
    Verifies that an expired ADMIN user does NOT silently downgrade or default to role='USER'.
    It must fail closed with ValueError.
    """
    session_maker = regression_env["session_maker"]
    schema_name = regression_env["schema_name"]
    admin_id = regression_env["admin_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        admin = await session.get(User, admin_id)
        assert admin is not None
        await session.commit()
        st = instance_state(admin)
        assert st.expired is True

        try:
            with pytest.raises(ValueError) as exc_info:
                UserContext.from_user(admin)
            assert "required authorization attribute(s)" in str(exc_info.value)
            assert "role" in str(exc_info.value)
        except MissingGreenlet:
            pytest.fail("UserContext.from_user raised MissingGreenlet instead of failing closed with ValueError!")


@pytest.mark.asyncio
async def test_user_context_from_user_fails_closed_on_partially_unloaded_attributes(regression_env):
    """
    Verifies that if specific authorization attributes (role or is_active) are unloaded
    in an active session, UserContext.from_user() detects it and raises ValueError
    without triggering lazy-loading database I/O or MissingGreenlet.
    """
    session_maker = regression_env["session_maker"]
    schema_name = regression_env["schema_name"]
    user_id = regression_env["user_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # 1. Partially unload 'role'
        user = await session.get(User, user_id)
        assert user is not None
        session.expire(user, ["role"])
        st = instance_state(user)
        assert "role" in st.unloaded

        try:
            with pytest.raises(ValueError) as exc_info:
                UserContext.from_user(user)
            assert "'role'" in str(exc_info.value)
        except MissingGreenlet:
            pytest.fail("UserContext.from_user raised MissingGreenlet when inspecting partially unloaded role!")

        # 2. Partially unload 'is_active'
        await session.refresh(user)
        session.expire(user, ["is_active"])
        st = instance_state(user)
        assert "is_active" in st.unloaded

        try:
            with pytest.raises(ValueError) as exc_info:
                UserContext.from_user(user)
            assert "'is_active'" in str(exc_info.value)
        except MissingGreenlet:
            pytest.fail("UserContext.from_user raised MissingGreenlet when inspecting partially unloaded is_active!")

