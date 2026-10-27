"""
Security and regression test suite for Phase 2B.2:
Conversation Ownership and IDOR Protection.

Verifies:
1. User accessing their own conversation succeeds.
2. User supplying another user's conversation ID is rejected with HTTP 403 Forbidden.
3. Unauthorized attempts do not silently create replacement conversations or mutate foreign conversations.
4. User supplying a non-existent conversation ID is rejected with HTTP 404 Not Found.
5. Malformed conversation ID strings are rejected with HTTP 404 Not Found.
6. Legitimate new conversation creation occurs when no conversation ID is provided.
7. Administrator cannot access another user's private conversation (per policy).
8. End-to-end /chat/ask rejects foreign conversation ID with HTTP 403.
"""

import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import (
    User,
    Department,
    Workspace,
    WorkspaceMember,
    Conversation,
    ChatMessage,
    Permission,
    UserDepartment,
    UserPermission,
)
from rag_app.schemas.users import UserContext
from rag_app.schemas.rag import GenerateRequest, GenerateResponse, RetrievedChunk
from rag_app.conversation.service import ConversationService
from rag_app.api.routes import ask_question


@pytest_asyncio.fixture
async def conv_test_env():
    """Sets up an isolated test schema with PostgreSQL tables."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_conv_auth_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        def create_tables(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
            User.__table__.create(sync_conn)
            Department.__table__.create(sync_conn)
            Workspace.__table__.create(sync_conn)
            WorkspaceMember.__table__.create(sync_conn)
            Permission.__table__.create(sync_conn)
            UserDepartment.__table__.create(sync_conn)
            UserPermission.__table__.create(sync_conn)
            Conversation.__table__.create(sync_conn)
            ChatMessage.__table__.create(sync_conn)
        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:6]
    admin_id = uuid.uuid4()
    user_a_id = uuid.uuid4()
    user_b_id = uuid.uuid4()
    dept_id = uuid.uuid4()
    ws_id = uuid.uuid4()

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        dept = Department(id=dept_id, code=f"SALES_{suffix}", name="Sales Department", is_active=True)
        session.add(dept)

        view_perm = Permission(id=uuid.uuid4(), code="VIEW", description="View files", is_active=True)
        session.add(view_perm)

        admin = User(id=admin_id, email=f"admin_{suffix}@acme.com", hashed_password="pw", role="ADMIN", is_active=True)
        user_a = User(id=user_a_id, email=f"user_a_{suffix}@acme.com", hashed_password="pw", role="USER", is_active=True)
        user_b = User(id=user_b_id, email=f"user_b_{suffix}@acme.com", hashed_password="pw", role="USER", is_active=True)
        session.add_all([admin, user_a, user_b])

        # Assign user department memberships and VIEW permissions
        session.add(UserDepartment(user_id=user_a_id, department_id=dept_id))
        session.add(UserDepartment(user_id=user_b_id, department_id=dept_id))
        await session.flush()

        session.add(UserPermission(user_id=user_a_id, department_id=dept_id, permission_id=view_perm.id))
        session.add(UserPermission(user_id=user_b_id, department_id=dept_id, permission_id=view_perm.id))

        ws = Workspace(
            id=ws_id,
            code=f"sales_ws_{suffix}",
            name="Sales Team Workspace",
            department_id=dept_id,
            created_by=admin_id,
            is_active=True,
        )
        session.add(ws)

        session.add(WorkspaceMember(workspace_id=ws_id, user_id=user_a_id, role="EDITOR"))
        session.add(WorkspaceMember(workspace_id=ws_id, user_id=user_b_id, role="EDITOR"))
        await session.commit()

    yield {
        "engine": engine,
        "session_maker": session_maker,
        "schema_name": schema_name,
        "admin_id": admin_id,
        "user_a_id": user_a_id,
        "user_b_id": user_b_id,
        "ws_id": ws_id,
        "dept_id": dept_id,
    }

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


# ── Test 1: User accessing their own conversation ─────────────────────────────

@pytest.mark.asyncio
async def test_user_access_own_conversation_succeeds(conv_test_env):
    """User accessing their own conversation succeeds and returns the conversation."""
    session_maker = conv_test_env["session_maker"]
    schema_name = conv_test_env["schema_name"]
    user_a_id = conv_test_env["user_a_id"]
    ws_id = conv_test_env["ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user_a = await session.get(User, user_a_id)
        conv_service = ConversationService(session)

        # 1. User A creates a conversation
        conv1 = await conv_service.get_or_create_conversation(
            user=user_a,
            conversation_id=None,
            initial_query="Q3 sales numbers",
            workspace_id=ws_id,
        )
        assert conv1.id is not None
        assert conv1.user_id == user_a_id

        # 2. User A retrieves the same conversation
        conv2 = await conv_service.get_or_create_conversation(
            user=user_a,
            conversation_id=conv1.id,
            initial_query="Follow up on Q3",
            workspace_id=ws_id,
        )
        assert conv2.id == conv1.id
        assert conv2.user_id == user_a_id


# ── Test 2: User supplying another user's conversation ID -> HTTP 403 ─────────

@pytest.mark.asyncio
async def test_user_supplying_foreign_conversation_id_rejected_403(conv_test_env):
    """User A supplying User B's conversation ID must be rejected with HTTP 403 Forbidden."""
    session_maker = conv_test_env["session_maker"]
    schema_name = conv_test_env["schema_name"]
    user_a_id = conv_test_env["user_a_id"]
    user_b_id = conv_test_env["user_b_id"]
    ws_id = conv_test_env["ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user_a = await session.get(User, user_a_id)
        user_b = await session.get(User, user_b_id)
        conv_service = ConversationService(session)

        # User B creates a private conversation
        conv_b = await conv_service.get_or_create_conversation(
            user=user_b,
            conversation_id=None,
            initial_query="User B private financial analysis",
            workspace_id=ws_id,
        )

        # User A tries to supply User B's conversation ID
        with pytest.raises(HTTPException) as exc_info:
            await conv_service.get_or_create_conversation(
                user=user_a,
                conversation_id=conv_b.id,
                initial_query="Attempting unauthorized access",
                workspace_id=ws_id,
            )

        assert exc_info.value.status_code == 403
        assert "Forbidden" in exc_info.value.detail


# ── Test 3: Foreign attempt does not create replacement or mutate foreign ──────

@pytest.mark.asyncio
async def test_foreign_attempt_does_not_create_replacement_or_mutate_foreign(conv_test_env):
    """
    Unauthorized attempt must NOT create a replacement conversation for User A,
    nor mutate User B's conversation title or messages.
    """
    session_maker = conv_test_env["session_maker"]
    schema_name = conv_test_env["schema_name"]
    user_a_id = conv_test_env["user_a_id"]
    user_b_id = conv_test_env["user_b_id"]
    ws_id = conv_test_env["ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user_a = await session.get(User, user_a_id)
        user_b = await session.get(User, user_b_id)
        conv_service = ConversationService(session)

        conv_b = await conv_service.get_or_create_conversation(
            user=user_b,
            conversation_id=None,
            initial_query="Original User B Title",
            workspace_id=ws_id,
        )
        orig_title = conv_b.title

        # Total conversation count before attack
        count_before = await session.scalar(select(func.count(Conversation.id)))

        # User A attempts unauthorized access
        with pytest.raises(HTTPException) as exc_info:
            await conv_service.get_or_create_conversation(
                user=user_a,
                conversation_id=conv_b.id,
                initial_query="Malicious Title Injection",
                workspace_id=ws_id,
            )
        assert exc_info.value.status_code == 403

        # Verify NO replacement conversation was created
        count_after = await session.scalar(select(func.count(Conversation.id)))
        assert count_after == count_before, "No new replacement conversation must be created"

        # Verify User B's conversation was NOT mutated
        await session.refresh(conv_b)
        assert conv_b.title == orig_title
        assert conv_b.user_id == user_b_id


# ── Test 4: User supplying non-existent conversation ID -> HTTP 404 ───────────

@pytest.mark.asyncio
async def test_user_supplying_non_existent_conversation_id_rejected_404(conv_test_env):
    """
    Supplying a non-existent conversation ID must return HTTP 404 Not Found,
    and must NOT silently auto-create a replacement conversation.
    """
    session_maker = conv_test_env["session_maker"]
    schema_name = conv_test_env["schema_name"]
    user_a_id = conv_test_env["user_a_id"]
    ws_id = conv_test_env["ws_id"]
    non_existent_id = uuid.uuid4()

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user_a = await session.get(User, user_a_id)
        conv_service = ConversationService(session)

        count_before = await session.scalar(select(func.count(Conversation.id)))

        with pytest.raises(HTTPException) as exc_info:
            await conv_service.get_or_create_conversation(
                user=user_a,
                conversation_id=non_existent_id,
                initial_query="Non existent conversation test",
                workspace_id=ws_id,
            )

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

        count_after = await session.scalar(select(func.count(Conversation.id)))
        assert count_after == count_before, "Must not silently create a conversation on 404"


# ── Test 5: Malformed conversation ID string -> HTTP 404 ─────────────────────

@pytest.mark.asyncio
async def test_malformed_conversation_id_rejected_404(conv_test_env):
    """Malformed conversation ID strings must be rejected with HTTP 404 without auto-creating."""
    session_maker = conv_test_env["session_maker"]
    schema_name = conv_test_env["schema_name"]
    user_a_id = conv_test_env["user_a_id"]
    ws_id = conv_test_env["ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user_a = await session.get(User, user_a_id)
        conv_service = ConversationService(session)

        with pytest.raises(HTTPException) as exc_info:
            await conv_service.get_or_create_conversation(
                user=user_a,
                conversation_id="not-a-valid-uuid-string",
                initial_query="Query with malformed id",
                workspace_id=ws_id,
            )

        assert exc_info.value.status_code == 404


# ── Test 6: Legitimate new conversation creation when conversation_id is None ──

@pytest.mark.asyncio
async def test_new_conversation_creation_when_no_id_provided(conv_test_env):
    """When conversation_id is None, a new conversation is legitimately created."""
    session_maker = conv_test_env["session_maker"]
    schema_name = conv_test_env["schema_name"]
    user_a_id = conv_test_env["user_a_id"]
    ws_id = conv_test_env["ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user_a = await session.get(User, user_a_id)
        conv_service = ConversationService(session)

        conv = await conv_service.get_or_create_conversation(
            user=user_a,
            conversation_id=None,
            initial_query="New customer discovery",
            workspace_id=ws_id,
        )

        assert conv.id is not None
        assert conv.user_id == user_a_id
        assert conv.workspace_id == ws_id
        assert "New Customer Discovery" in conv.title


# ── Test 7: Administrator cannot access another user's private thread ────────

@pytest.mark.asyncio
async def test_admin_cannot_access_other_users_conversation_403(conv_test_env):
    """
    Per policy, administrators do NOT bypass conversation ownership.
    Attempting to supply another user's conversation ID raises HTTP 403 Forbidden.
    """
    session_maker = conv_test_env["session_maker"]
    schema_name = conv_test_env["schema_name"]
    admin_id = conv_test_env["admin_id"]
    user_a_id = conv_test_env["user_a_id"]
    ws_id = conv_test_env["ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        admin = await session.get(User, admin_id)
        user_a = await session.get(User, user_a_id)
        conv_service = ConversationService(session)

        # User A creates a conversation
        conv_a = await conv_service.get_or_create_conversation(
            user=user_a,
            conversation_id=None,
            initial_query="User A confidential plans",
            workspace_id=ws_id,
        )

        # Admin tries to access User A's conversation via get_or_create_conversation
        with pytest.raises(HTTPException) as exc_info:
            await conv_service.get_or_create_conversation(
                user=admin,
                conversation_id=conv_a.id,
                initial_query="Admin inspecting user thread",
                workspace_id=ws_id,
            )

        assert exc_info.value.status_code == 403
        assert "Forbidden" in exc_info.value.detail


# ── Test 8: End-to-end route /chat/ask rejects foreign conversation ID ────────

@pytest.mark.asyncio
async def test_chat_ask_route_level_idor_rejected_403(conv_test_env):
    """
    Calling /chat/ask route with another user's conversation_id must reject with HTTP 403,
    and must not generate an answer or append messages to foreign thread.
    """
    session_maker = conv_test_env["session_maker"]
    schema_name = conv_test_env["schema_name"]
    user_a_id = conv_test_env["user_a_id"]
    user_b_id = conv_test_env["user_b_id"]
    ws_id = conv_test_env["ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user_a = await session.get(User, user_a_id)
        user_b = await session.get(User, user_b_id)
        conv_service = ConversationService(session)

        # User B creates a conversation
        conv_b = await conv_service.get_or_create_conversation(
            user=user_b,
            conversation_id=None,
            initial_query="User B private topic",
            workspace_id=ws_id,
        )

        mock_gen = MagicMock()
        mock_retrieval = MagicMock()

        # User A sends /chat/ask request referencing User B's conversation ID
        req = GenerateRequest(
            query="Can I see your secret data?",
            conversation_id=str(conv_b.id),
            workspace_id=ws_id,
        )

        with pytest.raises(HTTPException) as exc_info:
            await ask_question(
                request=req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=user_a,
            )

        assert exc_info.value.status_code == 403
        assert "Forbidden" in exc_info.value.detail
        assert not mock_gen.generate_from_context.called, "Generation must not run for unauthorized conversation"
