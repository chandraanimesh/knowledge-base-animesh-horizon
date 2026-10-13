import uuid

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_app.conversation.service import ConversationService
from rag_app.core.config import get_settings
from rag_app.db.models import ChatMessage, Conversation, User


@pytest_asyncio.fixture
async def db_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_global_threads_are_persistent_admin_only_and_owner_scoped(db_session: AsyncSession):
    """Global threads use the existing conversation store without cross-admin access."""
    suffix = uuid.uuid4().hex
    owner = User(
        email=f"global_owner_{suffix}@test.local",
        hashed_password="test",
        role="ADMIN",
        is_active=True,
    )
    other_admin = User(
        email=f"global_other_{suffix}@test.local",
        hashed_password="test",
        role="ADMIN",
        is_active=True,
    )
    standard_user = User(
        email=f"global_user_{suffix}@test.local",
        hashed_password="test",
        role="USER",
        is_active=True,
    )
    db_session.add_all([owner, other_admin, standard_user])
    await db_session.commit()
    user_ids = [owner.id, other_admin.id, standard_user.id]

    service = ConversationService(db_session)
    conversation_ids: list[uuid.UUID] = []
    try:
        first = await service.create_conversation(owner, title="First Global Thread")
        second = await service.create_conversation(owner, title="Second Global Thread")
        conversation_ids.extend([uuid.UUID(first.id), uuid.UUID(second.id)])

        await service.save_user_message(uuid.UUID(first.id), "Remember this thread only.")
        await service.save_assistant_message(uuid.UUID(first.id), "This reply belongs only to the first thread.")
        await service.save_user_message(uuid.UUID(second.id), "This is a separate thread.")

        # Commit + expire simulates reloading the persisted conversation after a refresh.
        await db_session.commit()
        db_session.expire_all()
        owner = await db_session.get(User, user_ids[0])
        other_admin = await db_session.get(User, user_ids[1])
        standard_user = await db_session.get(User, user_ids[2])

        listed = await service.list_conversations(owner)
        listed_ids = {item.id for item in listed}
        assert {first.id, second.id}.issubset(listed_ids)

        reopened_first = await service.get_conversation_detail(first.id, owner)
        reopened_second = await service.get_conversation_detail(second.id, owner)
        assert [message.content for message in reopened_first.messages] == [
            "Remember this thread only.",
            "This reply belongs only to the first thread.",
        ]
        assert [message.content for message in reopened_second.messages] == [
            "This is a separate thread.",
        ]

        # A different administrator cannot read or delete a private admin thread.
        with pytest.raises(HTTPException) as read_error:
            await service.get_conversation_detail(first.id, other_admin)
        assert read_error.value.status_code == 404
        with pytest.raises(HTTPException) as delete_error:
            await service.delete_conversation(first.id, other_admin)
        assert delete_error.value.status_code == 404

        # Direct service/API integration points reject non-admin Global Chat access.
        for operation in (
            lambda: service.list_conversations(standard_user),
            lambda: service.create_conversation(standard_user, title="Not allowed"),
            lambda: service.get_or_create_conversation(
                user=standard_user,
                conversation_id=None,
                initial_query="Not allowed",
                workspace_id=None,
            ),
        ):
            with pytest.raises(HTTPException) as denied:
                await operation()
            assert denied.value.status_code == 403
    finally:
        if conversation_ids:
            await db_session.execute(
                delete(ChatMessage).where(ChatMessage.conversation_id.in_(conversation_ids))
            )
            await db_session.execute(
                delete(Conversation).where(Conversation.id.in_(conversation_ids))
            )
        await db_session.execute(
            delete(User).where(User.id.in_(user_ids))
        )
        await db_session.commit()
