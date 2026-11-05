import os
import uuid
import pytest
from sqlalchemy import select

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.db.session import get_db_session
from rag_app.db.models import User, MemoryItem
from rag_app.memory.profile_compiler import compile_user_profile, invalidate_and_recompile_user_profile
from rag_app.generation.prompt import build_rag_messages


@pytest.mark.asyncio
async def test_living_user_profile_lifecycle():
    """Verify empty profile compilation, memory updates, supersession, and prompt injection."""
    async for session in get_db_session():
        # 1. Empty profile
        user = User(
            email=f"profile_lifecycle_{uuid.uuid4().hex[:6]}@local.dev",
            hashed_password="pw",
            role="USER",
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        profile = await compile_user_profile(user.id, session)
        assert profile["memory_version_count"] == 0
        assert profile["user_preferences"] == {}
        assert profile["work_context"] == {}
        assert profile["stable_technical_preferences"] == {}
        assert "last_recompiled_at" in profile

        # 2. Add initial active memories
        m1 = MemoryItem(
            user_id=user.id,
            canonical_key="preference:communication_style",
            memory_type="STATIC",
            status="ACTIVE",
            content="Prefers concise communication and direct tone.",
        )
        m2 = MemoryItem(
            user_id=user.id,
            canonical_key="work:role",
            memory_type="STATIC",
            status="ACTIVE",
            content="Role is Principal Engineer.",
        )
        m3 = MemoryItem(
            user_id=user.id,
            canonical_key="tech:database",
            memory_type="STATIC",
            status="ACTIVE",
            content="Preferred database is PostgreSQL.",
        )
        m4_old_framework = MemoryItem(
            user_id=user.id,
            canonical_key="preference:framework",
            memory_type="STATIC",
            status="ACTIVE",
            content="Django",
            structured_data={"section": "user_preferences", "key": "preferred_frameworks", "value": "Django"},
        )
        session.add_all([m1, m2, m3, m4_old_framework])
        await session.commit()

        # Compile and check
        profile2 = await compile_user_profile(user.id, session)
        assert profile2["memory_version_count"] == 4
        assert profile2["user_preferences"]["communication_style"] == "concise"
        assert profile2["work_context"]["role"] == "Principal Engineer"
        assert profile2["stable_technical_preferences"]["database"] == "PostgreSQL"
        assert "Django" in profile2["user_preferences"]["preferred_frameworks"]

        # 3. Supersede Django with FastAPI
        m4_old_framework.status = "SUPERSEDED"
        m5_new_framework = MemoryItem(
            user_id=user.id,
            canonical_key="preference:framework",
            memory_type="STATIC",
            status="ACTIVE",
            content="FastAPI",
            structured_data={"section": "user_preferences", "key": "preferred_frameworks", "value": "FastAPI"},
        )
        session.add(m5_new_framework)
        await session.commit()

        # Recompile
        updated_profile = await invalidate_and_recompile_user_profile(user.id, session)
        assert "FastAPI" in updated_profile["user_preferences"]["preferred_frameworks"]
        assert "Django" not in updated_profile["user_preferences"]["preferred_frameworks"]

        # 4. Test prompt injection
        messages = build_rag_messages(
            context="Some retrieved context",
            question="What framework do I use?",
            user_profile=updated_profile,
        )
        user_message_content = messages[-1]["content"]
        assert "Living User Profile:" in user_message_content
        assert "FastAPI" in user_message_content
        assert "Principal Engineer" in user_message_content

        from rag_app.db.session import engine
        await engine.dispose()
        break
