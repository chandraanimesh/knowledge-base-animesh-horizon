import os
import uuid
import pytest
from sqlalchemy import select

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.db.session import get_db_session
from rag_app.db.models import User, MemoryItem
from rag_app.memory.dreaming import run_memory_dreaming


@pytest.mark.asyncio
async def test_memory_dreaming_lifecycle():
    """Verify threshold check, cluster consolidation, provenance tracking, and living profile update."""
    async for session in get_db_session():
        user = User(
            email=f"dream_lifecycle_{uuid.uuid4().hex[:6]}@local.dev",
            hashed_password="pw",
            role="USER",
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        # 1. Below threshold: 3 items in preference category (threshold is 5) -> skipped
        for i in range(3):
            session.add(
                MemoryItem(
                    user_id=user.id,
                    canonical_key=f"preference:item_{i}",
                    memory_type="DYNAMIC",
                    status="ACTIVE",
                    content=f"Preference fact {i}",
                )
            )
        await session.commit()

        result_low = await run_memory_dreaming(user_id=user.id, session=session, threshold=5)
        assert result_low["clusters_consolidated"] == 0
        assert result_low["memories_consolidated"] == 0

        # 2. Add 5 fragmented memories in tech category (meets threshold=5)
        source_mems = []
        for i in range(5):
            m = MemoryItem(
                user_id=user.id,
                canonical_key=f"tech:item_{i}",
                memory_type="DYNAMIC",
                status="ACTIVE",
                content=f"Technical detail note {i}",
            )
            session.add(m)
            source_mems.append(m)
        await session.commit()

        # Run dreaming with threshold=5
        result = await run_memory_dreaming(user_id=user.id, session=session, threshold=5)
        assert result["clusters_consolidated"] == 1
        assert result["memories_consolidated"] == 5
        assert len(result["new_memory_ids"]) == 1

        new_id = uuid.UUID(result["new_memory_ids"][0])
        dreamt_mem = await session.get(MemoryItem, new_id)
        assert dreamt_mem is not None
        assert dreamt_mem.status == "ACTIVE"
        assert dreamt_mem.structured_data.get("is_consolidated") is True
        assert len(dreamt_mem.structured_data.get("source_memory_ids", [])) == 5

        # Verify sources are marked as SUPERSEDED pointing to dreamt_mem.id
        for m in source_mems:
            await session.refresh(m)
            assert m.status == "SUPERSEDED"
            assert m.superseded_by_id == dreamt_mem.id

        # Verify living profile was automatically updated
        await session.refresh(user)
        assert user.profile_json is not None
        assert "last_recompiled_at" in user.profile_json

        from rag_app.db.session import engine
        await engine.dispose()
        break
