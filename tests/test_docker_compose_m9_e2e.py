"""Docker Compose Integration Verification Suite (Milestone 9)

Tests operational behavior across the running multi-service development stack:
- Database identity check against rag_disposable_test
- Normal chat message to memory processing outbox job
- Background worker extraction & materialization
- Memory retrieval & ContextCompiler integration
- Periodic Dreaming scheduler execution (Step 4 & Step 7)
- Living profile synchronization
- Worker restart & stale job crash recovery
- Delayed-job tombstone anti-resurrection & authorization
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.db.models import (
    User,
    Department,
    Workspace,
    Conversation,
    ChatMessage,
    MemoryItem,
    MemoryRelationship,
    MemoryDreamingRun,
    MemoryProcessingJob,
    MemoryTombstone,
)
from rag_app.memory.jobs import (
    create_memory_job,
    execute_memory_job,
    process_pending_memory_jobs,
    recover_stale_memory_jobs,
)
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.retrieval import MemoryRetrievalService
from rag_app.memory.privacy_service import MemoryPrivacyService
from rag_app.memory.privacy_operations import MemoryPrivacyOperations
from rag_app.memory.profile_compiler import compile_user_profile
from rag_app.memory.schemas import (
    MemoryCandidate,
    MemoryType,
    JobStatus,
    ScopeType,
)
from rag_app.knowledge.scheduler import DurableReconciliationScheduler

test_engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=NullPool)
TestSession = async_sessionmaker(bind=test_engine, expire_on_commit=False, class_=AsyncSession)


@pytest.fixture(autouse=True)
def verify_disposable_database():
    assert "rag_disposable_test" in os.environ["DATABASE_URL"]


async def _create_test_env(session: AsyncSession, name: str):
    dept = Department(id=uuid.uuid4(), code=f"dp_{name}_{uuid.uuid4().hex[:6]}", name=f"Dept {name}")
    session.add(dept)
    await session.flush()
    ws = Workspace(id=uuid.uuid4(), code=f"ws_{name}_{uuid.uuid4().hex[:6]}", name=f"WS {name}", department_id=dept.id)
    session.add(ws)
    await session.flush()
    user = User(id=uuid.uuid4(), email=f"{name}_{uuid.uuid4().hex[:6]}@test.local", hashed_password="pw", role="USER", is_active=True)
    session.add(user)
    await session.flush()
    conv = Conversation(id=uuid.uuid4(), user_id=user.id, workspace_id=ws.id, title=f"Thread {name}")
    session.add(conv)
    await session.flush()
    return user, ws, conv


@pytest.mark.asyncio
async def test_docker_phase1_database_identity():
    """Phase 1: Read-only database identity query confirming rag_disposable_test."""
    async with TestSession() as session:
        res = await session.execute(text("SELECT current_database(), current_user, inet_server_addr(), inet_server_port();"))
        row = res.mappings().one()
        assert row["current_database"] == "rag_disposable_test"
        assert row["current_user"] == "rag"
        assert row["inet_server_port"] == 5432


@pytest.mark.asyncio
async def test_docker_phase4_chat_to_memory_e2e_journey():
    """Phase 4: Normal chat turn to memory outbox job and background materialization."""
    async with TestSession() as session:
        user, ws, conv = await _create_test_env(session, "chat_m9")

        # 1. Normal conversational message without /rem
        user_prompt = "I prefer technical explanations in English with practical examples. My current backend project uses FastAPI and PostgreSQL."
        msg = ChatMessage(id=uuid.uuid4(), conversation_id=conv.id, role="user", content=user_prompt)
        session.add(msg)
        await session.commit()

        # 2. Enqueue memory processing job
        job = await create_memory_job(
            session=session,
            owner_user_id=user.id,
            source_text=user_prompt,
            trigger_kind="conversational_turn",
            workspace_id=ws.id,
            source_message_id=msg.id,
            conversation_id=conv.id,
        )
        assert job.status == JobStatus.PENDING.value
        await session.commit()

        # 3. Background worker executes job
        processed_count = await process_pending_memory_jobs(worker_id="docker-worker-e2e", max_jobs=5, session_factory=TestSession)
        assert processed_count >= 1

        # 4. Refresh job and verify completion
        await session.refresh(job)
        assert job.status == JobStatus.SUCCEEDED.value

        # 5. Verify MemoryItem materialized
        mems = (await session.execute(
            select(MemoryItem).where(MemoryItem.user_id == user.id, MemoryItem.status == "ACTIVE")
        )).scalars().all()
        assert len(mems) >= 1
        active_mem = mems[0]
        assert active_mem.memory_type in ("SEMANTIC", "PROCEDURAL", "STATIC")
        assert active_mem.embedding is not None or "fastapi" in active_mem.content.lower()


@pytest.mark.asyncio
async def test_docker_phase5_memory_retrieval_and_context_compiler():
    """Phase 5: Retrieve relevant memory and compile living profile for context prompt."""
    async with TestSession() as session:
        user, ws, conv = await _create_test_env(session, "retrieval_m9")

        # Materialize preference
        await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
            user_id=user.id,
            content="Technical explanations in English with practical examples",
            canonical_key="MEMORY:STATIC:pref:style_english_examples",
            memory_type=MemoryType.STATIC,
        ), auto_embed=False)
        await session.commit()

        # Query memory
        retrieval_res = await MemoryRetrievalService.retrieve(session, user.id, query="How should you explain my backend implementation?")
        assert retrieval_res.total_count >= 1

        # Check compiled profile
        profile = await compile_user_profile(user.id, session)
        assert profile is not None
        assert profile.get("user_preferences") is not None


@pytest.mark.asyncio
async def test_docker_phase6_dreaming_scheduler_automatic_trigger():
    """Phase 6: Automatic Dreaming consolidation via DurableReconciliationScheduler."""
    async with TestSession() as session:
        user, ws, _ = await _create_test_env(session, "sched_dream")
        await MemoryPrivacyService.set_policy(
            session=session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_dreaming_enabled=True,
            memory_collection_enabled=True,
        )

        # Create 4 related memories for consolidation
        for i in range(4):
            await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
                user_id=user.id,
                content=f"Engineering convention {i}: enforce strict typing",
                canonical_key=f"MEMORY:SEMANTIC:pref:typing_{i}",
                memory_type=MemoryType.SEMANTIC,
            ), auto_embed=False)
        await session.commit()

        # Run scheduler cycle
        sched = DurableReconciliationScheduler(interval_seconds=60, session_factory=TestSession)
        metrics = await sched.run_once()
        assert metrics["dreamed_users"] >= 1

        # Check durable MemoryDreamingRun record
        runs = (await session.execute(
            select(MemoryDreamingRun).where(MemoryDreamingRun.user_id == user.id)
        )).scalars().all()
        assert len(runs) >= 1
        assert runs[-1].status == "COMPLETED"


@pytest.mark.asyncio
async def test_docker_phase7_worker_stale_job_recovery():
    """Phase 7: Stale job crash recovery by scheduler."""
    async with TestSession() as session:
        user, ws, conv = await _create_test_env(session, "stale_rec")
        now = datetime.now(timezone.utc)
        job = MemoryProcessingJob(
            id=uuid.uuid4(),
            owner_user_id=user.id,
            source_text_snapshot="Crashed worker uncommitted turn",
            trigger_kind="conversational_turn",
            status=JobStatus.PROCESSING.value,
            attempt_count=1,
            max_attempts=3,
            locked_at=now - timedelta(minutes=10),
            locked_by="crashed-container-42",
            created_at=now - timedelta(minutes=11),
            updated_at=now - timedelta(minutes=10),
        )
        session.add(job)
        await session.commit()

        recovered = await recover_stale_memory_jobs(session, lock_timeout_seconds=180)
        assert recovered >= 1
        await session.refresh(job)
        assert job.status == JobStatus.RETRY_PENDING.value


@pytest.mark.asyncio
async def test_docker_phase10_delayed_job_tombstone_anti_resurrection():
    """Phase 10: Forgotten fact creates tombstone that prevents delayed job resurrection."""
    async with TestSession() as session:
        user, ws, conv = await _create_test_env(session, "antires")
        canonical_key = "MEMORY:STATIC:pref:secret_codename"

        # Register tombstone (simulating user forget request)
        await MemoryPrivacyOperations.register_tombstone(session, user.id, canonical_key, "User forgot secret")
        await session.commit()

        # Delayed candidate attempts materialization
        cand = MemoryCandidate(
            user_id=user.id,
            content="Project secret codename is Apollo",
            canonical_key=canonical_key,
            memory_type=MemoryType.STATIC,
        )
        res = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert res is None, "Tombstone must permanently block candidate materialization!"


@pytest.mark.asyncio
async def test_docker_phase3_memory_routes_query_limits():
    """Phase 3: Verify _normalize_limit prevents unbounded queries and clamps valid limits."""
    from rag_app.api.memory_routes import _normalize_limit
    from fastapi import Query

    # 1. Default & None
    assert _normalize_limit(Query(50)) == 50
    assert _normalize_limit(None, 50) == 50

    # 2. Custom valid
    assert _normalize_limit(25) == 25
    assert _normalize_limit("35") == 35

    # 3. Minimum bound (< 1 clamps to 1)
    assert _normalize_limit(0) == 1
    assert _normalize_limit(-10) == 1

    # 4. Maximum bound (> 500 clamps to 500)
    assert _normalize_limit(999999) == 500

    # 5. Invalid string (falls back to clamped default)
    assert _normalize_limit("invalid_string", 50) == 50

