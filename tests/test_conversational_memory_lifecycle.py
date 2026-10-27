"""
Lifecycle, idempotency, retry, concurrency, and cross-user isolation tests
for automatic conversational memory extraction (Phase B).
"""

import asyncio
from datetime import datetime, timezone, timedelta
import uuid
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import (
    User,
    Department,
    Workspace,
    MemoryItem,
    MemoryProcessingJob,
    Conversation,
    ChatMessage,
    Document,
    DocumentSection,
    DocumentChunk,
)
from rag_app.memory.schemas import JobStatus, MemoryCandidate, MemoryType, StabilityTier
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.jobs import (
    create_memory_job,
    execute_memory_job,
    claim_next_memory_job,
    recover_stale_memory_jobs,
    process_pending_memory_jobs,
)
from rag_app.schemas.rag import GenerateRequest, GenerateResponse, RetrievalResponse
from rag_app.api.routes import ask_question


@pytest_asyncio.fixture
async def lifecycle_env():
    """Sets up an isolated test schema with live PostgreSQL engine."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_mem_lc_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        def create_tables(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
            MemoryProcessingJob.__table__.create(sync_conn)
            MemoryItem.__table__.create(sync_conn)
            Conversation.__table__.create(sync_conn)
            ChatMessage.__table__.create(sync_conn)
            Document.__table__.create(sync_conn)
            DocumentSection.__table__.create(sync_conn)
            DocumentChunk.__table__.create(sync_conn)
        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:8]
    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        dept = Department(name=f"MemLC Dept {suffix}", code=f"MLC_{suffix}")
        session.add(dept)
        await session.flush()

        user1 = User(email=f"u1_{suffix}@example.com", hashed_password="pw", role="ADMIN")
        user2 = User(email=f"u2_{suffix}@example.com", hashed_password="pw", role="USER")
        session.add_all([user1, user2])
        await session.flush()

        ws1 = Workspace(name=f"WS1 {suffix}", code=f"W1_{suffix}", department_id=dept.id, created_by=user1.id)
        session.add(ws1)
        await session.commit()

        ws1_id = ws1.id
        u1_id = user1.id
        u2_id = user2.id
        dept_id = dept.id

        yield {
            "session_maker": session_maker,
            "user1": user1,
            "user2": user2,
            "ws1": ws1,
            "dept": dept,
            "schema_name": schema_name,
            "ws1_id": ws1_id,
            "u1_id": u1_id,
            "u2_id": u2_id,
            "dept_id": dept_id,
        }

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        await conn.execute(text("DELETE FROM workspaces WHERE id = :wid"), {"wid": ws1_id})
        await conn.execute(text("DELETE FROM users WHERE id IN (:u1, :u2)"), {"u1": u1_id, "u2": u2_id})
        await conn.execute(text("DELETE FROM departments WHERE id = :did"), {"did": dept_id})
    await engine.dispose()


@pytest.mark.asyncio
async def test_memorable_message_creates_one_durable_job(lifecycle_env):
    """
    Test 1: When ENABLE_CONVERSATIONAL_MEMORY_EXTRACTION=True, an ordinary memorable
    statement creates exactly one durable MemoryProcessingJob without blocking RAG.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResponse(results=[], query="test"))

        mock_gen = MagicMock()
        mock_gen.generate_from_context = AsyncMock(
            return_value=GenerateResponse(answer="Understood. I will remember your preference.", citations=[], retrieved_chunks=[])
        )

        req = GenerateRequest(
            query="From now on, I prefer all financial summaries in concise bullet points with INR currency.",
            workspace_id=None,
        )

        with patch("rag_app.api.routes.get_settings") as mock_settings, \
             patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            cfg = get_settings()
            mock_settings.return_value = cfg.model_copy(update={"enable_conversational_memory_extraction": True})

            resp = await ask_question(
                request=req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=user1,
            )

        assert resp.answer is not None
        assert len(resp.answer) > 0

        # Verify durable job was persisted in DB
        stmt = select(MemoryProcessingJob).where(MemoryProcessingJob.owner_user_id == user1.id)
        jobs = (await session.execute(stmt)).scalars().all()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.trigger_kind == "conversational_turn"
        assert job.status == JobStatus.PENDING.value
        assert "concise bullet points" in job.source_text_snapshot
        assert job.source_message_id is not None


@pytest.mark.asyncio
async def test_greeting_and_transient_smalltalk_ignored(lifecycle_env):
    """
    Test 2: Transient conversation (greetings, simple questions, chit-chat)
    produces zero memory extraction jobs.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResponse(results=[], query="test"))
        mock_gen = MagicMock()
        mock_gen.generate_from_context = AsyncMock(
            return_value=GenerateResponse(answer="Hello! How can I help you?", citations=[], retrieved_chunks=[])
        )

        transient_queries = [
            "Hello, good morning!",
            "How are you doing today?",
            "Thanks a lot, that was helpful",
        ]

        with patch("rag_app.api.routes.get_settings") as mock_settings, \
             patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            cfg = get_settings()
            mock_settings.return_value = cfg.model_copy(update={"enable_conversational_memory_extraction": True})

            for q in transient_queries:
                await ask_question(
                    request=GenerateRequest(query=q, workspace_id=None),
                    service=mock_gen,
                    retrieval_service=mock_retrieval,
                    session=session,
                    user=user1,
                )

        stmt = select(MemoryProcessingJob).where(MemoryProcessingJob.owner_user_id == user1.id)
        jobs = (await session.execute(stmt)).scalars().all()
        assert len(jobs) == 0, "Transient queries must not create memory extraction jobs"


@pytest.mark.asyncio
async def test_feature_flag_disabled_creates_no_jobs(lifecycle_env):
    """
    Test 3: When ENABLE_CONVERSATIONAL_MEMORY_EXTRACTION=False (the default),
    ordinary conversational statements produce zero extraction jobs.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResponse(results=[], query="test"))
        mock_gen = MagicMock()
        mock_gen.generate_from_context = AsyncMock(
            return_value=GenerateResponse(answer="Got it.", citations=[], retrieved_chunks=[])
        )

        req = GenerateRequest(
            query="I work as a senior backend engineer in Bangalore.",
            workspace_id=None,
        )

        with patch("rag_app.api.routes.get_settings") as mock_settings, \
             patch("rag_app.security.authorization.get_allowed_departments", return_value=[]):
            cfg = get_settings()
            mock_settings.return_value = cfg.model_copy(update={"enable_conversational_memory_extraction": False})

            await ask_question(
                request=req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=user1,
            )

        stmt = select(MemoryProcessingJob).where(MemoryProcessingJob.owner_user_id == user1.id)
        jobs = (await session.execute(stmt)).scalars().all()
        assert len(jobs) == 0, "Disabled feature flag must not enqueue memory jobs"


@pytest.mark.asyncio
async def test_job_creation_idempotency_per_source_message(lifecycle_env):
    """
    Test 4: Creating a job with identical (source_message_id, trigger_kind) is idempotent.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        msg_id = uuid.uuid4()
        job1 = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text="I prefer TypeScript for frontend development.",
            trigger_kind="conversational_turn",
            source_message_id=msg_id,
        )
        await session.commit()

        # Repeated creation with same source_message_id
        job2 = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text="I prefer TypeScript for frontend development.",
            trigger_kind="conversational_turn",
            source_message_id=msg_id,
        )

        assert job1.id == job2.id
        stmt = select(MemoryProcessingJob).where(MemoryProcessingJob.source_message_id == msg_id)
        all_jobs = (await session.execute(stmt)).scalars().all()
        assert len(all_jobs) == 1


@pytest.mark.asyncio
async def test_sensitive_info_redacted_and_skipped(lifecycle_env):
    """
    Test 5: Sensitive credentials, passwords, or PII are redacted and never stored
    as reusable memory items.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        job = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text="My credit card number is 4111222233334444 and my password is secretPass123",
            trigger_kind="conversational_turn",
        )
        await session.commit()

        status, items, reason = await execute_memory_job(session=session, job_id=job.id)

        assert status == JobStatus.SKIPPED_NOT_MEMORABLE
        assert len(items) == 0
        assert "sensitive" in (reason or "").lower()

        # Check job in DB
        refreshed = await session.get(MemoryProcessingJob, job.id)
        assert refreshed.source_text_snapshot == "[REDACTED_SENSITIVE_CONTENT]"
        assert refreshed.status == JobStatus.SKIPPED_NOT_MEMORABLE.value


@pytest.mark.asyncio
async def test_stale_lock_recovery(lifecycle_env):
    """
    Test 6: Jobs stuck in PROCESSING due to worker crash or timeout are safely recovered
    back to RETRY_PENDING.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        old_lock_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        stale_job = MemoryProcessingJob(
            id=uuid.uuid4(),
            owner_user_id=user1.id,
            source_text_snapshot="I prefer dark mode in all editors",
            trigger_kind="explicit_rem",
            status=JobStatus.PROCESSING.value,
            attempt_count=1,
            max_attempts=3,
            locked_at=old_lock_time,
            locked_by="crashed-worker-pid-999",
        )
        session.add(stale_job)
        await session.commit()

        recovered_count = await recover_stale_memory_jobs(session=session, lock_timeout_seconds=180)
        assert recovered_count == 1

        refreshed = await session.get(MemoryProcessingJob, stale_job.id)
        assert refreshed.status == JobStatus.RETRY_PENDING.value
        assert refreshed.locked_at is None
        assert refreshed.locked_by is None


@pytest.mark.asyncio
async def test_bounded_retries_reach_dead_letter(lifecycle_env):
    """
    Test 7: A failing job retries with backoff and moves to DEAD_LETTER after max_attempts.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        job = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text="I prefer PostgreSQL for relational storage",
            trigger_kind="conversational_turn",
        )
        job.max_attempts = 2
        await session.commit()

        # Mock extractor to simulate an exception
        mock_extractor = MagicMock()
        mock_extractor.extract_from_conversation = AsyncMock(side_effect=RuntimeError("LLM API connection failed"))

        # Attempt 1
        status1, _, _ = await execute_memory_job(session=session, job_id=job.id, extractor=mock_extractor)
        assert status1 == JobStatus.RETRY_PENDING
        refreshed1 = await session.get(MemoryProcessingJob, job.id)
        assert refreshed1.attempt_count == 1
        assert refreshed1.next_attempt_at is not None

        # Attempt 2 (reaches max_attempts)
        status2, _, _ = await execute_memory_job(session=session, job_id=job.id, extractor=mock_extractor)
        assert status2 == JobStatus.DEAD_LETTER
        refreshed2 = await session.get(MemoryProcessingJob, job.id)
        assert refreshed2.attempt_count == 2
        assert refreshed2.status == JobStatus.DEAD_LETTER.value


@pytest.mark.asyncio
async def test_materializer_updates_and_supersedes_conflicting_memory(lifecycle_env):
    """
    Test 8: Changing preferences supersedes the old memory and creates an updated active memory.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # Step 1: User says "I live in Berlin"
        job1 = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text="/rem I live in Berlin",
            trigger_kind="explicit_rem",
        )
        await session.commit()
        await execute_memory_job(session=session, job_id=job1.id, auto_embed=False)

        mems1 = await MemoryMaterializer.get_user_memories(session=session, user_id=user1.id)
        assert len(mems1) == 1
        assert "Berlin" in mems1[0].content
        old_mem_id = mems1[0].id

        # Step 2: User updates to "I moved to Munich"
        job2 = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text="/rem I moved to Munich",
            trigger_kind="explicit_rem",
        )
        await session.commit()
        await execute_memory_job(session=session, job_id=job2.id, auto_embed=False)

        # Active memories should only contain Munich
        active_mems = await MemoryMaterializer.get_user_memories(session=session, user_id=user1.id)
        assert len(active_mems) == 1
        assert "Munich" in active_mems[0].content

        # Old memory should be SUPERSEDED
        old_mem = await session.get(MemoryItem, old_mem_id)
        assert old_mem.status == "SUPERSEDED"
        assert old_mem.superseded_by_id == active_mems[0].id


@pytest.mark.asyncio
async def test_cross_user_isolation(lifecycle_env):
    """
    Test 9: Memories extracted for User 1 are strictly invisible to User 2,
    even within the same workspace.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    user2 = lifecycle_env["user2"]
    ws1 = lifecycle_env["ws1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # User 1 memory
        cand1 = MemoryCandidate(
            user_id=user1.id,
            workspace_id=ws1.id,
            content="User 1 private salary requirement is 120k",
            canonical_key="MEMORY:STATIC:salary_expectation",
            memory_type=MemoryType.STATIC,
        )
        await MemoryMaterializer.materialize_memory(session=session, candidate=cand1, auto_embed=False)
        await session.commit()

        # Query User 1 memories
        u1_mems = await MemoryMaterializer.get_user_memories(session=session, user_id=user1.id, workspace_id=ws1.id)
        assert len(u1_mems) == 1

        # Query User 2 memories in same workspace
        u2_mems = await MemoryMaterializer.get_user_memories(session=session, user_id=user2.id, workspace_id=ws1.id)
        assert len(u2_mems) == 0, "User 2 must not see User 1's memory"


@pytest.mark.asyncio
async def test_disabled_flag_pauses_conversational_job_processing(lifecycle_env):
    """
    Test 10: When ENABLE_CONVERSATIONAL_MEMORY_EXTRACTION=False, worker does NOT claim
    conversational jobs, leaving them safely queued in the database.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        job = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text="I work as a senior backend engineer",
            trigger_kind="conversational_turn",
        )
        await session.commit()

        # Run claim with flag disabled
        with patch("rag_app.memory.jobs.get_settings") as mock_settings:
            cfg = get_settings()
            mock_settings.return_value = cfg.model_copy(update={"enable_conversational_memory_extraction": False})

            claimed = await claim_next_memory_job(session=session, worker_id="test-worker")
            assert claimed is None, "Conversational job must not be claimed when flag is disabled"

        # Refreshed job must still be PENDING (not lost or aborted)
        refreshed = await session.get(MemoryProcessingJob, job.id)
        assert refreshed.status == JobStatus.PENDING.value


@pytest.mark.asyncio
async def test_atomic_message_and_job_durability(lifecycle_env):
    """
    Test 11: Fix 1 - Atomic Message and Job Durability.
    Verifies that ChatMessage and MemoryProcessingJob are committed together in one transaction.
    Simulates a failure during job creation: verifies rollback ensures zero orphaned messages.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    ws1 = lifecycle_env["ws1"]
    schema_name = lifecycle_env["schema_name"]

    from rag_app.conversation.service import ConversationService

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        conv = Conversation(id=uuid.uuid4(), user_id=user1.id, workspace_id=ws1.id, title="Atomic Durability Conv")
        session.add(conv)
        await session.commit()

        conv_service = ConversationService(session)

        # 1. Normal atomic path: message and job commit together
        msg = await conv_service.save_user_message(
            conversation_id=conv.id,
            content="I am a Senior DevOps Engineer based in Chicago",
            commit=False,
        )
        job = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text=msg.content,
            trigger_kind="conversational_turn",
            workspace_id=ws1.id,
            source_message_id=msg.id,
            conversation_id=conv.id,
        )
        await session.commit()

        # Assert both records are persisted
        assert await session.get(ChatMessage, msg.id) is not None
        assert await session.get(MemoryProcessingJob, job.id) is not None

        # 2. Simulated failure path: error before commit -> rollback -> zero orphaned messages
        with pytest.raises(RuntimeError):
            fail_msg = await conv_service.save_user_message(
                conversation_id=conv.id,
                content="This message will fail before atomic commit",
                commit=False,
            )
            # Simulated crash/failure before commit
            raise RuntimeError("Simulated database/process crash before commit")

        await session.rollback()
        # Verify uncommitted message was rolled back
        check_msg = await session.scalar(select(ChatMessage).where(ChatMessage.content == "This message will fail before atomic commit"))
        assert check_msg is None, "Message must not be persisted if transaction rolled back"


@pytest.mark.asyncio
async def test_forgotten_memory_not_recreated_by_delayed_job(lifecycle_env):
    """
    Test 12: Fix 2 - Prevent Forgotten Memory from Being Recreated.
    1. User states a fact -> memory materialized (ACTIVE).
    2. A delayed retry job exists for that message.
    3. User runs /forget -> memory status becomes ARCHIVED (tombstone updated_at set).
    4. Delayed job executes -> materializer skips it, status becomes SKIPPED_NOT_MEMORABLE.
    5. Fact remains NOT retrievable.
    6. Genuinely new statement made AFTER forget CAN be materialized.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    ws1 = lifecycle_env["ws1"]
    schema_name = lifecycle_env["schema_name"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # Step 1: Create source message and initial job
        now = datetime.now(timezone.utc)
        conv = Conversation(id=uuid.uuid4(), user_id=user1.id, workspace_id=ws1.id, title="Forget Test Conv")
        session.add(conv)
        await session.flush()

        msg = ChatMessage(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="I live in Amsterdam", created_at=now)
        session.add(msg)
        await session.flush()

        cand = MemoryCandidate(
            user_id=user1.id,
            workspace_id=ws1.id,
            content="User lives in Amsterdam",
            canonical_key="MEMORY:STATIC:user_identity:location",
            memory_type=MemoryType.STATIC,
            source_quote="I live in Amsterdam",
            message_id=str(msg.id),
        )

        mem_item = await MemoryMaterializer.materialize_memory(session=session, candidate=cand, auto_embed=False)
        assert mem_item is not None
        assert mem_item.status == "ACTIVE"
        await session.commit()

        # Step 2: Queue a delayed job referencing this turn
        delayed_job = await create_memory_job(
            session=session,
            owner_user_id=user1.id,
            source_text="I live in Amsterdam",
            trigger_kind="explicit_rem",
            workspace_id=ws1.id,
            source_message_id=msg.id,
            conversation_id=conv.id,
        )
        delayed_job.created_at = now  # Originated before forget
        delayed_job.status = JobStatus.RETRY_PENDING.value
        await session.commit()

        # Step 3: User calls /forget
        await asyncio.sleep(0.05)  # Ensure distinct timestamp
        archived = await MemoryMaterializer.archive_memory(
            session=session,
            memory_id=mem_item.id,
            user_id=user1.id,
            workspace_id=ws1.id,
        )
        assert archived is True
        await session.commit()

        # Verify memory is ARCHIVED and not active
        active_mems = await MemoryMaterializer.get_user_memories(session=session, user_id=user1.id, workspace_id=ws1.id)
        assert len(active_mems) == 0, "Archived memory must not be in active memories"

        # Step 4: Delayed job runs now
        status, items, reason = await execute_memory_job(
            session=session,
            job_id=delayed_job.id,
            auto_embed=False,
        )
        await session.commit()

        assert status == JobStatus.SKIPPED_NOT_MEMORABLE
        assert len(items) == 0

        # Step 5: Verify the forgotten fact was NOT resurrected as ACTIVE
        active_after = await MemoryMaterializer.get_user_memories(session=session, user_id=user1.id, workspace_id=ws1.id)
        assert len(active_after) == 0, "Forgotten fact must NEVER be recreated by delayed job"

        # Step 6: Genuinely new statement made AFTER the forget operation
        await asyncio.sleep(0.05)
        new_now = datetime.now(timezone.utc)
        new_msg = ChatMessage(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="I moved to Rotterdam", created_at=new_now)
        session.add(new_msg)
        await session.flush()

        new_cand = MemoryCandidate(
            user_id=user1.id,
            workspace_id=ws1.id,
            content="User lives in Rotterdam",
            canonical_key="MEMORY:STATIC:user_identity:location",
            memory_type=MemoryType.STATIC,
            source_quote="I moved to Rotterdam",
            message_id=str(new_msg.id),
        )

        new_mem = await MemoryMaterializer.materialize_memory(session=session, candidate=new_cand, auto_embed=False)
        await session.commit()

        assert new_mem is not None
        assert new_mem.status == "ACTIVE"
        assert "Rotterdam" in new_mem.content


@pytest.mark.asyncio
async def test_concurrent_job_creation_idempotency_postgres(lifecycle_env):
    """
    Test 13: Fix 3 - PostgreSQL Engine-Level Idempotency.
    Proves that concurrent attempts to create the same (source_message_id, trigger_kind)
    cannot create duplicate rows. Both callers resolve cleanly to the same job.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    ws1 = lifecycle_env["ws1"]
    schema_name = lifecycle_env["schema_name"]

    source_msg_id = uuid.uuid4()
    trigger = "conversational_turn"

    async def try_create(worker_num: int):
        async with session_maker() as session:
            await session.execute(text(f"SET search_path TO {schema_name}, public"))
            job = await create_memory_job(
                session=session,
                owner_user_id=user1.id,
                source_text="Concurrent worker test statement",
                trigger_kind=trigger,
                workspace_id=ws1.id,
                source_message_id=source_msg_id,
            )
            await session.commit()
            return job.id

    # Execute two simultaneous creation calls concurrently
    results = await asyncio.gather(try_create(1), try_create(2))
    assert results[0] == results[1], "Both concurrent callers must resolve to the identical job ID"

    # Verify exactly one row in PostgreSQL
    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        stmt = select(MemoryProcessingJob).where(
            MemoryProcessingJob.source_message_id == source_msg_id,
            MemoryProcessingJob.trigger_kind == trigger,
        )
        all_jobs = (await session.execute(stmt)).scalars().all()
        assert len(all_jobs) == 1, f"Expected exactly 1 job row in PostgreSQL, found {len(all_jobs)}"


@pytest.mark.asyncio
async def test_scheduler_reconciles_missing_conversational_jobs(lifecycle_env):
    """
    Test 14: Scheduler reconciliation discovers committed user messages that missed job creation.
    """
    session_maker = lifecycle_env["session_maker"]
    user1 = lifecycle_env["user1"]
    ws1 = lifecycle_env["ws1"]
    schema_name = lifecycle_env["schema_name"]

    from rag_app.memory.jobs import reconcile_missing_conversational_memory_jobs

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        conv = Conversation(id=uuid.uuid4(), user_id=user1.id, workspace_id=ws1.id, title="Reconcile Test Conv")
        session.add(conv)
        await session.flush()

        # Add a memorable user message without creating a memory job (simulating crash window)
        orphaned_msg = ChatMessage(
            id=uuid.uuid4(),
            conversation_id=conv.id,
            role="user",
            content="I prefer dark mode in all applications",
            created_at=datetime.now(timezone.utc),
        )
        session.add(orphaned_msg)
        await session.commit()

        # Run reconciliation with flag enabled
        with patch("rag_app.memory.jobs.get_settings") as mock_settings:
            cfg = get_settings()
            mock_settings.return_value = cfg.model_copy(update={"enable_conversational_memory_extraction": True})

            reconciled = await reconcile_missing_conversational_memory_jobs(session=session, lookback_hours=24)
            assert reconciled >= 1, "Scheduler must discover and reconcile orphaned memorable messages"

        # Verify job was created
        stmt = select(MemoryProcessingJob).where(
            MemoryProcessingJob.source_message_id == orphaned_msg.id,
            MemoryProcessingJob.trigger_kind == "conversational_turn",
        )
        created_job = (await session.execute(stmt)).scalars().first()
        assert created_job is not None
        assert created_job.owner_user_id == user1.id

