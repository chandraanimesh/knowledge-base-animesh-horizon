"""
Milestone 9: Intelligent Memory & Autonomous Dreaming Engine Acceptance Test Suite.

Rigorously verifies all 20 deterministic scenarios:
 1. test_01_explicit_preference_capture
 2. test_02_repeated_information_deduplication
 3. test_03_new_semantic_fact_capture
 4. test_04_project_update
 5. test_05_temporal_change_supersession
 6. test_06_contradictory_facts_conflict
 7. test_07_ambiguous_information_handling
 8. test_08_temporary_information_expiration
 9. test_09_sensitive_information_rejection
 10. test_10_explicit_forget_request_tombstone
 11. test_11_paused_collection_exclusion
 12. test_12_disabled_retrieval_suppression
 13. test_13_cross_user_isolation
 14. test_14_cross_workspace_isolation
 15. test_15_unanswerable_memory_query
 16. test_16_dreaming_execution_and_clustering
 17. test_17_worker_restart_stale_job_recovery
 18. test_18_profile_rebuild_after_deletion
 19. test_19_tombstone_anti_resurrection
 20. test_20_legacy_memory_compatibility
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select, update, delete, text, func
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Strictly target disposable test database
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.main import app
from rag_app.db.session import get_db_session
from rag_app.db.models import (
    Department,
    Workspace,
    User,
    MemoryItem,
    MemoryRelationship,
    MemoryDreamingRun,
    MemoryEvalMetric,
    MemoryProcessingJob,
    MemoryPrivacyPolicy,
    MemoryTombstone,
)
from rag_app.security.authentication import create_access_token
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.retrieval import MemoryRetrievalService
from rag_app.memory.privacy_service import MemoryPrivacyService
from rag_app.memory.privacy_operations import MemoryPrivacyOperations
from rag_app.memory.consolidation import MemoryConsolidationService
from rag_app.memory.dreaming import run_memory_dreaming
from rag_app.memory.profile_compiler import compile_user_profile
from rag_app.memory.evaluation import MemoryEvaluationRunner
from rag_app.memory.jobs import create_memory_job, execute_memory_job, recover_stale_memory_jobs
from rag_app.memory.schemas import (
    MemoryCandidate,
    MemoryType,
    MemoryStatus,
    StabilityTier,
    RelationshipType,
    JobStatus,
    ScopeType,
)
from rag_app.memory.command_parser import parse_memory_command, detect_memory_candidate_rules
from rag_app.core.safety import check_sensitive_request


test_engine = create_async_engine(
    os.environ["DATABASE_URL"],
    echo=False,
    poolclass=NullPool,
)
TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_test_session():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db_session] = get_test_session


@pytest.fixture(autouse=True)
def ensure_disposable_db():
    assert "rag_disposable_test" in os.environ["DATABASE_URL"], "Must target rag_disposable_test exclusively!"


async def create_isolated_workspace(session: AsyncSession, prefix: str = "ws") -> Workspace:
    dept = Department(
        id=uuid.uuid4(),
        code=f"code_{prefix}_{uuid.uuid4().hex[:6]}",
        name=f"dept_{prefix}_{uuid.uuid4().hex[:6]}",
    )
    session.add(dept)
    await session.flush()
    ws = Workspace(
        id=uuid.uuid4(),
        code=f"code_ws_{prefix}_{uuid.uuid4().hex[:6]}",
        name=f"{prefix}_{uuid.uuid4().hex[:6]}",
        department_id=dept.id,
    )
    session.add(ws)
    await session.commit()
    await session.refresh(ws)
    return ws


async def create_isolated_user(session: AsyncSession, prefix: str = "m9") -> User:
    """Creates a clean test user for each scenario."""
    user = User(
        id=uuid.uuid4(),
        email=f"{prefix}_{uuid.uuid4().hex[:8]}@local.dev",
        hashed_password="mock_hashed_pass",
        role="USER",
        is_active=True,
        profile_json={},
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


# ==============================================================================
# 20 ACCEPTANCE TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_01_explicit_preference_capture():
    """1. Explicit preference capture via /rem command."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test01")
        parsed = parse_memory_command("/rem I prefer responses in concise bullet points.")
        assert parsed.is_command is True
        assert parsed.is_empty is False
        assert parsed.memory_type == MemoryType.STATIC
        assert parsed.stability_tier == StabilityTier.ETERNAL
        assert ("response_format" in parsed.canonical_key.lower() or "preference" in parsed.canonical_key.lower())

        cand = MemoryCandidate(
            user_id=user.id,
            content=parsed.content,
            canonical_key=parsed.canonical_key,
            memory_type=parsed.memory_type,
            stability_tier=parsed.stability_tier,
        )
        item = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert item is not None
        assert item.memory_type == "STATIC"
        assert item.status == "ACTIVE"


@pytest.mark.asyncio
async def test_02_repeated_information_deduplication():
    """2. Repeated information deduplication and DUPLICATE relationship."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test02")
        cand = MemoryCandidate(
            user_id=user.id,
            content="Always format output in Markdown tables",
            canonical_key="MEMORY:PROCEDURAL:table_format",
            memory_type=MemoryType.PROCEDURAL,
        )
        item1 = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert item1 is not None

        dups = await MemoryConsolidationService.detect_duplicates(
            session=session,
            user_id=user.id,
            content="Always format output in Markdown tables",
            exclude_id=uuid.uuid4(),
        )
        assert len(dups) >= 1
        dup_item, score, reason = dups[0]
        assert dup_item.id == item1.id
        assert score == 1.0

        rel = await MemoryConsolidationService.record_relationship(
            session=session,
            source_memory_id=item1.id,
            target_memory_id=dup_item.id,
            relationship_type=RelationshipType.DUPLICATE,
            confidence=score,
            reason=reason,
        )
        assert rel.relationship_type == "DUPLICATE"


@pytest.mark.asyncio
async def test_03_new_semantic_fact_capture():
    """3. New semantic fact capture from project directives."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test03")
        is_cand, is_trans, hints = detect_memory_candidate_rules("We decided to adopt PostgreSQL with pgvector for our vector store")
        assert is_cand is True
        assert is_trans is False
        assert hints.get("memory_type") == MemoryType.SEMANTIC
        assert hints.get("stability_tier") == StabilityTier.STABLE

        cand = MemoryCandidate(
            user_id=user.id,
            content="We decided to adopt PostgreSQL with pgvector for our vector store",
            canonical_key="MEMORY:SEMANTIC:tech_stack",
            memory_type=hints["memory_type"],
            stability_tier=hints["stability_tier"],
        )
        item = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert item is not None
        assert item.memory_type == "SEMANTIC"


@pytest.mark.asyncio
async def test_04_project_update():
    """4. Project update classified as DYNAMIC and EVOLVING."""
    is_cand, is_trans, hints = detect_memory_candidate_rules("I am currently working on SOC2 compliance audit documentation")
    assert is_cand is True
    assert is_trans is False
    assert hints.get("memory_type") == MemoryType.DYNAMIC
    assert hints.get("stability_tier") == StabilityTier.EVOLVING


@pytest.mark.asyncio
async def test_05_temporal_change_supersession():
    """5. Temporal change and supersession with validity closing."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test05")
        now = datetime.now(timezone.utc)
        old_cand = MemoryCandidate(
            user_id=user.id,
            content="User role is Junior Developer",
            canonical_key="MEMORY:STATIC:role",
            memory_type=MemoryType.STATIC,
            valid_from=now - timedelta(days=30),
        )
        old_item = await MemoryMaterializer.materialize_memory(session, old_cand, auto_embed=False)
        assert old_item is not None

        new_cand = MemoryCandidate(
            user_id=user.id,
            content="User role is Senior Lead Architect",
            canonical_key="MEMORY:STATIC:role",
            memory_type=MemoryType.STATIC,
            supersedes_id=old_item.id,
            valid_from=now,
        )
        new_item = await MemoryMaterializer.materialize_memory(session, new_cand, auto_embed=False)
        assert new_item is not None

        await session.refresh(old_item)
        assert old_item.status == "SUPERSEDED"
        assert old_item.superseded_by_id == new_item.id


@pytest.mark.asyncio
async def test_06_contradictory_facts_conflict():
    """6. Contradictory facts detection and CONTRADICTS relationship."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test06")
        cand1 = MemoryCandidate(
            user_id=user.id,
            content="User always requires dark mode interface",
            canonical_key="MEMORY:STATIC:pref:ui_theme",
            memory_type=MemoryType.STATIC,
        )
        item1 = await MemoryMaterializer.materialize_memory(session, cand1, auto_embed=False)

        conflicts = await MemoryConsolidationService.detect_conflicts(
            session=session,
            user_id=user.id,
            content="User always requires light mode interface",
            canonical_key=cand1.canonical_key,
        )
        assert len(conflicts) >= 1
        conf_item, reason = conflicts[0]
        assert conf_item.id == item1.id
        assert "opposing" in reason.lower() or "contradictory" in reason.lower()

        cand2 = MemoryCandidate(
            user_id=user.id,
            content="User always requires light mode interface",
            canonical_key=cand1.canonical_key,
            memory_type=MemoryType.STATIC,
        )
        item2 = await MemoryMaterializer.materialize_memory(session, cand2, auto_embed=False)
        assert item2 is not None

        rel = await MemoryConsolidationService.record_relationship(
            session=session,
            source_memory_id=item1.id,
            target_memory_id=item2.id,
            relationship_type=RelationshipType.CONTRADICTS,
            confidence=0.95,
            reason=reason,
        )
        assert rel.relationship_type == "CONTRADICTS"


@pytest.mark.asyncio
async def test_07_ambiguous_information_handling():
    """7. Ambiguous information handling (greetings/chit-chat dropped)."""
    is_cand, is_trans, hints = detect_memory_candidate_rules("Hi there, how are you doing today?")
    assert is_cand is False
    assert is_trans is True
    assert hints == {}


@pytest.mark.asyncio
async def test_08_temporary_information_expiration():
    """8. Temporary information expiration via valid_until."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test08")
        now = datetime.now(timezone.utc)
        cand = MemoryCandidate(
            user_id=user.id,
            content="Temporary staging token is active",
            canonical_key="MEMORY:DYNAMIC:staging_token",
            memory_type=MemoryType.DYNAMIC,
            valid_until=now - timedelta(hours=2),
        )
        item = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert item is not None

        res = await MemoryRetrievalService.retrieve(session=session, user_id=user.id, query="staging token")
        assert all(m.id != item.id for m in (res.dynamic_memories + res.relevant_memories))


@pytest.mark.asyncio
async def test_09_sensitive_information_rejection():
    """9. Sensitive information rejection (secrets/PII)."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test09")
        secret_text = "My secret token is sk-ant-api03-abcdef1234567890abcdef1234567890"
        safety = check_sensitive_request(secret_text, can_view_pii=False)
        assert safety.blocked is True

        job = await create_memory_job(
            session=session,
            owner_user_id=user.id,
            source_text=secret_text,
            trigger_kind="conversational_turn",
        )
        assert job is not None

        status, items, reason = await execute_memory_job(session=session, job_id=job.id)
        assert status == JobStatus.SKIPPED_NOT_MEMORABLE
        assert len(items) == 0
        assert "sensitive" in (reason or "").lower()


@pytest.mark.asyncio
async def test_10_explicit_forget_request_tombstone():
    """10. Explicit forget request writes MemoryTombstone."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test10")
        cand = MemoryCandidate(
            user_id=user.id,
            content="User lives in Paris",
            canonical_key="MEMORY:STATIC:location_paris",
            memory_type=MemoryType.STATIC,
        )
        item = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert item is not None

        del_res = await MemoryPrivacyOperations.delete_memory_by_id(session, item.id, user.id)
        assert del_res["status"] == "DELETED"

        tombstone = await session.scalar(
            select(MemoryTombstone).where(
                MemoryTombstone.user_id == user.id,
                MemoryTombstone.canonical_key == cand.canonical_key,
            )
        )
        assert tombstone is not None


@pytest.mark.asyncio
async def test_11_paused_collection_exclusion():
    """11. Paused collection exclusion."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test11")
        await MemoryPrivacyService.set_policy(
            session=session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_collection_enabled=False,
        )

        cand = MemoryCandidate(
            user_id=user.id,
            content="Should not be saved while paused",
            canonical_key="MEMORY:DYNAMIC:paused_note",
            memory_type=MemoryType.DYNAMIC,
        )
        item = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert item is None


@pytest.mark.asyncio
async def test_12_disabled_retrieval_suppression():
    """12. Disabled retrieval suppression."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test12")
        cand = MemoryCandidate(
            user_id=user.id,
            content="Important personal knowledge",
            canonical_key="MEMORY:STATIC:important_note",
            memory_type=MemoryType.STATIC,
        )
        await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)

        await MemoryPrivacyService.set_policy(
            session=session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_retrieval_enabled=False,
        )

        res = await MemoryRetrievalService.retrieve(session, user.id, query="important")
        assert res.total_count == 0
        assert len(res.static_memories) == 0


@pytest.mark.asyncio
async def test_13_cross_user_isolation():
    """13. Cross-user isolation guarantees."""
    async with TestSessionLocal() as session:
        user_a = await create_isolated_user(session, "user_a")
        user_b = await create_isolated_user(session, "user_b")

        cand = MemoryCandidate(
            user_id=user_a.id,
            content="Top secret project Alpha details",
            canonical_key="MEMORY:STATIC:project_alpha",
            memory_type=MemoryType.STATIC,
        )
        await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)

        res_b = await MemoryRetrievalService.retrieve(session, user_b.id, query="project Alpha")
        assert all(m.user_id != user_a.id for m in (res_b.static_memories + res_b.dynamic_memories + res_b.relevant_memories))


@pytest.mark.asyncio
async def test_14_cross_workspace_isolation():
    """14. Cross-workspace isolation."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test14")
        ws_1 = (await create_isolated_workspace(session, "ws1")).id
        ws_2 = (await create_isolated_workspace(session, "ws2")).id

        cand = MemoryCandidate(
            user_id=user.id,
            workspace_id=ws_1,
            content="Workspace 1 proprietary note",
            canonical_key="MEMORY:DYNAMIC:ws1_note",
            memory_type=MemoryType.DYNAMIC,
        )
        await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)

        mems_ws2 = await MemoryMaterializer.get_user_memories(
            session=session,
            user_id=user.id,
            workspace_id=ws_2,
            include_global=False,
        )
        assert len(mems_ws2) == 0


@pytest.mark.asyncio
async def test_15_unanswerable_memory_query():
    """15. Unanswerable memory query abstains gracefully."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test15")
        res = await MemoryRetrievalService.retrieve(session, user.id, query="quantum entanglement coordinates")
        assert res.total_count == 0


@pytest.mark.asyncio
async def test_16_dreaming_execution_and_clustering():
    """16. Autonomous Dreaming clusters memories and produces MemoryDreamingRun."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test16")
        await MemoryPrivacyService.set_policy(
            session=session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_dreaming_enabled=True,
            memory_collection_enabled=True,
        )

        for i in range(4):
            cand = MemoryCandidate(
                user_id=user.id,
                content=f"Team decision {i}: format code using black",
                canonical_key=f"MEMORY:SEMANTIC:pref:formatting_{i}",
                memory_type=MemoryType.SEMANTIC,
            )
            await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)

        res = await run_memory_dreaming(user.id, session, threshold=3, trigger_type="test")
        assert res["status"] == "SUCCESS"
        assert res["clusters_consolidated"] >= 1
        assert res["memories_consolidated"] >= 3

        run_rec = await session.scalar(
            select(MemoryDreamingRun).where(MemoryDreamingRun.run_id == res["run_id"])
        )
        assert run_rec is not None
        assert run_rec.status == "COMPLETED"


@pytest.mark.asyncio
async def test_17_worker_restart_stale_job_recovery():
    """17. Worker restart / stale job recovery."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test17")
        now = datetime.now(timezone.utc)
        job = MemoryProcessingJob(
            id=uuid.uuid4(),
            owner_user_id=user.id,
            source_text_snapshot="Recoverable user job snapshot",
            trigger_kind="conversational_turn",
            status=JobStatus.PROCESSING.value,
            attempt_count=1,
            max_attempts=3,
            locked_at=now - timedelta(minutes=10),
            locked_by="crashed-worker-1",
            created_at=now - timedelta(minutes=11),
            updated_at=now - timedelta(minutes=10),
        )
        session.add(job)
        await session.commit()

        recovered = await recover_stale_memory_jobs(session, lock_timeout_seconds=120)
        assert recovered >= 1
        await session.refresh(job)
        assert job.status == JobStatus.RETRY_PENDING.value


@pytest.mark.asyncio
async def test_18_profile_rebuild_after_deletion():
    """18. Profile rebuild excludes deleted memories."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test18")
        cand = MemoryCandidate(
            user_id=user.id,
            content="User role is Chief Reliability Officer",
            canonical_key="MEMORY:STATIC:role_rebuild",
            memory_type=MemoryType.STATIC,
        )
        item = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        prof1 = await compile_user_profile(user.id, session)
        assert prof1["work_context"].get("role") == "Chief Reliability Officer"

        await MemoryPrivacyOperations.delete_memory_by_id(session, item.id, user.id)
        prof2 = await compile_user_profile(user.id, session)
        assert prof2["work_context"].get("role") is None


@pytest.mark.asyncio
async def test_19_tombstone_anti_resurrection():
    """19. Tombstone anti-resurrection guarantees."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test19")
        key = "MEMORY:STATIC:forgotten_trait"
        tombstone = MemoryTombstone(
            id=uuid.uuid4(),
            user_id=user.id,
            canonical_key=key,
            reason="Explicit erasure",
        )
        session.add(tombstone)
        await session.commit()

        cand = MemoryCandidate(
            user_id=user.id,
            content="Resurrection attempt",
            canonical_key=key,
            memory_type=MemoryType.STATIC,
        )
        item = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert item is None


@pytest.mark.asyncio
async def test_20_legacy_memory_compatibility():
    """20. Legacy memory records (STATIC and DYNAMIC) compatibility."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "test20")
        cand_s = MemoryCandidate(
            user_id=user.id,
            content="Legacy static fact",
            canonical_key="MEMORY:STATIC:legacy_fact",
            memory_type="STATIC",
        )
        cand_d = MemoryCandidate(
            user_id=user.id,
            content="Legacy dynamic context",
            canonical_key="MEMORY:DYNAMIC:legacy_context",
            memory_type="DYNAMIC",
        )
        item_s = await MemoryMaterializer.materialize_memory(session, cand_s, auto_embed=False)
        item_d = await MemoryMaterializer.materialize_memory(session, cand_d, auto_embed=False)
        assert item_s is not None and item_s.memory_type == "STATIC"
        assert item_d is not None and item_d.memory_type == "DYNAMIC"

        res = await MemoryRetrievalService.retrieve(session, user.id, query="legacy")
        assert res.total_count >= 1
