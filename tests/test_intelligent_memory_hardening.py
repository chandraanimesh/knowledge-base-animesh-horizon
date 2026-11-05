"""
Milestone 9: Independent Verification & Hardening Test Suite (9A–9F).
Independently verifies runtime paths, scheduler integration, conversation capture,
temporal correctness, ContextCompiler integration, and worker resilience.
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
    Conversation,
    ChatMessage,
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
from rag_app.memory.jobs import (
    create_memory_job,
    execute_memory_job,
    recover_stale_memory_jobs,
    process_pending_memory_jobs,
)
from rag_app.memory.schemas import (
    MemoryCandidate,
    MemoryType,
    MemoryStatus,
    StabilityTier,
    RelationshipType,
    JobStatus,
    ScopeType,
    PrivacyPolicyPayload,
)
from rag_app.memory.command_parser import parse_memory_command, detect_memory_candidate_rules
from rag_app.core.safety import check_sensitive_request
from rag_app.knowledge.scheduler import DurableReconciliationScheduler


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


async def create_isolated_user(session: AsyncSession, prefix: str = "h9") -> User:
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


async def create_isolated_conversation(session: AsyncSession, user_id: uuid.UUID) -> Conversation:
    conv = Conversation(
        id=uuid.uuid4(),
        user_id=user_id,
        title="Hardening Test Conversation",
    )
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


# ==============================================================================
# AUDIT 9A — NORMAL CONVERSATION TO AUTOMATIC MEMORY
# ==============================================================================

@pytest.mark.asyncio
async def test_audit_9a_ordinary_conversation_to_memory():
    """Verify complete sequence from ordinary conversational turn to durable memory."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "audit9a")
        ws = await create_isolated_workspace(session, "ws_9a")
        conv = await create_isolated_conversation(session, user.id)

        # 1. Normal user message without /rem
        user_text = "I prefer technical explanations in English with practical examples."
        msg = ChatMessage(
            id=uuid.uuid4(),
            conversation_id=conv.id,
            role="user",
            content=user_text,
        )
        session.add(msg)
        await session.commit()

        # 2. Durable memory job enqueue
        job = await create_memory_job(
            session=session,
            owner_user_id=user.id,
            source_text=user_text,
            trigger_kind="conversational_turn",
            workspace_id=ws.id,
            source_message_id=msg.id,
            conversation_id=conv.id,
        )
        assert job is not None
        assert job.status == JobStatus.PENDING.value
        assert job.source_message_id == msg.id
        await session.commit()

        # 3. Background worker executes job
        status, items, reason = await execute_memory_job(session=session, job_id=job.id, auto_embed=False)
        assert status == JobStatus.SUCCEEDED
        assert len(items) == 1
        mem_item = items[0]

        # 4. Valid memory persisted and traceable to source message
        assert mem_item.status == "ACTIVE"
        assert mem_item.memory_type in ("STATIC", "PROCEDURAL")
        assert mem_item.source_job_id == job.id
        assert "technical explanations" in mem_item.content.lower()

        # 5. Refresh job and verify completion
        await session.refresh(job)
        assert job.status == JobStatus.SUCCEEDED.value
        assert job.processed_at is not None


@pytest.mark.asyncio
async def test_audit_9a_idempotent_capture_and_processing():
    """Verify idempotency: duplicate capture returns existing job; reprocessing returns existing memory."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "idemp")
        conv = await create_isolated_conversation(session, user.id)
        msg = ChatMessage(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="Our backend uses FastAPI.")
        session.add(msg)
        await session.commit()

        # First enqueue
        job1 = await create_memory_job(
            session=session,
            owner_user_id=user.id,
            source_text=msg.content,
            trigger_kind="conversational_turn",
            source_message_id=msg.id,
        )
        await session.commit()

        # Second enqueue with same source message
        job2 = await create_memory_job(
            session=session,
            owner_user_id=user.id,
            source_text=msg.content,
            trigger_kind="conversational_turn",
            source_message_id=msg.id,
        )
        assert job1.id == job2.id

        # First execution
        st1, items1, _ = await execute_memory_job(session=session, job_id=job1.id, auto_embed=False)
        assert st1 == JobStatus.SUCCEEDED
        assert len(items1) == 1

        # Second execution (idempotent guard)
        st2, items2, _ = await execute_memory_job(session=session, job_id=job1.id, auto_embed=False)
        assert st2 == JobStatus.SUCCEEDED
        assert len(items2) == 1
        assert items1[0].id == items2[0].id


@pytest.mark.asyncio
async def test_audit_9a_sensitivity_and_privacy_safeguards():
    """Verify secrets/PII are rejected and privacy pause halts collection/extraction."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "safety")

        # 1. API key rejected
        secret_text = "Here is my secret token: sk-ant-api03-99999888887777766666"
        safety = check_sensitive_request(secret_text, can_view_pii=False)
        assert safety.blocked is True

        job_sec = await create_memory_job(session=session, owner_user_id=user.id, source_text=secret_text)
        assert job_sec is not None
        st, items, reason = await execute_memory_job(session=session, job_id=job_sec.id, auto_embed=False)
        assert st == JobStatus.SKIPPED_NOT_MEMORABLE
        assert len(items) == 0
        assert "sensitive" in (reason or "").lower()

        # 2. Harmless technical text is NOT rejected
        safe_tech = "We use FastAPI and PostgreSQL for our microservices."
        safety_ok = check_sensitive_request(safe_tech, can_view_pii=False)
        assert safety_ok.blocked is False

        # 3. Privacy pause drops collection
        await MemoryPrivacyService.set_policy(
            session=session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_collection_enabled=False,
        )
        paused_job = await create_memory_job(
            session=session,
            owner_user_id=user.id,
            source_text="Should be dropped while collection paused",
        )
        assert paused_job is None


# ==============================================================================
# AUDIT 9B — MEMORY TYPE CLASSIFICATION & LEGACY COMPATIBILITY
# ==============================================================================

@pytest.mark.asyncio
async def test_audit_9b_type_classification():
    """Verify precise classification across EPISODIC, SEMANTIC, PROCEDURAL, and transient dropping."""
    # 1. EPISODIC
    cand1, trans1, hints1 = detect_memory_candidate_rules("Yesterday I attended a project review.")
    assert cand1 is True and trans1 is False
    assert hints1.get("memory_type") == MemoryType.EPISODIC

    # 2. SEMANTIC
    cand2, trans2, hints2 = detect_memory_candidate_rules("Our backend uses FastAPI.")
    assert cand2 is True and trans2 is False
    assert hints2.get("memory_type") == MemoryType.SEMANTIC

    # 3. PROCEDURAL
    cand3, trans3, hints3 = detect_memory_candidate_rules("I prefer step-by-step explanations.")
    assert cand3 is True and trans3 is False
    assert hints3.get("memory_type") == MemoryType.PROCEDURAL

    # 4. EPISODIC event
    cand4, trans4, hints4 = detect_memory_candidate_rules("We completed the migration yesterday.")
    assert cand4 is True and trans4 is False
    assert hints4.get("memory_type") == MemoryType.EPISODIC

    # 5. Transient query (no durable memory)
    cand5, trans5, hints5 = detect_memory_candidate_rules("What is PostgreSQL?")
    assert cand5 is False and trans5 is True


@pytest.mark.asyncio
async def test_audit_9b_legacy_memory_compatibility():
    """Verify backward compatibility: legacy STATIC and DYNAMIC items load and retrieve smoothly."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "legacy")
        c_static = MemoryCandidate(
            user_id=user.id,
            content="Legacy static fact: User timezone is UTC",
            canonical_key="MEMORY:STATIC:pref:timezone",
            memory_type="STATIC",
        )
        c_dynamic = MemoryCandidate(
            user_id=user.id,
            content="Legacy dynamic context: sprint goal is MVP",
            canonical_key="MEMORY:DYNAMIC:task:sprint_goal",
            memory_type="DYNAMIC",
        )
        item_s = await MemoryMaterializer.materialize_memory(session, c_static, auto_embed=False)
        item_d = await MemoryMaterializer.materialize_memory(session, c_dynamic, auto_embed=False)
        assert item_s.memory_type == "STATIC"
        assert item_d.memory_type == "DYNAMIC"

        res = await MemoryRetrievalService.retrieve(session=session, user_id=user.id, query="timezone")
        assert any("timezone" in m.content.lower() for m in res.static_memories + res.relevant_memories)


# ==============================================================================
# AUDIT 9C — DUPLICATION, RELATIONSHIPS & TEMPORAL INTELLIGENCE
# ==============================================================================

@pytest.mark.asyncio
async def test_audit_9c_same_entity_attribute_contradiction_vs_complements():
    """Verify contradiction on same attribute vs complements on distinct attributes."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "audit9c")

        # Frontend stack
        m1 = await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
            user_id=user.id,
            content="Project frontend = React",
            canonical_key="MEMORY:SEMANTIC:frontend_stack",
            memory_type=MemoryType.SEMANTIC,
        ), auto_embed=False)

        # Same attribute conflicting value -> CONTRADICTION detected
        conflicts = await MemoryConsolidationService.detect_conflicts(
            session=session,
            user_id=user.id,
            content="Project frontend = Vue",
            canonical_key="MEMORY:SEMANTIC:frontend_stack",
        )
        assert len(conflicts) >= 1
        conf_item, reason = conflicts[0]
        assert conf_item.id == m1.id

        # Distinct attribute -> Backend stack (not contradictory)
        m_backend = await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
            user_id=user.id,
            content="Project backend = FastAPI",
            canonical_key="MEMORY:SEMANTIC:backend_stack",
            memory_type=MemoryType.SEMANTIC,
        ), auto_embed=False)

        conflicts_other = await MemoryConsolidationService.detect_conflicts(
            session=session,
            user_id=user.id,
            content="Project database = PostgreSQL",
            canonical_key="MEMORY:SEMANTIC:db_stack",
        )
        assert len(conflicts_other) == 0

        # Record COMPLEMENTS relationship between frontend and backend
        rel = await MemoryConsolidationService.record_relationship(
            session=session,
            source_memory_id=m1.id,
            target_memory_id=m_backend.id,
            relationship_type=RelationshipType.COMPLEMENTS,
            confidence=0.9,
            reason="Complementary technology stack components",
        )
        assert rel.relationship_type == "COMPLEMENTS"


@pytest.mark.asyncio
async def test_audit_9c_temporal_correctness_and_supersession():
    """Verify temporal supersession closing valid_until and marking status SUPERSEDED."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "temporal")
        now = datetime.now(timezone.utc)

        # Old role
        m_old = await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
            user_id=user.id,
            content="User role is Junior Developer",
            canonical_key="MEMORY:STATIC:role",
            memory_type=MemoryType.STATIC,
            valid_from=now - timedelta(days=90),
        ), auto_embed=False)

        # New role superseding old
        m_new = await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
            user_id=user.id,
            content="User role is Senior Lead Architect",
            canonical_key="MEMORY:STATIC:role",
            memory_type=MemoryType.STATIC,
            supersedes_id=m_old.id,
            valid_from=now,
        ), auto_embed=False)

        await session.refresh(m_old)
        assert m_old.status == "SUPERSEDED"
        assert m_old.superseded_by_id == m_new.id
        assert m_old.valid_until is not None

        # Expired temporary fact
        m_exp = await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
            user_id=user.id,
            content="Temporary dev token active",
            canonical_key="MEMORY:DYNAMIC:temp_token",
            memory_type=MemoryType.DYNAMIC,
            valid_until=now - timedelta(hours=1),
        ), auto_embed=False)

        res = await MemoryRetrievalService.retrieve(session=session, user_id=user.id, query="token")
        assert all(m.id != m_exp.id for m in (res.dynamic_memories + res.relevant_memories))


# ==============================================================================
# AUDIT 9D — DREAMING ENGINE & AUTONOMOUS EXECUTION
# ==============================================================================

@pytest.mark.asyncio
async def test_audit_9d_scheduler_integration_and_dreaming_trigger():
    """Verify the ReconciliationScheduler triggers Memory Dreaming and records MemoryDreamingRun."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "sched_dream")
        await MemoryPrivacyService.set_policy(
            session=session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_dreaming_enabled=True,
            memory_collection_enabled=True,
        )

        for i in range(4):
            await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
                user_id=user.id,
                content=f"Design guideline {i}: follow clean architecture",
                canonical_key=f"MEMORY:SEMANTIC:pref:clean_arch_{i}",
                memory_type=MemoryType.SEMANTIC,
            ), auto_embed=False)

        await session.commit()

        # Instantiate scheduler and run one reconciliation cycle
        sched = DurableReconciliationScheduler(interval_seconds=60, session_factory=TestSessionLocal)
        metrics = await sched.run_once()

        assert "dreamed_users" in metrics
        assert metrics["dreamed_users"] >= 1

        # Verify durable MemoryDreamingRun record
        runs = (await session.execute(
            select(MemoryDreamingRun).where(MemoryDreamingRun.user_id == user.id)
        )).scalars().all()
        assert len(runs) >= 1
        latest_run = runs[-1]
        assert latest_run.status == "COMPLETED"
        assert latest_run.memories_inspected >= 1


@pytest.mark.asyncio
async def test_audit_9d_worker_crash_stale_job_recovery():
    """Verify worker crash recovery: stale PROCESSING job recovered to RETRY_PENDING."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "crash_rec")
        now = datetime.now(timezone.utc)
        crashed_job = MemoryProcessingJob(
            id=uuid.uuid4(),
            owner_user_id=user.id,
            source_text_snapshot="Recoverable text after worker termination",
            trigger_kind="conversational_turn",
            status=JobStatus.PROCESSING.value,
            attempt_count=1,
            max_attempts=3,
            locked_at=now - timedelta(minutes=15),
            locked_by="crashed-pod-99",
            created_at=now - timedelta(minutes=16),
            updated_at=now - timedelta(minutes=15),
        )
        session.add(crashed_job)
        await session.commit()

        recovered_count = await recover_stale_memory_jobs(session, lock_timeout_seconds=120)
        assert recovered_count >= 1

        await session.refresh(crashed_job)
        assert crashed_job.status == JobStatus.RETRY_PENDING.value
        assert crashed_job.locked_by is None


# ==============================================================================
# AUDIT 9E — LIVING PROFILE SYNCHRONIZATION & CONTEXT COMPILER
# ==============================================================================

@pytest.mark.asyncio
async def test_audit_9e_living_profile_lifecycle_transitions():
    """Verify living profile transitions: add -> duplicate -> supersede -> forget -> tombstone protection."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "profile_life")

        # 1. New memory reflects in profile
        cand_role = MemoryCandidate(
            user_id=user.id,
            content="User role is Staff AI Architect",
            canonical_key="MEMORY:STATIC:role",
            memory_type=MemoryType.STATIC,
        )
        item1 = await MemoryMaterializer.materialize_memory(session, cand_role, auto_embed=False)
        prof1 = await compile_user_profile(user.id, session)
        assert prof1["work_context"].get("role") == "Staff AI Architect"

        # 2. Procedural convention reflects in profile
        cand_proc = MemoryCandidate(
            user_id=user.id,
            content="Always format code blocks with language tags",
            canonical_key="MEMORY:PROCEDURAL:code_blocks",
            memory_type=MemoryType.PROCEDURAL,
        )
        await MemoryMaterializer.materialize_memory(session, cand_proc, auto_embed=False)
        prof2 = await compile_user_profile(user.id, session)
        assert len(prof2.get("procedural_conventions", [])) >= 1

        # 3. Forget memory wipes it and registers tombstone
        del_res = await MemoryPrivacyOperations.delete_memory_by_id(session, item1.id, user.id)
        assert del_res["status"] == "DELETED"

        prof3 = await compile_user_profile(user.id, session)
        assert prof3["work_context"].get("role") is None

        # 4. Tombstone prevents resurrection even if delayed job attempts insertion
        resurrect_cand = MemoryCandidate(
            user_id=user.id,
            content="User role is Staff AI Architect",
            canonical_key=cand_role.canonical_key,
            memory_type=MemoryType.STATIC,
        )
        blocked_item = await MemoryMaterializer.materialize_memory(session, resurrect_cand, auto_embed=False)
        assert blocked_item is None


# ==============================================================================
# AUDIT 9F — EVALUATION & OBSERVABILITY REST APIS
# ==============================================================================

@pytest.mark.asyncio
async def test_audit_9f_evaluation_runner_and_metrics_persistence():
    """Verify 20-scenario deterministic evaluation runner records MemoryEvalMetric in database."""
    async with TestSessionLocal() as session:
        user = await create_isolated_user(session, "eval_runner")

        runner = MemoryEvaluationRunner(session=session)
        eval_res = await runner.run_all()
        assert eval_res["total_scenarios"] == 20
        assert eval_res["passed_scenarios"] >= 18
        assert eval_res["pass_rate"] >= 0.90

        # Verify persisted metrics in DB
        metrics = (await session.execute(
            select(MemoryEvalMetric).where(MemoryEvalMetric.eval_run_id == eval_res["eval_run_id"])
        )).scalars().all()
        assert len(metrics) == 20


@pytest.mark.asyncio
async def test_audit_9f_rest_api_auth_and_isolation():
    """Verify REST API observability endpoints enforce auth and tenant/user isolation."""
    async with TestSessionLocal() as session:
        user_a = await create_isolated_user(session, "api_user_a")
        user_b = await create_isolated_user(session, "api_user_b")

        # Materialize memory for User A
        await MemoryMaterializer.materialize_memory(session, MemoryCandidate(
            user_id=user_a.id,
            content="Secret preference for User A",
            canonical_key="MEMORY:STATIC:pref:user_a_secret",
            memory_type=MemoryType.STATIC,
        ), auto_embed=False)

        token_b = create_access_token(str(user_b.id))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            headers = {"Authorization": f"Bearer {token_b}"}

            # 1. Overview for User B shows 0 memories
            res_ov = await ac.get("/api/memory/overview", headers=headers)
            assert res_ov.status_code == 200
            data_ov = res_ov.json()
            assert data_ov["total_memories"] == 0

            # 2. Jobs endpoint returns empty for User B
            res_jobs = await ac.get("/api/memory/jobs", headers=headers)
            assert res_jobs.status_code == 200
            assert len(res_jobs.json()) == 0

            # 3. Dreaming runs empty for User B
            res_dream = await ac.get("/api/memory/dreaming/runs", headers=headers)
            assert res_dream.status_code == 200
            assert len(res_dream.json()) == 0
