import uuid
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import text, select
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
    KnowledgeItem,
    KnowledgeEntity,
    KnowledgeRelationship,
    KnowledgeEvidence,
    MemoryItem,
    ConnectionCandidate,
    ConnectionCandidateReview,
    ConnectionCandidateEvidence,
    ConnectionDiscoveryJob,
)
from rag_app.knowledge.connection_service import ConnectionDiscoveryService, compute_hash
from rag_app.knowledge.connection_worker import (
    enqueue_discovery_job,
    claim_next_discovery_job,
    execute_discovery_job,
)
from rag_app.schemas.connections import ConnectionReviewRequest


@pytest_asyncio.fixture
async def conn_test_env():
    """
    Creates an isolated schema for connection discovery tests with full clean-up.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_conn_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))

        def create_tables(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
            Department.__table__.create(sync_conn)
            User.__table__.create(sync_conn)
            Permission.__table__.create(sync_conn)
            UserDepartment.__table__.create(sync_conn)
            UserPermission.__table__.create(sync_conn)
            Workspace.__table__.create(sync_conn)
            WorkspaceMember.__table__.create(sync_conn)
            KnowledgeItem.__table__.create(sync_conn)
            KnowledgeEntity.__table__.create(sync_conn)
            KnowledgeRelationship.__table__.create(sync_conn)
            KnowledgeEvidence.__table__.create(sync_conn)
            MemoryItem.__table__.create(sync_conn)
            ConnectionCandidate.__table__.create(sync_conn)
            ConnectionCandidateReview.__table__.create(sync_conn)
            ConnectionCandidateEvidence.__table__.create(sync_conn)
            ConnectionDiscoveryJob.__table__.create(sync_conn)

        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:8]
    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        dept_a = Department(name=f"Dept A {suffix}", code=f"DEPT_A_{suffix}", is_active=True)
        dept_b = Department(name=f"Dept B {suffix}", code=f"DEPT_B_{suffix}", is_active=True)
        session.add_all([dept_a, dept_b])
        await session.flush()

        perm_view = Permission(description="View", code="VIEW", is_active=True)
        session.add(perm_view)
        await session.flush()

        user_a = User(email=f"user_a_{suffix}@example.com", hashed_password="pwd", role="USER", is_active=True)
        user_b = User(email=f"user_b_{suffix}@example.com", hashed_password="pwd", role="USER", is_active=True)
        admin = User(email=f"admin_{suffix}@example.com", hashed_password="pwd", role="ADMIN", is_active=True)
        session.add_all([user_a, user_b, admin])
        await session.flush()

        # User A has VIEW in Dept A
        session.add(UserDepartment(user_id=user_a.id, department_id=dept_a.id))
        session.add(UserPermission(user_id=user_a.id, department_id=dept_a.id, permission_id=perm_view.id))

        # User B has VIEW in Dept B
        session.add(UserDepartment(user_id=user_b.id, department_id=dept_b.id))
        session.add(UserPermission(user_id=user_b.id, department_id=dept_b.id, permission_id=perm_view.id))

        await session.commit()

        env = {
            "schema_name": schema_name,
            "session_maker": session_maker,
            "dept_a": dept_a,
            "dept_b": dept_b,
            "user_a": user_a,
            "user_b": user_b,
            "admin": admin,
        }

    yield env

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_authorization_containment_pre_retrieval(conn_test_env):
    """
    Proves that unauthorized department knowledge items NEVER enter candidate generation queries.
    """
    sm = conn_test_env["session_maker"]
    schema = conn_test_env["schema_name"]
    user_a = conn_test_env["user_a"]
    admin = conn_test_env["admin"]
    dept_a = conn_test_env["dept_a"]
    dept_b = conn_test_env["dept_b"]
    now = datetime.now(timezone.utc)

    async with sm() as session:
        await session.execute(text(f"SET search_path TO {schema}, public"))

        # Source note in Dept A
        note_1 = KnowledgeItem(
            canonical_key="acme_earnings_q3",
            kind="ENTITY",
            source_type="DOCUMENT",
            visibility="SHARED",
            approval_status="APPROVED",
            approved_by_id=admin.id,
            approved_at=now,
            department_id=dept_a.id,
            status="ACTIVE",
        )
        # Target note in Dept A (authorized for user_a)
        note_2_dept_a = KnowledgeItem(
            canonical_key="acme_revenue_growth_q3",
            kind="ENTITY",
            source_type="DOCUMENT",
            visibility="SHARED",
            approval_status="APPROVED",
            approved_by_id=admin.id,
            approved_at=now,
            department_id=dept_a.id,
            status="ACTIVE",
        )
        # Target note in Dept B (UNAUTHORIZED for user_a)
        note_3_dept_b = KnowledgeItem(
            canonical_key="acme_merger_confidential",
            kind="ENTITY",
            source_type="DOCUMENT",
            visibility="SHARED",
            approval_status="APPROVED",
            approved_by_id=admin.id,
            approved_at=now,
            department_id=dept_b.id,
            status="ACTIVE",
        )
        session.add_all([note_1, note_2_dept_a, note_3_dept_b])
        await session.commit()
        await session.refresh(note_1)

        # Run discovery as User A
        service = ConnectionDiscoveryService(session)
        candidates = await service.generate_candidates_for_source(
            source_id=note_1.id,
            source_type="NOTE",
            user=user_a,
            max_candidates=10,
        )

        candidate_target_ids = [c.target_knowledge_item_id for c in candidates]

        # Authorized item must appear
        assert note_2_dept_a.id in candidate_target_ids
        # Unauthorized item in Dept B must NEVER appear in candidate results
        assert note_3_dept_b.id not in candidate_target_ids


@pytest.mark.asyncio
async def test_epistemic_separation_no_numerical_confirmation(conn_test_env):
    """
    Proves that a high ranking score NEVER automatically confirms a relationship.
    Candidates always start in SUGGESTED or REVIEW_REQUIRED, requiring explicit curator review.
    """
    sm = conn_test_env["session_maker"]
    schema = conn_test_env["schema_name"]
    user_a = conn_test_env["user_a"]
    admin = conn_test_env["admin"]
    dept_a = conn_test_env["dept_a"]
    now = datetime.now(timezone.utc)

    async with sm() as session:
        await session.execute(text(f"SET search_path TO {schema}, public"))

        note_a = KnowledgeItem(
            canonical_key="project_apollo_launch",
            kind="ENTITY",
            source_type="DOCUMENT",
            visibility="SHARED",
            approval_status="APPROVED",
            approved_by_id=admin.id,
            approved_at=now,
            department_id=dept_a.id,
            status="ACTIVE",
        )
        note_b = KnowledgeItem(
            canonical_key="project_apollo_launch_schedule",
            kind="ENTITY",
            source_type="DOCUMENT",
            visibility="SHARED",
            approval_status="APPROVED",
            approved_by_id=admin.id,
            approved_at=now,
            department_id=dept_a.id,
            status="ACTIVE",
        )
        session.add_all([note_a, note_b])
        await session.commit()

        service = ConnectionDiscoveryService(session)
        candidates = await service.generate_candidates_for_source(
            source_id=note_a.id,
            source_type="NOTE",
            user=user_a,
        )

        assert len(candidates) >= 1
        candidate = candidates[0]

        # Candidate is ranked as a semantic association in SUGGESTED state
        assert candidate.ranking_score > 0.0
        assert candidate.lifecycle_state == "SUGGESTED"
        assert candidate.connection_type == "SEMANTIC_ASSOCIATION"
        # Assert it was NOT automatically promoted to CONFIRMED
        assert candidate.lifecycle_state != "CONFIRMED"


@pytest.mark.asyncio
async def test_human_curator_review_promotes_to_canonical_knowledge(conn_test_env):
    """
    Verifies that human curator review with version-bound hashes transitions candidate
    to CONFIRMED and promotes the relationship into the canonical KnowledgeRelationship layer.
    """
    sm = conn_test_env["session_maker"]
    schema = conn_test_env["schema_name"]
    admin = conn_test_env["admin"]
    dept_a = conn_test_env["dept_a"]
    now = datetime.now(timezone.utc)

    async with sm() as session:
        await session.execute(text(f"SET search_path TO {schema}, public"))

        note_a = KnowledgeItem(
            canonical_key="customer_acme_corp",
            kind="ENTITY",
            source_type="DOCUMENT",
            visibility="SHARED",
            approval_status="APPROVED",
            approved_by_id=admin.id,
            approved_at=now,
            department_id=dept_a.id,
            status="ACTIVE",
        )
        note_b = KnowledgeItem(
            canonical_key="acme_enterprise_cloud",
            kind="ENTITY",
            source_type="DOCUMENT",
            visibility="SHARED",
            approval_status="APPROVED",
            approved_by_id=admin.id,
            approved_at=now,
            department_id=dept_a.id,
            status="ACTIVE",
        )
        session.add_all([note_a, note_b])
        await session.flush()

        # Canonical entities attached to knowledge items
        ent_a = KnowledgeEntity(knowledge_item_id=note_a.id, entity_type="COMPANY", canonical_name="Acme Corp", attributes={})
        ent_b = KnowledgeEntity(knowledge_item_id=note_b.id, entity_type="PRODUCT", canonical_name="Enterprise Cloud", attributes={})
        session.add_all([ent_a, ent_b])
        await session.commit()

        service = ConnectionDiscoveryService(session)
        candidates = await service.generate_candidates_for_source(
            source_id=note_a.id,
            source_type="NOTE",
            user=admin,
        )
        assert len(candidates) >= 1
        candidate = candidates[0]

        # Review with matching version hashes
        review_req = ConnectionReviewRequest(
            decision="CONFIRMED",
            rationale="Verified customer contract confirms Enterprise Cloud adoption",
            source_version_hash=candidate.source_version_hash,
            target_version_hash=candidate.target_version_hash,
            policy_version="v1.0",
        )
        res = await service.review_candidate(candidate_id=candidate.id, user=admin, request=review_req)

        assert res.lifecycle_state == "CONFIRMED"

        # Assert promoted to canonical KnowledgeRelationship
        canon_rel = await session.scalar(
            select(KnowledgeRelationship).where(
                KnowledgeRelationship.subject_entity_id == ent_a.id,
                KnowledgeRelationship.object_entity_id == ent_b.id,
            )
        )
        assert canon_rel is not None
        assert canon_rel.predicate == "associated_with"

        # Assert audit review is preserved
        review_record = await session.scalar(
            select(ConnectionCandidateReview).where(ConnectionCandidateReview.candidate_id == candidate.id)
        )
        assert review_record is not None
        assert review_record.decision == "CONFIRMED"
        assert review_record.reviewer_user_id == admin.id


@pytest.mark.asyncio
async def test_review_stale_version_mismatch_rejected(conn_test_env):
    """
    Verifies that a review request referencing outdated source version hashes is rejected with 409 Conflict.
    """
    sm = conn_test_env["session_maker"]
    schema = conn_test_env["schema_name"]
    admin = conn_test_env["admin"]
    dept_a = conn_test_env["dept_a"]
    now = datetime.now(timezone.utc)

    async with sm() as session:
        await session.execute(text(f"SET search_path TO {schema}, public"))

        note_a = KnowledgeItem(canonical_key="strategy_2026", kind="ENTITY", source_type="DOCUMENT", visibility="SHARED", approval_status="APPROVED", approved_by_id=admin.id, approved_at=now, department_id=dept_a.id, status="ACTIVE")
        note_b = KnowledgeItem(canonical_key="budget_2026", kind="ENTITY", source_type="DOCUMENT", visibility="SHARED", approval_status="APPROVED", approved_by_id=admin.id, approved_at=now, department_id=dept_a.id, status="ACTIVE")
        session.add_all([note_a, note_b])
        await session.commit()

        service = ConnectionDiscoveryService(session)
        candidates = await service.generate_candidates_for_source(source_id=note_a.id, source_type="NOTE", user=admin)
        candidate = candidates[0]

        stale_review = ConnectionReviewRequest(
            decision="CONFIRMED",
            rationale="Approved",
            source_version_hash="stale_outdated_hash_12345",
            target_version_hash=candidate.target_version_hash,
        )

        with pytest.raises(HTTPException) as exc_info:
            await service.review_candidate(candidate_id=candidate.id, user=admin, request=stale_review)

        assert exc_info.value.status_code == 409
        assert "revalidation required" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_source_mutation_triggers_revalidation(conn_test_env):
    """
    Verifies that mutating a source note causes previously confirmed candidates to transition
    to REVALIDATION_IN_PROGRESS rather than remaining silently confirmed.
    """
    sm = conn_test_env["session_maker"]
    schema = conn_test_env["schema_name"]
    admin = conn_test_env["admin"]
    dept_a = conn_test_env["dept_a"]
    now = datetime.now(timezone.utc)

    async with sm() as session:
        await session.execute(text(f"SET search_path TO {schema}, public"))

        note_a = KnowledgeItem(canonical_key="product_spec", kind="ENTITY", source_type="DOCUMENT", visibility="SHARED", approval_status="APPROVED", approved_by_id=admin.id, approved_at=now, department_id=dept_a.id, status="ACTIVE")
        note_b = KnowledgeItem(canonical_key="product_architecture", kind="ENTITY", source_type="DOCUMENT", visibility="SHARED", approval_status="APPROVED", approved_by_id=admin.id, approved_at=now, department_id=dept_a.id, status="ACTIVE")
        session.add_all([note_a, note_b])
        await session.commit()

        service = ConnectionDiscoveryService(session)
        candidates = await service.generate_candidates_for_source(source_id=note_a.id, source_type="NOTE", user=admin)
        candidate = candidates[0]

        # Confirm it
        review_req = ConnectionReviewRequest(
            decision="CONFIRMED",
            rationale="Approved spec",
            source_version_hash=candidate.source_version_hash,
            target_version_hash=candidate.target_version_hash,
        )
        await service.review_candidate(candidate_id=candidate.id, user=admin, request=review_req)

        # Mutate source note
        note_a.canonical_key = "product_spec_v2_redesigned"
        note_a.updated_at = datetime.now(timezone.utc)
        await session.commit()

        # Re-run discovery on updated note
        updated_candidates = await service.generate_candidates_for_source(source_id=note_a.id, source_type="NOTE", user=admin)
        assert len(updated_candidates) >= 1
        updated_cand = updated_candidates[0]

        # State must transition to REVALIDATION_IN_PROGRESS
        assert updated_cand.lifecycle_state == "REVALIDATION_IN_PROGRESS"


@pytest.mark.asyncio
async def test_worker_stale_source_protection(conn_test_env):
    """
    Verifies that background worker drops execution safely when source item mutated before processing.
    """
    sm = conn_test_env["session_maker"]
    schema = conn_test_env["schema_name"]
    admin = conn_test_env["admin"]
    dept_a = conn_test_env["dept_a"]
    now = datetime.now(timezone.utc)

    async with sm() as session:
        await session.execute(text(f"SET search_path TO {schema}, public"))

        note_a = KnowledgeItem(canonical_key="roadmap_q4", kind="ENTITY", source_type="DOCUMENT", visibility="SHARED", approval_status="APPROVED", approved_by_id=admin.id, approved_at=now, department_id=dept_a.id, status="ACTIVE")
        session.add(note_a)
        await session.commit()
        await session.refresh(note_a)

        updated_str = note_a.updated_at.isoformat() if note_a.updated_at else note_a.created_at.isoformat()
        initial_hash = compute_hash(f"{note_a.canonical_key}:{updated_str}")

        # Enqueue job with initial hash
        job = await enqueue_discovery_job(
            session=session,
            source_item_id=note_a.id,
            source_item_type="NOTE",
            requesting_user_id=admin.id,
            target_source_hash=initial_hash,
        )

        # Mutate source note before worker claims it
        note_a.canonical_key = "roadmap_q4_delayed"
        note_a.updated_at = datetime.now(timezone.utc)
        await session.commit()

        # Claim and execute job
        claimed = await claim_next_discovery_job(session, worker_id="test-worker")
        assert claimed is not None
        assert claimed.id == job.id

        success = await execute_discovery_job(session, claimed)
        assert success is True

        # Assert job was marked completed but stale source was detected and dropped
        await session.refresh(claimed)
        assert claimed.status == "COMPLETED"
        assert "Source mutated before job completion" in claimed.error_message


@pytest.mark.asyncio
async def test_personal_memory_strict_isolation(conn_test_env):
    """
    Proves that a user's private MemoryItem can NEVER be discovered or accessed by another user.
    """
    sm = conn_test_env["session_maker"]
    schema = conn_test_env["schema_name"]
    user_a = conn_test_env["user_a"]
    user_b = conn_test_env["user_b"]

    async with sm() as session:
        await session.execute(text(f"SET search_path TO {schema}, public"))

        # User A's private memory
        mem_a = MemoryItem(
            user_id=user_a.id,
            canonical_key="user_preference_coding_style",
            content="I prefer concise Python code with strict typing",
            status="ACTIVE",
        )
        # User B's memory
        mem_b = MemoryItem(
            user_id=user_b.id,
            canonical_key="user_preference_editor",
            content="I use VS Code for python development",
            status="ACTIVE",
        )
        session.add_all([mem_a, mem_b])
        await session.commit()

        service = ConnectionDiscoveryService(session)

        # User B attempts to run discovery on User A's memory -> Must be Forbidden
        with pytest.raises(HTTPException) as exc_info:
            await service.generate_candidates_for_source(source_id=mem_a.id, source_type="MEMORY", user=user_b)
        assert exc_info.value.status_code == 403

        # User B running discovery on User B's memory -> User A's memory must NEVER appear in targets
        candidates_b = await service.generate_candidates_for_source(source_id=mem_b.id, source_type="MEMORY", user=user_b)
        mem_target_ids = [c.target_memory_item_id for c in candidates_b]
        assert mem_a.id not in mem_target_ids
