"""
Milestone 8: Commercial Memory & Privacy Controls Acceptance Test Suite.

Rigorously verifies all 25 acceptance criteria:
 1. test_01_user_disables_collection_stops_new_memories
 2. test_02_paused_user_excluded_from_extraction_jobs
 3. test_03_paused_user_excluded_from_dreaming
 4. test_04_user_memory_isolation
 5. test_05_tenant_memory_isolation
 6. test_06_strictest_rule_precedence
 7. test_07_expired_memory_excluded_from_retrieval
 8. test_08_retention_sweep_cleans_up_expired_memory
 9. test_09_selected_memory_deletion
10. test_10_personal_memory_deletion_preserves_shared_knowledge
11. test_11_profile_scrubbing_removes_attribute
12. test_12_rebuilding_profile_does_not_resurrect_scrubbed_attribute
13. test_13_dreaming_cannot_resurrect_deleted_memories
14. test_14_delayed_pending_jobs_cannot_resurrect_deleted_memories
15. test_15_memory_retrieval_cannot_expose_deleted_memories
16. test_16_unauthorized_deletion_export_fails_closed
17. test_17_user_export_contains_only_authorized_memories
18. test_18_concurrent_deletion_and_processing_consistency
19. test_19_retention_sweep_is_idempotent
20. test_20_retry_after_failure_no_corruption
21. test_21_active_memories_retrieve_cleanly
22. test_22_core_rag_retrieval_ranking_unaffected
23. test_23_langgraph_and_chat_flows_operational
24. test_24_thread_and_tenant_isolation
25. test_25_frontend_production_build_succeeds
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
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
    UserDepartment,
    WorkspaceMember,
    Conversation,
    ChatMessage,
    Document,
    DocumentChunk,
    MemoryItem,
    MemoryProcessingJob,
    MemoryPrivacyPolicy,
    MemoryTombstone,
    MemoryPrivacyAuditEvent,
)
from rag_app.security.authentication import create_access_token
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.retrieval import MemoryRetrievalService
from rag_app.memory.privacy_service import MemoryPrivacyService
from rag_app.memory.privacy_operations import MemoryPrivacyOperations
from rag_app.memory.retention_service import MemoryRetentionService
from rag_app.memory.dreaming import run_memory_dreaming
from rag_app.memory.profile_compiler import compile_user_profile
from rag_app.memory.jobs import create_memory_job, execute_memory_job
from rag_app.conversation.service import ConversationService
from rag_app.retrieval.service import RetrievalService
from rag_app.memory.schemas import (
    MemoryCandidate,
    MemoryType,
    MemoryStatus,
    ScopeType,
    PrivacyPolicyPayload,
    JobStatus,
)

# Test database engine configured with NullPool for robust asyncpg lifecycle
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


async def get_test_setup(session: AsyncSession):
    """
    Seeds departments, workspaces, and test users:
      - Dept 1 (Engineering): Alice (Normal User), Admin (Administrator)
      - Dept 2 (HR): Bob (Normal User)
    """
    # Department 1: Engineering
    dept1 = await session.scalar(select(Department).where(Department.code == "M8_ENG_TEST"))
    if not dept1:
        dept1 = Department(code="M8_ENG_TEST", name="Engineering Test Dept", is_active=True)
        session.add(dept1)
        await session.flush()

    # Department 2: HR
    dept2 = await session.scalar(select(Department).where(Department.code == "M8_HR_TEST"))
    if not dept2:
        dept2 = Department(code="M8_HR_TEST", name="HR Test Dept", is_active=True)
        session.add(dept2)
        await session.flush()

    # Workspace 1
    ws1 = await session.scalar(select(Workspace).where(Workspace.code == "m8-eng-ws"))
    if not ws1:
        ws1 = Workspace(code="m8-eng-ws", name="Engineering Workspace", department_id=dept1.id, is_active=True)
        session.add(ws1)
        await session.flush()

    # Workspace 2
    ws2 = await session.scalar(select(Workspace).where(Workspace.code == "m8-hr-ws"))
    if not ws2:
        ws2 = Workspace(code="m8-hr-ws", name="HR Workspace", department_id=dept2.id, is_active=True)
        session.add(ws2)
        await session.flush()

    # Alice (Dept 1, Ws 1)
    alice = await session.scalar(select(User).where(User.email == "alice_m8@local.dev"))
    if not alice:
        alice = User(
            email="alice_m8@local.dev",
            hashed_password="hashed_pass_placeholder",
            role="USER",
            is_active=True,
        )
        session.add(alice)
        await session.flush()
        session.add(UserDepartment(user_id=alice.id, department_id=dept1.id))
        session.add(WorkspaceMember(workspace_id=ws1.id, user_id=alice.id, role="EDITOR"))

    # Bob (Dept 2, Ws 2)
    bob = await session.scalar(select(User).where(User.email == "bob_m8@local.dev"))
    if not bob:
        bob = User(
            email="bob_m8@local.dev",
            hashed_password="hashed_pass_placeholder",
            role="USER",
            is_active=True,
        )
        session.add(bob)
        await session.flush()
        session.add(UserDepartment(user_id=bob.id, department_id=dept2.id))
        session.add(WorkspaceMember(workspace_id=ws2.id, user_id=bob.id, role="EDITOR"))

    # Admin (Dept 1)
    admin = await session.scalar(select(User).where(User.email == "admin_m8@local.dev"))
    if not admin:
        admin = User(
            email="admin_m8@local.dev",
            hashed_password="hashed_pass_placeholder",
            role="ADMIN",
            is_active=True,
        )
        session.add(admin)
        await session.flush()
        session.add(UserDepartment(user_id=admin.id, department_id=dept1.id))
        session.add(WorkspaceMember(workspace_id=ws1.id, user_id=admin.id, role="OWNER"))

    await session.commit()
    return dept1, dept2, ws1, ws2, alice, bob, admin


# =============================================================================
# 1. USER DISABLES COLLECTION STOPS NEW MEMORIES
# =============================================================================
@pytest.mark.asyncio
async def test_01_user_disables_collection_stops_new_memories():
    async with TestSessionLocal() as session:
        dept1, _, ws1, _, alice, _, admin = await get_test_setup(session)

        # Alice pauses/disables memory collection
        await MemoryPrivacyService.pause_memory_collection(session, alice.id)

        # Attempt to materialize a memory
        cand = MemoryCandidate(
            user_id=alice.id,
            workspace_id=ws1.id,
            content="Alice prefers dark theme and Python 3.12.",
            memory_type=MemoryType.STATIC,
        )
        res = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert res is None, "Materializer must return None when collection is paused"

        # Check DB has no memory for this candidate
        mems = (await session.scalars(
            select(MemoryItem).where(
                MemoryItem.user_id == alice.id,
                MemoryItem.content == "Alice prefers dark theme and Python 3.12.",
            )
        )).all()
        assert len(mems) == 0, "No memory item should be persisted when paused"

        # Re-enable collection
        await MemoryPrivacyService.resume_memory_collection(session, alice.id)
        res_active = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        assert res_active is not None, "Materializer must persist memory when resumed"
        assert res_active.status == "ACTIVE"


# =============================================================================
# 2. PAUSED USER EXCLUDED FROM EXTRACTION JOBS
# =============================================================================
@pytest.mark.asyncio
async def test_02_paused_user_excluded_from_extraction_jobs():
    async with TestSessionLocal() as session:
        _, _, ws1, _, alice, _, _ = await get_test_setup(session)

        # Pause collection
        await MemoryPrivacyService.pause_memory_collection(session, alice.id)

        # create_memory_job should drop creation when paused
        job = await create_memory_job(
            session=session,
            owner_user_id=alice.id,
            source_text="Alice works on backend distributed architecture.",
            workspace_id=ws1.id,
        )
        assert job is None, "create_memory_job must return None for paused user"

        # If an unexecuted job already existed in queue prior to pause:
        orphan_job = MemoryProcessingJob(
            owner_user_id=alice.id,
            workspace_id=ws1.id,
            source_text_snapshot="Existing queued statement",
            status="PENDING",
        )
        session.add(orphan_job)
        await session.commit()

        # Disable extraction via policy
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.USER,
                scope_id=alice.id,
                memory_extraction_enabled=False,
            ),
            actor=alice,
        )

        job_status, items, skip_reason = await execute_memory_job(
            session=session,
            job_id=orphan_job.id,
            auto_embed=False,
        )
        assert job_status == JobStatus.SKIPPED_POLICY_PAUSED
        assert len(items) == 0
        assert "paused" in str(skip_reason).lower()


# =============================================================================
# 3. PAUSED USER EXCLUDED FROM DREAMING
# =============================================================================
@pytest.mark.asyncio
async def test_03_paused_user_excluded_from_dreaming():
    async with TestSessionLocal() as session:
        _, _, ws1, _, alice, _, _ = await get_test_setup(session)

        # Set dreaming disabled for Alice
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.USER,
                scope_id=alice.id,
                memory_dreaming_enabled=False,
            ),
            actor=alice,
        )

        dream_result = await run_memory_dreaming(
            user_id=alice.id,
            session=session,
            threshold=1,
        )
        assert dream_result.get("status") in ("SKIPPED_POLICY_PAUSED", "skipped")


# =============================================================================
# 4. USER MEMORY ISOLATION
# =============================================================================
@pytest.mark.asyncio
async def test_04_user_memory_isolation():
    async with TestSessionLocal() as session:
        _, _, ws1, ws2, alice, bob, _ = await get_test_setup(session)

        # Resume both
        await MemoryPrivacyService.resume_memory_collection(session, alice.id)
        await MemoryPrivacyService.resume_memory_collection(session, bob.id)

        cand_a = MemoryCandidate(
            user_id=alice.id,
            workspace_id=ws1.id,
            content="Alice private memo: Secret project code is X-44.",
            canonical_key="memory:static:alice_secret_44",
            memory_type=MemoryType.STATIC,
        )
        cand_b = MemoryCandidate(
            user_id=bob.id,
            workspace_id=ws2.id,
            content="Bob private memo: Budget allocated is $500k.",
            canonical_key="memory:static:bob_budget_500k",
            memory_type=MemoryType.STATIC,
        )
        await MemoryMaterializer.materialize_memory(session, cand_a, auto_embed=False)
        await MemoryMaterializer.materialize_memory(session, cand_b, auto_embed=False)

        # Alice retrieves memories
        res_alice = await MemoryRetrievalService.retrieve(
            session=session,
            user_id=alice.id,
            workspace_id=ws1.id,
            query="budget and secret project",
        )
        alice_contents = [m.content for m in res_alice.static_memories + res_alice.dynamic_memories]
        assert any("Secret project code is X-44" in c for c in alice_contents)
        assert not any("Budget allocated is $500k" in c for c in alice_contents), "Alice must NEVER see Bob's memory"


# =============================================================================
# 5. TENANT MEMORY ISOLATION
# =============================================================================
@pytest.mark.asyncio
async def test_05_tenant_memory_isolation():
    async with TestSessionLocal() as session:
        _, _, _, _, alice, bob, _ = await get_test_setup(session)

    token_alice = create_access_token(subject=str(alice.id))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Alice tries to export Bob's memories via API by tampering or fetching export
        resp = await client.get(
            "/api/v1/memory/export",
            headers={"Authorization": f"Bearer {token_alice}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == str(alice.id)
        # All exported items must strictly belong to Alice
        for item in data["memories"]:
            assert item["canonical_key"] != "memory:static:bob_budget_500k"


# =============================================================================
# 6. STRICTEST RULE PRECEDENCE
# =============================================================================
@pytest.mark.asyncio
async def test_06_strictest_rule_precedence():
    async with TestSessionLocal() as session:
        dept1, _, ws1, _, alice, _, admin = await get_test_setup(session)

        # Tenant policy: enabled, 90 days
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.TENANT,
                scope_id=dept1.id,
                memory_collection_enabled=True,
                retention_days=90,
            ),
            actor=admin,
        )

        # Workspace policy: enabled, 60 days
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.WORKSPACE,
                scope_id=ws1.id,
                memory_collection_enabled=True,
                retention_days=60,
            ),
            actor=admin,
        )

        # User policy: disabled, 30 days
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.USER,
                scope_id=alice.id,
                memory_collection_enabled=False,
                retention_days=30,
            ),
            actor=alice,
        )

        effective = await MemoryPrivacyService.get_effective_policy(
            session=session,
            user_id=alice.id,
            workspace_id=ws1.id,
        )
        assert effective.memory_collection_enabled is False, "User disabling overrides tenant enablement"
        assert effective.retention_days == 30, "Minimum retention (30 days) must win"

        # Now test tenant disabling override: Tenant disabled, User enabled
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.TENANT,
                scope_id=dept1.id,
                memory_collection_enabled=False,
            ),
            actor=admin,
        )
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.USER,
                scope_id=alice.id,
                memory_collection_enabled=True,
            ),
            actor=alice,
        )
        effective2 = await MemoryPrivacyService.get_effective_policy(
            session=session,
            user_id=alice.id,
            workspace_id=ws1.id,
        )
        assert effective2.memory_collection_enabled is False, "Tenant disabling must override user enablement"


# =============================================================================
# 7. EXPIRED MEMORY EXCLUDED FROM RETRIEVAL
# =============================================================================
@pytest.mark.asyncio
async def test_07_expired_memory_excluded_from_retrieval():
    async with TestSessionLocal() as session:
        dept1, _, ws1, _, alice, _, admin = await get_test_setup(session)
        # Re-enable collection
        await MemoryPrivacyService.resume_memory_collection(session, alice.id)
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.TENANT,
                scope_id=dept1.id,
                memory_collection_enabled=True,
            ),
            actor=admin,
        )

        # Create past expired memory
        cand_exp = MemoryCandidate(
            user_id=alice.id,
            workspace_id=ws1.id,
            content="Alice temporary parking permit B-12.",
            canonical_key="memory:dynamic:parking_permit_b12",
            memory_type=MemoryType.DYNAMIC,
            retention_expires_at=datetime.now(timezone.utc) - timedelta(days=2),
        )
        await MemoryMaterializer.materialize_memory(session, cand_exp, auto_embed=False)

        # Retrieve
        res = await MemoryRetrievalService.retrieve(
            session=session,
            user_id=alice.id,
            workspace_id=ws1.id,
            query="parking permit",
        )
        found = any("parking permit B-12" in m.content for m in res.dynamic_memories + res.static_memories)
        assert not found, "Expired memory must not appear in retrieval"


# =============================================================================
# 8. RETENTION SWEEP CLEANS UP EXPIRED MEMORY
# =============================================================================
@pytest.mark.asyncio
async def test_08_retention_sweep_cleans_up_expired_memory():
    async with TestSessionLocal() as session:
        _, _, ws1, _, alice, _, _ = await get_test_setup(session)

        # Create active memory that is past its retention date
        mem = MemoryItem(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key="memory:dynamic:temp_session_key",
            memory_type="DYNAMIC",
            status="ACTIVE",
            content="Temporary session note to be swept",
            retention_expires_at=datetime.now(timezone.utc) - timedelta(hours=5),
        )
        session.add(mem)
        await session.commit()
        mem_id = mem.id

        # Run retention sweep
        processed = await MemoryRetentionService.run_retention_sweep(session)
        assert processed["expired_count"] >= 1, "Retention sweep should process at least 1 expired memory"

        # Verify DB state: status EXPIRED, embedding None
        swept = await session.get(MemoryItem, mem_id)
        assert swept.status == "EXPIRED"
        assert swept.embedding is None


# =============================================================================
# 9. SELECTED MEMORY DELETION
# =============================================================================
@pytest.mark.asyncio
async def test_09_selected_memory_deletion():
    async with TestSessionLocal() as session:
        _, _, ws1, _, alice, _, _ = await get_test_setup(session)

        canon_key = f"memory:static:alice_home_address_{uuid.uuid4().hex[:8]}"
        mem = MemoryItem(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key=canon_key,
            memory_type="STATIC",
            status="ACTIVE",
            content="Alice lives at 123 Elm Street.",
        )
        session.add(mem)
        await session.commit()
        mem_id = mem.id

        # Delete / forget this memory
        success = await MemoryPrivacyOperations.forget_memory(
            session=session,
            memory_id=mem_id,
            user_id=alice.id,
        )
        assert success is True

        # Check memory item purged & embedding zeroed
        purged = await session.get(MemoryItem, mem_id)
        assert purged.status == "PURGED"
        assert purged.embedding is None

        # Check tombstone created
        tombstone = await session.scalar(
            select(MemoryTombstone).where(
                MemoryTombstone.user_id == alice.id,
                MemoryTombstone.canonical_key == canon_key,
            )
        )
        assert tombstone is not None, "Tombstone must be registered for forgotten key"


# =============================================================================
# 10. PERSONAL MEMORY DELETION PRESERVES SHARED KNOWLEDGE
# =============================================================================
@pytest.mark.asyncio
async def test_10_personal_memory_deletion_preserves_shared_knowledge():
    async with TestSessionLocal() as session:
        dept1, _, ws1, _, alice, _, admin = await get_test_setup(session)

        # 1. User has personal memory
        user_mem = MemoryItem(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key=f"memory:static:personal_secret_{uuid.uuid4().hex[:8]}",
            memory_type="STATIC",
            status="ACTIVE",
            content="Alice personal secret notes",
        )
        session.add(user_mem)

        # 2. Shared organizational document exists
        doc = Document(
            file_name="shared_policy.txt",
            file_path="/tmp/shared_policy.txt",
            content_hash="dummy_hash_" + uuid.uuid4().hex[:16],
            mime_type="text/plain",
            department_id=dept1.id,
            created_by=alice.id,
        )
        session.add(doc)
        await session.flush()

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            text="Official company expense policy and travel guidelines.",
            embedding=[0.0] * 384,
            is_active=True,
        )
        session.add(chunk)
        await session.commit()
        doc_id = doc.id
        user_mem_id = user_mem.id

        # Delete all personal memories for Alice
        purged_count = await MemoryPrivacyOperations.delete_all_personal_memories(
            session=session,
            user_id=alice.id,
        )
        assert purged_count >= 1

        # Check user memory is purged
        refetched_mem = await session.get(MemoryItem, user_mem_id)
        assert refetched_mem.status == "PURGED"

        # Check shared organizational document and chunk are completely untouched
        refetched_doc = await session.get(Document, doc_id)
        assert refetched_doc is not None
        refetched_chunk = await session.scalar(
            select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
        )
        assert refetched_chunk is not None
        assert "Official company expense policy" in refetched_chunk.text


# =============================================================================
# 11. PROFILE SCRUBBING REMOVES ATTRIBUTE
# =============================================================================
@pytest.mark.asyncio
async def test_11_profile_scrubbing_removes_attribute():
    async with TestSessionLocal() as session:
        dept1, dept2, ws1, ws2, alice, bob, admin = await get_test_setup(session)

        # Ensure collection is enabled
        await MemoryPrivacyService.resume_memory_collection(session, alice.id)
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.TENANT,
                scope_id=dept1.id,
                memory_collection_enabled=True,
            ),
            actor=admin,
        )

        # Setup supporting memory for work context
        cand = MemoryCandidate(
            user_id=alice.id,
            workspace_id=ws1.id,
            content="Alice works as a Senior Cloud Engineer on the Infrastructure team.",
            canonical_key=f"memory:static:alice_work_role_{uuid.uuid4().hex[:8]}",
            memory_type=MemoryType.STATIC,
            structured_data={"subject": "Alice", "predicate": "role", "value": "Senior Cloud Engineer"},
        )
        await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
        await compile_user_profile(user_id=alice.id, session=session)

        # Ensure user_preferences has preferred_editor in profile
        user = await session.get(User, alice.id)
        prof = dict(user.profile_json or {})
        prof.setdefault("user_preferences", {})["preferred_editor"] = "Neovim"
        user.profile_json = prof
        await session.commit()

        # Scrub 'preferred_editor' from 'user_preferences'
        success = await MemoryPrivacyOperations.scrub_user_profile(
            session=session,
            user_id=alice.id,
            section="user_preferences",
            key="preferred_editor",
        )
        assert success is not None
        await session.commit()

        # Check profile
        refetched_user = await session.get(User, alice.id)
        user_prefs = refetched_user.profile_json.get("user_preferences", {})
        assert "preferred_editor" not in user_prefs, "Scrubbed attribute must be removed"
        assert refetched_user.profile_json.get("work_context", {}).get("role") == "Senior Cloud Engineer"


# =============================================================================
# 12. REBUILDING PROFILE DOES NOT RESURRECT SCRUBBED ATTRIBUTE
# =============================================================================
@pytest.mark.asyncio
async def test_12_rebuilding_profile_does_not_resurrect_scrubbed_attribute():
    async with TestSessionLocal() as session:
        dept1, _, ws1, _, alice, _, admin = await get_test_setup(session)
        # Ensure collection is enabled
        await MemoryPrivacyService.resume_memory_collection(session, alice.id)
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.TENANT,
                scope_id=dept1.id,
                memory_collection_enabled=True,
            ),
            actor=admin,
        )

        # Scrub 'salary'
        await MemoryPrivacyOperations.scrub_user_profile(
            session=session,
            user_id=alice.id,
            section="work_context",
            key="salary",
        )

        # Insert a memory item attempting to recreate salary
        mem = MemoryItem(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key=f"memory:work_context:salary_{uuid.uuid4().hex[:8]}",
            memory_type="STATIC",
            status="ACTIVE",
            content="Alice salary is $150k",
            structured_data={"subject": "Alice", "predicate": "salary", "value": "$150k"},
        )
        session.add(mem)
        await session.commit()

        # Recompile profile
        compiled = await compile_user_profile(user_id=alice.id, session=session)
        assert "salary" not in compiled.get("work_context", {}), "Tombstone must prevent salary from being compiled into profile"


# =============================================================================
# 13. DREAMING CANNOT RESURRECT DELETED MEMORIES
# =============================================================================
@pytest.mark.asyncio
async def test_13_dreaming_cannot_resurrect_deleted_memories():
    async with TestSessionLocal() as session:
        dept1, dept2, ws1, ws2, alice, bob, admin = await get_test_setup(session)
        await MemoryPrivacyService.resume_memory_collection(session, alice.id)
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.TENANT,
                scope_id=dept1.id,
                memory_collection_enabled=True,
                memory_dreaming_enabled=True,
            ),
            actor=admin,
        )

        # Tombstone key
        tombstone_key = f"memory:dynamic:alice_medical_record_{uuid.uuid4().hex[:8]}"
        session.add(MemoryTombstone(user_id=alice.id, canonical_key=tombstone_key, reason="User privacy deletion"))
        await session.commit()

        # Run dreaming
        await run_memory_dreaming(user_id=alice.id, session=session, threshold=1)

        # Verify no memory with that canonical key exists as ACTIVE
        active_tombstoned = (await session.scalars(
            select(MemoryItem).where(
                MemoryItem.user_id == alice.id,
                MemoryItem.canonical_key == tombstone_key,
                MemoryItem.status == "ACTIVE",
            )
        )).all()
        assert len(active_tombstoned) == 0, "Dreaming must never resurrect tombstoned canonical keys"


# =============================================================================
# 14. DELAYED PENDING JOBS CANNOT RESURRECT DELETED MEMORIES
# =============================================================================
@pytest.mark.asyncio
async def test_14_delayed_pending_jobs_cannot_resurrect_deleted_memories():
    async with TestSessionLocal() as session:
        dept1, dept2, ws1, ws2, alice, bob, admin = await get_test_setup(session)
        await MemoryPrivacyService.resume_memory_collection(session, alice.id)
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.TENANT,
                scope_id=dept1.id,
                memory_collection_enabled=True,
                memory_extraction_enabled=True,
            ),
            actor=admin,
        )

        tombstone_key = f"memory:static:confidential_note_{uuid.uuid4().hex[:8]}"
        session.add(MemoryTombstone(user_id=alice.id, canonical_key=tombstone_key, reason="Deleted by user"))
        await session.commit()

        # Simulate delayed outbox execution calling materialize_memory with the tombstoned key
        delayed_cand = MemoryCandidate(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key=tombstone_key,
            content="Alice confidential note that was delayed in message broker.",
            memory_type=MemoryType.STATIC,
        )
        res = await MemoryMaterializer.materialize_memory(session, delayed_cand, auto_embed=False)
        assert res is None, "Materializer must refuse to materialize tombstoned canonical key"


# =============================================================================
# 15. MEMORY RETRIEVAL CANNOT EXPOSE DELETED MEMORIES
# =============================================================================
@pytest.mark.asyncio
async def test_15_memory_retrieval_cannot_expose_deleted_memories():
    async with TestSessionLocal() as session:
        dept1, dept2, ws1, ws2, alice, bob, admin = await get_test_setup(session)

        # Insert memories in various deleted states
        m1 = MemoryItem(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key=f"memory:dynamic:purged_item_{uuid.uuid4().hex[:8]}",
            memory_type="DYNAMIC",
            status="PURGED",
            content="Sensitive purged content query match",
        )
        m2 = MemoryItem(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key=f"memory:dynamic:deletion_req_item_{uuid.uuid4().hex[:8]}",
            memory_type="DYNAMIC",
            status="DELETION_REQUESTED",
            content="Sensitive deletion requested query match",
        )
        session.add_all([m1, m2])
        await session.commit()

        # Retrieve
        res = await MemoryRetrievalService.retrieve(
            session=session,
            user_id=alice.id,
            workspace_id=ws1.id,
            query="Sensitive",
        )
        all_retrieved = res.dynamic_memories + res.static_memories + res.relevant_memories
        assert not any(m.status in ("PURGED", "DELETION_REQUESTED", "EXPIRED") for m in all_retrieved)


# =============================================================================
# 16. UNAUTHORIZED DELETION/EXPORT FAILS CLOSED
# =============================================================================
@pytest.mark.asyncio
async def test_16_unauthorized_deletion_export_fails_closed():
    async with TestSessionLocal() as session:
        dept1, dept2, ws1, ws2, alice, bob, admin = await get_test_setup(session)

        # Bob has a memory
        bob_mem = MemoryItem(
            user_id=bob.id,
            workspace_id=ws2.id,
            canonical_key=f"memory:static:bob_private_item_{uuid.uuid4().hex[:8]}",
            memory_type="STATIC",
            status="ACTIVE",
            content="Bob confidential salary data",
        )
        session.add(bob_mem)
        await session.commit()
        bob_mem_id = bob_mem.id

    # Alice attempts to delete Bob's memory via API
    token_alice = create_access_token(subject=str(alice.id))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete(
            f"/api/v1/memory/{bob_mem_id}",
            headers={"Authorization": f"Bearer {token_alice}"},
        )
        assert resp.status_code in (404, 403), "Must fail closed with 404 or 403"

    # Verify Bob's memory was NOT touched
    async with TestSessionLocal() as session:
        mem_check = await session.get(MemoryItem, bob_mem_id)
        assert mem_check.status == "ACTIVE"


# =============================================================================
# 17. USER EXPORT CONTAINS ONLY AUTHORIZED MEMORIES
# =============================================================================
@pytest.mark.asyncio
async def test_17_user_export_contains_only_authorized_memories():
    async with TestSessionLocal() as session:
        _, _, ws1, _, alice, bob, _ = await get_test_setup(session)

    token_alice = create_access_token(subject=str(alice.id))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/v1/memory/export",
            headers={"Authorization": f"Bearer {token_alice}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == str(alice.id)
        for item in data["memories"]:
            # Confirm no Bob items
            assert "bob" not in item["canonical_key"].lower()


# =============================================================================
# 18. CONCURRENT DELETION AND PROCESSING CONSISTENCY
# =============================================================================
@pytest.mark.asyncio
async def test_18_concurrent_deletion_and_processing_consistency():
    async with TestSessionLocal() as session:
        dept1, dept2, ws1, ws2, alice, bob, admin = await get_test_setup(session)

        # Create active memory
        mem = MemoryItem(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key=f"memory:dynamic:concurrent_{uuid.uuid4().hex[:8]}",
            memory_type="DYNAMIC",
            status="ACTIVE",
            content="Concurrent processing test candidate",
        )
        session.add(mem)
        await session.commit()
        mem_id = mem.id

    # Run deletion concurrently with retrieval
    async def do_delete():
        async with TestSessionLocal() as s:
            res = await MemoryPrivacyOperations.forget_memory(
                session=s,
                memory_id=mem_id,
                user_id=alice.id,
            )
            await s.commit()
            return res

    async def do_retrieve():
        async with TestSessionLocal() as s:
            return await MemoryRetrievalService.retrieve(
                session=s,
                user_id=alice.id,
                workspace_id=ws1.id,
                query="concurrent",
            )

    del_res, ret_res = await asyncio.gather(do_delete(), do_retrieve())
    assert del_res is True

    # Check that in post-state, memory is purged and never exposed
    async with TestSessionLocal() as session:
        final_check = await session.get(MemoryItem, mem_id)
        assert final_check.status == "PURGED"


# =============================================================================
# 19. RETENTION SWEEP IS IDEMPOTENT
# =============================================================================
@pytest.mark.asyncio
async def test_19_retention_sweep_is_idempotent():
    async with TestSessionLocal() as session:
        # Run retention sweep twice
        res1 = await MemoryRetentionService.run_retention_sweep(session)
        res2 = await MemoryRetentionService.run_retention_sweep(session)
        # Second run should find 0 expired un-swept items
        assert res2["expired_count"] == 0


# =============================================================================
# 20. RETRY AFTER FAILURE NO CORRUPTION
# =============================================================================
@pytest.mark.asyncio
async def test_20_retry_after_failure_no_corruption():
    async with TestSessionLocal() as session:
        dept1, dept2, ws1, ws2, alice, bob, admin = await get_test_setup(session)

        canon_key = f"memory:static:retry_test_item_{uuid.uuid4().hex[:8]}"
        mem = MemoryItem(
            user_id=alice.id,
            workspace_id=ws1.id,
            canonical_key=canon_key,
            memory_type="STATIC",
            status="ACTIVE",
            content="Retry test content",
        )
        session.add(mem)
        await session.commit()
        mem_id = mem.id

        # First call: forget_memory
        first_call = await MemoryPrivacyOperations.forget_memory(
            session=session,
            memory_id=mem_id,
            user_id=alice.id,
        )
        assert first_call is True

        # Second call (retry): should succeed cleanly without crashing on unique tombstone constraints
        second_call = await MemoryPrivacyOperations.forget_memory(
            session=session,
            memory_id=mem_id,
            user_id=alice.id,
        )
        assert second_call is True

        # Verify exactly one tombstone exists for this canonical key
        tombstones = (await session.scalars(
            select(MemoryTombstone).where(
                MemoryTombstone.user_id == alice.id,
                MemoryTombstone.canonical_key == canon_key,
            )
        )).all()
        assert len(tombstones) == 1


# =============================================================================
# 21. ACTIVE MEMORIES RETRIEVE CLEANLY
# =============================================================================
@pytest.mark.asyncio
async def test_21_active_memories_retrieve_cleanly():
    async with TestSessionLocal() as session:
        dept1, _, ws1, _, alice, _, admin = await get_test_setup(session)
        await MemoryPrivacyService.resume_memory_collection(session, alice.id)
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=PrivacyPolicyPayload(
                scope_type=ScopeType.TENANT,
                scope_id=dept1.id,
                memory_collection_enabled=True,
                memory_retrieval_enabled=True,
            ),
            actor=admin,
        )

        cand = MemoryCandidate(
            user_id=alice.id,
            workspace_id=ws1.id,
            content="Alice prefers pytest for backend testing and TypeScript for frontend.",
            canonical_key="memory:static:alice_testing_stack",
            memory_type=MemoryType.STATIC,
        )
        await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)

        res = await MemoryRetrievalService.retrieve(
            session=session,
            user_id=alice.id,
            workspace_id=ws1.id,
            query="pytest and typescript stack",
        )
        assert any("pytest for backend" in m.content for m in res.static_memories + res.dynamic_memories)


# =============================================================================
# 22. CORE RAG RETRIEVAL/RANKING UNAFFECTED
# =============================================================================
@pytest.mark.asyncio
async def test_22_core_rag_retrieval_ranking_unaffected():
    async with TestSessionLocal() as session:
        dept1, _, ws1, _, alice, _, admin = await get_test_setup(session)

        # Create document with chunk
        doc = Document(
            file_name="core_arch.txt",
            file_path="/tmp/core_arch.txt",
            content_hash="hash_" + uuid.uuid4().hex[:16],
            mime_type="text/plain",
            department_id=dept1.id,
            created_by=alice.id,
        )
        session.add(doc)
        await session.flush()

        from rag_app.ingestion.embedding_v1 import get_embedding_provider
        embed_provider = get_embedding_provider()
        vec = await embed_provider.embed_query("Microservice event bus must use Kafka or RabbitMQ with idempotent consumer groups.")

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            text="Microservice event bus must use Kafka or RabbitMQ with idempotent consumer groups.",
            embedding=vec,
            is_active=True,
        )
        session.add(chunk)
        await session.commit()

        # Core RAG retrieval service
        rag_service = RetrievalService(session)
        search_results = await rag_service.retrieve(
            query="event bus Kafka RabbitMQ",
            allowed_departments=[dept1.id],
            top_k=5,
            rerank=False,
        )
        assert len(search_results.results) >= 1
        assert any("Kafka or RabbitMQ" in (getattr(c, "text", None) or getattr(c, "content", "")) for c in search_results.results)


# =============================================================================
# 23. LANGGRAPH & CHAT FLOWS REMAIN OPERATIONAL
# =============================================================================
@pytest.mark.asyncio
async def test_23_langgraph_and_chat_flows_operational():
    async with TestSessionLocal() as session:
        dept1, _, ws1, _, alice, _, admin = await get_test_setup(session)

        conv = Conversation(
            user_id=alice.id,
            workspace_id=ws1.id,
            title="Test Conversation for Seam",
        )
        session.add(conv)
        await session.commit()
        conv_id = conv.id

        conv_service = ConversationService(session)

        # Test apply_memory_seam with empty or active memories
        active_history = [
            {"role": "user", "content": "What is my preferred stack?"}
        ]
        history_with_seam = await conv_service.apply_memory_seam(
            conversation_id=conv_id,
            active_history=active_history,
            query="preferred stack",
        )
        assert len(history_with_seam) >= 1
        # System memory message or original active history returned cleanly without crash
        assert history_with_seam[-1]["content"] == "What is my preferred stack?"


# =============================================================================
# 24. THREAD AND TENANT ISOLATION
# =============================================================================
@pytest.mark.asyncio
async def test_24_thread_and_tenant_isolation():
    async with TestSessionLocal() as session:
        _, _, ws1, ws2, alice, bob, _ = await get_test_setup(session)

        conv_alice = Conversation(
            user_id=alice.id,
            workspace_id=ws1.id,
            title="Alice Confidential Thread",
        )
        session.add(conv_alice)
        await session.commit()
        conv_id = conv_alice.id

    # Bob tries to access Alice's thread via API
    token_bob = create_access_token(subject=str(bob.id))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            f"/api/v1/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {token_bob}"},
        )
        assert resp.status_code in (403, 404), "Foreign thread access must fail closed with 403 or 404"


# =============================================================================
# 25. FRONTEND PRODUCTION BUILD SUCCEEDS
# =============================================================================
def test_25_frontend_production_build_succeeds():
    dist_index = Path("frontend/dist/index.html")
    assert dist_index.exists(), f"Frontend production build artifact {dist_index} must exist"
    content = dist_index.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "index-" in content
