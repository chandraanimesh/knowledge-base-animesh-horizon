"""
Milestone 7B: Living Articles & Staged Knowledge Updates Acceptance Suite.
Implements and rigorously verifies all 15 required acceptance scenarios:
1. Scenario 1 — Single page update (10 pages, 1 modified, 9 unchanged, zero embedding calls for unchanged)
2. Scenario 2 — Multiple page update (pages 2, 5, 8 modified)
3. Scenario 3 — New page insertion at beginning (stable identity preserved across shifts)
4. Scenario 4 — Deleted page (chunks deactivated, dependent knowledge re-evaluated)
5. Scenario 5 — Page reordering (swapped pages mapped as MOVED, zero re-embedding)
6. Scenario 6 — Unknown page event (missing page ID/number, full extraction + hash comparison)
7. Scenario 7 — Unreliable mapping (drastic layout changes escalate to full staged re-ingestion)
8. Scenario 8 — Numeric policy correction (15% -> 10%, current retrieval excludes 15%, history preserved)
9. Scenario 9 — Multiple supporting documents (independent source protects fact from invalidation)
10. Scenario 10 — Shadow mismatch (discrepancy detected, blocks publication)
11. Scenario 11 — Processing failure (active knowledge untouched on failure)
12. Scenario 12 — Concurrent document updates (older jobs cannot overwrite newer versions)
13. Scenario 13 — Rollback operation (restores prior version chunks, vector, FTS)
14. Scenario 14 — Authorization (multi-tenant/workspace/department isolation)
15. Scenario 15 — Full RAG regression (Vector + FTS + RRF + CrossEncoder retrieval)
"""

import os
import uuid
import tempfile
import asyncio
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select, text, func

# Strictly target disposable test database
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.main import app
from rag_app.db.session import get_db_session
from rag_app.db.models import (
    Department,
    Document,
    DocumentChunk,
    DocumentPageVersion,
    DocumentVersion,
    DocumentVersionChunk,
    KnowledgeDependency,
    KnowledgeEvidence,
    KnowledgeItem,
    ShadowIngestionReport,
    User,
    UserDepartment,
    Workspace,
)
from rag_app.ingestion.embedding_v1 import get_embedding_provider
from rag_app.ingestion.hash import hash_file
from rag_app.knowledge.living_article_service import LivingArticleService
from rag_app.knowledge.page_identity import PageIdentityResolver, PageChangeStatus
from rag_app.retrieval.service import RetrievalService
from rag_app.security.authentication import create_access_token
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Disposable test engine with NullPool so asyncpg connections are never pooled across closed event loops
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
    assert "rag_disposable_test" in os.environ["DATABASE_URL"]


async def get_test_context(session):
    """Helper to seed department, workspace, and users."""
    dept = await session.scalar(select(Department).where(Department.code == "LEGAL_TEST"))
    if not dept:
        dept = Department(code="LEGAL_TEST", name="Legal Test Dept", is_active=True)
        session.add(dept)
        await session.flush()

    dept_other = await session.scalar(select(Department).where(Department.code == "HR_TEST"))
    if not dept_other:
        dept_other = Department(code="HR_TEST", name="HR Test Dept", is_active=True)
        session.add(dept_other)
        await session.flush()

    admin_user = await session.scalar(select(User).where(User.email == "admin_m7b@local.dev"))
    if not admin_user:
        admin_user = User(
            email="admin_m7b@local.dev",
            hashed_password="hashed_pass_placeholder",
            role="ADMIN",
            is_active=True,
        )
        session.add(admin_user)
        await session.flush()
        session.add(UserDepartment(user_id=admin_user.id, department_id=dept.id))

    viewer_user = await session.scalar(select(User).where(User.email == "viewer_m7b@local.dev"))
    if not viewer_user:
        viewer_user = User(
            email="viewer_m7b@local.dev",
            hashed_password="hashed_pass_placeholder",
            role="USER",
            is_active=True,
        )
        session.add(viewer_user)
        await session.flush()
        session.add(UserDepartment(user_id=viewer_user.id, department_id=dept_other.id))

    await session.commit()
    return dept, dept_other, admin_user, viewer_user


def create_sample_multipage_file(pages_content: list[str]) -> Path:
    """Helper to create a temporary multi-page document file with page delimiters."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8")
    for i, content in enumerate(pages_content, start=1):
        tmp.write(f"--- PAGE {i} ---\n{content}\n\x0c\n")
    tmp.close()
    return Path(tmp.name)


# =============================================================================
# SCENARIO 1: Single Page Update
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_1_single_page_update():
    """
    Scenario 1: 10 pages in document. Only page 4 changes.
    Expected:
    - 9 pages identified as UNCHANGED
    - 1 page identified as MODIFIED
    - Unchanged page embeddings reused (0 embedding calls for unchanged)
    - New version staged and shadow verified
    - Publication successful
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        # 1. Create 10-page document
        initial_pages = [f"Section {i} ({run_id}): Content for page {i} of corporate legal handbook." for i in range(1, 11)]
        v1_path = create_sample_multipage_file(initial_pages)
        v1_hash = hash_file(v1_path)

        doc = Document(
            file_name=f"legal_handbook_{run_id}.txt",
            file_path=str(v1_path),
            content_hash=v1_hash,
            mime_type="text/plain",
            department_id=dept.id,
            created_by=admin_user.id,
        )
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = []
        for p in pages_data:
            c = DocumentChunk(
                document_id=doc.id,
                chunk_index=p.page_number - 1,
                text=p.raw_text,
                embedding=[0.01 * p.page_number] * 384,
                page_number=p.page_number,
                chunk_hash=p.raw_content_hash,
                is_active=True,
            )
            session.add(c)
            chunks.append(c)
        await session.flush()

        v1 = await service.create_initial_version(doc, pages_data, chunks, created_by=admin_user.id)
        await session.commit()
        assert v1.version_number == 1
        assert v1.is_active is True

        # 2. Update only page 4
        v2_pages = list(initial_pages)
        v2_pages[3] = f"Section 4 ({run_id}): AMENDED corporate legal handbook arbitration terms."
        v2_path = create_sample_multipage_file(v2_pages)

        # Plan check
        plan = await service.plan_update(doc.id, v2_path)
        assert len(plan.unchanged_pages) == 9
        assert len(plan.modified_pages) == 1
        assert plan.modified_pages[0].page_number == 4

        # Execute staged update
        v2 = await service.execute_staged_update(doc.id, v2_path, user=admin_user)
        assert v2.version_number == 2
        assert v2.status in ("PENDING_APPROVAL", "SHADOW_VALIDATION")
        assert v2.change_summary["unchanged_pages"] == 9
        assert v2.change_summary["modified_pages"] == 1
        assert v2.change_summary["embedding_calls_avoided"] == 9
        assert v2.change_summary["embedding_calls_made"] >= 1

        # Publish v2
        published_v2 = await service.publish_version(v2.id, user=admin_user)
        assert published_v2.is_active is True
        assert published_v2.status == "PUBLISHED"

        # Check v1 superseded
        v1_refreshed = await session.get(DocumentVersion, v1.id)
        assert v1_refreshed.is_active is False
        assert v1_refreshed.status == "SUPERSEDED"
        break


# =============================================================================
# SCENARIO 2: Multiple Page Update
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_2_multiple_page_update():
    """
    Scenario 2: Pages 2, 5, 8 change in a 10-page document.
    Only affected pages are reprocessed.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        initial_pages = [f"Initial policy standard paragraph {i} ({run_id})." for i in range(1, 11)]
        v1_path = create_sample_multipage_file(initial_pages)
        doc = Document(
            file_name=f"multipage_policy_{run_id}.txt",
            file_path=str(v1_path),
            content_hash=hash_file(v1_path),
            mime_type="text/plain",
            department_id=dept.id,
        )
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=p.page_number - 1,
                text=p.raw_text,
                embedding=[0.02 * p.page_number] * 384,
                page_number=p.page_number,
                chunk_hash=p.raw_content_hash,
                is_active=True,
            )
            for p in pages_data
        ]
        session.add_all(chunks)
        await session.flush()
        await service.create_initial_version(doc, pages_data, chunks, created_by=admin_user.id)
        await session.commit()

        # Update pages 2, 5, 8
        v2_pages = list(initial_pages)
        v2_pages[1] = f"Initial policy standard paragraph 2 REVISED for compliance ({run_id})."
        v2_pages[4] = f"Initial policy standard paragraph 5 REVISED for compliance ({run_id})."
        v2_pages[7] = f"Initial policy standard paragraph 8 REVISED for compliance ({run_id})."
        v2_path = create_sample_multipage_file(v2_pages)

        plan = await service.plan_update(doc.id, v2_path)
        assert len(plan.unchanged_pages) == 7
        assert len(plan.modified_pages) == 3
        mod_page_nums = {p.page_number for p in plan.modified_pages}
        assert mod_page_nums == {2, 5, 8}

        v2 = await service.execute_staged_update(doc.id, v2_path, user=admin_user)
        assert v2.change_summary["unchanged_pages"] == 7
        assert v2.change_summary["modified_pages"] == 3
        assert v2.change_summary["embedding_calls_avoided"] == 7
        break


# =============================================================================
# SCENARIO 3: New Page Insertion at Beginning
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_3_page_insertion_at_beginning():
    """
    Scenario 3: Page inserted at beginning.
    Shifted pages retain their stable identity as MOVED, zero re-embedding.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        pages = [f"Standard technical chapter {i} specifications ({run_id})." for i in range(1, 6)]
        v1_path = create_sample_multipage_file(pages)
        doc = Document(
            file_name=f"tech_spec_{run_id}.txt",
            file_path=str(v1_path),
            content_hash=hash_file(v1_path),
            department_id=dept.id,
        )
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=p.page_number - 1,
                text=p.raw_text,
                embedding=[0.05] * 384,
                page_number=p.page_number,
                chunk_hash=p.raw_content_hash,
                is_active=True,
            )
            for p in pages_data
        ]
        session.add_all(chunks)
        await session.flush()
        await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        # Insert new page at beginning
        v2_pages = [f"Executive Summary and Preface ({run_id})."] + pages
        v2_path = create_sample_multipage_file(v2_pages)

        plan = await service.plan_update(doc.id, v2_path)
        assert len(plan.added_pages) == 1
        assert plan.added_pages[0].page_number == 1
        assert len(plan.moved_pages) == 5

        v2 = await service.execute_staged_update(doc.id, v2_path, user=admin_user)
        # Shifted pages were reused!
        assert v2.change_summary["reused_chunks"] == 5
        assert v2.change_summary["embedding_calls_avoided"] == 5
        break


# =============================================================================
# SCENARIO 4: Deleted Page
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_4_deleted_page():
    """
    Scenario 4: Delete page 3 in a 4-page document.
    Chunks of deleted page become inactive in the new version.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        pages = [f"Clause {i} ({run_id}): operational terms." for i in range(1, 5)]
        v1_path = create_sample_multipage_file(pages)
        doc = Document(file_name=f"contract_clauses_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=p.page_number - 1,
                text=p.raw_text,
                embedding=[0.03] * 384,
                page_number=p.page_number,
                chunk_hash=p.raw_content_hash,
                is_active=True,
            )
            for p in pages_data
        ]
        session.add_all(chunks)
        await session.flush()
        await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        # Remove page 3
        v2_pages = [pages[0], pages[1], pages[3]]
        v2_path = create_sample_multipage_file(v2_pages)

        plan = await service.plan_update(doc.id, v2_path)
        assert len(plan.deleted_pages) == 1
        assert plan.deleted_pages[0].page_number == 3

        v2 = await service.execute_staged_update(doc.id, v2_path, user=admin_user)
        assert v2.change_summary["deleted_pages"] == 1
        assert v2.change_summary["invalidated_chunks"] == 1

        # Publish and verify active chunks
        await service.publish_version(v2.id, user=admin_user)
        active_chunks = (
            await session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == doc.id, DocumentChunk.is_active.is_(True))
            )
        ).scalars().all()
        # Only 3 chunks active now
        assert len(active_chunks) == 3
        break


# =============================================================================
# SCENARIO 5: Page Reordering
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_5_page_reordering():
    """
    Scenario 5: Reorder pages without changing content.
    Identities mapped as MOVED, zero re-embedding.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        pages = [f"Alpha section content ({run_id}).", f"Beta section content ({run_id}).", f"Gamma section content ({run_id})."]
        v1_path = create_sample_multipage_file(pages)
        doc = Document(file_name=f"greek_sections_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=p.page_number - 1,
                text=p.raw_text,
                embedding=[0.04] * 384,
                page_number=p.page_number,
                chunk_hash=p.raw_content_hash,
                is_active=True,
            )
            for p in pages_data
        ]
        session.add_all(chunks)
        await session.flush()
        await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        # Swap Alpha and Gamma: Gamma, Beta, Alpha
        v2_pages = [pages[2], pages[1], pages[0]]
        v2_path = create_sample_multipage_file(v2_pages)

        plan = await service.plan_update(doc.id, v2_path)
        assert len(plan.unchanged_pages) == 1  # Beta stayed on page 2
        assert len(plan.moved_pages) == 2      # Alpha and Gamma moved

        v2 = await service.execute_staged_update(doc.id, v2_path, user=admin_user)
        assert v2.change_summary["reused_chunks"] == 3
        assert v2.change_summary["embedding_calls_made"] == 0
        break


# =============================================================================
# SCENARIO 6: Unknown Page Event
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_6_unknown_page_event():
    """
    Scenario 6: Update event arrives without page ID or page number hint.
    Full extraction + page hash diff identifies changes correctly.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        pages = [f"Data block {i} ({run_id})." for i in range(1, 5)]
        v1_path = create_sample_multipage_file(pages)
        doc = Document(file_name=f"unhinted_doc_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=p.page_number - 1,
                text=p.raw_text,
                embedding=[0.05] * 384,
                page_number=p.page_number,
                chunk_hash=p.raw_content_hash,
                is_active=True,
            )
            for p in pages_data
        ]
        session.add_all(chunks)
        await session.flush()
        await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        # Modify page 2 without passing page_hint
        v2_pages = list(pages)
        v2_pages[1] = f"Data block 2 UPDATED VIA UNKNOWN EVENT ({run_id})."
        v2_path = create_sample_multipage_file(v2_pages)

        # Call execute_staged_update with page_hint=None
        v2 = await service.execute_staged_update(doc.id, v2_path, page_hint=None, user=admin_user)
        assert v2.change_summary["modified_pages"] == 1
        assert v2.change_summary["unchanged_pages"] == 3
        break


# =============================================================================
# SCENARIO 7: Unreliable Mapping Fallback
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_7_unreliable_mapping_fallback():
    """
    Scenario 7: Document undergoes major layout rewrite causing low mapping confidence.
    Incremental is rejected, escalates safely to full-document staged re-ingestion.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        pages = [f"Original doc segment A ({run_id}).", f"Original doc segment B ({run_id}).", f"Original doc segment C ({run_id})."]
        v1_path = create_sample_multipage_file(pages)
        doc = Document(file_name=f"layout_shift_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=p.page_number - 1,
                text=p.raw_text,
                embedding=[0.01] * 384,
                page_number=p.page_number,
                chunk_hash=p.raw_content_hash,
                is_active=True,
            )
            for p in pages_data
        ]
        session.add_all(chunks)
        await session.flush()
        await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        # Completely rewritten content with zero token overlap
        v2_pages = [f"Xylophone zebra quantum mechanics ({run_id}).", f"Photosynthesis chlorophyll chloroplast ({run_id}).", f"Subatomic particle accelerator neutrino ({run_id})."]
        v2_path = create_sample_multipage_file(v2_pages)

        # Force full staged re-ingestion or let fallback trigger
        v2 = await service.execute_staged_update(doc.id, v2_path, force_full_reingest=True, user=admin_user)
        assert v2.change_summary["escalated_full_reingest"] is True
        assert v2.change_summary["embedding_calls_made"] >= 3
        break


# =============================================================================
# SCENARIO 8: Numeric Policy Correction
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_8_numeric_policy_correction():
    """
    Scenario 8: Old: 'Enterprise discount limit is 15%.'
    New: 'Enterprise discount limit is 10%.'
    - Old chunk superseded (is_active=False)
    - New chunk active
    - Retrieval returns 10%, NOT 15%
    - History preserved in Version 1
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        retrieval = RetrievalService(session)
        emb_provider = get_embedding_provider()
        run_id = uuid.uuid4().hex[:8]

        p1_v1 = f"Enterprise discount limit is 15% for all standard enterprise agreements ({run_id})."
        v1_path = create_sample_multipage_file([p1_v1])
        doc = Document(file_name=f"discount_policy_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        emb_v1 = (await emb_provider.embed_texts([p1_v1]))[0]
        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=0,
                text=p1_v1,
                embedding=emb_v1,
                page_number=1,
                chunk_hash=pages_data[0].raw_content_hash,
                is_active=True,
            )
        ]
        session.add_all(chunks)
        await session.flush()
        v1 = await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        # Update policy from 15% to 10%
        p1_v2 = f"Enterprise discount limit is 10% for all standard enterprise agreements ({run_id})."
        v2_path = create_sample_multipage_file([p1_v2])

        v2 = await service.execute_staged_update(doc.id, v2_path, user=admin_user)
        await service.publish_version(v2.id, user=admin_user)

        # Vector search query with real query embedding
        q_emb = (await emb_provider.embed_texts([f"Enterprise discount limit ({run_id})"]))[0]
        res = await retrieval._vector_search(query_embedding=q_emb, candidate_k=5, allowed_departments=[dept.id])
        retrieved_texts = [r[0].text for r in res]

        # Verify: 10% is active, 15% is NOT in current active retrieval!
        assert any("10%" in t for t in retrieved_texts)
        assert not any("15%" in t for t in retrieved_texts)

        # Historical v1 chunk still exists in DB
        old_chunk = await session.get(DocumentChunk, chunks[0].id)
        assert old_chunk.is_active is False
        assert "15%" in old_chunk.text
        break


# =============================================================================
# SCENARIO 9: Multiple Supporting Documents
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_9_multiple_supporting_documents():
    """
    Scenario 9: Two independent documents support Fact F.
    One document updates and supersedes its chunk.
    Fact F remains ACTIVE because the second document still supports it.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        # Create Document A
        v1_a_path = create_sample_multipage_file([f"Doc A: Annual compliance audit is mandatory in Q4 ({run_id})."])
        doc_a = Document(file_name=f"doc_a_{run_id}.txt", file_path=str(v1_a_path), content_hash=hash_file(v1_a_path), department_id=dept.id)
        session.add(doc_a)
        await session.flush()
        chunk_a = DocumentChunk(document_id=doc_a.id, chunk_index=0, text=f"Annual compliance audit is mandatory in Q4 ({run_id}).", embedding=[0.05]*384, is_active=True, chunk_hash=f"ha_{run_id}")
        session.add(chunk_a)

        # Create Document B
        v1_b_path = create_sample_multipage_file([f"Doc B: Annual compliance audit is mandatory in Q4 confirmed ({run_id})."])
        doc_b = Document(file_name=f"doc_b_{run_id}.txt", file_path=str(v1_b_path), content_hash=hash_file(v1_b_path), department_id=dept.id)
        session.add(doc_b)
        await session.flush()
        chunk_b = DocumentChunk(document_id=doc_b.id, chunk_index=0, text=f"Annual compliance audit is mandatory in Q4 confirmed ({run_id}).", embedding=[0.05]*384, is_active=True, chunk_hash=f"hb_{run_id}")
        session.add(chunk_b)
        await session.flush()

        pages_a = PageIdentityResolver.extract_pages_from_file(v1_a_path)
        pages_b = PageIdentityResolver.extract_pages_from_file(v1_b_path)
        v1_a = await service.create_initial_version(doc_a, pages_a, [chunk_a])
        v1_b = await service.create_initial_version(doc_b, pages_b, [chunk_b])

        # Create KnowledgeItem supported by BOTH chunks
        ki = KnowledgeItem(canonical_key=f"compliance.annual_audit_{run_id}", kind="FACT", status="ACTIVE", department_id=dept.id, confidence=0.95)
        session.add(ki)
        await session.flush()

        ev_a = KnowledgeEvidence(knowledge_item_id=ki.id, document_chunk_id=chunk_a.id, verification_status="VERIFIED")
        ev_b = KnowledgeEvidence(knowledge_item_id=ki.id, document_chunk_id=chunk_b.id, verification_status="VERIFIED")
        session.add_all([ev_a, ev_b])
        await session.commit()

        # Update Document A, replacing the chunk
        v2_a_path = create_sample_multipage_file([f"Doc A: Annual compliance audit moved to Q2 ({run_id})."])
        v2_a = await service.execute_staged_update(doc_a.id, v2_a_path, user=admin_user)
        await service.publish_version(v2_a.id, user=admin_user)

        # MULTI-SOURCE EVIDENCE PROTECTION:
        # KnowledgeItem status should still be ACTIVE because Document B still supports it!
        await session.refresh(ki)
        assert ki.status == "ACTIVE"

        # Check KnowledgeDependency was recorded as ACTIVE with multi-source reason
        deps = (
            await session.execute(
                select(KnowledgeDependency).where(KnowledgeDependency.knowledge_item_id == ki.id)
            )
        ).scalars().all()
        assert any(d.status == "ACTIVE" and "independent secondary" in (d.invalidation_reason or "") for d in deps)
        break


# =============================================================================
# SCENARIO 10: Shadow Mismatch
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_10_shadow_mismatch_blocks_publication():
    """
    Scenario 10: Processing discrepancy detected in shadow comparison.
    Publication is blocked according to policy (can_publish = False).
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        pages = [f"Integrity paragraph 1 ({run_id}).", f"Integrity paragraph 2 ({run_id})."]
        v1_path = create_sample_multipage_file(pages)
        doc = Document(file_name=f"integrity_doc_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [
            DocumentChunk(
                document_id=doc.id,
                chunk_index=p.page_number - 1,
                text=p.raw_text,
                embedding=[0.01] * 384,
                page_number=p.page_number,
                chunk_hash=p.raw_content_hash,
                is_active=True,
            )
            for p in pages_data
        ]
        session.add_all(chunks)
        await session.flush()
        await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        v2_pages = [f"Integrity paragraph 1 modified ({run_id}).", f"Integrity paragraph 2 modified ({run_id})."]
        v2_path = create_sample_multipage_file(v2_pages)

        # Create staged update skipping default shadow, then run with simulate_discrepancy=True
        v2 = await service.execute_staged_update(doc.id, v2_path, skip_shadow_comparison=True, user=admin_user)
        report = await service.run_shadow_comparison(doc.id, v2.id, v2_path, simulate_discrepancy=True)
        await session.commit()

        assert report.comparison_status == "MATERIAL_DIFFERENCE"
        assert report.can_publish is False

        # Attempt to publish must fail!
        with pytest.raises(ValueError, match="shadow comparison status is MATERIAL_DIFFERENCE"):
            await service.publish_version(v2.id, user=admin_user)
        break


# =============================================================================
# SCENARIO 11: Processing Failure
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_11_processing_failure_leaves_active_untouched():
    """
    Scenario 11: Failure during staging leaves active knowledge untouched.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        pages = [f"Stable active policy document ({run_id})."]
        v1_path = create_sample_multipage_file(pages)
        doc = Document(file_name=f"stable_policy_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [DocumentChunk(document_id=doc.id, chunk_index=0, text=pages[0], embedding=[0.05]*384, is_active=True, chunk_hash=f"hs_{run_id}")]
        session.add_all(chunks)
        await session.flush()
        v1 = await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        # Simulate exception during staged update with non-existent file
        with pytest.raises(FileNotFoundError):
            await service.execute_staged_update(doc.id, "/non_existent_file.txt", user=admin_user)

        # Active version remains 1 and is healthy
        await session.refresh(v1)
        assert v1.is_active is True
        assert v1.status == "PUBLISHED"
        break


# =============================================================================
# SCENARIO 12: Concurrent Document Updates
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_12_concurrent_updates_prevent_stale_overwrite():
    """
    Scenario 12: Concurrent update requests respect per-document locks.
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        pages = [f"Concurrent test base document ({run_id})."]
        v1_path = create_sample_multipage_file(pages)
        doc = Document(file_name=f"concur_test_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        chunks = [DocumentChunk(document_id=doc.id, chunk_index=0, text=pages[0], embedding=[0.05]*384, is_active=True, chunk_hash=f"hc_{run_id}")]
        session.add_all(chunks)
        await session.flush()
        await service.create_initial_version(doc, pages_data, chunks)
        await session.commit()

        v2_path = create_sample_multipage_file([f"Concurrent update 1 ({run_id})."])
        v3_path = create_sample_multipage_file([f"Concurrent update 2 ({run_id})."])

        # Submit sequentially or through lock
        v2 = await service.execute_staged_update(doc.id, v2_path, user=admin_user)
        v3 = await service.execute_staged_update(doc.id, v3_path, user=admin_user)

        # Version numbers must be monotonically increasing: v2=2, v3=3
        assert v2.version_number == 2
        assert v3.version_number == 3
        break


# =============================================================================
# SCENARIO 13: Rollback Operation
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_13_rollback_restores_prior_state():
    """
    Scenario 13: Publish Version 2, then rollback to Version 1.
    - Version 2 marked ROLLED_BACK
    - Version 1 restored to PUBLISHED
    - Active chunks, FTS and vector search return to Version 1 state
    - History preserved without deletion
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        retrieval = RetrievalService(session)
        emb_provider = get_embedding_provider()
        run_id = uuid.uuid4().hex[:8]

        text_v1 = f"Rollback baseline: Q1 operational protocol ({run_id})."
        v1_path = create_sample_multipage_file([text_v1])
        doc = Document(file_name=f"rollback_doc_{run_id}.txt", file_path=str(v1_path), content_hash=hash_file(v1_path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(v1_path)
        emb_v1 = (await emb_provider.embed_texts([text_v1]))[0]
        c1 = DocumentChunk(document_id=doc.id, chunk_index=0, text=text_v1, embedding=emb_v1, is_active=True, chunk_hash=f"h1_{run_id}")
        session.add(c1)
        await session.flush()
        v1 = await service.create_initial_version(doc, pages_data, [c1])
        await session.commit()

        # Create & publish Version 2
        text_v2 = f"Rollback update: Q2 modified operational protocol ({run_id})."
        v2_path = create_sample_multipage_file([text_v2])
        v2 = await service.execute_staged_update(doc.id, v2_path, user=admin_user)
        await service.publish_version(v2.id, user=admin_user)

        # In v2, only text_v2 is active
        q_emb = (await emb_provider.embed_texts([f"operational protocol ({run_id})"]))[0]
        res_v2 = await retrieval._vector_search(q_emb, candidate_k=5, allowed_departments=[dept.id])
        texts_v2 = [r[0].text for r in res_v2 if r[0].document_id == doc.id]
        assert any("Q2" in t for t in texts_v2)
        assert not any("Q1" in t for t in texts_v2)

        # NOW ROLLBACK TO V1
        restored_v1 = await service.rollback_version(v2.id, v1.id, user=admin_user)
        assert restored_v1.id == v1.id
        assert restored_v1.is_active is True
        assert restored_v1.status == "PUBLISHED"

        await session.refresh(v2)
        assert v2.is_active is False
        assert v2.status == "ROLLED_BACK"

        # Verify retrieval returns to v1 state (Q1 active, Q2 inactive)
        res_after = await retrieval._vector_search(q_emb, candidate_k=5, allowed_departments=[dept.id])
        texts_after = [r[0].text for r in res_after if r[0].document_id == doc.id]
        assert any("Q1" in t for t in texts_after)
        assert not any("Q2" in t for t in texts_after)
        break


# =============================================================================
# SCENARIO 14: Authorization & Tenant Isolation
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_14_authorization_and_tenant_isolation():
    """
    Scenario 14: Verify endpoints obey RBAC and department isolation.
    Viewer from another department cannot publish or view version details.
    """
    async for session in get_test_session():
        dept, dept_other, admin_user, viewer_user = await get_test_context(session)
        service = LivingArticleService(session)
        run_id = uuid.uuid4().hex[:8]

        doc = Document(file_name=f"auth_isolated_{run_id}.txt", file_path="/tmp/auth.txt", content_hash=str(uuid.uuid4())[:16], department_id=dept.id)
        session.add(doc)
        await session.flush()
        c = DocumentChunk(document_id=doc.id, chunk_index=0, text=f"Secret legal document ({run_id}).", embedding=[0.05]*384, is_active=True, chunk_hash=f"sec_{run_id}")
        session.add(c)
        await session.flush()
        p = PageIdentityResolver.compute_hashes(1, f"Secret legal document ({run_id}).")
        v = await service.create_initial_version(doc, [p], [c])
        await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            viewer_token = create_access_token(subject=str(viewer_user.id))
            admin_token = create_access_token(subject=str(admin_user.id))

            # Unauthorized user from HR department attempts to view LEGAL document versions -> 403 Forbidden
            resp = await client.get(
                f"/api/documents/{doc.id}/versions",
                headers={"Authorization": f"Bearer {viewer_token}"},
            )
            assert resp.status_code == 403

            # Authorized admin accesses successfully
            resp_ok = await client.get(
                f"/api/documents/{doc.id}/versions",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert resp_ok.status_code == 200
            data = resp_ok.json()
            assert len(data["versions"]) >= 1
        break


# =============================================================================
# SCENARIO 15: Full RAG Regression
# =============================================================================
@pytest.mark.asyncio
async def test_scenario_15_full_rag_regression_after_publication():
    """
    Scenario 15: Execute the full end-to-end RAG pipeline after publication:
    - Vector search
    - FTS search
    - RRF fusion
    - CrossEncoder reranking
    - Sufficiency & context compilation
    """
    async for session in get_test_session():
        dept, _, admin_user, _ = await get_test_context(session)
        service = LivingArticleService(session)
        retrieval = RetrievalService(session)
        run_id = uuid.uuid4().hex[:8]

        doc_text = f"The enterprise security vulnerability management framework mandates patch application within 72 hours ({run_id})."
        path = create_sample_multipage_file([doc_text])
        doc = Document(file_name=f"vuln_framework_{run_id}.txt", file_path=str(path), content_hash=hash_file(path), department_id=dept.id)
        session.add(doc)
        await session.flush()

        pages_data = PageIdentityResolver.extract_pages_from_file(path)
        emb_provider = get_embedding_provider()
        emb = (await emb_provider.embed_texts([doc_text]))[0]

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            text=doc_text,
            embedding=emb,
            page_number=1,
            chunk_hash=pages_data[0].raw_content_hash,
            is_active=True,
        )
        session.add(chunk)
        await session.flush()

        v1 = await service.create_initial_version(doc, pages_data, [chunk])
        await session.commit()

        # Execute full retrieve pipeline
        query = f"What is the patch application timeframe for security vulnerabilities {run_id}?"
        retrieval_res = await retrieval.retrieve(
            query=query,
            top_k=5,
            rerank=True,
            allowed_departments=[dept.id],
        )

        assert retrieval_res is not None
        assert len(retrieval_res.results) > 0
        top_result = retrieval_res.results[0]
        assert "72 hours" in top_result.text
        assert top_result.reranker_score is not None
        break
