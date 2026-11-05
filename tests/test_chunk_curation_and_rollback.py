"""
Milestone 7A.1 Acceptance Suite: Chunk Curation, Versioning, Diff, and Rollback Lifecycle.

Verifies:
1. First edit and version increment (v1 -> v2) with matching embedding snapshot.
2. Embedding-generation failure causes atomic rollback with zero partial state.
3. Concurrent edit conflict with stale base_version raises HTTP 409 without state change.
4. Accurate diff computation between versions with metrics.
5. Rollback creates forward version (v3) preserving all prior audit history.
6. Incompatible embedding snapshot triggers safe re-embedding during rollback.
7. Role enforcement: Viewer role allows read/diff but strictly rejects mutation (403 Forbidden).
8. Multi-tenant isolation: Cross-workspace access fails closed with 403 Forbidden.
9. Managed CRM content protection: Chunks from Twenty CRM sync cannot be directly edited (403 Forbidden).
10. Hierarchical OV-1/OV-2 integrity: Chunk edit flags section abstract/summary as stale.
11. Vector retrieval reflects active curated state immediately post-edit and post-rollback.
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timezone
import hashlib
from typing import Any, Optional
import uuid
from unittest.mock import AsyncMock

from fastapi import HTTPException
import pytest
import pytest_asyncio
from sqlalchemy import select, and_, text

from rag_app.core.config import get_settings
from rag_app.db.models import (
    Department,
    Document,
    DocumentChunk,
    DocumentChunkVersion,
    DocumentSection,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.knowledge.chunk_curation_service import ChunkCurationService
from rag_app.schemas.chunk_curation import (
    ChunkEditRequest,
    ChunkRollbackRequest,
)

settings = get_settings()


class MockEmbeddingProvider:
    """Deterministic embedding provider for fast, reproducible tests."""

    def __init__(self, dim: int = 384, fail_on_embed: bool = False):
        self.dim = dim
        self.fail_on_embed = fail_on_embed
        self.call_count = 0

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        if self.fail_on_embed:
            raise RuntimeError("Simulated embedding provider crash/timeout")

        results = []
        for text_str in texts:
            # Deterministic vector based on SHA-256 seed
            h = hashlib.sha256(text_str.encode("utf-8")).digest()
            vec = [float((h[i % len(h)] / 255.0) - 0.5) for i in range(self.dim)]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            results.append([x / norm for x in vec])
        return results


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_curation_context():
    """
    Sets up two isolated workspaces (Workspace A and Workspace B)
    with separate departments, owners, editors, and viewers.
    """
    async with AsyncSessionLocal() as session:
        # 1. Permissions
        perm_view = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not perm_view:
            perm_view = Permission(id=uuid.uuid4(), code="VIEW", description="View access", is_active=True)
            session.add(perm_view)
        perm_edit = await session.scalar(select(Permission).where(Permission.code == "EDIT"))
        if not perm_edit:
            perm_edit = Permission(id=uuid.uuid4(), code="EDIT", description="Edit access", is_active=True)
            session.add(perm_edit)
        await session.commit()

        # 2. Departments
        uid = uuid.uuid4().hex[:6]
        dept_a = Department(id=uuid.uuid4(), name=f"Dept-A-{uid}", code=f"DPT-A-{uid}", is_active=True)
        dept_b = Department(id=uuid.uuid4(), name=f"Dept-B-{uid}", code=f"DPT-B-{uid}", is_active=True)
        session.add_all([dept_a, dept_b])
        await session.commit()

        # 3. Workspaces
        ws_a = Workspace(id=uuid.uuid4(), name=f"Workspace A {uid}", code=f"WS-A-{uid}", department_id=dept_a.id, is_active=True)
        ws_b = Workspace(id=uuid.uuid4(), name=f"Workspace B {uid}", code=f"WS-B-{uid}", department_id=dept_b.id, is_active=True)
        session.add_all([ws_a, ws_b])
        await session.commit()

        # 4. Users
        owner_a = User(id=uuid.uuid4(), email=f"owner-a-{uid}@example.com", hashed_password="pw", role="USER", is_active=True)
        editor_a = User(id=uuid.uuid4(), email=f"editor-a-{uid}@example.com", hashed_password="pw", role="USER", is_active=True)
        viewer_a = User(id=uuid.uuid4(), email=f"viewer-a-{uid}@example.com", hashed_password="pw", role="USER", is_active=True)
        user_b = User(id=uuid.uuid4(), email=f"user-b-{uid}@example.com", hashed_password="pw", role="USER", is_active=True)
        admin = User(id=uuid.uuid4(), email=f"admin-{uid}@example.com", hashed_password="pw", role="ADMIN", is_active=True)
        session.add_all([owner_a, editor_a, viewer_a, user_b, admin])
        await session.commit()

        # 5. Memberships & Department links
        for u, role in [(owner_a, "OWNER"), (editor_a, "EDITOR"), (viewer_a, "VIEWER")]:
            session.add(UserDepartment(user_id=u.id, department_id=dept_a.id))
            session.add(UserPermission(user_id=u.id, department_id=dept_a.id, permission_id=perm_view.id))
            session.add(WorkspaceMember(workspace_id=ws_a.id, user_id=u.id, role=role))

        # Editor A also has EDIT in department
        session.add(UserPermission(user_id=editor_a.id, department_id=dept_a.id, permission_id=perm_edit.id))
        session.add(UserPermission(user_id=owner_a.id, department_id=dept_a.id, permission_id=perm_edit.id))

        # User B in Workspace B
        session.add(UserDepartment(user_id=user_b.id, department_id=dept_b.id))
        session.add(UserPermission(user_id=user_b.id, department_id=dept_b.id, permission_id=perm_view.id))
        session.add(UserPermission(user_id=user_b.id, department_id=dept_b.id, permission_id=perm_edit.id))
        session.add(WorkspaceMember(workspace_id=ws_b.id, user_id=user_b.id, role="OWNER"))

        await session.commit()

        # 6. Sample Hierarchical Document & Section in Workspace A
        mock_provider = MockEmbeddingProvider()
        init_text = "Original clause: The liability of the vendor is capped at $50,000."
        init_emb = (await mock_provider.embed_texts([init_text]))[0]
        init_hash = hashlib.sha256(init_text.encode("utf-8")).hexdigest()

        doc_a = Document(
            id=uuid.uuid4(),
            file_name=f"contract_{uid}.md",
            file_path=f"/docs/contract_{uid}.md",
            content_hash=hashlib.sha256(f"doc_{uid}".encode()).hexdigest(),
            department_id=dept_a.id,
            workspace_id=ws_a.id,
            created_by=owner_a.id,
        )
        session.add(doc_a)
        await session.flush()

        section_a = DocumentSection(
            id=uuid.uuid4(),
            document_id=doc_a.id,
            node_type="SECTION",
            title="Section 4: Liability",
            level=1,
            ordinal_path="4",
            l0_abstract="Liability terms abstract",
            l1_overview="Overview of liability limits and indemnities",
            metadata_={"is_stale": False},
        )
        session.add(section_a)
        await session.flush()

        chunk_a = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_a.id,
            chunk_index=0,
            text=init_text,
            embedding=init_emb,
            chunk_hash=init_hash,
            current_version=1,
            node_id=section_a.id,
        )
        session.add(chunk_a)

        # Managed CRM document in Workspace A
        crm_text = "# CRM Company: Acme Corp\nDetails from Twenty CRM"
        crm_emb = (await mock_provider.embed_texts([crm_text]))[0]
        crm_doc = Document(
            id=uuid.uuid4(),
            file_name=f"twenty_company_{uid}.md",
            file_path=f"/crm/twenty_company_{uid}.md",
            content_hash=hashlib.sha256(f"crm_{uid}".encode()).hexdigest(),
            department_id=dept_a.id,
            workspace_id=ws_a.id,
            created_by=owner_a.id,
        )
        session.add(crm_doc)
        await session.flush()

        crm_chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=crm_doc.id,
            chunk_index=0,
            text=crm_text,
            embedding=crm_emb,
            chunk_hash=hashlib.sha256(crm_text.encode()).hexdigest(),
            current_version=1,
            metadata_={"source": "twenty_crm_mcp", "doc_type": "CRM_RECORD"},
        )
        session.add(crm_chunk)

        await session.commit()

        context = {
            "dept_a": dept_a,
            "dept_b": dept_b,
            "ws_a": ws_a,
            "ws_b": ws_b,
            "owner_a": owner_a,
            "editor_a": editor_a,
            "viewer_a": viewer_a,
            "user_b": user_b,
            "admin": admin,
            "doc_a": doc_a,
            "section_a": section_a,
            "chunk_a": chunk_a,
            "crm_doc": crm_doc,
            "crm_chunk": crm_chunk,
            "init_text": init_text,
            "init_emb": init_emb,
            "init_hash": init_hash,
        }

        yield context

        # Teardown
        try:
            await session.execute(text("DELETE FROM document_chunk_versions WHERE chunk_id IN (:ca, :cc)"), {"ca": chunk_a.id, "cc": crm_chunk.id})
            await session.execute(text("DELETE FROM document_chunks WHERE id IN (:ca, :cc)"), {"ca": chunk_a.id, "cc": crm_chunk.id})
            await session.execute(text("DELETE FROM document_sections WHERE id = :sid"), {"sid": section_a.id})
            await session.execute(text("DELETE FROM documents WHERE id IN (:da, :dc)"), {"da": doc_a.id, "dc": crm_doc.id})
            await session.execute(text("DELETE FROM workspace_members WHERE workspace_id IN (:wa, :wb)"), {"wa": ws_a.id, "wb": ws_b.id})
            await session.execute(text("DELETE FROM user_permissions WHERE user_id IN (:oa, :ea, :va, :ub)"), {"oa": owner_a.id, "ea": editor_a.id, "va": viewer_a.id, "ub": user_b.id})
            await session.execute(text("DELETE FROM user_departments WHERE user_id IN (:oa, :ea, :va, :ub)"), {"oa": owner_a.id, "ea": editor_a.id, "va": viewer_a.id, "ub": user_b.id})
            await session.execute(text("DELETE FROM workspaces WHERE id IN (:wa, :wb)"), {"wa": ws_a.id, "wb": ws_b.id})
            await session.execute(text("DELETE FROM departments WHERE id IN (:da, :db)"), {"da": dept_a.id, "db": dept_b.id})
            await session.execute(text("DELETE FROM users WHERE id IN (:oa, :ea, :va, :ub, :adm)"), {"oa": owner_a.id, "ea": editor_a.id, "va": viewer_a.id, "ub": user_b.id, "adm": admin.id})
            await session.commit()
        except Exception:
            await session.rollback()


# ============================================================================
# TEST SUITE: Milestone 7A.1
# ============================================================================

@pytest.mark.asyncio
async def test_01_first_edit_and_version_increment(setup_curation_context):
    """Verifies first edit increments chunk from v1 to v2 and captures initial v1 snapshot."""
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        new_text = "Updated clause: The liability of the vendor is capped at $150,000."
        edit_req = ChunkEditRequest(
            new_text=new_text,
            base_version=1,
            edit_reason="Legal team agreed on increased liability cap",
        )

        updated_chunk, version_rec = await service.edit_chunk(doc.id, chunk.id, edit_req, editor)

        assert updated_chunk.current_version == 2
        assert updated_chunk.text == new_text
        assert updated_chunk.chunk_hash == hashlib.sha256(new_text.encode()).hexdigest()
        assert version_rec.version_number == 2
        assert version_rec.text_before == ctx["init_text"]
        assert version_rec.text_after == new_text
        assert version_rec.edited_by == editor.id
        assert version_rec.is_rollback is False

        # Verify historical versions in DB: both v1 and v2 exist
        versions = await service.list_chunk_versions(doc.id, chunk.id, editor)
        assert len(versions) == 2
        assert versions[0].version_number == 1
        assert versions[0].text_after == ctx["init_text"]
        assert versions[1].version_number == 2
        assert versions[1].text_after == new_text


@pytest.mark.asyncio
async def test_02_embedding_generation_failure_preserves_atomic_state(setup_curation_context):
    """
    INVARIANT 2: Failed embedding generation must abort before DB changes.
    Chunk text, version, and embedding must remain unmodified.
    """
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        failing_provider = MockEmbeddingProvider(fail_on_embed=True)
        service = ChunkCurationService(session, embedding_provider=failing_provider)

        edit_req = ChunkEditRequest(
            new_text="This text should never be persisted because embedding generation fails.",
            base_version=1,
            edit_reason="Failed attempt",
        )

        with pytest.raises(HTTPException) as exc_info:
            await service.edit_chunk(doc.id, chunk.id, edit_req, editor)

        assert exc_info.value.status_code == 500
        assert "Embedding provider failure" in exc_info.value.detail

        # Verify database chunk is untouched
        db_chunk = await session.get(DocumentChunk, chunk.id)
        assert db_chunk.current_version == 1
        assert db_chunk.text == ctx["init_text"]
        assert db_chunk.chunk_hash == ctx["init_hash"]

        # Verify no version rows created
        db_versions = (await session.scalars(
            select(DocumentChunkVersion).where(DocumentChunkVersion.chunk_id == chunk.id)
        )).all()
        assert len(db_versions) == 0


@pytest.mark.asyncio
async def test_03_concurrent_edit_conflict_optimistic_locking(setup_curation_context):
    """
    INVARIANT 4: Stale base_version returns HTTP 409 Conflict.
    Active chunk and history remain unchanged.
    """
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    owner = ctx["owner_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        # Editor advances version to 2
        await service.edit_chunk(
            doc.id, chunk.id,
            ChunkEditRequest(new_text="Version 2 content", base_version=1),
            editor,
        )

        # Owner tries to commit an edit assuming base_version=1 (concurrent stale write)
        with pytest.raises(HTTPException) as exc_info:
            await service.edit_chunk(
                doc.id, chunk.id,
                ChunkEditRequest(new_text="Owner conflicting content", base_version=1),
                owner,
            )

        assert exc_info.value.status_code == 409
        assert "Conflict" in exc_info.value.detail
        assert "current version is 2" in exc_info.value.detail

        # Active chunk remains at version 2
        db_chunk = await session.get(DocumentChunk, chunk.id)
        assert db_chunk.current_version == 2
        assert db_chunk.text == "Version 2 content"


@pytest.mark.asyncio
async def test_04_accurate_diff_between_versions(setup_curation_context):
    """Verifies diff API accurately generates unified diff and line metrics between versions."""
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        v2_text = "Original clause:\nThe liability of the vendor is capped at $200,000.\nNew indemnity clause added."
        await service.edit_chunk(doc.id, chunk.id, ChunkEditRequest(new_text=v2_text, base_version=1), editor)

        diff = await service.get_chunk_diff(doc.id, chunk.id, v1=1, v2=2, user=editor)

        assert diff.v1 == 1
        assert diff.v2 == 2
        assert diff.lines_added > 0
        assert diff.lines_removed > 0
        assert diff.characters_changed > 0
        assert "--- version_1" in diff.diff_patch
        assert "+++ version_2" in diff.diff_patch
        assert "+New indemnity clause added." in diff.diff_patch


@pytest.mark.asyncio
async def test_05_rollback_and_forward_version_numbering(setup_curation_context):
    """
    INVARIANT 5: Rollback creates forward-moving version (v3) restoring v1 text,
    without erasing v1 or v2.
    """
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        # v1 -> v2
        await service.edit_chunk(
            doc.id, chunk.id,
            ChunkEditRequest(new_text="Accidental bad edit in v2", base_version=1),
            editor,
        )

        # Rollback to v1 -> creates v3
        rolled_chunk, rollback_v = await service.rollback_chunk(
            doc.id, chunk.id,
            ChunkRollbackRequest(target_version=1, rollback_reason="Reverting accidental edit"),
            editor,
        )

        assert rolled_chunk.current_version == 3
        assert rolled_chunk.text == ctx["init_text"]
        assert rolled_chunk.chunk_hash == ctx["init_hash"]
        assert rollback_v.version_number == 3
        assert rollback_v.is_rollback is True
        assert rollback_v.reverted_from_version == 1

        # Check full history: v1, v2, v3 all exist
        versions = await service.list_chunk_versions(doc.id, chunk.id, editor)
        assert [v.version_number for v in versions] == [1, 2, 3]
        assert versions[0].text_after == ctx["init_text"]
        assert versions[1].text_after == "Accidental bad edit in v2"
        assert versions[2].text_after == ctx["init_text"]
        assert versions[2].is_rollback is True


@pytest.mark.asyncio
async def test_06_rollback_incompatible_embedding_reembeds(setup_curation_context):
    """
    INVARIANT 6: If historical snapshot has different embedding model or dimension,
    rollback safely re-embeds the text with active provider.
    """
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        # v1 -> v2
        await service.edit_chunk(
            doc.id, chunk.id,
            ChunkEditRequest(new_text="Edit in v2", base_version=1),
            editor,
        )

        # Manually alter v1 snapshot metadata to simulate a legacy model migration
        v1_rec = await session.scalar(
            select(DocumentChunkVersion).where(
                and_(
                    DocumentChunkVersion.chunk_id == chunk.id,
                    DocumentChunkVersion.version_number == 1,
                )
            )
        )
        v1_rec.embedding_model = "legacy-deprecated-embedder-v0"
        await session.commit()

        initial_calls = mock_provider.call_count

        # Rollback to v1: must detect model mismatch and call embed_texts
        rolled_chunk, rollback_v = await service.rollback_chunk(
            doc.id, chunk.id,
            ChunkRollbackRequest(target_version=1, rollback_reason="Rollback with re-embedding"),
            editor,
        )

        assert mock_provider.call_count == initial_calls + 1, "Must re-embed when historical model is incompatible"
        assert rollback_v.embedding_model == settings.embedding_model
        assert rolled_chunk.current_version == 3


@pytest.mark.asyncio
async def test_07_viewer_role_write_rejection(setup_curation_context):
    """Verifies that VIEWER role can read versions/diff, but is strictly rejected with 403 on mutation."""
    ctx = setup_curation_context
    viewer = ctx["viewer_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        # Read detail: allowed
        detail = await service.get_chunk_detail(doc.id, chunk.id, viewer)
        assert detail.id == chunk.id

        # Read versions: allowed
        versions = await service.list_chunk_versions(doc.id, chunk.id, viewer)
        assert len(versions) >= 1

        # Edit: rejected with 403
        with pytest.raises(HTTPException) as exc_info:
            await service.edit_chunk(
                doc.id, chunk.id,
                ChunkEditRequest(new_text="Viewer unauthorized edit", base_version=1),
                viewer,
            )
        assert exc_info.value.status_code == 403
        assert "Viewer role is read-only" in exc_info.value.detail

        # Rollback: rejected with 403
        with pytest.raises(HTTPException) as exc_rollback:
            await service.rollback_chunk(
                doc.id, chunk.id,
                ChunkRollbackRequest(target_version=1),
                viewer,
            )
        assert exc_rollback.value.status_code == 403


@pytest.mark.asyncio
async def test_08_cross_workspace_access_denial(setup_curation_context):
    """Verifies multi-tenant isolation: User in Workspace B cannot read or edit Workspace A chunks."""
    ctx = setup_curation_context
    user_b = ctx["user_b"]
    doc_a = ctx["doc_a"]
    chunk_a = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        # Read cross-workspace: 403
        with pytest.raises(HTTPException) as exc_read:
            await service.get_chunk_detail(doc_a.id, chunk_a.id, user_b)
        assert exc_read.value.status_code == 403

        # Edit cross-workspace: 403
        with pytest.raises(HTTPException) as exc_edit:
            await service.edit_chunk(
                doc_a.id, chunk_a.id,
                ChunkEditRequest(new_text="Cross-workspace illegal write", base_version=1),
                user_b,
            )
        assert exc_edit.value.status_code == 403


@pytest.mark.asyncio
async def test_09_managed_crm_chunk_protection(setup_curation_context):
    """Verifies that synchronized CRM chunks cannot be directly curated (403 Forbidden)."""
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    crm_doc = ctx["crm_doc"]
    crm_chunk = ctx["crm_chunk"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        # Attempt edit on CRM chunk
        with pytest.raises(HTTPException) as exc_info:
            await service.edit_chunk(
                crm_doc.id, crm_chunk.id,
                ChunkEditRequest(new_text="Modified CRM company name", base_version=1),
                editor,
            )
        assert exc_info.value.status_code == 403
        assert "Cannot edit managed external CRM" in exc_info.value.detail


@pytest.mark.asyncio
async def test_10_hierarchical_stale_summary_flagging(setup_curation_context):
    """
    INVARIANT 8: Chunk curation preserves OV-1/OV-2 hierarchy and flags
    parent DocumentSection as stale.
    """
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]
    section = ctx["section_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        # Verify initial section state
        sec_before = await session.get(DocumentSection, section.id)
        assert sec_before.metadata_.get("is_stale") is False

        # Edit chunk
        await service.edit_chunk(
            doc.id, chunk.id,
            ChunkEditRequest(new_text="Updated section liability text to $300,000", base_version=1),
            editor,
        )

        # Verify section marked as stale
        sec_after = await session.get(DocumentSection, section.id)
        assert sec_after.metadata_.get("is_stale") is True
        assert "edited to version 2" in sec_after.metadata_.get("stale_reason", "")
        assert "stale_at" in sec_after.metadata_


@pytest.mark.asyncio
async def test_11_vector_retrieval_reflects_active_curated_state(setup_curation_context):
    """Verifies that pgvector cosine similarity search retrieves active curated text post-edit and post-rollback."""
    ctx = setup_curation_context
    editor = ctx["editor_a"]
    doc = ctx["doc_a"]
    chunk = ctx["chunk_a"]

    async with AsyncSessionLocal() as session:
        mock_provider = MockEmbeddingProvider()
        service = ChunkCurationService(session, embedding_provider=mock_provider)

        # 1. Edit chunk to include specific unique keyword
        curated_text = "The unique vendor warranty period is extended to ninety days."
        await service.edit_chunk(
            doc.id, chunk.id,
            ChunkEditRequest(new_text=curated_text, base_version=1),
            editor,
        )

        # Compute query vector for the curated text
        query_vec = (await mock_provider.embed_texts([curated_text]))[0]

        # Query active document_chunks using pgvector cosine similarity
        query_stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc.id)
            .order_by(DocumentChunk.embedding.cosine_distance(query_vec))
            .limit(1)
        )
        matched_chunk = await session.scalar(query_stmt)
        assert matched_chunk is not None
        assert matched_chunk.text == curated_text
        assert matched_chunk.current_version == 2

        # 2. Rollback to v1
        await service.rollback_chunk(
            doc.id, chunk.id,
            ChunkRollbackRequest(target_version=1, rollback_reason="Restoring original"),
            editor,
        )

        # Query vector for original text
        orig_vec = (await mock_provider.embed_texts([ctx["init_text"]]))[0]
        matched_post_rollback = await session.scalar(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc.id)
            .order_by(DocumentChunk.embedding.cosine_distance(orig_vec))
            .limit(1)
        )
        assert matched_post_rollback is not None
        assert matched_post_rollback.text == ctx["init_text"]
        assert matched_post_rollback.current_version == 3
