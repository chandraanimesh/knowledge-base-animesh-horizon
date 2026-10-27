"""
Integration tests for CRM -> Knowledge Base Persistence and Offline Retrieval.

Covers:
1. Ingestion of CRM company state into Document and DocumentChunk rows with non-null
   embeddings and JSONB metadata_ containing CRM provenance (crm_record_id, source).
2. Content-hash deduplication: unchanged CRM records are cleanly skipped on subsequent sync.
3. Changed-record update: modifying CRM fields safely updates the document and re-indexes chunks.
4. Failure handling: sync fails closed if department is invalid or inactive.
5. Offline retrieval via RetrievalService: proves records are retrieved from the Knowledge Base
   while live CRM calls are genuinely isolated/prohibited.
6. Offline retrieval via MultiSourceEvidenceFusionService: proves Stream 3 retrieves persisted
   CRM document chunks even when live CRM (Stream 5) is offline.
"""

import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from sqlalchemy import delete, select

from rag_app.core.config import settings
from rag_app.crm.sync_service import CRMSyncService, normalize_crm_company_to_markdown
from rag_app.crm.twenty_client import (
    CRMCompany,
    CRMCompanyState,
    CRMOpportunity,
    CRMPerson,
)
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import CRMSyncRecord, Department, Document, DocumentChunk, User
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.graph.fusion import MultiSourceEvidenceFusionService
from rag_app.retrieval.service import RetrievalService


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def sync_test_setup():
    """Setup an active test department, admin user, and cleanup previous test docs."""
    async with AsyncSessionLocal() as session:
        # 1. Department
        dept = await session.scalar(
            select(Department).where(Department.code == "CRM_SYNC_TEST")
        )
        if not dept:
            dept = Department(
                code="CRM_SYNC_TEST",
                name="CRM Sync Test Department",
                is_active=True,
            )
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        # 2. Admin User
        admin_user = await session.scalar(
            select(User).where(User.email == "crm_sync_admin@test.com")
        )
        if not admin_user:
            admin_user = User(
                email="crm_sync_admin@test.com",
                hashed_password="hashed_test_password",
                role="ADMIN",
                is_active=True,
            )
            session.add(admin_user)
            await session.commit()
            await session.refresh(admin_user)

        # 3. Clean up any existing test documents & sync records for this company ID
        company_id_safe = "c776ee49"
        await session.execute(
            delete(CRMSyncRecord).where(
                CRMSyncRecord.external_record_id.like(f"{company_id_safe}%")
            )
        )
        old_docs = (
            await session.scalars(
                select(Document).where(
                    Document.file_name.like(f"twenty_company_{company_id_safe}%")
                )
            )
        ).all()
        for doc in old_docs:
            await session.delete(doc)
        await session.commit()

        yield dept, admin_user

        # Teardown: clean up test documents & sync records
        async with AsyncSessionLocal() as clean_session:
            await clean_session.execute(
                delete(CRMSyncRecord).where(
                    CRMSyncRecord.external_record_id.like(f"{company_id_safe}%")
                )
            )
            cleanup_docs = (
                await clean_session.scalars(
                    select(Document).where(
                        Document.file_name.like(f"twenty_company_{company_id_safe}%")
                    )
                )
            ).all()
            for doc in cleanup_docs:
                await clean_session.delete(doc)
            await clean_session.commit()


def make_airbnb_crm_state(revenue: float = 50000.0, address: str = "888 Brannan St, San Francisco, CA") -> CRMCompanyState:
    """Creates a deterministic CRMCompanyState adhering to the verified schema."""
    return CRMCompanyState(
        company_name="Airbnb",
        is_connected=True,
        company=CRMCompany(
            id="c776ee49-f608-4a77-8cc8-6fe96ae1e43f",
            name="Airbnb",
            domain="airbnb.com",
            address=address,
            annual_revenue=revenue,
            created_at="2026-03-10T16:26:00.000Z",
            updated_at="2026-03-10T16:26:00.000Z",
        ),
        people=[
            CRMPerson(
                id="p-airbnb-1",
                name="Brian Chesky",
                job_title="CEO",
                email="brian@airbnb.com",
                phone="+1-555-0199",
                company_id="c776ee49-f608-4a77-8cc8-6fe96ae1e43f",
            )
        ],
        opportunities=[
            CRMOpportunity(
                id="o-airbnb-1",
                name="Global Host Community Expansion",
                amount=250000.0,
                stage="NEGOTIATION",
                close_date="2026-12-31T00:00:00.000Z",
                company_id="c776ee49-f608-4a77-8cc8-6fe96ae1e43f",
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Normalization & Content-Hash Stability Test
# ---------------------------------------------------------------------------

def test_normalization_keeps_sync_timestamp_out_of_markdown():
    """Verify that sync_timestamp is isolated to doc_metadata to allow content-hash deduplication."""
    crm_state = make_airbnb_crm_state()
    markdown1, meta1 = normalize_crm_company_to_markdown(crm_state, verification_marker="MARKER-1")
    markdown2, meta2 = normalize_crm_company_to_markdown(crm_state, verification_marker="MARKER-1")

    # Markdown body must be byte-for-byte identical across runs for unchanged records
    assert markdown1 == markdown2
    assert "sync_timestamp" not in markdown1
    assert "sync_timestamp" in meta1
    assert meta1["crm_record_id"] == "c776ee49-f608-4a77-8cc8-6fe96ae1e43f"
    assert meta1["source"] == "twenty_crm_mcp"


# ---------------------------------------------------------------------------
# 2. Ingestion & Metadata Persistence Integration Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crm_sync_ingestion_and_metadata_persistence(sync_test_setup):
    """Verify one-record ingestion produces valid Document, DocumentChunks, embeddings, and metadata_."""
    dept, admin_user = sync_test_setup
    crm_state = make_airbnb_crm_state()
    verification_marker = "VERIFY-AIRBNB-MCP-SYNC-001"

    async with AsyncSessionLocal() as session:
        sync_service = CRMSyncService(session=session)
        resp = await sync_service.sync_company_state(
            crm_state=crm_state,
            department_id=dept.id,
            created_by=admin_user.id,
            verification_marker=verification_marker,
        )

        assert resp.skipped is False
        assert resp.chunks_created > 0
        assert resp.document_id is not None

        # Verify Document row
        doc = await session.get(Document, uuid.UUID(resp.document_id))
        assert doc is not None
        assert doc.department_id == dept.id
        assert doc.created_by == admin_user.id
        assert doc.content_hash is not None
        assert "twenty_company_c776ee49" in doc.file_name

        # Verify DocumentChunk rows
        chunks = (
            await session.scalars(
                select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
            )
        ).all()
        assert len(chunks) == resp.chunks_created
        for ch in chunks:
            # Vector embedding exists and matches dimension
            assert ch.embedding is not None
            assert len(ch.embedding) == 384
            # Metadata preserves CRM provenance
            assert ch.metadata_ is not None
            assert ch.metadata_.get("source") == "twenty_crm_mcp"
            assert ch.metadata_.get("crm_record_id") == "c776ee49-f608-4a77-8cc8-6fe96ae1e43f"
            assert ch.metadata_.get("crm_object") == "company"
            assert ch.metadata_.get("company_name") == "Airbnb"
            assert ch.metadata_.get("verification_marker") == verification_marker


# ---------------------------------------------------------------------------
# 3. Unchanged-Record Deduplication Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crm_sync_unchanged_record_deduplication(sync_test_setup):
    """Verify that re-syncing an unchanged record is cleanly skipped by content-hash deduplication."""
    dept, admin_user = sync_test_setup
    crm_state = make_airbnb_crm_state()

    async with AsyncSessionLocal() as session:
        sync_service = CRMSyncService(session=session)

        # 1. First sync -> creates document
        resp1 = await sync_service.sync_company_state(
            crm_state=crm_state,
            department_id=dept.id,
            created_by=admin_user.id,
        )
        assert resp1.skipped is False
        doc1_id = resp1.document_id

        # 2. Second sync with identical data -> skipped due to identical content_hash
        resp2 = await sync_service.sync_company_state(
            crm_state=crm_state,
            department_id=dept.id,
            created_by=admin_user.id,
        )
        assert resp2.skipped is True
        assert resp2.chunks_created == 0
        assert resp2.document_id == doc1_id
        assert "content_hash" in (resp2.reason or "")

        # Verify no duplicate chunks were created
        chunk_count = len(
            (
                await session.scalars(
                    select(DocumentChunk).where(DocumentChunk.document_id == uuid.UUID(doc1_id))
                )
            ).all()
        )
        assert chunk_count == resp1.chunks_created


# ---------------------------------------------------------------------------
# 4. Changed-Record Safe Update Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crm_sync_changed_record_updates(sync_test_setup):
    """Verify that updated CRM fields change the content hash, safely replacing the document."""
    dept, admin_user = sync_test_setup

    async with AsyncSessionLocal() as session:
        sync_service = CRMSyncService(session=session)

        # 1. Initial sync with revenue $50,000
        state1 = make_airbnb_crm_state(revenue=50000.0)
        resp1 = await sync_service.sync_company_state(
            crm_state=state1,
            department_id=dept.id,
            created_by=admin_user.id,
        )
        assert resp1.skipped is False
        doc1_id = uuid.UUID(resp1.document_id)

        # 2. Updated sync with revenue $95,000
        state2 = make_airbnb_crm_state(revenue=95000.0)
        resp2 = await sync_service.sync_company_state(
            crm_state=state2,
            department_id=dept.id,
            created_by=admin_user.id,
        )
        assert resp2.skipped is False
        doc2_id = uuid.UUID(resp2.document_id)

        # Old document was replaced
        assert doc1_id != doc2_id

        # Verify new chunks reflect the updated revenue
        chunks2 = (
            await session.scalars(
                select(DocumentChunk).where(DocumentChunk.document_id == doc2_id)
            )
        ).all()
        assert any("$95,000.00" in ch.text for ch in chunks2)


# ---------------------------------------------------------------------------
# 5. Invalid Department Failure Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crm_sync_fails_on_invalid_department():
    """Verify that sync fails closed when an invalid or non-existent department is supplied."""
    crm_state = make_airbnb_crm_state()
    random_dept_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        sync_service = CRMSyncService(session=session)
        with pytest.raises(ValueError) as exc_info:
            await sync_service.sync_company_state(
                crm_state=crm_state,
                department_id=random_dept_id,
            )
        assert "not found in database" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 6. Offline Retrieval Through RetrievalService (Live CRM Isolated)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_offline_retrieval_via_retrieval_service(sync_test_setup):
    """
    PROVE OFFLINE RETRIEVAL:
    Retrieves the persisted company record from DocumentChunk via RetrievalService
    while live CRM enrichment is genuinely isolated / prohibited.
    """
    dept, admin_user = sync_test_setup
    marker = "OFFLINE-RETRIEVAL-MARKER-AIRBNB-77"
    crm_state = make_airbnb_crm_state(address="888 Brannan St Suite 400")

    async with AsyncSessionLocal() as session:
        # Ingest the company into KB
        sync_service = CRMSyncService(session=session)
        resp = await sync_service.sync_company_state(
            crm_state=crm_state,
            department_id=dept.id,
            created_by=admin_user.id,
            verification_marker=marker,
        )
        assert resp.skipped is False

        # Now genuinely isolate/prohibit live CRM calls
        with patch.object(settings, "twenty_mcp_enabled", False), \
             patch.object(TwentyMCPClient, "get_live_company_state", side_effect=RuntimeError("LIVE CRM BLOCKED")):

            retrieval_service = RetrievalService(session=session)
            retrieval_resp = await retrieval_service.retrieve(
                query="What is the address of Airbnb?",
                allowed_departments=[dept.id],
                top_k=5,
            )

            assert len(retrieval_resp.results) > 0
            # Confirm that the retrieved chunk is the persisted Knowledge Base CRM chunk
            found_chunk = next(
                (ch for ch in retrieval_resp.results if "twenty_company" in ch.file_name),
                None,
            )
            assert found_chunk is not None, "Persisted CRM chunk was not retrieved from Knowledge Base!"
            assert "888 Brannan St" in found_chunk.text or marker in found_chunk.text
            assert "twenty_company_c776ee49" in found_chunk.file_name


# ---------------------------------------------------------------------------
# 7. Offline Retrieval Through MultiSourceEvidenceFusionService (Stream 3 Proof)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_offline_retrieval_via_fusion_stream3(sync_test_setup):
    """
    PROVE STREAM 3 OFFLINE RETRIEVAL:
    Proves that MultiSourceEvidenceFusionService retrieves the persisted CRM document
    via Stream 3 (DocumentChunk) even when Stream 5 (Live Operational CRM) is completely offline.
    """
    dept, admin_user = sync_test_setup
    marker = "FUSION-STREAM3-OFFLINE-AIRBNB-99"
    crm_state = make_airbnb_crm_state(address="888 Brannan St Global HQ")

    async with AsyncSessionLocal() as session:
        sync_service = CRMSyncService(session=session)
        resp = await sync_service.sync_company_state(
            crm_state=crm_state,
            department_id=dept.id,
            created_by=admin_user.id,
            verification_marker=marker,
        )
        assert resp.skipped is False

        # Live CRM is completely blocked/offline
        mock_crm = AsyncMock()
        mock_crm.get_live_company_state.return_value = CRMCompanyState(
            company_name="Airbnb",
            is_connected=False,
            error_message="Twenty CRM MCP is completely offline for this test",
        )

        fusion = MultiSourceEvidenceFusionService(session=session, crm_client=mock_crm)
        fused = await fusion.fuse_company_evidence(
            user=admin_user,
            company_name="Airbnb",
            query="Tell me about Airbnb",
            allowed_departments=[dept.id],
        )

        # Live CRM is offline
        assert fused.crm_status_available is False
        assert "offline" in fused.crm_notice.lower() or "unavailable" in fused.crm_notice.lower()

        # BUT Stream 3 (Document Chunks) successfully retrieved the persisted Knowledge Base record!
        assert len(fused.document_chunks) > 0
        crm_chunks = [s for s in fused.document_chunks if "twenty_company" in s.provenance.get("file_name", "")]
        assert len(crm_chunks) > 0, "Persisted CRM document was not found in Stream 3!"
        assert any(marker in s.content for s in crm_chunks)
        assert any("888 Brannan St Global HQ" in s.content for s in crm_chunks)

        # Grounded synthesis includes the document evidence
        assert "Document Evidence" in fused.formatted_synthesis
        assert crm_chunks[0].title in fused.formatted_synthesis
