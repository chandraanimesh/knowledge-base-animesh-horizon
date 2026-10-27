"""
Unit and Integration Tests for CRM -> KB Bounded Batch and Incremental Synchronization.

Covers:
1. Deterministic source fingerprinting (changes when fields change, stable otherwise).
2. Bounded batch execution with durable CRMSyncJob and CRMSyncRecord tracking.
3. Partial failure tolerance: malformed record does not abort remaining records.
4. Incremental change detection: unchanged fingerprints are cleanly skipped.
5. Checkpoint advancement and safe retry of transient failures with backoff.
6. Fail-closed department authorization.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.crm.sync_service import (
    CRMSyncService,
    CRMSyncStatus,
    compute_crm_company_fingerprint,
)
from rag_app.crm.twenty_client import (
    CRMCompany,
    CRMCompanyState,
    CRMOpportunity,
    CRMPerson,
)
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import CRMSyncJob, CRMSyncRecord, Department, Document
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.schemas.rag import IngestResponse


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def sync_dept():
    async with AsyncSessionLocal() as session:
        from sqlalchemy import delete
        await session.execute(delete(CRMSyncRecord))
        await session.execute(delete(CRMSyncJob))
        await session.commit()

        dept = await session.scalar(
            select(Department).where(Department.code == "CRM_BATCH_TEST")
        )
        if not dept:
            dept = Department(
                id=uuid.uuid4(),
                name="CRM Batch Test Dept",
                code="CRM_BATCH_TEST",
                is_active=True,
            )
            session.add(dept)
            await session.commit()
            await session.refresh(dept)
        return dept


async def _make_real_document_id(session, dept_id: uuid.UUID) -> str:
    doc = Document(
        id=uuid.uuid4(),
        file_name=f"test_{uuid.uuid4().hex[:6]}.md",
        file_path="uploaded_docs/test.md",
        content_hash=uuid.uuid4().hex,
        mime_type="text/markdown",
        department_id=dept_id,
        status="COMPLETED",
    )
    session.add(doc)
    await session.commit()
    return str(doc.id)


def _make_company_state(cid: str, name: str, revenue: float = 100000.0) -> CRMCompanyState:
    company = CRMCompany(
        id=cid,
        name=name,
        domain=f"{name.lower().replace(' ', '')}.com",
        address="123 Test St, San Francisco, CA",
        annual_revenue=revenue,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
    )
    person = CRMPerson(
        id=f"person-{cid[:4]}",
        name=f"Contact for {name}",
        job_title="VP Sales",
        email=f"contact@{name.lower().replace(' ', '')}.com",
        phone="555-1234",
        company_id=cid,
    )
    opp = CRMOpportunity(
        id=f"opp-{cid[:4]}",
        name=f"{name} Enterprise Deal",
        amount=50000.0,
        currency="USD",
        stage="PROPOSAL",
        close_date="2026-06-30",
        company_id=cid,
    )
    return CRMCompanyState(
        company_name=name,
        company=company,
        people=[person],
        opportunities=[opp],
        is_connected=True,
    )


def test_compute_crm_company_fingerprint_determinism():
    state1 = _make_company_state("comp-1", "Stripe", 1000000.0)
    state2 = _make_company_state("comp-1", "Stripe", 1000000.0)
    fp1 = compute_crm_company_fingerprint(state1)
    fp2 = compute_crm_company_fingerprint(state2)
    assert fp1 == fp2
    assert len(fp1) == 64

    # Changing revenue changes fingerprint
    state_mod = _make_company_state("comp-1", "Stripe", 2000000.0)
    fp_mod = compute_crm_company_fingerprint(state_mod)
    assert fp1 != fp_mod


@pytest.mark.asyncio
async def test_crm_batch_sync_bounded_execution(sync_dept):
    async with AsyncSessionLocal() as session:
        cid1 = f"comp-101-{uuid.uuid4().hex[:6]}"
        cid2 = f"comp-102-{uuid.uuid4().hex[:6]}"
        doc_id1 = await _make_real_document_id(session, sync_dept.id)
        doc_id2 = await _make_real_document_id(session, sync_dept.id)

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_companies = [
            CRMCompany(id=cid1, name="Alpha Corp"),
            CRMCompany(id=cid2, name="Beta Corp"),
        ]
        mock_mcp.fetch_companies_page.return_value = (mock_companies, False, 2)
        mock_mcp.get_live_company_state_by_id.side_effect = [
            _make_company_state(cid1, "Alpha Corp"),
            _make_company_state(cid2, "Beta Corp"),
        ]

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.side_effect = [
            IngestResponse(file_name="twenty_company_comp101.md", skipped=False, document_id=doc_id1),
            IngestResponse(file_name="twenty_company_comp102.md", skipped=False, document_id=doc_id2),
        ]

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)
        summary = await service.sync_batch(department_id=sync_dept.id, limit=10, offset=0)

        assert summary.status == "COMPLETED"
        assert summary.total_records == 2
        assert summary.processed_records == 2
        assert summary.succeeded_records == 2
        assert summary.failed_records == 0
        assert summary.checkpoint_cursor == "2"

        # Verify job in DB
        job = await session.get(CRMSyncJob, summary.job_id)
        assert job is not None
        assert job.status == "COMPLETED"
        assert job.checkpoint_cursor == "2"

        # Verify records in DB
        recs = (await session.scalars(
            select(CRMSyncRecord).where(CRMSyncRecord.job_id == summary.job_id)
        )).all()
        assert len(recs) == 2
        for r in recs:
            assert r.status == CRMSyncStatus.SUCCEEDED.value
            assert r.source_fingerprint is not None


@pytest.mark.asyncio
async def test_crm_batch_sync_partial_failure_isolation(sync_dept):
    async with AsyncSessionLocal() as session:
        cid1 = f"comp-201-{uuid.uuid4().hex[:6]}"
        cid2 = f"comp-202-{uuid.uuid4().hex[:6]}"
        doc_id1 = await _make_real_document_id(session, sync_dept.id)

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_companies = [
            CRMCompany(id=cid1, name="Good Corp"),
            CRMCompany(id=cid2, name="Broken Corp"),
        ]
        mock_mcp.fetch_companies_page.return_value = (mock_companies, False, 2)
        mock_mcp.get_live_company_state_by_id.side_effect = [
            _make_company_state(cid1, "Good Corp"),
            _make_company_state(cid2, "Broken Corp"),
        ]

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.side_effect = [
            IngestResponse(file_name="twenty_company_comp201.md", skipped=False, document_id=doc_id1),
            RuntimeError("Simulated vector embedding failure for Broken Corp"),
        ]

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)
        summary = await service.sync_batch(department_id=sync_dept.id, limit=10)

        # Batch should report PARTIAL_SUCCESS and NOT abort on the first failure
        assert summary.status == "PARTIAL_SUCCESS"
        assert summary.total_records == 2
        assert summary.succeeded_records == 1
        assert summary.failed_records == 1
        assert len(summary.records) == 2


@pytest.mark.asyncio
async def test_crm_incremental_sync_skips_unchanged_fingerprint(sync_dept):
    async with AsyncSessionLocal() as session:
        cid = f"comp-301-{uuid.uuid4().hex[:6]}"
        state = _make_company_state(cid, "Delta Corp", 500000.0)
        doc_id = await _make_real_document_id(session, sync_dept.id)

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.return_value = IngestResponse(
            file_name="twenty_company_comp301.md", skipped=False, document_id=doc_id
        )

        service = CRMSyncService(session, ingestion_service=mock_ingestion)

        # First run: Succeeded
        r1 = await service.sync_company_state(state, department_id=sync_dept.id)
        assert getattr(r1, "skipped", False) is False
        assert mock_ingestion.ingest_file.call_count == 1

        # Second run with same state: Skipped without calling ingestion
        r2 = await service.sync_company_state(state, department_id=sync_dept.id)
        assert getattr(r2, "skipped", False) is True
        assert mock_ingestion.ingest_file.call_count == 1  # Not called again


@pytest.mark.asyncio
async def test_crm_retry_eligible_failed_records(sync_dept):
    async with AsyncSessionLocal() as session:
        cid = f"comp-401-{uuid.uuid4().hex[:6]}"
        rec = CRMSyncRecord(
            id=uuid.uuid4(),
            provider="twenty_crm",
            entity_type="company",
            external_record_id=cid,
            department_id=sync_dept.id,
            status=CRMSyncStatus.RETRYABLE_FAILED.value,
            attempt_count=1,
            max_attempts=3,
            next_retry_at=datetime.now(timezone.utc) - timedelta(minutes=5),  # Elapsed
        )
        session.add(rec)
        await session.commit()

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.get_live_company_state_by_id.return_value = _make_company_state(cid, "Recovered Corp")
        doc_id = await _make_real_document_id(session, sync_dept.id)

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.return_value = IngestResponse(
            file_name="twenty_company_comp401.md", skipped=False, document_id=doc_id
        )

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)
        results = await service.retry_eligible_failed_records(department_id=sync_dept.id)

        assert len(results) == 1
        assert results[0].status == CRMSyncStatus.SUCCEEDED.value


@pytest.mark.asyncio
async def test_crm_batch_sync_fails_closed_on_invalid_dept():
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())
        with pytest.raises(ValueError, match="not found in database"):
            await service.sync_batch(department_id="NON_EXISTENT_DEPT")
