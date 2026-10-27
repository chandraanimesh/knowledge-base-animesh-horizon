"""
Unit and Integration Tests for Twenty CRM Webhooks and Scheduled Reconciliation.

Covers:
1. HMAC-SHA256 signature verification (rejects invalid/tampered requests).
2. Processing incoming record events (create/update) triggers idempotent sync.
3. Conservative deletion: delete event marks sync record as ARCHIVED without deleting KB document.
4. Scheduled reconciliation: detects drift between Twenty CRM and sync ledger, repairs drift.
"""

import hashlib
import hmac
import json
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.crm.sync_service import CRMSyncService, CRMSyncStatus
from rag_app.crm.twenty_client import CRMCompany, CRMCompanyState
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import CRMSyncJob, CRMSyncRecord, Department, Document
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.schemas.rag import IngestResponse


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def webhook_dept():
    async with AsyncSessionLocal() as session:
        from sqlalchemy import delete
        await session.execute(delete(CRMSyncRecord))
        await session.execute(delete(CRMSyncJob))
        await session.commit()

        dept = await session.scalar(
            select(Department).where(Department.code == "CRM_WEBHOOK_TEST")
        )
        if not dept:
            dept = Department(
                id=uuid.uuid4(),
                name="CRM Webhook Test Dept",
                code="CRM_WEBHOOK_TEST",
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


def _make_state(cid: str, name: str) -> CRMCompanyState:
    company = CRMCompany(
        id=cid,
        name=name,
        domain="example.com",
        address="123 Example St",
        annual_revenue=100000.0,
    )
    return CRMCompanyState(
        company_name=name,
        company=company,
        people=[],
        opportunities=[],
        is_connected=True,
    )


@pytest.mark.asyncio
async def test_webhook_hmac_signature_verification():
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())
        secret = "twenty-secret-key-12345"
        payload = {"event": "company.updated", "record": {"id": "c-100"}}
        raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        valid_sig = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

        # 1. Invalid signature raises ValueError
        with pytest.raises(ValueError, match="Invalid webhook HMAC-SHA256 signature"):
            await service.handle_webhook_event(
                payload=payload,
                signature_header="invalid_signature",
                secret=secret,
            )

        # 2. Valid signature passes verification
        res = await service.handle_webhook_event(
            payload=payload,
            signature_header=valid_sig,
            secret=secret,
        )
        assert res["status"] in ("acknowledged", "processed")


@pytest.mark.asyncio
async def test_webhook_company_update_triggers_sync(webhook_dept):
    async with AsyncSessionLocal() as session:
        cid = f"comp-hook-{uuid.uuid4().hex[:6]}"
        doc_id = await _make_real_document_id(session, webhook_dept.id)

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.get_live_company_state_by_id.return_value = _make_state(cid, "Hooked Company")

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.return_value = IngestResponse(
            file_name="twenty_company_comphook01.md", skipped=False, document_id=doc_id
        )

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)
        payload = {"event": "company.updated", "record": {"id": cid}}
        secret = "twenty-test-secret"
        raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        sig = hmac.new(secret.encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()

        res = await service.handle_webhook_event(
            payload=payload,
            signature_header=sig,
            secret=secret,
            department_id=webhook_dept.id,
        )

        assert res["status"] == "processed"
        assert res["record_id"] == cid
        assert mock_ingestion.ingest_file.call_count == 1


@pytest.mark.asyncio
async def test_webhook_conservative_deletion_archives_record(webhook_dept):
    async with AsyncSessionLocal() as session:
        cid = f"comp-del-{uuid.uuid4().hex[:6]}"
        # Pre-seed a synced record in the ledger
        rec = CRMSyncRecord(
            id=uuid.uuid4(),
            provider="twenty_crm",
            entity_type="company",
            external_record_id=cid,
            department_id=webhook_dept.id,
            status=CRMSyncStatus.SUCCEEDED.value,
        )
        session.add(rec)
        await session.commit()

        service = CRMSyncService(session, ingestion_service=AsyncMock())
        payload = {"event": "company.deleted", "record": {"id": cid}}
        secret = "twenty-test-secret"
        raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        sig = hmac.new(secret.encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()

        res = await service.handle_webhook_event(
            payload=payload,
            signature_header=sig,
            secret=secret,
        )
        assert res["status"] == "archived"
        assert res["record_id"] == cid

        # Verify status in database is ARCHIVED, not hard-deleted
        updated_rec = await session.get(CRMSyncRecord, rec.id)
        assert updated_rec is not None
        assert updated_rec.status == CRMSyncStatus.ARCHIVED.value


@pytest.mark.asyncio
async def test_scheduled_reconciliation_drift_detection(webhook_dept):
    async with AsyncSessionLocal() as session:
        cid = f"comp-recon-{uuid.uuid4().hex[:6]}"
        doc_id = await _make_real_document_id(session, webhook_dept.id)

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.return_value = (
            [CRMCompany(id=cid, name="Recon Corp")],
            False,
            1,
        )
        mock_mcp.get_live_company_state_by_id.return_value = _make_state(cid, "Recon Corp")

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.return_value = IngestResponse(
            file_name="twenty_company_comprecon01.md", skipped=False, document_id=doc_id
        )

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)
        recon_result = await service.reconcile_scheduled(department_id=webhook_dept.id)

        assert recon_result["examined_records"] == 1
        assert recon_result["drift_detected"] == 1
        assert recon_result["repaired_records"] == 1
        assert mock_ingestion.ingest_file.call_count == 1
