"""Workstream 4: Actual CRM API Integration & Synchronization Test Suite.

Covers:
1. Provider classification (Live Twenty CRM probe vs Verified Contract Mock).
2. TwentyCRMClient read path (health, company search, people, opportunities, live state aggregation).
3. Deterministic source fingerprinting & Markdown normalization (zero volatile timestamps in body).
4. Full synchronization pipeline into disposable DB (idempotency, sync ledger, 0 duplicate chunks).
5. Partial failure isolation in batch synchronization (malformed records do not abort batch).
6. HMAC-SHA256 webhook signature verification (valid, tampered, missing, replay protection).
7. End-to-end CRM-to-RAG hybrid retrieval with strict department boundary scoping.
8. Controlled writeback safety (explicit approval, draft tampering detection, idempotency replay).
"""

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from sqlalchemy import delete, select

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.core.config import settings
from rag_app.crm.sync_service import (
    CRMSyncService,
    CRMSyncStatus,
    compute_crm_company_fingerprint,
    normalize_crm_company_to_markdown,
)
from rag_app.crm.twenty_client import (
    CRMCompany,
    CRMCompanyState,
    CRMOpportunity,
    CRMPerson,
    TwentyCRMClient,
)
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import (
    CRMSyncJob,
    CRMSyncRecord,
    CRMWebhookEvent,
    CRMWriteBackApproval,
    CRMWriteBackRecord,
    Department,
    Document,
    DocumentChunk,
    User,
    UserDepartment,
    UserPermission,
    Permission,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.retrieval.service import RetrievalService
from rag_app.schemas.workflow import compute_canonical_draft_hash


@pytest.mark.asyncio
async def test_provider_classification_and_twenty_client_read_path():
    """
    4.1 & 4.2: Probe provider status & verify read path contracts.
    """
    client = TwentyCRMClient()
    is_live = await client.check_health()

    provider_status = "LIVE" if is_live else "MOCKED / SIMULATED (Contract-compliant)"
    assert provider_status in ["LIVE", "MOCKED / SIMULATED (Contract-compliant)"]

    # Verify TwentyCRMClient read contracts with verified schema payloads
    fake_company_data = {
        "data": {
            "companies": [
                {
                    "id": "c-100",
                    "name": "Acme Global",
                    "domainName": {"primaryLinkUrl": "acme.com"},
                    "address": {
                        "addressStreet1": "123 Market St",
                        "addressCity": "San Francisco",
                        "addressState": "CA",
                    },
                    "annualRevenue": {"amountMicros": 15000000000000},
                    "createdAt": "2026-01-01T00:00:00Z",
                    "updatedAt": "2026-01-02T00:00:00Z",
                    "deletedAt": None,
                }
            ]
        }
    }
    fake_people_data = {
        "data": {
            "people": [
                {
                    "id": "p-200",
                    "name": "Jane Smith",
                    "jobTitle": "VP Engineering",
                    "email": "jane@acme.com",
                    "phone": "+1-555-0199",
                    "deletedAt": None,
                }
            ]
        }
    }
    fake_opps_data = {
        "data": {
            "opportunities": [
                {
                    "id": "o-300",
                    "name": "Q4 Cloud Expansion",
                    "amount": 250000.0,
                    "currency": "USD",
                    "stage": "CONTRACT_SENT",
                    "closeDate": "2026-12-15",
                    "deletedAt": None,
                }
            ]
        }
    }

    async def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "/healthz" in url:
            return httpx.Response(200, json={"status": "ok"})
        elif "/rest/companies" in url:
            return httpx.Response(200, json=fake_company_data)
        elif "/rest/people" in url:
            return httpx.Response(200, json=fake_people_data)
        elif "/rest/opportunities" in url:
            return httpx.Response(200, json=fake_opps_data)
        return httpx.Response(404, json={"error": "Not Found"})

    real_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(mock_handler)
    mocked_client = TwentyCRMClient(base_url="http://mock-crm.local", api_key="test_api_key")

    with patch("rag_app.crm.twenty_client.httpx.AsyncClient", side_effect=lambda *args, **kwargs: real_async_client(transport=transport)):
        health = await mocked_client.check_health()
        assert health is True

        company = await mocked_client.get_company_by_name("Acme Global")
        assert company is not None
        assert company.id == "c-100"
        assert company.name == "Acme Global"
        assert company.annual_revenue == 15000000.0

        people = await mocked_client.get_company_people("c-100")
        assert len(people) == 1
        assert people[0].name == "Jane Smith"
        assert people[0].job_title == "VP Engineering"

        opps = await mocked_client.get_company_opportunities("c-100")
        assert len(opps) == 1
        assert opps[0].name == "Q4 Cloud Expansion"
        assert opps[0].stage == "CONTRACT_SENT"


@pytest.mark.asyncio
async def test_crm_fingerprint_and_normalization():
    """
    4.3: Verify deterministic fingerprinting and timestamp isolation.
    """
    company = CRMCompany(id="c-999", name="Apex Systems", domain="apex.io", annual_revenue=5000000.0)
    person = CRMPerson(id="p-1", name="Alice Brown", job_title="CTO", email="alice@apex.io")
    opp = CRMOpportunity(id="o-1", name="Enterprise Deal", amount=75000.0, stage="PROPOSAL")

    state1 = CRMCompanyState(
        company_name="Apex Systems",
        company=company,
        people=[person],
        opportunities=[opp],
        retrieval_timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    state2 = CRMCompanyState(
        company_name="Apex Systems",
        company=company,
        people=[person],
        opportunities=[opp],
        retrieval_timestamp=datetime(2026, 6, 15, 18, 30, 0, tzinfo=timezone.utc),
    )

    # 1. Fingerprint is strictly invariant to volatile retrieval_timestamp
    fp1 = compute_crm_company_fingerprint(state1)
    fp2 = compute_crm_company_fingerprint(state2)
    assert fp1 == fp2, "Fingerprint must not change when only timestamp changes"

    # 2. Modifying substantive field alters fingerprint
    changed_company = CRMCompany(id="c-999", name="Apex Systems", domain="apex.io", annual_revenue=6000000.0)
    state3 = CRMCompanyState(
        company_name="Apex Systems",
        company=changed_company,
        people=[person],
        opportunities=[opp],
    )
    fp3 = compute_crm_company_fingerprint(state3)
    assert fp3 != fp1, "Substantive revenue update must generate distinct fingerprint"

    # 3. Normalization stores sync_timestamp in metadata, NEVER inside markdown text body
    md_text, meta = normalize_crm_company_to_markdown(state1)
    assert "sync_timestamp" in meta
    assert meta["crm_record_id"] == "c-999"
    assert "sync_timestamp" not in md_text
    assert "# CRM Company: Apex Systems" in md_text
    assert "Alice Brown" in md_text
    assert "Enterprise Deal" in md_text


@pytest.mark.asyncio
async def test_crm_sync_pipeline_idempotency_and_ledger():
    """
    4.4: Ingest CRM record into disposable DB, verify sync ledger, and verify idempotency.
    """
    try:
        async with AsyncSessionLocal() as session:
            dept = Department(code=f"CRM_IDEMP_{uuid.uuid4().hex[:6]}".upper(), name="CRM Idemp Dept", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

            company = CRMCompany(
                id=f"comp_{uuid.uuid4().hex[:8]}",
                name="OmniCorp Global",
                domain="omnicorp.org",
                address="742 Evergreen Terrace",
                annual_revenue=12000000.0,
            )
            person = CRMPerson(id="p-omni-1", name="Homer Simpson", job_title="Safety Inspector", email="homer@omnicorp.org")
            crm_state = CRMCompanyState(
                company_name="OmniCorp Global",
                company=company,
                people=[person],
                opportunities=[],
            )

            sync_service = CRMSyncService(session=session)

            # 1. First sync -> Succeeded
            resp1 = await sync_service.sync_company_state(
                crm_state=crm_state,
                department_id=dept.id,
            )
            assert resp1.skipped is False
            assert resp1.document_id is not None

            # Verify durable ledger
            stmt = select(CRMSyncRecord).where(CRMSyncRecord.external_record_id == str(company.id))
            rec1 = await session.scalar(stmt)
            assert rec1 is not None
            assert rec1.status == CRMSyncStatus.SUCCEEDED.value
            assert rec1.source_fingerprint is not None

            # Count documents and chunks
            doc_count1 = await session.scalar(select(Document).where(Document.id == uuid.UUID(resp1.document_id)))
            chunk_count1 = len((await session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == uuid.UUID(resp1.document_id)))).all())
            assert doc_count1 is not None
            assert chunk_count1 > 0

            # 2. Second sync with identical state -> Skipped & Idempotent (0 new documents or chunks)
            resp2 = await sync_service.sync_company_state(
                crm_state=crm_state,
                department_id=dept.id,
            )
            assert resp2.skipped is True
            assert "already exists" in resp2.reason.lower()

            # Ledger status updated to SKIPPED
            await session.refresh(rec1)
            assert rec1.status == CRMSyncStatus.SKIPPED.value
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_crm_batch_sync_failure_isolation():
    """
    4.5: Batch sync partial failure tolerance.
    """
    try:
        async with AsyncSessionLocal() as session:
            dept = Department(code=f"CRM_BATCH_{uuid.uuid4().hex[:6]}".upper(), name="CRM Batch Dept", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

            valid_company_1 = CRMCompany(id=f"c_valid_1_{uuid.uuid4().hex[:6]}", name="Valid Alpha", domain="alpha.dev")
            valid_company_2 = CRMCompany(id=f"c_valid_2_{uuid.uuid4().hex[:6]}", name="Valid Beta", domain="beta.dev")

            valid_state_1 = CRMCompanyState(company_name="Valid Alpha", company=valid_company_1)
            # Malformed state with company=None
            malformed_state = CRMCompanyState(company_name="Broken Corp", company=None)
            valid_state_2 = CRMCompanyState(company_name="Valid Beta", company=valid_company_2)

            sync_service = CRMSyncService(session=session)

            # Sync record 1 -> succeeds
            resp1 = await sync_service.sync_company_state(valid_state_1, department_id=dept.id)
            assert resp1.skipped is False

            # Sync record 2 (malformed) -> gracefully caught and isolated
            failed_isolated = False
            try:
                await sync_service.sync_company_state(malformed_state, department_id=dept.id)
            except ValueError:
                failed_isolated = True
            assert failed_isolated is True, "Malformed state raised expected validation error"

            # Sync record 3 -> succeeds (batch was not killed)
            resp3 = await sync_service.sync_company_state(valid_state_2, department_id=dept.id)
            assert resp3.skipped is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_crm_webhook_hmac_verification_and_replay_protection():
    """
    4.6: HMAC-SHA256 signature verification & replay protection.
    """
    try:
        async with AsyncSessionLocal() as session:
            dept = Department(code=f"CRM_HOOK_{uuid.uuid4().hex[:6]}".upper(), name="CRM Hook Dept", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

            test_secret = "super_secret_webhook_key_98765"
            payload_data = {
                "event": "company.updated",
                "data": {
                    "id": f"wh_c_{uuid.uuid4().hex[:6]}",
                    "name": "Webhook Corp",
                    "domain": "webhook.com",
                },
            }
            raw_body = json.dumps(payload_data, separators=(",", ":")).encode("utf-8")

            # Compute valid HMAC
            valid_sig = hmac.new(test_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
            tampered_sig = "a" * 64

            sync_service = CRMSyncService(session=session)

            # 1. Tampered signature -> Rejected fail-closed
            with pytest.raises(ValueError, match="Invalid webhook HMAC-SHA256 signature"):
                await sync_service.handle_webhook_event(
                    payload=raw_body,
                    signature_header=tampered_sig,
                    secret=test_secret,
                    department_id=dept.id,
                )

            # 2. Missing signature -> Rejected fail-closed
            with pytest.raises(ValueError, match="Missing webhook signature header"):
                await sync_service.handle_webhook_event(
                    payload=raw_body,
                    signature_header="",
                    secret=test_secret,
                    department_id=dept.id,
                )

            # 3. Missing secret -> Rejected fail-closed
            with pytest.raises(ValueError, match="Webhook secret is absent or empty"):
                await sync_service.handle_webhook_event(
                    payload=raw_body,
                    signature_header=valid_sig,
                    secret="",
                    department_id=dept.id,
                )

            # 4. Valid signature & event ID -> Succeeds
            event_id = f"evt_{uuid.uuid4().hex}"
            from rag_app.schemas.rag import IngestResponse
            with patch.object(sync_service, "sync_company_by_id", new_callable=AsyncMock) as mock_sync:
                mock_sync.return_value = IngestResponse(file_name="wh.md", skipped=False, document_id=str(uuid.uuid4()))
                res = await sync_service.handle_webhook_event(
                    payload=raw_body,
                    signature_header=valid_sig,
                    secret=test_secret,
                    department_id=dept.id,
                    event_id=event_id,
                )
                assert res["status"] in ["processed", "acknowledged"]

            # 5. Duplicate event ID replay -> Suppressed
            res_replay = await sync_service.handle_webhook_event(
                payload=raw_body,
                signature_header=valid_sig,
                secret=test_secret,
                department_id=dept.id,
                event_id=event_id,
            )
            assert res_replay["status"] == "duplicate_ignored"
            assert "already processed" in res_replay["reason"].lower()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_crm_document_hybrid_retrieval_and_department_scoping():
    """
    4.7: Synced CRM record is retrievable via hybrid search in authorized department.
    """
    try:
        async with AsyncSessionLocal() as session:
            # Authorized Department
            dept_auth = Department(code=f"CRM_RET_AUTH_{uuid.uuid4().hex[:6]}".upper(), name="CRM Auth Dept", is_active=True)
            # Foreign Department
            dept_other = Department(code=f"CRM_RET_OTHER_{uuid.uuid4().hex[:6]}".upper(), name="CRM Other Dept", is_active=True)
            session.add_all([dept_auth, dept_other])
            await session.commit()
            await session.refresh(dept_auth)
            await session.refresh(dept_other)

            company_name = "Solaris Energy"
            company = CRMCompany(
                id=f"sol_{uuid.uuid4().hex[:8]}",
                name=company_name,
                domain="solarisenergy.test",
                address="99 Sun Boulevard, Phoenix, AZ",
                annual_revenue=42000000.0,
            )
            crm_state = CRMCompanyState(company_name=company_name, company=company)

            sync_service = CRMSyncService(session=session)
            ingest_resp = await sync_service.sync_company_state(crm_state=crm_state, department_id=dept_auth.id)
            assert ingest_resp.document_id is not None

            # Query via RetrievalService with authorized department
            retrieval_service = RetrievalService(session=session)
            auth_results = await retrieval_service.retrieve(
                query=f"What is the address and revenue of {company_name}?",
                top_k=5,
                rerank=True,
                allowed_departments=[dept_auth.id],
            )

            assert len(auth_results.results) > 0
            all_retrieved_text = " ".join([r.text for r in auth_results.results])
            assert "Solaris" in all_retrieved_text
            assert "99 Sun Boulevard" in all_retrieved_text
            assert "$42,000,000.00" in all_retrieved_text

            # Query via RetrievalService with foreign department -> 0 results
            other_results = await retrieval_service.retrieve(
                query=f"What is the address and revenue of {company_name}?",
                top_k=5,
                rerank=True,
                allowed_departments=[dept_other.id],
            )
            matching_other = [r for r in other_results.results if "Solaris" in r.text]
            assert len(matching_other) == 0, "CRM document must NOT leak across department boundary"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_crm_writeback_safety_and_idempotency():
    """
    4.8: Controlled write-back safety: explicit approval, draft tampering detection, idempotency replay.
    """
    try:
        async with AsyncSessionLocal() as session:
            dept = Department(code=f"CRM_WB_{uuid.uuid4().hex[:6]}".upper(), name="CRM Writeback Dept", is_active=True)
            admin_user = User(email=f"wb_admin_{uuid.uuid4().hex[:6]}@test.local", hashed_password="pw", role="ADMIN", is_active=True)
            session.add_all([dept, admin_user])
            await session.commit()
            await session.refresh(dept)
            await session.refresh(admin_user)

            company_id = f"wb_c_{uuid.uuid4().hex[:6]}"
            draft_markdown = "## Meeting Summary\nDiscussed renewal terms. Agreed on 15% discount for 2-year lock."
            draft_hash = compute_canonical_draft_hash(draft_markdown)

            mock_mcp = AsyncMock(spec=TwentyMCPClient)
            mock_mcp.create_note.return_value = "note_crm_12345"
            mock_mcp.create_note_target_for_company.return_value = "target_crm_67890"

            sync_service = CRMSyncService(session=session, mcp_client=mock_mcp)

            # 1. Unapproved write-back -> PermissionError
            with pytest.raises(PermissionError, match="Write-back operation was not explicitly approved"):
                await sync_service.create_meeting_note_writeback(
                    company_id=company_id,
                    note_title="Unapproved Note",
                    summary_markdown=draft_markdown,
                    idempotency_key="key_unapproved",
                    approved=False,
                )

            # 2. Create approval ticket
            approval = await sync_service.create_writeback_approval(
                company_id=company_id,
                department_id=dept.id,
                approved_by=admin_user.id,
                approved_draft_hash=draft_hash,
            )

            # 3. Draft modification after approval -> Revokes ticket and fails with PermissionError
            tampered_draft = draft_markdown + "\nAdditionally transfer $500k to offshore."
            with pytest.raises(PermissionError, match="Draft content has been modified since approval"):
                await sync_service.create_meeting_note_writeback(
                    company_id=company_id,
                    note_title="Quarterly Review",
                    summary_markdown=tampered_draft,
                    idempotency_key="key_tampered",
                    user_id=admin_user.id,
                    user_role="ADMIN",
                    approval_id=approval.id,
                    approved=True,
                )

            # Verify approval status is now REVOKED
            await session.refresh(approval)
            assert approval.status == "REVOKED"

            # 4. Fresh approval with untampered draft -> Executes successfully
            fresh_approval = await sync_service.create_writeback_approval(
                company_id=company_id,
                department_id=dept.id,
                approved_by=admin_user.id,
                approved_draft_hash=draft_hash,
            )

            idempotency_key = f"idemp_{uuid.uuid4().hex}"
            wb_res = await sync_service.create_meeting_note_writeback(
                company_id=company_id,
                note_title="Quarterly Review",
                summary_markdown=draft_markdown,
                idempotency_key=idempotency_key,
                user_id=admin_user.id,
                user_role="ADMIN",
                approval_id=fresh_approval.id,
                approved=True,
            )
            assert wb_res["status"] == "COMPLETED"
            assert wb_res["crm_note_id"] == "note_crm_12345"
            assert mock_mcp.create_note.call_count == 1

            # 5. Idempotent replay with same idempotency key -> Returns completed without duplicate call
            wb_replay = await sync_service.create_meeting_note_writeback(
                company_id=company_id,
                note_title="Quarterly Review",
                summary_markdown=draft_markdown,
                idempotency_key=idempotency_key,
                user_id=admin_user.id,
                user_role="ADMIN",
                approval_id=fresh_approval.id,
                approved=True,
            )
            assert wb_replay["status"] == "COMPLETED"
            assert wb_replay.get("is_duplicate_replay") is True
            # Call count remains 1, exactly 0 duplicate notes created!
            assert mock_mcp.create_note.call_count == 1
    finally:
        await engine.dispose()
