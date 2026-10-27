"""
Remediation and Verification Tests for Production Readiness Audit Findings.

Covers:
- SEC-01 & SEC-04: Webhook HMAC signature verification (fail-closed, raw bytes, constant-time, formatting, tampered payload, replay protection)
- SEC-05: Webhook payload size enforcement (413 Payload Too Large)
- SEC-02 & SEC-03: Server-side write-back approval and department authorization
- REL-01: Two-phase Note creation and safe retry recovery without duplicating Notes
- REL-02: Concurrency-safe _get_or_create_sync_record with PostgreSQL savepoints
"""

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.crm.sync_service import CRMSyncService, CRMSyncStatus
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import (
    CRMSyncRecord,
    CRMWebhookEvent,
    CRMWriteBackApproval,
    CRMWriteBackRecord,
    Department,
    User,
    UserDepartment,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.main import app


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_dept_and_users():
    async with AsyncSessionLocal() as session:
        dept = await session.scalar(select(Department).where(Department.code == "AUDIT_DEPT"))
        if not dept:
            dept = Department(code="AUDIT_DEPT", name="Audit Department", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        admin = await session.scalar(select(User).where(User.email == "audit_admin@test.com"))
        if not admin:
            admin = User(email="audit_admin@test.com", hashed_password="pwd", role="ADMIN", is_active=True)
            session.add(admin)
            await session.commit()
            await session.refresh(admin)

        user = await session.scalar(select(User).where(User.email == "audit_user@test.com"))
        if not user:
            user = User(email="audit_user@test.com", hashed_password="pwd", role="USER", is_active=True)
            session.add(user)
            await session.commit()
            await session.refresh(user)

        # Assign user to department
        user_dept = await session.scalar(
            select(UserDepartment).where(
                UserDepartment.user_id == user.id,
                UserDepartment.department_id == dept.id,
            )
        )
        if not user_dept:
            user_dept = UserDepartment(user_id=user.id, department_id=dept.id)
            session.add(user_dept)
            await session.commit()

        yield dept, admin, user


# ============================================================================
# SEC-01 & SEC-04: Webhook Signature Verification Tests
# ============================================================================

@pytest.mark.asyncio
async def test_remediation_webhook_fail_closed_without_secret():
    """SEC-01: Absent or empty secret must fail closed and reject webhook requests."""
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())
        payload = b'{"event":"company.deleted","record":{"id":"comp-1"}}'

        with pytest.raises(ValueError, match="Webhook secret is absent or empty"):
            await service.handle_webhook_event(
                payload=payload,
                signature_header="some_sig",
                secret=None,
            )


@pytest.mark.asyncio
async def test_remediation_webhook_signature_verification_and_tampering():
    """SEC-01 & SEC-04: Constant-time HMAC verification against raw bytes rejects invalid/tampered signatures."""
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())
        secret = "super-secret-twenty-key"
        raw_body = b'{"event": "company.deleted", "record": {"id": "comp-raw-10"}}'
        valid_sig = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

        # 1. Missing signature rejected
        with pytest.raises(ValueError, match="Missing webhook signature header"):
            await service.handle_webhook_event(
                payload=raw_body,
                signature_header=None,
                secret=secret,
            )

        # 2. Forged signature rejected
        with pytest.raises(ValueError, match="Invalid webhook HMAC-SHA256 signature"):
            await service.handle_webhook_event(
                payload=raw_body,
                signature_header="forged_signature_hex",
                secret=secret,
            )

        # 3. Tampered payload rejected (even by 1 byte)
        tampered_body = b'{"event": "company.deleted", "record": {"id": "comp-raw-11"}}'
        with pytest.raises(ValueError, match="Invalid webhook HMAC-SHA256 signature"):
            await service.handle_webhook_event(
                payload=tampered_body,
                signature_header=valid_sig,
                secret=secret,
            )

        # 4. Valid raw bytes with exact formatting passes
        res = await service.handle_webhook_event(
            payload=raw_body,
            signature_header=f"sha256={valid_sig}",
            secret=secret,
        )
        assert res["status"] in ("archived", "ignored")


@pytest.mark.asyncio
async def test_remediation_webhook_replay_protection_deduplication():
    """SEC-01: Replay protection detects duplicate event deliveries and skips re-execution."""
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())
        secret = "replay-test-secret"
        event_id = f"evt-unique-{uuid.uuid4().hex[:8]}"
        payload_dict = {
            "id": event_id,
            "event": "company.deleted",
            "record": {"id": "comp-replay-100"},
        }
        raw_bytes = json.dumps(payload_dict).encode("utf-8")
        sig = hmac.new(secret.encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()

        # First delivery: processed/recorded
        res1 = await service.handle_webhook_event(
            payload=raw_bytes,
            signature_header=sig,
            secret=secret,
            event_id=event_id,
        )
        assert res1["status"] in ("archived", "ignored")

        # Second delivery with identical event_id: duplicate ignored
        res2 = await service.handle_webhook_event(
            payload=raw_bytes,
            signature_header=sig,
            secret=secret,
            event_id=event_id,
        )
        assert res2["status"] == "duplicate_ignored"
        assert res2["event_id"] == event_id


# ============================================================================
# SEC-02 & SEC-03: Server-Side Approval and Authorization Tests
# ============================================================================

@pytest.mark.asyncio
async def test_remediation_writeback_requires_server_side_approval(setup_dept_and_users):
    """SEC-02: Regular user cannot authorize themselves by submitting approved=True without approval record."""
    dept, admin, user = setup_dept_and_users
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())

        # Non-admin user with approved=True but no approval_id must be strictly rejected
        with pytest.raises(PermissionError, match="Server-side approval record is required"):
            await service.create_meeting_note_writeback(
                company_id="company-target-1",
                note_title="Unapproved User Note",
                summary_markdown="Notes",
                idempotency_key="idem-no-approval",
                user_id=user.id,
                user_role="USER",
                approval_id=None,
                approved=True,
            )


@pytest.mark.asyncio
async def test_remediation_writeback_valid_server_approval_succeeds(setup_dept_and_users):
    """SEC-02 & SEC-03: Valid server-side approval ticket allows write-back and marks approval consumed."""
    dept, admin, user = setup_dept_and_users
    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.create_note.return_value = "note-uuid-approved"
        mock_mcp.create_note_target_for_company.return_value = "target-uuid-approved"

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        cid = f"company-auth-{uuid.uuid4().hex[:6]}"

        from rag_app.schemas.workflow import compute_canonical_draft_hash

        # 1. Admin creates server-side approval ticket
        approval = await service.create_writeback_approval(
            company_id=cid,
            department_id=dept.id,
            approved_by=admin.id,
            requested_by=user.id,
            approved_draft_hash=compute_canonical_draft_hash("Summary content"),
        )
        assert approval.status == "APPROVED"

        # 2. User executes write-back referencing approval ticket
        res = await service.create_meeting_note_writeback(
            company_id=cid,
            note_title="Approved Executive Meeting",
            summary_markdown="Summary content",
            idempotency_key=f"idem-approved-{uuid.uuid4().hex[:6]}",
            user_id=user.id,
            user_role="USER",
            approval_id=approval.id,
            approved=True,
        )

        assert res["status"] == "COMPLETED"
        assert res["crm_note_id"] == "note-uuid-approved"

        # 3. Verify approval record is marked CONSUMED and cannot be reused
        refreshed_app = await session.get(CRMWriteBackApproval, approval.id)
        assert refreshed_app.status == "CONSUMED"

        # 4. Attempting to reuse consumed approval raises error
        with pytest.raises(PermissionError, match="already consumed"):
            await service.create_meeting_note_writeback(
                company_id=cid,
                note_title="Reused Ticket Note",
                summary_markdown="Duplicate attempt",
                idempotency_key=f"idem-reuse-{uuid.uuid4().hex[:6]}",
                user_id=user.id,
                user_role="USER",
                approval_id=approval.id,
                approved=True,
            )


@pytest.mark.asyncio
async def test_remediation_writeback_approval_target_mismatch(setup_dept_and_users):
    """SEC-02: Approval for company A cannot authorize company B."""
    dept, admin, user = setup_dept_and_users
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())
        approval = await service.create_writeback_approval(
            company_id="company-A",
            department_id=dept.id,
            approved_by=admin.id,
        )

        with pytest.raises(PermissionError, match="cannot authorize company"):
            await service.create_meeting_note_writeback(
                company_id="company-B",  # Mismatch!
                note_title="Mismatch Note",
                summary_markdown="Content",
                idempotency_key="idem-mismatch",
                user_id=user.id,
                user_role="USER",
                approval_id=approval.id,
            )


# ============================================================================
# REL-01: Two-Phase Note Creation Recovery (No Duplicates on Retry)
# ============================================================================

@pytest.mark.asyncio
async def test_remediation_writeback_partial_failure_retry_avoids_duplicate():
    """
    REL-01: If create_note succeeds but create_note_target fails, note_id is durably
    persisted. Retrying with the same idempotency_key skips create_note and only retries linking.
    """
    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.create_note.return_value = "note-durable-999"
        mock_mcp.create_note_target_for_company.side_effect = [
            RuntimeError("Network disconnect during linking"),
            "target-linked-888",
        ]

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        idem_key = f"idem-rel01-{uuid.uuid4().hex[:8]}"

        # Attempt 1: Fails at step 2
        with pytest.raises(RuntimeError, match="Network disconnect"):
            await service.create_meeting_note_writeback(
                company_id="comp-rel01-10",
                note_title="Durable Note",
                summary_markdown="Summary",
                idempotency_key=idem_key,
                user_role="ADMIN",  # Admin can act directly
                approved=True,
            )

        # Check DB state: note_id was durably persisted despite step 2 failure!
        rec = await session.scalar(
            select(CRMWriteBackRecord).where(CRMWriteBackRecord.idempotency_key == idem_key)
        )
        assert rec is not None
        assert rec.crm_note_id == "note-durable-999"
        assert rec.status == "LINK_FAILED"
        assert mock_mcp.create_note.call_count == 1

        # Attempt 2 (Retry): Uses saved crm_note_id and executes ONLY step 2
        res2 = await service.create_meeting_note_writeback(
            company_id="comp-rel01-10",
            note_title="Durable Note",
            summary_markdown="Summary",
            idempotency_key=idem_key,
            user_role="ADMIN",
            approved=True,
        )

        assert res2["status"] == "COMPLETED"
        assert res2["crm_note_id"] == "note-durable-999"
        assert res2["crm_note_target_id"] == "target-linked-888"
        # Crucial: create_note was NOT called again on retry! (call_count remains 1)
        assert mock_mcp.create_note.call_count == 1


# ============================================================================
# REL-02: Concurrency-Safe Sync Record Creation
# ============================================================================

@pytest.mark.asyncio
async def test_remediation_concurrency_sync_record(setup_dept_and_users):
    """REL-02: Simultaneous workers processing the same record converge safely without unhandled IntegrityError."""
    dept, admin, _ = setup_dept_and_users
    cid = f"comp-concurrent-{uuid.uuid4().hex[:6]}"

    async def worker():
        async with AsyncSessionLocal() as session:
            service = CRMSyncService(session, ingestion_service=AsyncMock())
            return await service._get_or_create_sync_record(
                external_record_id=cid,
                department_id=dept.id,
            )

    # Run two simultaneous worker tasks for the exact same company ID
    rec1, rec2 = await asyncio.gather(worker(), worker())

    assert rec1 is not None
    assert rec2 is not None
    # Both workers converge on the exact same database record ID
    assert rec1.id == rec2.id
    assert rec1.external_record_id == cid


# ============================================================================
# SEC-05 & HTTP Route Webhook Protection Tests
# ============================================================================

@pytest.mark.asyncio
async def test_remediation_webhook_http_oversized_payload_rejected():
    """SEC-05: Webhook HTTP endpoint rejects payloads exceeding maximum size limit with 413."""
    from rag_app.core.config import settings

    # Temporarily set max_bytes to 500 bytes to test limit deterministically
    orig_max = settings.twenty_webhook_max_bytes
    orig_secret = settings.twenty_webhook_secret
    try:
        settings.twenty_webhook_max_bytes = 500
        settings.twenty_webhook_secret = "test-secret-route"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            oversized_data = b"x" * 600
            sig = hmac.new(b"test-secret-route", oversized_data, hashlib.sha256).hexdigest()

            # Oversized request rejected with 413
            resp = await client.post(
                "/crm/webhooks/twenty",
                content=oversized_data,
                headers={"X-Twenty-Signature": sig},
            )
            assert resp.status_code == 413
            assert "exceeds maximum size limit" in resp.json()["detail"]

    finally:
        settings.twenty_webhook_max_bytes = orig_max
        settings.twenty_webhook_secret = orig_secret


@pytest.mark.asyncio
async def test_remediation_webhook_http_missing_signature_rejected():
    """SEC-01: Webhook HTTP endpoint rejects missing signature header with 401."""
    from rag_app.core.config import settings

    orig_secret = settings.twenty_webhook_secret
    try:
        settings.twenty_webhook_secret = "test-secret-route"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/crm/webhooks/twenty",
                json={"event": "company.deleted", "record": {"id": "123"}},
            )
            assert resp.status_code == 401
            assert "Missing webhook signature header" in resp.json()["detail"]
    finally:
        settings.twenty_webhook_secret = orig_secret

