"""
Unit and Integration Tests for Knowledge Base -> Twenty CRM Write-Back Operations.

Covers:
1. Execution of approved Note creation linked to Company via NoteTarget.
2. Idempotency enforcement: replaying with the same idempotency_key returns cached result
   without issuing duplicate API calls.
3. Fail-closed security: unapproved write-back attempts are strictly rejected.
4. Client-side allowlist enforcement: call_write_tool rejects any tool outside
   APPROVED_WRITE_BACK_TOOLS.
"""

import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.crm.sync_service import CRMSyncService
from rag_app.crm.twenty_mcp_client import (
    APPROVED_WRITE_BACK_TOOLS,
    TwentyMCPAllowlistViolation,
    TwentyMCPClient,
)
from rag_app.db.models import CRMWriteBackRecord
from rag_app.db.session import AsyncSessionLocal, engine


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest.mark.asyncio
async def test_write_tool_security_allowlist_enforcement():
    client = TwentyMCPClient(enabled=True)

    # 1. Unapproved tool should be rejected immediately
    with pytest.raises(TwentyMCPAllowlistViolation, match="not in the approved write-back allowlist"):
        await client.call_write_tool("delete_one_company", {"id": "123"})

    with pytest.raises(TwentyMCPAllowlistViolation, match="not in the approved write-back allowlist"):
        await client.call_write_tool("create_one_company", {"name": "Malicious Corp"})

    # 2. Approved tools are in the allowlist
    assert "create_one_note" in APPROVED_WRITE_BACK_TOOLS
    assert "create_one_note_target" in APPROVED_WRITE_BACK_TOOLS


@pytest.mark.asyncio
async def test_meeting_note_writeback_success():
    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.create_note.return_value = "note-uuid-999"
        mock_mcp.create_note_target_for_company.return_value = "target-uuid-888"

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        idem_key = f"idem-key-{uuid.uuid4().hex[:8]}"

        result = await service.create_meeting_note_writeback(
            company_id="company-uuid-111",
            note_title="Q4 Strategy Review",
            summary_markdown="Discussed enterprise contract expansion and SLA terms.",
            idempotency_key=idem_key,
            user_role="ADMIN",
            approved=True,
        )

        assert result["status"] == "COMPLETED"
        assert result["crm_note_id"] == "note-uuid-999"
        assert result["crm_note_target_id"] == "target-uuid-888"
        assert result["is_duplicate_replay"] is False

        # Verify durable ledger in DB
        db_rec = await session.scalar(
            select(CRMWriteBackRecord).where(CRMWriteBackRecord.idempotency_key == idem_key)
        )
        assert db_rec is not None
        assert db_rec.status == "COMPLETED"
        assert db_rec.crm_note_id == "note-uuid-999"
        assert db_rec.crm_note_target_id == "target-uuid-888"


@pytest.mark.asyncio
async def test_meeting_note_writeback_idempotency():
    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.create_note.return_value = "note-uuid-123"
        mock_mcp.create_note_target_for_company.return_value = "target-uuid-456"

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        idem_key = f"idem-replay-{uuid.uuid4().hex[:8]}"

        # First call: executes write-back
        res1 = await service.create_meeting_note_writeback(
            company_id="company-uuid-222",
            note_title="Initial Sync",
            summary_markdown="Meeting notes content",
            idempotency_key=idem_key,
            user_role="ADMIN",
            approved=True,
        )
        assert res1["is_duplicate_replay"] is False
        assert mock_mcp.create_note.call_count == 1

        # Second call with same idempotency key: returns cached result without issuing CRM calls
        res2 = await service.create_meeting_note_writeback(
            company_id="company-uuid-222",
            note_title="Initial Sync",
            summary_markdown="Meeting notes content",
            idempotency_key=idem_key,
            user_role="ADMIN",
            approved=True,
        )
        assert res2["is_duplicate_replay"] is True
        assert res2["crm_note_id"] == "note-uuid-123"
        assert mock_mcp.create_note.call_count == 1  # Not called again!


@pytest.mark.asyncio
async def test_meeting_note_writeback_requires_approval():
    async with AsyncSessionLocal() as session:
        service = CRMSyncService(session, ingestion_service=AsyncMock())
        with pytest.raises(PermissionError, match="not explicitly approved"):
            await service.create_meeting_note_writeback(
                company_id="company-uuid-333",
                note_title="Unapproved Note",
                summary_markdown="Secret meeting",
                idempotency_key="unapproved-key",
                approved=False,
            )
