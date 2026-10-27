"""
Comprehensive Test Suite for Milestone 6H:
Durable Connection-Scoped External CRM Sync and Automated Reconciliation.

Covers:
1. Connection-scoped bounded batch sync with durable CRMSyncJob and CRMSyncRecord tracking.
2. Incremental synchronization: deterministic fingerprint skip and changed-record updates.
3. Concurrency lock & lease fencing: overlapping sync runs for the same connection are rejected (409 Conflict).
4. Worker restart recovery: stale RUNNING jobs past 15-minute lease are recovered to FAILED; clean resumption.
5. Strict Checkpoint Advancement Invariant: failed pages never advance the checkpoint cursor.
6. Error classification: transient errors trigger bounded exponential backoff; permanent errors fail immediately.
7. Multi-tenant workspace connection isolation: separate connections never collide on identical external IDs.
8. Automated reconciliation: repairs drift, indexes missed records, applies conservative archival (ARCHIVED).
9. Uncertain write-back operation recovery: UNKNOWN operation records transition to RECONCILIATION_REQUIRED.
10. Data governance: Global chat retrieval (workspace_id=None) strictly excludes workspace-isolated CRM documents.
11. DurableReconciliationScheduler integration: scheduler cycle executes CRM sync recovery and retries.
"""

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from rag_app.crm.sync_service import (
    CRMSyncService,
    CRMSyncStatus,
    ConcurrencyConflictError,
    compute_crm_company_fingerprint,
)
from rag_app.crm.twenty_client import (
    CRMCompany,
    CRMCompanyState,
    CRMOpportunity,
    CRMPerson,
)
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import (
    Base,
    CRMSyncJob,
    CRMSyncRecord,
    Department,
    Document,
    MCPToolOperationRecord,
    User,
    Workspace,
    WorkspaceMCPConnection,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.knowledge.scheduler import DurableReconciliationScheduler
from rag_app.retrieval.service import RetrievalService
from rag_app.schemas.rag import IngestResponse


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_pipeline_context():
    """Sets up an active department, workspace, user, and workspace MCP connection."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        # Clean previous test records
        await session.execute(delete(MCPToolOperationRecord))
        await session.execute(delete(CRMSyncRecord))
        await session.execute(delete(CRMSyncJob))
        await session.commit()

        # 1. Department
        dept = await session.scalar(
            select(Department).where(Department.code == "CRM_PIPE_TEST")
        )
        if not dept:
            dept = Department(
                id=uuid.uuid4(),
                code="CRM_PIPE_TEST",
                name="CRM Pipeline Test Dept",
                is_active=True,
            )
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        # 2. User
        user = await session.scalar(
            select(User).where(User.email == "crm_pipeline_user@test.com")
        )
        if not user:
            user = User(
                id=uuid.uuid4(),
                email="crm_pipeline_user@test.com",
                hashed_password="hashed_pw",
                role="ADMIN",
                is_active=True,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

        # 3. Workspace
        ws = await session.scalar(
            select(Workspace).where(Workspace.name == "CRM Pipeline Workspace")
        )
        if not ws:
            ws = Workspace(
                id=uuid.uuid4(),
                name="CRM Pipeline Workspace",
                code="crm_pipe_ws",
                department_id=dept.id,
                is_active=True,
            )
            session.add(ws)
            await session.commit()
            await session.refresh(ws)

        # 4. Connection
        conn = await session.scalar(
            select(WorkspaceMCPConnection).where(
                WorkspaceMCPConnection.workspace_id == ws.id,
                WorkspaceMCPConnection.name == "Twenty CRM Pipeline",
            )
        )
        if not conn:
            conn = WorkspaceMCPConnection(
                id=uuid.uuid4(),
                workspace_id=ws.id,
                name="Twenty CRM Pipeline",
                endpoint_url="http://test-twenty-pipe:4000/mcp",
                transport="streamable-http",
                status="CONNECTED",
                is_enabled=True,
                server_info={"version": "1.0"},
                discovered_tools=[],
                enabled_tools=["read_company"],
                configuration={"credential_mode": "default"},
            )
            session.add(conn)
            await session.commit()
            await session.refresh(conn)

        return dept, user, ws, conn


def _make_company_state(cid: str, name: str, revenue: float = 250000.0) -> CRMCompanyState:
    company = CRMCompany(
        id=cid,
        name=name,
        domain=f"{name.lower().replace(' ', '')}.com",
        address="100 Pipeline Way, Silicon Valley, CA",
        annual_revenue=revenue,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T00:00:00Z",
    )
    person = CRMPerson(
        id=f"per-{cid[:6]}",
        name=f"Lead {name}",
        job_title="VP Engineering",
        email=f"lead@{name.lower().replace(' ', '')}.com",
        phone="555-9876",
        company_id=cid,
    )
    opp = CRMOpportunity(
        id=f"opp-{cid[:6]}",
        name=f"Deal for {name}",
        amount=revenue * 0.2,
        currency="USD",
        stage="Proposal",
        close_date="2026-12-31",
        company_id=cid,
    )
    return CRMCompanyState(
        company_name=name,
        company=company,
        people=[person],
        opportunities=[opp],
        is_connected=True,
    )


async def _make_test_document(session, dept_id: uuid.UUID, ws_id: uuid.UUID | None = None) -> Document:
    doc = Document(
        id=uuid.uuid4(),
        file_name=f"test_doc_{uuid.uuid4().hex[:6]}.md",
        file_path="uploaded_docs/test_doc.md",
        content_hash=uuid.uuid4().hex,
        mime_type="text/markdown",
        department_id=dept_id,
        workspace_id=ws_id,
        status="COMPLETED",
    )
    session.add(doc)
    await session.commit()
    return doc


@pytest.mark.asyncio
async def test_connection_bounded_batch_sync_success(setup_pipeline_context):
    """Proves bounded connection-scoped sync records durable job and sync records."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        cid1 = f"c-pipe-1-{uuid.uuid4().hex[:6]}"
        cid2 = f"c-pipe-2-{uuid.uuid4().hex[:6]}"

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.return_value = (
            [CRMCompany(id=cid1, name="Alpha Corp"), CRMCompany(id=cid2, name="Beta Corp")],
            False,
            2,
        )
        mock_mcp.get_live_company_state_by_id.side_effect = lambda cid: (
            _make_company_state(cid1, "Alpha Corp") if cid == cid1 else _make_company_state(cid2, "Beta Corp")
        )

        async def _mock_ingest1(**kwargs):
            doc = await _make_test_document(session, dept.id, ws.id)
            return IngestResponse(
                file_name=str(kwargs.get("file_path")),
                skipped=False,
                document_id=str(doc.id),
            )

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.side_effect = _mock_ingest1

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)
        summary = await service.sync_connection_batch(
            workspace_id=ws.id,
            connection_id=conn.id,
            department_id=dept.id,
            limit=10,
            offset=0,
            created_by=user.id,
        )

        assert summary.status == "COMPLETED"
        assert summary.total_records == 2
        assert summary.succeeded_records == 2
        assert summary.failed_records == 0
        assert summary.skipped_records == 0

        cursor_data = json.loads(summary.checkpoint_cursor)
        assert cursor_data["offset"] == 2
        assert cursor_data["connection_id"] == str(conn.id)

        # Verify durable CRMSyncJob in DB
        job = await session.get(CRMSyncJob, summary.job_id)
        assert job is not None
        assert job.status == "COMPLETED"
        assert job.job_type == "CONNECTION_SYNC"
        assert job.workspace_id == ws.id

        # Verify durable CRMSyncRecords partitioned by provider=twenty_crm:{conn.id}
        expected_provider = f"twenty_crm:{conn.id}"
        rec1 = await session.scalar(
            select(CRMSyncRecord).where(
                CRMSyncRecord.provider == expected_provider,
                CRMSyncRecord.external_record_id == cid1,
            )
        )
        assert rec1 is not None
        assert rec1.status == CRMSyncStatus.SUCCEEDED.value
        assert rec1.source_fingerprint is not None
        assert rec1.workspace_id == ws.id


@pytest.mark.asyncio
async def test_connection_incremental_sync_skips_unchanged_and_updates_changed(setup_pipeline_context):
    """Proves unchanged fingerprints are cleanly skipped and modified records trigger re-sync."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        cid = f"c-inc-{uuid.uuid4().hex[:6]}"

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.return_value = ([CRMCompany(id=cid, name="Gamma Inc")], False, 1)
        mock_mcp.get_live_company_state_by_id.return_value = _make_company_state(cid, "Gamma Inc", revenue=100000.0)

        async def _mock_ingest2(**kwargs):
            doc = await _make_test_document(session, dept.id, ws.id)
            return IngestResponse(
                file_name=str(kwargs.get("file_path", "test.md")),
                skipped=False,
                document_id=str(doc.id),
            )

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.side_effect = _mock_ingest2

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)

        # 1. First sync -> Succeeded
        s1 = await service.sync_connection_batch(ws.id, conn.id, dept.id)
        assert s1.succeeded_records == 1
        assert s1.skipped_records == 0
        assert mock_ingestion.ingest_file.call_count == 1

        # 2. Second sync with identical data -> Cleanly skipped, no ingestion call
        mock_ingestion.ingest_file.reset_mock()
        s2 = await service.sync_connection_batch(ws.id, conn.id, dept.id)
        assert s2.succeeded_records == 0
        assert s2.skipped_records == 1
        assert mock_ingestion.ingest_file.call_count == 0

        # 3. Modify revenue -> fingerprint changes -> triggers ingestion
        mock_mcp.get_live_company_state_by_id.return_value = _make_company_state(cid, "Gamma Inc", revenue=555000.0)
        s3 = await service.sync_connection_batch(ws.id, conn.id, dept.id)
        assert s3.succeeded_records == 1
        assert s3.skipped_records == 0
        assert mock_ingestion.ingest_file.call_count == 1


@pytest.mark.asyncio
async def test_connection_concurrency_lock_rejects_overlapping_sync(setup_pipeline_context):
    """Proves an active RUNNING job blocks concurrent sync runs with 409 ConcurrencyConflictError."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        # Pre-seed an active RUNNING job started 2 minutes ago
        active_job = CRMSyncJob(
            id=uuid.uuid4(),
            job_type="CONNECTION_SYNC",
            status="RUNNING",
            department_id=dept.id,
            workspace_id=ws.id,
            started_at=datetime.now(timezone.utc) - timedelta(minutes=2),
            checkpoint_cursor=json.dumps({"offset": 0, "connection_id": str(conn.id)}),
        )
        session.add(active_job)
        await session.commit()

        service = CRMSyncService(session, mcp_client=AsyncMock())
        with pytest.raises(ConcurrencyConflictError, match="already actively running"):
            await service.sync_connection_batch(ws.id, conn.id, dept.id)


@pytest.mark.asyncio
async def test_connection_stale_lease_recovery_on_worker_restart(setup_pipeline_context):
    """Proves a stale RUNNING job past the 15-minute lease is automatically recovered to FAILED."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=25)
        stale_job = CRMSyncJob(
            id=uuid.uuid4(),
            job_type="CONNECTION_SYNC",
            status="RUNNING",
            department_id=dept.id,
            workspace_id=ws.id,
            started_at=stale_time,
            checkpoint_cursor=json.dumps({"offset": 10, "connection_id": str(conn.id)}),
        )
        session.add(stale_job)
        await session.commit()

        # Calling recover_abandoned_sync_jobs directly transitions it
        recovered = await CRMSyncService.recover_abandoned_sync_jobs(session, lease_timeout_seconds=900)
        assert recovered == 1

        updated_stale = await session.get(CRMSyncJob, stale_job.id)
        assert updated_stale.status == "FAILED"
        assert "expired" in updated_stale.error_message.lower()

        # Now a new sync can claim the lock cleanly and run
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.return_value = ([], False, 0)
        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        new_summary = await service.sync_connection_batch(ws.id, conn.id, dept.id, offset=10)
        assert new_summary.status == "COMPLETED"


@pytest.mark.asyncio
async def test_connection_checkpoint_advancement_invariant(setup_pipeline_context):
    """Proves checkpoint cursor is NOT advanced if page retrieval fails."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.side_effect = httpx.NetworkError("Remote Twenty server disconnected")

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        with pytest.raises(httpx.NetworkError):
            await service.sync_connection_batch(ws.id, conn.id, dept.id, offset=15)

        # Verify job marked FAILED and checkpoint remains at initial offset 15
        stmt = select(CRMSyncJob).where(
            CRMSyncJob.workspace_id == ws.id,
            CRMSyncJob.job_type == "CONNECTION_SYNC",
        ).order_by(CRMSyncJob.created_at.desc())
        job = await session.scalar(stmt)
        assert job.status == "FAILED"
        cursor_data = json.loads(job.checkpoint_cursor)
        assert cursor_data["offset"] == 15  # Did NOT advance!


@pytest.mark.asyncio
async def test_connection_transient_error_backoff_and_permanent_classification(setup_pipeline_context):
    """Proves transient errors set retryable backoff while permanent errors fail immediately."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        cid_transient = f"c-trans-{uuid.uuid4().hex[:6]}"
        cid_permanent = f"c-perm-{uuid.uuid4().hex[:6]}"

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.return_value = (
            [CRMCompany(id=cid_transient, name="Transient Co"), CRMCompany(id=cid_permanent, name="Perm Co")],
            False,
            2,
        )

        def mock_get_company(cid):
            if cid == cid_transient:
                raise httpx.TimeoutException("Remote timeout (504)")
            else:
                raise ValueError("Invalid schema attributes")

        mock_mcp.get_live_company_state_by_id.side_effect = mock_get_company

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        summary = await service.sync_connection_batch(ws.id, conn.id, dept.id)

        assert summary.status == "FAILED"
        assert summary.failed_records == 2

        provider_key = f"twenty_crm:{conn.id}"
        rec_trans = await session.scalar(
            select(CRMSyncRecord).where(
                CRMSyncRecord.provider == provider_key,
                CRMSyncRecord.external_record_id == cid_transient,
            )
        )
        assert rec_trans.status == CRMSyncStatus.RETRYABLE_FAILED.value
        assert rec_trans.next_retry_at is not None
        assert rec_trans.attempt_count == 1

        rec_perm = await session.scalar(
            select(CRMSyncRecord).where(
                CRMSyncRecord.provider == provider_key,
                CRMSyncRecord.external_record_id == cid_permanent,
            )
        )
        assert rec_perm.status == CRMSyncStatus.PERMANENT_FAILED.value
        assert rec_perm.next_retry_at is None


@pytest.mark.asyncio
async def test_multi_tenant_workspace_connection_isolation(setup_pipeline_context):
    """Proves two distinct workspaces/connections syncing the same external ID do not collide."""
    dept, user, ws1, conn1 = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        # Create second workspace & connection
        ws2 = Workspace(
            id=uuid.uuid4(),
            name="Workspace Two Isolation",
            code=f"ws2_iso_{uuid.uuid4().hex[:6]}",
            department_id=dept.id,
            is_active=True,
        )
        session.add(ws2)
        await session.commit()

        conn2 = WorkspaceMCPConnection(
            id=uuid.uuid4(),
            workspace_id=ws2.id,
            name="Twenty CRM Conn 2",
            endpoint_url="http://twenty-ws2:4000/mcp",
            transport="streamable-http",
            status="CONNECTED",
            is_enabled=True,
            server_info={"version": "1.0"},
            discovered_tools=[],
            enabled_tools=["read_company"],
            configuration={"credential_mode": "default"},
        )
        session.add(conn2)
        await session.commit()

        shared_cid = "same-external-crm-uuid-12345"
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.return_value = ([CRMCompany(id=shared_cid, name="Shared Corp")], False, 1)
        mock_mcp.get_live_company_state_by_id.return_value = _make_company_state(shared_cid, "Shared Corp")

        async def _mock_ingest3(**kwargs):
            doc = await _make_test_document(session, dept.id, kwargs.get("workspace_id"))
            return IngestResponse(
                file_name=str(kwargs.get("file_path", "doc.md")),
                skipped=False,
                document_id=str(doc.id),
            )

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.side_effect = _mock_ingest3

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)

        # Sync on Workspace 1 / Connection 1
        s1 = await service.sync_connection_batch(ws1.id, conn1.id, dept.id)
        assert s1.succeeded_records == 1

        # Sync on Workspace 2 / Connection 2
        s2 = await service.sync_connection_batch(ws2.id, conn2.id, dept.id)
        assert s2.succeeded_records == 1

        # Verify two distinct sync records partitioned by provider key
        rec1 = await session.scalar(
            select(CRMSyncRecord).where(
                CRMSyncRecord.provider == f"twenty_crm:{conn1.id}",
                CRMSyncRecord.external_record_id == shared_cid,
            )
        )
        rec2 = await session.scalar(
            select(CRMSyncRecord).where(
                CRMSyncRecord.provider == f"twenty_crm:{conn2.id}",
                CRMSyncRecord.external_record_id == shared_cid,
            )
        )

        assert rec1 is not None and rec2 is not None
        assert rec1.id != rec2.id
        assert rec1.workspace_id == ws1.id
        assert rec2.workspace_id == ws2.id


@pytest.mark.asyncio
async def test_connection_drift_reconciliation_and_conservative_deletion(setup_pipeline_context):
    """Proves reconciliation repairs drift and transitions deleted remote records to ARCHIVED."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        provider_key = f"twenty_crm:{conn.id}"
        cid_existing_deleted = f"c-del-{uuid.uuid4().hex[:6]}"
        cid_remote_active = f"c-act-{uuid.uuid4().hex[:6]}"

        # Pre-seed record that will be deleted remotely
        pre_rec = CRMSyncRecord(
            id=uuid.uuid4(),
            provider=provider_key,
            entity_type="company",
            external_record_id=cid_existing_deleted,
            department_id=dept.id,
            workspace_id=ws.id,
            status=CRMSyncStatus.SUCCEEDED.value,
        )
        session.add(pre_rec)
        await session.commit()

        # Remote Twenty CRM now returns ONLY cid_remote_active
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.return_value = (
            [CRMCompany(id=cid_remote_active, name="Active Co")],
            False,
            1,
        )
        mock_mcp.get_live_company_state_by_id.return_value = _make_company_state(cid_remote_active, "Active Co")

        async def _mock_ingest4(**kwargs):
            doc = await _make_test_document(session, dept.id, ws.id)
            return IngestResponse(
                file_name=str(kwargs.get("file_path", "active.md")),
                skipped=False,
                document_id=str(doc.id),
            )

        mock_ingestion = AsyncMock()
        mock_ingestion.ingest_file.side_effect = _mock_ingest4

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=mock_ingestion)
        recon_report = await service.reconcile_connection(ws.id, conn.id, dept.id)

        assert recon_report["drift_detected"] == 1
        assert recon_report["repaired_records"] == 1
        assert recon_report["archived_records"] == 1

        # Check DB: deleted record is ARCHIVED (not hard deleted)
        updated_pre = await session.get(CRMSyncRecord, pre_rec.id)
        assert updated_pre.status == CRMSyncStatus.ARCHIVED.value


@pytest.mark.asyncio
async def test_uncertain_write_recovery_during_reconciliation(setup_pipeline_context):
    """Proves ambiguous UNKNOWN mutating operations transition to RECONCILIATION_REQUIRED."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        # Pre-seed an ambiguous mutating operation in UNKNOWN status
        op_id = uuid.uuid4()
        op = MCPToolOperationRecord(
            id=op_id,
            workspace_id=ws.id,
            connection_id=conn.id,
            tool_name="create_one_note",
            arguments={"title": "Meeting with Client"},
            idempotency_key=f"idem-{uuid.uuid4().hex[:12]}",
            status="UNKNOWN",
        )
        session.add(op)
        await session.commit()

        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.fetch_companies_page.return_value = ([], False, 0)

        service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        recon_report = await service.reconcile_connection(ws.id, conn.id, dept.id)

        assert recon_report["reconciled_operations"] == 1

        updated_op = await session.get(MCPToolOperationRecord, op_id)
        assert updated_op.status == "RECONCILIATION_REQUIRED"


@pytest.mark.asyncio
async def test_data_governance_global_chat_retrieval_isolation(setup_pipeline_context):
    """Proves global chat queries (workspace_id=None) strictly exclude workspace-restricted documents."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        # 1. Ingest workspace-isolated document
        ws_doc = Document(
            id=uuid.uuid4(),
            file_name=f"ws_crm_doc_{uuid.uuid4().hex[:6]}.md",
            file_path="uploaded_docs/ws_crm.md",
            content_hash=uuid.uuid4().hex,
            mime_type="text/markdown",
            department_id=dept.id,
            workspace_id=ws.id,
            status="COMPLETED",
        )
        # 2. Ingest global document (workspace_id=None)
        global_doc = Document(
            id=uuid.uuid4(),
            file_name=f"global_doc_{uuid.uuid4().hex[:6]}.md",
            file_path="uploaded_docs/global.md",
            content_hash=uuid.uuid4().hex,
            mime_type="text/markdown",
            department_id=dept.id,
            workspace_id=None,
            status="COMPLETED",
        )
        session.add_all([ws_doc, global_doc])
        await session.commit()

        from rag_app.db.models import DocumentChunk
        from rag_app.core.config import settings

        dim = settings.embedding_dim or 1536
        chunk_ws = DocumentChunk(
            id=uuid.uuid4(),
            document_id=ws_doc.id,
            chunk_index=0,
            text="Acme CRM company details and meeting notes",
            embedding=[0.0] * dim,
        )
        chunk_global = DocumentChunk(
            id=uuid.uuid4(),
            document_id=global_doc.id,
            chunk_index=0,
            text="Global company policies and overview",
            embedding=[0.0] * dim,
        )
        session.add_all([chunk_ws, chunk_global])
        await session.commit()

        from rag_app.retrieval.bm25 import GlobalBM25Index

        retrieval_svc = RetrievalService(session=session)

        # 1. Keyword search with workspace_id returns ws_doc only
        with patch.object(GlobalBM25Index, "get_scores", return_value={chunk_ws.id: 1.0, chunk_global.id: 1.0}):
            res_ws = await retrieval_svc._keyword_search("company", candidate_k=10, workspace_id=ws.id)
            assert len(res_ws) == 1
            assert res_ws[0][1].id == ws_doc.id

            # 2. Global chat keyword search (workspace_id=None) strictly returns ONLY global_doc
            res_global = await retrieval_svc._keyword_search("company", candidate_k=10, workspace_id=None)
            assert len(res_global) == 1
            assert res_global[0][1].id == global_doc.id


@pytest.mark.asyncio
async def test_periodic_scheduler_reconciliation_integration(setup_pipeline_context):
    """Proves DurableReconciliationScheduler executes Step 5 for CRM sync recovery and retries."""
    dept, user, ws, conn = setup_pipeline_context
    async with AsyncSessionLocal() as session:
        # Pre-seed an abandoned CRM sync job
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        abandoned_job = CRMSyncJob(
            id=uuid.uuid4(),
            job_type="CONNECTION_SYNC",
            status="RUNNING",
            department_id=dept.id,
            workspace_id=ws.id,
            started_at=stale_time,
            checkpoint_cursor=json.dumps({"offset": 0, "connection_id": str(conn.id)}),
        )
        session.add(abandoned_job)
        await session.commit()

        scheduler = DurableReconciliationScheduler(interval_seconds=60, session_factory=AsyncSessionLocal)
        metrics = await scheduler.run_once()

        assert "recovered_crm_sync_jobs" in metrics
        assert metrics["recovered_crm_sync_jobs"] >= 1

        async with AsyncSessionLocal() as verify_session:
            updated_job = await verify_session.get(CRMSyncJob, abandoned_job.id)
            assert updated_job is not None
            assert updated_job.status == "FAILED"
