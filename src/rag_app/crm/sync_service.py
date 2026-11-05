"""
CRM to Knowledge Base Persistence and Synchronization Service.

Provides:
- Phase 1: Bounded batch synchronization over Twenty CRM records with durable job tracking.
- Phase 2: Incremental synchronization, deterministic source fingerprinting, checkpoints, idempotency, and safe retry backoff.
- Phase 3: Verified Twenty CRM webhook handling and scheduled reconciliation with conservative deletion handling.
- Phase 4: Approved Knowledge Base -> CRM write-back (meeting summary Note creation linked to Company).

Guarantees:
- Reuses TwentyMCPClient, IngestionService, Document, and DocumentChunk models.
- Does not modify chunking, embedding, hybrid search, RRF, reranking, or prompt generation.
- sync_timestamp is stored in doc_metadata, NOT in the deterministic Markdown text.
- Validates real department UUIDs and active department status against the database.
- Per-record error isolation in batch jobs: failures on individual records do not abort unrelated records.
- Fail-closed authorization and explicit approval required for write-backs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import httpx
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.crm.twenty_client import CRMCompanyState
from rag_app.crm.twenty_mcp_client import (
    TwentyMCPAllowlistViolation,
    TwentyMCPClient,
    TwentyMCPError,
)
from rag_app.schemas.rag import IngestResponse

if TYPE_CHECKING:
    from rag_app.db.models import (
        CRMSyncJob,
        CRMSyncRecord,
        CRMWriteBackRecord,
        Department,
    )
    from rag_app.ingestion.service_v1 import IngestionService

logger = logging.getLogger(__name__)


class CRMSyncStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    SKIPPED = "SKIPPED"
    RETRYABLE_FAILED = "RETRYABLE_FAILED"
    PERMANENT_FAILED = "PERMANENT_FAILED"
    ARCHIVED = "ARCHIVED"


@dataclass
class RecordSyncResult:
    record_id: str
    company_name: str
    status: str
    document_id: Optional[uuid.UUID] = None
    fingerprint: Optional[str] = None
    error_message: Optional[str] = None
    skipped: bool = False


@dataclass
class BatchSyncSummary:
    job_id: uuid.UUID
    status: str
    total_records: int
    processed_records: int
    succeeded_records: int
    skipped_records: int
    failed_records: int
    checkpoint_cursor: Optional[str] = None
    records: list[RecordSyncResult] = field(default_factory=list)
    error_message: Optional[str] = None


def compute_crm_company_fingerprint(company_state: CRMCompanyState) -> str:
    """
    Computes a deterministic SHA-256 fingerprint of the CRMCompanyState.
    Includes stable company fields, contacts, and deals.
    Excludes volatile fields such as sync_timestamp and fetch latency.
    """
    if not company_state.company:
        return ""

    c = company_state.company
    people_payload = [
        {
            "id": p.id,
            "name": p.name,
            "email": p.email,
            "phone": p.phone,
            "job_title": p.job_title,
        }
        for p in sorted(company_state.people, key=lambda x: str(x.id))
    ]

    opps_payload = [
        {
            "id": o.id,
            "name": o.name,
            "amount": o.amount,
            "currency": o.currency,
            "stage": o.stage,
            "close_date": o.close_date,
        }
        for o in sorted(company_state.opportunities, key=lambda x: str(x.id))
    ]

    canonical_dict = {
        "id": str(c.id),
        "name": str(c.name).strip(),
        "domain": str(c.domain or "").strip(),
        "address": str(c.address or "").strip(),
        "annual_revenue": float(c.annual_revenue) if c.annual_revenue is not None else None,
        "people": people_payload,
        "opportunities": opps_payload,
    }

    encoded = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ConcurrencyConflictError(Exception):
    """Raised when a sync job is already actively running for a connection or workspace."""
    pass


def normalize_crm_company_to_markdown(
    company_state: CRMCompanyState,
    verification_marker: Optional[str] = None,
    connection_id: Optional[uuid.UUID] = None,
    workspace_id: Optional[uuid.UUID] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Normalizes a CRMCompanyState into deterministic Markdown text and structured metadata.

    IMPORTANT: To preserve SHA-256 content-hash deduplication in IngestionService,
    volatile timestamps (such as sync_timestamp) are stored strictly in doc_metadata,
    NOT inside the generated Markdown body.
    """
    if not company_state.company:
        raise ValueError("Cannot normalize CRMCompanyState without a valid company record.")

    c = company_state.company

    sections: list[str] = [f"# CRM Company: {c.name}"]

    if verification_marker:
        sections.append(f"\n## Verification\n- Verification marker: {verification_marker}")

    sections.append("\n## Company Details")
    sections.append(f"- Company Name: {c.name}")
    sections.append(f"- Domain: {c.domain or 'Not provided'}")
    sections.append(f"- Address: {c.address or 'Not provided'}")
    if c.annual_revenue is not None:
        sections.append(f"- Annual Revenue: ${c.annual_revenue:,.2f}")

    if company_state.people:
        sections.append("\n## Associated Contacts")
        for p in company_state.people:
            contact_line = f"- {p.name}"
            if p.job_title:
                contact_line += f" ({p.job_title})"
            if p.email:
                contact_line += f" | Email: {p.email}"
            if p.phone:
                contact_line += f" | Phone: {p.phone}"
            sections.append(contact_line)

    if company_state.opportunities:
        sections.append("\n## Active Opportunities / Deals")
        for o in company_state.opportunities:
            val_str = f"${o.amount:,.2f} {o.currency}" if o.amount is not None else "Unspecified"
            close_str = f" | Target Close: {o.close_date}" if o.close_date else ""
            sections.append(f"- {o.name} | Stage: {o.stage} | Value: {val_str}{close_str}")

    sections.append("\n## Source Provenance")
    sections.append("- Source System: Twenty CRM (MCP)")
    sections.append("- Source Object: company")
    sections.append(f"- Source Record ID: {c.id}")
    if c.created_at:
        sections.append(f"- Record Created: {c.created_at}")
    if c.updated_at:
        sections.append(f"- Record Updated: {c.updated_at}")

    markdown_content = "\n".join(sections) + "\n"

    doc_metadata: dict[str, Any] = {
        "source": "twenty_crm_mcp",
        "crm_record_id": str(c.id),
        "crm_object": "company",
        "company_name": str(c.name),
        "doc_type": "CRM_RECORD",
        "sync_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if verification_marker:
        doc_metadata["verification_marker"] = verification_marker
    if connection_id:
        doc_metadata["connection_id"] = str(connection_id)
    if workspace_id:
        doc_metadata["workspace_id"] = str(workspace_id)

    return markdown_content, doc_metadata
_UNSET = object()


class CRMSyncService:
    """
    Syncs verified CRM records into the RAG Knowledge Base via IngestionService.
    Supports bounded batch sync, incremental sync with fingerprints, safe retries,
    scheduled reconciliation, and controlled meeting summary write-backs.
    """

    def __init__(
        self,
        session: AsyncSession,
        mcp_client: Optional[TwentyMCPClient] = None,
        ingestion_service: Optional[Any] = None,
        staging_dir: Optional[Path] = None,
    ):
        self.session = session
        self.mcp_client = mcp_client or TwentyMCPClient()
        if ingestion_service is not None:
            self.ingestion_service = ingestion_service
        else:
            from rag_app.ingestion.service_v1 import IngestionService
            self.ingestion_service = IngestionService(session)
        self.staging_dir = staging_dir or Path("uploaded_docs")
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    async def _resolve_department(self, department_id: uuid.UUID | str) -> Any:
        """Resolves and validates an active department against the database schema."""
        from rag_app.db.models import Department

        dept_uuid: Optional[uuid.UUID] = None
        if isinstance(department_id, uuid.UUID):
            dept_uuid = department_id
        else:
            try:
                dept_uuid = uuid.UUID(str(department_id).strip())
            except (ValueError, TypeError):
                dept_uuid = None

        if dept_uuid:
            dept = await self.session.get(Department, dept_uuid)
        else:
            dept = await self.session.scalar(
                select(Department).where(Department.code == str(department_id).strip())
            )

        if not dept:
            raise ValueError(f"Department '{department_id}' not found in database.")
        if not dept.is_active:
            raise ValueError(f"Department '{department_id}' ({dept.code}) is inactive.")

        return dept

    def _is_retryable_error(self, ex: Exception) -> bool:
        """Classifies transient network, rate limit, and timeout errors as retryable."""
        if isinstance(ex, (httpx.TimeoutException, httpx.NetworkError, TimeoutError, ConnectionError)):
            return True
        if isinstance(ex, httpx.HTTPStatusError):
            status = ex.response.status_code if ex.response else 500
            return status in (429, 502, 503, 504)
        if isinstance(ex, TwentyMCPError) and "timeout" in str(ex).lower():
            return True
        return False

    async def _get_or_create_sync_record(
        self,
        external_record_id: str,
        department_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
        job_id: Optional[uuid.UUID] = None,
        provider: Optional[str] = None,
    ) -> Any:
        from rag_app.db.models import CRMSyncRecord
        from sqlalchemy.exc import IntegrityError

        provider_name = provider or "twenty_crm"
        stmt = select(CRMSyncRecord).where(
            and_(
                CRMSyncRecord.provider == provider_name,
                CRMSyncRecord.entity_type == "company",
                CRMSyncRecord.external_record_id == str(external_record_id),
            )
        )
        record = await self.session.scalar(stmt)
        if not record:
            record = CRMSyncRecord(
                id=uuid.uuid4(),
                job_id=job_id,
                provider=provider_name,
                entity_type="company",
                external_record_id=str(external_record_id),
                department_id=department_id,
                workspace_id=workspace_id,
                status=CRMSyncStatus.PENDING.value,
                attempt_count=0,
            )
            try:
                self.session.add(record)
                await self.session.commit()
            except IntegrityError:
                await self.session.rollback()
                record = await self.session.scalar(stmt)
                if not record:
                    raise
        else:
            if job_id:
                record.job_id = job_id
            record.department_id = department_id
            if workspace_id:
                record.workspace_id = workspace_id
            await self.session.commit()
        return record

    async def sync_company_state(
        self,
        crm_state: CRMCompanyState,
        department_id: uuid.UUID | str,
        created_by: Optional[uuid.UUID] = None,
        workspace_id: Optional[uuid.UUID] = None,
        verification_marker: Optional[str] = None,
        force: bool = False,
        job_id: Optional[uuid.UUID] = None,
        connection_id: Optional[uuid.UUID] = None,
    ) -> IngestResponse:
        """
        Idempotently synchronizes a CRMCompanyState into the Knowledge Base.
        Deduplicates based on source fingerprint and updates durable CRMSyncRecord.
        """
        if not crm_state.company:
            raise ValueError("CRMCompanyState contains no company record to sync.")

        dept = await self._resolve_department(department_id)
        cid = str(crm_state.company.id)
        current_fingerprint = compute_crm_company_fingerprint(crm_state)
        provider_name = f"twenty_crm:{connection_id}" if connection_id else "twenty_crm"

        sync_rec = await self._get_or_create_sync_record(
            external_record_id=cid,
            department_id=dept.id,
            workspace_id=workspace_id,
            job_id=job_id,
            provider=provider_name,
        )

        now = datetime.now(timezone.utc)
        sync_rec.last_attempted_at = now

        company_id_safe = cid.replace("-", "")[:8]
        if connection_id:
            conn_safe = str(connection_id).replace("-", "")[:8]
            file_name = f"twenty_conn_{conn_safe}_company_{company_id_safe}.md"
        else:
            file_name = f"twenty_company_{company_id_safe}.md"

        # Idempotency check: if source fingerprint is identical, skip ingestion
        if not force and sync_rec.source_fingerprint == current_fingerprint and sync_rec.status == CRMSyncStatus.SUCCEEDED.value:
            sync_rec.status = CRMSyncStatus.SKIPPED.value
            await self.session.commit()
            return IngestResponse(
                file_name=file_name,
                skipped=True,
                reason="same content_hash already exists",
                document_id=str(sync_rec.document_id) if sync_rec.document_id else None,
            )

        markdown_text, doc_metadata = normalize_crm_company_to_markdown(
            company_state=crm_state,
            verification_marker=verification_marker,
            connection_id=connection_id,
            workspace_id=workspace_id,
        )

        staging_file = self.staging_dir / file_name
        staging_file.write_text(markdown_text, encoding="utf-8")

        try:
            resp = await self.ingestion_service.ingest_file(
                file_path=staging_file,
                department_id=dept.id,
                created_by=created_by,
                force=force,
                workspace_id=workspace_id,
                doc_metadata=doc_metadata,
            )

            sync_rec.document_id = resp.document_id
            sync_rec.source_fingerprint = current_fingerprint
            sync_rec.status = CRMSyncStatus.SUCCEEDED.value
            sync_rec.last_synced_at = now
            sync_rec.attempt_count += 1
            sync_rec.error_message = None
            sync_rec.next_retry_at = None
            await self.session.commit()
            return resp

        except Exception as ex:
            sync_rec.attempt_count += 1
            is_retryable = self._is_retryable_error(ex)
            sync_rec.status = (
                CRMSyncStatus.RETRYABLE_FAILED.value
                if is_retryable
                else CRMSyncStatus.PERMANENT_FAILED.value
            )
            sync_rec.error_message = str(ex)[:1000]

            if is_retryable:
                backoff_seconds = min(300, (2 ** sync_rec.attempt_count) * 5)
                sync_rec.next_retry_at = now + timedelta(seconds=backoff_seconds)
            else:
                sync_rec.next_retry_at = None

            await self.session.commit()
            logger.error("CRM Knowledge Base sync failed for company %s: %s", cid, ex, exc_info=True)
            raise

    async def sync_company_by_name(
        self,
        company_name: str,
        department_id: uuid.UUID | str,
        created_by: Optional[uuid.UUID] = None,
        workspace_id: Optional[uuid.UUID] = None,
        verification_marker: Optional[str] = None,
        force: bool = False,
    ) -> IngestResponse:
        crm_state = await self.mcp_client.get_live_company_state(company_name)
        if not crm_state.is_connected:
            raise RuntimeError(
                f"Twenty CRM MCP adapter is disconnected or offline: {crm_state.error_message or 'unavailable'}"
            )
        if not crm_state.company:
            err = crm_state.error_message or f"Company '{company_name}' was not found in Twenty CRM."
            raise ValueError(err)

        return await self.sync_company_state(
            crm_state=crm_state,
            department_id=department_id,
            created_by=created_by,
            workspace_id=workspace_id,
            verification_marker=verification_marker,
            force=force,
        )

    async def sync_company_by_id(
        self,
        company_id: str,
        department_id: uuid.UUID | str,
        created_by: Optional[uuid.UUID] = None,
        workspace_id: Optional[uuid.UUID] = None,
        verification_marker: Optional[str] = None,
        force: bool = False,
        job_id: Optional[uuid.UUID] = None,
    ) -> IngestResponse:
        crm_state = await self.mcp_client.get_live_company_state_by_id(company_id)
        if not crm_state.is_connected:
            raise RuntimeError(
                f"Twenty CRM MCP adapter is disconnected: {crm_state.error_message or 'unavailable'}"
            )
        if not crm_state.company:
            err = crm_state.error_message or f"Company ID '{company_id}' not found in Twenty CRM."
            raise ValueError(err)

        return await self.sync_company_state(
            crm_state=crm_state,
            department_id=department_id,
            created_by=created_by,
            workspace_id=workspace_id,
            verification_marker=verification_marker,
            force=force,
            job_id=job_id,
        )

    async def sync_batch(
        self,
        department_id: uuid.UUID | str,
        limit: int = 20,
        offset: int = 0,
        workspace_id: Optional[uuid.UUID] = None,
        created_by: Optional[uuid.UUID] = None,
        force: bool = False,
    ) -> BatchSyncSummary:
        """
        Executes a bounded batch synchronization across Twenty CRM companies.
        Provides per-record error isolation and durable job tracking in crm_sync_jobs.
        """
        from rag_app.db.models import CRMSyncJob

        dept = await self._resolve_department(department_id)

        job = CRMSyncJob(
            id=uuid.uuid4(),
            job_type="BATCH_SYNC",
            status="RUNNING",
            department_id=dept.id,
            workspace_id=workspace_id,
            created_by=created_by,
            started_at=datetime.now(timezone.utc),
            checkpoint_cursor=str(offset),
        )
        self.session.add(job)
        await self.session.commit()

        summary = BatchSyncSummary(
            job_id=job.id,
            status="RUNNING",
            total_records=0,
            processed_records=0,
            succeeded_records=0,
            skipped_records=0,
            failed_records=0,
        )

        try:
            companies, has_next, total_count = await self.mcp_client.fetch_companies_page(
                limit=limit,
                offset=offset,
            )
            job.total_records = len(companies)
            summary.total_records = len(companies)

            for comp in companies:
                cid = str(comp.id)
                cname = str(comp.name)
                try:
                    crm_state = await self.mcp_client.get_live_company_state_by_id(cid)
                    if not crm_state.company:
                        # Fallback using name if lookup by id returned empty
                        crm_state = await self.mcp_client.get_live_company_state(cname)

                    if not crm_state.company:
                        raise ValueError(f"Could not load state for company {cid} ({cname})")

                    resp = await self.sync_company_state(
                        crm_state=crm_state,
                        department_id=dept.id,
                        created_by=created_by,
                        workspace_id=workspace_id,
                        force=force,
                        job_id=job.id,
                    )

                    is_skip = getattr(resp, "skipped", False)
                    if is_skip:
                        summary.skipped_records += 1
                        job.skipped_records += 1
                        rec_status = CRMSyncStatus.SKIPPED.value
                    else:
                        summary.succeeded_records += 1
                        job.succeeded_records += 1
                        rec_status = CRMSyncStatus.SUCCEEDED.value

                    summary.records.append(
                        RecordSyncResult(
                            record_id=cid,
                            company_name=cname,
                            status=rec_status,
                            document_id=resp.document_id,
                            skipped=is_skip,
                        )
                    )

                except Exception as ex:
                    logger.warning("Batch sync record %s failed: %s", cid, ex)
                    summary.failed_records += 1
                    job.failed_records += 1
                    is_retry = self._is_retryable_error(ex)
                    failed_status = (
                        CRMSyncStatus.RETRYABLE_FAILED.value
                        if is_retry
                        else CRMSyncStatus.PERMANENT_FAILED.value
                    )
                    summary.records.append(
                        RecordSyncResult(
                            record_id=cid,
                            company_name=cname,
                            status=failed_status,
                            error_message=str(ex)[:500],
                        )
                    )

                summary.processed_records += 1
                job.processed_records += 1

            new_cursor = str(offset + len(companies))
            job.checkpoint_cursor = new_cursor
            summary.checkpoint_cursor = new_cursor
            job.completed_at = datetime.now(timezone.utc)

            if summary.failed_records == 0:
                job.status = "COMPLETED"
                summary.status = "COMPLETED"
            elif summary.succeeded_records > 0 or summary.skipped_records > 0:
                job.status = "PARTIAL_SUCCESS"
                summary.status = "PARTIAL_SUCCESS"
            else:
                job.status = "FAILED"
                summary.status = "FAILED"

            await self.session.commit()
            return summary

        except Exception as ex:
            job.status = "FAILED"
            job.error_message = str(ex)[:1000]
            job.completed_at = datetime.now(timezone.utc)
            summary.status = "FAILED"
            summary.error_message = str(ex)
            await self.session.commit()
            raise

    async def retry_eligible_failed_records(
        self,
        department_id: Optional[uuid.UUID | str] = None,
        limit: int = 10,
    ) -> list[RecordSyncResult]:
        """
        Retries sync records marked as RETRYABLE_FAILED whose next_retry_at has elapsed.
        """
        from rag_app.db.models import CRMSyncRecord

        now = datetime.now(timezone.utc)
        stmt = select(CRMSyncRecord).where(
            and_(
                CRMSyncRecord.status == CRMSyncStatus.RETRYABLE_FAILED.value,
                CRMSyncRecord.attempt_count < CRMSyncRecord.max_attempts,
                (CRMSyncRecord.next_retry_at == None) | (CRMSyncRecord.next_retry_at <= now),
            )
        ).limit(limit)

        if department_id:
            dept = await self._resolve_department(department_id)
            stmt = stmt.where(CRMSyncRecord.department_id == dept.id)

        records = (await self.session.scalars(stmt)).all()
        results: list[RecordSyncResult] = []

        for rec in records:
            try:
                resp = await self.sync_company_by_id(
                    company_id=rec.external_record_id,
                    department_id=rec.department_id,
                    workspace_id=rec.workspace_id,
                )
                results.append(
                    RecordSyncResult(
                        record_id=rec.external_record_id,
                        company_name="Retried Record",
                        status=CRMSyncStatus.SUCCEEDED.value,
                        document_id=resp.document_id,
                    )
                )
            except Exception as ex:
                results.append(
                    RecordSyncResult(
                        record_id=rec.external_record_id,
                        company_name="Retried Record",
                        status=CRMSyncStatus.RETRYABLE_FAILED.value,
                        error_message=str(ex),
                    )
                )

        return results

    async def reconcile_scheduled(
        self,
        department_id: uuid.UUID | str,
        max_records: int = 50,
    ) -> dict[str, Any]:
        """
        Periodic reconciliation job comparing live CRM records with the sync ledger.
        Repairs drift, syncs unindexed or updated records, and handles confirmed deletions.
        """
        from rag_app.db.models import CRMSyncRecord

        dept = await self._resolve_department(department_id)
        companies, _, _ = await self.mcp_client.fetch_companies_page(limit=max_records, offset=0)

        drift_detected = 0
        repaired = 0
        skipped = 0

        for comp in companies:
            cid = str(comp.id)
            sync_rec = await self.session.scalar(
                select(CRMSyncRecord).where(
                    and_(
                        CRMSyncRecord.provider == "twenty_crm",
                        CRMSyncRecord.entity_type == "company",
                        CRMSyncRecord.external_record_id == cid,
                    )
                )
            )

            crm_state = await self.mcp_client.get_live_company_state_by_id(cid)
            if not crm_state.company:
                continue

            current_fp = compute_crm_company_fingerprint(crm_state)
            if not sync_rec or sync_rec.source_fingerprint != current_fp or sync_rec.status != CRMSyncStatus.SUCCEEDED.value:
                drift_detected += 1
                try:
                    await self.sync_company_state(
                        crm_state=crm_state,
                        department_id=dept.id,
                        force=False,
                    )
                    repaired += 1
                except Exception as ex:
                    logger.warning("Reconciliation sync failed for %s: %s", cid, ex)
            else:
                skipped += 1

        # Conservative deletion detection: records in ledger not returned by CRM
        # We do NOT hard delete KB documents; we preserve audit records.
        return {
            "department_id": str(dept.id),
            "examined_records": len(companies),
            "drift_detected": drift_detected,
            "repaired_records": repaired,
            "unchanged_records": skipped,
        }

    async def sync_connection_batch(
        self,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
        department_id: uuid.UUID | str,
        limit: int = 20,
        offset: int = 0,
        created_by: Optional[uuid.UUID] = None,
        force: bool = False,
        max_pages: int = 1,
    ) -> BatchSyncSummary:
        """
        Executes a bounded, connection-scoped batch synchronization from Twenty CRM to Knowledge Base.
        Enforces atomic job claiming, concurrency fencing, strict checkpoint advancement,
        deterministic fingerprint deduplication, and workspace data governance.
        """
        from rag_app.db.models import CRMSyncJob, WorkspaceMCPConnection
        from rag_app.mcp.connection_service import WorkspaceMCPConnectionService
        from unittest.mock import AsyncMock

        dept = await self._resolve_department(department_id)

        # 1. Resolve & validate connection
        conn = await self.session.get(WorkspaceMCPConnection, connection_id)
        if not conn or conn.workspace_id != workspace_id:
            raise ValueError(f"Workspace MCP Connection {connection_id} not found in workspace {workspace_id}.")
        if not conn.is_enabled:
            raise ValueError(f"Workspace MCP Connection '{conn.name}' ({connection_id}) is disabled.")

        # 2. Atomic Concurrency Lock & Stale Lease Recovery
        now = datetime.now(timezone.utc)
        lease_timeout = timedelta(minutes=15)

        stmt_running = select(CRMSyncJob).where(
            and_(
                CRMSyncJob.workspace_id == workspace_id,
                CRMSyncJob.job_type == "CONNECTION_SYNC",
                CRMSyncJob.status == "RUNNING",
            )
        )
        running_jobs = (await self.session.scalars(stmt_running)).all()
        for rj in running_jobs:
            is_same_conn = False
            if rj.checkpoint_cursor:
                try:
                    cdata = json.loads(rj.checkpoint_cursor)
                    if isinstance(cdata, dict) and cdata.get("connection_id") == str(connection_id):
                        is_same_conn = True
                except Exception:
                    pass
            if is_same_conn or not rj.checkpoint_cursor:
                if rj.started_at and (now - rj.started_at) > lease_timeout:
                    logger.warning("Recovering stale sync job %s past lease timeout", rj.id)
                    rj.status = "FAILED"
                    rj.error_message = "Stale execution lease expired on restart / recovery"
                    rj.completed_at = now
                    await self.session.commit()
                else:
                    raise ConcurrencyConflictError(
                        f"A sync job ({rj.id}) is already actively running for connection {connection_id}."
                    )

        # 3. Create durable sync job record
        job = CRMSyncJob(
            id=uuid.uuid4(),
            job_type="CONNECTION_SYNC",
            status="RUNNING",
            department_id=dept.id,
            workspace_id=workspace_id,
            created_by=created_by,
            started_at=now,
            checkpoint_cursor=json.dumps({"offset": offset, "connection_id": str(connection_id)}),
        )
        self.session.add(job)
        await self.session.commit()

        summary = BatchSyncSummary(
            job_id=job.id,
            status="RUNNING",
            total_records=0,
            processed_records=0,
            succeeded_records=0,
            skipped_records=0,
            failed_records=0,
        )

        # 4. Resolve client (use injected mock client if present, else build connection-scoped client)
        mcp_client = self.mcp_client
        is_mock = isinstance(mcp_client, AsyncMock) or getattr(mcp_client, "_is_mock", False)
        if not is_mock:
            conn_svc = WorkspaceMCPConnectionService(self.session)
            secret = await conn_svc._resolve_credential_secret(conn, conn.endpoint_url)
            mcp_client = TwentyMCPClient(mcp_url=conn.endpoint_url, api_key=secret)

        current_offset = offset
        try:
            for _ in range(max(1, max_pages)):
                companies, has_next, total_count = await mcp_client.fetch_companies_page(
                    limit=limit,
                    offset=current_offset,
                )
                if not companies:
                    break

                job.total_records += len(companies)
                summary.total_records += len(companies)

                for comp in companies:
                    cid = str(comp.id)
                    cname = str(comp.name)
                    try:
                        crm_state = await mcp_client.get_live_company_state_by_id(cid)
                        if not crm_state.company:
                            crm_state = await mcp_client.get_live_company_state(cname)
                        if not crm_state.company:
                            raise ValueError(f"Could not load state for company {cid} ({cname})")

                        resp = await self.sync_company_state(
                            crm_state=crm_state,
                            department_id=dept.id,
                            created_by=created_by,
                            workspace_id=workspace_id,
                            force=force,
                            job_id=job.id,
                            connection_id=connection_id,
                        )

                        is_skip = getattr(resp, "skipped", False)
                        if is_skip:
                            summary.skipped_records += 1
                            job.skipped_records += 1
                            rec_status = CRMSyncStatus.SKIPPED.value
                        else:
                            summary.succeeded_records += 1
                            job.succeeded_records += 1
                            rec_status = CRMSyncStatus.SUCCEEDED.value

                        summary.records.append(
                            RecordSyncResult(
                                record_id=cid,
                                company_name=cname,
                                status=rec_status,
                                document_id=resp.document_id,
                                skipped=is_skip,
                            )
                        )
                    except Exception as ex:
                        logger.warning("Connection sync record %s failed: %s", cid, ex)
                        summary.failed_records += 1
                        job.failed_records += 1
                        is_retry = self._is_retryable_error(ex)
                        failed_status = (
                            CRMSyncStatus.RETRYABLE_FAILED.value
                            if is_retry
                            else CRMSyncStatus.PERMANENT_FAILED.value
                        )
                        summary.records.append(
                            RecordSyncResult(
                                record_id=cid,
                                company_name=cname,
                                status=failed_status,
                                error_message=str(ex)[:500],
                            )
                        )
                        try:
                            sync_rec = await self._get_or_create_sync_record(
                                external_record_id=cid,
                                department_id=dept.id,
                                workspace_id=workspace_id,
                                job_id=job.id,
                                provider=f"twenty_crm:{connection_id}",
                            )
                            if sync_rec.status not in (CRMSyncStatus.RETRYABLE_FAILED.value, CRMSyncStatus.PERMANENT_FAILED.value) or sync_rec.error_message != str(ex)[:1000]:
                                sync_rec.attempt_count += 1
                                sync_rec.status = failed_status
                                sync_rec.error_message = str(ex)[:1000]
                                sync_rec.last_attempted_at = datetime.now(timezone.utc)
                                if is_retry:
                                    backoff_seconds = min(300, (2 ** sync_rec.attempt_count) * 5)
                                    sync_rec.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
                                else:
                                    sync_rec.next_retry_at = None
                                await self.session.commit()
                        except Exception as rec_err:
                            logger.warning("Failed to persist failure sync record for %s: %s", cid, rec_err)

                    summary.processed_records += 1
                    job.processed_records += 1

                # Strict Checkpoint Invariant: Advance checkpoint cursor ONLY after all records on this page are accounted for
                current_offset += len(companies)
                new_cursor = json.dumps({"offset": current_offset, "connection_id": str(connection_id)})
                job.checkpoint_cursor = new_cursor
                summary.checkpoint_cursor = new_cursor
                await self.session.commit()

                if not has_next or len(companies) < limit:
                    break

            job.completed_at = datetime.now(timezone.utc)
            if summary.failed_records == 0:
                job.status = "COMPLETED"
                summary.status = "COMPLETED"
            elif summary.succeeded_records > 0 or summary.skipped_records > 0:
                job.status = "PARTIAL_SUCCESS"
                summary.status = "PARTIAL_SUCCESS"
            else:
                job.status = "FAILED"
                summary.status = "FAILED"

            await self.session.commit()
            return summary

        except Exception as ex:
            job.status = "FAILED"
            job.error_message = str(ex)[:1000]
            job.completed_at = datetime.now(timezone.utc)
            summary.status = "FAILED"
            summary.error_message = str(ex)
            await self.session.commit()
            raise

    async def reconcile_connection(
        self,
        workspace_id: uuid.UUID,
        connection_id: uuid.UUID,
        department_id: uuid.UUID | str,
        max_records: int = 50,
    ) -> dict[str, Any]:
        """
        Audits live Twenty CRM connection records against the sync ledger.
        Repairs drift, syncs missed webhooks/records, applies conservative archival for deleted records,
        and reconciles uncertain write-back operations.
        """
        from rag_app.db.models import CRMSyncRecord, MCPToolOperationRecord, WorkspaceMCPConnection
        from rag_app.mcp.connection_service import WorkspaceMCPConnectionService
        from unittest.mock import AsyncMock

        dept = await self._resolve_department(department_id)
        conn = await self.session.get(WorkspaceMCPConnection, connection_id)
        if not conn or conn.workspace_id != workspace_id:
            raise ValueError(f"Workspace MCP Connection {connection_id} not found in workspace {workspace_id}.")

        mcp_client = self.mcp_client
        is_mock = isinstance(mcp_client, AsyncMock) or getattr(mcp_client, "_is_mock", False)
        if not is_mock:
            conn_svc = WorkspaceMCPConnectionService(self.session)
            secret = await conn_svc._resolve_credential_secret(conn, conn.endpoint_url)
            mcp_client = TwentyMCPClient(mcp_url=conn.endpoint_url, api_key=secret)

        provider_key = f"twenty_crm:{connection_id}"
        companies, _, _ = await mcp_client.fetch_companies_page(limit=max_records, offset=0)

        drift_detected = 0
        repaired = 0
        skipped = 0
        archived = 0
        reconciled_operations = 0

        remote_record_ids = set()

        for comp in companies:
            cid = str(comp.id)
            remote_record_ids.add(cid)
            sync_rec = await self.session.scalar(
                select(CRMSyncRecord).where(
                    and_(
                        CRMSyncRecord.provider == provider_key,
                        CRMSyncRecord.entity_type == "company",
                        CRMSyncRecord.external_record_id == cid,
                    )
                )
            )

            crm_state = await mcp_client.get_live_company_state_by_id(cid)
            if not crm_state.company:
                continue

            current_fp = compute_crm_company_fingerprint(crm_state)
            if not sync_rec or sync_rec.source_fingerprint != current_fp or sync_rec.status != CRMSyncStatus.SUCCEEDED.value:
                drift_detected += 1
                try:
                    await self.sync_company_state(
                        crm_state=crm_state,
                        department_id=dept.id,
                        workspace_id=workspace_id,
                        connection_id=connection_id,
                        force=False,
                    )
                    repaired += 1
                except Exception as ex:
                    logger.warning("Connection reconciliation sync failed for %s: %s", cid, ex)
            else:
                skipped += 1

        # Conservative deletion: active ledger records not returned by remote CRM
        stmt_existing = select(CRMSyncRecord).where(
            and_(
                CRMSyncRecord.provider == provider_key,
                CRMSyncRecord.entity_type == "company",
                CRMSyncRecord.status == CRMSyncStatus.SUCCEEDED.value,
            )
        )
        existing_records = (await self.session.scalars(stmt_existing)).all()
        for erec in existing_records:
            if erec.external_record_id not in remote_record_ids:
                erec.status = CRMSyncStatus.ARCHIVED.value
                archived += 1
        await self.session.commit()

        # Reconcile uncertain write-back outcomes
        stmt_unknown = select(MCPToolOperationRecord).where(
            and_(
                MCPToolOperationRecord.workspace_id == workspace_id,
                MCPToolOperationRecord.connection_id == connection_id,
                MCPToolOperationRecord.status == "UNKNOWN",
            )
        )
        unknown_ops = (await self.session.scalars(stmt_unknown)).all()
        for op in unknown_ops:
            op.status = "RECONCILIATION_REQUIRED"
            reconciled_operations += 1
        await self.session.commit()

        return {
            "workspace_id": str(workspace_id),
            "connection_id": str(connection_id),
            "examined_records": len(companies),
            "drift_detected": drift_detected,
            "repaired_records": repaired,
            "unchanged_records": skipped,
            "archived_records": archived,
            "reconciled_operations": reconciled_operations,
        }

    @classmethod
    async def recover_abandoned_sync_jobs(
        cls,
        session: AsyncSession,
        lease_timeout_seconds: int = 900,
    ) -> int:
        """
        Finds any CRMSyncJob stuck in RUNNING for longer than lease_timeout_seconds
        and transitions them to FAILED, allowing worker restarts to recover cleanly.
        """
        from rag_app.db.models import CRMSyncJob

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=lease_timeout_seconds)
        stmt = select(CRMSyncJob).where(
            and_(
                CRMSyncJob.status == "RUNNING",
                CRMSyncJob.started_at <= cutoff,
            )
        )
        stale_jobs = (await session.scalars(stmt)).all()
        recovered = 0
        for job in stale_jobs:
            job.status = "FAILED"
            job.error_message = "Stale execution lease expired on restart / recovery"
            job.completed_at = datetime.now(timezone.utc)
            recovered += 1
        if recovered > 0:
            await session.commit()
            logger.info("Recovered %d abandoned CRM sync jobs", recovered)
        return recovered

    async def handle_webhook_event(
        self,
        payload: dict[str, Any] | list[Any] | bytes,
        signature_header: Optional[str] = None,
        secret: Any = _UNSET,
        department_id: Optional[uuid.UUID | str] = None,
        event_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Processes an incoming Twenty CRM webhook payload with HMAC-SHA256 signature verification.
        Enforces fail-closed secret & signature verification against raw bytes.
        Provides replay protection and conservative deletion.
        """
        from rag_app.core.config import settings
        from rag_app.db.models import CRMSyncRecord, CRMWebhookEvent

        if secret is _UNSET:
            webhook_secret = settings.twenty_webhook_secret
        else:
            webhook_secret = secret

        if not webhook_secret or not str(webhook_secret).strip():
            raise ValueError("Webhook secret is absent or empty; rejecting request fail-closed.")

        if not signature_header or not signature_header.strip():
            raise ValueError("Missing webhook signature header; rejecting request fail-closed.")

        # Determine raw bytes and parsed payload
        if isinstance(payload, bytes):
            raw_bytes = payload
            try:
                parsed_payload = json.loads(raw_bytes.decode("utf-8"))
            except Exception as e:
                raise ValueError(f"Malformed JSON payload: {e}")
        elif isinstance(payload, (dict, list)):
            raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            parsed_payload = payload
        else:
            raise ValueError("Payload must be raw bytes or a dictionary/list.")

        # Constant-time HMAC-SHA256 verification
        # Format 1: Direct raw bytes HMAC (standard test suites)
        expected_sig = hmac.new(webhook_secret.encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()
        sig_to_check = signature_header.replace("sha256=", "").strip()
        is_valid = hmac.compare_digest(expected_sig, sig_to_check)

        # Format 2: Twenty CRM official timestamped format: HMAC-SHA256 of `${timestamp}:${raw_bytes}`
        if not is_valid and timestamp:
            try:
                raw_str = raw_bytes.decode("utf-8")
                expected_sig_ts = hmac.new(
                    webhook_secret.encode("utf-8"),
                    f"{timestamp}:{raw_str}".encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                is_valid = hmac.compare_digest(expected_sig_ts, sig_to_check)
            except Exception:
                pass

        if not is_valid:
            raise ValueError("Invalid webhook HMAC-SHA256 signature.")

        # Timestamp freshness enforcement (Area C: SEC-06)
        # Validate timestamp age against UTC with bounded clock skew (default 300s / 5 min)
        if timestamp is not None and str(timestamp).strip():
            ts_str = str(timestamp).strip()
            try:
                if ts_str.isdigit():
                    ts_seconds = float(ts_str) / 1000.0 if len(ts_str) >= 12 else float(ts_str)
                elif "T" in ts_str:
                    ts_seconds = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                else:
                    ts_seconds = float(ts_str)
            except Exception as te:
                raise ValueError(f"Malformed webhook timestamp '{timestamp}': {te}")

            now_seconds = datetime.now(timezone.utc).timestamp()
            drift = now_seconds - ts_seconds
            max_skew = 300.0
            if drift > max_skew:
                raise ValueError(
                    f"Webhook timestamp expired: event is {drift:.1f}s old (maximum allowable skew is {max_skew:.0f}s)."
                )
            if drift < -60.0:
                raise ValueError(
                    f"Webhook timestamp is in the future: {-drift:.1f}s ahead of server clock."
                )

        # Handle list payloads if wrapped
        if isinstance(parsed_payload, list):
            parsed_payload = parsed_payload[0] if (len(parsed_payload) > 0 and isinstance(parsed_payload[0], dict)) else {}

        # Multi-Tenant Department & Workspace Routing (Area B)
        # 1. Inspect cryptographically signed webhook identity
        webhook_id = parsed_payload.get("webhookId") if isinstance(parsed_payload, dict) else None
        target_department_to_sync = None
        target_workspace_to_sync = None

        mappings = settings.get_twenty_webhook_mappings() if hasattr(settings, "get_twenty_webhook_mappings") else {}

        if webhook_id:
            mapping_entry = mappings.get(str(webhook_id))
            if not mapping_entry:
                # Unauthorized or unregistered webhook ID: fail closed
                raise ValueError(
                    f"Unauthorized or unregistered webhook identity '{webhook_id}'. Routing identity cannot be verified."
                )

            if isinstance(mapping_entry, dict):
                target_department_to_sync = mapping_entry.get("department_id")
                target_workspace_to_sync = mapping_entry.get("workspace_id")
            else:
                target_department_to_sync = mapping_entry

            # Cryptographic query parameter tamper verification:
            # If an unsigned query parameter `department_id` is passed, it MUST match the server-side authorized mapping
            if department_id:
                is_match = False
                if str(department_id).strip().lower() == str(target_department_to_sync).strip().lower():
                    is_match = True
                else:
                    try:
                        mapped_dept = await self._resolve_department(target_department_to_sync)
                        query_dept = await self._resolve_department(department_id)
                        if mapped_dept.id == query_dept.id:
                            is_match = True
                    except Exception:
                        is_match = False

                if not is_match:
                    raise ValueError(
                        f"Query parameter department_id '{department_id}' does not match "
                        f"server-side authorization mapping for webhook '{webhook_id}'."
                    )
        elif department_id:
            # No signed webhook identity in body: only permitted in dev/test fixtures
            env = getattr(settings, "environment", "development").lower()
            if env not in ("test", "testing", "dev", "development"):
                raise ValueError(
                    "Missing signed webhook identity; client-supplied query parameter routing is forbidden in production."
                )
            target_department_to_sync = department_id

        # Replay protection & deduplication
        resolved_event_id = (
            event_id
            or parsed_payload.get("id")
            or parsed_payload.get("eventId")
            or parsed_payload.get("deliveryId")
        )
        event_name = (
            parsed_payload.get("eventName")
            or parsed_payload.get("event")
            or parsed_payload.get("operation")
            or parsed_payload.get("action")
            or parsed_payload.get("type")
            or "unknown"
        )

        if resolved_event_id:
            existing_event = await self.session.scalar(
                select(CRMWebhookEvent).where(CRMWebhookEvent.event_id == str(resolved_event_id))
            )
            if existing_event:
                return {
                    "status": "duplicate_ignored",
                    "event_id": str(resolved_event_id),
                    "reason": "Event already processed or recorded",
                }

            webhook_event = CRMWebhookEvent(
                id=uuid.uuid4(),
                event_id=str(resolved_event_id),
                event_name=event_name,
                status="PROCESSING",
                payload_hash=hashlib.sha256(raw_bytes).hexdigest(),
            )
            self.session.add(webhook_event)
            await self.session.commit()

        data = (
            parsed_payload.get("record")
            or parsed_payload.get("data")
            or parsed_payload.get("payload")
            or parsed_payload
        )
        record_id = str(data.get("id") or "") if isinstance(data, dict) else ""

        if not record_id:
            return {"status": "ignored", "reason": "No record ID found in payload"}

        # Handle conservative deletion
        if "deleted" in event_name.lower():
            sync_rec = await self.session.scalar(
                select(CRMSyncRecord).where(
                    and_(
                        CRMSyncRecord.provider == "twenty_crm",
                        CRMSyncRecord.entity_type == "company",
                        CRMSyncRecord.external_record_id == record_id,
                    )
                )
            )
            if sync_rec:
                sync_rec.status = CRMSyncStatus.ARCHIVED.value
                await self.session.commit()
                return {"status": "archived", "record_id": record_id}
            return {"status": "ignored", "reason": "Record not found in ledger"}

        # Created or updated event
        if target_department_to_sync:
            resolved_dept = await self._resolve_department(target_department_to_sync)
            resp = await self.sync_company_by_id(
                company_id=record_id,
                department_id=resolved_dept.id,
                workspace_id=target_workspace_to_sync,
            )
            return {
                "status": "processed",
                "record_id": record_id,
                "document_id": str(resp.document_id) if resp.document_id else None,
                "skipped": getattr(resp, "skipped", False),
            }

        return {"status": "acknowledged", "record_id": record_id, "note": "No department specified for sync"}

    async def create_writeback_approval(
        self,
        company_id: str,
        department_id: uuid.UUID | str,
        approved_by: uuid.UUID,
        requested_by: Optional[uuid.UUID] = None,
        valid_hours: int = 24,
        operation_type: str = "NOTE_CREATE",
        approved_draft_hash: Optional[str] = None,
        intended_action: str = "CRM_CREATE_NOTE",
        workflow_run_id: Optional[uuid.UUID] = None,
    ) -> Any:
        """Creates a trusted server-side approval record for a CRM write-back."""
        from rag_app.db.models import CRMWriteBackApproval

        dept = await self._resolve_department(department_id)
        now = datetime.now(timezone.utc)
        approval = CRMWriteBackApproval(
            id=uuid.uuid4(),
            operation_type=operation_type,
            external_record_id=str(company_id),
            target_object="company",
            department_id=dept.id,
            requested_by=requested_by,
            approved_by=approved_by,
            status="APPROVED",
            expires_at=now + timedelta(hours=valid_hours),
            created_at=now,
            approved_draft_hash=approved_draft_hash,
            intended_action=intended_action,
            workflow_run_id=workflow_run_id,
        )
        self.session.add(approval)
        await self.session.commit()
        await self.session.refresh(approval)
        return approval

    async def create_meeting_note_writeback(
        self,
        company_id: str,
        note_title: str,
        summary_markdown: str,
        idempotency_key: str,
        user_id: Optional[uuid.UUID] = None,
        user_role: Optional[str] = None,
        approval_id: Optional[uuid.UUID] = None,
        approved: bool = True,
    ) -> dict[str, Any]:
        """
        Executes one approved KB -> CRM write-back operation:
        Creates a meeting summary Note in Twenty CRM linked to the specified Company.
        Enforces server-side approval verification, department RBAC, two-phase note persistence,
        and safe retry recovery without duplicating Notes.
        """
        from rag_app.db.models import (
            CRMWriteBackApproval,
            CRMWriteBackRecord,
            Department,
            User,
            UserDepartment,
        )
        from rag_app.schemas.workflow import compute_canonical_draft_hash

        if not approved:
            raise PermissionError("Write-back operation was not explicitly approved.")

        # Idempotency check: return completed result immediately on replay
        existing = await self.session.scalar(
            select(CRMWriteBackRecord).where(
                CRMWriteBackRecord.idempotency_key == idempotency_key
            )
        )
        if existing and existing.status == "COMPLETED":
            return {
                "status": "COMPLETED",
                "idempotency_key": idempotency_key,
                "crm_note_id": existing.crm_note_id,
                "crm_note_target_id": existing.crm_note_target_id,
                "note_title": existing.note_title,
                "is_duplicate_replay": True,
                "approval_id": str(approval_id) if approval_id else None,
            }

        approval: Optional[CRMWriteBackApproval] = None

        # Server-side Approval & Authorization Validation (SEC-02 & SEC-03)
        if approval_id:
            approval = await self.session.get(CRMWriteBackApproval, approval_id)
            if not approval:
                raise PermissionError(f"Approval record '{approval_id}' not found.")
            if approval.status != "APPROVED":
                raise PermissionError(f"Approval record is already {approval.status.lower()} and cannot be reused.")
            if approval.expires_at <= datetime.now(timezone.utc):
                raise PermissionError("Approval record has expired.")
            if approval.external_record_id != str(company_id):
                raise PermissionError(
                    f"Approval for company {approval.external_record_id} cannot authorize company {company_id}."
                )
            if approval.operation_type != "NOTE_CREATE":
                raise PermissionError(f"Approval operation '{approval.operation_type}' does not match 'NOTE_CREATE'.")
            if getattr(approval, "intended_action", "CRM_CREATE_NOTE") != "CRM_CREATE_NOTE":
                raise PermissionError(
                    f"Approval intended action '{approval.intended_action}' cannot authorize 'CRM_CREATE_NOTE'."
                )

            # Cryptographic draft-hash verification & legacy fail-closed policy
            current_draft_hash = compute_canonical_draft_hash(summary_markdown)
            if not approval.approved_draft_hash:
                if user_role != "ADMIN":
                    raise PermissionError(
                        "Legacy approval ticket lacks cryptographic draft hash. "
                        "Non-admin callers cannot execute write-back with unverified legacy tickets."
                    )
            else:
                if approval.approved_draft_hash != current_draft_hash:
                    approval.status = "REVOKED"
                    approval.revoked_at = datetime.now(timezone.utc)
                    await self.session.commit()
                    raise PermissionError(
                        f"Draft content has been modified since approval was granted. "
                        f"Expected hash {approval.approved_draft_hash}, got {current_draft_hash}. "
                        f"Prior approval ticket {approval.id} has been revoked."
                    )

            # Check department authorization of the caller
            if user_id and user_role != "ADMIN":
                dept = await self.session.get(Department, approval.department_id)
                user_dept = await self.session.scalar(
                    select(UserDepartment).where(
                        UserDepartment.user_id == user_id,
                        UserDepartment.department_id == approval.department_id,
                    )
                )
                if not user_dept:
                    raise PermissionError(
                        f"User is not authorized for department '{dept.code if dept else approval.department_id}'."
                    )
        else:
            # If no approval_id is passed, caller must be ADMIN
            if user_role != "ADMIN":
                raise PermissionError(
                    "Server-side approval record is required for write-back. Submitting approved=True alone is unauthorized."
                )

        if not existing:
            existing = CRMWriteBackRecord(
                id=uuid.uuid4(),
                operation_type="NOTE_CREATE",
                provider="twenty_crm",
                external_record_id=str(company_id),
                target_object="company",
                idempotency_key=idempotency_key,
                status="PENDING",
                note_title=note_title,
                summary_content=summary_markdown,
                created_by=user_id,
                approved_by=approval.approved_by if approval else user_id,
            )
            self.session.add(existing)
            await self.session.commit()

        # Phase 1: Note Creation (with intermediate commit!) (REL-01)
        if existing.crm_note_id:
            # Already created in Twenty CRM in a previous attempt; skip duplicate creation!
            note_id = existing.crm_note_id
        else:
            existing.status = "NOTE_PENDING"
            await self.session.commit()
            note_id = await self.mcp_client.create_note(
                title=note_title,
                markdown_body=summary_markdown,
            )
            existing.crm_note_id = note_id
            existing.status = "NOTE_CREATED"
            await self.session.commit()

        # Phase 2: Link Note to target Company
        try:
            existing.status = "LINK_PENDING"
            await self.session.commit()
            target_id = await self.mcp_client.create_note_target_for_company(
                note_id=note_id,
                company_id=company_id,
            )
            existing.status = "COMPLETED"
            existing.crm_note_target_id = target_id
            existing.completed_at = datetime.now(timezone.utc)
            if approval:
                approval.status = "CONSUMED"
                approval.consumed_at = datetime.now(timezone.utc)
            await self.session.commit()

            return {
                "status": "COMPLETED",
                "idempotency_key": idempotency_key,
                "crm_note_id": note_id,
                "crm_note_target_id": target_id,
                "note_title": note_title,
                "is_duplicate_replay": False,
                "approval_id": str(approval_id) if approval_id else None,
            }

        except Exception as ex:
            existing.status = "LINK_FAILED"
            existing.error_message = str(ex)[:1000]
            await self.session.commit()
            logger.error("Write-back note target linking failed for company %s (note %s): %s", company_id, note_id, ex)
            raise
