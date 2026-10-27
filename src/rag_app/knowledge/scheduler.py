"""
Durable Background Scheduler for Orphan-Evidence Reconciliation and Job Recovery.

Periodically executes:
1. Abandoned background job recovery (processes jobs interrupted by server restart/crash).
2. Processing of pending extraction jobs.
3. Idempotent orphan-evidence reconciliation via CompanyNotesService.

Ensures auditable lifecycle transitions and preserves the manual admin endpoint.
Retrieval authorization does NOT depend on this scheduler (real-time pre-context joins
and RBAC filter inaccessible evidence synchronously).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_app.db.session import AsyncSessionLocal
from rag_app.knowledge.company_notes_service import CompanyNotesService
from rag_app.knowledge.worker import process_pending_jobs, recover_abandoned_jobs
from rag_app.knowledge.connection_worker import process_pending_discovery_jobs
from rag_app.memory.jobs import (
    process_pending_memory_jobs,
    recover_stale_memory_jobs,
    reconcile_missing_conversational_memory_jobs,
)

logger = logging.getLogger(__name__)



class DurableReconciliationScheduler:
    """
    In-process durable background runner tied to FastAPI lifespan.
    Periodically executes job recovery, extraction queue polling,
    and orphan evidence reconciliation.
    """

    def __init__(
        self,
        interval_seconds: int = 900,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
    ):
        self.interval_seconds = max(10, interval_seconds)
        self.session_factory = session_factory or AsyncSessionLocal
        self._task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self) -> None:
        """Starts the periodic background scheduler task."""
        if self._is_running:
            return
        self._is_running = True
        self._task = asyncio.create_task(self._loop(), name="reconciliation_scheduler_loop")
        logger.info(
            "scheduler_started",
            extra={
                "event": "durable_reconciliation_scheduler_started",
                "interval_seconds": self.interval_seconds,
            },
        )

    async def stop(self) -> None:
        """Gracefully stops the background scheduler task."""
        self._is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("scheduler_stopped", extra={"event": "durable_reconciliation_scheduler_stopped"})

    async def run_once(self) -> dict[str, int | float]:
        """
        Executes a single cycle of:
        1. Abandoned extraction job recovery.
        2. Pending extraction job processing.
        3. Orphan-evidence reconciliation.
        Returns execution metrics for auditing.
        """
        start = time.perf_counter()
        logger.info("scheduler_cycle_start", extra={"event": "reconciliation_cycle_start"})

        recovered_jobs = 0
        processed_jobs = 0
        reconciled_notes = 0

        sf = self.session_factory
        # 1. Recover abandoned jobs & process pending jobs
        try:
            async with sf() as session:
                recovered_jobs = await recover_abandoned_jobs(session, lock_timeout_seconds=300)
            processed_jobs = await process_pending_jobs(
                worker_id="scheduler-worker",
                max_jobs=10,
                session_factory=sf,
            )
        except Exception as job_err:
            logger.error(
                "scheduler_job_recovery_error",
                extra={"event": "reconciliation_job_recovery_error", "error": str(job_err)},
            )

        # 2. Reconcile orphaned company notes
        try:
            async with sf() as session:
                company_service = CompanyNotesService(session)
                reconciled_notes = await company_service.reconcile_orphaned_notes()
        except Exception as rec_err:
            logger.error(
                "scheduler_reconciliation_error",
                extra={"event": "orphan_notes_reconciliation_error", "error": str(rec_err)},
            )

        # 3. Process pending connection discovery jobs
        discovered_jobs = 0
        try:
            discovered_jobs = await process_pending_discovery_jobs(
                worker_id="scheduler-connection-worker",
                max_jobs=5,
                session_factory=sf,
            )
        except Exception as conn_job_err:
            logger.error(
                "scheduler_connection_jobs_error",
                extra={"event": "connection_jobs_error", "error": str(conn_job_err)},
            )

        # 4. Recover stale memory jobs, reconcile missing conversational jobs, & process pending memory extraction jobs
        recovered_mem_jobs = 0
        reconciled_mem_jobs = 0
        processed_mem_jobs = 0
        try:
            async with sf() as session:
                recovered_mem_jobs = await recover_stale_memory_jobs(session, lock_timeout_seconds=180)
                reconciled_mem_jobs = await reconcile_missing_conversational_memory_jobs(session, lookback_hours=24)
            processed_mem_jobs = await process_pending_memory_jobs(
                worker_id="scheduler-memory-worker",
                max_jobs=10,
                session_factory=sf,
            )
        except Exception as mem_job_err:
            logger.error(
                "scheduler_memory_jobs_error",
                extra={"event": "memory_jobs_error", "error": str(mem_job_err)},
            )

        # 5. Recover abandoned CRM sync jobs and retry eligible failed records
        recovered_crm_sync_jobs = 0
        retried_crm_records = 0
        try:
            from rag_app.crm.sync_service import CRMSyncService
            async with sf() as session:
                recovered_crm_sync_jobs = await CRMSyncService.recover_abandoned_sync_jobs(session, lease_timeout_seconds=900)
                crm_svc = CRMSyncService(session)
                retried = await crm_svc.retry_eligible_failed_records(limit=10)
                retried_crm_records = len(retried)
        except Exception as crm_err:
            logger.error(
                "scheduler_crm_sync_error",
                extra={"event": "crm_sync_scheduler_error", "error": str(crm_err)},
            )

        duration = round(time.perf_counter() - start, 2)
        metrics = {
            "recovered_jobs": recovered_jobs,
            "processed_jobs": processed_jobs,
            "reconciled_notes": reconciled_notes,
            "discovered_jobs": discovered_jobs,
            "recovered_memory_jobs": recovered_mem_jobs,
            "reconciled_memory_jobs": reconciled_mem_jobs,
            "processed_memory_jobs": processed_mem_jobs,
            "recovered_crm_sync_jobs": recovered_crm_sync_jobs,
            "retried_crm_records": retried_crm_records,
            "duration_seconds": duration,
        }


        logger.info(
            "scheduler_cycle_complete",
            extra={
                "event": "reconciliation_cycle_complete",
                **metrics,
            },
        )
        return metrics

    async def _loop(self) -> None:
        """Periodic background execution loop."""
        # Initial run shortly after startup (5 seconds delay to allow DB ready)
        try:
            await asyncio.sleep(5)
            await self.run_once()
        except asyncio.CancelledError:
            return
        except Exception as ex:
            logger.warning("Initial reconciliation cycle error: %s", type(ex).__name__)

        while self._is_running:
            try:
                await asyncio.sleep(self.interval_seconds)
                if not self._is_running:
                    break
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as loop_ex:
                logger.error(
                    "scheduler_loop_exception",
                    extra={"event": "scheduler_loop_exception", "error": str(loop_ex)},
                )


_global_scheduler: Optional[DurableReconciliationScheduler] = None


def get_reconciliation_scheduler(interval_seconds: int = 900) -> DurableReconciliationScheduler:
    """Returns singleton scheduler instance."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = DurableReconciliationScheduler(interval_seconds=interval_seconds)
    return _global_scheduler
