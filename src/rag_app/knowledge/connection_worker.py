import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_app.db.models import ConnectionDiscoveryJob, User
from rag_app.db.session import AsyncSessionLocal
from rag_app.knowledge.connection_service import ConnectionDiscoveryService

logger = logging.getLogger(__name__)


async def enqueue_discovery_job(
    session: AsyncSession,
    source_item_id: uuid.UUID,
    source_item_type: str,
    requesting_user_id: uuid.UUID,
    target_source_hash: str,
    max_attempts: int = 3,
) -> ConnectionDiscoveryJob:
    """
    Idempotently enqueues a connection discovery job.
    Deduplicates if an identical pending or running job exists.
    """
    stmt = (
        select(ConnectionDiscoveryJob)
        .where(
            ConnectionDiscoveryJob.source_item_id == source_item_id,
            ConnectionDiscoveryJob.target_source_hash == target_source_hash,
            ConnectionDiscoveryJob.status.in_(("PENDING", "RUNNING")),
        )
        .order_by(ConnectionDiscoveryJob.created_at.desc())
        .limit(1)
    )
    existing = await session.scalar(stmt)
    if existing:
        logger.info(
            "discovery_job_deduplicated",
            extra={
                "source_item_id": str(source_item_id),
                "job_id": str(existing.id),
                "status": existing.status,
            },
        )
        return existing

    job = ConnectionDiscoveryJob(
        source_item_id=source_item_id,
        source_item_type=source_item_type.upper(),
        target_source_hash=target_source_hash,
        requesting_user_id=requesting_user_id,
        status="PENDING",
        attempt_count=0,
        max_attempts=max_attempts,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    logger.info(
        "discovery_job_enqueued",
        extra={
            "source_item_id": str(source_item_id),
            "job_id": str(job.id),
        },
    )
    return job


async def claim_next_discovery_job(
    session: AsyncSession,
    worker_id: str = "connection-worker-default",
    lock_timeout_seconds: int = 300,
) -> Optional[ConnectionDiscoveryJob]:
    """
    Claims the next available job using SELECT ... FOR UPDATE SKIP LOCKED.
    Safely reclaims abandoned jobs where locked_at > lock_timeout_seconds.
    """
    now = datetime.now(timezone.utc)
    lock_cutoff = now - timedelta(seconds=lock_timeout_seconds)

    stmt = (
        select(ConnectionDiscoveryJob)
        .where(
            or_(
                ConnectionDiscoveryJob.status == "PENDING",
                and_(
                    ConnectionDiscoveryJob.status == "RUNNING",
                    ConnectionDiscoveryJob.locked_at < lock_cutoff,
                ),
            )
        )
        .order_by(ConnectionDiscoveryJob.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = await session.scalar(stmt)
    if not job:
        return None

    job.status = "RUNNING"
    job.locked_at = now
    job.locked_by = worker_id
    job.attempt_count += 1
    job.updated_at = now
    await session.commit()
    await session.refresh(job)
    return job


async def execute_discovery_job(
    session: AsyncSession,
    job: ConnectionDiscoveryJob,
) -> bool:
    """
    Executes discovery for a claimed job with stale-result protection.
    Returns True on success, False on failure.
    """
    now = datetime.now(timezone.utc)
    try:
        user = await session.get(User, job.requesting_user_id)
        if not user or not user.is_active:
            raise ValueError(f"Requesting user {job.requesting_user_id} not found or inactive")

        service = ConnectionDiscoveryService(session)

        # 1. Verify source still matches job target_source_hash (Stale result check)
        _, current_hash, _, _ = await service.resolve_source_and_version(
            source_id=job.source_item_id,
            source_type=job.source_item_type,
            user=user,
        )
        if current_hash != job.target_source_hash:
            logger.warning(
                "discovery_job_stale_source_dropped",
                extra={
                    "job_id": str(job.id),
                    "expected_hash": job.target_source_hash,
                    "current_hash": current_hash,
                },
            )
            job.status = "COMPLETED"
            job.error_message = "Source mutated before job completion; discarded as stale"
            job.completed_at = now
            job.locked_at = None
            job.locked_by = None
            await session.commit()
            return True

        # 2. Run discovery
        await service.generate_candidates_for_source(
            source_id=job.source_item_id,
            source_type=job.source_item_type,
            user=user,
            max_candidates=15,
        )

        job.status = "COMPLETED"
        job.error_message = None
        job.completed_at = now
        job.locked_at = None
        job.locked_by = None
        job.updated_at = now
        await session.commit()
        return True

    except Exception as exc:
        logger.error(
            "discovery_job_failed",
            extra={
                "job_id": str(job.id),
                "attempt": job.attempt_count,
                "error": str(exc),
            },
            exc_info=True,
        )
        job.updated_at = now
        job.locked_at = None
        job.locked_by = None
        job.error_message = str(exc)

        if job.attempt_count >= job.max_attempts:
            job.status = "FAILED"
        else:
            job.status = "PENDING"  # Eligible for retry

        await session.commit()
        return False


async def process_pending_discovery_jobs(
    worker_id: str = "connection-worker-scheduler",
    max_jobs: int = 5,
    session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
) -> int:
    """Batch loop processing pending discovery jobs."""
    sf = session_factory or AsyncSessionLocal
    processed_count = 0

    for _ in range(max_jobs):
        async with sf() as session:
            job = await claim_next_discovery_job(session, worker_id=worker_id)
            if not job:
                break
            await execute_discovery_job(session, job)
            processed_count += 1

    return processed_count
