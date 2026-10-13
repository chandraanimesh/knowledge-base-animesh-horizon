"""
Durable Background Processing Worker for Document Knowledge Extraction.

Backed by PostgreSQL (KnowledgeExtractionJob model):
- Jobs survive application server crashes and restarts.
- Concurrency-safe job claiming via SELECT ... FOR UPDATE SKIP LOCKED.
- Automatic recovery of abandoned jobs (locked_at > lock_timeout_seconds).
- Configurable retries (max_attempts).
- Idempotent processing via PostgreSQL ON CONFLICT DO UPDATE constraints.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_app.db.models import Document, DocumentChunk, KnowledgeExtractionJob
from rag_app.db.session import AsyncSessionLocal
from rag_app.knowledge.extractor import KnowledgeExtractorService
from rag_app.knowledge.materializer import materialize_knowledge

logger = logging.getLogger(__name__)


async def enqueue_extraction_job(
    session: AsyncSession,
    document_id: uuid.UUID,
    max_attempts: int = 3,
) -> KnowledgeExtractionJob:
    """
    Idempotently enqueues a knowledge extraction job for a document.
    If a PENDING or RUNNING job already exists, returns the existing job.
    If a COMPLETED or FAILED job exists, creates a fresh PENDING job or resets it.
    """
    stmt = (
        select(KnowledgeExtractionJob)
        .where(KnowledgeExtractionJob.document_id == document_id)
        .order_by(KnowledgeExtractionJob.created_at.desc())
        .limit(1)
    )
    existing = await session.scalar(stmt)

    if existing and existing.status in ("PENDING", "RUNNING"):
        logger.info(
            "job_deduplicated",
            extra={
                "event": "knowledge_extraction_job_already_enqueued",
                "document_id": str(document_id),
                "job_id": str(existing.id),
                "status": existing.status,
            },
        )
        return existing

    job = KnowledgeExtractionJob(
        document_id=document_id,
        status="PENDING",
        attempts=0,
        max_attempts=max_attempts,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    logger.info(
        "job_enqueued",
        extra={
            "event": "knowledge_extraction_job_enqueued",
            "document_id": str(document_id),
            "job_id": str(job.id),
        },
    )
    return job


async def claim_next_extraction_job(
    session: AsyncSession,
    worker_id: str = "worker-default",
    lock_timeout_seconds: int = 300,
) -> Optional[KnowledgeExtractionJob]:
    """
    Safely claims the next pending or abandoned job using FOR UPDATE SKIP LOCKED.
    """
    now = datetime.now(timezone.utc)
    lock_cutoff = now - timedelta(seconds=lock_timeout_seconds)

    stmt = (
        select(KnowledgeExtractionJob)
        .where(
            or_(
                KnowledgeExtractionJob.status == "PENDING",
                # Abandoned job recovery: was RUNNING but worker crashed/timed out
                (
                    (KnowledgeExtractionJob.status == "RUNNING")
                    & (KnowledgeExtractionJob.locked_at < lock_cutoff)
                ),
            ),
            KnowledgeExtractionJob.attempts < KnowledgeExtractionJob.max_attempts,
        )
        .order_by(KnowledgeExtractionJob.created_at.asc())
        .with_for_update(skip_locked=True)
        .limit(1)
    )

    job = await session.scalar(stmt)
    if not job:
        return None

    job.status = "RUNNING"
    job.locked_at = now
    job.locked_by = worker_id
    job.attempts += 1
    job.updated_at = now

    await session.commit()
    await session.refresh(job)

    logger.info(
        "job_claimed",
        extra={
            "event": "knowledge_extraction_job_claimed",
            "job_id": str(job.id),
            "document_id": str(job.document_id),
            "attempt": job.attempts,
            "worker_id": worker_id,
        },
    )
    return job


async def execute_claimed_job(
    session: AsyncSession,
    job: KnowledgeExtractionJob,
) -> bool:
    """
    Executes knowledge extraction for the claimed job and transitions its status.
    Idempotent: Re-executing over existing chunks safely updates knowledge items.
    """
    start_time = time.perf_counter()
    job_id = job.id
    document_id = job.document_id
    job_attempts = job.attempts
    job_max_attempts = job.max_attempts

    doc = await session.scalar(select(Document).where(Document.id == document_id))
    if not doc:
        job_to_fail = await session.get(KnowledgeExtractionJob, job_id)
        if job_to_fail:
            job_to_fail.status = "FAILED"
            job_to_fail.error_message = f"Document {document_id} not found."
            job_to_fail.completed_at = datetime.now(timezone.utc)
            await session.commit()
        return False

    chunk_stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    chunks = (await session.scalars(chunk_stmt)).all()

    if not chunks:
        job_to_complete = await session.get(KnowledgeExtractionJob, job_id)
        if job_to_complete:
            job_to_complete.status = "COMPLETED"
            job_to_complete.completed_at = datetime.now(timezone.utc)
            await session.commit()
        return True

    extractor = KnowledgeExtractorService()
    success_count = 0
    failure_count = 0
    candidate_count = 0

    try:
        for chunk in chunks:
            try:
                candidates = await extractor.extract_chunk(chunk, doc)
                if candidates:
                    for candidate in candidates:
                        await materialize_knowledge(session, candidate)
                        candidate_count += 1
                    await session.commit()
                success_count += 1
            except Exception as chunk_err:
                await session.rollback()
                logger.error(
                    "chunk_failure",
                    extra={
                        "event": "knowledge_extraction_chunk_failure",
                        "document_id": str(document_id),
                        "chunk_id": str(chunk.id),
                        "error": str(chunk_err),
                    },
                )
                failure_count += 1

        duration = round(time.perf_counter() - start_time, 2)
        job_to_complete = await session.get(KnowledgeExtractionJob, job_id)
        if job_to_complete:
            job_to_complete.status = "COMPLETED"
            job_to_complete.error_message = None
            job_to_complete.completed_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "job_complete",
            extra={
                "event": "knowledge_extraction_job_complete",
                "job_id": str(job_id),
                "document_id": str(document_id),
                "total_chunks": len(chunks),
                "success_count": success_count,
                "failure_count": failure_count,
                "candidate_count": candidate_count,
                "duration_seconds": duration,
            },
        )
        return True

    except Exception as job_err:
        await session.rollback()
        duration = round(time.perf_counter() - start_time, 2)
        logger.error(
            "job_execution_error",
            extra={
                "event": "knowledge_extraction_job_error",
                "job_id": str(job_id),
                "document_id": str(document_id),
                "attempt": job_attempts,
                "error": str(job_err),
            },
        )

        # Retry logic: if max attempts not yet reached, put back to PENDING
        job_to_retry = await session.get(KnowledgeExtractionJob, job_id)
        if job_to_retry:
            if job_attempts < job_max_attempts:
                job_to_retry.status = "PENDING"
                job_to_retry.error_message = f"Attempt {job_attempts} failed: {str(job_err)}"
                job_to_retry.locked_at = None
                job_to_retry.locked_by = None
            else:
                job_to_retry.status = "FAILED"
                job_to_retry.error_message = f"Exhausted {job_max_attempts} attempts: {str(job_err)}"
                job_to_retry.completed_at = datetime.now(timezone.utc)
            await session.commit()
        return False


async def recover_abandoned_jobs(
    session: AsyncSession,
    lock_timeout_seconds: int = 300,
) -> int:
    """
    Recovers jobs left in 'RUNNING' state due to an unexpected process restart or crash.
    Transitions abandoned jobs back to 'PENDING' so they can be processed by available workers.
    """
    now = datetime.now(timezone.utc)
    lock_cutoff = now - timedelta(seconds=lock_timeout_seconds)

    stmt = (
        select(KnowledgeExtractionJob)
        .where(
            KnowledgeExtractionJob.status == "RUNNING",
            KnowledgeExtractionJob.locked_at < lock_cutoff,
            KnowledgeExtractionJob.attempts < KnowledgeExtractionJob.max_attempts,
        )
        .with_for_update()
    )
    abandoned_jobs = (await session.scalars(stmt)).all()

    for job in abandoned_jobs:
        job.status = "PENDING"
        job.locked_at = None
        job.locked_by = None
        job.updated_at = now

    if abandoned_jobs:
        await session.commit()
        logger.info(
            "abandoned_jobs_recovered",
            extra={
                "event": "knowledge_extraction_abandoned_jobs_recovered",
                "count": len(abandoned_jobs),
            },
        )

    return len(abandoned_jobs)


async def process_pending_jobs(
    worker_id: str = "worker-poll",
    max_jobs: int = 10,
    session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
) -> int:
    """
    Claims and processes up to max_jobs pending or recovered extraction jobs.
    Useful for periodic background polling and test verification.
    """
    sf = session_factory or AsyncSessionLocal
    processed = 0
    for _ in range(max_jobs):
        async with sf() as session:
            job = await claim_next_extraction_job(session=session, worker_id=worker_id)
            if not job:
                break
            await execute_claimed_job(session=session, job=job)
            processed += 1
    return processed


async def run_document_knowledge_extraction(document_id: uuid.UUID) -> None:
    """
    Durable entrypoint compatible with FastAPI BackgroundTasks and direct invocation.
    1. Enqueues job in PostgreSQL (durable queue).
    2. Immediately claims and executes it.
    If server restarts midway, job remains in PostgreSQL and is safely reclaimed by the scheduler.
    """
    async with AsyncSessionLocal() as session:
        job = await enqueue_extraction_job(session=session, document_id=document_id)

    async with AsyncSessionLocal() as session:
        # Fetch and claim this specific job or next available
        claimed_job = await session.get(KnowledgeExtractionJob, job.id)
        if claimed_job and claimed_job.status == "PENDING":
            claimed_job.status = "RUNNING"
            claimed_job.locked_at = datetime.now(timezone.utc)
            claimed_job.locked_by = "immediate-worker"
            claimed_job.attempts += 1
            await session.commit()
            fresh_job = await session.get(KnowledgeExtractionJob, job.id)
            if fresh_job:
                await execute_claimed_job(session=session, job=fresh_job)
