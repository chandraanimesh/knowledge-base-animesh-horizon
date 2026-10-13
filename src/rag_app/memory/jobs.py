import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, update

from rag_app.db.models import MemoryProcessingJob, MemoryItem
from rag_app.memory.schemas import JobStatus, MemoryCandidate, MemoryType
from rag_app.memory.command_parser import parse_memory_command
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.extractor import MemoryExtractor

logger = logging.getLogger(__name__)


async def create_memory_job(
    session: AsyncSession,
    owner_user_id: uuid.UUID,
    source_text: str,
    trigger_kind: str = "rule_candidate",
    workspace_id: Optional[uuid.UUID] = None,
    source_message_id: Optional[uuid.UUID] = None,
    conversation_id: Optional[uuid.UUID] = None,
) -> MemoryProcessingJob:
    """
    Creates a durable memory evaluation job / outbox record.
    The source text is snapshot before interpretation to prevent silent loss.
    """
    job = MemoryProcessingJob(
        id=uuid.uuid4(),
        owner_user_id=owner_user_id,
        workspace_id=workspace_id,
        source_message_id=source_message_id,
        conversation_id=conversation_id,
        source_text_snapshot=source_text,
        trigger_kind=trigger_kind,
        status=JobStatus.PENDING.value,
        attempt_count=0,
        max_attempts=3,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(job)
    await session.flush()
    return job


async def execute_memory_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    auto_embed: bool = True,
    extractor: Optional[MemoryExtractor] = None,
) -> tuple[JobStatus, list[MemoryItem], Optional[str]]:
    """
    Safely and idempotently processes a single memory processing job.
    Never silently loses the captured statement on failure.
    Returns: (job_status, materialized_items, error_or_skip_reason)
    """
    job = await session.get(MemoryProcessingJob, job_id)
    if not job:
        raise ValueError(f"MemoryProcessingJob {job_id} not found")

    # Idempotency guard: do not re-process completed jobs
    if job.status == JobStatus.SUCCEEDED.value:
        # Retrieve already materialized items for this job
        stmt = select(MemoryItem).where(MemoryItem.source_job_id == job.id)
        existing = (await session.execute(stmt)).scalars().all()
        return JobStatus.SUCCEEDED, list(existing), None

    if job.status == JobStatus.SKIPPED_NOT_MEMORABLE.value:
        return JobStatus.SKIPPED_NOT_MEMORABLE, [], job.error_message

    now = datetime.now(timezone.utc)
    job.status = JobStatus.PROCESSING.value
    job.attempt_count += 1
    job.locked_at = now
    job.updated_at = now
    await session.flush()

    try:
        text_content = job.source_text_snapshot.strip()
        materialized_items: list[MemoryItem] = []

        # 1. Check if slash command
        if text_content.startswith("/"):
            parsed = parse_memory_command(text_content)
            if parsed.is_command:
                if parsed.is_empty:
                    job.status = JobStatus.SKIPPED_NOT_MEMORABLE.value
                    job.error_message = parsed.error_message or "Empty command payload"
                    job.processed_at = now
                    await session.flush()
                    return JobStatus.SKIPPED_NOT_MEMORABLE, [], job.error_message

                # Build candidate
                candidate = MemoryCandidate(
                    user_id=job.owner_user_id,
                    workspace_id=job.workspace_id,
                    content=parsed.content,
                    canonical_key=parsed.canonical_key,
                    memory_type=parsed.memory_type,
                    stability_tier=parsed.stability_tier,
                    importance_signals=parsed.importance_signals,
                    subject=parsed.subject,
                    predicate=parsed.predicate,
                    source_quote=parsed.content,
                    source_type="EXPLICIT_USER_STATEMENT",
                    source_job_id=job.id,
                    conversation_id=str(job.conversation_id) if job.conversation_id else None,
                    message_id=str(job.source_message_id) if job.source_message_id else None,
                    extraction_method="command_parser",
                )

                # Look up existing active memory to supersede if same canonical key
                existing_mems = await MemoryMaterializer.get_user_memories(
                    session=session,
                    user_id=job.owner_user_id,
                    workspace_id=job.workspace_id,
                    include_global=True,
                    limit=50,
                )
                for ex in existing_mems:
                    if ex.canonical_key == parsed.canonical_key and ex.status == "ACTIVE":
                        candidate.supersedes_id = ex.id
                        break

                rec = await MemoryMaterializer.materialize_memory(
                    session=session,
                    candidate=candidate,
                    auto_embed=auto_embed,
                )
                materialized_items.append(rec)

        # 2. Ordinary conversational turn
        if not materialized_items:
            mem_extractor = extractor or MemoryExtractor()
            existing_mems = await MemoryMaterializer.get_user_memories(
                session=session,
                user_id=job.owner_user_id,
                workspace_id=job.workspace_id,
                include_global=True,
                limit=50,
            )

            candidates = await mem_extractor.extract_from_conversation(
                text=text_content,
                user_id=job.owner_user_id,
                workspace_id=job.workspace_id or uuid.uuid4(),  # fallback uuid for extraction validator
                conversation_id=str(job.conversation_id) if job.conversation_id else None,
                message_id=str(job.source_message_id) if job.source_message_id else None,
                existing_memories=existing_mems,
            )

            if not candidates:
                job.status = JobStatus.SKIPPED_NOT_MEMORABLE.value
                job.processed_at = now
                job.updated_at = now
                await session.flush()
                return JobStatus.SKIPPED_NOT_MEMORABLE, [], "No memorable candidates extracted"

            for cand in candidates:
                cand.source_job_id = job.id
                cand.workspace_id = job.workspace_id
                rec = await MemoryMaterializer.materialize_memory(
                    session=session,
                    candidate=cand,
                    auto_embed=auto_embed,
                )
                materialized_items.append(rec)

        job.status = JobStatus.SUCCEEDED.value
        job.processed_at = datetime.now(timezone.utc)
        job.updated_at = datetime.now(timezone.utc)
        job.locked_at = None
        await session.flush()
        return JobStatus.SUCCEEDED, materialized_items, None

    except Exception as exc:
        logger.error(
            "memory_job_processing_failed",
            extra={
                "event": "memory_job_processing_failed",
                "job_id": str(job.id),
                "attempt": job.attempt_count,
                "error": str(exc),
            },
        )
        job.last_error_code = exc.__class__.__name__
        job.error_message = str(exc)
        job.locked_at = None
        job.updated_at = datetime.now(timezone.utc)

        if job.attempt_count >= job.max_attempts:
            job.status = JobStatus.DEAD_LETTER.value
            final_status = JobStatus.DEAD_LETTER
        else:
            job.status = JobStatus.RETRY_PENDING.value
            backoff_secs = 2 ** job.attempt_count
            job.next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_secs)
            final_status = JobStatus.RETRY_PENDING

        await session.flush()
        return final_status, [], str(exc)


async def retry_pending_jobs(
    session: AsyncSession,
    limit: int = 10,
    auto_embed: bool = True,
) -> list[tuple[uuid.UUID, JobStatus]]:
    """
    Finds due retry_pending memory processing jobs and executes them safely.
    """
    now = datetime.now(timezone.utc)
    stmt = (
        select(MemoryProcessingJob)
        .where(
            MemoryProcessingJob.status == JobStatus.RETRY_PENDING.value,
            or_(
                MemoryProcessingJob.next_attempt_at.is_(None),
                MemoryProcessingJob.next_attempt_at <= now,
            ),
        )
        .order_by(MemoryProcessingJob.next_attempt_at.asc())
        .limit(limit)
    )
    jobs = (await session.execute(stmt)).scalars().all()
    results: list[tuple[uuid.UUID, JobStatus]] = []

    for job in jobs:
        status, _, _ = await execute_memory_job(
            session=session,
            job_id=job.id,
            auto_embed=auto_embed,
        )
        results.append((job.id, status))

    return results
