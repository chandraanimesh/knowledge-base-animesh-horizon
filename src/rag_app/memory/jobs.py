import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
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
    Idempotent per (source_message_id, trigger_kind) when source_message_id is present.
    Uses PostgreSQL savepoints and unique constraint to safely resolve concurrent insertions.
    """
    stmt = None
    if source_message_id is not None:
        stmt = (
            select(MemoryProcessingJob)
            .where(
                MemoryProcessingJob.source_message_id == source_message_id,
                MemoryProcessingJob.trigger_kind == trigger_kind,
            )
            .limit(1)
        )
        existing = (await session.execute(stmt)).scalars().first()
        if existing:
            return existing

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

    try:
        async with session.begin_nested():
            session.add(job)
            await session.flush()
        return job
    except IntegrityError:
        # Concurrent worker inserted equivalent (source_message_id, trigger_kind).
        # Savepoint rolled back cleanly; outer transaction remains usable.
        if stmt is not None:
            existing = (await session.execute(stmt)).scalars().first()
            if existing:
                return existing
        raise



import asyncio
from rag_app.core.config import get_settings
from rag_app.core.safety import check_sensitive_request
from rag_app.db.session import AsyncSessionLocal


async def execute_memory_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    auto_embed: bool = True,
    extractor: Optional[MemoryExtractor] = None,
    execution_timeout_seconds: float = 15.0,
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

        # Privacy Guard: Never persist secrets or unsuitable sensitive information
        safety = check_sensitive_request(text_content, can_view_pii=False)
        if safety.blocked:
            job.status = JobStatus.SKIPPED_NOT_MEMORABLE.value
            job.error_message = "Skipped sensitive information (PII/secrets)"
            job.source_text_snapshot = "[REDACTED_SENSITIVE_CONTENT]"
            job.processed_at = now
            job.locked_at = None
            job.locked_by = None
            await session.flush()
            return JobStatus.SKIPPED_NOT_MEMORABLE, [], job.error_message

        # 1. Check if slash command
        if text_content.startswith("/"):
            parsed = parse_memory_command(text_content)
            if parsed.is_command:
                if parsed.is_empty:
                    job.status = JobStatus.SKIPPED_NOT_MEMORABLE.value
                    job.error_message = parsed.error_message or "Empty command payload"
                    job.processed_at = now
                    job.locked_at = None
                    job.locked_by = None
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
                if rec is not None:
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

            # Bounded extraction call with strict timeout
            candidates = await asyncio.wait_for(
                mem_extractor.extract_from_conversation(
                    text=text_content,
                    user_id=job.owner_user_id,
                    workspace_id=job.workspace_id or uuid.uuid4(),
                    conversation_id=str(job.conversation_id) if job.conversation_id else None,
                    message_id=str(job.source_message_id) if job.source_message_id else None,
                    existing_memories=existing_mems,
                ),
                timeout=execution_timeout_seconds,
            )

            if not candidates:
                job.status = JobStatus.SKIPPED_NOT_MEMORABLE.value
                job.processed_at = now
                job.updated_at = now
                job.locked_at = None
                job.locked_by = None
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
                if rec is not None:
                    materialized_items.append(rec)

        if not materialized_items:
            job.status = JobStatus.SKIPPED_NOT_MEMORABLE.value
            job.error_message = "Skipped: all extracted candidates were forgotten or empty"
            job.processed_at = datetime.now(timezone.utc)
            job.updated_at = datetime.now(timezone.utc)
            job.locked_at = None
            job.locked_by = None
            await session.flush()
            return JobStatus.SKIPPED_NOT_MEMORABLE, [], job.error_message

        job.status = JobStatus.SUCCEEDED.value
        job.processed_at = datetime.now(timezone.utc)
        job.updated_at = datetime.now(timezone.utc)
        job.locked_at = None
        job.locked_by = None
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
        job.locked_by = None
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


async def claim_next_memory_job(
    session: AsyncSession,
    worker_id: str = "memory-worker-default",
    lock_timeout_seconds: int = 180,
) -> Optional[MemoryProcessingJob]:
    """
    Safely claims the next pending, due retry, or abandoned memory job using FOR UPDATE SKIP LOCKED.
    Respects the ENABLE_CONVERSATIONAL_MEMORY_EXTRACTION feature flag: if disabled,
    only claims explicit commands, keeping conversational jobs safely queued.
    """
    now = datetime.now(timezone.utc)
    lock_cutoff = now - timedelta(seconds=lock_timeout_seconds)
    settings = get_settings()

    allowed_triggers = ["explicit_rem", "command_parser"]
    if getattr(settings, "enable_conversational_memory_extraction", False):
        allowed_triggers.extend(["conversational_turn", "rule_candidate"])

    stmt = (
        select(MemoryProcessingJob)
        .where(
            MemoryProcessingJob.trigger_kind.in_(allowed_triggers),
            or_(
                MemoryProcessingJob.status == JobStatus.PENDING.value,
                and_(
                    MemoryProcessingJob.status == JobStatus.RETRY_PENDING.value,
                    or_(
                        MemoryProcessingJob.next_attempt_at.is_(None),
                        MemoryProcessingJob.next_attempt_at <= now,
                    ),
                ),
                and_(
                    MemoryProcessingJob.status == JobStatus.PROCESSING.value,
                    MemoryProcessingJob.locked_at < lock_cutoff,
                ),
            ),
            MemoryProcessingJob.attempt_count < MemoryProcessingJob.max_attempts,
        )
        .order_by(MemoryProcessingJob.created_at.asc())
        .with_for_update(skip_locked=True)
        .limit(1)
    )

    job = await session.scalar(stmt)
    if not job:
        return None

    job.status = JobStatus.PROCESSING.value
    job.locked_at = now
    job.locked_by = worker_id
    job.updated_at = now
    await session.commit()
    await session.refresh(job)
    return job


async def recover_stale_memory_jobs(
    session: AsyncSession,
    lock_timeout_seconds: int = 180,
) -> int:
    """
    Recovers jobs abandoned in PROCESSING status due to unhandled worker crash or timeout.
    Transitions stale jobs to RETRY_PENDING (or DEAD_LETTER if max_attempts reached).
    """
    now = datetime.now(timezone.utc)
    lock_cutoff = now - timedelta(seconds=lock_timeout_seconds)

    stmt = (
        select(MemoryProcessingJob)
        .where(
            MemoryProcessingJob.status == JobStatus.PROCESSING.value,
            MemoryProcessingJob.locked_at < lock_cutoff,
        )
        .with_for_update(skip_locked=True)
    )
    stale_jobs = (await session.execute(stmt)).scalars().all()
    recovered_count = 0

    for job in stale_jobs:
        job.locked_at = None
        job.locked_by = None
        job.updated_at = now
        if job.attempt_count >= job.max_attempts:
            job.status = JobStatus.DEAD_LETTER.value
        else:
            job.status = JobStatus.RETRY_PENDING.value
            job.next_attempt_at = now
        recovered_count += 1

    if recovered_count > 0:
        await session.commit()
        logger.info(
            "stale_memory_jobs_recovered",
            extra={"event": "stale_memory_jobs_recovered", "count": recovered_count},
        )

    return recovered_count


async def process_pending_memory_jobs(
    worker_id: str = "memory-worker-scheduler",
    max_jobs: int = 10,
    session_factory: Optional[any] = None,
    auto_embed: bool = True,
) -> int:
    """
    Processes a batch of eligible memory extraction jobs safely and idempotently.
    Returns the number of processed jobs.
    """
    sf = session_factory or AsyncSessionLocal
    processed_count = 0

    for _ in range(max_jobs):
        async with sf() as session:
            job = await claim_next_memory_job(session, worker_id=worker_id)
            if not job:
                break

            await execute_memory_job(
                session=session,
                job_id=job.id,
                auto_embed=auto_embed,
            )
            await session.commit()
            processed_count += 1

    return processed_count


async def retry_pending_jobs(
    session: AsyncSession,
    limit: int = 10,
    auto_embed: bool = True,
) -> list[tuple[uuid.UUID, JobStatus]]:
    """
    Finds due retry_pending memory processing jobs and executes them safely.
    Kept for backward compatibility.
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


async def reconcile_missing_conversational_memory_jobs(
    session: AsyncSession,
    lookback_hours: int = 24,
    limit: int = 20,
) -> int:
    """
    Discovers committed user ChatMessages within the lookback window that do not have
    a corresponding MemoryProcessingJob for 'conversational_turn'.
    Enqueues the missing job idempotently with PostgreSQL unique index protection.
    """
    settings = get_settings()
    if not getattr(settings, "enable_conversational_memory_extraction", False):
        return 0

    from rag_app.db.models import ChatMessage, Conversation
    from rag_app.memory.command_parser import detect_memory_candidate_rules
    from rag_app.core.safety import check_sensitive_request

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=lookback_hours)

    stmt = (
        select(ChatMessage, Conversation)
        .join(Conversation, ChatMessage.conversation_id == Conversation.id)
        .outerjoin(
            MemoryProcessingJob,
            and_(
                MemoryProcessingJob.source_message_id == ChatMessage.id,
                MemoryProcessingJob.trigger_kind == "conversational_turn",
            ),
        )
        .where(
            ChatMessage.role == "user",
            ChatMessage.created_at >= since,
            MemoryProcessingJob.id.is_(None),
        )
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    )

    results = (await session.execute(stmt)).all()
    reconciled_count = 0

    for msg, conv in results:
        content = (msg.content or "").strip()
        if content.startswith("/"):
            continue

        is_cand, is_trans, _ = detect_memory_candidate_rules(content)
        safety = check_sensitive_request(content, can_view_pii=False)

        if not is_trans and not safety.blocked:
            try:
                await create_memory_job(
                    session=session,
                    owner_user_id=conv.user_id,
                    source_text=content,
                    trigger_kind="conversational_turn",
                    workspace_id=conv.workspace_id,
                    source_message_id=msg.id,
                    conversation_id=conv.id,
                )
                reconciled_count += 1
            except Exception as err:
                logger.warning(
                    "reconcile_missing_job_failed",
                    extra={"event": "reconcile_missing_job_failed", "msg_id": str(msg.id), "error": str(err)},
                )

    if reconciled_count > 0:
        await session.commit()
        logger.info(
            "reconciled_missing_conversational_jobs",
            extra={"event": "reconciled_missing_conversational_jobs", "count": reconciled_count},
        )

    return reconciled_count
