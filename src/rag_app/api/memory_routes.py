import logging
from typing import Annotated, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.session import get_db_session
from rag_app.db.models import User
from rag_app.api.dependencies.auth import get_current_user
from rag_app.conversation.service import ConversationService
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.retrieval import MemoryRetrievalService
from rag_app.memory.privacy_service import MemoryPrivacyService
from rag_app.memory.privacy_operations import MemoryPrivacyOperations
from rag_app.memory.retention_service import MemoryRetentionService
from rag_app.memory.schemas import (
    MemoryCandidate,
    MemoryExportResponse,
    MemoryPrivacyStatusResponse,
    MemoryRecord,
    MemoryRetrievalResponse,
    MemoryType,
    MemoryStatus,
    PrivacyAuditEventRecord,
    PrivacyPolicyPayload,
    PrivacyPolicyRecord,
    ProfileScrubPayload,
    ScopeType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


class CreateMemoryPayload(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)
    memory_type: MemoryType = MemoryType.DYNAMIC
    canonical_key: Optional[str] = None
    workspace_id: Optional[UUID] = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    structured_data: Optional[dict[str, Any]] = None


class UpdateMemoryPayload(BaseModel):
    content: Optional[str] = Field(None, min_length=1, max_length=2000)
    structured_data: Optional[dict[str, Any]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class SearchMemoryPayload(BaseModel):
    query: str = Field(..., min_length=1)
    workspace_id: Optional[UUID] = None
    limit: int = Field(10, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


@router.get("", response_model=list[MemoryRecord])
async def list_memories(
    session: DbSession,
    user: CurrentUser,
    workspace_id: Optional[UUID] = Query(None, description="Workspace ID filter"),
    include_global: bool = Query(True, description="Whether to include global memories"),
    limit: int = Query(50, ge=1, le=200),
) -> list[MemoryRecord]:
    """
    List active memories for the authenticated user.
    If workspace_id is provided, verifies user membership/access before returning.
    """
    if workspace_id is not None:
        conv_service = ConversationService(session)
        await conv_service._validate_workspace_access(user, workspace_id)

    records = await MemoryMaterializer.get_user_memories(
        session=session,
        user_id=user.id,
        workspace_id=workspace_id,
        limit=limit,
        include_global=include_global,
    )
    return records


@router.post("", response_model=MemoryRecord, status_code=status.HTTP_201_CREATED)
async def create_memory(
    session: DbSession,
    user: CurrentUser,
    payload: CreateMemoryPayload,
) -> MemoryRecord:
    """
    Create a new memory from explicit user input.
    Enforces user ownership and workspace access.
    """
    if payload.workspace_id is not None:
        conv_service = ConversationService(session)
        await conv_service._validate_workspace_access(user, payload.workspace_id)

    candidate = MemoryCandidate(
        user_id=user.id,
        workspace_id=payload.workspace_id,
        memory_type=payload.memory_type,
        content=payload.content.strip(),
        canonical_key=payload.canonical_key,
        confidence=payload.confidence,
        structured_data=payload.structured_data,
        source_type="EXPLICIT_USER_STATEMENT",
    )

    created = await MemoryMaterializer.materialize_memory(session, candidate)
    return created


@router.patch("/{memory_id}", response_model=MemoryRecord)
async def edit_memory(
    memory_id: UUID,
    payload: UpdateMemoryPayload,
    session: DbSession,
    user: CurrentUser,
) -> MemoryRecord:
    """
    Edit a selected active memory owned by the authenticated user.
    """
    updated = await MemoryMaterializer.update_memory(
        session=session,
        memory_id=memory_id,
        user_id=user.id,
        content=payload.content,
        structured_data=payload.structured_data,
        confidence=payload.confidence,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or unauthorized.",
        )
    return updated


@router.delete("/{memory_id}", status_code=status.HTTP_200_OK)
async def delete_memory(
    memory_id: UUID,
    session: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """
    Forget and purge a selected memory owned by the authenticated user.
    Removes embedding, registers anti-resurrection tombstone, cancels pending jobs,
    and recompiles living user profile.
    """
    try:
        success = await MemoryPrivacyOperations.forget_memory(
            session=session,
            memory_id=memory_id,
            user_id=user.id,
        )
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found or unauthorized.",
        )
    await session.commit()
    return {"status": "PURGED", "memory_id": str(memory_id)}


@router.delete("", status_code=status.HTTP_200_OK)
async def delete_all_personal_memories(
    session: DbSession,
    user: CurrentUser,
    workspace_id: Optional[UUID] = Query(None, description="Optional workspace scope"),
) -> dict[str, Any]:
    """
    Delete all personal memories for the authenticated user.
    Guaranteed not to affect shared organizational documents, other users, or CRM records.
    """
    try:
        deleted_count = await MemoryPrivacyOperations.delete_all_personal_memories(
            session=session,
            user_id=user.id,
            workspace_id=workspace_id,
        )
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )

    await session.commit()
    return {"status": "DELETED_ALL", "deleted_count": deleted_count}


@router.get("/privacy/policy", response_model=PrivacyPolicyRecord)
async def get_effective_privacy_policy(
    session: DbSession,
    user: CurrentUser,
    workspace_id: Optional[UUID] = Query(None),
) -> PrivacyPolicyRecord:
    """
    Returns the resolved effective privacy policy for the user, accounting for
    tenant defaults, workspace policies, and user overrides under the Strictest Rule Invariant.
    """
    return await MemoryPrivacyService.get_effective_policy(
        session=session,
        user_id=user.id,
        workspace_id=workspace_id,
    )


@router.put("/privacy/policy", response_model=PrivacyPolicyRecord)
async def update_privacy_policy(
    payload: PrivacyPolicyPayload,
    session: DbSession,
    user: CurrentUser,
) -> PrivacyPolicyRecord:
    """
    Configures a privacy policy for USER, WORKSPACE, or TENANT scope.
    Enforces RBAC: Admins for tenant/workspace, User for their own settings.
    """
    try:
        await MemoryPrivacyService.set_policy(
            session=session,
            payload=payload,
            actor=user,
        )
        await session.commit()
        return await MemoryPrivacyService.get_effective_policy(
            session=session,
            user_id=payload.scope_id if payload.scope_type == ScopeType.USER and payload.scope_id else user.id,
            workspace_id=payload.scope_id if payload.scope_type == ScopeType.WORKSPACE else None,
        )
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )


@router.post("/privacy/pause", response_model=PrivacyPolicyRecord)
async def pause_memory_collection(
    session: DbSession,
    user: CurrentUser,
) -> PrivacyPolicyRecord:
    """Pauses memory collection for the current user."""
    policy = await MemoryPrivacyService.pause_memory_collection(session, user.id)
    await session.commit()
    return policy


@router.post("/privacy/resume", response_model=PrivacyPolicyRecord)
async def resume_memory_collection(
    session: DbSession,
    user: CurrentUser,
) -> PrivacyPolicyRecord:
    """Resumes memory collection for the current user."""
    policy = await MemoryPrivacyService.resume_memory_collection(session, user.id)
    await session.commit()
    return policy


@router.get("/privacy/status", response_model=MemoryPrivacyStatusResponse)
async def get_privacy_status(
    session: DbSession,
    user: CurrentUser,
    workspace_id: Optional[UUID] = Query(None),
) -> MemoryPrivacyStatusResponse:
    """
    Provides a comprehensive privacy overview: collection state (ENABLED/PAUSED/DISABLED),
    effective policy, active/expired/purged memory counts, and living profile summary.
    """
    from sqlalchemy import func
    from rag_app.db.models import MemoryItem

    policy = await MemoryPrivacyService.get_effective_policy(session, user.id, workspace_id)

    # State determination
    if not policy.memory_collection_enabled:
        collection_state = "PAUSED" if policy.user_can_pause else "DISABLED"
    else:
        collection_state = "ENABLED"

    # Memory counts
    base_q = select(func.count()).where(MemoryItem.user_id == user.id)
    if workspace_id is not None:
        base_q = base_q.where(or_(MemoryItem.workspace_id == workspace_id, MemoryItem.workspace_id.is_(None)))

    total_active = (await session.execute(base_q.where(MemoryItem.status == "ACTIVE"))).scalar() or 0
    total_expired = (await session.execute(base_q.where(MemoryItem.status == "EXPIRED"))).scalar() or 0
    total_purged = (await session.execute(base_q.where(MemoryItem.status == "PURGED"))).scalar() or 0

    return MemoryPrivacyStatusResponse(
        user_id=user.id,
        collection_state=collection_state,
        effective_policy=policy,
        total_active_memories=total_active,
        total_expired_memories=total_expired,
        total_purged_memories=total_purged,
        living_profile_summary=user.profile_json or {},
        last_dreaming_run=user.profile_json.get("last_recompiled_at") if user.profile_json else None,
    )


@router.post("/profile/scrub", response_model=dict[str, Any])
async def scrub_profile_attribute(
    payload: ProfileScrubPayload,
    session: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """
    Granular commercial profile scrubbing:
    Removes the specified attribute (e.g. role, communication_style, database),
    purges its supporting memories, registers anti-resurrection tombstones,
    and recompiles the profile without sensitive data.
    """
    try:
        res = await MemoryPrivacyOperations.scrub_user_profile(
            session=session,
            user_id=user.id,
            section=payload.section,
            key=payload.key,
        )
        await session.commit()
        return res
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )


@router.get("/export", response_model=MemoryExportResponse)
async def export_my_memories(
    session: DbSession,
    user: CurrentUser,
    workspace_id: Optional[UUID] = Query(None),
) -> MemoryExportResponse:
    """
    Exports all memories owned by the user in structured JSON format.
    Guaranteed zero leakage of other tenants' or users' data.
    """
    try:
        res = await MemoryPrivacyOperations.export_user_memories(
            session=session,
            user_id=user.id,
            workspace_id=workspace_id,
        )
        return res
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )


@router.post("/privacy/sweep", response_model=dict[str, Any])
async def trigger_retention_sweep(
    session: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """
    Runs an idempotent retention lifecycle sweep.
    Requires ADMIN role.
    """
    if user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can trigger system retention sweeps",
        )
    return await MemoryRetentionService.run_retention_sweep(session)


@router.get("/privacy/audit-logs", response_model=list[PrivacyAuditEventRecord])
async def list_privacy_audit_logs(
    session: DbSession,
    user: CurrentUser,
    limit: int = Query(50, ge=1, le=200),
) -> list[PrivacyAuditEventRecord]:
    """
    Lists privacy audit events for the authenticated user (or tenant if admin).
    Strictly free of sensitive personal data or memory content.
    """
    from rag_app.db.models import MemoryPrivacyAuditEvent
    stmt = (
        select(MemoryPrivacyAuditEvent)
        .where(MemoryPrivacyAuditEvent.user_id == user.id)
        .order_by(MemoryPrivacyAuditEvent.created_at.desc())
        .limit(limit)
    )
    records = (await session.execute(stmt)).scalars().all()
    return list(records)


@router.post("/search", response_model=MemoryRetrievalResponse)
async def search_memories(
    payload: SearchMemoryPayload,
    session: DbSession,
    user: CurrentUser,
) -> MemoryRetrievalResponse:
    """
    Retrieve relevant memories within the authorized scope.
    """
    if payload.workspace_id is not None:
        conv_service = ConversationService(session)
        await conv_service._validate_workspace_access(user, payload.workspace_id)

    response = await MemoryRetrievalService.retrieve(
        session=session,
        user_id=user.id,
        workspace_id=payload.workspace_id,
        query=payload.query,
        limit_relevant=payload.limit,
        similarity_threshold=payload.similarity_threshold,
    )
    return response


@router.get("/profile", response_model=dict[str, Any])
async def get_user_profile(
    session: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """Returns the living compiled profile for the current user."""
    from rag_app.memory.profile_compiler import compile_user_profile
    profile = user.profile_json
    if not profile:
        profile = await compile_user_profile(user_id=user.id, session=session)
        await session.commit()
    return profile


@router.post("/profile/recompile", response_model=dict[str, Any])
async def recompile_user_profile_endpoint(
    session: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """Forces invalidation and immediate recompilation of the user profile."""
    from rag_app.memory.profile_compiler import invalidate_and_recompile_user_profile
    profile = await invalidate_and_recompile_user_profile(user_id=user.id, session=session)
    await session.commit()
    return profile


@router.post("/dream", response_model=dict[str, Any])
async def trigger_memory_dreaming(
    session: DbSession,
    user: CurrentUser,
    threshold: Optional[int] = Query(None, ge=2, le=50),
) -> dict[str, Any]:
    """Runs memory dreaming / consolidation for the current user."""
    from rag_app.memory.dreaming import run_memory_dreaming
    result = await run_memory_dreaming(user_id=user.id, session=session, threshold=threshold)
    return result


def _normalize_limit(val: Any, default: int = 50, min_val: int = 1, max_val: int = 500) -> int:
    """Safely normalizes integer limit parameter from FastAPI Query with upper/lower bounds."""
    result = default
    if isinstance(val, int):
        result = val
    elif hasattr(val, "default"):
        try:
            result = int(val.default)
        except Exception:
            result = default
    else:
        try:
            result = int(val)
        except (ValueError, TypeError):
            result = default

    if result < min_val:
        return min_val
    if result > max_val:
        return max_val
    return result


@router.get("/overview", response_model=dict[str, Any])
async def get_memory_overview(
    session: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """
    Returns an aggregated overview of memory items, structured categories,
    lifecycle statuses, relationships, and dreaming runs for the user.
    """
    from sqlalchemy import func, select, or_
    from rag_app.db.models import MemoryItem, MemoryRelationship, MemoryDreamingRun
    from rag_app.memory.profile_compiler import compile_user_profile

    by_type = {}
    by_status = {}
    total_memories = 0
    active_memories = 0
    rel_count = 0
    total_dreaming_runs = 0
    last_dreaming_run = None

    try:
        # Count by memory_type
        type_counts_rows = (
            await session.execute(
                select(MemoryItem.memory_type, func.count(MemoryItem.id))
                .where(MemoryItem.user_id == user.id)
                .group_by(MemoryItem.memory_type)
            )
        ).all()
        by_type = {str(row[0]): int(row[1]) for row in type_counts_rows}
    except Exception as e:
        logger.debug(f"Could not aggregate memory types: {e}")

    try:
        # Count by status
        status_counts_rows = (
            await session.execute(
                select(MemoryItem.status, func.count(MemoryItem.id))
                .where(MemoryItem.user_id == user.id)
                .group_by(MemoryItem.status)
            )
        ).all()
        by_status = {str(row[0]): int(row[1]) for row in status_counts_rows}
        total_memories = sum(by_status.values())
        active_memories = by_status.get("ACTIVE", 0)
    except Exception as e:
        logger.debug(f"Could not aggregate memory statuses: {e}")

    try:
        # Relationships count
        user_mem_ids_subquery = select(MemoryItem.id).where(MemoryItem.user_id == user.id).scalar_subquery()
        rel_count = (
            await session.scalar(
                select(func.count(MemoryRelationship.id)).where(
                    or_(
                        MemoryRelationship.source_memory_id.in_(user_mem_ids_subquery),
                        MemoryRelationship.target_memory_id.in_(user_mem_ids_subquery),
                    )
                )
            )
        ) or 0
    except Exception as e:
        logger.debug(f"Could not count memory relationships: {e}")

    try:
        # Dreaming runs count & last run
        dreaming_stmt = (
            select(MemoryDreamingRun)
            .where(MemoryDreamingRun.user_id == user.id)
            .order_by(MemoryDreamingRun.started_at.desc())
        )
        dreaming_runs = (await session.execute(dreaming_stmt)).scalars().all()
        total_dreaming_runs = len(dreaming_runs)
        last_dreaming_run = dreaming_runs[0].started_at.isoformat() if dreaming_runs else None
    except Exception as e:
        logger.debug(f"Could not query dreaming runs: {e}")

    # Current profile summary
    profile = user.profile_json or {}
    if not profile:
        try:
            profile = await compile_user_profile(user_id=user.id, session=session)
            await session.commit()
        except Exception:
            profile = {}

    return {
        "user_id": str(user.id),
        "total_memories": total_memories,
        "active_memories": active_memories,
        "by_type": by_type,
        "by_status": by_status,
        "total_relationships": rel_count,
        "total_dreaming_runs": total_dreaming_runs,
        "last_dreaming_run": last_dreaming_run,
        "living_profile_summary": profile,
    }


@router.get("/jobs", response_model=list[dict[str, Any]])
async def list_memory_processing_jobs(
    session: DbSession,
    user: CurrentUser,
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Lists recent memory processing jobs for the authenticated user."""
    from sqlalchemy import select
    from rag_app.db.models import MemoryProcessingJob

    try:
        limit_val = _normalize_limit(limit, 50)
        stmt = (
            select(MemoryProcessingJob)
            .where(MemoryProcessingJob.owner_user_id == user.id)
            .order_by(MemoryProcessingJob.created_at.desc())
            .limit(limit_val)
        )
        jobs = (await session.execute(stmt)).scalars().all()
        return [
            {
                "id": str(j.id),
                "source_message_id": str(j.source_message_id) if j.source_message_id else None,
                "conversation_id": str(j.conversation_id) if j.conversation_id else None,
                "owner_user_id": str(j.owner_user_id),
                "workspace_id": str(j.workspace_id) if j.workspace_id else None,
                "source_text_snapshot": j.source_text_snapshot,
                "trigger_kind": j.trigger_kind,
                "status": j.status,
                "attempt_count": j.attempt_count,
                "max_attempts": j.max_attempts,
                "error_message": j.error_message,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "processed_at": j.processed_at.isoformat() if j.processed_at else None,
            }
            for j in jobs
        ]
    except Exception as e:
        logger.error(f"Error querying memory processing jobs: {e}", exc_info=True)
        return []


@router.get("/relationships", response_model=list[dict[str, Any]])
async def list_memory_relationships(
    session: DbSession,
    user: CurrentUser,
    limit: int = Query(100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Lists consolidation and temporal relationship links for the user's memories."""
    from sqlalchemy import select, or_
    from rag_app.db.models import MemoryItem, MemoryRelationship

    try:
        limit_val = _normalize_limit(limit, 100)
        user_mem_subquery = select(MemoryItem.id).where(MemoryItem.user_id == user.id).scalar_subquery()
        stmt = (
            select(MemoryRelationship)
            .where(
                or_(
                    MemoryRelationship.source_memory_id.in_(user_mem_subquery),
                    MemoryRelationship.target_memory_id.in_(user_mem_subquery),
                )
            )
            .order_by(MemoryRelationship.created_at.desc())
            .limit(limit_val)
        )
        relationships = (await session.execute(stmt)).scalars().all()
        return [
            {
                "id": str(r.id),
                "source_memory_id": str(r.source_memory_id) if r.source_memory_id else "",
                "target_memory_id": str(r.target_memory_id) if r.target_memory_id else "",
                "relationship_type": r.relationship_type,
                "confidence": r.confidence,
                "reason": r.reason,
                "source_evidence": r.source_evidence,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in relationships
        ]
    except Exception as e:
        logger.error(f"Error querying memory relationships: {e}", exc_info=True)
        return []


@router.get("/dreaming/runs", response_model=list[dict[str, Any]])
async def list_dreaming_runs(
    session: DbSession,
    user: CurrentUser,
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Lists recent Autonomous Dreaming execution runs for the authenticated user."""
    from sqlalchemy import select
    from rag_app.db.models import MemoryDreamingRun

    try:
        limit_val = _normalize_limit(limit, 50)
        stmt = (
            select(MemoryDreamingRun)
            .where(MemoryDreamingRun.user_id == user.id)
            .order_by(MemoryDreamingRun.started_at.desc())
            .limit(limit_val)
        )
        runs = (await session.execute(stmt)).scalars().all()
        return [
            {
                "id": str(r.id),
                "run_id": r.run_id,
                "user_id": str(r.user_id),
                "workspace_id": str(r.workspace_id) if r.workspace_id else None,
                "trigger_type": r.trigger_type,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "memories_inspected": r.memories_inspected,
                "memories_skipped": r.memories_skipped,
                "duplicates_detected": r.duplicates_detected,
                "conflicts_detected": r.conflicts_detected,
                "new_memories_created": r.new_memories_created,
                "memories_superseded": r.memories_superseded,
                "profile_updated": r.profile_updated,
                "error_message": r.error_message,
                "duration_seconds": r.duration_seconds,
            }
            for r in runs
        ]
    except Exception as e:
        logger.error(f"Error querying dreaming runs: {e}", exc_info=True)
        return []


@router.post("/dreaming/run", response_model=dict[str, Any])
async def trigger_dreaming_run(
    session: DbSession,
    user: CurrentUser,
    threshold: Optional[int] = Query(None, ge=1, le=50),
    max_batch_size: int = Query(50, ge=5, le=200),
    workspace_id: Optional[UUID] = Query(None),
) -> dict[str, Any]:
    """Manually triggers an Autonomous Dreaming consolidation run for the user."""
    from rag_app.memory.dreaming import run_memory_dreaming
    try:
        result = await run_memory_dreaming(
            user_id=user.id,
            session=session,
            threshold=threshold,
            trigger_type="manual",
            max_batch_size=max_batch_size,
            workspace_id=workspace_id,
        )
        return result
    except Exception as e:
        logger.error(f"Error executing manual dreaming run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dreaming run failed: {str(e)}")


@router.get("/evaluation", response_model=list[dict[str, Any]])
async def get_evaluation_metrics(
    session: DbSession,
    user: CurrentUser,
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Returns recent deterministic evaluation quality metrics."""
    from sqlalchemy import select
    from rag_app.db.models import MemoryEvalMetric

    try:
        limit_val = _normalize_limit(limit, 50)
        stmt = select(MemoryEvalMetric).order_by(MemoryEvalMetric.created_at.desc()).limit(limit_val)
        metrics = (await session.execute(stmt)).scalars().all()
        return [
            {
                "id": str(m.id),
                "eval_run_id": m.eval_run_id,
                "scenario_name": m.scenario_name,
                "metric_category": m.metric_category,
                "metrics_json": m.metrics_json,
                "passed": m.passed,
                "details": m.details,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in metrics
        ]
    except Exception as e:
        logger.error(f"Error querying evaluation metrics: {e}", exc_info=True)
        return []


@router.post("/evaluation/run", response_model=dict[str, Any])
async def trigger_evaluation_run(
    session: DbSession,
    user: CurrentUser,
) -> dict[str, Any]:
    """Executes the complete 20-scenario deterministic evaluation suite and returns results."""
    from rag_app.memory.evaluation import MemoryEvaluationRunner
    runner = MemoryEvaluationRunner(session=session)
    results = await runner.run_all()
    return results

