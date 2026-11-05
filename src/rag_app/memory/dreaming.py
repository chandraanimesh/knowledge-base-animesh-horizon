"""Memory Dreaming / Autonomous Consolidation Engine (Milestone 9D)

Periodically or on-demand consolidates fragmented, redundant, or accumulating
memories into cohesive, authoritative summaries under strict provenance, safety,
and temporal intelligence rules.
Tracks execution runs in memory_dreaming_runs table.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import select, func, update, or_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.core.config import get_settings
from rag_app.db.models import MemoryItem, MemoryDreamingRun, MemoryTombstone
from rag_app.memory.profile_compiler import compile_user_profile
from rag_app.memory.privacy_service import MemoryPrivacyService
from rag_app.memory.consolidation import MemoryConsolidationService

logger = logging.getLogger(__name__)


def normalize_string(s: str) -> str:
    """Normalizes string for comparison by lowercasing and collapsing whitespace."""
    return " ".join(s.lower().strip().split())


def _extract_cluster_key(mem: MemoryItem) -> str:
    """Extracts a grouping key for clustering related memories."""
    struct = mem.structured_data or {}
    if "category" in struct and struct["category"]:
        return str(struct["category"]).lower()

    m_type = str(mem.memory_type or "").upper()
    if m_type in ("EPISODIC", "SEMANTIC", "PROCEDURAL"):
        # Group by topic if present
        canonical = mem.canonical_key or ""
        parts = canonical.split(":")
        if len(parts) >= 3:
            return parts[2].lower()
        return m_type.lower()

    canonical = mem.canonical_key or ""
    if ":" in canonical:
        parts = canonical.split(":")
        for p in parts:
            p_lower = p.lower()
            if p_lower in ("preference", "work", "tech", "user_identity", "project", "style", "convention"):
                return p_lower
        if len(parts) >= 3:
            return parts[2].lower()

    return "general"


def _synthesize_cluster_content(memories: list[MemoryItem], category: str) -> str:
    """
    Deterministic synthesis fallback that preserves hard constraints,
    resolves temporal overrides, and generates a cohesive summary.
    """
    sorted_mems = sorted(memories, key=lambda m: m.created_at or m.updated_at)
    unique_facts: dict[str, str] = {}
    free_statements: list[str] = []

    for m in sorted_mems:
        c_text = (m.content or "").strip()
        if not c_text:
            continue

        # If it's a key-value style
        if ":" in c_text and len(c_text.split(":", 1)[0].split()) <= 3:
            k, v = c_text.split(":", 1)
            unique_facts[k.strip().lower()] = v.strip()
        else:
            if c_text not in free_statements:
                free_statements.append(c_text)

    summary_parts = []
    if unique_facts:
        summary_parts.append("; ".join(f"{k.capitalize()}: {v}" for k, v in unique_facts.items()))
    if free_statements:
        summary_parts.append(" ".join(free_statements))

    return "\n".join(summary_parts) if summary_parts else f"Consolidated {category} profile."


async def run_memory_dreaming(
    user_id: uuid.UUID,
    session: AsyncSession,
    threshold: int | None = None,
    trigger_type: str = "scheduled",
    max_batch_size: int = 50,
    workspace_id: Optional[uuid.UUID] = None,
) -> dict[str, Any]:
    """
    Executes memory dreaming / consolidation for a given user.
    Records a durable MemoryDreamingRun, groups active memories into clusters,
    synthesizes qualifying clusters (>= threshold) into consolidated items,
    supersedes source items, updates relationships, and recompiles the user profile.
    """
    settings = get_settings()
    consolidation_threshold = (
        threshold
        if threshold is not None
        else getattr(settings, "dreaming_consolidation_threshold", 5)
    )

    started_at = datetime.now(timezone.utc)
    run_id = f"dream_{uuid.uuid4().hex[:12]}"

    # Initialize tracking record
    dreaming_run = MemoryDreamingRun(
        id=uuid.uuid4(),
        run_id=run_id,
        user_id=user_id,
        workspace_id=workspace_id,
        trigger_type=trigger_type,
        status="RUNNING",
        started_at=started_at,
        memories_inspected=0,
        memories_skipped=0,
        duplicates_detected=0,
        conflicts_detected=0,
        new_memories_created=0,
        memories_superseded=0,
        profile_updated=False,
        token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        checkpoint_data={"max_batch_size": max_batch_size, "threshold": consolidation_threshold},
    )
    session.add(dreaming_run)
    try:
        await session.flush()
    except Exception as d_err:
        logger.warning(f"Failed to record initial MemoryDreamingRun: {d_err}")

    # Privacy Policy Check: Is memory dreaming permitted?
    is_dreaming_permitted = await MemoryPrivacyService.is_dreaming_enabled(session, user_id)
    if not is_dreaming_permitted:
        logger.info(
            "memory_dreaming_skipped_policy_disabled",
            extra={"event": "memory_dreaming_skipped_policy_disabled", "user_id": str(user_id)},
        )
        dreaming_run.status = "SKIPPED_POLICY_PAUSED"
        dreaming_run.completed_at = datetime.now(timezone.utc)
        dreaming_run.error_message = "Dreaming disabled or paused by privacy policy"
        dreaming_run.duration_seconds = 0.0
        try:
            await session.commit()
        except Exception:
            pass
        return {
            "run_id": run_id,
            "user_id": str(user_id),
            "clusters_consolidated": 0,
            "memories_consolidated": 0,
            "new_memory_ids": [],
            "status": "SKIPPED_POLICY_PAUSED",
            "duration_seconds": 0.0,
        }

    try:
        now = datetime.now(timezone.utc)

        # 1. Fetch active, non-expired memories for user up to max_batch_size
        stmt = (
            select(MemoryItem)
            .where(
                MemoryItem.user_id == user_id,
                MemoryItem.status == "ACTIVE",
                or_(
                    MemoryItem.retention_expires_at.is_(None),
                    MemoryItem.retention_expires_at > now,
                ),
                or_(
                    MemoryItem.valid_until.is_(None),
                    MemoryItem.valid_until > now,
                ),
            )
            .order_by(MemoryItem.created_at.asc())
            .limit(max_batch_size)
        )
        if workspace_id is not None:
            stmt = stmt.where(or_(MemoryItem.workspace_id == workspace_id, MemoryItem.workspace_id.is_(None)))

        all_candidates = (await session.execute(stmt)).scalars().all()
        dreaming_run.memories_inspected = len(all_candidates)

        # Anti-resurrection: filter out any memory whose canonical key has been tombstoned
        tombstones = set()
        try:
            tombstones = set(
                (
                    await session.execute(
                        select(MemoryTombstone.canonical_key).where(MemoryTombstone.user_id == user_id)
                    )
                )
                .scalars()
                .all()
            )
        except Exception as t_err:
            try:
                await session.rollback()
            except Exception:
                pass
            logger.debug(f"Could not load tombstones for user {user_id}: {t_err}")

        active_mems = [m for m in all_candidates if m.canonical_key not in tombstones]
        dreaming_run.memories_skipped = len(all_candidates) - len(active_mems)

        if not active_mems:
            dreaming_run.status = "COMPLETED"
            dreaming_run.completed_at = datetime.now(timezone.utc)
            dreaming_run.duration_seconds = (dreaming_run.completed_at - started_at).total_seconds()
            await session.commit()
            return {
                "run_id": run_id,
                "user_id": str(user_id),
                "clusters_consolidated": 0,
                "memories_consolidated": 0,
                "new_memory_ids": [],
                "status": "NO_ACTIVE_MEMORIES",
                "duration_seconds": dreaming_run.duration_seconds,
            }

        # 2. Analyze relationships across active memories (duplicates & conflicts)
        for i, mem in enumerate(active_mems):
            # Duplicate check
            for other in active_mems[i + 1:]:
                if normalize_string(mem.content or "") == normalize_string(other.content or ""):
                    dreaming_run.duplicates_detected += 1
                    await MemoryConsolidationService.record_relationship(
                        session=session,
                        source_memory_id=mem.id,
                        target_memory_id=other.id,
                        relationship_type="DUPLICATE",
                        confidence=1.0,
                        reason="Exact text match during Dreaming run",
                    )
            # Conflict check
            conflicts = await MemoryConsolidationService.detect_conflicts(
                session=session,
                user_id=user_id,
                content=mem.content,
                canonical_key=mem.canonical_key,
                exclude_id=mem.id,
            )
            for conf_item, reason in conflicts:
                dreaming_run.conflicts_detected += 1
                await MemoryConsolidationService.record_relationship(
                    session=session,
                    source_memory_id=conf_item.id,
                    target_memory_id=mem.id,
                    relationship_type="CONTRADICTS",
                    confidence=0.9,
                    reason=f"Dreaming conflict: {reason}",
                )

        # 3. Cluster memories
        clusters: dict[str, list[MemoryItem]] = defaultdict(list)
        for mem in active_mems:
            c_key = _extract_cluster_key(mem)
            clusters[c_key].append(mem)

        new_memory_ids: list[uuid.UUID] = []
        total_consolidated_count = 0
        consolidated_clusters = 0

        # 4. Consolidate qualifying clusters
        for category, mem_list in clusters.items():
            if len(mem_list) < consolidation_threshold:
                continue

            consolidated_clusters += 1
            source_ids = [m.id for m in mem_list]
            total_consolidated_count += len(source_ids)

            consolidated_text = _synthesize_cluster_content(mem_list, category)

            # Generate embedding for consolidated memory
            embedding = None
            try:
                from rag_app.ingestion.embedding_v1 import get_embedding_provider
                provider = get_embedding_provider()
                embedding = await provider.embed_query(consolidated_text)
            except Exception as e_err:
                logger.warning(f"Could not generate embedding for dreamt memory: {e_err}")

            # Create new consolidated MemoryItem
            new_mem = MemoryItem(
                user_id=user_id,
                workspace_id=mem_list[0].workspace_id if all(m.workspace_id == mem_list[0].workspace_id for m in mem_list) else None,
                canonical_key=f"consolidated:{category}:{uuid.uuid4().hex[:8]}",
                memory_type="SEMANTIC",
                status="ACTIVE",
                content=consolidated_text,
                embedding=embedding,
                valid_from=now,
                structured_data={
                    "is_consolidated": True,
                    "category": category,
                    "source_memory_ids": [str(sid) for sid in source_ids],
                    "source_count": len(source_ids),
                    "dreaming_run_id": run_id,
                },
            )
            session.add(new_mem)
            await session.flush()
            new_memory_ids.append(new_mem.id)

            # Mark source memories as SUPERSEDED pointing to new_mem.id
            await session.execute(
                update(MemoryItem)
                .where(MemoryItem.id.in_(source_ids))
                .values(
                    status="SUPERSEDED",
                    superseded_by_id=new_mem.id,
                    valid_until=now,
                    updated_at=func.now(),
                )
            )

            # Record SUPERSEDES relationships
            for sid in source_ids:
                await MemoryConsolidationService.record_relationship(
                    session=session,
                    source_memory_id=sid,
                    target_memory_id=new_mem.id,
                    relationship_type="SUPERSEDES",
                    confidence=1.0,
                    reason=f"Consolidated into {new_mem.canonical_key} during Dreaming",
                )

            logger.info(
                "memory_cluster_dreamt_and_consolidated",
                extra={
                    "event": "memory_cluster_dreamt_and_consolidated",
                    "user_id": str(user_id),
                    "category": category,
                    "source_count": len(source_ids),
                    "new_memory_id": str(new_mem.id),
                    "run_id": run_id,
                },
            )

        # 5. Living User Profile recompile hook
        if consolidated_clusters > 0:
            await compile_user_profile(user_id=user_id, session=session)
            dreaming_run.profile_updated = True

        dreaming_run.status = "COMPLETED"
        dreaming_run.completed_at = datetime.now(timezone.utc)
        dreaming_run.duration_seconds = (dreaming_run.completed_at - started_at).total_seconds()
        dreaming_run.new_memories_created = len(new_memory_ids)
        dreaming_run.memories_superseded = total_consolidated_count

        await session.commit()

        return {
            "run_id": run_id,
            "user_id": str(user_id),
            "clusters_consolidated": consolidated_clusters,
            "memories_consolidated": total_consolidated_count,
            "new_memory_ids": [str(i) for i in new_memory_ids],
            "status": "SUCCESS",
            "duration_seconds": dreaming_run.duration_seconds,
        }

    except Exception as exc:
        logger.error(
            "memory_dreaming_run_failed",
            extra={"event": "memory_dreaming_run_failed", "user_id": str(user_id), "run_id": run_id, "error": str(exc)},
            exc_info=True,
        )
        try:
            await session.rollback()
            dreaming_run.status = "FAILED"
            dreaming_run.completed_at = datetime.now(timezone.utc)
            dreaming_run.error_message = str(exc)
            dreaming_run.duration_seconds = (dreaming_run.completed_at - started_at).total_seconds()
            session.add(dreaming_run)
            await session.commit()
        except Exception:
            pass
        return {
            "run_id": run_id,
            "user_id": str(user_id),
            "status": "FAILED",
            "error": str(exc),
            "clusters_consolidated": 0,
            "memories_consolidated": 0,
            "new_memory_ids": [],
            "duration_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
        }
