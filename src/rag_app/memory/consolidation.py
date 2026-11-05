"""Memory Consolidation & Temporal Intelligence Engine (Milestone 9C)

Provides:
- Deduplication (exact & semantic cosine similarity)
- Contradiction & conflict detection
- Complementary memory linking
- Temporal supersession & valid_from/valid_until tracking
- Immutable MemoryRelationship ledger records
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from sqlalchemy import select, update, or_, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import MemoryItem, MemoryRelationship
from rag_app.memory.schemas import MemoryCandidate, RelationshipType
from rag_app.memory.materializer import normalize_string

logger = logging.getLogger(__name__)


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


class MemoryConsolidationService:
    """
    Manages relationship detection, reconciliation, deduplication,
    conflict detection, and temporal intelligence for personal memories.
    """

    @staticmethod
    async def record_relationship(
        session: AsyncSession,
        source_memory_id: uuid.UUID,
        target_memory_id: uuid.UUID,
        relationship_type: str | RelationshipType,
        confidence: float = 1.0,
        reason: Optional[str] = None,
        source_evidence: Optional[dict[str, Any]] = None,
    ) -> MemoryRelationship:
        """
        Records a relationship link between two memories in the memory_relationships ledger.
        Idempotent: prevents duplicate relationship links of the exact same type between the pair.
        """
        rel_type_str = (
            relationship_type.value
            if hasattr(relationship_type, "value")
            else str(relationship_type).upper()
        )

        # Check existing relationship
        stmt = (
            select(MemoryRelationship)
            .where(
                MemoryRelationship.source_memory_id == source_memory_id,
                MemoryRelationship.target_memory_id == target_memory_id,
                MemoryRelationship.relationship_type == rel_type_str,
            )
            .limit(1)
        )
        existing = await session.scalar(stmt)
        if existing:
            return existing

        rel = MemoryRelationship(
            id=uuid.uuid4(),
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            relationship_type=rel_type_str,
            confidence=max(0.0, min(1.0, float(confidence))),
            reason=reason,
            source_evidence=source_evidence or {},
            created_at=datetime.now(timezone.utc),
        )
        session.add(rel)
        await session.flush()

        logger.info(
            "memory_relationship_recorded",
            extra={
                "event": "memory_relationship_recorded",
                "relationship_id": str(rel.id),
                "source_memory_id": str(source_memory_id),
                "target_memory_id": str(target_memory_id),
                "relationship_type": rel_type_str,
            },
        )
        return rel

    @staticmethod
    async def get_relationships_for_memory(
        session: AsyncSession,
        memory_id: uuid.UUID,
    ) -> Sequence[MemoryRelationship]:
        """Fetches all relationships where memory_id is source or target."""
        stmt = (
            select(MemoryRelationship)
            .where(
                or_(
                    MemoryRelationship.source_memory_id == memory_id,
                    MemoryRelationship.target_memory_id == memory_id,
                )
            )
            .order_by(MemoryRelationship.created_at.desc())
        )
        return (await session.execute(stmt)).scalars().all()

    @staticmethod
    async def detect_duplicates(
        session: AsyncSession,
        user_id: uuid.UUID,
        content: str,
        embedding: Optional[list[float]] = None,
        exclude_id: Optional[uuid.UUID] = None,
        similarity_threshold: float = 0.88,
    ) -> list[tuple[MemoryItem, float, str]]:
        """
        Detects exact and semantic duplicates among active memories for a given user.
        Returns: list of (existing_memory, similarity_score, reason)
        """
        content_norm = normalize_string(content)
        if not content_norm:
            return []

        now = datetime.now(timezone.utc)
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
        )
        if exclude_id is not None:
            stmt = stmt.where(MemoryItem.id != exclude_id)

        candidates = (await session.execute(stmt)).scalars().all()
        duplicates: list[tuple[MemoryItem, float, str]] = []

        for item in candidates:
            item_norm = normalize_string(item.content or "")
            # 1. Exact match on normalized text
            if item_norm == content_norm:
                duplicates.append((item, 1.0, "Exact content match"))
                continue

            # 2. Semantic vector cosine similarity
            if embedding and item.embedding:
                try:
                    # item.embedding may be a list or pgvector array
                    item_vec = list(item.embedding) if not isinstance(item.embedding, list) else item.embedding
                    sim = compute_cosine_similarity(embedding, item_vec)
                    if sim >= similarity_threshold:
                        duplicates.append((item, sim, f"Semantic similarity ({sim:.2f}) >= {similarity_threshold}"))
                except Exception as sim_err:
                    logger.debug(f"Cosine similarity check failed: {sim_err}")

        return duplicates

    @staticmethod
    async def detect_conflicts(
        session: AsyncSession,
        user_id: uuid.UUID,
        content: str,
        canonical_key: Optional[str] = None,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> list[tuple[MemoryItem, str]]:
        """
        Detects factual contradictions or opposing constraints on the same topic.
        Returns: list of (conflicting_memory, conflict_reason)
        """
        content_norm = normalize_string(content)
        if not content_norm:
            return []

        now = datetime.now(timezone.utc)
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
        )
        if exclude_id is not None:
            stmt = stmt.where(MemoryItem.id != exclude_id)

        candidates = (await session.execute(stmt)).scalars().all()
        conflicts: list[tuple[MemoryItem, str]] = []

        # Common opposing value pairs for conflict detection
        opposing_pairs = [
            ("dark mode", "light mode"),
            ("concise", "detailed"),
            ("brief", "verbose"),
            ("in usd", "in eur"),
            ("in usd", "in inr"),
            ("in eur", "in inr"),
            ("python", "typescript"),
            ("react", "vue"),
        ]

        for item in candidates:
            item_norm = normalize_string(item.content or "")

            # If same canonical key prefix / topic, but different content
            if canonical_key and item.canonical_key:
                # Compare topic keys (parts after type)
                c_topic = ":".join(canonical_key.split(":")[2:]) if canonical_key.count(":") >= 2 else canonical_key
                i_topic = ":".join(item.canonical_key.split(":")[2:]) if item.canonical_key.count(":") >= 2 else item.canonical_key
                if c_topic and i_topic and c_topic == i_topic and item_norm != content_norm:
                    conflicts.append((item, f"Contradictory values for topic '{c_topic}': '{item.content}' vs '{content}'"))
                    continue

            # Check opposing pairs
            for a, b in opposing_pairs:
                if (a in content_norm and b in item_norm) or (b in content_norm and a in item_norm):
                    conflicts.append((item, f"Opposing directives detected: '{a}' vs '{b}'"))
                    break

        return conflicts

    @staticmethod
    async def detect_complements(
        session: AsyncSession,
        user_id: uuid.UUID,
        content: str,
        canonical_key: Optional[str] = None,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> list[tuple[MemoryItem, str]]:
        """
        Detects complementary memories that enrich the current memory on the same domain/project.
        Returns: list of (complementary_memory, reason)
        """
        content_norm = normalize_string(content)
        if not content_norm:
            return []

        now = datetime.now(timezone.utc)
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
        )
        if exclude_id is not None:
            stmt = stmt.where(MemoryItem.id != exclude_id)

        candidates = (await session.execute(stmt)).scalars().all()
        complements: list[tuple[MemoryItem, str]] = []

        keywords = [k for k in content_norm.split() if len(k) > 4 and k not in ("prefer", "always", "working", "reviewing")]
        if not keywords:
            return []

        for item in candidates:
            item_norm = normalize_string(item.content or "")
            if item_norm == content_norm:
                continue

            shared_keywords = [k for k in keywords if k in item_norm]
            if len(shared_keywords) >= 2:
                complements.append((
                    item,
                    f"Complementary domain context (shared entities: {', '.join(shared_keywords[:3])})",
                ))

        return complements

    @staticmethod
    async def supersede_memory(
        session: AsyncSession,
        old_memory_id: uuid.UUID,
        new_memory_id: uuid.UUID,
        user_id: uuid.UUID,
        reason: str = "Superseded by newer memory item",
    ) -> bool:
        """
        Marks an old memory item as SUPERSEDED, sets superseded_by_id,
        closes its valid_until timestamp to now, and records a SUPERSEDES relationship.
        """
        now = datetime.now(timezone.utc)
        stmt = (
            update(MemoryItem)
            .where(
                MemoryItem.id == old_memory_id,
                MemoryItem.user_id == user_id,
            )
            .values(
                status="SUPERSEDED",
                superseded_by_id=new_memory_id,
                valid_until=now,
                updated_at=func.now(),
            )
        )
        res = await session.execute(stmt)
        if res.rowcount > 0:
            await MemoryConsolidationService.record_relationship(
                session=session,
                source_memory_id=old_memory_id,
                target_memory_id=new_memory_id,
                relationship_type=RelationshipType.SUPERSEDES,
                confidence=1.0,
                reason=reason,
                source_evidence={"superseded_at": now.isoformat()},
            )
            return True
        return False

