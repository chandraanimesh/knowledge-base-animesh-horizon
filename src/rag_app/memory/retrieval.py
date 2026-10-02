import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Union, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, desc, asc

from rag_app.db.models import MemoryItem
from rag_app.memory.schemas import (
    MemoryRecord,
    MemoryRetrievalResponse,
    MemoryStatus,
    MemoryType,
)

logger = logging.getLogger(__name__)


class MemoryRetrievalService:
    """
    Focused service for retrieving user-scoped persistent memories, supporting:
    - User-level data isolation
    - ACTIVE-only status filter
    - Dynamic validity window expiration check
    - Static vs Dynamic memory partitioning
    - Native pgvector cosine similarity semantic retrieval
    - Deterministic tie-breaking
    """

    @classmethod
    async def retrieve(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        query: Optional[str] = None,
        memory_type: Optional[Union[MemoryType, str]] = None,
        limit_static: int = 20,
        limit_dynamic: int = 20,
        limit_relevant: int = 10,
        similarity_threshold: Optional[float] = None,
        reference_time: Optional[datetime] = None,
        embedding_provider=None,
    ) -> MemoryRetrievalResponse:
        """
        Retrieves active, non-expired memories for a single user, separated into:
        - static_memories: durable persona / preference facts
        - dynamic_memories: transient context / active tasks
        - relevant_memories: semantically ranked user memories matching query
        """
        if not user_id:
            raise ValueError("user_id is required for memory retrieval")

        now = reference_time or datetime.now(timezone.utc)

        # Base filter: strict user isolation, ACTIVE status only, valid time window
        base_filters = [
            MemoryItem.user_id == user_id,
            MemoryItem.status == MemoryStatus.ACTIVE.value,
            or_(
                MemoryItem.valid_until.is_(None),
                MemoryItem.valid_until > now,
            ),
            or_(
                MemoryItem.valid_from.is_(None),
                MemoryItem.valid_from <= now,
            ),
        ]

        # Memory type filter if explicitly requested
        target_type = None
        if memory_type is not None:
            target_type = (
                memory_type.value
                if isinstance(memory_type, MemoryType)
                else str(memory_type).upper()
            )

        # 1. Retrieve static profile memories
        static_records: list[MemoryRecord] = []
        if target_type is None or target_type == MemoryType.STATIC.value:
            static_stmt = (
                select(MemoryItem)
                .where(
                    *base_filters,
                    MemoryItem.memory_type == MemoryType.STATIC.value,
                )
                .order_by(
                    desc(MemoryItem.confidence).nullslast(),
                    desc(MemoryItem.created_at),
                    asc(MemoryItem.id),
                )
                .limit(limit_static)
            )
            static_results = (await session.execute(static_stmt)).scalars().all()
            static_records = [MemoryRecord.model_validate(m) for m in static_results]

        # 2. Retrieve dynamic context memories
        dynamic_records: list[MemoryRecord] = []
        if target_type is None or target_type == MemoryType.DYNAMIC.value:
            dynamic_stmt = (
                select(MemoryItem)
                .where(
                    *base_filters,
                    MemoryItem.memory_type == MemoryType.DYNAMIC.value,
                )
                .order_by(
                    desc(MemoryItem.confidence).nullslast(),
                    desc(MemoryItem.created_at),
                    asc(MemoryItem.id),
                )
                .limit(limit_dynamic)
            )
            dynamic_results = (await session.execute(dynamic_stmt)).scalars().all()
            dynamic_records = [MemoryRecord.model_validate(m) for m in dynamic_results]

        # 3. Retrieve semantically relevant memories if query is provided
        relevant_records: list[MemoryRecord] = []
        if query and query.strip():
            query_clean = query.strip()
            try:
                if embedding_provider is None:
                    from rag_app.ingestion.embedding_v1 import get_embedding_provider
                    provider = get_embedding_provider()
                else:
                    provider = embedding_provider

                query_embedding = await provider.embed_query(query_clean)

                if query_embedding:
                    vector_filters = list(base_filters) + [MemoryItem.embedding.is_not(None)]
                    if target_type is not None:
                        vector_filters.append(MemoryItem.memory_type == target_type)

                    distance_col = MemoryItem.embedding.cosine_distance(query_embedding)

                    relevant_stmt = (
                        select(MemoryItem, distance_col.label("distance"))
                        .where(*vector_filters)
                        .order_by(
                            asc("distance"),
                            desc(MemoryItem.confidence).nullslast(),
                            desc(MemoryItem.created_at),
                            asc(MemoryItem.id),
                        )
                        .limit(limit_relevant)
                    )

                    rows = (await session.execute(relevant_stmt)).all()
                    for row in rows:
                        mem_item = row[0]
                        dist_val = float(row[1]) if row[1] is not None else 1.0
                        score = round(max(0.0, 1.0 - dist_val), 4)

                        if similarity_threshold is not None and score < similarity_threshold:
                            continue

                        rec = MemoryRecord.model_validate(mem_item)
                        rec.score = score
                        relevant_records.append(rec)
            except Exception as exc:
                logger.error(
                    "semantic_memory_retrieval_failed",
                    extra={
                        "event": "semantic_memory_retrieval_failed",
                        "user_id": str(user_id),
                        "query": query_clean,
                        "error": str(exc),
                    },
                )

        return MemoryRetrievalResponse(
            user_id=user_id,
            static_memories=static_records,
            dynamic_memories=dynamic_records,
            relevant_memories=relevant_records,
            total_count=len(static_records) + len(dynamic_records) + len(relevant_records),
        )

    @classmethod
    async def get_static_profile(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        limit: int = 50,
        reference_time: Optional[datetime] = None,
    ) -> list[MemoryRecord]:
        """Convenience method to retrieve durable profile facts and preferences."""
        resp = await cls.retrieve(
            session=session,
            user_id=user_id,
            memory_type=MemoryType.STATIC,
            limit_static=limit,
            reference_time=reference_time,
        )
        return resp.static_memories

    @classmethod
    async def get_dynamic_context(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        query: Optional[str] = None,
        limit: int = 20,
        reference_time: Optional[datetime] = None,
    ) -> list[MemoryRecord]:
        """Convenience method to retrieve active contextual memories and current task facts."""
        resp = await cls.retrieve(
            session=session,
            user_id=user_id,
            query=query,
            memory_type=MemoryType.DYNAMIC,
            limit_dynamic=limit,
            reference_time=reference_time,
        )
        return resp.dynamic_memories

    @classmethod
    async def search_relevant_memories(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        query: str,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        reference_time: Optional[datetime] = None,
        embedding_provider=None,
    ) -> list[MemoryRecord]:
        """Convenience method to search semantically relevant user memories."""
        resp = await cls.retrieve(
            session=session,
            user_id=user_id,
            query=query,
            limit_relevant=limit,
            similarity_threshold=similarity_threshold,
            reference_time=reference_time,
            embedding_provider=embedding_provider,
        )
        return resp.relevant_memories
