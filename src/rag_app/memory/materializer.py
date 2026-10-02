import logging
import uuid
import unicodedata
import re
from typing import Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func, select, update, text

from rag_app.db.models import MemoryItem
from rag_app.memory.schemas import MemoryCandidate, MemoryStatus, MemoryType

logger = logging.getLogger(__name__)


def normalize_string(text_val: str) -> str:
    """
    Deterministic normalization rules for fields used in canonical identity.
    - NFKD Unicode normalization
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse repeated whitespace into a single space
    """
    if not text_val:
        return ""
    normalized = unicodedata.normalize('NFKD', text_val.strip())
    normalized = normalized.lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def generate_memory_canonical_key(
    memory_type: str,
    key_or_content: str,
) -> str:
    """
    Generates a deterministic canonical identity for a user-scoped memory item.
    - Format: MEMORY:<UPPERCASE_TYPE>:<normalized_key>
    """
    m_type = str(memory_type).strip().upper()
    norm_key = normalize_string(key_or_content)
    return f"MEMORY:{m_type}:{norm_key}"


class MemoryMaterializer:
    """
    Materializes MemoryCandidates into PostgreSQL with:
    - User-level strict isolation
    - Idempotency via ON CONFLICT (user_id, canonical_key) WHERE status = 'ACTIVE'
    - Supersession tracking (updates superseded memory to SUPERSEDED and sets superseded_by_id)
    - Full provenance retention back to conversation/event
    """

    @staticmethod
    async def generate_embedding(content: str) -> Optional[list[float]]:
        """Generates embedding for memory content using project embedding provider."""
        if not content or not content.strip():
            return None
        try:
            from rag_app.ingestion.embedding_v1 import get_embedding_provider
            provider = get_embedding_provider()
            return await provider.embed_query(content.strip())
        except Exception as exc:
            logger.warning(
                "memory_embedding_generation_failed",
                extra={"event": "memory_embedding_generation_failed", "error": str(exc)},
            )
            return None

    @staticmethod
    async def materialize_memory(
        session: AsyncSession,
        candidate: MemoryCandidate,
        auto_embed: bool = True,
    ) -> MemoryItem:
        """
        Materialize a single MemoryCandidate into PostgreSQL idempotently with embedding.
        """
        mem_type = (
            candidate.memory_type.value
            if isinstance(candidate.memory_type, MemoryType)
            else str(candidate.memory_type).upper()
        )
        mem_status = (
            candidate.status.value
            if isinstance(candidate.status, MemoryStatus)
            else str(candidate.status).upper()
        )

        canonical_key = candidate.canonical_key or generate_memory_canonical_key(
            mem_type,
            candidate.content,
        )

        embedding = candidate.embedding
        if embedding is None and auto_embed and candidate.content:
            embedding = await MemoryMaterializer.generate_embedding(candidate.content)

        # 1. If superseding an old memory item, mark old memory as SUPERSEDED first
        if candidate.supersedes_id:
            await session.execute(
                update(MemoryItem)
                .where(
                    MemoryItem.id == candidate.supersedes_id,
                    MemoryItem.user_id == candidate.user_id,
                )
                .values(
                    status="SUPERSEDED",
                    updated_at=func.now(),
                )
            )

        # 2. Upsert MemoryItem
        stmt = insert(MemoryItem).values(
            user_id=candidate.user_id,
            canonical_key=canonical_key,
            memory_type=mem_type,
            status=mem_status,
            content=candidate.content,
            embedding=embedding,
            structured_data=candidate.structured_data or {},
            confidence=candidate.confidence,
            valid_from=candidate.valid_from,
            valid_until=candidate.valid_until,
            source_type=candidate.source_type,
            conversation_id=candidate.conversation_id,
            message_id=candidate.message_id,
            event_id=candidate.event_id,
            extraction_method=candidate.extraction_method,
        ).on_conflict_do_update(
            index_elements=['user_id', 'canonical_key'],
            index_where=text("status = 'ACTIVE'"),
            set_={
                'content': insert(MemoryItem).excluded.content,
                'embedding': func.coalesce(
                    insert(MemoryItem).excluded.embedding,
                    MemoryItem.embedding,
                ),
                'status': insert(MemoryItem).excluded.status,
                'structured_data': MemoryItem.structured_data.op('||')(insert(MemoryItem).excluded.structured_data),
                'confidence': func.coalesce(
                    insert(MemoryItem).excluded.confidence,
                    MemoryItem.confidence,
                ),
                'valid_from': insert(MemoryItem).excluded.valid_from,
                'valid_until': insert(MemoryItem).excluded.valid_until,
                'source_type': insert(MemoryItem).excluded.source_type,
                'conversation_id': insert(MemoryItem).excluded.conversation_id,
                'message_id': insert(MemoryItem).excluded.message_id,
                'event_id': insert(MemoryItem).excluded.event_id,
                'extraction_method': insert(MemoryItem).excluded.extraction_method,
                'updated_at': func.now(),
            }
        ).returning(MemoryItem)

        result = await session.execute(stmt)
        memory_item = result.scalar_one()
        await session.refresh(memory_item)

        # 3. Handle supersession pointer if candidate supersedes an existing memory item
        if candidate.supersedes_id and candidate.supersedes_id != memory_item.id:
            await MemoryMaterializer.supersede_memory(
                session=session,
                old_memory_id=candidate.supersedes_id,
                new_memory_id=memory_item.id,
                user_id=candidate.user_id,
            )

        return memory_item

    @staticmethod
    async def supersede_memory(
        session: AsyncSession,
        old_memory_id: uuid.UUID,
        new_memory_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
    ) -> bool:
        """
        Marks an old memory as SUPERSEDED and points superseded_by_id to new_memory_id.
        Preserves complete history without deleting rows.
        """
        stmt = update(MemoryItem).where(
            MemoryItem.id == old_memory_id,
        )
        if user_id:
            stmt = stmt.where(MemoryItem.user_id == user_id)

        stmt = stmt.values(
            status="SUPERSEDED",
            superseded_by_id=new_memory_id,
            updated_at=func.now(),
        )
        result = await session.execute(stmt)
        return result.rowcount > 0

    @staticmethod
    async def get_user_memories(
        session: AsyncSession,
        user_id: uuid.UUID,
        status: Optional[str] = "ACTIVE",
        memory_type: Optional[str] = None,
    ) -> Sequence[MemoryItem]:
        """
        Queries memories strictly isolated to user_id.
        """
        stmt = select(MemoryItem).where(MemoryItem.user_id == user_id)
        if status is not None:
            stmt = stmt.where(MemoryItem.status == status)
        if memory_type is not None:
            stmt = stmt.where(MemoryItem.memory_type == memory_type)

        stmt = stmt.order_by(MemoryItem.created_at.desc())
        result = await session.execute(stmt)
        return result.scalars().all()


async def materialize_memory(
    session: AsyncSession,
    candidate: MemoryCandidate,
    auto_embed: bool = True,
) -> MemoryItem:
    """Convenience functional wrapper matching existing knowledge materializer pattern."""
    return await MemoryMaterializer.materialize_memory(
        session=session,
        candidate=candidate,
        auto_embed=auto_embed,
    )

