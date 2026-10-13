import logging
import uuid
import unicodedata
import re
from typing import Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import func, select, update, text, or_

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


def extract_memory_topic_key(text_val: str) -> str:
    """
    Deterministically maps user memory statements to canonical topic keys.
    Prevents duplicate or contradictory active memories on the same subject.
    """
    if not text_val:
        return "general"
    norm = normalize_string(text_val)

    # 1. Response formatting / verbosity / style
    if any(k in norm for k in ["bullet point", "bulletpoint", "paragraph", "concise", "detailed", "verbosity", "briefly", "short response", "long response"]):
        return "preference:response_format"

    # 2. Programming language / tech stack preferences
    if any(k in norm for k in ["prefer python", "prefer typescript", "prefer javascript", "prefer go", "prefer rust", "prefer java", "use python", "use typescript"]):
        return "preference:programming_language"

    # 3. UI Theme / visual preferences
    if any(k in norm for k in ["dark mode", "light mode", "theme", "dark theme", "light theme"]):
        return "preference:ui_theme"

    # 4. User identity / name
    if any(k in norm for k in ["call me", "my name is", "refer to me as"]):
        return "user_identity:name"

    # 5. User role / occupation
    if any(k in norm for k in ["my role is", "i am a", "i'm a", "work as a", "my job is", "my title is"]):
        return "user_identity:role"

    # 6. User location / residency
    if any(k in norm for k in ["live in", "located in", "based in", "moving to", "relocated to"]):
        return "user_identity:location"

    # 7. Language / locale preference
    if any(k in norm for k in ["speak in", "respond in", "language is", "reply in"]):
        return "preference:natural_language"

    # 8. Currency preference
    if any(k in norm for k in ["currency", "in usd", "in eur", "in inr", "in gbp"]):
        return "preference:currency"

    # Fallback to normalized text
    return norm


class MemoryMaterializer:
    """
    Materializes MemoryCandidates into PostgreSQL with:
    - User-level strict isolation
    - Dual-scope: Global (workspace_id IS NULL) and Workspace-scoped (workspace_id IS NOT NULL)
    - Idempotency via partial ON CONFLICT unique indexes
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
        Supports both workspace-scoped (workspace_id is UUID) and global (workspace_id is None).
        """
        if not candidate.user_id:
            raise ValueError("user_id is required to materialize memory")

        raw_type = (
            candidate.memory_type.value
            if hasattr(candidate.memory_type, "value")
            else str(candidate.memory_type).upper()
        )
        # Map to database constraint ('STATIC', 'DYNAMIC') while preserving granular category in structured_data
        if raw_type in ("STATIC", "SEMANTIC"):
            mem_type = "STATIC"
        else:
            mem_type = "DYNAMIC"

        mem_status = (
            candidate.status.value
            if hasattr(candidate.status, "value")
            else str(candidate.status).upper()
        )

        canonical_key = candidate.canonical_key or generate_memory_canonical_key(
            mem_type,
            candidate.content,
        )

        embedding = candidate.embedding
        if embedding is None and auto_embed and candidate.content:
            embedding = await MemoryMaterializer.generate_embedding(candidate.content)

        struct_data = dict(candidate.structured_data or {})
        struct_data["category"] = raw_type
        if candidate.stability_tier:
            struct_data["stability_tier"] = (
                candidate.stability_tier.value
                if hasattr(candidate.stability_tier, "value")
                else str(candidate.stability_tier)
            )
        if candidate.importance_signals:
            struct_data["importance_signals"] = candidate.importance_signals
        if candidate.subject:
            struct_data["subject"] = candidate.subject
        if candidate.predicate:
            struct_data["predicate"] = candidate.predicate
        if candidate.source_quote:
            struct_data["source_quote"] = candidate.source_quote
        if candidate.source_job_id:
            struct_data["source_job_id"] = str(candidate.source_job_id)

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

        # 2. Upsert MemoryItem with appropriate conflict target based on scope
        insert_values = dict(
            user_id=candidate.user_id,
            workspace_id=candidate.workspace_id,
            canonical_key=canonical_key,
            memory_type=mem_type,
            status=mem_status,
            content=candidate.content,
            embedding=embedding,
            structured_data=struct_data,
            confidence=candidate.confidence,
            valid_from=candidate.valid_from,
            valid_until=candidate.valid_until,
            retention_expires_at=candidate.retention_expires_at,
            source_job_id=candidate.source_job_id,
            source_type=candidate.source_type,
            conversation_id=candidate.conversation_id,
            message_id=candidate.message_id,
            event_id=candidate.event_id,
            extraction_method=candidate.extraction_method,
        )

        update_set = {
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
            'retention_expires_at': insert(MemoryItem).excluded.retention_expires_at,
            'source_job_id': insert(MemoryItem).excluded.source_job_id,
            'source_type': insert(MemoryItem).excluded.source_type,
            'conversation_id': insert(MemoryItem).excluded.conversation_id,
            'message_id': insert(MemoryItem).excluded.message_id,
            'event_id': insert(MemoryItem).excluded.event_id,
            'extraction_method': insert(MemoryItem).excluded.extraction_method,
            'updated_at': func.now(),
        }

        if candidate.workspace_id is not None:
            conflict_kwargs = {
                "index_elements": ['user_id', 'workspace_id', 'canonical_key'],
                "index_where": text("status = 'ACTIVE' AND workspace_id IS NOT NULL"),
            }
        else:
            conflict_kwargs = {
                "index_elements": ['user_id', 'canonical_key'],
                "index_where": text("status = 'ACTIVE' AND workspace_id IS NULL"),
            }

        stmt = (
            insert(MemoryItem)
            .values(**insert_values)
            .on_conflict_do_update(
                **conflict_kwargs,
                set_=update_set,
            )
            .returning(MemoryItem)
        )

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
                workspace_id=candidate.workspace_id,
            )

        return memory_item

    @staticmethod
    async def supersede_memory(
        session: AsyncSession,
        old_memory_id: uuid.UUID,
        new_memory_id: uuid.UUID,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> bool:
        """
        Marks an old memory as SUPERSEDED and points superseded_by_id to new_memory_id.
        Preserves complete history without deleting rows.
        Requires valid user_id to guarantee strict user isolation.
        """
        if not user_id:
            raise ValueError("user_id is required for supersession")
        if not old_memory_id:
            raise ValueError("old_memory_id is required for supersession")
        if not new_memory_id:
            raise ValueError("new_memory_id is required for supersession")

        filters = [
            MemoryItem.id == old_memory_id,
            MemoryItem.user_id == user_id,
        ]
        if workspace_id is not None:
            filters.append(
                or_(
                    MemoryItem.workspace_id == workspace_id,
                    MemoryItem.workspace_id.is_(None),
                )
            )

        stmt = (
            update(MemoryItem)
            .where(*filters)
            .values(
                status="SUPERSEDED",
                superseded_by_id=new_memory_id,
                updated_at=func.now(),
            )
        )
        result = await session.execute(stmt)
        return result.rowcount > 0

    @staticmethod
    async def archive_memory(
        session: AsyncSession,
        memory_id: uuid.UUID,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> bool:
        """
        Archives a memory item (status='ARCHIVED').
        Requires matching user_id to enforce authorization boundary.
        """
        if not user_id:
            raise ValueError("user_id is required to archive memory")
        if not memory_id:
            raise ValueError("memory_id is required to archive memory")

        filters = [
            MemoryItem.id == memory_id,
            MemoryItem.user_id == user_id,
            MemoryItem.status == "ACTIVE",
        ]
        if workspace_id is not None:
            filters.append(
                or_(MemoryItem.workspace_id == workspace_id, MemoryItem.workspace_id.is_(None))
            )

        stmt = (
            update(MemoryItem)
            .where(*filters)
            .values(
                status="ARCHIVED",
                updated_at=func.now(),
            )
        )
        result = await session.execute(stmt)
        return result.rowcount > 0

    @staticmethod
    async def update_memory(
        session: AsyncSession,
        memory_id: uuid.UUID,
        user_id: uuid.UUID,
        content: Optional[str] = None,
        structured_data: Optional[dict] = None,
        confidence: Optional[float] = None,
        auto_embed: bool = True,
    ) -> Optional[MemoryItem]:
        """
        Updates content, structured_data, and/or confidence of an existing ACTIVE memory item.
        Regenerates embedding if content is modified.
        Strictly enforces user_id ownership.
        """
        if not user_id:
            raise ValueError("user_id is required to update memory")

        item = await session.scalar(
            select(MemoryItem).where(
                MemoryItem.id == memory_id,
                MemoryItem.user_id == user_id,
                MemoryItem.status == "ACTIVE",
            )
        )
        if not item:
            return None

        if content is not None and content.strip():
            clean_content = content.strip()
            item.content = clean_content
            if auto_embed:
                item.embedding = await MemoryMaterializer.generate_embedding(clean_content)

        if structured_data is not None:
            existing_sd = dict(item.structured_data or {})
            existing_sd.update(structured_data)
            item.structured_data = existing_sd

        if confidence is not None:
            item.confidence = confidence

        item.updated_at = func.now()
        await session.flush()
        await session.refresh(item)
        return item

    @staticmethod
    async def get_user_memories(
        session: AsyncSession,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
        status: Optional[str] = "ACTIVE",
        memory_type: Optional[str] = None,
        include_global: bool = True,
        limit: Optional[int] = None,
    ) -> Sequence[MemoryItem]:
        """
        Queries memories strictly isolated to user_id.
        - If workspace_id is provided and include_global is True: returns workspace memories + global memories.
        - If workspace_id is provided and include_global is False: returns strictly workspace memories.
        - If workspace_id is None: returns all active memories for this user (or only global if include_global=False).
        """
        if not user_id:
            raise ValueError("user_id is required")

        stmt = select(MemoryItem).where(MemoryItem.user_id == user_id)

        if workspace_id is not None:
            if include_global:
                stmt = stmt.where(
                    or_(MemoryItem.workspace_id == workspace_id, MemoryItem.workspace_id.is_(None))
                )
            else:
                stmt = stmt.where(MemoryItem.workspace_id == workspace_id)
        elif not include_global:
            stmt = stmt.where(MemoryItem.workspace_id.is_(None))

        if status is not None:
            stmt = stmt.where(MemoryItem.status == status)
        if memory_type is not None:
            stmt = stmt.where(MemoryItem.memory_type == memory_type)

        stmt = stmt.order_by(MemoryItem.created_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
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

