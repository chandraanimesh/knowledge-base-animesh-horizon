"""User Privacy Operations & Anti-Resurrection Engine

Implements individual user controls:
- Forget selected memory
- Delete all personal memories
- Granular commercial profile scrubbing
- Export memory
- Tombstone registration to prevent memory resurrection
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update, delete, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import (
    MemoryItem,
    MemoryPrivacyPolicy,
    MemoryProcessingJob,
    MemoryTombstone,
    User,
)
from rag_app.memory.privacy_service import MemoryPrivacyService
from rag_app.memory.profile_compiler import (
    _extract_category_and_data,
    compile_user_profile,
)
from rag_app.memory.schemas import (
    JobStatus,
    MemoryExportItem,
    MemoryExportResponse,
    MemoryStatus,
)

logger = logging.getLogger(__name__)


class MemoryPrivacyOperations:
    """
    Handles user memory deletion, profile scrubbing, export, and anti-resurrection guarantees.
    """

    @classmethod
    async def is_canonical_key_tombstoned(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        canonical_key: str,
    ) -> bool:
        """Checks if a canonical memory key has been tombstoned for this user."""
        stmt = (
            select(MemoryTombstone)
            .where(
                MemoryTombstone.user_id == user_id,
                MemoryTombstone.canonical_key == canonical_key,
            )
            .limit(1)
        )
        return (await session.execute(stmt)).scalars().first() is not None

    @classmethod
    async def register_tombstone(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        canonical_key: str,
        reason: str = "FORGET",
        section: Optional[str] = None,
        key: Optional[str] = None,
    ) -> MemoryTombstone:
        """
        Registers an immutable tombstone preventing resurrection of deleted facts
        from delayed jobs, dreaming, or cached inputs.
        """
        existing = await session.scalar(
            select(MemoryTombstone).where(
                MemoryTombstone.user_id == user_id,
                MemoryTombstone.canonical_key == canonical_key,
            )
        )
        if existing:
            return existing

        tombstone = MemoryTombstone(
            user_id=user_id,
            canonical_key=canonical_key,
            reason=reason,
            section=section,
            key=key,
            created_at=datetime.now(timezone.utc),
        )
        session.add(tombstone)
        await session.flush()
        return tombstone

    @classmethod
    async def forget_memory(
        cls,
        session: AsyncSession,
        memory_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """
        Forgets a selected memory record:
        - Validates user ownership
        - Sets status to PURGED and zeroes embedding
        - Creates a persistent MemoryTombstone
        - Inactivates any pending jobs that could recreate this memory
        - Recompiles living user profile without this memory
        """
        mem = await session.get(MemoryItem, memory_id)
        if not mem or mem.user_id != user_id:
            return False

        # Check policy
        policy = await MemoryPrivacyService.get_effective_policy(session, user_id, mem.workspace_id)
        if not policy.user_deletion_enabled:
            raise PermissionError("User memory deletion is disabled by policy")

        # 1. Register anti-resurrection tombstone
        await cls.register_tombstone(
            session=session,
            user_id=user_id,
            canonical_key=mem.canonical_key,
            reason="FORGET",
        )

        # 2. Mark PURGED and zero out embedding
        mem.status = "PURGED"
        mem.embedding = None
        mem.content = "[DELETED_BY_USER]"
        mem.updated_at = datetime.now(timezone.utc)
        await session.flush()

        # 3. Suppress pending jobs for this user referencing this memory or job
        if mem.source_job_id:
            await session.execute(
                update(MemoryProcessingJob)
                .where(
                    MemoryProcessingJob.id == mem.source_job_id,
                    MemoryProcessingJob.status.in_(["PENDING", "PROCESSING", "RETRY_PENDING"]),
                )
                .values(
                    status="SKIPPED_POLICY_PAUSED",
                    error_message="Job cancelled: source memory forgotten by user",
                    updated_at=datetime.now(timezone.utc),
                )
            )

        # 4. Recompile user living profile
        await compile_user_profile(user_id=user_id, session=session)

        # 5. Audit event
        await MemoryPrivacyService.log_audit_event(
            session=session,
            user_id=user_id,
            action="MEMORY_FORGET",
            status="SUCCESS",
            workspace_id=mem.workspace_id,
            metadata={"memory_id": str(memory_id), "canonical_key": mem.canonical_key},
        )

        return True

    @classmethod
    async def delete_memory_by_id(
        cls,
        session: AsyncSession,
        memory_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> dict[str, Any]:
        """
        Convenience method for deleting/forgetting a memory by ID.
        Invokes forget_memory, guaranteeing anti-resurrection tombstoning and profile update.
        """
        success = await cls.forget_memory(session=session, memory_id=memory_id, user_id=user_id)
        return {
            "status": "DELETED" if success else "NOT_FOUND",
            "memory_id": str(memory_id),
            "user_id": str(user_id),
        }

    @classmethod
    async def delete_all_personal_memories(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> int:
        """
        User-authorized operation to delete all personal memories:
        - Strictly scoped to user_id (and optional workspace_id)
        - Never touches shared organizational knowledge, canonical documents, or CRM records
        - Zeros all embeddings, marks status PURGED, registers tombstones
        - Resets living user profile to empty dict
        - Cancels all pending memory processing jobs for this user
        """
        # Check policy
        policy = await MemoryPrivacyService.get_effective_policy(session, user_id, workspace_id)
        if not policy.user_deletion_enabled:
            raise PermissionError("User memory deletion is disabled by policy")

        query = select(MemoryItem).where(
            MemoryItem.user_id == user_id,
            MemoryItem.status != "PURGED",
        )
        if workspace_id is not None:
            query = query.where(MemoryItem.workspace_id == workspace_id)

        memories = (await session.execute(query)).scalars().all()

        deleted_count = 0
        now = datetime.now(timezone.utc)
        for mem in memories:
            # Register tombstone for each fact
            await cls.register_tombstone(
                session=session,
                user_id=user_id,
                canonical_key=mem.canonical_key,
                reason="DELETE_ALL",
            )
            mem.status = "PURGED"
            mem.embedding = None
            mem.content = "[DELETED_BY_USER]"
            mem.updated_at = now
            deleted_count += 1

        # Cancel all pending memory jobs for user
        job_stmt = (
            update(MemoryProcessingJob)
            .where(
                MemoryProcessingJob.owner_user_id == user_id,
                MemoryProcessingJob.status.in_(["PENDING", "PROCESSING", "RETRY_PENDING"]),
            )
            .values(
                status="SKIPPED_POLICY_PAUSED",
                error_message="Job cancelled: user deleted all personal memory",
                updated_at=now,
            )
        )
        if workspace_id is not None:
            job_stmt = job_stmt.where(MemoryProcessingJob.workspace_id == workspace_id)
        await session.execute(job_stmt)

        # Clear living user profile
        user = await session.get(User, user_id)
        if user:
            user.profile_json = {}
            await session.flush()

        # Audit event
        await MemoryPrivacyService.log_audit_event(
            session=session,
            user_id=user_id,
            action="MEMORY_DELETE_ALL",
            status="SUCCESS",
            workspace_id=workspace_id,
            metadata={"deleted_count": deleted_count},
        )

        return deleted_count

    @classmethod
    async def scrub_user_profile(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        section: str,
        key: str,
    ) -> dict[str, Any]:
        """
        Granular commercial user profile scrubbing:
        - Identifies all active memories supporting the targeted (section, key)
        - Purges them, zeroes embeddings, registers tombstones
        - Rebuilds living user profile from remaining valid memories
        - Guarantees the scrubbed attribute is removed and cannot be resurrected
        """
        policy = await MemoryPrivacyService.get_effective_policy(session, user_id)
        if not policy.profile_scrubbing_enabled:
            raise PermissionError("Profile scrubbing is disabled by policy")

        # 1. Fetch active memories for user
        active_stmt = select(MemoryItem).where(
            MemoryItem.user_id == user_id,
            MemoryItem.status == "ACTIVE",
        )
        active_mems = (await session.execute(active_stmt)).scalars().all()

        scrubbed_mem_ids: list[uuid.UUID] = []
        now = datetime.now(timezone.utc)

        for mem in active_mems:
            sec, k, _ = _extract_category_and_data(
                canonical_key=mem.canonical_key,
                content=mem.content,
                structured_data=mem.structured_data,
            )
            if sec == section and k == key:
                # This memory supports the scrubbed attribute!
                scrubbed_mem_ids.append(mem.id)
                await cls.register_tombstone(
                    session=session,
                    user_id=user_id,
                    canonical_key=mem.canonical_key,
                    reason="PROFILE_SCRUB",
                    section=section,
                    key=key,
                )
                mem.status = "PURGED"
                mem.embedding = None
                mem.content = "[SCRUBBED_BY_USER]"
                mem.updated_at = now

        await session.flush()

        # 2. Recompile living user profile from surviving memories
        updated_profile = await compile_user_profile(user_id=user_id, session=session)

        # 3. Explicit safety check: Ensure the attribute is removed from living profile
        user = await session.get(User, user_id)
        if user:
            existing_prof = dict(user.profile_json or {})
            if section in existing_prof and isinstance(existing_prof[section], dict):
                existing_prof[section].pop(key, None)
            for s_name, s_data in updated_profile.items():
                if isinstance(s_data, dict) and s_data:
                    existing_prof.setdefault(s_name, {}).update(s_data)
            if section in existing_prof and isinstance(existing_prof[section], dict):
                existing_prof[section].pop(key, None)
            user.profile_json = existing_prof
            await session.flush()

        # 4. Audit event (strictly metadata only, no values)
        await MemoryPrivacyService.log_audit_event(
            session=session,
            user_id=user_id,
            action="PROFILE_SCRUB",
            status="SUCCESS",
            metadata={
                "section": section,
                "key": key,
                "scrubbed_memory_records": len(scrubbed_mem_ids),
            },
        )

        return {
            "user_id": str(user_id),
            "section": section,
            "key": key,
            "scrubbed_memory_records": len(scrubbed_mem_ids),
            "updated_profile": updated_profile,
            "status": "SUCCESS",
        }

    @classmethod
    async def export_user_memories(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> MemoryExportResponse:
        """
        Exports all memory records owned by the authenticated user in portable JSON format.
        Guarantees zero leakage of other users' data or system credentials.
        """
        policy = await MemoryPrivacyService.get_effective_policy(session, user_id, workspace_id)
        if not policy.memory_export_enabled:
            raise PermissionError("Memory export is disabled by policy")

        stmt = select(MemoryItem).where(
            MemoryItem.user_id == user_id,
            MemoryItem.status != "PURGED",
        )
        if workspace_id is not None:
            stmt = stmt.where(
                or_(MemoryItem.workspace_id == workspace_id, MemoryItem.workspace_id.is_(None))
            )
        stmt = stmt.order_by(MemoryItem.created_at.asc())

        items = (await session.execute(stmt)).scalars().all()

        export_items = [
            MemoryExportItem(
                id=m.id,
                canonical_key=m.canonical_key,
                memory_type=m.memory_type,
                status=m.status,
                content=m.content,
                structured_data=m.structured_data,
                confidence=m.confidence,
                source_type=m.source_type,
                conversation_id=m.conversation_id,
                message_id=m.message_id,
                valid_from=m.valid_from,
                valid_until=m.valid_until,
                retention_expires_at=m.retention_expires_at,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in items
        ]

        await MemoryPrivacyService.log_audit_event(
            session=session,
            user_id=user_id,
            action="MEMORY_EXPORT",
            status="SUCCESS",
            workspace_id=workspace_id,
            metadata={"exported_count": len(export_items)},
        )

        return MemoryExportResponse(
            user_id=user_id,
            workspace_id=workspace_id,
            exported_at=datetime.now(timezone.utc),
            total_memories=len(export_items),
            memories=export_items,
        )

