"""Memory Retention and Expiration Engine

Executes automated lifecycle sweeps to expire memories past retention limits,
clean up vector embeddings, and recompile derived user profiles idempotently.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from sqlalchemy import select, update, or_, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import MemoryItem
from rag_app.memory.privacy_service import MemoryPrivacyService
from rag_app.memory.profile_compiler import compile_user_profile

logger = logging.getLogger(__name__)


class MemoryRetentionService:
    """
    Manages memory retention lifecycles and sweeps expired memories.
    """

    @classmethod
    def calculate_retention_expiration(
        cls,
        created_at: datetime,
        retention_days: Optional[int],
    ) -> Optional[datetime]:
        """Calculates retention expiration timestamp from created_at and retention_days."""
        if not retention_days or retention_days <= 0:
            return None
        return created_at + timedelta(days=retention_days)

    @classmethod
    async def run_retention_sweep(
        cls,
        session: AsyncSession,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Executes an idempotent retention sweep across all active memories:
        1. Finds memories with retention_expires_at <= now.
        2. Evaluates dynamic retention_days from effective privacy policies.
        3. Transitions expired memories to EXPIRED status and cleans up embeddings.
        4. Recompiles living profiles for all affected users.
        """
        sweep_time = now or datetime.now(timezone.utc)
        logger.info(
            "starting_memory_retention_sweep",
            extra={"event": "starting_memory_retention_sweep", "sweep_time": sweep_time.isoformat()},
        )

        # 1. Direct expiration via retention_expires_at
        direct_stmt = (
            select(MemoryItem)
            .where(
                MemoryItem.status == "ACTIVE",
                MemoryItem.retention_expires_at.is_not(None),
                MemoryItem.retention_expires_at <= sweep_time,
            )
        )
        direct_expired = (await session.execute(direct_stmt)).scalars().all()

        # 2. Dynamic policy check for remaining active memories
        active_stmt = (
            select(MemoryItem)
            .where(
                MemoryItem.status == "ACTIVE",
                or_(
                    MemoryItem.retention_expires_at.is_(None),
                    MemoryItem.retention_expires_at > sweep_time,
                ),
            )
        )
        active_candidates = (await session.execute(active_stmt)).scalars().all()

        policy_expired: list[MemoryItem] = []
        user_policy_cache: dict[uuid.UUID, Any] = {}

        for mem in active_candidates:
            if mem.user_id not in user_policy_cache:
                policy = await MemoryPrivacyService.get_effective_policy(
                    session=session,
                    user_id=mem.user_id,
                    workspace_id=mem.workspace_id,
                )
                user_policy_cache[mem.user_id] = policy

            eff_policy = user_policy_cache[mem.user_id]
            if eff_policy.retention_days is not None and eff_policy.auto_expiration_enabled:
                cutoff = mem.created_at + timedelta(days=eff_policy.retention_days)
                if cutoff <= sweep_time:
                    policy_expired.append(mem)

        all_expired = list({m.id: m for m in (direct_expired + policy_expired)}.values())

        if not all_expired:
            return {
                "sweep_time": sweep_time.isoformat(),
                "expired_count": 0,
                "affected_users_count": 0,
                "status": "NO_EXPIRED_MEMORIES",
            }

        expired_ids = [m.id for m in all_expired]
        affected_user_ids = list({m.user_id for m in all_expired})

        # 3. Transition to EXPIRED and zero out embeddings
        await session.execute(
            update(MemoryItem)
            .where(MemoryItem.id.in_(expired_ids))
            .values(
                status="EXPIRED",
                embedding=None,
                updated_at=sweep_time,
            )
        )
        await session.flush()

        # 4. Invalidate and recompile living user profiles for affected users
        for u_id in affected_user_ids:
            try:
                await compile_user_profile(user_id=u_id, session=session)
            except Exception as comp_err:
                logger.warning(f"Error recompiling profile for user {u_id} during sweep: {comp_err}")

            # Audit event per affected user
            await MemoryPrivacyService.log_audit_event(
                session=session,
                user_id=u_id,
                action="RETENTION_SWEEP",
                status="SUCCESS",
                metadata={"expired_records": len([m for m in all_expired if m.user_id == u_id])},
            )

        await session.commit()

        logger.info(
            "memory_retention_sweep_completed",
            extra={
                "event": "memory_retention_sweep_completed",
                "expired_count": len(all_expired),
                "affected_users": len(affected_user_ids),
            },
        )

        return {
            "sweep_time": sweep_time.isoformat(),
            "expired_count": len(all_expired),
            "affected_users_count": len(affected_user_ids),
            "status": "SUCCESS",
        }

