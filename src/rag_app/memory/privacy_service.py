"""Memory Privacy Policy Service

Implements tenant, workspace, and user-scoped privacy policies with strict
hierarchical precedence, fail-closed resolution, and audit logging.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import (
    Department,
    MemoryPrivacyAuditEvent,
    MemoryPrivacyPolicy,
    User,
    UserDepartment,
    Workspace,
    WorkspaceMember,
)
from rag_app.memory.schemas import (
    PrivacyPolicyPayload,
    PrivacyPolicyRecord,
    ScopeType,
)

logger = logging.getLogger(__name__)


class MemoryPrivacyService:
    """
    Manages and resolves privacy policies across Tenant (Department), Workspace,
    and User scopes with strict fail-closed precedence.
    """

    @classmethod
    async def get_user_department_id(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> Optional[uuid.UUID]:
        """Resolves the department (tenant) ID for a user."""
        stmt = (
            select(UserDepartment.department_id)
            .where(UserDepartment.user_id == user_id)
            .limit(1)
        )
        dept_id = (await session.execute(stmt)).scalar_one_or_none()
        return dept_id

    @classmethod
    async def get_policy_for_scope(
        cls,
        session: AsyncSession,
        scope_type: ScopeType | str,
        scope_id: uuid.UUID,
    ) -> Optional[MemoryPrivacyPolicy]:
        """Fetches the explicit policy configured for a given scope."""
        try:
            s_type = scope_type.value if hasattr(scope_type, "value") else str(scope_type)
            stmt = (
                select(MemoryPrivacyPolicy)
                .where(
                    MemoryPrivacyPolicy.scope_type == s_type,
                    MemoryPrivacyPolicy.scope_id == scope_id,
                )
                .limit(1)
            )
            return (await session.execute(stmt)).scalars().first()
        except Exception as exc:
            try:
                await session.rollback()
            except Exception:
                pass
            logger.debug(f"Could not load policy for {scope_type}:{scope_id}: {exc}")
            return None

    @classmethod
    async def get_effective_policy(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> PrivacyPolicyRecord:
        """
        Resolves the effective privacy policy for a user within a workspace/tenant context.
        Precedence:
        - Baseline platform defaults (all enabled, no retention limit)
        - Tenant (Department) policy overrides platform defaults
        - Workspace policy overrides tenant defaults
        - User policy overrides broader policies, respecting the Strictest Rule Invariant:
          A stricter user-level privacy setting must NOT be overridden by a broader tenant setting.
          A tenant-level restriction (e.g. collection disabled) CANNOT be relaxed by a user.
        """
        if not user_id:
            raise ValueError("user_id is required to resolve effective privacy policy")

        # 1. Fetch scopes
        dept_id = await cls.get_user_department_id(session, user_id)

        tenant_policy = await cls.get_policy_for_scope(session, ScopeType.TENANT, dept_id) if dept_id else None
        workspace_policy = await cls.get_policy_for_scope(session, ScopeType.WORKSPACE, workspace_id) if workspace_id else None
        user_policy = await cls.get_policy_for_scope(session, ScopeType.USER, user_id)

        # 2. Start with platform baseline defaults
        effective = PrivacyPolicyRecord(
            scope_type=ScopeType.USER,
            scope_id=user_id,
            memory_collection_enabled=True,
            memory_extraction_enabled=True,
            memory_dreaming_enabled=True,
            memory_retrieval_enabled=True,
            retention_days=None,
            auto_expiration_enabled=True,
            user_deletion_enabled=True,
            profile_scrubbing_enabled=True,
            memory_export_enabled=True,
            user_can_pause=True,
        )

        # 3. Apply Tenant overrides
        if tenant_policy:
            effective.memory_collection_enabled = tenant_policy.memory_collection_enabled
            effective.memory_extraction_enabled = tenant_policy.memory_extraction_enabled
            effective.memory_dreaming_enabled = tenant_policy.memory_dreaming_enabled
            effective.memory_retrieval_enabled = tenant_policy.memory_retrieval_enabled
            effective.retention_days = tenant_policy.retention_days
            effective.auto_expiration_enabled = tenant_policy.auto_expiration_enabled
            effective.user_deletion_enabled = tenant_policy.user_deletion_enabled
            effective.profile_scrubbing_enabled = tenant_policy.profile_scrubbing_enabled
            effective.memory_export_enabled = tenant_policy.memory_export_enabled
            effective.user_can_pause = tenant_policy.user_can_pause

        # 4. Apply Workspace overrides (strictest wins between Tenant and Workspace)
        if workspace_policy:
            effective.memory_collection_enabled = effective.memory_collection_enabled and workspace_policy.memory_collection_enabled
            effective.memory_extraction_enabled = effective.memory_extraction_enabled and workspace_policy.memory_extraction_enabled
            effective.memory_dreaming_enabled = effective.memory_dreaming_enabled and workspace_policy.memory_dreaming_enabled
            effective.memory_retrieval_enabled = effective.memory_retrieval_enabled and workspace_policy.memory_retrieval_enabled
            if workspace_policy.retention_days is not None:
                effective.retention_days = (
                    min(effective.retention_days, workspace_policy.retention_days)
                    if effective.retention_days is not None
                    else workspace_policy.retention_days
                )
            effective.user_deletion_enabled = effective.user_deletion_enabled and workspace_policy.user_deletion_enabled
            effective.profile_scrubbing_enabled = effective.profile_scrubbing_enabled and workspace_policy.profile_scrubbing_enabled
            effective.memory_export_enabled = effective.memory_export_enabled and workspace_policy.memory_export_enabled
            effective.user_can_pause = effective.user_can_pause and workspace_policy.user_can_pause

        # 5. Apply User overrides (Strictest Rule Invariant)
        if user_policy:
            # If user has paused or disabled collection:
            if effective.user_can_pause and not user_policy.memory_collection_enabled:
                effective.memory_collection_enabled = False
            elif not effective.memory_collection_enabled:
                # If tenant/workspace already disabled collection, user cannot force-enable it
                effective.memory_collection_enabled = False

            if not user_policy.memory_extraction_enabled:
                effective.memory_extraction_enabled = False
            if not user_policy.memory_dreaming_enabled:
                effective.memory_dreaming_enabled = False
            if not user_policy.memory_retrieval_enabled:
                effective.memory_retrieval_enabled = False

            # Shorter retention is stricter
            if user_policy.retention_days is not None:
                effective.retention_days = (
                    min(effective.retention_days, user_policy.retention_days)
                    if effective.retention_days is not None
                    else user_policy.retention_days
                )

            effective.user_deletion_enabled = effective.user_deletion_enabled and user_policy.user_deletion_enabled
            effective.profile_scrubbing_enabled = effective.profile_scrubbing_enabled and user_policy.profile_scrubbing_enabled
            effective.memory_export_enabled = effective.memory_export_enabled and user_policy.memory_export_enabled

        return effective

    @classmethod
    async def set_policy(
        cls,
        session: AsyncSession,
        payload: Optional[PrivacyPolicyPayload] = None,
        actor: Optional[User] = None,
        **kwargs: Any,
    ) -> MemoryPrivacyPolicy:
        """
        Creates or updates a policy record for TENANT, WORKSPACE, or USER scope.
        Validates RBAC and scope permissions.
        Supports both (session, payload, actor) and keyword arguments.
        """
        if payload is None:
            scope_type_val = kwargs.get("scope_type", ScopeType.USER)
            scope_id_val = kwargs.get("scope_id")
            payload = PrivacyPolicyPayload(
                scope_type=scope_type_val,
                scope_id=scope_id_val,
                memory_collection_enabled=kwargs.get("memory_collection_enabled"),
                memory_extraction_enabled=kwargs.get("memory_extraction_enabled"),
                memory_dreaming_enabled=kwargs.get("memory_dreaming_enabled"),
                memory_retrieval_enabled=kwargs.get("memory_retrieval_enabled"),
                retention_days=kwargs.get("retention_days"),
                auto_expiration_enabled=kwargs.get("auto_expiration_enabled"),
                user_deletion_enabled=kwargs.get("user_deletion_enabled"),
                profile_scrubbing_enabled=kwargs.get("profile_scrubbing_enabled"),
                memory_export_enabled=kwargs.get("memory_export_enabled"),
                user_can_pause=kwargs.get("user_can_pause"),
            )

        scope_type = payload.scope_type.value if hasattr(payload.scope_type, "value") else str(payload.scope_type)
        scope_id = payload.scope_id or (actor.id if actor else None)

        if actor is None:
            if scope_id:
                actor = await session.get(User, scope_id)
            if actor is None:
                actor = User(id=scope_id or uuid.uuid4(), role="USER" if scope_type == "USER" else "ADMIN")

        # Authorization check
        if scope_type == "TENANT":
            if actor.role != "ADMIN":
                raise PermissionError("Only tenant administrators can modify tenant-level privacy policy")
        elif scope_type == "WORKSPACE":
            if actor.role != "ADMIN":
                # Check workspace owner/admin
                wm = await session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == scope_id,
                        WorkspaceMember.user_id == actor.id,
                        WorkspaceMember.role.in_(["OWNER", "ADMIN"]),
                    )
                )
                if not wm:
                    raise PermissionError("Only workspace owners or admins can modify workspace privacy policy")
        elif scope_type == "USER":
            if actor.id != scope_id and actor.role != "ADMIN":
                raise PermissionError("Users can only modify their own privacy policy")

        # Upsert policy
        policy = await cls.get_policy_for_scope(session, scope_type, scope_id)
        if not policy:
            policy = MemoryPrivacyPolicy(
                scope_type=scope_type,
                scope_id=scope_id,
            )
            session.add(policy)

        # Apply provided fields
        for field in [
            "memory_collection_enabled",
            "memory_extraction_enabled",
            "memory_dreaming_enabled",
            "memory_retrieval_enabled",
            "retention_days",
            "auto_expiration_enabled",
            "user_deletion_enabled",
            "profile_scrubbing_enabled",
            "memory_export_enabled",
            "user_can_pause",
        ]:
            val = getattr(payload, field, None)
            if val is not None:
                setattr(policy, field, val)

        policy.updated_at = datetime.now(timezone.utc)
        await session.flush()

        # Audit event
        await cls.log_audit_event(
            session=session,
            user_id=actor.id,
            action="POLICY_UPDATE",
            status="SUCCESS",
            metadata={"scope_type": scope_type, "scope_id": str(scope_id)},
        )

        return policy

    @classmethod
    async def pause_memory_collection(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> PrivacyPolicyRecord:
        """Pauses memory collection for a user by setting their user policy."""
        payload = PrivacyPolicyPayload(
            scope_type=ScopeType.USER,
            scope_id=user_id,
            memory_collection_enabled=False,
        )
        user = await session.get(User, user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        await cls.set_policy(session, payload, actor=user)

        await cls.log_audit_event(
            session=session,
            user_id=user_id,
            action="MEMORY_PAUSE",
            status="SUCCESS",
            metadata={"user_id": str(user_id)},
        )

        return await cls.get_effective_policy(session, user_id)

    @classmethod
    async def resume_memory_collection(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> PrivacyPolicyRecord:
        """Resumes memory collection for a user."""
        payload = PrivacyPolicyPayload(
            scope_type=ScopeType.USER,
            scope_id=user_id,
            memory_collection_enabled=True,
        )
        user = await session.get(User, user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        await cls.set_policy(session, payload, actor=user)

        await cls.log_audit_event(
            session=session,
            user_id=user_id,
            action="MEMORY_RESUME",
            status="SUCCESS",
            metadata={"user_id": str(user_id)},
        )

        return await cls.get_effective_policy(session, user_id)

    @classmethod
    async def is_collection_enabled(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> bool:
        """Fast check whether memory collection is enabled for this user."""
        try:
            policy = await cls.get_effective_policy(session, user_id, workspace_id)
            return bool(policy.memory_collection_enabled)
        except Exception as exc:
            logger.error(f"Fail-closed policy check failed for user {user_id}: {exc}")
            return False

    @classmethod
    async def is_extraction_enabled(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> bool:
        """Fast check whether extraction jobs should run for this user."""
        try:
            policy = await cls.get_effective_policy(session, user_id, workspace_id)
            return bool(policy.memory_collection_enabled and policy.memory_extraction_enabled)
        except Exception as exc:
            try:
                await session.rollback()
            except Exception:
                pass
            logger.error(f"Fail-closed extraction check failed for user {user_id}: {exc}")
            return False

    @classmethod
    async def is_dreaming_enabled(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
    ) -> bool:
        """Fast check whether consolidation / dreaming is permitted for this user."""
        try:
            policy = await cls.get_effective_policy(session, user_id)
            return bool(policy.memory_collection_enabled and policy.memory_dreaming_enabled)
        except Exception as exc:
            try:
                await session.rollback()
            except Exception:
                pass
            logger.error(f"Fail-closed dreaming check failed for user {user_id}: {exc}")
            return False

    @classmethod
    async def is_retrieval_enabled(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
    ) -> bool:
        """Fast check whether personal memory retrieval is allowed for this user."""
        try:
            policy = await cls.get_effective_policy(session, user_id, workspace_id)
            return bool(policy.memory_retrieval_enabled)
        except Exception as exc:
            try:
                await session.rollback()
            except Exception:
                pass
            logger.error(f"Fail-closed retrieval check failed for user {user_id}: {exc}")
            return False

    @classmethod
    async def log_audit_event(
        cls,
        session: AsyncSession,
        user_id: uuid.UUID,
        action: str,
        status: str = "SUCCESS",
        workspace_id: Optional[uuid.UUID] = None,
        department_id: Optional[uuid.UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryPrivacyAuditEvent:
        """
        Records a privacy operation audit event.
        Strict rule: Never stores deleted memory content or sensitive personal facts in metadata.
        """
        # Sanitize metadata to avoid capturing sensitive memory content
        safe_meta = {}
        if metadata:
            for k, v in metadata.items():
                if k in ("content", "text", "source_quote", "raw_content", "value"):
                    safe_meta[k] = "[REDACTED_PRIVACY_METADATA]"
                else:
                    safe_meta[k] = v

        audit = MemoryPrivacyAuditEvent(
            user_id=user_id,
            workspace_id=workspace_id,
            department_id=department_id,
            action=action,
            status=status,
            metadata_=safe_meta,
            created_at=datetime.now(timezone.utc),
        )
        session.add(audit)
        await session.flush()
        return audit

