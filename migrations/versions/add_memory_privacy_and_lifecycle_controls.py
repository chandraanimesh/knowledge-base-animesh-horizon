"""add_memory_privacy_and_lifecycle_controls

Revision ID: add_memory_privacy_and_lifecycle_controls
Revises: add_living_articles_and_staged_updates
Create Date: 2026-11-04 17:00:00.000000

Milestone 8: Commercial Memory & Privacy Controls.
Adds:
- memory_privacy_policies
- memory_tombstones
- memory_privacy_audit_events
- updates ck_memoryitem_status constraint
- updates ck_memory_processing_job_status constraint
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "add_memory_privacy_and_lifecycle_controls"
down_revision: Union[str, Sequence[str], None] = "add_living_articles_and_staged_updates"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # 1. Create memory_privacy_policies table
    if "memory_privacy_policies" not in tables:
        op.create_table(
            "memory_privacy_policies",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("scope_type", sa.String(20), nullable=False),
            sa.Column("scope_id", UUID(as_uuid=True), nullable=False),
            sa.Column("memory_collection_enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("memory_extraction_enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("memory_dreaming_enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("memory_retrieval_enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("retention_days", sa.Integer(), nullable=True),
            sa.Column("auto_expiration_enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("user_deletion_enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("profile_scrubbing_enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("memory_export_enabled", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("user_can_pause", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint("scope_type IN ('TENANT', 'WORKSPACE', 'USER')", name="ck_memory_privacy_scope_type"),
            sa.UniqueConstraint("scope_type", "scope_id", name="uq_memory_privacy_policies_scope"),
        )
        op.create_index("ix_memory_privacy_policies_scope_type", "memory_privacy_policies", ["scope_type"])
        op.create_index("ix_memory_privacy_policies_scope_id", "memory_privacy_policies", ["scope_id"])

    # 2. Create memory_tombstones table
    if "memory_tombstones" not in tables:
        op.create_table(
            "memory_tombstones",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("canonical_key", sa.String(), nullable=False),
            sa.Column("reason", sa.String(50), nullable=False, server_default="FORGET"),
            sa.Column("section", sa.String(50), nullable=True),
            sa.Column("key", sa.String(100), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("user_id", "canonical_key", name="uq_memory_tombstones_user_key"),
        )
        op.create_index("ix_memory_tombstones_user_id", "memory_tombstones", ["user_id"])
        op.create_index("ix_memory_tombstones_canonical_key", "memory_tombstones", ["canonical_key"])

    # 3. Create memory_privacy_audit_events table
    if "memory_privacy_audit_events" not in tables:
        op.create_table(
            "memory_privacy_audit_events",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True),
            sa.Column("department_id", UUID(as_uuid=True), sa.ForeignKey("departments.id", ondelete="SET NULL"), nullable=True),
            sa.Column("action", sa.String(50), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="SUCCESS"),
            sa.Column("metadata_", JSONB, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.CheckConstraint(
                "action IN ('POLICY_UPDATE', 'MEMORY_PAUSE', 'MEMORY_RESUME', 'MEMORY_FORGET', 'MEMORY_DELETE_ALL', 'PROFILE_SCRUB', 'RETENTION_SWEEP', 'MEMORY_EXPORT')",
                name="ck_memory_privacy_audit_action",
            ),
            sa.CheckConstraint("status IN ('SUCCESS', 'FAILED')", name="ck_memory_privacy_audit_status"),
        )
        op.create_index("ix_memory_privacy_audit_events_user_id", "memory_privacy_audit_events", ["user_id"])
        op.create_index("ix_memory_privacy_audit_events_workspace_id", "memory_privacy_audit_events", ["workspace_id"])
        op.create_index("ix_memory_privacy_audit_events_department_id", "memory_privacy_audit_events", ["department_id"])
        op.create_index("ix_memory_privacy_audit_events_action", "memory_privacy_audit_events", ["action"])
        op.create_index("ix_memory_privacy_audit_events_created_at", "memory_privacy_audit_events", ["created_at"])

    # 4. Safely update constraints on memory_items and memory_processing_jobs
    try:
        op.drop_constraint("ck_memoryitem_status", "memory_items", type_="check")
    except Exception:
        pass
    op.create_check_constraint(
        "ck_memoryitem_status",
        "memory_items",
        "status IN ('ACTIVE', 'SUPERSEDED', 'ARCHIVED', 'EXPIRED', 'DELETION_REQUESTED', 'DELETION_PROCESSING', 'PURGED')",
    )

    try:
        op.drop_constraint("ck_memory_processing_job_status", "memory_processing_jobs", type_="check")
    except Exception:
        pass
    op.create_check_constraint(
        "ck_memory_processing_job_status",
        "memory_processing_jobs",
        "status IN ('PENDING', 'PROCESSING', 'SUCCEEDED', 'RETRY_PENDING', 'DEAD_LETTER', 'SKIPPED_NOT_MEMORABLE', 'SKIPPED_POLICY_PAUSED')",
    )


def downgrade() -> None:
    try:
        op.drop_constraint("ck_memory_processing_job_status", "memory_processing_jobs", type_="check")
        op.create_check_constraint(
            "ck_memory_processing_job_status",
            "memory_processing_jobs",
            "status IN ('PENDING', 'PROCESSING', 'SUCCEEDED', 'RETRY_PENDING', 'DEAD_LETTER', 'SKIPPED_NOT_MEMORABLE')",
        )
    except Exception:
        pass

    try:
        op.drop_constraint("ck_memoryitem_status", "memory_items", type_="check")
        op.create_check_constraint(
            "ck_memoryitem_status",
            "memory_items",
            "status IN ('ACTIVE', 'SUPERSEDED', 'ARCHIVED', 'EXPIRED')",
        )
    except Exception:
        pass

    op.drop_table("memory_privacy_audit_events")
    op.drop_table("memory_tombstones")
    op.drop_table("memory_privacy_policies")

