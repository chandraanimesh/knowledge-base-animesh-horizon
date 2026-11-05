"""add_intelligent_memory_and_autonomous_dreaming

Revision ID: add_intelligent_memory_and_autonomous_dreaming
Revises: add_memory_privacy_and_lifecycle_controls
Create Date: 2026-11-04 18:40:00.000000

Adds structured memory types (EPISODIC, SEMANTIC, PROCEDURAL),
memory relationships table for consolidation and temporal intelligence,
autonomous dreaming runs tracking table, and memory evaluation metrics table.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "add_intelligent_memory_and_autonomous_dreaming"
down_revision: Union[str, Sequence[str], None] = "add_memory_privacy_and_lifecycle_controls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # 1. Expand ck_memoryitem_type constraint on memory_items to include EPISODIC, SEMANTIC, PROCEDURAL
    # Drop existing constraint safely if present
    op.execute(
        sa.text("ALTER TABLE memory_items DROP CONSTRAINT IF EXISTS ck_memoryitem_type;")
    )
    op.execute(
        sa.text(
            "ALTER TABLE memory_items ADD CONSTRAINT ck_memoryitem_type "
            "CHECK (memory_type IN ('STATIC', 'DYNAMIC', 'EPISODIC', 'SEMANTIC', 'PROCEDURAL'));"
        )
    )

    # 2. Create memory_relationships table
    if "memory_relationships" not in tables:
        op.create_table(
            "memory_relationships",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "source_memory_id",
                UUID(as_uuid=True),
                sa.ForeignKey("memory_items.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "target_memory_id",
                UUID(as_uuid=True),
                sa.ForeignKey("memory_items.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column(
                "relationship_type",
                sa.String(50),
                sa.CheckConstraint(
                    "relationship_type IN ('DUPLICATE', 'COMPLEMENTS', 'SUPERSEDES', 'CONTRADICTS', 'RELATED', 'UNRESOLVED')",
                    name="ck_memory_relationship_type",
                ),
                nullable=False,
                index=True,
            ),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("source_evidence", JSONB(), nullable=True, server_default=sa.text("'{}'::jsonb")),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
        op.create_index(
            "ix_memory_relationships_source_target",
            "memory_relationships",
            ["source_memory_id", "target_memory_id"],
        )

    # 3. Create memory_dreaming_runs table
    if "memory_dreaming_runs" not in tables:
        op.create_table(
            "memory_dreaming_runs",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("run_id", sa.String(100), unique=True, nullable=False, index=True),
            sa.Column(
                "user_id",
                UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=True,
                index=True,
            ),
            sa.Column(
                "workspace_id",
                UUID(as_uuid=True),
                sa.ForeignKey("workspaces.id", ondelete="SET NULL"),
                nullable=True,
                index=True,
            ),
            sa.Column("trigger_type", sa.String(50), nullable=False, server_default="SCHEDULED"),
            sa.Column(
                "status",
                sa.String(50),
                sa.CheckConstraint(
                    "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'PARTIALLY_COMPLETED', 'FAILED', 'CANCELLED', 'SKIPPED_POLICY_PAUSED')",
                    name="ck_memory_dreaming_run_status",
                ),
                nullable=False,
                server_default="PENDING",
                index=True,
            ),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("memories_inspected", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("memories_skipped", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("duplicates_detected", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("conflicts_detected", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("new_memories_created", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("memories_superseded", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("profile_updated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("token_usage", JSONB, nullable=False, server_default="{}"),
            sa.Column("checkpoint_data", JSONB, nullable=False, server_default="{}"),
            sa.Column("duration_seconds", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
        op.create_index(
            "ix_memory_dreaming_runs_user_status",
            "memory_dreaming_runs",
            ["user_id", "status"],
        )

    # 4. Create memory_eval_metrics table
    if "memory_eval_metrics" not in tables:
        op.create_table(
            "memory_eval_metrics",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("eval_run_id", sa.String(100), nullable=False, index=True),
            sa.Column("scenario_name", sa.String(100), nullable=False, index=True),
            sa.Column("metric_category", sa.String(50), nullable=False, index=True),
            sa.Column("metrics_json", JSONB, nullable=False, server_default="{}"),
            sa.Column("passed", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("details", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
        op.create_index(
            "ix_memory_eval_metrics_run_category",
            "memory_eval_metrics",
            ["eval_run_id", "metric_category"],
        )


def downgrade() -> None:
    op.drop_table("memory_eval_metrics")
    op.drop_table("memory_dreaming_runs")
    op.drop_table("memory_relationships")
    op.execute(
        sa.text("ALTER TABLE memory_items DROP CONSTRAINT IF EXISTS ck_memoryitem_type;")
    )
    op.execute(
        sa.text(
            "ALTER TABLE memory_items ADD CONSTRAINT ck_memoryitem_type "
            "CHECK (memory_type IN ('STATIC', 'DYNAMIC'));"
        )
    )
