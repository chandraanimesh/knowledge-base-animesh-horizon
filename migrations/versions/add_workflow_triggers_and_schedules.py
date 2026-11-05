"""add_workflow_triggers_and_schedules

Revision ID: add_workflow_triggers_and_schedules
Revises: add_workflow_definitions_and_versions
Create Date: 2026-11-02 15:00:00.000000

Adds transactional outbox and schedule tracking tables for Workflow Triggers:
- workflow_trigger_events: Transactional outbox for manual, webhook, and scheduled triggers
- workflow_schedule_states: State tracking for scheduled workflows (next run, window key, misfire)
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "add_workflow_triggers_and_schedules"
down_revision: Union[str, Sequence[str], None] = "add_workflow_definitions_and_versions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # 1. Create workflow_trigger_events table (Transactional Outbox)
    if "workflow_trigger_events" not in tables:
        op.create_table(
            "workflow_trigger_events",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("event_id", sa.String(255), unique=True, nullable=False),
            sa.Column("event_source", sa.String(50), nullable=False),
            sa.Column("event_name", sa.String(100), nullable=False),
            sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False),
            sa.Column("workflow_id", UUID(as_uuid=True), sa.ForeignKey("workflow_definitions.id", ondelete="CASCADE"), nullable=False),
            sa.Column("workflow_version_id", UUID(as_uuid=True), sa.ForeignKey("workflow_versions.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("payload", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("idempotency_key", sa.String(255), nullable=True),
            sa.Column("status", sa.String(50), nullable=False, server_default="PENDING"),
            sa.Column("workflow_run_id", UUID(as_uuid=True), sa.ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "status IN ('PENDING', 'PROCESSING', 'DISPATCHED', 'FAILED')",
                name="ck_workflow_trigger_event_status",
            ),
        )
        op.create_index("ix_workflow_trigger_events_event_id", "workflow_trigger_events", ["event_id"], unique=True)
        op.create_index("ix_workflow_trigger_events_workspace_id", "workflow_trigger_events", ["workspace_id"])
        op.create_index("ix_workflow_trigger_events_workflow_id", "workflow_trigger_events", ["workflow_id"])
        op.create_index("ix_workflow_trigger_events_version_id", "workflow_trigger_events", ["workflow_version_id"])
        op.create_index("ix_workflow_trigger_events_status", "workflow_trigger_events", ["status"])
        op.create_index("ix_workflow_trigger_events_idempotency_key", "workflow_trigger_events", ["idempotency_key"])

    # 2. Create workflow_schedule_states table
    if "workflow_schedule_states" not in tables:
        op.create_table(
            "workflow_schedule_states",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("workflow_id", UUID(as_uuid=True), sa.ForeignKey("workflow_definitions.id", ondelete="CASCADE"), unique=True, nullable=False),
            sa.Column("last_scheduled_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_window_key", sa.String(255), nullable=True),
            sa.Column("misfire_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("status", sa.String(50), nullable=False, server_default="IDLE"),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_workflow_schedule_states_workflow_id", "workflow_schedule_states", ["workflow_id"], unique=True)
        op.create_index("ix_workflow_schedule_states_next_run_at", "workflow_schedule_states", ["next_run_at"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "workflow_schedule_states" in tables:
        op.drop_table("workflow_schedule_states")
    if "workflow_trigger_events" in tables:
        op.drop_table("workflow_trigger_events")

