"""add_workflow_run_and_tasks_persistence

Revision ID: add_workflow_run_and_tasks_persistence
Revises: add_memory_processing_job_source_trigger_unique_index
Create Date: 2026-10-23 18:30:00.000000

Adds tables and constraints for Milestone 1 Workflow Persistence:
- workflow_runs: Multi-tenant, scoped-idempotency, budget-reserved orchestrator lifecycle state.
- workflow_tasks: Normalized task DAG execution, atomic claiming, lease recovery, and fencing tokens.
- generated_artifacts: Backward-compatible optional workflow_run_id FK.
- crm_writeback_approvals: Backward-compatible optional workflow_run_id FK, approved_draft_hash, intended_action.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "add_workflow_run_and_tasks_persistence"
down_revision: Union[str, Sequence[str], None] = "add_memory_processing_job_source_trigger_unique_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # 1. Create workflow_runs table
    if "workflow_runs" not in tables:
        op.create_table(
            "workflow_runs",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=True),
            sa.Column("department_id", UUID(as_uuid=True), sa.ForeignKey("departments.id", ondelete="RESTRICT"), nullable=True),
            sa.Column("conversation_id", UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True),
            sa.Column("scope_type", sa.String(20), nullable=False, server_default="WORKSPACE"),
            sa.Column("thread_id", sa.String(255), nullable=True),
            sa.Column("status", sa.String(50), nullable=False, server_default="PENDING"),
            sa.Column("idempotency_key", sa.String(255), nullable=True),
            sa.Column("request_hash", sa.String(64), nullable=False),
            sa.Column("draft_hash", sa.String(64), nullable=True),
            sa.Column("initial_query", sa.Text(), nullable=False),
            sa.Column("max_tokens_budget", sa.Integer(), nullable=False, server_default="25000"),
            sa.Column("tokens_spent", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("tokens_reserved", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("max_provider_calls_budget", sa.Integer(), nullable=False, server_default="15"),
            sa.Column("provider_calls_spent", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("provider_calls_reserved", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("wall_clock_timeout_seconds", sa.Integer(), nullable=False, server_default="180"),
            sa.Column("failure_category", sa.String(100), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("awaiting_approval_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("executing_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.CheckConstraint(
                "status IN ('PENDING', 'PLANNING', 'RUNNING', 'AWAITING_APPROVAL', 'EXECUTING', 'COMPLETED', 'REJECTED', 'EXPIRED', 'CANCELLED', 'FAILED')",
                name="ck_workflow_runs_status",
            ),
            sa.CheckConstraint(
                "scope_type IN ('GLOBAL', 'WORKSPACE')",
                name="ck_workflow_runs_scope_type",
            ),
            sa.CheckConstraint(
                "(scope_type = 'GLOBAL' AND workspace_id IS NULL) OR (scope_type = 'WORKSPACE' AND workspace_id IS NOT NULL)",
                name="ck_workflow_runs_scope_workspace_consistency",
            ),
            sa.CheckConstraint("tokens_spent >= 0", name="ck_workflow_runs_tokens_spent_nonneg"),
            sa.CheckConstraint("tokens_reserved >= 0", name="ck_workflow_runs_tokens_reserved_nonneg"),
            sa.CheckConstraint("provider_calls_spent >= 0", name="ck_workflow_runs_calls_spent_nonneg"),
            sa.CheckConstraint("provider_calls_reserved >= 0", name="ck_workflow_runs_calls_reserved_nonneg"),
        )
        op.create_index("ix_workflow_runs_user_id", "workflow_runs", ["user_id"])
        op.create_index("ix_workflow_runs_workspace_id", "workflow_runs", ["workspace_id"])
        op.create_index("ix_workflow_runs_department_id", "workflow_runs", ["department_id"])
        op.create_index("ix_workflow_runs_conversation_id", "workflow_runs", ["conversation_id"])
        op.create_index("ix_workflow_runs_thread_id", "workflow_runs", ["thread_id"])
        op.create_index("ix_workflow_runs_status_created", "workflow_runs", ["status", "created_at"])
        op.create_index(
            "uq_workflow_runs_user_ws_idempotency",
            "workflow_runs",
            ["user_id", "workspace_id", "idempotency_key"],
            unique=True,
            postgresql_where=sa.text("idempotency_key IS NOT NULL AND workspace_id IS NOT NULL"),
        )
        op.create_index(
            "uq_workflow_runs_user_global_idempotency",
            "workflow_runs",
            ["user_id", "idempotency_key"],
            unique=True,
            postgresql_where=sa.text("idempotency_key IS NOT NULL AND workspace_id IS NULL"),
        )

    # 2. Create workflow_tasks table
    if "workflow_tasks" not in tables:
        op.create_table(
            "workflow_tasks",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("workflow_run_id", UUID(as_uuid=True), sa.ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("task_key", sa.String(100), nullable=False),
            sa.Column("task_type", sa.String(50), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
            sa.Column("dependencies", JSONB, nullable=False, server_default="[]"),
            sa.Column("input_payload", JSONB, nullable=False, server_default="{}"),
            sa.Column("output_result", JSONB, nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("tokens_spent", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("provider_calls", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
            sa.Column("locked_by", sa.String(255), nullable=True),
            sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("fencing_token", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "status IN ('PENDING', 'READY', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'SKIPPED')",
                name="ck_workflow_task_status",
            ),
            sa.UniqueConstraint("workflow_run_id", "task_key", name="uq_workflow_tasks_run_task_key"),
        )
        op.create_index("ix_workflow_tasks_workflow_run_id", "workflow_tasks", ["workflow_run_id"])
        op.create_index("ix_workflow_tasks_status", "workflow_tasks", ["status"])
        op.create_index("ix_workflow_tasks_lease_expires_at", "workflow_tasks", ["lease_expires_at"])
        op.create_index("ix_workflow_tasks_claim", "workflow_tasks", ["status", "lease_expires_at"])

    # 3. Extend generated_artifacts
    artifact_cols = [c["name"] for c in inspector.get_columns("generated_artifacts")]
    if "workflow_run_id" not in artifact_cols:
        op.add_column(
            "generated_artifacts",
            sa.Column("workflow_run_id", UUID(as_uuid=True), sa.ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True),
        )
        op.create_index("ix_generated_artifacts_workflow_run_id", "generated_artifacts", ["workflow_run_id"])

    # 4. Extend crm_writeback_approvals
    crm_cols = [c["name"] for c in inspector.get_columns("crm_writeback_approvals")]
    if "workflow_run_id" not in crm_cols:
        op.add_column(
            "crm_writeback_approvals",
            sa.Column("workflow_run_id", UUID(as_uuid=True), sa.ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True),
        )
        op.create_index("ix_crm_writeback_approvals_workflow_run_id", "crm_writeback_approvals", ["workflow_run_id"])
    if "approved_draft_hash" not in crm_cols:
        op.add_column(
            "crm_writeback_approvals",
            sa.Column("approved_draft_hash", sa.String(64), nullable=True),
        )
    if "intended_action" not in crm_cols:
        op.add_column(
            "crm_writeback_approvals",
            sa.Column("intended_action", sa.String(50), nullable=False, server_default="CRM_CREATE_NOTE"),
        )


def downgrade() -> None:
    # 1. Drop added columns from crm_writeback_approvals
    op.drop_column("crm_writeback_approvals", "intended_action")
    op.drop_column("crm_writeback_approvals", "approved_draft_hash")
    op.drop_index("ix_crm_writeback_approvals_workflow_run_id", table_name="crm_writeback_approvals")
    op.drop_column("crm_writeback_approvals", "workflow_run_id")

    # 2. Drop added column from generated_artifacts
    op.drop_index("ix_generated_artifacts_workflow_run_id", table_name="generated_artifacts")
    op.drop_column("generated_artifacts", "workflow_run_id")

    # 3. Drop workflow_tasks table
    op.drop_table("workflow_tasks")

    # 4. Drop workflow_runs table
    op.drop_table("workflow_runs")
