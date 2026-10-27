"""add_mcp_tool_operation_records

Revision ID: add_mcp_tool_operation_records
Revises: add_workspace_mcp_connections
Create Date: 2026-10-26 13:00:00.000000

Adds mcp_tool_operation_records table for Milestone 6C.2:
- Durable operation records for external MCP mutating tool calls
- Idempotency key tracking and replay protection
- Two-phase execution states (PENDING -> IN_FLIGHT -> CONFIRMED / UNKNOWN / RECONCILIATION_REQUIRED)
- Prevents ambiguous timeouts from being treated as failures or blindly retried
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "add_mcp_tool_operation_records"
down_revision: Union[str, Sequence[str], None] = "add_workspace_mcp_connections"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "mcp_tool_operation_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workspace_id",
            UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "connection_id",
            UUID(as_uuid=True),
            sa.ForeignKey("workspace_mcp_connections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(100), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="PENDING"),
        sa.Column("arguments", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("result_payload", JSONB, nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reconciliation_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_reconciled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approval_ticket_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("workflow_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "task_id",
            UUID(as_uuid=True),
            sa.ForeignKey("workflow_tasks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("workspace_id", "idempotency_key", name="uq_mcp_tool_op_ws_idempotency_key"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'IN_FLIGHT', 'CONFIRMED', 'FAILED', 'UNKNOWN', 'RECONCILIATION_REQUIRED')",
            name="ck_mcp_tool_operation_records_status",
        ),
    )
    op.create_index(
        "ix_mcp_tool_operation_records_workspace_id",
        "mcp_tool_operation_records",
        ["workspace_id"],
    )
    op.create_index(
        "ix_mcp_tool_operation_records_connection_id",
        "mcp_tool_operation_records",
        ["connection_id"],
    )
    op.create_index(
        "ix_mcp_tool_operation_records_ws_status",
        "mcp_tool_operation_records",
        ["workspace_id", "status"],
    )
    op.create_index(
        "ix_mcp_tool_operation_records_conn_tool",
        "mcp_tool_operation_records",
        ["connection_id", "tool_name"],
    )


def downgrade() -> None:
    op.drop_index("ix_mcp_tool_operation_records_conn_tool", table_name="mcp_tool_operation_records")
    op.drop_index("ix_mcp_tool_operation_records_ws_status", table_name="mcp_tool_operation_records")
    op.drop_index("ix_mcp_tool_operation_records_connection_id", table_name="mcp_tool_operation_records")
    op.drop_index("ix_mcp_tool_operation_records_workspace_id", table_name="mcp_tool_operation_records")
    op.drop_table("mcp_tool_operation_records")
