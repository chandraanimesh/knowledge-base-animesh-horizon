"""add_workspace_mcp_connections

Revision ID: add_workspace_mcp_connections
Revises: add_workflow_run_and_tasks_persistence
Create Date: 2026-10-26 12:00:00.000000

Adds workspace_mcp_connections table for Milestone 6C:
- Workspace-scoped outbound MCP connection persistence
- Server-managed credential references (never raw API keys)
- Discovered tool metadata caching (untrusted)
- Default-deny tool allowlist
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "add_workspace_mcp_connections"
down_revision: Union[str, Sequence[str], None] = "add_workflow_run_and_tasks_persistence"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workspace_mcp_connections",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workspace_id",
            UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("endpoint_url", sa.Text(), nullable=False),
        sa.Column("transport", sa.String(50), nullable=False, server_default="streamable-http"),
        sa.Column("credential_ref", sa.String(255), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("status", sa.String(50), nullable=False, server_default="DISCONNECTED"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("last_connected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("server_info", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("discovered_tools", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("enabled_tools", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("workspace_id", "name", name="uq_workspace_mcp_connections_ws_name"),
        sa.CheckConstraint(
            "status IN ('DISCONNECTED', 'CONNECTED', 'ERROR', 'INITIALIZING')",
            name="ck_workspace_mcp_connections_status",
        ),
        sa.CheckConstraint(
            "transport IN ('streamable-http', 'sse')",
            name="ck_workspace_mcp_connections_transport",
        ),
    )
    op.create_index(
        "ix_workspace_mcp_connections_workspace_id",
        "workspace_mcp_connections",
        ["workspace_id"],
    )
    op.create_index(
        "ix_workspace_mcp_connections_ws_enabled",
        "workspace_mcp_connections",
        ["workspace_id", "is_enabled"],
    )


def downgrade() -> None:
    op.drop_index("ix_workspace_mcp_connections_ws_enabled", table_name="workspace_mcp_connections")
    op.drop_index("ix_workspace_mcp_connections_workspace_id", table_name="workspace_mcp_connections")
    op.drop_table("workspace_mcp_connections")
