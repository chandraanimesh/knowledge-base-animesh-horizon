"""add_workspace_to_conversations

Revision ID: add_workspace_to_conversations
Revises: add_workspace_members
Create Date: 2026-10-01 23:20:00.000000

Associates conversations with workspaces (workspace_id nullable FK).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'add_workspace_to_conversations'
down_revision: Union[str, Sequence[str], None] = 'add_workspace_members'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_conversations_workspace_updated",
        "conversations",
        ["workspace_id", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_conversations_workspace_updated", table_name="conversations")
    op.drop_column("conversations", "workspace_id")
