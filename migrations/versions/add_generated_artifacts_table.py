"""add_generated_artifacts_table

Revision ID: add_generated_artifacts_table
Revises: add_crm_approvals_and_webhook_events
Create Date: 2026-10-16 12:30:00.000000

Adds generated_artifacts table for multi-format RAG artifacts persistence:
- Secure metadata, server-controlled paths, SHA-256 checksums, expiration TTL, and download audit
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "add_generated_artifacts_table"
down_revision: Union[str, Sequence[str], None] = "add_crm_approvals_and_webhook_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "generated_artifacts" not in tables:
        op.create_table(
            "generated_artifacts",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("conversation_id", UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True),
            sa.Column("message_id", UUID(as_uuid=True), sa.ForeignKey("chat_messages.id", ondelete="SET NULL"), nullable=True, index=True),
            sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True),
            sa.Column("organization_id", UUID(as_uuid=True), nullable=True, index=True),
            sa.Column("title", sa.String(255), nullable=False),
            sa.Column("format", sa.String(20), nullable=False),
            sa.Column("safe_filename", sa.String(255), nullable=False),
            sa.Column("mime_type", sa.String(100), nullable=False),
            sa.Column("byte_size", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("storage_path", sa.Text(), nullable=False),
            sa.Column("checksum_sha256", sa.String(64), nullable=False),
            sa.Column("source_departments", JSONB, nullable=True),
            sa.Column("authorized_roles", JSONB, nullable=True),
            sa.Column("tabular_data", JSONB, nullable=True),
            sa.Column("coverage_metadata", JSONB, nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False, index=True),
            sa.Column("download_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_downloaded_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index(
            "ix_artifacts_user_created",
            "generated_artifacts",
            ["user_id", "created_at"],
        )
        op.create_index(
            "ix_artifacts_expires",
            "generated_artifacts",
            ["expires_at"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "generated_artifacts" in tables:
        op.drop_index("ix_artifacts_expires", table_name="generated_artifacts")
        op.drop_index("ix_artifacts_user_created", table_name="generated_artifacts")
        op.drop_table("generated_artifacts")
