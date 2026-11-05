"""add_chunk_versions_and_curation

Revision ID: add_chunk_versions_and_curation
Revises: add_mcp_tool_operation_records
Create Date: 2026-10-28 11:45:00.000000

Adds document_chunk_versions table and version columns to document_chunks:
- current_version and updated_at on document_chunks
- document_chunk_versions for immutable audit history, diff provenance, and rollback snapshots
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from rag_app.core.config import get_settings

settings = get_settings()

revision: str = "add_chunk_versions_and_curation"
down_revision: Union[str, Sequence[str], None] = "add_mcp_tool_operation_records"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add current_version and updated_at to document_chunks
    op.add_column(
        "document_chunks",
        sa.Column(
            "current_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
    )
    op.add_column(
        "document_chunks",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # 2. Create document_chunk_versions table
    op.create_table(
        "document_chunk_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "chunk_id",
            UUID(as_uuid=True),
            sa.ForeignKey("document_chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("text_before", sa.Text(), nullable=False),
        sa.Column("text_after", sa.Text(), nullable=False),
        sa.Column("diff_patch", sa.Text(), nullable=True),
        sa.Column(
            "embedding_snapshot",
            Vector(settings.embedding_dim),
            nullable=False,
        ),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("chunk_hash", sa.String(64), nullable=False),
        sa.Column(
            "edited_by",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("edit_reason", sa.String(500), nullable=True),
        sa.Column(
            "is_rollback",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("reverted_from_version", sa.Integer(), nullable=True),
        sa.UniqueConstraint("chunk_id", "version_number", name="uq_chunk_version_number"),
    )
    op.create_index(
        "ix_document_chunk_versions_chunk_id",
        "document_chunk_versions",
        ["chunk_id"],
    )
    op.create_index(
        "ix_chunk_versions_created_at",
        "document_chunk_versions",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_versions_created_at", table_name="document_chunk_versions")
    op.drop_index("ix_document_chunk_versions_chunk_id", table_name="document_chunk_versions")
    op.drop_table("document_chunk_versions")
    op.drop_column("document_chunks", "updated_at")
    op.drop_column("document_chunks", "current_version")
