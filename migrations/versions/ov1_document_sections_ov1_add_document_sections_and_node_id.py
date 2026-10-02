"""ov1_add_document_sections_and_node_id

Revision ID: ov1_document_sections
Revises: a1b2c3d4e5f6
Create Date: 2026-10-01 16:45:50.683462

OV-1: Native Hierarchical Resource Model
- Creates the `document_sections` table (self-referential section tree).
- Adds nullable `node_id` FK column on `document_chunks`.
- Existing documents/chunks are unaffected; node_id starts as NULL everywhere.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'ov1_document_sections'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create document_sections table
    op.create_table(
        "document_sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_sections.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "node_type",
            sa.String(20),
            nullable=False,
            server_default="SECTION",
        ),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ordinal_path", sa.String(255), nullable=False, server_default=""),
        sa.Column("l0_abstract", sa.Text(), nullable=True),
        sa.Column("l1_overview", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "node_type IN ('DOCUMENT', 'SECTION', 'SUBSECTION', 'FIGURE')",
            name="ck_documentsection_node_type",
        ),
    )
    op.create_index("ix_document_sections_document_id", "document_sections", ["document_id"])
    op.create_index("ix_document_sections_parent_id", "document_sections", ["parent_id"])
    op.create_index(
        "ix_document_sections_doc_ordinal",
        "document_sections",
        ["document_id", "ordinal_path"],
    )

    # 2. Add nullable node_id FK to document_chunks
    op.add_column(
        "document_chunks",
        sa.Column(
            "node_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_document_chunks_node_id",
        "document_chunks",
        "document_sections",
        ["node_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_document_chunks_node_id", "document_chunks", ["node_id"])


def downgrade() -> None:
    op.drop_index("ix_document_chunks_node_id", table_name="document_chunks")
    op.drop_constraint("fk_document_chunks_node_id", "document_chunks", type_="foreignkey")
    op.drop_column("document_chunks", "node_id")

    op.drop_index("ix_document_sections_doc_ordinal", table_name="document_sections")
    op.drop_index("ix_document_sections_parent_id", table_name="document_sections")
    op.drop_index("ix_document_sections_document_id", table_name="document_sections")
    op.drop_table("document_sections")
