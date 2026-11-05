"""add_living_articles_and_staged_updates

Revision ID: add_living_articles_and_staged_updates
Revises: add_document_notes_and_user_profile_registration
Create Date: 2026-11-04 16:00:00.000000

Milestone 7B: Living Articles & Staged Knowledge Updates.
Adds:
- document_versions
- document_page_versions
- document_version_chunks
- knowledge_dependencies
- shadow_ingestion_reports
- documents.current_version_id
- document_chunks.is_active, document_chunks.document_version_id
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "add_living_articles_and_staged_updates"
down_revision: Union[str, Sequence[str], None] = "add_document_notes_and_user_profile_registration"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # 1. Create document_versions table
    if "document_versions" not in tables:
        op.create_table(
            "document_versions",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("version_number", sa.Integer(), nullable=False, default=1),
            sa.Column("parent_version_id", UUID(as_uuid=True), sa.ForeignKey("document_versions.id", ondelete="SET NULL"), nullable=True, index=True),
            sa.Column("source_content_hash", sa.String(64), nullable=False, index=True),
            sa.Column("extraction_config_version", sa.String(50), server_default="v1", nullable=False),
            sa.Column("chunking_config_version", sa.String(50), server_default="v1", nullable=False),
            sa.Column("embedding_model", sa.String(100), server_default="sentence-transformers/all-MiniLM-L6-v2", nullable=False),
            sa.Column("embedding_dim", sa.Integer(), server_default="384", nullable=False),
            sa.Column(
                "status",
                sa.String(30),
                sa.CheckConstraint(
                    "status IN ('DETECTED', 'PROCESSING', 'SHADOW_VALIDATION', 'PENDING_APPROVAL', 'PUBLISHED', 'FAILED', 'SUPERSEDED', 'ROLLED_BACK')",
                    name="ck_document_version_status",
                ),
                server_default="DETECTED",
                nullable=False,
                index=True,
            ),
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False, index=True),
            sa.Column("change_summary", JSONB, nullable=True),
            sa.Column("metadata", JSONB, nullable=True),
            sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint("document_id", "version_number", name="uq_doc_version_number"),
        )

    # 2. Add current_version_id to documents
    doc_cols = [c["name"] for c in inspector.get_columns("documents")]
    if "current_version_id" not in doc_cols:
        op.add_column(
            "documents",
            sa.Column(
                "current_version_id",
                UUID(as_uuid=True),
                sa.ForeignKey("document_versions.id", ondelete="SET NULL", use_alter=True),
                nullable=True,
                index=True,
            ),
        )

    # 3. Add is_active and document_version_id to document_chunks
    chunk_cols = [c["name"] for c in inspector.get_columns("document_chunks")]
    if "is_active" not in chunk_cols:
        op.add_column(
            "document_chunks",
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False, index=True),
        )
    if "document_version_id" not in chunk_cols:
        op.add_column(
            "document_chunks",
            sa.Column(
                "document_version_id",
                UUID(as_uuid=True),
                sa.ForeignKey("document_versions.id", ondelete="SET NULL"),
                nullable=True,
                index=True,
            ),
        )

    # 4. Create document_page_versions table
    if "document_page_versions" not in tables:
        op.create_table(
            "document_page_versions",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("document_version_id", UUID(as_uuid=True), sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("page_number", sa.Integer(), nullable=False),
            sa.Column("page_fingerprint", sa.String(64), nullable=False, index=True),
            sa.Column("raw_content_hash", sa.String(64), nullable=False),
            sa.Column("structural_hash", sa.String(64), nullable=True),
            sa.Column(
                "status",
                sa.String(20),
                sa.CheckConstraint(
                    "status IN ('UNCHANGED', 'MODIFIED', 'ADDED', 'DELETED', 'MOVED', 'SPLIT', 'MERGED', 'UNKNOWN')",
                    name="ck_doc_page_version_status",
                ),
                nullable=False,
                server_default="ADDED",
            ),
            sa.Column("reused_from_page_id", UUID(as_uuid=True), sa.ForeignKey("document_page_versions.id", ondelete="SET NULL"), nullable=True),
            sa.Column("mapping_confidence", sa.Float(), server_default="1.0", nullable=False),
            sa.Column("metadata", JSONB, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )

    # 5. Create document_version_chunks table
    if "document_version_chunks" not in tables:
        op.create_table(
            "document_version_chunks",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("document_version_id", UUID(as_uuid=True), sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("chunk_id", UUID(as_uuid=True), sa.ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("ordinal", sa.Integer(), server_default="0", nullable=False),
            sa.Column("is_reused", sa.Boolean(), server_default=sa.text("false"), nullable=False),
            sa.Column("reused_from_version_id", UUID(as_uuid=True), sa.ForeignKey("document_versions.id", ondelete="SET NULL"), nullable=True),
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("document_version_id", "chunk_id", name="uq_doc_version_chunk"),
        )

    # 6. Create knowledge_dependencies table
    if "knowledge_dependencies" not in tables:
        op.create_table(
            "knowledge_dependencies",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("document_version_id", UUID(as_uuid=True), sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("page_version_id", UUID(as_uuid=True), sa.ForeignKey("document_page_versions.id", ondelete="SET NULL"), nullable=True, index=True),
            sa.Column("chunk_id", UUID(as_uuid=True), sa.ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=True, index=True),
            sa.Column("knowledge_item_id", UUID(as_uuid=True), sa.ForeignKey("knowledge_items.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("relationship_id", UUID(as_uuid=True), sa.ForeignKey("knowledge_relationships.id", ondelete="SET NULL"), nullable=True, index=True),
            sa.Column(
                "status",
                sa.String(30),
                sa.CheckConstraint(
                    "status IN ('ACTIVE', 'STALE', 'PENDING_REVALIDATION', 'SUPERSEDED', 'REVOKED', 'ARCHIVED')",
                    name="ck_knowledge_dependency_status",
                ),
                server_default="ACTIVE",
                nullable=False,
                index=True,
            ),
            sa.Column("invalidation_reason", sa.Text(), nullable=True),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )

    # 7. Create shadow_ingestion_reports table
    if "shadow_ingestion_reports" not in tables:
        op.create_table(
            "shadow_ingestion_reports",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("document_version_id", UUID(as_uuid=True), sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column(
                "comparison_status",
                sa.String(30),
                sa.CheckConstraint(
                    "comparison_status IN ('EXACT_MATCH', 'ACCEPTABLE_DIFFERENCE', 'MATERIAL_DIFFERENCE', 'FAILED')",
                    name="ck_shadow_comparison_status",
                ),
                nullable=False,
                index=True,
            ),
            sa.Column("page_diff_summary", JSONB, nullable=True),
            sa.Column("chunk_diff_summary", JSONB, nullable=True),
            sa.Column("embedding_diff_summary", JSONB, nullable=True),
            sa.Column("knowledge_diff_summary", JSONB, nullable=True),
            sa.Column("retrieval_diff_summary", JSONB, nullable=True),
            sa.Column("can_publish", sa.Boolean(), server_default=sa.text("false"), nullable=False),
            sa.Column("reasons", JSONB, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "shadow_ingestion_reports" in tables:
        op.drop_table("shadow_ingestion_reports")
    if "knowledge_dependencies" in tables:
        op.drop_table("knowledge_dependencies")
    if "document_version_chunks" in tables:
        op.drop_table("document_version_chunks")
    if "document_page_versions" in tables:
        op.drop_table("document_page_versions")

    chunk_cols = [c["name"] for c in inspector.get_columns("document_chunks")]
    if "document_version_id" in chunk_cols:
        op.drop_column("document_chunks", "document_version_id")
    if "is_active" in chunk_cols:
        op.drop_column("document_chunks", "is_active")

    doc_cols = [c["name"] for c in inspector.get_columns("documents")]
    if "current_version_id" in doc_cols:
        op.drop_column("documents", "current_version_id")

    if "document_versions" in tables:
        op.drop_table("document_versions")

