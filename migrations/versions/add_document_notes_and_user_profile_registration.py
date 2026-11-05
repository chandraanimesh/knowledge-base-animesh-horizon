"""add_document_notes_and_user_profile_registration

Revision ID: add_document_notes_and_user_profile_registration
Revises: add_workflow_run_provider_snapshot
Create Date: 2026-11-04 10:45:00.000000

Adds document_notes table for verified errata layer,
and adds registration_status and profile_json columns to users table.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "add_document_notes_and_user_profile_registration"
down_revision: Union[str, Sequence[str], None] = "add_workflow_run_provider_snapshot"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # 1. Create document_notes table
    if "document_notes" not in tables:
        op.create_table(
            "document_notes",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("author_user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True),
            sa.Column("correction_text", sa.Text(), nullable=False),
            sa.Column("evidence_reference", sa.Text(), nullable=True),
            sa.Column("approval_status", sa.String(20), sa.CheckConstraint("approval_status IN ('PENDING', 'APPROVED', 'REJECTED')", name="ck_document_notes_approval_status"), server_default="APPROVED", nullable=False),
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False, index=True),
            sa.Column("effective_from", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("effective_until", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index(
            "ix_document_notes_doc_active",
            "document_notes",
            ["document_id", "is_active", "approval_status"],
        )

    # 2. Add columns to users table
    user_cols = [c["name"] for c in inspector.get_columns("users")]
    if "registration_status" not in user_cols:
        op.add_column(
            "users",
            sa.Column(
                "registration_status",
                sa.String(20),
                sa.CheckConstraint("registration_status IN ('PENDING', 'APPROVED', 'REJECTED')", name="ck_user_registration_status"),
                server_default="APPROVED",
                nullable=False,
            ),
        )
    if "profile_json" not in user_cols:
        op.add_column(
            "users",
            sa.Column(
                "profile_json",
                JSONB,
                server_default="{}",
                nullable=True,
            ),
        )

    # 3. Ensure chat_messages has visualization and execution_metrics
    if "chat_messages" in tables:
        chat_cols = [c["name"] for c in inspector.get_columns("chat_messages")]
        if "visualization" not in chat_cols:
            op.add_column("chat_messages", sa.Column("visualization", JSONB, nullable=True))
        if "execution_metrics" not in chat_cols:
            op.add_column("chat_messages", sa.Column("execution_metrics", JSONB, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "document_notes" in tables:
        op.drop_index("ix_document_notes_doc_active", table_name="document_notes")
        op.drop_table("document_notes")

    user_cols = [c["name"] for c in inspector.get_columns("users")]
    if "profile_json" in user_cols:
        op.drop_column("users", "profile_json")
    if "registration_status" in user_cols:
        op.drop_column("users", "registration_status")

    if "chat_messages" in tables:
        chat_cols = [c["name"] for c in inspector.get_columns("chat_messages")]
        if "execution_metrics" in chat_cols:
            op.drop_column("chat_messages", "execution_metrics")
        if "visualization" in chat_cols:
            op.drop_column("chat_messages", "visualization")

