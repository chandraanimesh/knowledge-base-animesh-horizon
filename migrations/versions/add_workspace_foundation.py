"""add_workspace_foundation

Revision ID: add_workspace_foundation
Revises: add_document_perms
Create Date: 2026-10-01 22:20:00.000000

Adds first-class Workspace entity and nullable workspace_id FK to documents.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'add_workspace_foundation'
down_revision: Union[str, Sequence[str], None] = 'add_document_perms'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("department_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("departments.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("department_id", "code", name="uq_workspaces_department_code"),
    )
    op.create_index(
        "ix_workspaces_department_id",
        "workspaces",
        ["department_id"],
    )
    op.create_index(
        "ix_workspaces_dept_active",
        "workspaces",
        ["department_id", "is_active"],
    )

    op.add_column(
        "documents",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True),
    )
    op.create_index(
        "ix_documents_workspace_id",
        "documents",
        ["workspace_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_documents_workspace_id", table_name="documents")
    op.drop_column("documents", "workspace_id")
    op.drop_table("workspaces")
