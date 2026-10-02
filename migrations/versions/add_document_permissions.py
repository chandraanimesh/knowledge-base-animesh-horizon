"""add_document_permissions

Revision ID: add_document_perms
Revises: add_conversations_chat
Create Date: 2026-10-01 21:40:00.000000

Small additive authorization layer for document-level permissions:
- Creates `document_permissions` table: (document_id, user_id, permission_id)
- Primary key enforces uniqueness on (document_id, user_id, permission_id)
- Reuses existing permissions and document/user tables
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'add_document_perms'
down_revision: Union[str, Sequence[str], None] = 'add_conversations_chat'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_permissions",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("permission_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("permissions.id", ondelete="RESTRICT"), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_document_permissions_doc_id",
        "document_permissions",
        ["document_id"],
    )
    op.create_index(
        "ix_document_permissions_user_id",
        "document_permissions",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_table("document_permissions")
