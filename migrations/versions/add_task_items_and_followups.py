"""add_task_items_and_followups

Revision ID: add_task_items_and_followups
Revises: allow_private_visibility_drafts
Create Date: 2026-10-11 15:30:00.000000

Stage H: Structured Events, Tasks and Follow-ups
Adds task_items table with explicit owner, due date, timezone policy, status,
company/contact associations, and provenance tracking.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


revision: str = 'add_task_items_and_followups'
down_revision: Union[str, Sequence[str], None] = 'allow_private_visibility_drafts'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    existing_tables = inspector.get_table_names(schema=target_schema)
    if 'task_items' not in existing_tables:
        op.create_table(
            'task_items',
            sa.Column('id', UUID(as_uuid=True), primary_key=True),
            sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='SET NULL'), nullable=True),
            sa.Column('department_id', UUID(as_uuid=True), sa.ForeignKey('departments.id', ondelete='SET NULL'), nullable=True),
            sa.Column('title', sa.String(255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('status', sa.String(20), sa.CheckConstraint("status IN ('OPEN', 'COMPLETED', 'CANCELLED')", name='ck_task_status'), nullable=False, server_default='OPEN'),
            sa.Column('due_date', sa.DateTime(timezone=True), nullable=True),
            sa.Column('timezone', sa.String(64), nullable=False, server_default='UTC'),
            sa.Column('company_name', sa.String(255), nullable=True),
            sa.Column('contact_name', sa.String(255), nullable=True),
            sa.Column('provenance_type', sa.String(32), sa.CheckConstraint("provenance_type IN ('EXPLICIT_USER_ACTION', 'EXTRACTED_FROM_INTERACTION', 'EXTRACTED_FROM_DOCUMENT')", name='ck_task_provenance_type'), nullable=False, server_default='EXPLICIT_USER_ACTION'),
            sa.Column('conversation_id', UUID(as_uuid=True), sa.ForeignKey('conversations.id', ondelete='SET NULL'), nullable=True),
            sa.Column('message_id', UUID(as_uuid=True), sa.ForeignKey('chat_messages.id', ondelete='SET NULL'), nullable=True),
            sa.Column('document_chunk_id', UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='SET NULL'), nullable=True),
            sa.Column('attributes', JSONB, nullable=True, server_default='{}'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
            schema=target_schema,
        )

        op.create_index('ix_task_items_user_id', 'task_items', ['user_id'], schema=target_schema)
        op.create_index('ix_task_items_workspace_id', 'task_items', ['workspace_id'], schema=target_schema)
        op.create_index('ix_task_items_department_id', 'task_items', ['department_id'], schema=target_schema)
        op.create_index('ix_task_items_status', 'task_items', ['status'], schema=target_schema)
        op.create_index('ix_task_items_due_date', 'task_items', ['due_date'], schema=target_schema)
        op.create_index('ix_task_items_company_name', 'task_items', ['company_name'], schema=target_schema)
        op.create_index('ix_task_items_user_status_due', 'task_items', ['user_id', 'status', 'due_date'], schema=target_schema)
        op.create_index('ix_task_items_user_company', 'task_items', ['user_id', 'company_name'], schema=target_schema)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    existing_tables = inspector.get_table_names(schema=target_schema)
    if 'task_items' in existing_tables:
        op.drop_table('task_items', schema=target_schema)
