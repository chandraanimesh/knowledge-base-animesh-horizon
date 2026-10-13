"""add_reliable_memory_processing_jobs

Revision ID: add_reliable_memory_processing_jobs
Revises: add_website_sources_to_evidence
Create Date: 2026-10-12 14:00:00.000000

Adds durable memory processing jobs table and memory_items lifecycle/job provenance fields:
- memory_processing_jobs table (durable capture outbox and evaluation jobs)
- memory_items.source_job_id (FK to memory_processing_jobs)
- memory_items.retention_expires_at (lifecycle storage retention expiry)
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision: str = 'add_reliable_memory_processing_jobs'
down_revision: Union[str, Sequence[str], None] = 'add_website_sources_to_evidence'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    existing_tables = inspector.get_table_names(schema=target_schema)
    if 'memory_processing_jobs' not in existing_tables:
        op.create_table(
            'memory_processing_jobs',
            sa.Column('id', UUID(as_uuid=True), primary_key=True),
            sa.Column('source_message_id', UUID(as_uuid=True), nullable=True),
            sa.Column('conversation_id', UUID(as_uuid=True), nullable=True),
            sa.Column('owner_user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='SET NULL'), nullable=True),
            sa.Column('source_text_snapshot', sa.Text(), nullable=False),
            sa.Column('trigger_kind', sa.String(50), nullable=False, server_default='rule_candidate'),
            sa.Column(
                'status',
                sa.String(50),
                sa.CheckConstraint(
                    "status IN ('PENDING', 'PROCESSING', 'SUCCEEDED', 'RETRY_PENDING', 'DEAD_LETTER', 'SKIPPED_NOT_MEMORABLE')",
                    name='ck_memory_processing_job_status',
                ),
                nullable=False,
                server_default='PENDING',
            ),
            sa.Column('attempt_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('max_attempts', sa.Integer(), nullable=False, server_default='3'),
            sa.Column('next_attempt_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('last_error_code', sa.String(100), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('locked_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('locked_by', sa.String(255), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
            schema=target_schema,
        )
        op.create_index('ix_memory_processing_jobs_owner_user_id', 'memory_processing_jobs', ['owner_user_id'], schema=target_schema)
        op.create_index('ix_memory_processing_jobs_workspace_id', 'memory_processing_jobs', ['workspace_id'], schema=target_schema)
        op.create_index('ix_memory_processing_jobs_status_created', 'memory_processing_jobs', ['status', 'created_at'], schema=target_schema)
        op.create_index('ix_memory_processing_jobs_user_status', 'memory_processing_jobs', ['owner_user_id', 'status'], schema=target_schema)
        op.create_index('ix_memory_processing_jobs_next_attempt', 'memory_processing_jobs', ['status', 'next_attempt_at'], schema=target_schema)

    # Columns on memory_items:
    existing_cols = [c['name'] for c in inspector.get_columns('memory_items', schema=target_schema)]
    if 'source_job_id' not in existing_cols:
        op.add_column(
            'memory_items',
            sa.Column('source_job_id', UUID(as_uuid=True), sa.ForeignKey('memory_processing_jobs.id', ondelete='SET NULL'), nullable=True),
            schema=target_schema,
        )
        op.create_index('ix_memory_items_source_job_id', 'memory_items', ['source_job_id'], schema=target_schema)
    if 'retention_expires_at' not in existing_cols:
        op.add_column(
            'memory_items',
            sa.Column('retention_expires_at', sa.DateTime(timezone=True), nullable=True),
            schema=target_schema,
        )
        op.create_index('ix_memory_items_retention_expires_at', 'memory_items', ['retention_expires_at'], schema=target_schema)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    existing_cols = [c['name'] for c in inspector.get_columns('memory_items', schema=target_schema)]
    if 'retention_expires_at' in existing_cols:
        op.drop_index('ix_memory_items_retention_expires_at', table_name='memory_items', schema=target_schema)
        op.drop_column('memory_items', 'retention_expires_at', schema=target_schema)
    if 'source_job_id' in existing_cols:
        op.drop_index('ix_memory_items_source_job_id', table_name='memory_items', schema=target_schema)
        op.drop_column('memory_items', 'source_job_id', schema=target_schema)

    existing_tables = inspector.get_table_names(schema=target_schema)
    if 'memory_processing_jobs' in existing_tables:
        op.drop_table('memory_processing_jobs', schema=target_schema)
