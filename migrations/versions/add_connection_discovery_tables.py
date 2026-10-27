"""add_connection_discovery_tables

Revision ID: add_connection_discovery_tables
Revises: add_reliable_memory_processing_jobs
Create Date: 2026-10-13 18:00:00.000000

Adds tables for Knowledge Connection Discovery:
- connection_candidates (candidate suggestions with typed relational foreign keys)
- connection_candidate_reviews (version-bound human curator audit log)
- connection_candidate_evidence (verifiable chunk/message evidence links)
- connection_discovery_jobs (durable background job queue)
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision: str = 'add_connection_discovery_tables'
down_revision: Union[str, Sequence[str], None] = 'add_reliable_memory_processing_jobs'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    existing_tables = inspector.get_table_names(schema=target_schema)

    # 1. connection_candidates
    if 'connection_candidates' not in existing_tables:
        op.create_table(
            'connection_candidates',
            sa.Column('id', UUID(as_uuid=True), primary_key=True),
            sa.Column('source_knowledge_item_id', UUID(as_uuid=True), sa.ForeignKey('knowledge_items.id', ondelete='CASCADE'), nullable=True),
            sa.Column('source_memory_item_id', UUID(as_uuid=True), sa.ForeignKey('memory_items.id', ondelete='CASCADE'), nullable=True),
            sa.Column('target_knowledge_item_id', UUID(as_uuid=True), sa.ForeignKey('knowledge_items.id', ondelete='CASCADE'), nullable=True),
            sa.Column('target_memory_item_id', UUID(as_uuid=True), sa.ForeignKey('memory_items.id', ondelete='CASCADE'), nullable=True),
            sa.Column('connection_type', sa.String(30), nullable=False),
            sa.Column('predicate', sa.String(100), nullable=False, server_default='associated_with'),
            sa.Column('lifecycle_state', sa.String(30), nullable=False, server_default='SUGGESTED'),
            sa.Column('is_directed', sa.Boolean(), nullable=False, server_default=sa.text('true')),
            sa.Column('ranking_score', sa.Float(), nullable=False, server_default='0.0'),
            sa.Column('source_version_hash', sa.String(64), nullable=False),
            sa.Column('target_version_hash', sa.String(64), nullable=False),
            sa.Column('valid_from', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True),
            sa.Column('transaction_created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('transaction_superseded_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('superseded_by_id', UUID(as_uuid=True), sa.ForeignKey('connection_candidates.id', ondelete='SET NULL'), nullable=True),
            sa.Column('department_id', UUID(as_uuid=True), sa.ForeignKey('departments.id', ondelete='RESTRICT'), nullable=True),
            sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=True),
            sa.Column('author_user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.CheckConstraint(
                "(source_knowledge_item_id IS NOT NULL AND source_memory_item_id IS NULL) OR "
                "(source_knowledge_item_id IS NULL AND source_memory_item_id IS NOT NULL)",
                name='chk_candidate_source',
            ),
            sa.CheckConstraint(
                "(target_knowledge_item_id IS NOT NULL AND target_memory_item_id IS NULL) OR "
                "(target_knowledge_item_id IS NULL AND target_memory_item_id IS NOT NULL)",
                name='chk_candidate_target',
            ),
            schema=target_schema,
        )
        op.create_index('ix_connection_candidates_source_k', 'connection_candidates', ['source_knowledge_item_id'], schema=target_schema)
        op.create_index('ix_connection_candidates_source_m', 'connection_candidates', ['source_memory_item_id'], schema=target_schema)
        op.create_index('ix_connection_candidates_target_k', 'connection_candidates', ['target_knowledge_item_id'], schema=target_schema)
        op.create_index('ix_connection_candidates_target_m', 'connection_candidates', ['target_memory_item_id'], schema=target_schema)
        op.create_index('ix_connection_candidates_state', 'connection_candidates', ['lifecycle_state'], schema=target_schema)

    # 2. connection_candidate_reviews
    if 'connection_candidate_reviews' not in existing_tables:
        op.create_table(
            'connection_candidate_reviews',
            sa.Column('id', UUID(as_uuid=True), primary_key=True),
            sa.Column('candidate_id', UUID(as_uuid=True), sa.ForeignKey('connection_candidates.id', ondelete='CASCADE'), nullable=False),
            sa.Column('reviewer_user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='RESTRICT'), nullable=False),
            sa.Column('decision', sa.String(30), nullable=False),
            sa.Column('rationale', sa.Text(), nullable=False),
            sa.Column('source_version_hash', sa.String(64), nullable=False),
            sa.Column('target_version_hash', sa.String(64), nullable=False),
            sa.Column('policy_version', sa.String(20), nullable=False, server_default='v1.0'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            schema=target_schema,
        )
        op.create_index('ix_candidate_reviews_cand_id', 'connection_candidate_reviews', ['candidate_id'], schema=target_schema)

    # 3. connection_candidate_evidence
    if 'connection_candidate_evidence' not in existing_tables:
        op.create_table(
            'connection_candidate_evidence',
            sa.Column('id', UUID(as_uuid=True), primary_key=True),
            sa.Column('candidate_id', UUID(as_uuid=True), sa.ForeignKey('connection_candidates.id', ondelete='CASCADE'), nullable=False),
            sa.Column('document_chunk_id', UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='CASCADE'), nullable=True),
            sa.Column('chat_message_id', UUID(as_uuid=True), sa.ForeignKey('chat_messages.id', ondelete='CASCADE'), nullable=True),
            sa.Column('research_evidence_id', UUID(as_uuid=True), sa.ForeignKey('research_evidence.id', ondelete='CASCADE'), nullable=True),
            sa.Column('claim_excerpt', sa.Text(), nullable=False),
            sa.Column('content_hash', sa.String(64), nullable=False),
            sa.Column('verification_status', sa.String(30), nullable=False, server_default='VERIFIED'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            schema=target_schema,
        )
        op.create_index('ix_candidate_evidence_cand_id', 'connection_candidate_evidence', ['candidate_id'], schema=target_schema)

    # 4. connection_discovery_jobs
    if 'connection_discovery_jobs' not in existing_tables:
        op.create_table(
            'connection_discovery_jobs',
            sa.Column('id', UUID(as_uuid=True), primary_key=True),
            sa.Column('source_item_id', UUID(as_uuid=True), nullable=False),
            sa.Column('source_item_type', sa.String(20), nullable=False),
            sa.Column('target_source_hash', sa.String(64), nullable=False),
            sa.Column('requesting_user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('status', sa.String(50), nullable=False, server_default='PENDING'),
            sa.Column('attempt_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('max_attempts', sa.Integer(), nullable=False, server_default='3'),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('locked_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('locked_by', sa.String(255), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
            schema=target_schema,
        )
        op.create_index('ix_connection_jobs_status_created', 'connection_discovery_jobs', ['status', 'created_at'], schema=target_schema)
        op.create_index('ix_connection_jobs_locked_at', 'connection_discovery_jobs', ['locked_at'], schema=target_schema)


def downgrade() -> None:
    bind = op.get_bind()
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    op.drop_table('connection_discovery_jobs', schema=target_schema)
    op.drop_table('connection_candidate_evidence', schema=target_schema)
    op.drop_table('connection_candidate_reviews', schema=target_schema)
    op.drop_table('connection_candidates', schema=target_schema)
