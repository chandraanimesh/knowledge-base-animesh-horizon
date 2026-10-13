"""add_website_sources_to_knowledge_evidence

Revision ID: add_website_sources_to_evidence
Revises: add_task_items_and_followups
Create Date: 2026-10-12 01:35:00.000000

Adds website source fields to knowledge_evidence:
- source_url (validated URL string)
- source_title (optional page title/label)
- user_excerpt (optional user-provided supporting text)
- verification_status (NOT_VERIFIED, VERIFIED, FAILED)
- verified_at (timestamp)
- content_hash (future verification hash)
Updates ck_knowledgeevidence_single_source to allow website sources.
Updates uq_knowledge_evidence_source unique index to prevent duplicate URL attachments.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'add_website_sources_to_evidence'
down_revision: Union[str, Sequence[str], None] = 'add_task_items_and_followups'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None
    schema_prefix = f"{current_schema}." if current_schema else ""

    ev_cols = [c['name'] for c in inspector.get_columns('knowledge_evidence', schema=target_schema)]

    if 'source_url' not in ev_cols:
        op.add_column(
            'knowledge_evidence',
            sa.Column('source_url', sa.String(2048), nullable=True),
            schema=target_schema,
        )
    if 'source_title' not in ev_cols:
        op.add_column(
            'knowledge_evidence',
            sa.Column('source_title', sa.String(512), nullable=True),
            schema=target_schema,
        )
    if 'user_excerpt' not in ev_cols:
        op.add_column(
            'knowledge_evidence',
            sa.Column('user_excerpt', sa.Text(), nullable=True),
            schema=target_schema,
        )
    if 'verification_status' not in ev_cols:
        op.add_column(
            'knowledge_evidence',
            sa.Column('verification_status', sa.String(30), nullable=True, server_default='NOT_VERIFIED'),
            schema=target_schema,
        )
    if 'verified_at' not in ev_cols:
        op.add_column(
            'knowledge_evidence',
            sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
            schema=target_schema,
        )
    if 'content_hash' not in ev_cols:
        op.add_column(
            'knowledge_evidence',
            sa.Column('content_hash', sa.String(128), nullable=True),
            schema=target_schema,
        )

    # Index on source_url
    ev_indexes = [idx['name'] for idx in inspector.get_indexes('knowledge_evidence', schema=target_schema)]
    if 'ix_knowledge_evidence_source_url' not in ev_indexes:
        op.create_index(
            'ix_knowledge_evidence_source_url',
            'knowledge_evidence',
            ['source_url'],
            unique=False,
            schema=target_schema,
        )

    # Check constraint on verification_status
    ev_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_evidence', schema=target_schema)]
    if 'ck_knowledgeevidence_verification_status' not in ev_checks:
        op.create_check_constraint(
            'ck_knowledgeevidence_verification_status',
            'knowledge_evidence',
            "verification_status IS NULL OR verification_status IN ('NOT_VERIFIED', 'VERIFIED', 'FAILED')",
            schema=target_schema,
        )

    # Update single-source check constraint
    if 'ck_knowledgeevidence_single_source' in ev_checks:
        op.drop_constraint('ck_knowledgeevidence_single_source', 'knowledge_evidence', type_='check', schema=target_schema)

    op.create_check_constraint(
        'ck_knowledgeevidence_single_source',
        'knowledge_evidence',
        "(document_chunk_id IS NOT NULL AND research_evidence_id IS NULL AND conversation_id IS NULL AND chat_message_id IS NULL AND source_url IS NULL) OR "
        "(document_chunk_id IS NULL AND research_evidence_id IS NOT NULL AND conversation_id IS NULL AND chat_message_id IS NULL AND source_url IS NULL) OR "
        "(document_chunk_id IS NULL AND research_evidence_id IS NULL AND conversation_id IS NOT NULL AND source_url IS NULL) OR "
        "(document_chunk_id IS NULL AND research_evidence_id IS NULL AND conversation_id IS NULL AND chat_message_id IS NULL AND source_url IS NOT NULL)",
        schema=target_schema,
    )

    # Update unique index
    op.execute(sa.text(f"DROP INDEX IF EXISTS {schema_prefix}uq_knowledge_evidence_source"))
    op.execute(sa.text(
        f"CREATE UNIQUE INDEX uq_knowledge_evidence_source ON {schema_prefix}knowledge_evidence "
        "(knowledge_item_id, document_chunk_id, research_evidence_id, conversation_id, chat_message_id, source_url) "
        "NULLS NOT DISTINCT"
    ))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None
    schema_prefix = f"{current_schema}." if current_schema else ""

    op.execute(sa.text(f"DROP INDEX IF EXISTS {schema_prefix}uq_knowledge_evidence_source"))

    ev_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_evidence', schema=target_schema)]
    if 'ck_knowledgeevidence_single_source' in ev_checks:
        op.drop_constraint('ck_knowledgeevidence_single_source', 'knowledge_evidence', type_='check', schema=target_schema)
    if 'ck_knowledgeevidence_verification_status' in ev_checks:
        op.drop_constraint('ck_knowledgeevidence_verification_status', 'knowledge_evidence', type_='check', schema=target_schema)

    op.create_check_constraint(
        'ck_knowledgeevidence_single_source',
        'knowledge_evidence',
        "(document_chunk_id IS NOT NULL AND research_evidence_id IS NULL AND conversation_id IS NULL AND chat_message_id IS NULL) OR "
        "(document_chunk_id IS NULL AND research_evidence_id IS NOT NULL AND conversation_id IS NULL AND chat_message_id IS NULL) OR "
        "(document_chunk_id IS NULL AND research_evidence_id IS NULL AND conversation_id IS NOT NULL)",
        schema=target_schema,
    )

    op.execute(sa.text(
        f"CREATE UNIQUE INDEX uq_knowledge_evidence_source ON {schema_prefix}knowledge_evidence "
        "(knowledge_item_id, document_chunk_id, research_evidence_id, conversation_id, chat_message_id) "
        "NULLS NOT DISTINCT"
    ))

    ev_indexes = [idx['name'] for idx in inspector.get_indexes('knowledge_evidence', schema=target_schema)]
    if 'ix_knowledge_evidence_source_url' in ev_indexes:
        op.drop_index('ix_knowledge_evidence_source_url', 'knowledge_evidence', schema=target_schema)

    ev_cols = [c['name'] for c in inspector.get_columns('knowledge_evidence', schema=target_schema)]
    for col in ('content_hash', 'verified_at', 'verification_status', 'user_excerpt', 'source_title', 'source_url'):
        if col in ev_cols:
            op.drop_column('knowledge_evidence', col, schema=target_schema)
