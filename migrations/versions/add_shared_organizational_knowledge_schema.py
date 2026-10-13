"""add_shared_organizational_knowledge_schema

Revision ID: add_shared_knowledge_schema
Revises: add_workspace_scoped_memory_items
Create Date: 2026-10-10 14:15:00.000000

Adds schema extensions for shared organizational knowledge:
- Extends knowledge_items with visibility, approval_status, source_type,
  workspace_id, author_id, approved_by_id, approved_at.
- Extends knowledge_evidence with conversation_id and chat_message_id,
  updating the single-source constraint and unique source constraint.
- Migration is non-destructive on downgrade.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'add_shared_knowledge_schema'
down_revision: Union[str, Sequence[str], None] = 'add_workspace_scoped_memory_items'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()

    # 1. Extend knowledge_items table
    ki_cols = [c['name'] for c in inspector.get_columns('knowledge_items', schema=current_schema)]

    if 'visibility' not in ki_cols:
        op.add_column(
            'knowledge_items',
            sa.Column('visibility', sa.String(20), nullable=False, server_default='WORKSPACE'),
            schema=current_schema,
        )
    if 'approval_status' not in ki_cols:
        op.add_column(
            'knowledge_items',
            sa.Column('approval_status', sa.String(20), nullable=False, server_default='DRAFT'),
            schema=current_schema,
        )
    if 'source_type' not in ki_cols:
        op.add_column(
            'knowledge_items',
            sa.Column('source_type', sa.String(20), nullable=False, server_default='DOCUMENT'),
            schema=current_schema,
        )
    if 'workspace_id' not in ki_cols:
        op.add_column(
            'knowledge_items',
            sa.Column(
                'workspace_id',
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey(f'{current_schema}.workspaces.id' if current_schema else 'workspaces.id', ondelete='SET NULL'),
                nullable=True,
            ),
            schema=current_schema,
        )
    if 'author_id' not in ki_cols:
        op.add_column(
            'knowledge_items',
            sa.Column(
                'author_id',
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey(f'{current_schema}.users.id' if current_schema else 'users.id', ondelete='SET NULL'),
                nullable=True,
            ),
            schema=current_schema,
        )
    if 'approved_by_id' not in ki_cols:
        op.add_column(
            'knowledge_items',
            sa.Column(
                'approved_by_id',
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey(f'{current_schema}.users.id' if current_schema else 'users.id', ondelete='SET NULL'),
                nullable=True,
            ),
            schema=current_schema,
        )
    if 'approved_at' not in ki_cols:
        op.add_column(
            'knowledge_items',
            sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
            schema=current_schema,
        )

    # Check constraints on knowledge_items
    ki_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_items', schema=current_schema)]
    if 'ck_knowledgeitem_visibility' not in ki_checks:
        op.create_check_constraint(
            'ck_knowledgeitem_visibility',
            'knowledge_items',
            "visibility IN ('WORKSPACE', 'SHARED')",
            schema=current_schema,
        )
    if 'ck_knowledgeitem_approval_status' not in ki_checks:
        op.create_check_constraint(
            'ck_knowledgeitem_approval_status',
            'knowledge_items',
            "approval_status IN ('DRAFT', 'PENDING', 'APPROVED', 'REJECTED')",
            schema=current_schema,
        )
    if 'ck_knowledgeitem_source_type' not in ki_checks:
        op.create_check_constraint(
            'ck_knowledgeitem_source_type',
            'knowledge_items',
            "source_type IN ('DOCUMENT', 'RESEARCH', 'CONVERSATION')",
            schema=current_schema,
        )
    if 'ck_knowledgeitem_approval_consistency' not in ki_checks:
        op.create_check_constraint(
            'ck_knowledgeitem_approval_consistency',
            'knowledge_items',
            "(approval_status != 'APPROVED') OR (approved_by_id IS NOT NULL AND approved_at IS NOT NULL)",
            schema=current_schema,
        )

    # Indexes on knowledge_items
    ki_indexes = [idx['name'] for idx in inspector.get_indexes('knowledge_items', schema=current_schema)]
    if 'ix_knowledge_items_workspace_id' not in ki_indexes:
        op.create_index(
            'ix_knowledge_items_workspace_id',
            'knowledge_items',
            ['workspace_id'],
            unique=False,
            schema=current_schema,
        )
    if 'ix_knowledge_items_author_id' not in ki_indexes:
        op.create_index(
            'ix_knowledge_items_author_id',
            'knowledge_items',
            ['author_id'],
            unique=False,
            schema=current_schema,
        )
    if 'ix_knowledge_items_approved_by_id' not in ki_indexes:
        op.create_index(
            'ix_knowledge_items_approved_by_id',
            'knowledge_items',
            ['approved_by_id'],
            unique=False,
            schema=current_schema,
        )
    if 'ix_knowledge_items_visibility_approval' not in ki_indexes:
        op.create_index(
            'ix_knowledge_items_visibility_approval',
            'knowledge_items',
            ['visibility', 'approval_status'],
            unique=False,
            schema=current_schema,
        )

    # 2. Extend knowledge_evidence table
    ev_cols = [c['name'] for c in inspector.get_columns('knowledge_evidence', schema=current_schema)]
    if 'conversation_id' not in ev_cols:
        op.add_column(
            'knowledge_evidence',
            sa.Column(
                'conversation_id',
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey(f'{current_schema}.conversations.id' if current_schema else 'conversations.id', ondelete='CASCADE'),
                nullable=True,
            ),
            schema=current_schema,
        )
    if 'chat_message_id' not in ev_cols:
        op.add_column(
            'knowledge_evidence',
            sa.Column(
                'chat_message_id',
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey(f'{current_schema}.chat_messages.id' if current_schema else 'chat_messages.id', ondelete='CASCADE'),
                nullable=True,
            ),
            schema=current_schema,
        )

    ev_indexes = [idx['name'] for idx in inspector.get_indexes('knowledge_evidence', schema=current_schema)]
    if 'ix_knowledge_evidence_conversation_id' not in ev_indexes:
        op.create_index(
            'ix_knowledge_evidence_conversation_id',
            'knowledge_evidence',
            ['conversation_id'],
            unique=False,
            schema=current_schema,
        )
    if 'ix_knowledge_evidence_chat_message_id' not in ev_indexes:
        op.create_index(
            'ix_knowledge_evidence_chat_message_id',
            'knowledge_evidence',
            ['chat_message_id'],
            unique=False,
            schema=current_schema,
        )

    # Update check constraint on knowledge_evidence
    ev_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_evidence', schema=current_schema)]
    if 'ck_knowledgeevidence_single_source' in ev_checks:
        op.drop_constraint('ck_knowledgeevidence_single_source', 'knowledge_evidence', type_='check', schema=current_schema)

    op.create_check_constraint(
        'ck_knowledgeevidence_single_source',
        'knowledge_evidence',
        "(document_chunk_id IS NOT NULL AND research_evidence_id IS NULL AND conversation_id IS NULL AND chat_message_id IS NULL) OR "
        "(document_chunk_id IS NULL AND research_evidence_id IS NOT NULL AND conversation_id IS NULL AND chat_message_id IS NULL) OR "
        "(document_chunk_id IS NULL AND research_evidence_id IS NULL AND conversation_id IS NOT NULL)",
        schema=current_schema,
    )

    # Update unique constraint on knowledge_evidence
    ev_uniques = [u['name'] for u in inspector.get_unique_constraints('knowledge_evidence', schema=current_schema)]
    if 'uq_knowledge_evidence_source' in ev_uniques:
        op.drop_constraint('uq_knowledge_evidence_source', 'knowledge_evidence', type_='unique', schema=current_schema)
    op.execute(sa.text(f"DROP INDEX IF EXISTS {current_schema + '.' if current_schema else ''}uq_knowledge_evidence_source"))

    schema_qual = f"{current_schema}." if current_schema else ""
    op.execute(sa.text(
        f"CREATE UNIQUE INDEX uq_knowledge_evidence_source ON {schema_qual}knowledge_evidence "
        "(knowledge_item_id, document_chunk_id, research_evidence_id, conversation_id, chat_message_id) "
        "NULLS NOT DISTINCT"
    ))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    schema_qual = f"{current_schema}." if current_schema else ""

    # 1. Revert knowledge_evidence constraints
    op.execute(sa.text(f"DROP INDEX IF EXISTS {schema_qual}uq_knowledge_evidence_source"))
    ev_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_evidence', schema=current_schema)]
    if 'ck_knowledgeevidence_single_source' in ev_checks:
        op.drop_constraint('ck_knowledgeevidence_single_source', 'knowledge_evidence', type_='check', schema=current_schema)

    # Restore legacy single-source check constraint
    op.create_check_constraint(
        'ck_knowledgeevidence_single_source',
        'knowledge_evidence',
        "(document_chunk_id IS NOT NULL AND research_evidence_id IS NULL) OR "
        "(document_chunk_id IS NULL AND research_evidence_id IS NOT NULL)",
        schema=current_schema,
    )
    op.execute(sa.text(
        f"CREATE UNIQUE INDEX uq_knowledge_evidence_source ON {schema_qual}knowledge_evidence "
        "(knowledge_item_id, document_chunk_id, research_evidence_id) "
        "NULLS NOT DISTINCT"
    ))

    # Drop knowledge_evidence columns and indexes
    ev_indexes = [idx['name'] for idx in inspector.get_indexes('knowledge_evidence', schema=current_schema)]
    if 'ix_knowledge_evidence_chat_message_id' in ev_indexes:
        op.drop_index('ix_knowledge_evidence_chat_message_id', table_name='knowledge_evidence', schema=current_schema)
    if 'ix_knowledge_evidence_conversation_id' in ev_indexes:
        op.drop_index('ix_knowledge_evidence_conversation_id', table_name='knowledge_evidence', schema=current_schema)

    ev_cols = [c['name'] for c in inspector.get_columns('knowledge_evidence', schema=current_schema)]
    if 'chat_message_id' in ev_cols:
        op.drop_column('knowledge_evidence', 'chat_message_id', schema=current_schema)
    if 'conversation_id' in ev_cols:
        op.drop_column('knowledge_evidence', 'conversation_id', schema=current_schema)

    # 2. Revert knowledge_items indexes, constraints, and columns
    ki_indexes = [idx['name'] for idx in inspector.get_indexes('knowledge_items', schema=current_schema)]
    if 'ix_knowledge_items_visibility_approval' in ki_indexes:
        op.drop_index('ix_knowledge_items_visibility_approval', table_name='knowledge_items', schema=current_schema)
    if 'ix_knowledge_items_approved_by_id' in ki_indexes:
        op.drop_index('ix_knowledge_items_approved_by_id', table_name='knowledge_items', schema=current_schema)
    if 'ix_knowledge_items_author_id' in ki_indexes:
        op.drop_index('ix_knowledge_items_author_id', table_name='knowledge_items', schema=current_schema)
    if 'ix_knowledge_items_workspace_id' in ki_indexes:
        op.drop_index('ix_knowledge_items_workspace_id', table_name='knowledge_items', schema=current_schema)

    ki_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_items', schema=current_schema)]
    if 'ck_knowledgeitem_approval_consistency' in ki_checks:
        op.drop_constraint('ck_knowledgeitem_approval_consistency', 'knowledge_items', type_='check', schema=current_schema)
    if 'ck_knowledgeitem_source_type' in ki_checks:
        op.drop_constraint('ck_knowledgeitem_source_type', 'knowledge_items', type_='check', schema=current_schema)
    if 'ck_knowledgeitem_approval_status' in ki_checks:
        op.drop_constraint('ck_knowledgeitem_approval_status', 'knowledge_items', type_='check', schema=current_schema)
    if 'ck_knowledgeitem_visibility' in ki_checks:
        op.drop_constraint('ck_knowledgeitem_visibility', 'knowledge_items', type_='check', schema=current_schema)

    ki_cols = [c['name'] for c in inspector.get_columns('knowledge_items', schema=current_schema)]
    if 'approved_at' in ki_cols:
        op.drop_column('knowledge_items', 'approved_at', schema=current_schema)
    if 'approved_by_id' in ki_cols:
        op.drop_column('knowledge_items', 'approved_by_id', schema=current_schema)
    if 'author_id' in ki_cols:
        op.drop_column('knowledge_items', 'author_id', schema=current_schema)
    if 'workspace_id' in ki_cols:
        op.drop_column('knowledge_items', 'workspace_id', schema=current_schema)
    if 'source_type' in ki_cols:
        op.drop_column('knowledge_items', 'source_type', schema=current_schema)
    if 'approval_status' in ki_cols:
        op.drop_column('knowledge_items', 'approval_status', schema=current_schema)
    if 'visibility' in ki_cols:
        op.drop_column('knowledge_items', 'visibility', schema=current_schema)
