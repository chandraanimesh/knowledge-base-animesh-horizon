"""Add Knowledge Models

Revision ID: 8e0cc0dc0ed6
Revises: 038390e5e21a
Create Date: 2026-09-28 12:14:19.018715
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8e0cc0dc0ed6'
down_revision: Union[str, Sequence[str], None] = '038390e5e21a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'knowledge_items',
        sa.Column('id', sa.UUID(as_uuid=True), primary_key=True),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='ACTIVE'),
        sa.Column('department_id', sa.UUID(as_uuid=True), sa.ForeignKey('departments.id', ondelete='RESTRICT'), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("kind IN ('ENTITY', 'FACT', 'EVENT', 'RELATIONSHIP')", name='ck_knowledgeitem_kind'),
        sa.CheckConstraint("status IN ('ACTIVE', 'SUPERSEDED', 'ARCHIVED')", name='ck_knowledgeitem_status')
    )
    op.create_index(op.f('ix_knowledge_items_department_id'), 'knowledge_items', ['department_id'], unique=False)

    op.create_table(
        'knowledge_entities',
        sa.Column('id', sa.UUID(as_uuid=True), primary_key=True),
        sa.Column('knowledge_item_id', sa.UUID(as_uuid=True), sa.ForeignKey('knowledge_items.id', ondelete='CASCADE'), nullable=False),
        sa.Column('entity_type', sa.String(), nullable=False),
        sa.Column('canonical_name', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    op.create_index(op.f('ix_knowledge_entities_knowledge_item_id'), 'knowledge_entities', ['knowledge_item_id'], unique=False)
    op.create_index(op.f('ix_knowledge_entities_canonical_name'), 'knowledge_entities', ['canonical_name'], unique=False)

    op.create_table(
        'knowledge_relationships',
        sa.Column('id', sa.UUID(as_uuid=True), primary_key=True),
        sa.Column('knowledge_item_id', sa.UUID(as_uuid=True), sa.ForeignKey('knowledge_items.id', ondelete='CASCADE'), nullable=False),
        sa.Column('subject_entity_id', sa.UUID(as_uuid=True), sa.ForeignKey('knowledge_entities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('predicate', sa.String(), nullable=False),
        sa.Column('object_entity_id', sa.UUID(as_uuid=True), sa.ForeignKey('knowledge_entities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )
    op.create_index(op.f('ix_knowledge_relationships_knowledge_item_id'), 'knowledge_relationships', ['knowledge_item_id'], unique=False)
    op.create_index(op.f('ix_knowledge_relationships_subject_entity_id'), 'knowledge_relationships', ['subject_entity_id'], unique=False)
    op.create_index(op.f('ix_knowledge_relationships_object_entity_id'), 'knowledge_relationships', ['object_entity_id'], unique=False)

    op.create_table(
        'knowledge_evidence',
        sa.Column('id', sa.UUID(as_uuid=True), primary_key=True),
        sa.Column('knowledge_item_id', sa.UUID(as_uuid=True), sa.ForeignKey('knowledge_items.id', ondelete='CASCADE'), nullable=False),
        sa.Column('document_chunk_id', sa.UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='CASCADE'), nullable=True),
        sa.Column('research_evidence_id', sa.UUID(as_uuid=True), sa.ForeignKey('research_evidence.id', ondelete='CASCADE'), nullable=True),
        sa.Column('extraction_method', sa.String(), nullable=True),
        sa.Column('extraction_run_id', sa.UUID(as_uuid=True), sa.ForeignKey('research_runs.id', ondelete='SET NULL'), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint(
            "(document_chunk_id IS NOT NULL AND research_evidence_id IS NULL) OR (document_chunk_id IS NULL AND research_evidence_id IS NOT NULL)",
            name='ck_knowledgeevidence_single_source'
        )
    )
    op.create_index(op.f('ix_knowledge_evidence_knowledge_item_id'), 'knowledge_evidence', ['knowledge_item_id'], unique=False)
    op.create_index(op.f('ix_knowledge_evidence_document_chunk_id'), 'knowledge_evidence', ['document_chunk_id'], unique=False)
    op.create_index(op.f('ix_knowledge_evidence_research_evidence_id'), 'knowledge_evidence', ['research_evidence_id'], unique=False)
    op.create_index(op.f('ix_knowledge_evidence_extraction_run_id'), 'knowledge_evidence', ['extraction_run_id'], unique=False)


def downgrade() -> None:
    op.drop_table('knowledge_evidence')
    op.drop_table('knowledge_relationships')
    op.drop_table('knowledge_entities')
    op.drop_table('knowledge_items')
