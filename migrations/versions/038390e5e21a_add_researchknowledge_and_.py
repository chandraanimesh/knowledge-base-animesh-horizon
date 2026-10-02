"""Add ResearchKnowledge and ResearchEvidence tables

Revision ID: 038390e5e21a
Revises: 135fb346053d
Create Date: 2026-09-24 14:07:04.599021
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '038390e5e21a'
down_revision: Union[str, Sequence[str], None] = '135fb346053d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


from sqlalchemy.dialects.postgresql import UUID, JSONB

def upgrade() -> None:
    op.create_table(
        'research_knowledge',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('research_run_id', UUID(as_uuid=True), sa.ForeignKey('research_runs.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('department_id', UUID(as_uuid=True), sa.ForeignKey('departments.id', ondelete='RESTRICT'), nullable=True, index=True),
        sa.Column('entity_name', sa.String(), nullable=False),
        sa.Column('entity_type', sa.String(), nullable=False),
        sa.Column('attributes', JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('research_run_id', 'entity_name', 'entity_type', name='uq_research_knowledge_entity')
    )
    
    op.create_table(
        'research_evidence',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('knowledge_id', UUID(as_uuid=True), sa.ForeignKey('research_knowledge.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('source_url', sa.String(), nullable=False),
        sa.Column('source_title', sa.String(), nullable=True),
        sa.Column('evidence_text', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False)
    )

def downgrade() -> None:
    op.drop_table('research_evidence')
    op.drop_table('research_knowledge')
