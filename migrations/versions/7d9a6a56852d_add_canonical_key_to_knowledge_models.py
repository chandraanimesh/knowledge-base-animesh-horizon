"""Add canonical_key to Knowledge models

Revision ID: 7d9a6a56852d
Revises: 8e0cc0dc0ed6
Create Date: 2026-09-28 12:57:08.918760
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7d9a6a56852d'
down_revision: Union[str, Sequence[str], None] = '8e0cc0dc0ed6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # KnowledgeItem canonical_key
    op.add_column("knowledge_items", sa.Column("canonical_key", sa.String(), nullable=True))
    op.execute("UPDATE knowledge_items SET canonical_key = 'LEGACY:' || id::text WHERE canonical_key IS NULL")
    op.alter_column("knowledge_items", "canonical_key", nullable=False)
    
    op.execute("CREATE UNIQUE INDEX ix_knowledge_items_canonical_key_dept ON knowledge_items (canonical_key, department_id) NULLS NOT DISTINCT")
    
    # KnowledgeEntity uniqueness
    op.create_unique_constraint("uq_knowledge_entity_item", "knowledge_entities", ["knowledge_item_id"])
    
    # KnowledgeRelationship uniqueness
    op.create_unique_constraint("uq_knowledge_relationship_item", "knowledge_relationships", ["knowledge_item_id"])


def downgrade() -> None:
    op.drop_constraint("uq_knowledge_relationship_item", "knowledge_relationships", type_="unique")
    op.drop_constraint("uq_knowledge_entity_item", "knowledge_entities", type_="unique")
    op.execute("DROP INDEX ix_knowledge_items_canonical_key_dept")
    op.drop_column("knowledge_items", "canonical_key")
