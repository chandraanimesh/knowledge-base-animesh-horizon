"""Add attributes to KnowledgeEntity

Revision ID: 56f9f2a95820
Revises: 7d9a6a56852d
Create Date: 2026-09-28 15:30:33.469787
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '56f9f2a95820'
down_revision: Union[str, Sequence[str], None] = '7d9a6a56852d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


from sqlalchemy.dialects import postgresql

def upgrade() -> None:
    op.add_column('knowledge_entities', sa.Column('attributes', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False))


def downgrade() -> None:
    op.drop_column('knowledge_entities', 'attributes')
