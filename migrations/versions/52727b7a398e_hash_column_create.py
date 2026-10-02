"""hash column create

Revision ID: 52727b7a398e
Revises: 20260817_0001
Create Date: 2026-08-22 11:31:09.152168
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '52727b7a398e'
down_revision: Union[str, Sequence[str], None] = '20260817_0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('document_chunks', sa.Column('chunk_hash', sa.String(), nullable=True))
    op.create_index(op.f('ix_document_chunks_chunk_hash'), 'document_chunks', ['chunk_hash'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_document_chunks_chunk_hash'), table_name='document_chunks')
    op.drop_column('document_chunks', 'chunk_hash')
