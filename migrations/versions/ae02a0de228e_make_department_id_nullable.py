"""make department_id nullable

Revision ID: ae02a0de228e
Revises: 07db973a6a43
Create Date: 2026-09-24 13:10:05.326197
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ae02a0de228e'
down_revision: Union[str, Sequence[str], None] = '07db973a6a43'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('documents', 'department_id',
               existing_type=sa.UUID(),
               nullable=True)


def downgrade() -> None:
    op.alter_column('documents', 'department_id',
               existing_type=sa.UUID(),
               nullable=False)
