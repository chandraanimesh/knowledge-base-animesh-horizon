"""make researchrun department_id nullable

Revision ID: 135fb346053d
Revises: ae02a0de228e
Create Date: 2026-09-24 13:17:53.913564
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '135fb346053d'
down_revision: Union[str, Sequence[str], None] = 'ae02a0de228e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('research_runs', 'department_id',
               existing_type=sa.UUID(),
               nullable=True)


def downgrade() -> None:
    op.alter_column('research_runs', 'department_id',
               existing_type=sa.UUID(),
               nullable=False)
