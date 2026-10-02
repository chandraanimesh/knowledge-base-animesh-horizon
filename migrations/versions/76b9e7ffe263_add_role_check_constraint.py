"""add_role_check_constraint

Revision ID: 76b9e7ffe263
Revises: 88cd8f977c32
Create Date: 2026-09-22 15:37:44.263447
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '76b9e7ffe263'
down_revision: Union[str, Sequence[str], None] = '88cd8f977c32'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_check_constraint('ck_user_role', 'users', "role IN ('ADMIN', 'USER')")


def downgrade() -> None:
    op.drop_constraint('ck_user_role', 'users', type_='check')
