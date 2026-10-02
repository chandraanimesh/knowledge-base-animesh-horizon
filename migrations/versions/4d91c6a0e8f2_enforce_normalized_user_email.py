"""enforce normalized user email uniqueness

Revision ID: 4d91c6a0e8f2
Revises: 76b9e7ffe263
Create Date: 2026-09-22 00:00:00
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "4d91c6a0e8f2"
down_revision: Union[str, Sequence[str], None] = "76b9e7ffe263"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM users
                GROUP BY lower(btrim(email))
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION
                    'Cannot enforce normalized email uniqueness: duplicate normalized emails exist';
            END IF;
        END $$;
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_users_email_normalized
        ON users (lower(btrim(email)))
        """
    )


def downgrade() -> None:
    op.drop_index("uq_users_email_normalized", table_name="users")
