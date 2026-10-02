"""add_access_requests_persistence

Revision ID: a1b2c3d4e5f6
Revises: 56f9f2a95820
Create Date: 2026-10-01 12:12:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '56f9f2a95820'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. access_requests table
    op.create_table(
        'access_requests',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('department_id', sa.UUID(), nullable=False),
        sa.Column(
            'status',
            sa.String(),
            server_default='PENDING',
            nullable=False,
        ),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column(
            'requested_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reviewed_by', sa.UUID(), nullable=True),
        sa.Column('review_note', sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('PENDING', 'APPROVED', 'REJECTED', 'CANCELLED')",
            name='ck_accessrequest_status',
        ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['department_id'], ['departments.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['reviewed_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_access_requests_user_id', 'access_requests', ['user_id'])
    op.create_index('ix_access_requests_department_id', 'access_requests', ['department_id'])
    op.create_index('ix_access_requests_status', 'access_requests', ['status'])

    # Partial unique index: prevent more than one PENDING request for the same (user_id, department_id)
    op.execute(
        """
        CREATE UNIQUE INDEX uq_access_requests_user_dept_pending
        ON access_requests (user_id, department_id)
        WHERE status = 'PENDING'
        """
    )

    # 2. access_request_permissions child table
    op.create_table(
        'access_request_permissions',
        sa.Column('access_request_id', sa.UUID(), nullable=False),
        sa.Column('permission_id', sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ['access_request_id'],
            ['access_requests.id'],
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['permission_id'],
            ['permissions.id'],
            ondelete='RESTRICT',
        ),
        sa.PrimaryKeyConstraint('access_request_id', 'permission_id'),
    )


def downgrade() -> None:
    op.drop_table('access_request_permissions')
    op.drop_index('uq_access_requests_user_dept_pending', table_name='access_requests')
    op.drop_index('ix_access_requests_status', table_name='access_requests')
    op.drop_index('ix_access_requests_department_id', table_name='access_requests')
    op.drop_index('ix_access_requests_user_id', table_name='access_requests')
    op.drop_table('access_requests')
