"""add_authorization_schema

Revision ID: 88cd8f977c32
Revises: 52727b7a398e
Create Date: 2026-09-22 15:35:14.175832
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '88cd8f977c32'
down_revision: Union[str, Sequence[str], None] = '52727b7a398e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # permissions table
    op.create_table(
        'permissions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('code', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code')
    )

    # users table
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # departments table
    op.create_table(
        'departments',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('code', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code')
    )

    # user_departments table
    op.create_table(
        'user_departments',
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('department_id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['department_id'], ['departments.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'department_id')
    )
    op.create_index(op.f('ix_user_departments_department_id'), 'user_departments', ['department_id'], unique=False)

    # user_permissions table
    op.create_table(
        'user_permissions',
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('department_id', sa.UUID(), nullable=False),
        sa.Column('permission_id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['permission_id'], ['permissions.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['user_id', 'department_id'], ['user_departments.user_id', 'user_departments.department_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'department_id', 'permission_id')
    )

    # Seed permissions
    op.execute(
        """
        INSERT INTO permissions (id, code, description, is_active)
        VALUES 
            (gen_random_uuid(), 'VIEW', 'View access', true),
            (gen_random_uuid(), 'UPLOAD', 'Upload access', true),
            (gen_random_uuid(), 'EDIT', 'Edit access', true),
            (gen_random_uuid(), 'DELETE', 'Delete access', true)
        ON CONFLICT (code) DO NOTHING;
        """
    )


def downgrade() -> None:
    op.drop_table('user_permissions')
    op.drop_index(op.f('ix_user_departments_department_id'), table_name='user_departments')
    op.drop_table('user_departments')
    op.drop_table('departments')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    op.drop_table('permissions')
