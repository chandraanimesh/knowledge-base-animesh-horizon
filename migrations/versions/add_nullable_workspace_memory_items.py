"""add_nullable_workspace_memory_items

Revision ID: add_nullable_workspace_memory_items
Revises: add_publication_invariants
Create Date: 2026-10-10 18:00:00.000000

Supports dual-scope personal memory:
- Global personal memories: workspace_id IS NULL (applies across the entire organization/platform)
- Workspace-scoped memories: workspace_id IS NOT NULL (applies strictly to a specific workspace)

Alters memory_items.workspace_id to be NULLABLE.
Updates partial unique indexes to guarantee:
1. (user_id, workspace_id, canonical_key) is unique where status = 'ACTIVE' AND workspace_id IS NOT NULL
2. (user_id, canonical_key) is unique where status = 'ACTIVE' AND workspace_id IS NULL
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'add_nullable_workspace_memory_items'
down_revision: Union[str, Sequence[str], None] = 'add_publication_invariants'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()

    # 1. Alter workspace_id column to nullable
    op.alter_column(
        'memory_items',
        'workspace_id',
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
        schema=current_schema,
    )

    # 2. Inspect existing indexes
    existing_indexes = [idx['name'] for idx in inspector.get_indexes('memory_items', schema=current_schema)]

    # Drop old index if present
    if 'ix_memory_items_user_workspace_canonical_key_active' in existing_indexes:
        op.drop_index(
            'ix_memory_items_user_workspace_canonical_key_active',
            table_name='memory_items',
            schema=current_schema,
        )

    # Create workspace-scoped active unique index (workspace_id IS NOT NULL)
    op.create_index(
        'ix_memory_items_user_workspace_canonical_key_active',
        'memory_items',
        ['user_id', 'workspace_id', 'canonical_key'],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE' AND workspace_id IS NOT NULL"),
        schema=current_schema,
    )

    # Create global personal memory active unique index (workspace_id IS NULL)
    if 'ix_memory_items_user_global_canonical_key_active' not in existing_indexes:
        op.create_index(
            'ix_memory_items_user_global_canonical_key_active',
            'memory_items',
            ['user_id', 'canonical_key'],
            unique=True,
            postgresql_where=sa.text("status = 'ACTIVE' AND workspace_id IS NULL"),
            schema=current_schema,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()

    # 1. Drop global unique index
    existing_indexes = [idx['name'] for idx in inspector.get_indexes('memory_items', schema=current_schema)]
    if 'ix_memory_items_user_global_canonical_key_active' in existing_indexes:
        op.drop_index(
            'ix_memory_items_user_global_canonical_key_active',
            table_name='memory_items',
            schema=current_schema,
        )

    # 2. Drop modified workspace index
    if 'ix_memory_items_user_workspace_canonical_key_active' in existing_indexes:
        op.drop_index(
            'ix_memory_items_user_workspace_canonical_key_active',
            table_name='memory_items',
            schema=current_schema,
        )

    # 3. Re-create original workspace index
    op.create_index(
        'ix_memory_items_user_workspace_canonical_key_active',
        'memory_items',
        ['user_id', 'workspace_id', 'canonical_key'],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
        schema=current_schema,
    )

    # 4. Make workspace_id NOT NULL again (only if no null rows exist)
    null_count = bind.execute(sa.text(f"SELECT COUNT(*) FROM {current_schema + '.' if current_schema else ''}memory_items WHERE workspace_id IS NULL")).scalar()
    if null_count == 0:
        op.alter_column(
            'memory_items',
            'workspace_id',
            existing_type=postgresql.UUID(as_uuid=True),
            nullable=False,
            schema=current_schema,
        )
