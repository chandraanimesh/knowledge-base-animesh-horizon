"""fix_memory_workspace_constraints

Revision ID: fix_memory_workspace_constraints
Revises: add_nullable_workspace_memory_items
Create Date: 2026-10-10 18:30:00.000000

Corrects MemoryItem workspace database constraints:
1. Replaces non-unique ix_memory_items_user_workspace_canonical_key_active with a UNIQUE partial index
   on (user_id, workspace_id, canonical_key) WHERE status = 'ACTIVE' AND workspace_id IS NOT NULL.
2. Restores referential integrity by adding a foreign key from memory_items.workspace_id to workspaces.id
   with ON DELETE RESTRICT (preventing workspace memories from silently becoming global or orphan memories).
3. Executes within a single transaction with duplicate and orphan precondition checks.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'fix_memory_workspace_constraints'
down_revision: Union[str, Sequence[str], None] = 'add_nullable_workspace_memory_items'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    qual_mem = f'"{current_schema}".memory_items' if current_schema and current_schema != 'public' else 'memory_items'
    qual_ws = f'"{current_schema}".workspaces' if current_schema and current_schema != 'public' else 'workspaces'

    # Precondition 1: Check for active duplicates before creating unique index
    duplicates = bind.execute(sa.text(f"""
        SELECT user_id, workspace_id, canonical_key, COUNT(*)
        FROM {qual_mem}
        WHERE status = 'ACTIVE' AND workspace_id IS NOT NULL
        GROUP BY user_id, workspace_id, canonical_key
        HAVING COUNT(*) > 1
    """)).fetchall()
    if duplicates:
        raise RuntimeError(
            f"Precondition failed for migration {revision}: found {len(duplicates)} duplicate active workspace memories"
        )

    # Precondition 2: Check for orphan workspace references before creating foreign key
    orphans = bind.execute(sa.text(f"""
        SELECT m.id, m.workspace_id
        FROM {qual_mem} m
        LEFT JOIN {qual_ws} w ON w.id = m.workspace_id
        WHERE m.workspace_id IS NOT NULL AND w.id IS NULL
    """)).fetchall()
    if orphans:
        raise RuntimeError(
            f"Precondition failed for migration {revision}: found {len(orphans)} orphan memory_items referencing non-existent workspaces"
        )

    # 1. Replace workspace index with UNIQUE partial index
    existing_indexes = [idx['name'] for idx in inspector.get_indexes('memory_items', schema=target_schema)]
    if 'ix_memory_items_user_workspace_canonical_key_active' in existing_indexes:
        op.drop_index(
            'ix_memory_items_user_workspace_canonical_key_active',
            table_name='memory_items',
            schema=target_schema,
        )

    op.create_index(
        'ix_memory_items_user_workspace_canonical_key_active',
        'memory_items',
        ['user_id', 'workspace_id', 'canonical_key'],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE' AND workspace_id IS NOT NULL"),
        schema=target_schema,
    )

    # 2. Add foreign key with ON DELETE RESTRICT
    existing_fks = [fk['name'] for fk in inspector.get_foreign_keys('memory_items', schema=target_schema)]
    if 'fk_memory_items_workspace_id_workspaces' not in existing_fks:
        op.create_foreign_key(
            'fk_memory_items_workspace_id_workspaces',
            'memory_items',
            'workspaces',
            ['workspace_id'],
            ['id'],
            ondelete='RESTRICT',
            source_schema=target_schema,
            referent_schema=target_schema,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    # 1. Drop foreign key constraint
    existing_fks = [fk['name'] for fk in inspector.get_foreign_keys('memory_items', schema=target_schema)]
    if 'fk_memory_items_workspace_id_workspaces' in existing_fks:
        op.drop_constraint(
            'fk_memory_items_workspace_id_workspaces',
            'memory_items',
            type_='foreignkey',
            schema=target_schema,
        )

    # 2. Re-create non-unique workspace index
    existing_indexes = [idx['name'] for idx in inspector.get_indexes('memory_items', schema=target_schema)]
    if 'ix_memory_items_user_workspace_canonical_key_active' in existing_indexes:
        op.drop_index(
            'ix_memory_items_user_workspace_canonical_key_active',
            table_name='memory_items',
            schema=target_schema,
        )

    op.create_index(
        'ix_memory_items_user_workspace_canonical_key_active',
        'memory_items',
        ['user_id', 'workspace_id', 'canonical_key'],
        unique=False,
        postgresql_where=sa.text("status = 'ACTIVE' AND workspace_id IS NOT NULL"),
        schema=target_schema,
    )
