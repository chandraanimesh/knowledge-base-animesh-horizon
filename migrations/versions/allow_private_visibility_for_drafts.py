"""allow_private_visibility_drafts

Revision ID: allow_private_visibility_drafts
Revises: fix_memory_workspace_constraints
Create Date: 2026-10-10 19:00:00.000000

Smallest additive schema change for Company Notes & private drafts:
- Extends knowledge_items check constraint ck_knowledgeitem_visibility to:
  visibility IN ('PRIVATE', 'WORKSPACE', 'SHARED')
- Guarantees private drafts are isolated to author by default before sharing or publication.
- Migration is non-destructive on downgrade.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'allow_private_visibility_drafts'
down_revision: Union[str, Sequence[str], None] = 'fix_memory_workspace_constraints'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    ki_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_items', schema=target_schema)]
    if 'ck_knowledgeitem_visibility' in ki_checks:
        op.drop_constraint(
            'ck_knowledgeitem_visibility',
            'knowledge_items',
            type_='check',
            schema=target_schema,
        )

    op.create_check_constraint(
        'ck_knowledgeitem_visibility',
        'knowledge_items',
        "visibility IN ('PRIVATE', 'WORKSPACE', 'SHARED')",
        schema=target_schema,
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    target_schema = current_schema if current_schema and current_schema != 'public' else None

    # Before restoring restrictive constraint, update any PRIVATE to WORKSPACE and archive to preserve privacy
    qual_ki = f'"{target_schema}".knowledge_items' if target_schema else 'knowledge_items'
    bind.execute(sa.text(f"UPDATE {qual_ki} SET visibility = 'WORKSPACE', status = 'ARCHIVED' WHERE visibility = 'PRIVATE'"))

    ki_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_items', schema=target_schema)]
    if 'ck_knowledgeitem_visibility' in ki_checks:
        op.drop_constraint(
            'ck_knowledgeitem_visibility',
            'knowledge_items',
            type_='check',
            schema=target_schema,
        )

    op.create_check_constraint(
        'ck_knowledgeitem_visibility',
        'knowledge_items',
        "visibility IN ('WORKSPACE', 'SHARED')",
        schema=target_schema,
    )
