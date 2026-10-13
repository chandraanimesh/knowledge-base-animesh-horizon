"""add_shared_knowledge_publication_invariants

Revision ID: add_publication_invariants
Revises: add_shared_knowledge_schema
Create Date: 2026-10-10 14:40:00.000000

Adds database-level invariant constraint:
- ck_knowledgeitem_shared_requires_approved:
  Guarantees that a record cannot have visibility='SHARED' unless approval_status='APPROVED'.
- Non-destructive downgrade drops the check constraint cleanly without deleting tables or data.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'add_publication_invariants'
down_revision: Union[str, Sequence[str], None] = 'add_shared_knowledge_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()

    ki_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_items', schema=current_schema)]
    if 'ck_knowledgeitem_shared_requires_approved' not in ki_checks:
        op.create_check_constraint(
            'ck_knowledgeitem_shared_requires_approved',
            'knowledge_items',
            "(visibility != 'SHARED') OR (approval_status = 'APPROVED')",
            schema=current_schema,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()

    ki_checks = [c['name'] for c in inspector.get_check_constraints('knowledge_items', schema=current_schema)]
    if 'ck_knowledgeitem_shared_requires_approved' in ki_checks:
        op.drop_constraint(
            'ck_knowledgeitem_shared_requires_approved',
            'knowledge_items',
            type_='check',
            schema=current_schema,
        )
