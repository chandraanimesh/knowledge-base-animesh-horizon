"""add_memory_processing_job_source_trigger_unique_index

Revision ID: add_memory_processing_job_source_trigger_unique_index
Revises: add_generated_artifacts_table
Create Date: 2026-10-21 18:00:00.000000

Adds unique partial index on memory_processing_jobs(source_message_id, trigger_kind)
WHERE source_message_id IS NOT NULL to guarantee PostgreSQL engine-level idempotency.
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = "add_memory_processing_job_source_trigger_unique_index"
down_revision: Union[str, Sequence[str], None] = "add_generated_artifacts_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Safety check for existing data: detect duplicates before index creation
    dup_check = bind.execute(sa.text("""
        SELECT source_message_id, trigger_kind, COUNT(*) AS cnt
        FROM memory_processing_jobs
        WHERE source_message_id IS NOT NULL
        GROUP BY source_message_id, trigger_kind
        HAVING COUNT(*) > 1;
    """)).fetchall()

    if dup_check:
        dup_details = ", ".join(f"(source_msg={row[0]}, trigger={row[1]}, count={row[2]})" for row in dup_check)
        raise RuntimeError(
            f"Cannot create unique index on memory_processing_jobs: duplicate entries exist: {dup_details}. "
            "Please resolve duplicate entries before applying this migration."
        )

    # 2. Create unique partial index
    op.create_index(
        "ix_memory_processing_jobs_source_trigger_unique",
        "memory_processing_jobs",
        ["source_message_id", "trigger_kind"],
        unique=True,
        postgresql_where=sa.text("source_message_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_processing_jobs_source_trigger_unique",
        table_name="memory_processing_jobs",
    )
