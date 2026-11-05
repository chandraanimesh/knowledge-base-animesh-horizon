"""add_workflow_run_provider_snapshot

Revision ID: add_workflow_run_provider_snapshot
Revises: add_workflow_triggers_and_schedules
Create Date: 2026-11-03 15:10:00.000000

Adds provider_snapshot JSONB column to workflow_runs table for immutable runtime configuration capture (Phase 5E).
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "add_workflow_run_provider_snapshot"
down_revision: Union[str, Sequence[str], None] = "add_workflow_triggers_and_schedules"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns("workflow_runs")]

    if "provider_snapshot" not in columns:
        op.add_column(
            "workflow_runs",
            sa.Column("provider_snapshot", JSONB, nullable=True),
        )

    # Phase 5E.1: Enforce snapshot immutability at database engine level against bulk/Core updates
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_workflow_run_snapshot_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
            IF OLD.provider_snapshot IS NOT NULL AND OLD.provider_snapshot != 'null'::jsonb AND NEW.provider_snapshot IS DISTINCT FROM OLD.provider_snapshot THEN
                RAISE EXCEPTION 'ImmutableRuntimeSnapshotError: provider_snapshot column cannot be mutated or cleared once initialized.';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("DROP TRIGGER IF EXISTS trg_workflow_runs_provider_snapshot_immutable ON workflow_runs;")
    op.execute("""
        CREATE TRIGGER trg_workflow_runs_provider_snapshot_immutable
        BEFORE UPDATE ON workflow_runs
        FOR EACH ROW
        EXECUTE FUNCTION prevent_workflow_run_snapshot_mutation();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_workflow_runs_provider_snapshot_immutable ON workflow_runs;")
    op.execute("DROP FUNCTION IF EXISTS prevent_workflow_run_snapshot_mutation();")

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns("workflow_runs")]

    if "provider_snapshot" in columns:
        op.drop_column("workflow_runs", "provider_snapshot")

