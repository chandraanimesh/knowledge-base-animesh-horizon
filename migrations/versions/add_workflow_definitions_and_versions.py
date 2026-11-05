"""add_workflow_definitions_and_versions

Revision ID: add_workflow_definitions_and_versions
Revises: add_chunk_versions_and_curation
Create Date: 2026-11-02 13:00:00.000000

Adds persistent Workflow Definition and Versioning tables and links to WorkflowRun:
- workflow_definitions: Workspace-scoped workflow metadata, owner, active version pointer
- workflow_versions: Immutable published snapshots, version numbering, trigger/steps DAG specs, canvas layout
- workflow_runs: Optional foreign keys to workflow_definitions.id and workflow_versions.id
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "add_workflow_definitions_and_versions"
down_revision: Union[str, Sequence[str], None] = "add_chunk_versions_and_curation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # 1. Create workflow_definitions table
    if "workflow_definitions" not in tables:
        op.create_table(
            "workflow_definitions",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("created_by_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("active_version_id", UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_workflow_definitions_workspace_id", "workflow_definitions", ["workspace_id"])
        op.create_index("ix_workflow_definitions_created_by_id", "workflow_definitions", ["created_by_id"])

    # 2. Create workflow_versions table
    if "workflow_versions" not in tables:
        op.create_table(
            "workflow_versions",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("workflow_id", UUID(as_uuid=True), sa.ForeignKey("workflow_definitions.id", ondelete="CASCADE"), nullable=False),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="DRAFT"),
            sa.Column("trigger_spec", JSONB, nullable=False, server_default="{}"),
            sa.Column("steps_spec", JSONB, nullable=False, server_default="[]"),
            sa.Column("canvas_layout", JSONB, nullable=True, server_default="{}"),
            sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("created_by_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "status IN ('DRAFT', 'ACTIVE', 'ARCHIVED')",
                name="ck_workflow_version_status",
            ),
            sa.UniqueConstraint("workflow_id", "version_number", name="uq_workflow_versions_wf_version"),
        )
        op.create_index("ix_workflow_versions_workflow_id", "workflow_versions", ["workflow_id"])
        op.create_index("ix_workflow_versions_wf_status", "workflow_versions", ["workflow_id", "status"])

    # 3. Add foreign key from workflow_definitions.active_version_id to workflow_versions.id
    # Re-inspect to see if constraint exists
    fks = inspector.get_foreign_keys("workflow_definitions") if "workflow_definitions" in tables else []
    fk_names = [fk["name"] for fk in fks]
    if "fk_workflow_definitions_active_version" not in fk_names:
        op.create_foreign_key(
            "fk_workflow_definitions_active_version",
            "workflow_definitions",
            "workflow_versions",
            ["active_version_id"],
            ["id"],
            ondelete="SET NULL",
            use_alter=True,
        )

    # 4. Extend workflow_runs table with optional workflow_id and workflow_version_id
    wf_run_cols = [c["name"] for c in inspector.get_columns("workflow_runs")]
    if "workflow_id" not in wf_run_cols:
        op.add_column(
            "workflow_runs",
            sa.Column("workflow_id", UUID(as_uuid=True), sa.ForeignKey("workflow_definitions.id", ondelete="SET NULL"), nullable=True),
        )
        op.create_index("ix_workflow_runs_workflow_id", "workflow_runs", ["workflow_id"])

    if "workflow_version_id" not in wf_run_cols:
        op.add_column(
            "workflow_runs",
            sa.Column("workflow_version_id", UUID(as_uuid=True), sa.ForeignKey("workflow_versions.id", ondelete="RESTRICT"), nullable=True),
        )
        op.create_index("ix_workflow_runs_workflow_version_id", "workflow_runs", ["workflow_version_id"])


def downgrade() -> None:
    # 1. Drop workflow_runs columns
    op.drop_index("ix_workflow_runs_workflow_version_id", table_name="workflow_runs")
    op.drop_column("workflow_runs", "workflow_version_id")
    op.drop_index("ix_workflow_runs_workflow_id", table_name="workflow_runs")
    op.drop_column("workflow_runs", "workflow_id")

    # 2. Drop active_version_id foreign key constraint
    op.drop_constraint("fk_workflow_definitions_active_version", "workflow_definitions", type_="foreignkey")

    # 3. Drop tables
    op.drop_table("workflow_versions")
    op.drop_table("workflow_definitions")

