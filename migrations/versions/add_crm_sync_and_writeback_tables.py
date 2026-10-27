"""add_crm_sync_and_writeback_tables

Revision ID: add_crm_sync_and_writeback_tables
Revises: add_connection_discovery_tables
Create Date: 2026-10-14 01:25:00.000000

Adds tables for CRM Knowledge Base synchronization and controlled write-back:
- crm_sync_jobs: Durable batch/incremental sync job state and checkpoints
- crm_sync_records: Per-record sync ledger, fingerprints, idempotency, retry tracking
- crm_writeback_records: One approved write-back ledger (meeting summary Note creation)
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "add_crm_sync_and_writeback_tables"
down_revision: Union[str, Sequence[str], None] = "add_connection_discovery_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "crm_sync_jobs" not in tables:
        op.create_table(
            "crm_sync_jobs",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("job_type", sa.String(50), nullable=False, server_default="BATCH_SYNC"),
            sa.Column("status", sa.String(50), nullable=False, server_default="PENDING"),
            sa.Column("department_id", UUID(as_uuid=True), sa.ForeignKey("departments.id", ondelete="SET NULL"), nullable=True),
            sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True),
            sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("checkpoint_cursor", sa.String(255), nullable=True),
            sa.Column("total_records", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("processed_records", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("succeeded_records", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("skipped_records", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("failed_records", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_crm_sync_jobs_department_id", "crm_sync_jobs", ["department_id"])
        op.create_index("ix_crm_sync_jobs_workspace_id", "crm_sync_jobs", ["workspace_id"])

    if "crm_sync_records" not in tables:
        op.create_table(
            "crm_sync_records",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("crm_sync_jobs.id", ondelete="SET NULL"), nullable=True),
            sa.Column("provider", sa.String(50), nullable=False, server_default="twenty_crm"),
            sa.Column("entity_type", sa.String(50), nullable=False, server_default="company"),
            sa.Column("external_record_id", sa.String(255), nullable=False),
            sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
            sa.Column("department_id", UUID(as_uuid=True), sa.ForeignKey("departments.id", ondelete="RESTRICT"), nullable=True),
            sa.Column("workspace_id", UUID(as_uuid=True), sa.ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True),
            sa.Column("source_fingerprint", sa.String(64), nullable=True),
            sa.Column("status", sa.String(50), nullable=False, server_default="PENDING"),
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
            sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_attempted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("provider", "entity_type", "external_record_id", name="uq_crm_sync_records_provider_entity_external"),
        )
        op.create_index("ix_crm_sync_records_job_id", "crm_sync_records", ["job_id"])
        op.create_index("ix_crm_sync_records_external_record_id", "crm_sync_records", ["external_record_id"])
        op.create_index("ix_crm_sync_records_document_id", "crm_sync_records", ["document_id"])
        op.create_index("ix_crm_sync_records_department_id", "crm_sync_records", ["department_id"])
        op.create_index("ix_crm_sync_records_workspace_id", "crm_sync_records", ["workspace_id"])
        op.create_index("ix_crm_sync_records_status_retry", "crm_sync_records", ["status", "next_retry_at"])

    if "crm_writeback_records" not in tables:
        op.create_table(
            "crm_writeback_records",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("operation_type", sa.String(50), nullable=False, server_default="NOTE_CREATE"),
            sa.Column("provider", sa.String(50), nullable=False, server_default="twenty_crm"),
            sa.Column("external_record_id", sa.String(255), nullable=False),
            sa.Column("target_object", sa.String(50), nullable=False, server_default="company"),
            sa.Column("idempotency_key", sa.String(255), unique=True, nullable=False),
            sa.Column("status", sa.String(50), nullable=False, server_default="PENDING"),
            sa.Column("crm_note_id", sa.String(255), nullable=True),
            sa.Column("crm_note_target_id", sa.String(255), nullable=True),
            sa.Column("note_title", sa.String(255), nullable=False),
            sa.Column("summary_content", sa.Text(), nullable=False),
            sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("approved_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_crm_writeback_records_external_record_id", "crm_writeback_records", ["external_record_id"])


def downgrade() -> None:
    op.drop_table("crm_writeback_records")
    op.drop_table("crm_sync_records")
    op.drop_table("crm_sync_jobs")
