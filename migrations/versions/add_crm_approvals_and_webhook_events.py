"""add_crm_approvals_and_webhook_events

Revision ID: add_crm_approvals_and_webhook_events
Revises: add_crm_sync_and_writeback_tables
Create Date: 2026-10-14 02:00:00.000000

Adds tables for CRM Write-Back Approvals and Webhook Event Deduplication:
- crm_writeback_approvals: Server-side trusted approval records for KB -> CRM write-back
- crm_webhook_events: Durable webhook event log for replay protection and deduplication
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "add_crm_approvals_and_webhook_events"
down_revision: Union[str, Sequence[str], None] = "add_crm_sync_and_writeback_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "crm_writeback_approvals" not in tables:
        op.create_table(
            "crm_writeback_approvals",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("operation_type", sa.String(50), nullable=False, server_default="NOTE_CREATE"),
            sa.Column("external_record_id", sa.String(255), nullable=False),
            sa.Column("target_object", sa.String(50), nullable=False, server_default="company"),
            sa.Column("department_id", UUID(as_uuid=True), sa.ForeignKey("departments.id", ondelete="CASCADE"), nullable=False),
            sa.Column("requested_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("approved_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("status", sa.String(50), nullable=False, server_default="APPROVED"),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_crm_writeback_approvals_external_record_id", "crm_writeback_approvals", ["external_record_id"])
        op.create_index("ix_crm_writeback_approvals_department_id", "crm_writeback_approvals", ["department_id"])

    if "crm_webhook_events" not in tables:
        op.create_table(
            "crm_webhook_events",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("event_id", sa.String(255), unique=True, nullable=False),
            sa.Column("event_name", sa.String(100), nullable=False),
            sa.Column("status", sa.String(50), nullable=False, server_default="RECEIVED"),
            sa.Column("payload_hash", sa.String(64), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_crm_webhook_events_event_id", "crm_webhook_events", ["event_id"])


def downgrade() -> None:
    op.drop_table("crm_webhook_events")
    op.drop_table("crm_writeback_approvals")
