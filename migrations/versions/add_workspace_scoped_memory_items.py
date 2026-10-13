"""add_workspace_scoped_memory_items

Revision ID: add_workspace_scoped_memory_items
Revises: add_workspace_provider_profile
Create Date: 2026-10-10 12:00:00.000000

Adds workspace-scoped persistent memory items table and indexes.
Handles both clean database environments (full table creation) and existing
databases where memory_items was created without workspace_id.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = 'add_workspace_scoped_memory_items'
down_revision: Union[str, Sequence[str], None] = 'add_workspace_provider_profile'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    tables = inspector.get_table_names(schema=current_schema)

    if "memory_items" not in tables:
        # Clean database path: create memory_items table with workspace scoping
        op.create_table(
            "memory_items",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                nullable=False,
                server_default=sa.text("gen_random_uuid()"),
            ),
            sa.Column(
                "user_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "workspace_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("canonical_key", sa.String(), nullable=False),
            sa.Column(
                "memory_type",
                sa.String(),
                sa.CheckConstraint(
                    "memory_type IN ('STATIC', 'DYNAMIC')",
                    name="ck_memoryitem_type",
                ),
                nullable=False,
                server_default="DYNAMIC",
            ),
            sa.Column(
                "status",
                sa.String(),
                sa.CheckConstraint(
                    "status IN ('ACTIVE', 'SUPERSEDED', 'ARCHIVED', 'EXPIRED')",
                    name="ck_memoryitem_status",
                ),
                nullable=False,
                server_default="ACTIVE",
            ),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("embedding", Vector(384), nullable=True),
            sa.Column(
                "structured_data",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
            sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "superseded_by_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("memory_items.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("source_type", sa.String(), nullable=True, server_default="CONVERSATION"),
            sa.Column("conversation_id", sa.String(), nullable=True),
            sa.Column("message_id", sa.String(), nullable=True),
            sa.Column("event_id", sa.String(), nullable=True),
            sa.Column("extraction_method", sa.String(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )

        op.create_index("ix_memory_items_user_id", "memory_items", ["user_id"])
        op.create_index("ix_memory_items_workspace_id", "memory_items", ["workspace_id"])
        op.create_index("ix_memory_items_canonical_key", "memory_items", ["canonical_key"])
        op.create_index("ix_memory_items_conversation_id", "memory_items", ["conversation_id"])
        op.create_index("ix_memory_items_superseded_by_id", "memory_items", ["superseded_by_id"])
        op.create_index(
            "ix_memory_items_user_workspace_canonical_key_active",
            "memory_items",
            ["user_id", "workspace_id", "canonical_key"],
            unique=True,
            postgresql_where=sa.text("status = 'ACTIVE'"),
        )
        op.create_index(
            "ix_memory_items_user_workspace_status",
            "memory_items",
            ["user_id", "workspace_id", "status"],
        )
        # Mark table as created by this migration for safe rollback
        op.execute(sa.text(f"COMMENT ON TABLE {current_schema}.memory_items IS 'created_by_alembic_add_workspace_scoped_memory_items'"))
    else:
        # Existing database path: table exists, check if workspace_id column is present
        columns = [c["name"] for c in inspector.get_columns("memory_items", schema=current_schema)]
        if "workspace_id" not in columns:
            # Check for existing rows that lack workspace scope
            count_res = bind.execute(sa.text("SELECT count(*) FROM memory_items")).scalar()
            if count_res and count_res > 0:
                raise RuntimeError(
                    f"Found {count_res} existing unscoped rows in memory_items. "
                    "Cannot safely assign workspace_id without manual intervention. "
                    "Aborting migration to prevent data integrity violations."
                )

            op.add_column(
                "memory_items",
                sa.Column(
                    "workspace_id",
                    postgresql.UUID(as_uuid=True),
                    sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
                    nullable=False,
                ),
            )

            # Drop old unscoped indexes if they exist
            existing_indexes = [idx["name"] for idx in inspector.get_indexes("memory_items", schema=current_schema)]
            if "ix_memory_items_user_canonical_key_active" in existing_indexes:
                op.drop_index("ix_memory_items_user_canonical_key_active", table_name="memory_items")
            if "ix_memory_items_user_status" in existing_indexes:
                op.drop_index("ix_memory_items_user_status", table_name="memory_items")

            # Create new workspace-scoped indexes
            if "ix_memory_items_workspace_id" not in existing_indexes:
                op.create_index("ix_memory_items_workspace_id", "memory_items", ["workspace_id"])
            op.create_index(
                "ix_memory_items_user_workspace_canonical_key_active",
                "memory_items",
                ["user_id", "workspace_id", "canonical_key"],
                unique=True,
                postgresql_where=sa.text("status = 'ACTIVE'"),
            )
            op.create_index(
                "ix_memory_items_user_workspace_status",
                "memory_items",
                ["user_id", "workspace_id", "status"],
            )

            # Mark table as altered (pre-existing) by this migration for non-destructive rollback
            op.execute(sa.text(f"COMMENT ON TABLE {current_schema}.memory_items IS 'altered_by_alembic_add_workspace_scoped_memory_items'"))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    current_schema = op.get_context().version_table_schema or bind.execute(sa.text("SELECT current_schema()")).scalar()
    tables = inspector.get_table_names(schema=current_schema)

    if "memory_items" in tables:
        comment_info = inspector.get_table_comment("memory_items", schema=current_schema)
        table_comment = (comment_info.get("text") or "") if comment_info else ""

        if table_comment == "created_by_alembic_add_workspace_scoped_memory_items":
            # Clean database path: Table was created from scratch by this migration, safe to drop
            op.drop_table("memory_items")
        else:
            # Pre-existing database path: Preserve existing table and all records!
            # Only remove workspace_id and restore pre-existing indexes non-destructively.
            existing_indexes = [idx["name"] for idx in inspector.get_indexes("memory_items", schema=current_schema)]
            if "ix_memory_items_user_workspace_canonical_key_active" in existing_indexes:
                op.drop_index("ix_memory_items_user_workspace_canonical_key_active", table_name="memory_items")
            if "ix_memory_items_user_workspace_status" in existing_indexes:
                op.drop_index("ix_memory_items_user_workspace_status", table_name="memory_items")
            if "ix_memory_items_workspace_id" in existing_indexes:
                op.drop_index("ix_memory_items_workspace_id", table_name="memory_items")

            columns = [c["name"] for c in inspector.get_columns("memory_items", schema=current_schema)]
            if "workspace_id" in columns:
                op.drop_column("memory_items", "workspace_id")

            # Re-create legacy un-scoped indexes
            op.create_index(
                "ix_memory_items_user_canonical_key_active",
                "memory_items",
                ["user_id", "canonical_key"],
                unique=True,
                postgresql_where=sa.text("status = 'ACTIVE'"),
            )
            op.create_index(
                "ix_memory_items_user_status",
                "memory_items",
                ["user_id", "status"],
            )
            op.execute(sa.text(f"COMMENT ON TABLE {current_schema}.memory_items IS NULL"))
