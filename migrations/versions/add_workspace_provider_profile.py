"""add_workspace_provider_profile

Revision ID: add_workspace_provider_profile
Revises: add_workspace_to_conversations
Create Date: 2026-10-02 00:00:00.000000

Adds workspace_provider_profiles table.
A Workspace may have at most one provider profile (unique on workspace_id).
Secrets (API keys) are NEVER stored here; credential_ref is a plain label/reference name.
When no profile exists, the platform default provider registry configuration is used.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'add_workspace_provider_profile'
down_revision: Union[str, Sequence[str], None] = 'add_workspace_to_conversations'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workspace_provider_profiles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False, server_default="Default"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("TRUE")),

        # LLM
        sa.Column("llm_provider", sa.String(100), nullable=True),
        sa.Column("llm_model", sa.String(255), nullable=True),
        sa.Column("llm_credential_ref", sa.String(255), nullable=True),

        # Embedding
        sa.Column("embedding_provider", sa.String(100), nullable=True),
        sa.Column("embedding_model", sa.String(255), nullable=True),

        # Document reader
        sa.Column("document_reader_provider", sa.String(100), nullable=True),

        # Reranker
        sa.Column("reranker_provider", sa.String(100), nullable=True),
        sa.Column("reranker_model", sa.String(255), nullable=True),

        # OCR
        sa.Column("ocr_provider", sa.String(100), nullable=True),

        # Web
        sa.Column("web_provider", sa.String(100), nullable=True),
        sa.Column("web_credential_ref", sa.String(255), nullable=True),

        # Non-secret extra configuration
        sa.Column(
            "configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),

        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # One profile per workspace
    op.create_unique_constraint(
        "uq_workspace_provider_profile_workspace_id",
        "workspace_provider_profiles",
        ["workspace_id"],
    )

    op.create_index(
        "ix_workspace_provider_profiles_workspace_id",
        "workspace_provider_profiles",
        ["workspace_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_workspace_provider_profiles_workspace_id",
        table_name="workspace_provider_profiles",
    )
    op.drop_constraint(
        "uq_workspace_provider_profile_workspace_id",
        "workspace_provider_profiles",
        type_="unique",
    )
    op.drop_table("workspace_provider_profiles")
