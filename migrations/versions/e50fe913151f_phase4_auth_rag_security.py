"""phase4_auth_rag_security

Revision ID: e50fe913151f
Revises: 4d91c6a0e8f2
Create Date: 2026-09-22 18:19:39.163879
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e50fe913151f'
down_revision: Union[str, Sequence[str], None] = '4d91c6a0e8f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add hashed_password to users
    op.add_column("users", sa.Column("hashed_password", sa.String(), nullable=True))
    op.execute("UPDATE users SET hashed_password = '!unusable'")
    op.alter_column("users", "hashed_password", nullable=False)

    # 2. Add columns to documents
    op.add_column("documents", sa.Column("department_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("documents", sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("documents", sa.Column("status", sa.String(), server_default="READY", nullable=True))

    # 3. Create UNCLASSIFIED department
    # Note: gen_random_uuid() is available in postgresql >= 13.
    op.execute("""
        INSERT INTO departments (id, code, name, is_active, created_at)
        VALUES (gen_random_uuid(), 'UNCLASSIFIED', 'Unclassified (Quarantine)', true, now())
        ON CONFLICT (code) DO NOTHING
    """)

    # 4. Assign existing documents to UNCLASSIFIED
    op.execute("""
        UPDATE documents
        SET department_id = (SELECT id FROM departments WHERE code = 'UNCLASSIFIED'),
            status = 'READY'
    """)

    # 5. Make columns NOT NULL and add constraints
    op.alter_column("documents", "department_id", nullable=False)
    op.alter_column("documents", "status", nullable=False)
    
    op.create_foreign_key(
        "fk_documents_department_id", 
        "documents", "departments", 
        ["department_id"], ["id"], 
        ondelete="RESTRICT"
    )
    op.create_foreign_key(
        "fk_documents_created_by", 
        "documents", "users", 
        ["created_by"], ["id"], 
        ondelete="SET NULL"
    )
    op.create_index("ix_documents_department_id", "documents", ["department_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_documents_department_id", table_name="documents")
    op.drop_constraint("fk_documents_created_by", "documents", type_="foreignkey")
    op.drop_constraint("fk_documents_department_id", "documents", type_="foreignkey")
    op.drop_column("documents", "status")
    op.drop_column("documents", "created_by")
    op.drop_column("documents", "department_id")
    
    op.drop_column("users", "hashed_password")
    
    op.execute("DELETE FROM departments WHERE code = 'UNCLASSIFIED'")
