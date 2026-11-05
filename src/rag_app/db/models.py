import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID,JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates

from rag_app.core.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True,
        default=uuid.uuid4
        )
    file_name: Mapped[str] = mapped_column(
        String(255), 
        nullable=False
        )
    file_path: Mapped[str] = mapped_column(
        Text, 
        nullable=False
        )
    content_hash: Mapped[str] = mapped_column(
        String(64), 
        nullable=False, 
        unique=True
        )
    mime_type: Mapped[str] = mapped_column(
        String(80), 
        nullable=False, 
        default="text/plain"
        )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=True,
        index=True
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    status: Mapped[str] = mapped_column(
        String,
        default="READY",
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
        )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    sections: Mapped[list["DocumentSection"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        foreign_keys="[DocumentSection.document_id]",
    )
    workspace: Mapped["Workspace | None"] = relationship(
        back_populates="documents",
        foreign_keys="[Document.workspace_id]",
    )


class DocumentSection(Base):
    """
    OV-1: Hierarchical resource node.

    Represents a structural unit of a Document — the document root, a heading
    section, a sub-section, or a figure grouping.  Nodes form a tree via the
    nullable `parent_id` self-FK.  Each node carries:
    - L0: a short deterministic abstract (≤ settings.node_l0_max_chars chars)
    - L1: an overview including child section titles (≤ settings.node_l1_max_chars chars)

    DocumentChunks that belong to a section reference it via the nullable `node_id`
    FK on DocumentChunk.  Chunks without a node_id are still fully functional —
    the hierarchy is additive, not a replacement.

    Authorization is inherited from Document.department_id; no separate auth table.
    """
    __tablename__ = "document_sections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    node_type: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint(
            "node_type IN ('DOCUMENT', 'SECTION', 'SUBSECTION', 'FIGURE')",
            name="ck_documentsection_node_type",
        ),
        nullable=False,
        default="SECTION",
    )
    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )
    level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="0=document root, 1=H1, 2=H2, 3=H3, …",
    )
    ordinal_path: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="",
        comment="Dot-separated ordinal e.g. '1', '1.2', '1.2.3' for deterministic ordering",
    )
    l0_abstract: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Short deterministic abstract derived from section content/title",
    )
    l1_overview: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Section overview: title + child section navigation",
    )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        back_populates="sections",
        foreign_keys=[document_id],
    )
    parent: Mapped["DocumentSection | None"] = relationship(
        back_populates="children",
        remote_side=[id],
        foreign_keys=[parent_id],
    )
    children: Mapped[list["DocumentSection"]] = relationship(
        back_populates="parent",
        foreign_keys=[parent_id],
        cascade="all, delete-orphan",
        order_by="DocumentSection.ordinal_path",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="node",
        foreign_keys="[DocumentChunk.node_id]",
    )

    __table_args__ = (
        Index(
            "ix_document_sections_doc_ordinal",
            "document_id",
            "ordinal_path",
        ),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True, 
        default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False
        )

    text: Mapped[str] = mapped_column(
        Text, 
        nullable=False
    )
    
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dim),
        nullable=False
        )
    page_number: Mapped[int | None] = mapped_column(
        Integer, 
        nullable=True
        )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    # OV-1: nullable FK to the structural node this chunk belongs to.
    # NULL means the chunk predates hierarchy or belongs to a document
    # without detectable heading structure — it is still fully functional.
    node_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    node: Mapped["DocumentSection | None"] = relationship(
        back_populates="chunks",
        foreign_keys=[node_id],
    )
    chunk_hash: Mapped[str] = mapped_column(
        String, 
        index=True
    ) # index=True zaruri hai fast search ke liye
    current_version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunk_index"
        ),
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True,
        index=True
    )
    hashed_password: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    role: Mapped[str] = mapped_column(
        String,
        CheckConstraint("role IN ('ADMIN', 'USER')", name='ck_user_role'),
        nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    registration_status: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("registration_status IN ('PENDING', 'APPROVED', 'REJECTED')", name="ck_user_registration_status"),
        default="APPROVED",
        nullable=False,
    )
    profile_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    __table_args__ = (
        Index(
            "uq_users_email_normalized",
            func.lower(func.btrim(email)),
            unique=True,
        ),
    )

    @validates("email")
    def normalize_email(self, _key: str, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("email must not be empty")
        return normalized


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    code: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )


class Permission(Base):
    __tablename__ = "permissions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    code: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True
    )
    description: Mapped[str | None] = mapped_column(
        String,
        nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class UserDepartment(Base):
    __tablename__ = "user_departments"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )
    department_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        primary_key=True,
        index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class UserPermission(Base):
    __tablename__ = "user_permissions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True
    )
    department_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True
    )
    permission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="RESTRICT"),
        primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["user_id", "department_id"],
            ["user_departments.user_id", "user_departments.department_id"],
            ondelete="CASCADE",
        ),
    )


class DocumentPermission(Base):
    """
    Minimal additive authorization layer for document-specific permissions.
    Grants VIEW, UPLOAD, EDIT, or DELETE on a specific document to a user.
    Enforces uniqueness on (document_id, user_id, permission_id).
    """
    __tablename__ = "document_permissions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )
    permission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="RESTRICT"),
        primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class Workspace(Base):
    """
    First-class Workspace entity.
    Belongs to exactly one department.
    Serves as an additional knowledge boundary scoped within department authorization.
    """
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    department_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    department: Mapped["Department"] = relationship(foreign_keys=[department_id])
    creator: Mapped["User | None"] = relationship(foreign_keys=[created_by])
    documents: Mapped[list["Document"]] = relationship(
        back_populates="workspace",
        foreign_keys="[Document.workspace_id]",
    )
    members: Mapped[list["WorkspaceMember"]] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    provider_profile: Mapped["WorkspaceProviderProfile | None"] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
        uselist=False,
    )
    mcp_connections: Mapped[list["WorkspaceMCPConnection"]] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "department_id",
            "code",
            name="uq_workspaces_department_code",
        ),
        Index(
            "ix_workspaces_dept_active",
            "department_id",
            "is_active",
        ),
    )


class WorkspaceMember(Base):
    """
    Explicit Workspace Membership and Role.
    Roles: OWNER | EDITOR | VIEWER
    """
    __tablename__ = "workspace_members"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("role IN ('OWNER', 'EDITOR', 'VIEWER')", name="ck_workspace_member_role"),
        default="VIEWER",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="members")
    user: Mapped["User"] = relationship()

    __table_args__ = (
        Index("ix_workspace_members_user_id", "user_id"),
    )


class WorkspaceProviderProfile(Base):
    """
    Workspace-level provider configuration profile.
    A Workspace may have at most one active profile.
    Stores provider name + model + non-secret configuration.
    Secrets (API keys) are NEVER stored here; credential_ref is just a label.
    When no profile exists, the platform-level defaults from the provider registry are used.
    """
    __tablename__ = "workspace_provider_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="Default",
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )

    # LLM provider selection
    llm_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    llm_credential_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Embedding provider selection
    embedding_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Document reader provider selection
    document_reader_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Reranker provider selection
    reranker_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    reranker_model: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # OCR provider selection
    ocr_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Web provider selection
    web_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    web_credential_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Provider-specific non-secret extra configuration
    configuration: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="provider_profile")

    __table_args__ = (
        # Enforce one profile per workspace
        UniqueConstraint(
            "workspace_id",
            name="uq_workspace_provider_profile_workspace_id",
        ),
        Index("ix_workspace_provider_profiles_workspace_id", "workspace_id"),
    )


class WorkspaceMCPConnection(Base):
    """
    Workspace-scoped External MCP Connection.
    Enforces strict multi-tenant boundary, server-managed credential reference,
    and default-deny tool allowlist.
    """
    __tablename__ = "workspace_mcp_connections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    endpoint_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    transport: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="streamable-http",
    )
    credential_ref: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="DISCONNECTED",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    last_connected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    server_info: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
    )
    discovered_tools: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        server_default="[]",
    )
    enabled_tools: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        server_default="[]",
    )
    configuration: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="mcp_connections")
    creator: Mapped["User | None"] = relationship(foreign_keys=[created_by])

    @property
    def credential_mode(self) -> str:
        """Explicit credential mode: 'default', 'custom', or 'env_ref'."""
        if not self.configuration:
            return "env_ref" if self.credential_ref else "default"
        return self.configuration.get(
            "credential_mode",
            "env_ref" if self.credential_ref else "default",
        )

    @credential_mode.setter
    def credential_mode(self, value: str) -> None:
        cfg = dict(self.configuration or {})
        cfg["credential_mode"] = str(value).lower()
        self.configuration = cfg

    @property
    def encrypted_credential(self) -> str | None:
        """Ciphertext token of the encrypted custom credential, if configured."""
        if not self.configuration:
            return None
        return self.configuration.get("encrypted_credential")

    @encrypted_credential.setter
    def encrypted_credential(self, value: str | None) -> None:
        cfg = dict(self.configuration or {})
        if value is None:
            cfg.pop("encrypted_credential", None)
        else:
            cfg["encrypted_credential"] = str(value)
        self.configuration = cfg

    @property
    def credential_key_version(self) -> str | None:
        """Version tag of the key used to encrypt the credential."""
        if not self.configuration:
            return None
        return self.configuration.get("credential_key_version")

    @credential_key_version.setter
    def credential_key_version(self, value: str | None) -> None:
        cfg = dict(self.configuration or {})
        if value is None:
            cfg.pop("credential_key_version", None)
        else:
            cfg["credential_key_version"] = str(value)
        self.configuration = cfg

    @property
    def has_custom_credential(self) -> bool:
        """True if an encrypted custom credential is saved on this connection."""
        return bool(self.encrypted_credential)


    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_workspace_mcp_connections_ws_name"),
        CheckConstraint("status IN ('DISCONNECTED', 'CONNECTED', 'ERROR', 'INITIALIZING')", name="ck_workspace_mcp_connections_status"),
        CheckConstraint("transport IN ('streamable-http', 'sse')", name="ck_workspace_mcp_connections_transport"),
        Index("ix_workspace_mcp_connections_ws_enabled", "workspace_id", "is_enabled"),
    )


class MCPToolOperationRecord(Base):
    """
    Durable operation record for external MCP tool invocations (Milestone 6C.2).
    Tracks external write lifecycle, idempotency key correlation, bounded reconciliation,
    and prevents ambiguous timeouts from being treated as failures or blindly retried.
    """
    __tablename__ = "mcp_tool_operation_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    connection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspace_mcp_connections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tool_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PENDING",
    )
    arguments: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default="{}",
    )
    result_payload: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    reconciliation_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    last_reconciled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    approval_ticket_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_tasks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    workspace: Mapped["Workspace"] = relationship()
    connection: Mapped["WorkspaceMCPConnection"] = relationship()

    __table_args__ = (
        UniqueConstraint("workspace_id", "idempotency_key", name="uq_mcp_tool_op_ws_idempotency_key"),
        CheckConstraint(
            "status IN ('PENDING', 'IN_FLIGHT', 'CONFIRMED', 'FAILED', 'UNKNOWN', 'RECONCILIATION_REQUIRED')",
            name="ck_mcp_tool_operation_records_status",
        ),
        Index("ix_mcp_tool_operation_records_ws_status", "workspace_id", "status"),
        Index("ix_mcp_tool_operation_records_conn_tool", "connection_id", "tool_name"),
    )





class AccessRequest(Base):
    """
    USER-submitted request for department access.
    PENDING status grants zero authorization.
    Approval/rejection does not automatically grant permissions;
    it tracks the lifecycle of access requests.
    """
    __tablename__ = "access_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    department_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String,
        CheckConstraint(
            "status IN ('PENDING', 'APPROVED', 'REJECTED', 'CANCELLED')",
            name="ck_accessrequest_status",
        ),
        default="PENDING",
        nullable=False,
        index=True,
    )
    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    review_note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    department: Mapped["Department"] = relationship("Department", foreign_keys=[department_id])
    reviewer: Mapped["User | None"] = relationship("User", foreign_keys=[reviewed_by])
    requested_permissions: Mapped[list["AccessRequestPermission"]] = relationship(
        "AccessRequestPermission",
        cascade="all, delete-orphan",
        back_populates="access_request",
    )

    __table_args__ = (
        Index(
            "uq_access_requests_user_dept_pending",
            "user_id",
            "department_id",
            unique=True,
            postgresql_where=text("status = 'PENDING'"),
        ),
    )


class AccessRequestPermission(Base):
    """
    Normalized child table storing requested permissions for an AccessRequest.
    Composite PK (access_request_id, permission_id).
    """
    __tablename__ = "access_request_permissions"

    access_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("access_requests.id", ondelete="CASCADE"),
        primary_key=True,
    )
    permission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="RESTRICT"),
        primary_key=True,
    )

    access_request: Mapped["AccessRequest"] = relationship(
        "AccessRequest",
        back_populates="requested_permissions",
    )
    permission: Mapped["Permission"] = relationship("Permission")



class ResearchRun(Base):
    __tablename__ = "research_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    status: Mapped[str] = mapped_column(
        String,
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')",
            name="ck_researchrun_status"
        ),
        default="PENDING",
        nullable=False
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=True,
        index=True
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

class ResearchKnowledge(Base):
    __tablename__ = "research_knowledge"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    research_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        index=True
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=True,
        index=True
    )
    entity_name: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    entity_type: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    attributes: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    evidence: Mapped[list["ResearchEvidence"]] = relationship(
        back_populates="knowledge",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "research_run_id",
            "entity_name",
            "entity_type",
            name="uq_research_knowledge_entity"
        ),
    )


class ResearchEvidence(Base):
    __tablename__ = "research_evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    knowledge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("research_knowledge.id", ondelete="CASCADE"),
        index=True
    )
    source_url: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    source_title: Mapped[str | None] = mapped_column(
        String,
        nullable=True
    )
    evidence_text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    knowledge: Mapped[ResearchKnowledge] = relationship(
        back_populates="evidence"
    )

class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    canonical_key: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    kind: Mapped[str] = mapped_column(
        String,
        CheckConstraint(
            "kind IN ('ENTITY', 'FACT', 'EVENT', 'RELATIONSHIP')",
            name="ck_knowledgeitem_kind"
        ),
        nullable=False
    )
    status: Mapped[str] = mapped_column(
        String,
        CheckConstraint(
            "status IN ('ACTIVE', 'SUPERSEDED', 'ARCHIVED')",
            name="ck_knowledgeitem_status"
        ),
        default="ACTIVE",
        nullable=False
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=True,
        index=True
    )
    confidence: Mapped[float | None] = mapped_column(
        nullable=True
    )
    visibility: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint(
            "visibility IN ('PRIVATE', 'WORKSPACE', 'SHARED')",
            name="ck_knowledgeitem_visibility",
        ),
        default="WORKSPACE",
        server_default="WORKSPACE",
        nullable=False,
    )
    approval_status: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint(
            "approval_status IN ('DRAFT', 'PENDING', 'APPROVED', 'REJECTED')",
            name="ck_knowledgeitem_approval_status",
        ),
        default="DRAFT",
        server_default="DRAFT",
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint(
            "source_type IN ('DOCUMENT', 'RESEARCH', 'CONVERSATION')",
            name="ck_knowledgeitem_source_type",
        ),
        default="DOCUMENT",
        server_default="DOCUMENT",
        nullable=False,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    author_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    approved_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    workspace: Mapped["Workspace | None"] = relationship(foreign_keys=[workspace_id])
    department: Mapped["Department | None"] = relationship(foreign_keys=[department_id])
    author: Mapped["User | None"] = relationship(foreign_keys=[author_id])
    approved_by: Mapped["User | None"] = relationship(foreign_keys=[approved_by_id])

    entities: Mapped[list["KnowledgeEntity"]] = relationship(
        back_populates="knowledge_item",
        cascade="all, delete-orphan",
        foreign_keys="[KnowledgeEntity.knowledge_item_id]"
    )
    relationships_as_subject: Mapped[list["KnowledgeRelationship"]] = relationship(
        back_populates="knowledge_item",
        cascade="all, delete-orphan",
        foreign_keys="[KnowledgeRelationship.knowledge_item_id]"
    )
    evidence_links: Mapped[list["KnowledgeEvidence"]] = relationship(
        back_populates="knowledge_item",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_knowledge_items_canonical_key_dept",
            "canonical_key",
            "department_id",
            unique=True,
            postgresql_nulls_not_distinct=True,
        ),
        Index(
            "ix_knowledge_items_visibility_approval",
            "visibility",
            "approval_status",
        ),
        CheckConstraint(
            "(approval_status != 'APPROVED') OR (approved_by_id IS NOT NULL AND approved_at IS NOT NULL)",
            name="ck_knowledgeitem_approval_consistency",
        ),
        CheckConstraint(
            "(visibility != 'SHARED') OR (approval_status = 'APPROVED')",
            name="ck_knowledgeitem_shared_requires_approved",
        ),
    )

class KnowledgeEntity(Base):
    __tablename__ = "knowledge_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    knowledge_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    entity_type: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    canonical_name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True
    )
    attributes: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default='{}'
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    knowledge_item: Mapped[KnowledgeItem] = relationship(
        back_populates="entities",
        foreign_keys=[knowledge_item_id]
    )

    __table_args__ = (
        UniqueConstraint(
            "knowledge_item_id",
            name="uq_knowledge_entity_item"
        ),
    )

class KnowledgeRelationship(Base):
    __tablename__ = "knowledge_relationships"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    knowledge_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    subject_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    predicate: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    object_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    knowledge_item: Mapped[KnowledgeItem] = relationship(
        back_populates="relationships_as_subject",
        foreign_keys=[knowledge_item_id]
    )
    subject_entity: Mapped[KnowledgeEntity] = relationship(
        foreign_keys=[subject_entity_id]
    )
    object_entity: Mapped[KnowledgeEntity] = relationship(
        foreign_keys=[object_entity_id]
    )

    __table_args__ = (
        UniqueConstraint(
            "knowledge_item_id",
            name="uq_knowledge_relationship_item"
        ),
    )

class KnowledgeEvidence(Base):
    __tablename__ = "knowledge_evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    knowledge_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    document_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    research_evidence_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("research_evidence.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    chat_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    extraction_method: Mapped[str | None] = mapped_column(
        String,
        nullable=True
    )
    extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("research_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    # Website source fields (non-destructive extension)
    source_url: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        index=True,
    )
    source_title: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )
    user_excerpt: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    verification_status: Mapped[str | None] = mapped_column(
        String(30),
        CheckConstraint(
            "verification_status IS NULL OR verification_status IN ('NOT_VERIFIED', 'VERIFIED', 'FAILED')",
            name="ck_knowledgeevidence_verification_status",
        ),
        default="NOT_VERIFIED",
        server_default="NOT_VERIFIED",
        nullable=True,
    )
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    content_hash: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )
    confidence: Mapped[float | None] = mapped_column(
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    knowledge_item: Mapped[KnowledgeItem] = relationship(
        back_populates="evidence_links"
    )
    document_chunk: Mapped["DocumentChunk"] = relationship()
    research_evidence: Mapped["ResearchEvidence"] = relationship()
    conversation: Mapped["Conversation | None"] = relationship(foreign_keys=[conversation_id])
    chat_message: Mapped["ChatMessage | None"] = relationship(foreign_keys=[chat_message_id])

    __table_args__ = (
        CheckConstraint(
            "(document_chunk_id IS NOT NULL AND research_evidence_id IS NULL AND conversation_id IS NULL AND chat_message_id IS NULL AND source_url IS NULL) OR "
            "(document_chunk_id IS NULL AND research_evidence_id IS NOT NULL AND conversation_id IS NULL AND chat_message_id IS NULL AND source_url IS NULL) OR "
            "(document_chunk_id IS NULL AND research_evidence_id IS NULL AND conversation_id IS NOT NULL AND source_url IS NULL) OR "
            "(document_chunk_id IS NULL AND research_evidence_id IS NULL AND conversation_id IS NULL AND chat_message_id IS NULL AND source_url IS NOT NULL)",
            name="ck_knowledgeevidence_single_source",
        ),
        UniqueConstraint(
            "knowledge_item_id",
            "document_chunk_id",
            "research_evidence_id",
            "conversation_id",
            "chat_message_id",
            "source_url",
            name="uq_knowledge_evidence_source",
            postgresql_nulls_not_distinct=True,
        ),
    )


class MemoryProcessingJob(Base):
    """
    Durable memory evaluation job / outbox record.
    Captures user turn snapshots before structured interpretation and supports
    deterministic candidate execution, LLM fallbacks, and observable retries.
    """
    __tablename__ = "memory_processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    owner_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_text_snapshot: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    trigger_kind: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="rule_candidate",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint(
            "status IN ('PENDING', 'PROCESSING', 'SUCCEEDED', 'RETRY_PENDING', 'DEAD_LETTER', 'SKIPPED_NOT_MEMORABLE')",
            name="ck_memory_processing_job_status",
        ),
        default="PENDING",
        nullable=False,
        index=True,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    last_error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    locked_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    owner: Mapped["User"] = relationship(foreign_keys=[owner_user_id])
    workspace: Mapped["Workspace | None"] = relationship(foreign_keys=[workspace_id])

    __table_args__ = (
        Index("ix_memory_processing_jobs_status_created", "status", "created_at"),
        Index("ix_memory_processing_jobs_user_status", "owner_user_id", "status"),
        Index("ix_memory_processing_jobs_next_attempt", "status", "next_attempt_at"),
        Index(
            "ix_memory_processing_jobs_source_trigger_unique",
            "source_message_id",
            "trigger_kind",
            unique=True,
            postgresql_where=text("source_message_id IS NOT NULL"),
        ),
    )



class MemoryItem(Base):
    __tablename__ = "memory_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="RESTRICT", name="fk_memory_items_workspace_id_workspaces"),
        nullable=True,
        index=True,
    )
    canonical_key: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    memory_type: Mapped[str] = mapped_column(
        String,
        CheckConstraint(
            "memory_type IN ('STATIC', 'DYNAMIC')",
            name="ck_memoryitem_type",
        ),
        default="DYNAMIC",
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String,
        CheckConstraint(
            "status IN ('ACTIVE', 'SUPERSEDED', 'ARCHIVED', 'EXPIRED')",
            name="ck_memoryitem_status",
        ),
        default="ACTIVE",
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dim),
        nullable=True,
    )
    structured_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
    )
    confidence: Mapped[float | None] = mapped_column(
        nullable=True,
    )
    valid_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    retention_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    source_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memory_processing_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memory_items.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_type: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        default="CONVERSATION",
    )
    conversation_id: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
        index=True,
    )
    message_id: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )
    event_id: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )
    extraction_method: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    workspace: Mapped["Workspace | None"] = relationship(foreign_keys=[workspace_id])
    superseded_by: Mapped["MemoryItem | None"] = relationship(
        remote_side=[id],
        foreign_keys=[superseded_by_id],
    )
    source_job: Mapped["MemoryProcessingJob | None"] = relationship(foreign_keys=[source_job_id])

    __table_args__ = (
        Index(
            "ix_memory_items_user_workspace_canonical_key_active",
            "user_id",
            "workspace_id",
            "canonical_key",
            unique=True,
            postgresql_where=text("status = 'ACTIVE' AND workspace_id IS NOT NULL"),
        ),
        Index(
            "ix_memory_items_user_global_canonical_key_active",
            "user_id",
            "canonical_key",
            unique=True,
            postgresql_where=text("status = 'ACTIVE' AND workspace_id IS NULL"),
        ),
        Index(
            "ix_memory_items_user_workspace_status",
            "user_id",
            "workspace_id",
            "status",
        ),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Chat",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    workspace: Mapped["Workspace | None"] = relationship(foreign_keys=[workspace_id])
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at.asc()",
    )

    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
        Index("ix_conversations_workspace_updated", "workspace_id", "updated_at"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="ck_chatmessage_role",
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    sources: Mapped[list[dict] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    visualization: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    execution_metrics: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages"
    )

    __table_args__ = (
        Index("ix_chat_messages_conv_created", "conversation_id", "created_at"),
    )


class TaskItem(Base):
    """
    Structured Follow-up Tasks, Commitments, and Events.
    Maintains strict separation from personal memory (MemoryItem) and external CRM state.
    Carries explicit owner, due date, timezone policy, status, company/contact associations,
    and provenance tracking.
    """
    __tablename__ = "task_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint(
            "status IN ('OPEN', 'COMPLETED', 'CANCELLED')",
            name="ck_task_status",
        ),
        default="OPEN",
        nullable=False,
        index=True,
    )
    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    timezone: Mapped[str] = mapped_column(
        String(64),
        default="UTC",
        nullable=False,
    )
    company_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    contact_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    provenance_type: Mapped[str] = mapped_column(
        String(32),
        CheckConstraint(
            "provenance_type IN ('EXPLICIT_USER_ACTION', 'EXTRACTED_FROM_INTERACTION', 'EXTRACTED_FROM_DOCUMENT')",
            name="ck_task_provenance_type",
        ),
        default="EXPLICIT_USER_ACTION",
        nullable=False,
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    attributes: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    workspace: Mapped["Workspace | None"] = relationship(foreign_keys=[workspace_id])
    department: Mapped["Department | None"] = relationship(foreign_keys=[department_id])

    __table_args__ = (
        Index("ix_task_items_user_status_due", "user_id", "status", "due_date"),
        Index("ix_task_items_user_company", "user_id", "company_name"),
    )


class KnowledgeExtractionJob(Base):
    """
    Durable PostgreSQL-backed queue record for document knowledge extraction.
    Ensures background jobs survive application server restarts, support safe concurrent
    claiming via FOR UPDATE SKIP LOCKED, automatic abandoned-job recovery, and retries.
    """
    __tablename__ = "knowledge_extraction_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')",
            name="ck_extraction_job_status",
        ),
        default="PENDING",
        nullable=False,
        index=True,
    )
    attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    locked_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    document: Mapped["Document"] = relationship(foreign_keys=[document_id])

    __table_args__ = (
        Index("ix_extraction_jobs_status_created", "status", "created_at"),
        Index("ix_extraction_jobs_locked_at", "locked_at"),
    )


class ConnectionCandidate(Base):
    __tablename__ = "connection_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source_knowledge_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_items.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    source_memory_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memory_items.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    target_knowledge_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_items.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    target_memory_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memory_items.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    connection_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )
    predicate: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="associated_with",
    )
    lifecycle_state: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="SUGGESTED",
        index=True,
    )
    is_directed: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
    )
    ranking_score: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
    )
    source_version_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    target_version_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    transaction_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    transaction_superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    superseded_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("connection_candidates.id", ondelete="SET NULL"),
        nullable=True,
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    author_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    source_knowledge_item: Mapped["KnowledgeItem | None"] = relationship(
        foreign_keys=[source_knowledge_item_id]
    )
    source_memory_item: Mapped["MemoryItem | None"] = relationship(
        foreign_keys=[source_memory_item_id]
    )
    target_knowledge_item: Mapped["KnowledgeItem | None"] = relationship(
        foreign_keys=[target_knowledge_item_id]
    )
    target_memory_item: Mapped["MemoryItem | None"] = relationship(
        foreign_keys=[target_memory_item_id]
    )
    reviews: Mapped[list["ConnectionCandidateReview"]] = relationship(
        back_populates="candidate",
        cascade="all, delete-orphan",
    )
    evidence_links: Mapped[list["ConnectionCandidateEvidence"]] = relationship(
        back_populates="candidate",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint(
            "(source_knowledge_item_id IS NOT NULL AND source_memory_item_id IS NULL) OR "
            "(source_knowledge_item_id IS NULL AND source_memory_item_id IS NOT NULL)",
            name="chk_candidate_source",
        ),
        CheckConstraint(
            "(target_knowledge_item_id IS NOT NULL AND target_memory_item_id IS NULL) OR "
            "(target_knowledge_item_id IS NULL AND target_memory_item_id IS NOT NULL)",
            name="chk_candidate_target",
        ),
        Index("ix_connection_candidates_lookup", "source_knowledge_item_id", "target_knowledge_item_id", "predicate"),
    )


class ConnectionCandidateReview(Base):
    __tablename__ = "connection_candidate_reviews"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("connection_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reviewer_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    decision: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )
    rationale: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    source_version_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    target_version_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    policy_version: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="v1.0",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    candidate: Mapped["ConnectionCandidate"] = relationship(
        back_populates="reviews",
        foreign_keys=[candidate_id],
    )
    reviewer: Mapped["User"] = relationship(
        foreign_keys=[reviewer_user_id],
    )


class ConnectionCandidateEvidence(Base):
    __tablename__ = "connection_candidate_evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("connection_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=True,
    )
    chat_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=True,
    )
    research_evidence_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("research_evidence.id", ondelete="CASCADE"),
        nullable=True,
    )
    claim_excerpt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    verification_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="VERIFIED",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    candidate: Mapped["ConnectionCandidate"] = relationship(
        back_populates="evidence_links",
        foreign_keys=[candidate_id],
    )


class ConnectionDiscoveryJob(Base):
    __tablename__ = "connection_discovery_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    source_item_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    target_source_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    requesting_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="PENDING",
        nullable=False,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    locked_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    requesting_user: Mapped["User"] = relationship(foreign_keys=[requesting_user_id])

    __table_args__ = (
        Index("ix_connection_jobs_status_created", "status", "created_at"),
        Index("ix_connection_jobs_locked_at", "locked_at"),
    )


class CRMSyncJob(Base):
    __tablename__ = "crm_sync_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_type: Mapped[str] = mapped_column(
        String(50),
        default="BATCH_SYNC",
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="PENDING",
        nullable=False,
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    checkpoint_cursor: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    total_records: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    processed_records: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    succeeded_records: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    skipped_records: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    failed_records: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class CRMSyncRecord(Base):
    __tablename__ = "crm_sync_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("crm_sync_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider: Mapped[str] = mapped_column(
        String(50),
        default="twenty_crm",
        nullable=False,
    )
    entity_type: Mapped[str] = mapped_column(
        String(50),
        default="company",
        nullable=False,
    )
    external_record_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="PENDING",
        nullable=False,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "entity_type",
            "external_record_id",
            name="uq_crm_sync_records_provider_entity_external",
        ),
        Index("ix_crm_sync_records_status_retry", "status", "next_retry_at"),
    )


class CRMWriteBackRecord(Base):
    __tablename__ = "crm_writeback_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    operation_type: Mapped[str] = mapped_column(
        String(50),
        default="NOTE_CREATE",
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(
        String(50),
        default="twenty_crm",
        nullable=False,
    )
    external_record_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    target_object: Mapped[str] = mapped_column(
        String(50),
        default="company",
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="PENDING",
        nullable=False,
    )
    crm_note_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    crm_note_target_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    note_title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    summary_content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class CRMWriteBackApproval(Base):
    __tablename__ = "crm_writeback_approvals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    operation_type: Mapped[str] = mapped_column(
        String(50),
        default="NOTE_CREATE",
        nullable=False,
    )
    external_record_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    target_object: Mapped[str] = mapped_column(
        String(50),
        default="company",
        nullable=False,
    )
    department_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    approved_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="APPROVED",
        nullable=False,
    )  # APPROVED, CONSUMED, REVOKED, EXPIRED
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    consumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    workflow_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    approved_draft_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    intended_action: Mapped[str] = mapped_column(
        String(50),
        default="CRM_CREATE_NOTE",
        server_default="CRM_CREATE_NOTE",
        nullable=False,
    )

    workflow_run: Mapped["WorkflowRun | None"] = relationship(
        back_populates="approvals",
        foreign_keys="[CRMWriteBackApproval.workflow_run_id]",
    )


class CRMWebhookEvent(Base):
    __tablename__ = "crm_webhook_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    event_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    event_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="RECEIVED",
        nullable=False,
    )
    payload_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class GeneratedArtifact(Base):
    __tablename__ = "generated_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    format: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    safe_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    byte_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    storage_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    checksum_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    source_departments: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
    )
    authorized_roles: Mapped[list[str] | None] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
    )
    tabular_data: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    coverage_metadata: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    download_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    last_downloaded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    workflow_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    workflow_run: Mapped["WorkflowRun | None"] = relationship(
        back_populates="artifacts",
        foreign_keys="[GeneratedArtifact.workflow_run_id]",
    )

    __table_args__ = (
        Index("ix_artifacts_user_created", "user_id", "created_at"),
        Index("ix_artifacts_expires", "expires_at"),
    )


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    scope_type: Mapped[str] = mapped_column(
        String(20),
        default="WORKSPACE",
        nullable=False,
    )
    thread_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="PENDING",
        nullable=False,
        index=True,
    )
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    request_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    draft_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    initial_query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    workflow_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_definitions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    workflow_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_versions.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    provider_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Budget limits & real-time accounting
    max_tokens_budget: Mapped[int] = mapped_column(Integer, nullable=False, default=25000)
    tokens_spent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_reserved: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_provider_calls_budget: Mapped[int] = mapped_column(Integer, nullable=False, default=15)
    provider_calls_spent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    provider_calls_reserved: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    wall_clock_timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=180)

    # Error classification
    failure_category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Auditable timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    awaiting_approval_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    executing_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    tasks: Mapped[list["WorkflowTask"]] = relationship(
        back_populates="workflow_run",
        cascade="all, delete-orphan",
        order_by="WorkflowTask.created_at.asc()",
    )
    artifacts: Mapped[list["GeneratedArtifact"]] = relationship(
        back_populates="workflow_run",
    )
    workflow: Mapped["WorkflowDefinition | None"] = relationship(
        foreign_keys="[WorkflowRun.workflow_id]",
    )
    workflow_version: Mapped["WorkflowVersion | None"] = relationship(
        foreign_keys="[WorkflowRun.workflow_version_id]",
    )
    approvals: Mapped[list["CRMWriteBackApproval"]] = relationship(
        back_populates="workflow_run",
    )
    @validates("provider_snapshot")
    def validate_provider_snapshot(self, key, value):
        if getattr(self, "provider_snapshot", None) is not None and value != self.provider_snapshot:
            raise ValueError("ImmutableRuntimeSnapshotError: Runtime provider snapshot cannot be mutated once initialized.")
        return value

    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'PLANNING', 'RUNNING', 'AWAITING_APPROVAL', 'EXECUTING', 'COMPLETED', 'REJECTED', 'EXPIRED', 'CANCELLED', 'FAILED')",
            name="ck_workflow_runs_status",
        ),
        CheckConstraint(
            "scope_type IN ('GLOBAL', 'WORKSPACE')",
            name="ck_workflow_runs_scope_type",
        ),
        CheckConstraint(
            "(scope_type = 'GLOBAL' AND workspace_id IS NULL) OR (scope_type = 'WORKSPACE' AND workspace_id IS NOT NULL)",
            name="ck_workflow_runs_scope_workspace_consistency",
        ),
        CheckConstraint("tokens_spent >= 0", name="ck_workflow_runs_tokens_spent_nonneg"),
        CheckConstraint("tokens_reserved >= 0", name="ck_workflow_runs_tokens_reserved_nonneg"),
        CheckConstraint("provider_calls_spent >= 0", name="ck_workflow_runs_calls_spent_nonneg"),
        CheckConstraint("provider_calls_reserved >= 0", name="ck_workflow_runs_calls_reserved_nonneg"),
        Index("ix_workflow_runs_status_created", "status", "created_at"),
        Index(
            "uq_workflow_runs_user_ws_idempotency",
            "user_id",
            "workspace_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL AND workspace_id IS NOT NULL"),
        ),
        Index(
            "uq_workflow_runs_user_global_idempotency",
            "user_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL AND workspace_id IS NULL"),
        ),
    )


class WorkflowTask(Base):
    __tablename__ = "workflow_tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workflow_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    task_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="PENDING",
        nullable=False,
        index=True,
    )
    dependencies: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    input_payload: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    output_result: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    tokens_spent: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    provider_calls: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    max_retries: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    locked_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    fencing_token: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    workflow_run: Mapped["WorkflowRun"] = relationship(
        back_populates="tasks",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'READY', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'SKIPPED')",
            name="ck_workflow_task_status",
        ),
        UniqueConstraint(
            "workflow_run_id",
            "task_key",
            name="uq_workflow_tasks_run_task_key",
        ),
        Index("ix_workflow_tasks_claim", "status", "lease_expires_at"),
    )




class WorkflowDefinition(Base):
    __tablename__ = "workflow_definitions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    active_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    workspace: Mapped["Workspace"] = relationship()
    created_by: Mapped["User"] = relationship()
    versions: Mapped[list["WorkflowVersion"]] = relationship(back_populates="workflow", cascade="all, delete-orphan", foreign_keys="[WorkflowVersion.workflow_id]")


class WorkflowVersion(Base):
    __tablename__ = "workflow_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workflow_definitions.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), CheckConstraint("status IN ('DRAFT', 'ACTIVE', 'ARCHIVED')", name="ck_workflow_version_status"), default="DRAFT", nullable=False)
    trigger_spec: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    steps_spec: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    canvas_layout: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=dict, nullable=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_by_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    workflow: Mapped["WorkflowDefinition"] = relationship(back_populates="versions", foreign_keys=[workflow_id])
    created_by: Mapped["User"] = relationship()

    __table_args__ = (
        UniqueConstraint("workflow_id", "version_number", name="uq_workflow_versions_wf_version"),
        Index("ix_workflow_versions_wf_status", "workflow_id", "status"),
    )


class WorkflowTriggerEvent(Base):
    __tablename__ = "workflow_trigger_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    event_source: Mapped[str] = mapped_column(String(50), nullable=False)
    event_name: Mapped[str] = mapped_column(String(100), nullable=False)
    workspace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    workflow_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workflow_definitions.id", ondelete="CASCADE"), nullable=False, index=True)
    workflow_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workflow_versions.id", ondelete="RESTRICT"), nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint("status IN ('PENDING', 'PROCESSING', 'DISPATCHED', 'FAILED')", name="ck_workflow_trigger_event_status"),
        default="PENDING",
        nullable=False,
        index=True,
    )
    workflow_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkflowScheduleState(Base):
    __tablename__ = "workflow_schedule_states"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workflow_definitions.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    last_scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_window_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    misfire_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="IDLE", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class MemoryPrivacyPolicy(Base):
    __tablename__ = "memory_privacy_policies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_type: Mapped[str] = mapped_column(String(20), nullable=False)
    scope_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    memory_collection_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    memory_extraction_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    memory_dreaming_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    memory_retrieval_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    retention_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    auto_expiration_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    user_deletion_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    profile_scrubbing_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    memory_export_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    user_can_pause: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint("scope_type IN ('TENANT', 'WORKSPACE', 'USER')", name="ck_memory_privacy_scope_type"),
        UniqueConstraint("scope_type", "scope_id", name="uq_memory_privacy_policies_scope"),
        Index("ix_memory_privacy_policies_scope_type", "scope_type"),
        Index("ix_memory_privacy_policies_scope_id", "scope_id"),
    )


class MemoryTombstone(Base):
    __tablename__ = "memory_tombstones"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    canonical_key: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str] = mapped_column(String(50), default="FORGET", nullable=False)
    section: Mapped[str | None] = mapped_column(String(50), nullable=True)
    key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped["User"] = relationship()

    __table_args__ = (
        UniqueConstraint("user_id", "canonical_key", name="uq_memory_tombstones_user_key"),
        Index("ix_memory_tombstones_user_id", "user_id"),
        Index("ix_memory_tombstones_canonical_key", "canonical_key"),
    )


class MemoryPrivacyAuditEvent(Base):
    __tablename__ = "memory_privacy_audit_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True)
    department_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="SUCCESS", nullable=False)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata_", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "action IN ('POLICY_UPDATE', 'MEMORY_PAUSE', 'MEMORY_RESUME', 'MEMORY_FORGET', 'MEMORY_DELETE_ALL', 'PROFILE_SCRUB', 'RETENTION_SWEEP', 'MEMORY_EXPORT')",
            name="ck_memory_privacy_audit_action",
        ),
        CheckConstraint("status IN ('SUCCESS', 'FAILED')", name="ck_memory_privacy_audit_status"),
        Index("ix_memory_privacy_audit_events_user_id", "user_id"),
        Index("ix_memory_privacy_audit_events_workspace_id", "workspace_id"),
        Index("ix_memory_privacy_audit_events_department_id", "department_id"),
        Index("ix_memory_privacy_audit_events_action", "action"),
        Index("ix_memory_privacy_audit_events_created_at", "created_at"),
    )


class MemoryRelationship(Base):
    __tablename__ = "memory_relationships"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("memory_items.id", ondelete="CASCADE"), nullable=False, index=True)
    target_memory_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("memory_items.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_type: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint("relationship_type IN ('DUPLICATE', 'COMPLEMENTS', 'SUPERSEDES', 'CONTRADICTS', 'RELATED', 'UNRESOLVED')", name="ck_memory_relationship_type"),
        nullable=False,
        index=True,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB, server_default=text("'{}'::jsonb"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    source_memory: Mapped["MemoryItem"] = relationship(foreign_keys=[source_memory_id])
    target_memory: Mapped["MemoryItem"] = relationship(foreign_keys=[target_memory_id])

    __table_args__ = (
        Index("ix_memory_relationships_source_target", "source_memory_id", "target_memory_id"),
    )


class MemoryDreamingRun(Base):
    __tablename__ = "memory_dreaming_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True)
    trigger_type: Mapped[str] = mapped_column(String(50), default="SCHEDULED", nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint("status IN ('PENDING', 'RUNNING', 'COMPLETED', 'PARTIALLY_COMPLETED', 'FAILED', 'CANCELLED', 'SKIPPED_POLICY_PAUSED')", name="ck_memory_dreaming_run_status"),
        default="PENDING",
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    memories_inspected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    memories_skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duplicates_detected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    conflicts_detected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    new_memories_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    memories_superseded: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    profile_updated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_usage: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    checkpoint_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_memory_dreaming_runs_user_status", "user_id", "status"),
    )


class MemoryEvalMetric(Base):
    __tablename__ = "memory_eval_metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    eval_run_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    scenario_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_memory_eval_metrics_run_category", "eval_run_id", "metric_category"),
    )


class DocumentChunkVersion(Base):
    __tablename__ = "document_chunk_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text_before: Mapped[str] = mapped_column(Text, nullable=False)
    text_after: Mapped[str] = mapped_column(Text, nullable=False)
    diff_patch: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_snapshot: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    edited_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    edit_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_rollback: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reverted_from_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    chunk: Mapped["DocumentChunk"] = relationship()
    editor: Mapped["User | None"] = relationship(foreign_keys=[edited_by])

    __table_args__ = (
        UniqueConstraint("chunk_id", "version_number", name="uq_chunk_version_number"),
        Index("ix_chunk_versions_created_at", "created_at"),
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    source_content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    extraction_config_version: Mapped[str] = mapped_column(String(50), default="v1", nullable=False)
    chunking_config_version: Mapped[str] = mapped_column(String(50), default="v1", nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), default="sentence-transformers/all-MiniLM-L6-v2", nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, default=384, nullable=False)
    status: Mapped[str] = mapped_column(
        String(30),
        CheckConstraint("status IN ('DETECTED', 'PROCESSING', 'SHADOW_VALIDATION', 'PENDING_APPROVAL', 'PUBLISHED', 'FAILED', 'SUPERSEDED', 'ROLLED_BACK')", name="ck_document_version_status"),
        default="DETECTED",
        nullable=False,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    change_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    document: Mapped["Document"] = relationship(foreign_keys=[document_id])

    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_doc_version_number"),
    )


class DocumentPageVersion(Base):
    __tablename__ = "document_page_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    page_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    structural_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("status IN ('UNCHANGED', 'MODIFIED', 'ADDED', 'DELETED', 'MOVED', 'SPLIT', 'MERGED', 'UNKNOWN')", name="ck_doc_page_version_status"),
        default="ADDED",
        nullable=False,
    )
    reused_from_page_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_page_versions.id", ondelete="SET NULL"), nullable=True)
    mapping_confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DocumentVersionChunk(Base):
    __tablename__ = "document_version_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_reused: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reused_from_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="SET NULL"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("document_version_id", "chunk_id", name="uq_doc_version_chunk"),
    )


class KnowledgeDependency(Base):
    __tablename__ = "knowledge_dependencies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    page_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_page_versions.id", ondelete="SET NULL"), nullable=True, index=True)
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=True, index=True)
    knowledge_item_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_items.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_relationships.id", ondelete="SET NULL"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(30),
        CheckConstraint("status IN ('ACTIVE', 'STALE', 'PENDING_REVALIDATION', 'SUPERSEDED', 'REVOKED', 'ARCHIVED')", name="ck_knowledge_dependency_status"),
        default="ACTIVE",
        nullable=False,
    )
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ShadowIngestionReport(Base):
    __tablename__ = "shadow_ingestion_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    comparison_status: Mapped[str] = mapped_column(
        String(30),
        CheckConstraint("comparison_status IN ('EXACT_MATCH', 'ACCEPTABLE_DIFFERENCE', 'MATERIAL_DIFFERENCE', 'FAILED')", name="ck_shadow_comparison_status"),
        nullable=False,
    )
    exact_matches: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reused_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    new_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    deleted_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    modified_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    divergence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    validation_passed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class DocumentNote(Base):
    __tablename__ = "document_notes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    author_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    correction_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    approval_status: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("approval_status IN ('PENDING', 'APPROVED', 'REJECTED')", name="ck_document_notes_approval_status"),
        default="APPROVED",
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    effective_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    document: Mapped["Document"] = relationship()
    author: Mapped["User | None"] = relationship(foreign_keys=[author_user_id])

    __table_args__ = (
        Index("ix_document_notes_doc_active", "document_id", "is_active", "approval_status"),
    )
