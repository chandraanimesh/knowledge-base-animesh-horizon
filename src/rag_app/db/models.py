import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, Enum, ForeignKey, ForeignKeyConstraint, Index, Integer, String, Text, UniqueConstraint, func, text
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

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

    __table_args__ = (
        CheckConstraint(
            "(document_chunk_id IS NOT NULL AND research_evidence_id IS NULL) OR "
            "(document_chunk_id IS NULL AND research_evidence_id IS NOT NULL)",
            name="ck_knowledgeevidence_single_source"
        ),
        UniqueConstraint(
            "knowledge_item_id",
            "document_chunk_id",
            "research_evidence_id",
            name="uq_knowledge_evidence_source",
            postgresql_nulls_not_distinct=True,
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
    superseded_by: Mapped["MemoryItem | None"] = relationship(
        remote_side=[id],
        foreign_keys=[superseded_by_id],
    )

    __table_args__ = (
        Index(
            "ix_memory_items_user_canonical_key_active",
            "user_id",
            "canonical_key",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
        ),
        Index(
            "ix_memory_items_user_status",
            "user_id",
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


