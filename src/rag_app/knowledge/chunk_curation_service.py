"""
Chunk Curation & Rollback Lifecycle Service (Milestone 7A.1).

Guarantees:
1. Version numbers identify exact, reproducible chunk states with matching embedding snapshots.
2. Embedding-first generation: failed embedding generation never leaves updated text paired with an old embedding.
3. Row locking (with_for_update) and fresh base_version check prevents lost updates.
4. Stale base_version returns HTTP 409 Conflict.
5. Rollback creates a new forward-moving version without erasing historical versions.
6. Restores embedding snapshot only if model identity and dimensions are compatible; re-embeds otherwise.
7. External CRM synchronized chunks are protected from manual direct edits.
8. Hierarchical OV-1/OV-2 sections mark derived summaries/abstracts as stale upon chunk mutation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import difflib
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional
import uuid

from fastapi import HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from rag_app.core.config import get_settings
from rag_app.db.models import (
    Department,
    Document,
    DocumentChunk,
    DocumentChunkVersion,
    DocumentSection,
    User,
    UserDepartment,
    Workspace,
    WorkspaceMember,
)
from rag_app.ingestion.embedding_v1 import EmbeddingProvider, get_embedding_provider
from rag_app.knowledge.quality_checks import run_deterministic_quality_checks
from rag_app.schemas.chunk_curation import (
    ChunkDetailResponse,
    ChunkDiffResponse,
    ChunkEditRequest,
    ChunkEvidenceResponse,
    ChunkIssueReportRequest,
    ChunkRollbackRequest,
    ChunkVersionResponse,
    QualityCheckFlag,
    QualityReportItem,
)
from rag_app.security.authorization import authorize

logger = logging.getLogger(__name__)
settings = get_settings()


class ChunkCurationService:
    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.session = session
        self._embedding_provider = embedding_provider

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    async def _resolve_chunk_and_document(
        self,
        document_id: uuid.UUID,
        chunk_id: uuid.UUID,
    ) -> tuple[DocumentChunk, Document]:
        stmt = (
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                and_(
                    DocumentChunk.id == chunk_id,
                    DocumentChunk.document_id == document_id,
                )
            )
        )
        res = (await self.session.execute(stmt)).first()
        if not res:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk {chunk_id} not found in document {document_id}.",
            )
        return res[0], res[1]

    async def _check_read_authorization(self, user: User, document: Document) -> None:
        """
        Ensures user has read access to the document and chunk.
        Fails closed by default.
        """
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user.",
            )
        if user.role == "ADMIN":
            return

        if document.workspace_id:
            ws = await self.session.get(Workspace, document.workspace_id)
            if not ws or not ws.is_active or ws.deleted_at is not None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workspace not found or inactive.",
                )

            # Department check
            dept = await self.session.get(Department, ws.department_id)
            if not dept or not dept.is_active or dept.deleted_at is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Workspace department inactive or unavailable.",
                )
            is_dept_auth = await authorize(self.session, user, "VIEW", dept.code)
            if not is_dept_auth:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Forbidden: Missing VIEW permission for department {dept.code}.",
                )

            # Workspace membership check
            member = await self.session.scalar(
                select(WorkspaceMember).where(
                    and_(
                        WorkspaceMember.workspace_id == ws.id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
            )
            if not member or member.role not in ("OWNER", "EDITOR", "VIEWER"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Not an active member of this workspace.",
                )
            return

        if document.department_id:
            dept = await self.session.get(Department, document.department_id)
            if not dept or not dept.is_active or dept.deleted_at is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Document department inactive or unavailable.",
                )
            is_dept_auth = await authorize(self.session, user, "VIEW", dept.code)
            if not is_dept_auth:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Forbidden: Missing VIEW permission for department {dept.code}.",
                )
            return

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Document has no active tenant partition.",
        )

    async def _check_write_authorization(self, user: User, document: Document) -> None:
        """
        Ensures caller has mutation authorization for chunk curation and rollback.
        VIEWER roles are strictly rejected with 403 Forbidden.
        """
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user.",
            )
        if user.role == "ADMIN":
            return

        if document.workspace_id:
            ws = await self.session.get(Workspace, document.workspace_id)
            if not ws or not ws.is_active or ws.deleted_at is not None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workspace not found or inactive.",
                )

            dept = await self.session.get(Department, ws.department_id)
            if not dept or not dept.is_active or dept.deleted_at is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Workspace department inactive or unavailable.",
                )
            is_dept_auth = await authorize(self.session, user, "VIEW", dept.code)
            if not is_dept_auth:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Forbidden: Missing VIEW permission for department {dept.code}.",
                )

            member = await self.session.scalar(
                select(WorkspaceMember).where(
                    and_(
                        WorkspaceMember.workspace_id == ws.id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
            )
            if not member:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Not an active member of this workspace.",
                )
            if member.role == "VIEWER":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Viewer role is read-only. Chunk curation requires Editor or Owner privileges.",
                )
            if member.role not in ("OWNER", "EDITOR"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient workspace role for chunk curation.",
                )
            return

        if document.department_id:
            dept = await self.session.get(Department, document.department_id)
            if not dept or not dept.is_active or dept.deleted_at is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Document department inactive or unavailable.",
                )
            is_edit_auth = await authorize(self.session, user, "EDIT", dept.code)
            if is_edit_auth:
                return

            # Check if user is the document creator and member of department
            if document.created_by == user.id:
                is_member = await self.session.scalar(
                    select(UserDepartment).where(
                        and_(
                            UserDepartment.user_id == user.id,
                            UserDepartment.department_id == dept.id,
                        )
                    )
                )
                if is_member:
                    return

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Missing EDIT permission for department {dept.code}.",
            )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Document has no active tenant partition.",
        )

    def _is_managed_external(self, chunk: DocumentChunk, document: Document) -> bool:
        meta = chunk.metadata_ or {}
        if (
            meta.get("source") == "twenty_crm_mcp"
            or meta.get("doc_type") == "CRM_RECORD"
            or meta.get("managed_external") is True
            or document.file_name.startswith(("twenty_", "CRM_"))
        ):
            return True
        return False

    def _check_not_managed_external(self, chunk: DocumentChunk, document: Document) -> None:
        if self._is_managed_external(chunk, document):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot edit managed external CRM synchronized chunk directly. Content is governed by CRM sync lifecycle.",
            )

    async def get_chunk_detail(
        self,
        document_id: uuid.UUID,
        chunk_id: uuid.UUID,
        user: User,
    ) -> ChunkDetailResponse:
        chunk, document = await self._resolve_chunk_and_document(document_id, chunk_id)
        await self._check_read_authorization(user, document)

        return ChunkDetailResponse(
            id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            chunk_hash=chunk.chunk_hash,
            current_version=chunk.current_version,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
            page_number=chunk.page_number,
            node_id=chunk.node_id,
            is_managed_external=self._is_managed_external(chunk, document),
        )

    async def list_chunk_versions(
        self,
        document_id: uuid.UUID,
        chunk_id: uuid.UUID,
        user: User,
    ) -> list[ChunkVersionResponse]:
        chunk, document = await self._resolve_chunk_and_document(document_id, chunk_id)
        await self._check_read_authorization(user, document)

        stmt = (
            select(DocumentChunkVersion)
            .where(DocumentChunkVersion.chunk_id == chunk.id)
            .order_by(DocumentChunkVersion.version_number.asc())
        )
        versions = (await self.session.scalars(stmt)).all()

        # If chunk is at version 1 and has no version records yet, dynamically construct initial version representation
        if not versions and chunk.current_version == 1:
            return [
                ChunkVersionResponse(
                    id=uuid.uuid4(),
                    chunk_id=chunk.id,
                    version_number=1,
                    text_before=chunk.text,
                    text_after=chunk.text,
                    diff_patch=None,
                    embedding_model=settings.embedding_model,
                    embedding_dim=settings.embedding_dim,
                    chunk_hash=chunk.chunk_hash,
                    edited_by=document.created_by,
                    created_at=chunk.created_at,
                    edit_reason="Initial ingestion version",
                    is_rollback=False,
                    reverted_from_version=None,
                )
            ]

        return [ChunkVersionResponse.model_validate(v) for v in versions]

    async def get_chunk_diff(
        self,
        document_id: uuid.UUID,
        chunk_id: uuid.UUID,
        v1: int,
        v2: int,
        user: User,
    ) -> ChunkDiffResponse:
        chunk, document = await self._resolve_chunk_and_document(document_id, chunk_id)
        await self._check_read_authorization(user, document)

        if v1 < 1 or v2 < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Version numbers must be >= 1.",
            )

        text_v1 = await self._get_version_text(chunk, v1)
        text_v2 = await self._get_version_text(chunk, v2)

        lines1 = text_v1.splitlines(keepends=True)
        lines2 = text_v2.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                lines1,
                lines2,
                fromfile=f"version_{v1}",
                tofile=f"version_{v2}",
            )
        )
        diff_patch = "".join(diff_lines)

        lines_added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        lines_removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
        chars_changed = abs(len(text_v2) - len(text_v1))

        return ChunkDiffResponse(
            chunk_id=chunk.id,
            v1=v1,
            v2=v2,
            text_v1=text_v1,
            text_v2=text_v2,
            diff_patch=diff_patch,
            lines_added=lines_added,
            lines_removed=lines_removed,
            characters_changed=chars_changed,
        )

    async def _get_version_text(self, chunk: DocumentChunk, version_number: int) -> str:
        if version_number == chunk.current_version:
            return chunk.text

        stmt = select(DocumentChunkVersion).where(
            and_(
                DocumentChunkVersion.chunk_id == chunk.id,
                DocumentChunkVersion.version_number == version_number,
            )
        )
        ver = await self.session.scalar(stmt)
        if not ver:
            # If requesting v1 on a chunk that hasn't been edited yet
            if version_number == 1 and chunk.current_version == 1:
                return chunk.text
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version_number} does not exist for chunk {chunk.id}.",
            )
        return ver.text_after

    async def edit_chunk(
        self,
        document_id: uuid.UUID,
        chunk_id: uuid.UUID,
        payload: ChunkEditRequest,
        user: User,
    ) -> tuple[DocumentChunk, DocumentChunkVersion]:
        chunk, document = await self._resolve_chunk_and_document(document_id, chunk_id)
        await self._check_write_authorization(user, document)
        self._check_not_managed_external(chunk, document)

        clean_text = payload.new_text.strip()
        if not clean_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chunk text cannot be empty or whitespace only.",
            )

        if clean_text == chunk.text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New text is identical to current chunk text.",
            )

        # -------------------------------------------------------------
        # INVARIANT 2: Embedding-First Generation
        # Generate embedding before modifying persistent chunk state.
        # If this raises, no database state is touched.
        # -------------------------------------------------------------
        try:
            embeddings = await self.embedding_provider.embed_texts([clean_text])
            new_embedding = embeddings[0]
            if len(new_embedding) != settings.embedding_dim:
                raise ValueError(
                    f"Generated embedding dimension {len(new_embedding)} does not match expected {settings.embedding_dim}"
                )
        except Exception as ex:
            logger.error("Embedding generation failed for chunk edit: %s", ex)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding provider failure: {str(ex)[:200]}",
            )

        new_chunk_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)

        # -------------------------------------------------------------
        # INVARIANTS 3 & 4: Row Locking and Base Version Optimistic Check
        # -------------------------------------------------------------
        lock_stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.id == chunk_id)
            .with_for_update()
        )
        locked_chunk = await self.session.scalar(lock_stmt)
        if not locked_chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chunk disappeared during update locking.",
            )

        if locked_chunk.current_version != payload.base_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Conflict: chunk current version is {locked_chunk.current_version}, "
                    f"but base_version was {payload.base_version}. Please refresh and retry."
                ),
            )

        # If this is the first edit from version 1, ensure version 1 snapshot is persisted
        if locked_chunk.current_version == 1:
            has_v1 = await self.session.scalar(
                select(DocumentChunkVersion).where(
                    and_(
                        DocumentChunkVersion.chunk_id == locked_chunk.id,
                        DocumentChunkVersion.version_number == 1,
                    )
                )
            )
            if not has_v1:
                v1_record = DocumentChunkVersion(
                    id=uuid.uuid4(),
                    chunk_id=locked_chunk.id,
                    version_number=1,
                    text_before=locked_chunk.text,
                    text_after=locked_chunk.text,
                    diff_patch=None,
                    embedding_snapshot=locked_chunk.embedding,
                    embedding_model=settings.embedding_model,
                    embedding_dim=settings.embedding_dim,
                    chunk_hash=locked_chunk.chunk_hash,
                    edited_by=document.created_by or user.id,
                    created_at=locked_chunk.created_at,
                    edit_reason="Initial ingestion version",
                    is_rollback=False,
                    reverted_from_version=None,
                )
                self.session.add(v1_record)
                await self.session.flush()

        new_version_num = locked_chunk.current_version + 1

        # Compute diff patch
        diff_lines = list(
            difflib.unified_diff(
                locked_chunk.text.splitlines(keepends=True),
                clean_text.splitlines(keepends=True),
                fromfile=f"v{locked_chunk.current_version}",
                tofile=f"v{new_version_num}",
            )
        )
        diff_patch = "".join(diff_lines)

        # Create new version snapshot record
        version_record = DocumentChunkVersion(
            id=uuid.uuid4(),
            chunk_id=locked_chunk.id,
            version_number=new_version_num,
            text_before=locked_chunk.text,
            text_after=clean_text,
            diff_patch=diff_patch,
            embedding_snapshot=new_embedding,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            chunk_hash=new_chunk_hash,
            edited_by=user.id,
            created_at=now,
            edit_reason=payload.edit_reason,
            is_rollback=False,
            reverted_from_version=None,
        )
        self.session.add(version_record)

        # Update active chunk
        locked_chunk.text = clean_text
        locked_chunk.embedding = new_embedding
        locked_chunk.chunk_hash = new_chunk_hash
        locked_chunk.current_version = new_version_num
        locked_chunk.updated_at = now

        # -------------------------------------------------------------
        # INVARIANT 8: Hierarchical Stale Summary Flagging
        # -------------------------------------------------------------
        if locked_chunk.node_id:
            section = await self.session.get(DocumentSection, locked_chunk.node_id)
            if section:
                sec_meta = dict(section.metadata_ or {})
                sec_meta["is_stale"] = True
                sec_meta["stale_reason"] = f"Chunk {locked_chunk.id} edited to version {new_version_num}"
                sec_meta["stale_at"] = now.isoformat()
                section.metadata_ = sec_meta

        await self.session.commit()
        await self.session.refresh(locked_chunk)
        await self.session.refresh(version_record)

        return locked_chunk, version_record

    async def rollback_chunk(
        self,
        document_id: uuid.UUID,
        chunk_id: uuid.UUID,
        payload: ChunkRollbackRequest,
        user: User,
    ) -> tuple[DocumentChunk, DocumentChunkVersion]:
        chunk, document = await self._resolve_chunk_and_document(document_id, chunk_id)
        await self._check_write_authorization(user, document)
        self._check_not_managed_external(chunk, document)

        # Locate historical target version
        target_v = await self.session.scalar(
            select(DocumentChunkVersion).where(
                and_(
                    DocumentChunkVersion.chunk_id == chunk.id,
                    DocumentChunkVersion.version_number == payload.target_version,
                )
            )
        )
        if not target_v:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target version {payload.target_version} does not exist for chunk {chunk.id}.",
            )

        target_text = target_v.text_after
        if target_text == chunk.text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chunk is already in target version {payload.target_version} state.",
            )

        # -------------------------------------------------------------
        # INVARIANT 6: Embedding Compatibility Check
        # Restore embedding snapshot only if model identity and dimensions match;
        # otherwise re-embed target text.
        # -------------------------------------------------------------
        if (
            target_v.embedding_model == settings.embedding_model
            and target_v.embedding_dim == settings.embedding_dim
            and len(target_v.embedding_snapshot) == settings.embedding_dim
        ):
            restored_embedding = list(target_v.embedding_snapshot)
        else:
            logger.info(
                "Historical embedding snapshot incompatible (model=%s, dim=%s). Re-embedding target text with active provider.",
                target_v.embedding_model,
                target_v.embedding_dim,
            )
            try:
                embeddings = await self.embedding_provider.embed_texts([target_text])
                restored_embedding = embeddings[0]
            except Exception as ex:
                logger.error("Embedding generation failed during rollback: %s", ex)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Embedding provider failure during rollback: {str(ex)[:200]}",
                )

        now = datetime.now(timezone.utc)

        # -------------------------------------------------------------
        # INVARIANT 5: Forward-Moving Rollback Version
        # -------------------------------------------------------------
        lock_stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.id == chunk_id)
            .with_for_update()
        )
        locked_chunk = await self.session.scalar(lock_stmt)
        if not locked_chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chunk not found during rollback locking.",
            )

        new_version_num = locked_chunk.current_version + 1

        # Diff from current text to target text
        diff_lines = list(
            difflib.unified_diff(
                locked_chunk.text.splitlines(keepends=True),
                target_text.splitlines(keepends=True),
                fromfile=f"v{locked_chunk.current_version}",
                tofile=f"v{new_version_num}_rollback_from_v{payload.target_version}",
            )
        )
        diff_patch = "".join(diff_lines)

        rollback_record = DocumentChunkVersion(
            id=uuid.uuid4(),
            chunk_id=locked_chunk.id,
            version_number=new_version_num,
            text_before=locked_chunk.text,
            text_after=target_text,
            diff_patch=diff_patch,
            embedding_snapshot=restored_embedding,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            chunk_hash=target_v.chunk_hash,
            edited_by=user.id,
            created_at=now,
            edit_reason=payload.rollback_reason or f"Rollback to version {payload.target_version}",
            is_rollback=True,
            reverted_from_version=payload.target_version,
        )
        self.session.add(rollback_record)

        # Update active chunk
        locked_chunk.text = target_text
        locked_chunk.embedding = restored_embedding
        locked_chunk.chunk_hash = target_v.chunk_hash
        locked_chunk.current_version = new_version_num
        locked_chunk.updated_at = now

        # INVARIANT 8: Hierarchical Stale Summary Flagging
        if locked_chunk.node_id:
            section = await self.session.get(DocumentSection, locked_chunk.node_id)
            if section:
                sec_meta = dict(section.metadata_ or {})
                sec_meta["is_stale"] = True
                sec_meta["stale_reason"] = (
                    f"Chunk {locked_chunk.id} rolled back to version {payload.target_version} (as v{new_version_num})"
                )
                sec_meta["stale_at"] = now.isoformat()
                section.metadata_ = sec_meta

        await self.session.commit()
        await self.session.refresh(locked_chunk)
        await self.session.refresh(rollback_record)

        return locked_chunk, rollback_record

    async def get_chunk_evidence(
        self,
        document_id: uuid.UUID,
        chunk_id: uuid.UUID,
        user: User,
    ) -> ChunkEvidenceResponse:
        """
        Inspect available source evidence and run deterministic quality checks (Milestone 7A.3).
        Provides honest source-availability state, distinguishes original ingested vs curated text,
        and aggregates substantiated quality flags.
        """
        chunk, document = await self._resolve_chunk_and_document(document_id, chunk_id)
        await self._check_read_authorization(user, document)

        # 1. Source availability check
        source_path = Path(document.file_path) if document.file_path else None
        source_available = False
        source_availability_reason = None
        if source_path and source_path.exists() and source_path.is_file():
            source_available = True
        else:
            source_available = False
            if not document.file_path:
                source_availability_reason = "No source file path recorded in document registry."
            else:
                source_availability_reason = f"Source file does not exist on host storage: {document.file_path}"

        # 2. Original vs Curated text distinction
        # Version 1 holds the unedited ingested content in text_after
        v1_stmt = (
            select(DocumentChunkVersion)
            .where(
                and_(
                    DocumentChunkVersion.chunk_id == chunk.id,
                    DocumentChunkVersion.version_number == 1,
                )
            )
        )
        v1_record = await self.session.scalar(v1_stmt)
        if v1_record:
            original_text = v1_record.text_after
        else:
            # Fallback if chunk has not yet undergone versioned curation
            original_text = chunk.text

        is_curated = (chunk.current_version > 1) or (original_text != chunk.text)

        # 3. Section hierarchy location (OV-1 / OV-2)
        section_title: Optional[str] = None
        section_ordinal: Optional[str] = None
        section_abstract: Optional[str] = None
        if chunk.node_id:
            section = await self.session.get(DocumentSection, chunk.node_id)
            if section:
                section_title = section.title
                section_ordinal = section.ordinal_path
                section_abstract = section.l0_abstract or section.l1_overview

        # 4. Source URL extraction
        meta = dict(chunk.metadata_ or {})
        source_url = meta.get("source_url") or meta.get("url")
        if not source_url and document.file_path and document.file_path.startswith(("http://", "https://")):
            source_url = document.file_path

        is_managed = self._is_managed_external(chunk, document)

        # 5. Deterministic quality checks
        quality_flags = run_deterministic_quality_checks(
            text=chunk.text,
            original_text=original_text,
            current_version=chunk.current_version,
            mime_type=document.mime_type,
            file_name=document.file_name,
            page_number=chunk.page_number,
            is_managed_external=is_managed,
            source_available=source_available,
        )

        # 6. Parse user-reported quality issues
        raw_reports = meta.get("quality_reports", [])
        quality_reports: list[QualityReportItem] = []
        for r in raw_reports:
            try:
                quality_reports.append(QualityReportItem.model_validate(r))
            except Exception as ex:
                logger.warning("Could not parse quality report item %s: %s", r, ex)

        return ChunkEvidenceResponse(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            file_name=document.file_name,
            file_path=document.file_path,
            mime_type=document.mime_type,
            source_available=source_available,
            source_availability_reason=source_availability_reason,
            source_url=source_url,
            page_number=chunk.page_number,
            node_id=chunk.node_id,
            section_title=section_title,
            section_ordinal=section_ordinal,
            section_abstract=section_abstract,
            current_text=chunk.text,
            original_text=original_text,
            is_curated=is_curated,
            current_version=chunk.current_version,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
            chunk_hash=chunk.chunk_hash,
            is_managed_external=is_managed,
            quality_flags=quality_flags,
            quality_reports=quality_reports,
        )

    async def report_chunk_issue(
        self,
        document_id: uuid.UUID,
        chunk_id: uuid.UUID,
        payload: ChunkIssueReportRequest,
        user: User,
    ) -> dict[str, Any]:
        """
        Record a suspicious or incorrect source/chunk report from an authorized user (Milestone 7A.3).
        Preserves existing schema by storing reports under DocumentChunk.metadata_['quality_reports'].
        """
        chunk, document = await self._resolve_chunk_and_document(document_id, chunk_id)
        await self._check_read_authorization(user, document)

        now = datetime.now(timezone.utc)
        report_item = {
            "report_id": f"rep_{uuid.uuid4().hex[:12]}",
            "reported_by": user.email or str(user.id),
            "reported_at": now.isoformat(),
            "issue_type": payload.issue_type.strip(),
            "description": payload.description.strip(),
            "suggested_remediation": payload.suggested_remediation,
            "status": "OPEN",
        }

        meta = dict(chunk.metadata_ or {})
        existing_reports = list(meta.get("quality_reports", []))
        existing_reports.append(report_item)
        meta["quality_reports"] = existing_reports
        chunk.metadata_ = meta
        flag_modified(chunk, "metadata_")

        await self.session.commit()
        await self.session.refresh(chunk)

        logger.info(
            "User %s reported quality issue on chunk %s: %s (%s)",
            user.email,
            chunk.id,
            payload.issue_type,
            payload.suggested_remediation,
        )

        return {
            "message": "Quality issue report logged successfully.",
            "report": report_item,
            "total_reports": len(existing_reports),
        }
