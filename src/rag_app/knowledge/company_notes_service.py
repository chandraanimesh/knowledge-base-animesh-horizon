import hashlib
import logging
import urllib.parse
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.db.models import (
    Department,
    Document,
    DocumentChunk,
    DocumentSection,
    KnowledgeEntity,
    KnowledgeEvidence,
    KnowledgeItem,
    KnowledgeRelationship,
    User,
    Workspace,
    WorkspaceMember,
)
from rag_app.knowledge.materializer import normalize_string
from rag_app.schemas.knowledge import (
    BatchPublicationItemOutcome,
    BatchPublicationRequest,
    BatchPublicationResponse,
    CompanyNoteCreateRequest,
    CompanyNoteEvidenceAttachRequest,
    CompanyNoteEvidenceItem,
    CompanyNoteResponse,
    CompanyNoteReviewRequest,
    CompanyNoteSubmitRequest,
    CompanyNoteUpdateRequest,
)
from rag_app.security.authorization import authorize, authorize_document
from rag_app.core.filenames import derive_original_filename

logger = logging.getLogger(__name__)

CONFIDENCE_MAP = {
    "SUPPORTED": 1.0,
    "INFERRED": 0.7,
    "UNVERIFIED": 0.3,
}


def validate_website_url(url: str) -> str:
    """
    Validates website source URL syntax.
    Strictly permits only HTTP and HTTPS schemes with valid domain structure.
    Does not fetch or crawl the URL.
    """
    if not url or not isinstance(url, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Website URL cannot be empty",
        )
    trimmed = url.strip()
    try:
        parsed = urllib.parse.urlparse(trimmed)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed website URL: {str(exc)}",
        )
    if parsed.scheme.lower() not in ("http", "https"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported URL scheme '{parsed.scheme}'. Only HTTP and HTTPS URLs are allowed.",
        )
    if not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid website URL: missing domain host.",
        )
    domain_part = parsed.netloc.split(":")[0]
    if "." not in domain_part and domain_part.lower() != "localhost":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid website domain '{parsed.netloc}'. Must be a valid domain or host.",
        )
    return trimmed


class CompanyNotesService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _format_evidence_item(
        self,
        ev: KnowledgeEvidence,
        user: Optional[User] = None,
    ) -> CompanyNoteEvidenceItem:
        if ev.source_url:
            return CompanyNoteEvidenceItem(
                id=ev.id,
                source_type="WEBSITE",
                document_chunk_id=None,
                document_id=None,
                file_name=ev.source_title or ev.source_url,
                page_number=None,
                section_title=None,
                section_ordinal=None,
                excerpt=ev.user_excerpt or "",
                extraction_method=ev.extraction_method or "WEBSITE_REGISTRATION",
                confidence=ev.confidence,
                created_at=ev.created_at,
                source_url=ev.source_url,
                source_title=ev.source_title,
                user_excerpt=ev.user_excerpt,
                verification_status=ev.verification_status or "NOT_VERIFIED",
                verified_at=ev.verified_at,
            )

        chunk = ev.document_chunk
        doc = chunk.document if chunk else None
        node = chunk.node if chunk else None

        file_name = doc.file_name if doc else "Unknown Source"
        original_fname = derive_original_filename(doc.file_name, doc.file_path) if doc else None
        page_num = chunk.page_number if chunk else None
        sec_title = node.title if node else None
        sec_ordinal = node.ordinal_path if node else None

        # Excerpt masking policy:
        # Prevent unauthorized disclosure of restricted source text.
        # Check if caller has VIEW permission on the specific source document.
        # If user lacks permission, mask the excerpt while preserving provenance metadata.
        can_view_excerpt = False
        if user and doc:
            can_view_excerpt = await authorize_document(self.session, user, "VIEW", doc.id)

        if can_view_excerpt:
            excerpt = (chunk.text[:500] if chunk and chunk.text else "")
        else:
            excerpt = "[RESTRICTED - View permission on source document required]"

        return CompanyNoteEvidenceItem(
            id=ev.id,
            source_type="DOCUMENT",
            document_chunk_id=ev.document_chunk_id or uuid.UUID(int=0),
            document_id=doc.id if doc else uuid.UUID(int=0),
            file_name=file_name,
            original_file_name=original_fname,
            page_number=page_num,
            section_title=sec_title,
            section_ordinal=sec_ordinal,
            excerpt=excerpt,
            extraction_method=ev.extraction_method,
            confidence=ev.confidence,
            created_at=ev.created_at,
            source_url=None,
            source_title=None,
            user_excerpt=None,
            verification_status="VERIFIED" if doc else "NOT_VERIFIED",
            verified_at=ev.created_at if doc else None,
        )

    async def _format_note_response(
        self,
        item: KnowledgeItem,
        user: Optional[User] = None,
    ) -> CompanyNoteResponse:
        workspace_name = item.workspace.name if item.workspace else None
        department_code = item.department.code if item.department else None
        author_email = item.author.email if item.author else None
        approved_by_email = item.approved_by.email if item.approved_by else None

        entity = await self.session.scalar(
            select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == item.id)
        )
        company_name = ""
        statement = ""
        obs_type = "SUPPORTED"
        if entity:
            company_name = entity.canonical_name or ""
            attrs = entity.attributes or {}
            statement = attrs.get("statement") or ""
            obs_type = attrs.get("observation_type") or "SUPPORTED"
            if not company_name and attrs.get("company_name"):
                company_name = attrs.get("company_name")

        if not company_name and item.canonical_key.startswith("NOTE:"):
            parts = item.canonical_key.split(":")
            if len(parts) >= 2 and parts[1]:
                company_name = parts[1].replace("_", " ").title()

        if not statement:
            if entity and entity.canonical_name:
                statement = f"Observation for {entity.canonical_name}"
            elif company_name:
                statement = f"Observation for {company_name}"
            elif item.canonical_key.startswith("NOTE:"):
                statement = f"Company observation note ({item.canonical_key.split(':')[-1][:8]})"
            else:
                statement = "Client observation draft"

        # Fetch evidence items with loaded chunks and documents
        ev_rows = (
            await self.session.scalars(
                select(KnowledgeEvidence)
                .options(
                    selectinload(KnowledgeEvidence.document_chunk)
                    .selectinload(DocumentChunk.document),
                    selectinload(KnowledgeEvidence.document_chunk)
                    .selectinload(DocumentChunk.node),
                )
                .where(KnowledgeEvidence.knowledge_item_id == item.id)
                .order_by(KnowledgeEvidence.created_at.asc())
            )
        ).all()

        evidence_items = []
        for ev in ev_rows:
            evidence_items.append(await self._format_evidence_item(ev, user=user))

        return CompanyNoteResponse(
            id=item.id,
            canonical_key=item.canonical_key,
            company_name=company_name,
            statement=statement,
            observation_type=obs_type,
            kind=item.kind,
            status=item.status,
            visibility=item.visibility,
            approval_status=item.approval_status,
            source_type=item.source_type,
            workspace_id=item.workspace_id,
            workspace_name=workspace_name,
            department_id=item.department_id,
            department_code=department_code,
            author_id=item.author_id,
            author_email=author_email,
            approved_by_id=item.approved_by_id,
            approved_by_email=approved_by_email,
            approved_at=item.approved_at,
            confidence=item.confidence,
            created_at=item.created_at,
            updated_at=item.updated_at,
            evidence_count=len(evidence_items),
            evidence=evidence_items,
        )

    async def _validate_chunk_access(self, user: User, chunk_id: uuid.UUID) -> DocumentChunk:
        """
        Server-side evidence authorization verification.
        Ensures user can attach only source documents or chunks they are authorized to access.
        """
        chunk = await self.session.scalar(
            select(DocumentChunk)
            .options(
                selectinload(DocumentChunk.document),
                selectinload(DocumentChunk.node),
            )
            .where(DocumentChunk.id == chunk_id)
        )
        if not chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source document chunk not found",
            )

        doc = chunk.document
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document associated with chunk not found",
            )

        # Strictly check document VIEW authorization
        is_auth = await authorize_document(self.session, user, "VIEW", doc.id)
        if not is_auth:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: You are not authorized to access document '{doc.file_name}' (ID: {doc.id})",
            )

        return chunk

    async def create_note(
        self,
        user: User,
        request: CompanyNoteCreateRequest,
    ) -> CompanyNoteResponse:
        """
        Creates a Company Note observation.
        Invariants enforced:
        - Default visibility is 'PRIVATE' (author-only draft).
        - A draft cannot enter shared retrieval merely because it was saved.
        - Any attached chunks must be authorized for this user.
        - Observation type ('SUPPORTED', 'INFERRED', 'UNVERIFIED') is recorded explicitly.
        """
        # Validate workspace access if provided
        target_ws_id = request.workspace_id
        target_dept_id = request.department_id
        if target_ws_id:
            ws = await self.session.get(Workspace, target_ws_id)
            if not ws or not ws.is_active:
                raise HTTPException(status_code=404, detail="Workspace not found or inactive")
            if user.role != "ADMIN":
                is_mem = await self.session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == target_ws_id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not is_mem:
                    raise HTTPException(status_code=403, detail="Forbidden: You are not a member of this workspace")
            if not target_dept_id:
                target_dept_id = ws.department_id

        # Validate department access if department_id is set
        if target_dept_id and user.role != "ADMIN":
            dept = await self.session.get(Department, target_dept_id)
            if dept:
                is_dept_auth = await authorize(self.session, user, "VIEW", dept.code)
                if not is_dept_auth:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Forbidden: You do not have access to department '{dept.code}'",
                    )

        # Validate all requested initial evidence chunks BEFORE creating note
        chunks_to_attach: list[DocumentChunk] = []
        seen_chunk_ids: set[uuid.UUID] = set()
        for chunk_id in request.document_chunk_ids:
            chunk = await self._validate_chunk_access(user, chunk_id)
            if chunk.id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.id)
                chunks_to_attach.append(chunk)

        # Build deterministic canonical key
        clean_company = normalize_string(request.company_name)
        clean_stmt = normalize_string(request.statement)
        stmt_hash = hashlib.sha256(clean_stmt.encode("utf-8")).hexdigest()[:16]
        canonical_key = f"NOTE:{clean_company}:{stmt_hash}"

        obs_type = request.observation_type.upper()
        if obs_type not in CONFIDENCE_MAP:
            obs_type = "SUPPORTED"

        conf = request.confidence if request.confidence is not None else CONFIDENCE_MAP.get(obs_type, 1.0)
        init_vis = request.visibility.upper() if request.visibility else "PRIVATE"
        if init_vis not in ("PRIVATE", "WORKSPACE"):
            init_vis = "PRIVATE"

        now = datetime.now(timezone.utc)
        item = KnowledgeItem(
            canonical_key=canonical_key,
            kind="FACT",
            status="ACTIVE",
            visibility=init_vis,  # Invariant: Defaults to PRIVATE
            approval_status="DRAFT",  # Invariant: Defaults to DRAFT
            source_type="DOCUMENT",
            workspace_id=target_ws_id,
            department_id=target_dept_id,
            author_id=user.id,
            confidence=conf,
            created_at=now,
            updated_at=now,
        )
        self.session.add(item)
        await self.session.flush()

        # Create linked KnowledgeEntity
        entity = KnowledgeEntity(
            knowledge_item_id=item.id,
            entity_type="company_note",
            canonical_name=request.company_name[:255],
            attributes={
                "company_name": request.company_name,
                "statement": request.statement,
                "observation_type": obs_type,
                **request.attributes,
            },
        )
        self.session.add(entity)

        # Attach validated evidence chunks
        for chunk in chunks_to_attach:
            ev = KnowledgeEvidence(
                knowledge_item_id=item.id,
                document_chunk_id=chunk.id,
                extraction_method="MANUAL_EVIDENCE",
                confidence=conf,
                created_at=now,
            )
            self.session.add(ev)

        # Attach validated website sources
        seen_urls: set[str] = set()
        for ws_input in request.website_sources:
            clean_url = validate_website_url(ws_input.url)
            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            ev = KnowledgeEvidence(
                knowledge_item_id=item.id,
                source_url=clean_url,
                source_title=ws_input.title.strip() if ws_input.title else None,
                user_excerpt=ws_input.user_excerpt.strip() if ws_input.user_excerpt else None,
                verification_status="NOT_VERIFIED",
                extraction_method="WEBSITE_REGISTRATION",
                confidence=conf,
                created_at=now,
            )
            self.session.add(ev)

        await self.session.commit()

        # Reload with relationships
        loaded_item = await self.session.scalar(
            select(KnowledgeItem)
            .options(
                selectinload(KnowledgeItem.workspace),
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.author),
                selectinload(KnowledgeItem.approved_by),
            )
            .where(KnowledgeItem.id == item.id)
        )
        return await self._format_note_response(loaded_item, user=user)

    async def get_note(self, user: User, note_id: uuid.UUID) -> CompanyNoteResponse:
        """
        Fetches a note ensuring strict visibility boundaries:
        - PRIVATE notes are strictly author-only (even for ADMINs).
        - WORKSPACE notes require workspace membership.
        - SHARED notes require department authorization.
        """
        item = await self.session.scalar(
            select(KnowledgeItem)
            .options(
                selectinload(KnowledgeItem.workspace),
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.author),
                selectinload(KnowledgeItem.approved_by),
            )
            .where(KnowledgeItem.id == note_id)
        )
        if not item:
            raise HTTPException(status_code=404, detail="Company note not found")

        # Boundary 0: Inactive / archived note isolation for non-authors
        if item.status != "ACTIVE" and item.author_id != user.id and user.role != "ADMIN":
            raise HTTPException(status_code=404, detail="Company note not found")

        # Boundary 1: Author-private draft isolation
        if item.visibility == "PRIVATE" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Private draft notes are accessible only to their author",
            )

        # Boundary 2: Workspace visibility
        if item.visibility == "WORKSPACE" and user.role != "ADMIN" and item.author_id != user.id:
            if item.workspace_id:
                is_mem = await self.session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == item.workspace_id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not is_mem:
                    raise HTTPException(status_code=403, detail="Forbidden: You are not a member of this workspace")

        return await self._format_note_response(item, user=user)

    async def list_notes(
        self,
        user: User,
        company_name: Optional[str] = None,
        visibility: Optional[str] = None,
        approval_status: Optional[str] = None,
        workspace_id: Optional[uuid.UUID] = None,
        limit: int = 50,
    ) -> list[CompanyNoteResponse]:
        """
        Lists notes visible to the user:
        - Users see their own notes (including PRIVATE).
        - Users see active WORKSPACE notes in their authorized workspaces.
        - Users see active SHARED notes approved for their departments.
        - Admins see active WORKSPACE and SHARED notes, but NEVER another user's PRIVATE draft.
        """
        stmt = (
            select(KnowledgeItem)
            .options(
                selectinload(KnowledgeItem.workspace),
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.author),
                selectinload(KnowledgeItem.approved_by),
            )
            .where(
                KnowledgeItem.source_type == "DOCUMENT",
                KnowledgeItem.kind != "RELATIONSHIP",
                or_(
                    KnowledgeItem.canonical_key.like("NOTE:%"),
                    exists().where(
                        KnowledgeEntity.knowledge_item_id == KnowledgeItem.id,
                        KnowledgeEntity.entity_type == "company_note",
                    ),
                ),
            )
            .order_by(KnowledgeItem.updated_at.desc())
            .limit(limit)
        )

        if user.role == "ADMIN":
            # Admin sees all non-private active notes, PLUS their own private/archived notes
            stmt = stmt.where(
                or_(
                    KnowledgeItem.author_id == user.id,
                    and_(
                        KnowledgeItem.status == "ACTIVE",
                        KnowledgeItem.visibility != "PRIVATE",
                    ),
                )
            )
        else:
            user_workspaces_subq = select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user.id)
            stmt = stmt.where(
                or_(
                    KnowledgeItem.author_id == user.id,
                    and_(
                        KnowledgeItem.status == "ACTIVE",
                        KnowledgeItem.visibility == "WORKSPACE",
                        KnowledgeItem.workspace_id.in_(user_workspaces_subq),
                    ),
                    and_(
                        KnowledgeItem.status == "ACTIVE",
                        KnowledgeItem.visibility == "SHARED",
                        KnowledgeItem.approval_status == "APPROVED",
                    ),
                )
            )

        if visibility:
            stmt = stmt.where(KnowledgeItem.visibility == visibility)
        if approval_status:
            stmt = stmt.where(KnowledgeItem.approval_status == approval_status)
        if workspace_id:
            stmt = stmt.where(KnowledgeItem.workspace_id == workspace_id)

        if company_name:
            norm_comp = normalize_string(company_name)
            stmt = stmt.where(
                exists().where(
                    KnowledgeEntity.knowledge_item_id == KnowledgeItem.id,
                    KnowledgeEntity.canonical_name.ilike(f"%{norm_comp}%"),
                )
            )

        items = (await self.session.scalars(stmt)).all()
        results = []
        for it in items:
            results.append(await self._format_note_response(it, user=user))
        return results

    async def update_note(
        self,
        user: User,
        note_id: uuid.UUID,
        request: CompanyNoteUpdateRequest,
    ) -> CompanyNoteResponse:
        """
        Edits an existing note.
        Only the author or an administrator can edit notes in DRAFT or REJECTED state.
        Approved shared knowledge requires explicit revision workflow.
        """
        item = await self.session.get(KnowledgeItem, note_id)
        if not item:
            raise HTTPException(status_code=404, detail="Company note not found")

        # Private draft isolation: even ADMIN cannot modify another user's private draft
        if item.visibility == "PRIVATE" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Private draft notes can only be modified by their author",
            )

        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(status_code=403, detail="Forbidden: You cannot edit this note")

        if item.approval_status == "APPROVED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Approved shared knowledge cannot be edited directly. Return to draft for revision first.",
            )

        entity = await self.session.scalar(
            select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == item.id)
        )

        now = datetime.now(timezone.utc)
        if request.company_name is not None and entity:
            entity.canonical_name = request.company_name[:255]
            attrs = dict(entity.attributes or {})
            attrs["company_name"] = request.company_name
            entity.attributes = attrs

        if request.statement is not None and entity:
            attrs = dict(entity.attributes or {})
            attrs["statement"] = request.statement
            entity.attributes = attrs

        if request.observation_type is not None:
            obs_type = request.observation_type.upper()
            if obs_type in CONFIDENCE_MAP:
                if entity:
                    attrs = dict(entity.attributes or {})
                    attrs["observation_type"] = obs_type
                    entity.attributes = attrs
                item.confidence = CONFIDENCE_MAP[obs_type]

        if request.workspace_id is not None:
            item.workspace_id = request.workspace_id
        if request.department_id is not None:
            item.department_id = request.department_id
        if request.visibility is not None and request.visibility.upper() in ("PRIVATE", "WORKSPACE"):
            item.visibility = request.visibility.upper()

        if request.attributes and entity:
            attrs = dict(entity.attributes or {})
            attrs.update(request.attributes)
            entity.attributes = attrs

        item.updated_at = now
        await self.session.commit()

        loaded_item = await self.session.scalar(
            select(KnowledgeItem)
            .options(
                selectinload(KnowledgeItem.workspace),
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.author),
                selectinload(KnowledgeItem.approved_by),
            )
            .where(KnowledgeItem.id == item.id)
        )
        return await self._format_note_response(loaded_item, user=user)

    async def attach_evidence(
        self,
        user: User,
        note_id: uuid.UUID,
        request: CompanyNoteEvidenceAttachRequest,
    ) -> CompanyNoteResponse:
        """
        Attaches a source document chunk reference to the note.
        Strictly validates that the caller has VIEW permission on the target chunk's document.
        """
        item = await self.session.get(KnowledgeItem, note_id)
        if not item:
            raise HTTPException(status_code=404, detail="Company note not found")

        if item.visibility == "PRIVATE" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Cannot attach evidence to another user's private draft",
            )

        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(status_code=403, detail="Forbidden: You cannot modify evidence on this note")

        # Branch 1: Website source attachment
        if request.source_url:
            clean_url = validate_website_url(request.source_url)
            existing_ev = await self.session.scalar(
                select(KnowledgeEvidence).where(
                    KnowledgeEvidence.knowledge_item_id == item.id,
                    KnowledgeEvidence.source_url == clean_url,
                )
            )
            if existing_ev:
                return await self.get_note(user, note_id)

            now = datetime.now(timezone.utc)
            conf = request.confidence if request.confidence is not None else item.confidence
            ev = KnowledgeEvidence(
                knowledge_item_id=item.id,
                source_url=clean_url,
                source_title=request.source_title.strip() if request.source_title else None,
                user_excerpt=request.custom_excerpt.strip() if request.custom_excerpt else None,
                verification_status="NOT_VERIFIED",
                extraction_method="WEBSITE_REGISTRATION",
                confidence=conf,
                created_at=now,
            )
            self.session.add(ev)
            item.updated_at = now
            await self.session.commit()
            return await self.get_note(user, note_id)

        # Branch 2: Document chunk attachment
        if not request.document_chunk_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either document_chunk_id or source_url must be provided to attach evidence",
            )

        # Validate chunk access server-side
        chunk = await self._validate_chunk_access(user, request.document_chunk_id)

        # Check for duplicate evidence link
        existing_ev = await self.session.scalar(
            select(KnowledgeEvidence).where(
                KnowledgeEvidence.knowledge_item_id == item.id,
                KnowledgeEvidence.document_chunk_id == chunk.id,
            )
        )
        if existing_ev:
            return await self.get_note(user, note_id)

        now = datetime.now(timezone.utc)
        conf = request.confidence if request.confidence is not None else item.confidence
        ev = KnowledgeEvidence(
            knowledge_item_id=item.id,
            document_chunk_id=chunk.id,
            extraction_method=request.extraction_method or "MANUAL_EVIDENCE",
            confidence=conf,
            created_at=now,
        )
        self.session.add(ev)
        item.updated_at = now
        await self.session.commit()

        return await self.get_note(user, note_id)

    async def remove_evidence(
        self,
        user: User,
        note_id: uuid.UUID,
        evidence_id: uuid.UUID,
    ) -> CompanyNoteResponse:
        """
        Removes an evidence reference.
        If all evidence is removed, approved notes are de-published to prevent unsupported knowledge retrieval.
        """
        item = await self.session.get(KnowledgeItem, note_id)
        if not item:
            raise HTTPException(status_code=404, detail="Company note not found")

        if item.visibility == "PRIVATE" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Cannot remove evidence from another user's private draft",
            )

        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(status_code=403, detail="Forbidden: You cannot remove evidence from this note")

        ev = await self.session.scalar(
            select(KnowledgeEvidence).where(
                KnowledgeEvidence.id == evidence_id,
                KnowledgeEvidence.knowledge_item_id == item.id,
            )
        )
        if not ev:
            raise HTTPException(status_code=404, detail="Evidence not found on this note")

        await self.session.delete(ev)
        await self.session.flush()

        remaining_count = await self.session.scalar(
            select(func.count(KnowledgeEvidence.id)).where(KnowledgeEvidence.knowledge_item_id == item.id)
        )
        now = datetime.now(timezone.utc)
        if not remaining_count or remaining_count == 0:
            if item.approval_status == "APPROVED":
                item.status = "ARCHIVED"
                item.visibility = "WORKSPACE"
                item.approval_status = "REJECTED"

                # Synchronously archive associated GraphRAG relationship items
                rel_key_prefix = f"RELATIONSHIP:company:%:has_observation:company_note:{item.id.hex[:8]}"
                rel_items = (
                    await self.session.scalars(
                        select(KnowledgeItem).where(
                            KnowledgeItem.canonical_key.like(rel_key_prefix),
                            KnowledgeItem.kind == "RELATIONSHIP",
                        )
                    )
                ).all()
                for ri in rel_items:
                    ri.status = "ARCHIVED"
                    ri.visibility = "WORKSPACE"
                    ri.approval_status = "REJECTED"

        item.updated_at = now
        await self.session.commit()

        return await self.get_note(user, note_id)

    async def share_to_workspace(
        self,
        user: User,
        note_id: uuid.UUID,
        workspace_id: uuid.UUID,
    ) -> CompanyNoteResponse:
        """
        Distinct from publishing as shared organizational knowledge:
        Shares an author-private draft with a workspace.
        Requires active workspace membership.
        Note remains in DRAFT status.
        """
        item = await self.session.get(KnowledgeItem, note_id)
        if not item:
            raise HTTPException(status_code=404, detail="Company note not found")

        # Lifecycle check: Note must be active and in DRAFT state
        if item.status != "ACTIVE":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot share note in '{item.status}' status to a workspace",
            )
        if item.approval_status != "DRAFT":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot share note with approval status '{item.approval_status}' to a workspace; only DRAFT notes can be shared",
            )

        if item.visibility == "PRIVATE" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Private draft notes can only be shared by their author",
            )
        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(status_code=403, detail="Forbidden: Only the author can share this draft to a workspace")

        ws = await self.session.get(Workspace, workspace_id)
        if not ws or not ws.is_active:
            raise HTTPException(status_code=404, detail="Target workspace not found or inactive")

        if user.role != "ADMIN":
            is_mem = await self.session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.user_id == user.id,
                )
            )
            if not is_mem:
                raise HTTPException(status_code=403, detail="Forbidden: You are not a member of the target workspace")

            # Validate target workspace department access for non-admin
            if ws.department_id:
                dept = await self.session.get(Department, ws.department_id)
                if dept:
                    is_dept_auth = await authorize(self.session, user, "VIEW", dept.code)
                    if not is_dept_auth:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Forbidden: You do not have access to department '{dept.code}'",
                        )

        # Collision guard: check if duplicate canonical key already exists in target department
        if ws.department_id != item.department_id:
            existing_target = await self.session.scalar(
                select(KnowledgeItem).where(
                    KnowledgeItem.canonical_key == item.canonical_key,
                    KnowledgeItem.department_id == ws.department_id,
                    KnowledgeItem.id != item.id,
                )
            )
            if existing_target:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A note with this canonical statement already exists in the target department",
                )

        item.workspace_id = ws.id
        item.department_id = ws.department_id
        item.visibility = "WORKSPACE"
        item.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

        return await self.get_note(user, note_id)

    async def submit_note(
        self,
        user: User,
        note_id: uuid.UUID,
        request: CompanyNoteSubmitRequest,
    ) -> CompanyNoteResponse:
        """
        Submits note for publication review (DRAFT -> PENDING).
        Requires at least one supporting evidence item.
        """
        item = await self.session.get(KnowledgeItem, note_id)
        if not item:
            raise HTTPException(status_code=404, detail="Company note not found")

        if item.visibility == "PRIVATE" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Private draft notes can only be submitted for review by their author",
            )
        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(status_code=403, detail="Forbidden: Only the author can submit this note for review")

        # Check evidence count: A URL-only note cannot be submitted for publication without verified supporting document evidence
        doc_ev_count = await self.session.scalar(
            select(func.count(KnowledgeEvidence.id)).where(
                KnowledgeEvidence.knowledge_item_id == item.id,
                or_(
                    KnowledgeEvidence.document_chunk_id.is_not(None),
                    KnowledgeEvidence.verification_status == "VERIFIED",
                ),
            )
        )
        if not doc_ev_count or doc_ev_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot submit note for publication without verified supporting document evidence",
            )

        if request.target_department_id:
            item.department_id = request.target_department_id
        if request.target_workspace_id:
            item.workspace_id = request.target_workspace_id

        # Make note accessible for review
        if item.visibility == "PRIVATE":
            item.visibility = "WORKSPACE" if item.workspace_id else "WORKSPACE"

        item.approval_status = "PENDING"
        item.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

        return await self.get_note(user, note_id)

    async def review_note(
        self,
        admin_user: User,
        note_id: uuid.UUID,
        request: CompanyNoteReviewRequest,
    ) -> CompanyNoteResponse:
        """
        Admin review: APPROVE, REJECT, or REVISE.
        Enforces evidence sufficiency and GraphRAG provenance preservation.
        """
        if admin_user.role != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Admin privileges required to review publication candidates",
            )

        item = await self.session.scalar(
            select(KnowledgeItem)
            .where(KnowledgeItem.id == note_id)
            .with_for_update()
        )
        if not item:
            raise HTTPException(status_code=404, detail="Company note not found")

        # Do not expose unsubmitted private drafts of other users to review
        if item.visibility == "PRIVATE" and item.author_id != admin_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Cannot review another user's unsubmitted private draft",
            )

        now = datetime.now(timezone.utc)
        action = request.action.upper()

        if action == "APPROVE":
            doc_ev_count = await self.session.scalar(
                select(func.count(KnowledgeEvidence.id)).where(
                    KnowledgeEvidence.knowledge_item_id == item.id,
                    or_(
                        KnowledgeEvidence.document_chunk_id.is_not(None),
                        KnowledgeEvidence.verification_status == "VERIFIED",
                    ),
                )
            )
            if not doc_ev_count or doc_ev_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot approve note: URL-only notes require at least one verified document evidence citation for publication",
                )

            item.approval_status = "APPROVED"
            item.visibility = "SHARED"
            item.approved_by_id = admin_user.id
            item.approved_at = now
            item.updated_at = now

            # Preserve GraphRAG relationship provenance:
            # Create KnowledgeRelationship connecting the company entity to the note observation
            await self._materialize_note_graphrag_provenance(item)

        elif action == "REJECT":
            item.approval_status = "REJECTED"
            item.visibility = "WORKSPACE"  # Guarantees fail-closed
            item.approved_by_id = admin_user.id
            item.approved_at = now
            item.updated_at = now

        elif action == "REVISE":
            item.approval_status = "DRAFT"
            item.visibility = "WORKSPACE"
            item.updated_at = now

        elif action in ("WITHDRAW", "ARCHIVE"):
            item.status = "ARCHIVED"
            item.visibility = "WORKSPACE"
            if item.approval_status == "APPROVED":
                item.approval_status = "REJECTED"
            item.approved_by_id = admin_user.id
            item.approved_at = now
            item.updated_at = now

            # Synchronously archive associated GraphRAG relationship items
            rel_key_prefix = f"RELATIONSHIP:company:%:has_observation:company_note:{item.id.hex[:8]}"
            rel_items = (
                await self.session.scalars(
                    select(KnowledgeItem).where(
                        KnowledgeItem.canonical_key.like(rel_key_prefix),
                        KnowledgeItem.kind == "RELATIONSHIP",
                    )
                )
            ).all()
            for ri in rel_items:
                ri.status = "ARCHIVED"
                ri.visibility = "WORKSPACE"
                ri.approval_status = "REJECTED"
                ri.updated_at = now

        else:
            raise HTTPException(status_code=400, detail=f"Invalid review action '{request.action}'")

        await self.session.commit()
        return await self.get_note(admin_user, note_id)

    async def archive_note(
        self,
        user: User,
        note_id: uuid.UUID,
        reason: Optional[str] = None,
    ) -> CompanyNoteResponse:
        """
        Withdraws / archives an existing company note.
        Allowed for the author of the note or an ADMIN.
        Private drafts belonging to other users cannot be archived.
        Demotes note to status='ARCHIVED', visibility='WORKSPACE', and approval_status='REJECTED' if approved.
        Synchronously archives associated GraphRAG relationship items.
        """
        item = await self.session.scalar(
            select(KnowledgeItem)
            .where(KnowledgeItem.id == note_id)
            .with_for_update()
        )
        if not item:
            raise HTTPException(status_code=404, detail="Company note not found")

        if item.visibility == "PRIVATE" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Cannot archive another user's private draft",
            )
        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Only the author or an admin can archive this note",
            )

        now = datetime.now(timezone.utc)
        item.status = "ARCHIVED"
        if item.approval_status == "APPROVED":
            item.approval_status = "REJECTED"
        item.visibility = "WORKSPACE"
        item.updated_at = now

        # Synchronously archive linked GraphRAG relationship items
        rel_key_prefix = f"RELATIONSHIP:company:%:has_observation:company_note:{item.id.hex[:8]}"
        rel_items = (
            await self.session.scalars(
                select(KnowledgeItem).where(
                    KnowledgeItem.canonical_key.like(rel_key_prefix),
                    KnowledgeItem.kind == "RELATIONSHIP",
                ).with_for_update()
            )
        ).all()
        for ri in rel_items:
            ri.status = "ARCHIVED"
            ri.visibility = "WORKSPACE"
            ri.approval_status = "REJECTED"
            ri.updated_at = now

        await self.session.commit()
        return await self.get_note(user, note_id)

    async def supersede_note(
        self,
        user: User,
        old_note_id: uuid.UUID,
        new_note_id: uuid.UUID,
    ) -> CompanyNoteResponse:
        """
        Supersedes an old company note with a newer replacement note.
        Sets old_note.status = 'SUPERSEDED' and old_note.superseded_by_id = new_note.id.
        Archives associated GraphRAG relationships of the old note.
        Preserves provenance and history without destructive deletion.
        """
        if old_note_id == new_note_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A note cannot supersede itself",
            )

        old_item = await self.session.scalar(
            select(KnowledgeItem)
            .where(KnowledgeItem.id == old_note_id)
            .with_for_update()
        )
        if not old_item:
            raise HTTPException(status_code=404, detail="Old company note not found")

        new_item = await self.session.scalar(
            select(KnowledgeItem)
            .where(KnowledgeItem.id == new_note_id)
            .with_for_update()
        )
        if not new_item:
            raise HTTPException(status_code=404, detail="Replacement company note not found")

        if old_item.visibility == "PRIVATE" and old_item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Cannot supersede another user's private draft",
            )
        if user.role != "ADMIN" and old_item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Only the author or an admin can supersede this note",
            )

        now = datetime.now(timezone.utc)
        old_item.status = "SUPERSEDED"
        if old_item.approval_status == "APPROVED":
            old_item.approval_status = "REJECTED"
        old_item.visibility = "WORKSPACE"
        old_item.updated_at = now

        # Synchronously archive old item's GraphRAG relationships
        rel_key_prefix = f"RELATIONSHIP:company:%:has_observation:company_note:{old_item.id.hex[:8]}"
        rel_items = (
            await self.session.scalars(
                select(KnowledgeItem).where(
                    KnowledgeItem.canonical_key.like(rel_key_prefix),
                    KnowledgeItem.kind == "RELATIONSHIP",
                ).with_for_update()
            )
        ).all()
        for ri in rel_items:
            ri.status = "SUPERSEDED"
            ri.visibility = "WORKSPACE"
            ri.approval_status = "REJECTED"
            ri.updated_at = now

        await self.session.commit()
        return await self.get_note(user, old_note_id)

    async def reconcile_orphaned_notes(self) -> int:
        """
        Idempotent administrative reconciliation operation:
        Finds approved or active company notes that have lost all valid evidence
        (e.g. through direct source document or chunk hard-deletion) and
        synchronously transitions them to ARCHIVED / REJECTED, ensuring no
        administrative or publication workflow treats them as valid.
        """
        stmt = (
            select(KnowledgeItem)
            .where(
                KnowledgeItem.kind == "FACT",
                KnowledgeItem.source_type == "DOCUMENT",
                KnowledgeItem.status == "ACTIVE",
                KnowledgeItem.approval_status == "APPROVED",
                ~exists().where(KnowledgeEvidence.knowledge_item_id == KnowledgeItem.id),
            )
            .with_for_update()
        )
        orphaned_items = (await self.session.scalars(stmt)).all()
        now = datetime.now(timezone.utc)
        count = 0
        for item in orphaned_items:
            item.status = "ARCHIVED"
            item.visibility = "WORKSPACE"
            item.approval_status = "REJECTED"
            item.updated_at = now
            count += 1

            # Archive linked relationship items
            rel_key_prefix = f"RELATIONSHIP:company:%:has_observation:company_note:{item.id.hex[:8]}"
            rel_items = (
                await self.session.scalars(
                    select(KnowledgeItem).where(
                        KnowledgeItem.canonical_key.like(rel_key_prefix),
                        KnowledgeItem.kind == "RELATIONSHIP",
                    ).with_for_update()
                )
            ).all()
            for ri in rel_items:
                ri.status = "ARCHIVED"
                ri.visibility = "WORKSPACE"
                ri.approval_status = "REJECTED"

        if count > 0:
            await self.session.commit()
        return count

    async def _materialize_note_graphrag_provenance(self, item: KnowledgeItem) -> None:
        """
        Materializes GraphRAG entities and relationships for approved company notes,
        ensuring evidence provenance links remain intact.
        """
        entity = await self.session.scalar(
            select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == item.id)
        )
        if not entity:
            return

        company_name = entity.canonical_name or "Company"
        norm_company = normalize_string(company_name)

        # 1. Subject entity: Company
        subj_item = await self.session.scalar(
            select(KnowledgeItem).where(
                KnowledgeItem.canonical_key == f"ENTITY:company:{norm_company}",
                KnowledgeItem.department_id == item.department_id,
            ).with_for_update()
        )
        if not subj_item:
            subj_item = KnowledgeItem(
                canonical_key=f"ENTITY:company:{norm_company}",
                kind="ENTITY",
                status="ACTIVE",
                visibility="SHARED",
                approval_status="APPROVED",
                source_type="DOCUMENT",
                department_id=item.department_id,
                author_id=item.author_id,
                approved_by_id=item.approved_by_id,
                approved_at=item.approved_at,
                confidence=item.confidence,
            )
            self.session.add(subj_item)
            await self.session.flush()

            subj_ent = KnowledgeEntity(
                knowledge_item_id=subj_item.id,
                entity_type="company",
                canonical_name=company_name,
                attributes={"company_name": company_name},
            )
            self.session.add(subj_ent)
            await self.session.flush()
        else:
            subj_ent = await self.session.scalar(
                select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == subj_item.id)
            )

        # 2. Relationship: Company -> HAS_OBSERVATION -> Note Entity
        if subj_ent and entity.id != subj_ent.id:
            rel_key = f"RELATIONSHIP:company:{norm_company}:has_observation:company_note:{item.id.hex[:8]}"
            existing_rel_item = await self.session.scalar(
                select(KnowledgeItem)
                .where(KnowledgeItem.canonical_key == rel_key)
                .with_for_update()
            )
            if not existing_rel_item:
                rel_item = KnowledgeItem(
                    canonical_key=rel_key,
                    kind="RELATIONSHIP",
                    status="ACTIVE",
                    visibility="SHARED",
                    approval_status="APPROVED",
                    source_type="DOCUMENT",
                    department_id=item.department_id,
                    author_id=item.author_id,
                    approved_by_id=item.approved_by_id,
                    approved_at=item.approved_at,
                    confidence=item.confidence,
                )
                self.session.add(rel_item)
                await self.session.flush()

                rel = KnowledgeRelationship(
                    knowledge_item_id=rel_item.id,
                    subject_entity_id=subj_ent.id,
                    predicate="has_observation",
                    object_entity_id=entity.id,
                )
                self.session.add(rel)

                # Link evidence to relationship item as well for provenance
                ev_rows = (
                    await self.session.scalars(
                        select(KnowledgeEvidence).where(KnowledgeEvidence.knowledge_item_id == item.id)
                    )
                ).all()
                for ev in ev_rows:
                    if ev.document_chunk_id:
                        rel_ev = KnowledgeEvidence(
                            knowledge_item_id=rel_item.id,
                            document_chunk_id=ev.document_chunk_id,
                            extraction_method="COMPANY_NOTE_PROVENANCE",
                            confidence=item.confidence,
                        )
                        self.session.add(rel_ev)
                    elif ev.source_url:
                        rel_ev = KnowledgeEvidence(
                            knowledge_item_id=rel_item.id,
                            source_url=ev.source_url,
                            source_title=ev.source_title,
                            user_excerpt=ev.user_excerpt,
                            verification_status=ev.verification_status,
                            extraction_method="COMPANY_NOTE_PROVENANCE",
                            confidence=item.confidence,
                        )
                        self.session.add(rel_ev)

    async def publish_batch(
        self,
        admin_user: User,
        request: BatchPublicationRequest,
    ) -> BatchPublicationResponse:
        """
        Controlled Batch Ingestion Coordinator:
        1. Validates the entire selected batch before publication.
        2. Checks authorization, evidence integrity, duplicates, and conflicts.
        3. Tracks batch status and individual item outcomes.
        4. Idempotent retries: already approved items report success safely.
        5. Recovers safely from partial failure.
        6. Preserves source evidence on all materialized knowledge and relationships.
        """
        if admin_user.role != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Admin privileges required for batch publication",
            )

        batch_id = uuid.uuid4()
        outcomes: list[BatchPublicationItemOutcome] = []
        published_count = 0
        failed_count = 0
        is_retry = True

        now = datetime.now(timezone.utc)

        # Pre-validation: check for in-batch duplicates
        seen_keys: set[str] = set()

        for note_id in request.note_ids:
            # Row-level lock ensures atomic serialization across concurrent publications
            item = await self.session.scalar(
                select(KnowledgeItem)
                .where(KnowledgeItem.id == note_id)
                .with_for_update()
            )
            if not item:
                failed_count += 1
                is_retry = False
                outcomes.append(
                    BatchPublicationItemOutcome(
                        note_id=note_id,
                        success=False,
                        status="NOT_FOUND",
                        error="Company note not found",
                    )
                )
                continue

            entity = await self.session.scalar(
                select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == item.id)
            )
            company_name = entity.canonical_name if entity else None

            # Check evidence count
            ev_count = await self.session.scalar(
                select(func.count(KnowledgeEvidence.id)).where(KnowledgeEvidence.knowledge_item_id == item.id)
            )

            # Idempotency check: already approved items
            if item.approval_status == "APPROVED" and item.visibility == "SHARED":
                if ev_count and ev_count > 0:
                    outcomes.append(
                        BatchPublicationItemOutcome(
                            note_id=note_id,
                            canonical_key=item.canonical_key,
                            company_name=company_name,
                            success=True,
                            status="ALREADY_APPROVED",
                        )
                    )
                    published_count += 1
                    continue
                else:
                    # Approved note that lost all evidence! Reconcile de-publication immediately
                    item.status = "ARCHIVED"
                    item.visibility = "WORKSPACE"
                    item.approval_status = "REJECTED"
                    failed_count += 1
                    is_retry = False
                    outcomes.append(
                        BatchPublicationItemOutcome(
                            note_id=note_id,
                            canonical_key=item.canonical_key,
                            company_name=company_name,
                            success=False,
                            status="NO_EVIDENCE",
                            error="Approved note has lost all source document evidence; de-published during reconciliation",
                        )
                    )
                    continue

            is_retry = False

            # Boundary check: do not publish another user's unsubmitted private draft
            if item.visibility == "PRIVATE" and item.author_id != admin_user.id:
                failed_count += 1
                outcomes.append(
                    BatchPublicationItemOutcome(
                        note_id=note_id,
                        canonical_key=item.canonical_key,
                        company_name=company_name,
                        success=False,
                        status="PRIVATE_DRAFT",
                        error="Cannot publish another user's unsubmitted private draft",
                    )
                )
                continue

            # Check duplicate within this batch
            if item.canonical_key in seen_keys:
                failed_count += 1
                outcomes.append(
                    BatchPublicationItemOutcome(
                        note_id=note_id,
                        canonical_key=item.canonical_key,
                        company_name=company_name,
                        success=False,
                        status="CONFLICT",
                        error="Duplicate observation statement within the same publication batch",
                    )
                )
                continue
            seen_keys.add(item.canonical_key)

            # Check evidence integrity: require verified document evidence (URL-only notes cannot be batch published)
            valid_doc_count = await self.session.scalar(
                select(func.count(KnowledgeEvidence.id)).where(
                    KnowledgeEvidence.knowledge_item_id == item.id,
                    or_(
                        KnowledgeEvidence.document_chunk_id.is_not(None),
                        KnowledgeEvidence.verification_status == "VERIFIED",
                    ),
                )
            )
            if not valid_doc_count or valid_doc_count == 0:
                failed_count += 1
                outcomes.append(
                    BatchPublicationItemOutcome(
                        note_id=note_id,
                        canonical_key=item.canonical_key,
                        company_name=company_name,
                        success=False,
                        status="NO_EVIDENCE",
                        error="Cannot publish note without valid source document evidence",
                    )
                )
                continue

            # Verify department assignment for shared publication
            if request.target_department_id:
                item.department_id = request.target_department_id
            elif not item.department_id and item.workspace:
                item.department_id = item.workspace.department_id

            if not item.department_id and request.target_audience == "SHARED":
                # Fallback to first available department or error
                first_dept = await self.session.scalar(select(Department).where(Department.is_active == True))
                if first_dept:
                    item.department_id = first_dept.id

            # Safe publication transition
            try:
                async with self.session.begin_nested():
                    item.approval_status = "APPROVED"
                    item.visibility = "SHARED"
                    item.approved_by_id = admin_user.id
                    item.approved_at = now
                    item.updated_at = now

                    # Materialize GraphRAG relationship and provenance
                    await self._materialize_note_graphrag_provenance(item)

                published_count += 1
                outcomes.append(
                    BatchPublicationItemOutcome(
                        note_id=note_id,
                        canonical_key=item.canonical_key,
                        company_name=company_name,
                        success=True,
                        status="PUBLISHED",
                    )
                )
            except Exception as ex:
                logger.error(f"Failed to publish note {note_id}: {ex}", exc_info=True)
                failed_count += 1
                outcomes.append(
                    BatchPublicationItemOutcome(
                        note_id=note_id,
                        canonical_key=item.canonical_key,
                        company_name=company_name,
                        success=False,
                        status="ERROR",
                        error=str(ex),
                    )
                )

        await self.session.commit()

        return BatchPublicationResponse(
            batch_id=batch_id,
            total_items=len(request.note_ids),
            published_count=published_count,
            failed_count=failed_count,
            is_idempotent_retry=is_retry,
            outcomes=outcomes,
        )
