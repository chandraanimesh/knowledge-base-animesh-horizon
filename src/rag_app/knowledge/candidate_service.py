import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import String, func, or_, select, exists
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.db.models import (
    ChatMessage,
    Conversation,
    Department,
    KnowledgeEntity,
    KnowledgeEvidence,
    KnowledgeItem,
    User,
    Workspace,
    WorkspaceMember,
)
from rag_app.knowledge.materializer import normalize_string
from rag_app.schemas.knowledge import (
    CandidateCreateRequest,
    KnowledgeCandidateResponse,
    KnowledgeEvidenceSummary,
)
from rag_app.schemas.rag import RetrievedChunk
from rag_app.security.authorization import authorize

logger = logging.getLogger(__name__)


class KnowledgeCandidateService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _format_candidate_response(self, item: KnowledgeItem) -> KnowledgeCandidateResponse:
        # Load relationships if needed
        workspace_name = item.workspace.name if item.workspace else None
        department_code = item.department.code if item.department else None
        author_email = item.author.email if item.author else None
        approved_by_email = item.approved_by.email if item.approved_by else None

        # Fetch statement from linked entity attributes or fallback to canonical_key
        entity = await self.session.scalar(
            select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == item.id)
        )
        statement = ""
        if entity:
            statement = (entity.attributes or {}).get("statement") or entity.canonical_name or ""
        if not statement:
            statement = item.canonical_key

        # Fetch evidence links
        ev_rows = (
            await self.session.scalars(
                select(KnowledgeEvidence)
                .where(KnowledgeEvidence.knowledge_item_id == item.id)
                .order_by(KnowledgeEvidence.created_at.asc())
            )
        ).all()

        evidence_summaries = []
        for ev in ev_rows:
            src_type = "UNKNOWN"
            if ev.document_chunk_id is not None:
                src_type = "DOCUMENT"
            elif ev.research_evidence_id is not None:
                src_type = "RESEARCH"
            elif ev.conversation_id is not None:
                src_type = "CONVERSATION"

            evidence_summaries.append(
                KnowledgeEvidenceSummary(
                    id=ev.id,
                    source_type=src_type,
                    conversation_id=ev.conversation_id,
                    chat_message_id=ev.chat_message_id,
                    document_chunk_id=ev.document_chunk_id,
                    research_evidence_id=ev.research_evidence_id,
                    extraction_method=ev.extraction_method,
                    confidence=ev.confidence,
                    created_at=ev.created_at,
                )
            )

        return KnowledgeCandidateResponse(
            id=item.id,
            canonical_key=item.canonical_key,
            statement=statement,
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
            evidence_count=len(evidence_summaries),
            evidence=evidence_summaries,
        )

    async def _validate_conversation_and_message_access(
        self,
        user: User,
        conversation_id: uuid.UUID,
        chat_message_id: uuid.UUID,
    ) -> tuple[Conversation, ChatMessage, Optional[Workspace], Optional[Department]]:
        """
        Validates ownership and access to the conversation and chat message.
        Guarantees server-side verification:
        1. User must have access to conversation (either owner or ADMIN).
        2. If conversation has workspace, user must have department access.
        3. chat_message_id must belong to conversation_id.
        """
        conv = await self.session.get(Conversation, conversation_id)
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source conversation not found",
            )

        # Personal direct conversation threads (workspace_id is None) are strictly private to their owner, even for ADMINs
        if conv.workspace_id is None and conv.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Personal conversation threads without a workspace are strictly private to their owner",
            )

        if user.role != "ADMIN" and conv.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: You are not authorized to access this conversation",
            )

        ws: Optional[Workspace] = None
        dept: Optional[Department] = None
        if conv.workspace_id is not None:
            ws = await self.session.scalar(
                select(Workspace)
                .options(selectinload(Workspace.department))
                .where(Workspace.id == conv.workspace_id, Workspace.is_active == True)
            )
            if not ws:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workspace associated with conversation not found or inactive",
                )
            dept = ws.department
            if user.role != "ADMIN" and dept:
                is_auth = await authorize(self.session, user, "VIEW", dept.code)
                if not is_auth:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Forbidden: You do not have access to department '{dept.code}'",
                    )

        # Validate chat_message_id
        msg = await self.session.get(ChatMessage, chat_message_id)
        if not msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source chat message not found",
            )

        if msg.conversation_id != conv.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provided chat_message_id does not belong to the supplied conversation_id",
            )

        return conv, msg, ws, dept

    async def create_candidate(
        self,
        user: User,
        request: CandidateCreateRequest,
    ) -> KnowledgeCandidateResponse:
        """
        Extracts and materializes an unpublished knowledge candidate from an authorized conversation.
        Always sets visibility='WORKSPACE' and approval_status='DRAFT' by default.
        """
        conv, msg, ws, dept = await self._validate_conversation_and_message_access(
            user=user,
            conversation_id=request.conversation_id,
            chat_message_id=request.chat_message_id,
        )

        target_ws_id = ws.id if ws else None
        target_dept_id = dept.id if dept else None

        # Build deterministic canonical key
        clean_stmt = normalize_string(request.statement)
        topic_norm = normalize_string(request.topic) or "general"
        stmt_hash = hashlib.sha256(clean_stmt.encode("utf-8")).hexdigest()[:16]
        canonical_key = f"{request.kind.upper()}:{topic_norm}:{stmt_hash}"

        # Check for pre-existing knowledge item for idempotency
        stmt_existing = select(KnowledgeItem).where(
            KnowledgeItem.canonical_key == canonical_key,
            KnowledgeItem.department_id == target_dept_id,
        )
        existing_item = await self.session.scalar(stmt_existing)

        if existing_item:
            # Check if evidence link already exists
            existing_ev = await self.session.scalar(
                select(KnowledgeEvidence).where(
                    KnowledgeEvidence.knowledge_item_id == existing_item.id,
                    KnowledgeEvidence.conversation_id == conv.id,
                    KnowledgeEvidence.chat_message_id == msg.id,
                )
            )
            if not existing_ev:
                new_ev = KnowledgeEvidence(
                    knowledge_item_id=existing_item.id,
                    conversation_id=conv.id,
                    chat_message_id=msg.id,
                    extraction_method="CONVERSATION_EXTRACTOR",
                    confidence=request.confidence,
                )
                self.session.add(new_ev)
                await self.session.commit()
            return await self._format_candidate_response(existing_item)

        # Create new KnowledgeItem
        now = datetime.now(timezone.utc)
        item = KnowledgeItem(
            canonical_key=canonical_key,
            kind=request.kind.upper(),
            status="ACTIVE",
            visibility="WORKSPACE",  # Invariant: Never automatically SHARED
            approval_status="DRAFT",  # Invariant: Never automatically APPROVED
            source_type="CONVERSATION",
            workspace_id=target_ws_id,
            department_id=target_dept_id,
            author_id=user.id,
            confidence=request.confidence,
            created_at=now,
            updated_at=now,
        )
        self.session.add(item)
        await self.session.flush()

        # Create linked KnowledgeEntity storing the verbatim business statement
        entity = KnowledgeEntity(
            knowledge_item_id=item.id,
            entity_type=topic_norm,
            canonical_name=clean_stmt[:255],
            attributes={
                "statement": request.statement,
                **request.attributes,
            },
        )
        self.session.add(entity)

        # Create linked KnowledgeEvidence
        evidence = KnowledgeEvidence(
            knowledge_item_id=item.id,
            conversation_id=conv.id,
            chat_message_id=msg.id,
            extraction_method="CONVERSATION_EXTRACTOR",
            confidence=request.confidence,
            created_at=now,
        )
        self.session.add(evidence)

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
        return await self._format_candidate_response(loaded_item)

    async def submit_candidate(
        self,
        user: User,
        candidate_id: uuid.UUID,
    ) -> KnowledgeCandidateResponse:
        """
        Submits a candidate for publication review: transitions status from DRAFT or REJECTED to PENDING.
        """
        item = await self.session.get(KnowledgeItem, candidate_id)
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge candidate not found")

        # Verify caller has right to submit: author or ADMIN
        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(
                status_code=403,
                detail="Forbidden: Only the author or an administrator can submit this candidate for review",
            )

        if item.approval_status == "APPROVED":
            return await self._format_candidate_response(item)

        # Require valid supporting evidence
        ev_count = await self.session.scalar(
            select(func.count(KnowledgeEvidence.id)).where(KnowledgeEvidence.knowledge_item_id == item.id)
        )
        if not ev_count or ev_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot submit candidate without supporting evidence",
            )

        item.approval_status = "PENDING"
        item.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

        return await self._format_candidate_response(item)

    async def approve_candidate(
        self,
        user: User,
        candidate_id: uuid.UUID,
    ) -> KnowledgeCandidateResponse:
        """
        Approves publication of candidate: sets approval_status='APPROVED', visibility='SHARED',
        records approver and timestamp.
        Strict server-side check: caller MUST be global ADMIN.
        """
        if user.role != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Admin privileges required for organization-wide shared publication",
            )

        item = await self.session.get(KnowledgeItem, candidate_id)
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge candidate not found")

        # Invariant: Must have valid supporting evidence
        ev_count = await self.session.scalar(
            select(func.count(KnowledgeEvidence.id)).where(KnowledgeEvidence.knowledge_item_id == item.id)
        )
        if not ev_count or ev_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot approve knowledge candidate without valid supporting evidence",
            )

        # State transition: Must be DRAFT or PENDING
        if item.approval_status not in ("PENDING", "DRAFT"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot approve candidate in status '{item.approval_status}'",
            )

        now = datetime.now(timezone.utc)
        item.approval_status = "APPROVED"
        item.visibility = "SHARED"
        item.approved_by_id = user.id
        item.approved_at = now
        item.updated_at = now

        await self.session.commit()
        return await self._format_candidate_response(item)

    async def reject_candidate(
        self,
        user: User,
        candidate_id: uuid.UUID,
        reason: Optional[str] = None,
    ) -> KnowledgeCandidateResponse:
        """
        Rejects publication: sets approval_status='REJECTED', visibility='WORKSPACE'.
        Requires global ADMIN authorization.
        """
        if user.role != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Admin privileges required to reject publication candidate",
            )

        item = await self.session.get(KnowledgeItem, candidate_id)
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge candidate not found")

        now = datetime.now(timezone.utc)
        item.approval_status = "REJECTED"
        item.visibility = "WORKSPACE"  # Guarantees fail-closed
        item.approved_by_id = user.id
        item.approved_at = now
        item.updated_at = now

        await self.session.commit()
        return await self._format_candidate_response(item)

    async def return_to_draft(
        self,
        user: User,
        candidate_id: uuid.UUID,
    ) -> KnowledgeCandidateResponse:
        """
        Returns a rejected or pending candidate back to DRAFT for revision.
        """
        item = await self.session.get(KnowledgeItem, candidate_id)
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge candidate not found")

        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Only the author or an administrator can revise this candidate",
            )

        now = datetime.now(timezone.utc)
        item.approval_status = "DRAFT"
        item.visibility = "WORKSPACE"
        item.updated_at = now
        await self.session.commit()
        return await self._format_candidate_response(item)

    async def withdraw_evidence(
        self,
        user: User,
        candidate_id: uuid.UUID,
        evidence_id: uuid.UUID,
    ) -> KnowledgeCandidateResponse:
        """
        Withdraws/deletes a supporting evidence link.
        If all evidence is removed, automatically de-activates and de-publishes the knowledge item
        so that unsupported knowledge cannot remain retrievable as an approved shared fact.
        """
        item = await self.session.get(KnowledgeItem, candidate_id)
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge candidate not found")

        if user.role != "ADMIN" and item.author_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Only the author or an administrator can withdraw evidence",
            )

        ev = await self.session.scalar(
            select(KnowledgeEvidence).where(
                KnowledgeEvidence.id == evidence_id,
                KnowledgeEvidence.knowledge_item_id == item.id,
            )
        )
        if not ev:
            raise HTTPException(status_code=404, detail="Evidence not found for this candidate")

        await self.session.delete(ev)
        await self.session.flush()

        # Check remaining evidence count
        remaining = await self.session.scalar(
            select(func.count(KnowledgeEvidence.id)).where(KnowledgeEvidence.knowledge_item_id == item.id)
        )
        if not remaining or remaining == 0:
            # Unsupported knowledge cannot remain approved or shared
            item.status = "ARCHIVED"
            item.visibility = "WORKSPACE"
            item.approval_status = "REJECTED"
            item.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        return await self._format_candidate_response(item)

    async def get_candidate(
        self,
        user: User,
        candidate_id: uuid.UUID,
    ) -> KnowledgeCandidateResponse:
        item = await self.session.scalar(
            select(KnowledgeItem)
            .options(
                selectinload(KnowledgeItem.workspace),
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.author),
                selectinload(KnowledgeItem.approved_by),
            )
            .where(KnowledgeItem.id == candidate_id)
        )
        if not item:
            raise HTTPException(status_code=404, detail="Knowledge candidate not found")

        # Authorization check: ADMIN or author or workspace member
        if user.role != "ADMIN" and item.author_id != user.id:
            if item.workspace_id:
                is_member = await self.session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == item.workspace_id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not is_member:
                    raise HTTPException(status_code=403, detail="Forbidden: You cannot view this candidate")
            else:
                raise HTTPException(status_code=403, detail="Forbidden: You cannot view this candidate")

        return await self._format_candidate_response(item)

    async def list_candidates(
        self,
        user: User,
        status: Optional[str] = None,
        visibility: Optional[str] = None,
        approval_status: Optional[str] = None,
        workspace_id: Optional[uuid.UUID] = None,
        limit: int = 50,
    ) -> list[KnowledgeCandidateResponse]:
        stmt = (
            select(KnowledgeItem)
            .options(
                selectinload(KnowledgeItem.workspace),
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.author),
                selectinload(KnowledgeItem.approved_by),
            )
            .order_by(KnowledgeItem.updated_at.desc())
            .limit(limit)
        )

        if user.role != "ADMIN":
            # Filter to items authored by user or in workspaces user is a member of
            user_workspaces_subq = select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user.id)
            stmt = stmt.where(
                or_(
                    KnowledgeItem.author_id == user.id,
                    KnowledgeItem.workspace_id.in_(user_workspaces_subq),
                )
            )

        if status:
            stmt = stmt.where(KnowledgeItem.status == status)
        if visibility:
            stmt = stmt.where(KnowledgeItem.visibility == visibility)
        if approval_status:
            stmt = stmt.where(KnowledgeItem.approval_status == approval_status)
        if workspace_id:
            stmt = stmt.where(KnowledgeItem.workspace_id == workspace_id)

        items = (await self.session.scalars(stmt)).all()
        results = []
        for it in items:
            results.append(await self._format_candidate_response(it))
        return results

    async def get_approved_shared_knowledge(
        self,
        allowed_departments: Optional[list[uuid.UUID]] = None,
        limit: int = 50,
    ) -> list[KnowledgeCandidateResponse]:
        """
        Enforces strict retrieval boundaries for organization-wide shared knowledge:
        - visibility == 'SHARED'
        - approval_status == 'APPROVED'
        - status == 'ACTIVE'
        - Evidence MUST exist (unsupported facts are never returned)
        - department_id is non-null and within allowed_departments (if specified)
        """
        if allowed_departments is not None and len(allowed_departments) == 0:
            return []

        stmt = (
            select(KnowledgeItem)
            .options(
                selectinload(KnowledgeItem.workspace),
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.author),
                selectinload(KnowledgeItem.approved_by),
            )
            .where(
                KnowledgeItem.visibility == "SHARED",
                KnowledgeItem.approval_status == "APPROVED",
                KnowledgeItem.status == "ACTIVE",
                KnowledgeItem.department_id.is_not(None),
                exists().where(KnowledgeEvidence.knowledge_item_id == KnowledgeItem.id),
            )
            .order_by(KnowledgeItem.updated_at.desc())
            .limit(limit)
        )
        if allowed_departments is not None:
            stmt = stmt.where(KnowledgeItem.department_id.in_(allowed_departments))

        items = (await self.session.scalars(stmt)).all()
        results = []
        for it in items:
            results.append(await self._format_candidate_response(it))
        return results

    async def search_approved_shared_knowledge(
        self,
        query: str,
        allowed_departments: Optional[list[uuid.UUID]] = None,
        limit: int = 5,
    ) -> list[tuple[KnowledgeItem, Optional[KnowledgeEntity], str, float]]:
        """
        Database-level authorized search for approved shared organizational knowledge.
        Enforces strict retrieval criteria in SQL:
        1. visibility == 'SHARED'
        2. approval_status == 'APPROVED'
        3. status == 'ACTIVE'
        4. department_id IS NOT NULL and department_id IN (allowed_departments)
        5. At least one valid KnowledgeEvidence row exists
        6. Relevance filtering based on query tokens across canonical_key, canonical_name, and statement.
        """
        if not query or not query.strip():
            return []

        if allowed_departments is not None and len(allowed_departments) == 0:
            return []

        STOP_WORDS = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "these", "those", "then", "just", "so", "than",
            "such", "both", "through", "about", "for", "is", "of", "while", "during",
            "to", "from", "in", "out", "on", "off", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
            "don", "should", "now", "our", "my", "we", "us", "are", "tell", "me", "do",
            "does", "did", "have", "has", "had", "with"
        }
        raw_tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
        tokens = [t for t in raw_tokens if t not in STOP_WORDS and len(t) >= 2]
        if not tokens:
            tokens = [t for t in raw_tokens if len(t) >= 2]
        if not tokens:
            return []

        stmt = (
            select(KnowledgeItem)
            .options(
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.entities),
                selectinload(KnowledgeItem.workspace),
            )
            .where(
                KnowledgeItem.visibility == "SHARED",
                KnowledgeItem.approval_status == "APPROVED",
                KnowledgeItem.status == "ACTIVE",
                KnowledgeItem.department_id.is_not(None),
                exists().where(KnowledgeEvidence.knowledge_item_id == KnowledgeItem.id),
            )
        )

        if allowed_departments is not None:
            stmt = stmt.where(KnowledgeItem.department_id.in_(allowed_departments))

        token_clauses = []
        for tok in tokens[:8]:
            token_clauses.append(KnowledgeItem.canonical_key.ilike(f"%{tok}%"))
            token_clauses.append(
                exists().where(
                    KnowledgeEntity.knowledge_item_id == KnowledgeItem.id,
                    or_(
                        KnowledgeEntity.canonical_name.ilike(f"%{tok}%"),
                        func.cast(KnowledgeEntity.attributes, String).ilike(f"%{tok}%"),
                    ),
                )
            )

        stmt = stmt.where(or_(*token_clauses))
        stmt = stmt.order_by(KnowledgeItem.updated_at.desc()).limit(limit * 3)

        items = (await self.session.scalars(stmt)).all()

        scored_results: list[tuple[KnowledgeItem, Optional[KnowledgeEntity], str, float]] = []
        for item in items:
            entity = item.entities[0] if item.entities else None
            stmt_text = ""
            if entity:
                stmt_text = (entity.attributes or {}).get("statement") or entity.canonical_name or ""
            if not stmt_text:
                stmt_text = item.canonical_key

            search_blob = f"{item.canonical_key} {stmt_text}".lower()
            match_count = sum(1 for tok in tokens if tok in search_blob)
            if match_count > 0:
                relevance_score = match_count / len(tokens)
                scored_results.append((item, entity, stmt_text, relevance_score))

        scored_results.sort(key=lambda x: (x[3], x[0].updated_at or x[0].created_at), reverse=True)
        return scored_results[:limit]

    async def retrieve_shared_knowledge_as_chunks(
        self,
        query: str,
        allowed_departments: Optional[list[uuid.UUID]] = None,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieves approved shared organizational knowledge and formats them
        as safe RetrievedChunk instances adhering to standard RAG evidence contracts.
        Ensures NO raw ChatMessage text is exposed.
        """
        candidates = await self.search_approved_shared_knowledge(
            query=query,
            allowed_departments=allowed_departments,
            limit=limit,
        )

        chunks: list[RetrievedChunk] = []
        for idx, (item, entity, stmt_text, rel_score) in enumerate(candidates):
            topic = entity.entity_type if (entity and entity.entity_type) else item.kind
            dept_code = item.department.code if item.department else "ORG"
            approval_date = item.approved_at.strftime("%Y-%m-%d") if item.approved_at else ""
            confidence_str = f" | Confidence: {item.confidence:.2f}" if item.confidence is not None else ""

            formatted_content = (
                f"[Approved Shared Organizational Knowledge: {topic}]\n"
                f"{stmt_text}\n"
                f"Status: Approved | Department: {dept_code} | Date: {approval_date}{confidence_str}"
            )

            chunk = RetrievedChunk(
                chunk_id=f"shared_kn_{item.id}",
                document_id=f"shared_kn_doc_{item.id}",
                department_id=str(item.department_id),
                file_name=f"Approved Shared Knowledge: {topic}",
                chunk_index=idx,
                text=formatted_content,
                hybrid_score=rel_score,
                reranker_score=None,
            )
            chunks.append(chunk)

        return chunks
