import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.db.models import (
    ConnectionCandidate,
    ConnectionCandidateEvidence,
    ConnectionCandidateReview,
    Department,
    KnowledgeEntity,
    KnowledgeEvidence,
    KnowledgeItem,
    KnowledgeRelationship,
    MemoryItem,
    User,
    Workspace,
    WorkspaceMember,
)
from rag_app.schemas.connections import (
    ConnectionCandidateResponse,
    ConnectionEvidenceResponse,
    ConnectionReviewRequest,
)
from rag_app.security.authorization import (
    get_allowed_departments,
    get_allowed_workspaces,
)

logger = logging.getLogger(__name__)


def compute_hash(text: str) -> str:
    """Computes a deterministic SHA-256 hash."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class ConnectionDiscoveryService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def resolve_source_and_version(
        self,
        source_id: uuid.UUID,
        source_type: str,
        user: User,
    ) -> Tuple[Any, str, Optional[uuid.UUID], Optional[uuid.UUID]]:
        """
        Validates user access to the source item and derives its deterministic version hash.
        Returns: (item, version_hash, department_id, workspace_id)
        """
        source_type_upper = source_type.upper()

        if source_type_upper in ("NOTE", "KNOWLEDGE"):
            item = await self.session.get(KnowledgeItem, source_id)
            if not item or item.status != "ACTIVE":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Source knowledge item {source_id} not found or inactive",
                )

            # Enforce authorization
            if user.role != "ADMIN":
                if item.visibility == "PRIVATE":
                    if item.author_id != user.id:
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
                elif item.visibility == "WORKSPACE":
                    allowed_ws = await get_allowed_workspaces(self.session, user)
                    if allowed_ws is not None and (item.workspace_id not in allowed_ws):
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
                elif item.visibility == "SHARED":
                    allowed_depts = await get_allowed_departments(self.session, user)
                    if allowed_depts is not None and (item.department_id not in allowed_depts):
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

            # Determine hash from content/canonical_key and timestamp
            updated_str = item.updated_at.isoformat() if item.updated_at else item.created_at.isoformat()
            version_hash = compute_hash(f"{item.canonical_key}:{updated_str}")
            return item, version_hash, item.department_id, item.workspace_id

        elif source_type_upper == "MEMORY":
            item = await self.session.get(MemoryItem, source_id)
            if not item or item.status != "ACTIVE":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Source memory item {source_id} not found or inactive",
                )

            # Strictly user-scoped
            if user.role != "ADMIN" and item.user_id != user.id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

            if item.workspace_id and user.role != "ADMIN":
                allowed_ws = await get_allowed_workspaces(self.session, user)
                if allowed_ws is not None and item.workspace_id not in allowed_ws:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

            updated_str = item.updated_at.isoformat() if item.updated_at else item.created_at.isoformat()
            version_hash = compute_hash(f"{item.canonical_key}:{item.content}:{updated_str}")
            return item, version_hash, None, item.workspace_id

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported source item type '{source_type}'",
            )

    async def generate_candidates_for_source(
        self,
        source_id: uuid.UUID,
        source_type: str,
        user: User,
        max_candidates: int = 10,
    ) -> List[ConnectionCandidate]:
        """
        Discovers and persists candidate connections with pre-retrieval SQL authorization containment.
        """
        source_item, source_hash, source_dept_id, source_ws_id = await self.resolve_source_and_version(
            source_id=source_id,
            source_type=source_type,
            user=user,
        )

        allowed_depts = await get_allowed_departments(self.session, user) if user.role != "ADMIN" else None
        allowed_workspaces = await get_allowed_workspaces(self.session, user) if user.role != "ADMIN" else None

        source_text = getattr(source_item, "canonical_key", "")
        if hasattr(source_item, "content"):
            source_text += " " + (source_item.content or "")

        discovered_candidates: List[ConnectionCandidate] = []

        # -------------------------------------------------------------
        # 1. Search other authorized KnowledgeItem records
        # -------------------------------------------------------------
        ki_stmt = select(KnowledgeItem).where(
            KnowledgeItem.status == "ACTIVE",
            KnowledgeItem.id != source_id,
        )

        if user.role != "ADMIN":
            auth_clauses = [
                and_(KnowledgeItem.visibility == "PRIVATE", KnowledgeItem.author_id == user.id),
            ]
            if allowed_workspaces is not None and len(allowed_workspaces) > 0:
                auth_clauses.append(
                    and_(KnowledgeItem.visibility == "WORKSPACE", KnowledgeItem.workspace_id.in_(allowed_workspaces))
                )
            if allowed_depts is not None and len(allowed_depts) > 0:
                auth_clauses.append(
                    and_(KnowledgeItem.visibility == "SHARED", KnowledgeItem.department_id.in_(allowed_depts))
                )
            ki_stmt = ki_stmt.where(or_(*auth_clauses))

        ki_stmt = ki_stmt.order_by(KnowledgeItem.created_at.desc()).limit(max_candidates)
        target_knowledge_items = (await self.session.scalars(ki_stmt)).all()

        for target_ki in target_knowledge_items:
            target_updated_str = (
                target_ki.updated_at.isoformat() if target_ki.updated_at else target_ki.created_at.isoformat()
            )
            target_hash = compute_hash(f"{target_ki.canonical_key}:{target_updated_str}")

            # Calculate ranking score (lexical heuristic for candidate ordering only)
            target_text = getattr(target_ki, "canonical_key", "")
            if hasattr(target_ki, "content"):
                target_text += " " + (target_ki.content or "")
            score = self._compute_lexical_similarity(source_text, target_text)

            # Determine connection type and lifecycle state
            # Factual relationships require evidence; otherwise it is a semantic association
            if score >= 0.1:
                conn_type = "SEMANTIC_ASSOCIATION"
                predicate = "associated_with"
                lifecycle = "SUGGESTED"
            else:
                continue

            candidate = await self._upsert_candidate(
                source_knowledge_id=source_id if source_type.upper() in ("NOTE", "KNOWLEDGE") else None,
                source_memory_id=source_id if source_type.upper() == "MEMORY" else None,
                target_knowledge_id=target_ki.id,
                target_memory_id=None,
                connection_type=conn_type,
                predicate=predicate,
                lifecycle_state=lifecycle,
                ranking_score=score,
                source_hash=source_hash,
                target_hash=target_hash,
                department_id=target_ki.department_id or source_dept_id,
                workspace_id=target_ki.workspace_id or source_ws_id,
                author_user_id=user.id,
            )
            discovered_candidates.append(candidate)

        # -------------------------------------------------------------
        # 2. Search other authorized MemoryItem records
        # -------------------------------------------------------------
        mem_stmt = select(MemoryItem).where(
            MemoryItem.status == "ACTIVE",
            MemoryItem.id != source_id,
        )
        if user.role != "ADMIN":
            mem_stmt = mem_stmt.where(MemoryItem.user_id == user.id)
            if allowed_workspaces is not None:
                mem_stmt = mem_stmt.where(
                    or_(MemoryItem.workspace_id.is_(None), MemoryItem.workspace_id.in_(allowed_workspaces))
                )

        mem_stmt = mem_stmt.order_by(MemoryItem.created_at.desc()).limit(max_candidates)
        target_memories = (await self.session.scalars(mem_stmt)).all()

        for target_mem in target_memories:
            target_updated_str = (
                target_mem.updated_at.isoformat() if target_mem.updated_at else target_mem.created_at.isoformat()
            )
            target_hash = compute_hash(f"{target_mem.canonical_key}:{target_mem.content}:{target_updated_str}")
            score = self._compute_lexical_similarity(source_text, f"{target_mem.canonical_key} {target_mem.content}")

            if score >= 0.1:
                candidate = await self._upsert_candidate(
                    source_knowledge_id=source_id if source_type.upper() in ("NOTE", "KNOWLEDGE") else None,
                    source_memory_id=source_id if source_type.upper() == "MEMORY" else None,
                    target_knowledge_id=None,
                    target_memory_id=target_mem.id,
                    connection_type="SEMANTIC_ASSOCIATION",
                    predicate="associated_with",
                    lifecycle_state="SUGGESTED",
                    ranking_score=score,
                    source_hash=source_hash,
                    target_hash=target_hash,
                    department_id=None,
                    workspace_id=target_mem.workspace_id or source_ws_id,
                    author_user_id=user.id,
                )
                discovered_candidates.append(candidate)

        await self.session.commit()
        return discovered_candidates

    def _compute_lexical_similarity(self, text_a: str, text_b: str) -> float:
        """Deterministic Jaccard token overlap similarity in [0.0, 1.0]."""
        tokens_a = set(re.findall(r"[a-zA-Z0-9]+", text_a.lower()))
        tokens_b = set(re.findall(r"[a-zA-Z0-9]+", text_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return round(len(intersection) / len(union), 4)

    async def _upsert_candidate(
        self,
        source_knowledge_id: Optional[uuid.UUID],
        source_memory_id: Optional[uuid.UUID],
        target_knowledge_id: Optional[uuid.UUID],
        target_memory_id: Optional[uuid.UUID],
        connection_type: str,
        predicate: str,
        lifecycle_state: str,
        ranking_score: float,
        source_hash: str,
        target_hash: str,
        department_id: Optional[uuid.UUID],
        workspace_id: Optional[uuid.UUID],
        author_user_id: uuid.UUID,
    ) -> ConnectionCandidate:
        """
        Upserts candidate or detects mutation to trigger revalidation.
        """
        stmt = select(ConnectionCandidate).where(
            ConnectionCandidate.source_knowledge_item_id == source_knowledge_id,
            ConnectionCandidate.source_memory_item_id == source_memory_id,
            ConnectionCandidate.target_knowledge_item_id == target_knowledge_id,
            ConnectionCandidate.target_memory_item_id == target_memory_id,
            ConnectionCandidate.predicate == predicate,
        )
        existing = await self.session.scalar(stmt)

        if existing:
            # Check if source or target version mutated
            if existing.source_version_hash != source_hash or existing.target_version_hash != target_hash:
                logger.info("Source or target version changed; setting to REVALIDATION_IN_PROGRESS")
                existing.source_version_hash = source_hash
                existing.target_version_hash = target_hash
                existing.ranking_score = ranking_score
                if existing.lifecycle_state == "CONFIRMED":
                    existing.lifecycle_state = "REVALIDATION_IN_PROGRESS"
                existing.updated_at = datetime.now(timezone.utc)
            else:
                existing.ranking_score = ranking_score
                existing.updated_at = datetime.now(timezone.utc)
            return existing

        candidate = ConnectionCandidate(
            source_knowledge_item_id=source_knowledge_id,
            source_memory_item_id=source_memory_id,
            target_knowledge_item_id=target_knowledge_id,
            target_memory_item_id=target_memory_id,
            connection_type=connection_type,
            predicate=predicate,
            lifecycle_state=lifecycle_state,
            ranking_score=ranking_score,
            source_version_hash=source_hash,
            target_version_hash=target_hash,
            department_id=department_id,
            workspace_id=workspace_id,
            author_user_id=author_user_id,
        )
        self.session.add(candidate)
        await self.session.flush()
        return candidate

    async def review_candidate(
        self,
        candidate_id: uuid.UUID,
        user: User,
        request: ConnectionReviewRequest,
    ) -> ConnectionCandidateResponse:
        """
        Processes human curator review decision with version-bound verification.
        Promotes confirmed relationships into canonical KnowledgeRelationship layer.
        """
        stmt = (
            select(ConnectionCandidate)
            .options(
                selectinload(ConnectionCandidate.evidence_links),
                selectinload(ConnectionCandidate.source_knowledge_item),
                selectinload(ConnectionCandidate.target_knowledge_item),
            )
            .where(ConnectionCandidate.id == candidate_id)
        )
        candidate = await self.session.scalar(stmt)
        if not candidate:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection candidate not found")

        # Verify version binding to prevent stale approval
        if (
            candidate.source_version_hash != request.source_version_hash
            or candidate.target_version_hash != request.target_version_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Source or target version has changed since candidate was reviewed. Revalidation required.",
            )

        decision = request.decision.upper()
        if decision not in ("CONFIRMED", "REJECTED", "CONTRADICTED"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid decision '{request.decision}'")

        # Record review history
        review_record = ConnectionCandidateReview(
            candidate_id=candidate.id,
            reviewer_user_id=user.id,
            decision=decision,
            rationale=request.rationale,
            source_version_hash=request.source_version_hash,
            target_version_hash=request.target_version_hash,
            policy_version=request.policy_version,
        )
        self.session.add(review_record)

        candidate.lifecycle_state = decision
        candidate.updated_at = datetime.now(timezone.utc)

        # If CONFIRMED, promote to canonical KnowledgeRelationship if applicable
        if decision == "CONFIRMED":
            await self._promote_to_canonical_knowledge(candidate)

        await self.session.commit()
        stmt = (
            select(ConnectionCandidate)
            .options(
                selectinload(ConnectionCandidate.evidence_links),
                selectinload(ConnectionCandidate.source_knowledge_item),
                selectinload(ConnectionCandidate.source_memory_item),
                selectinload(ConnectionCandidate.target_knowledge_item),
                selectinload(ConnectionCandidate.target_memory_item),
            )
            .where(ConnectionCandidate.id == candidate.id)
        )
        loaded = await self.session.scalar(stmt)
        return await self._format_candidate_response(loaded or candidate)

    async def _promote_to_canonical_knowledge(self, candidate: ConnectionCandidate) -> None:
        """
        Materializes approved relationship into the canonical knowledge graph layer.
        """
        if candidate.source_knowledge_item_id and candidate.target_knowledge_item_id:
            # Resolve or create KnowledgeEntity stubs for subject and object
            src_entity = await self.session.scalar(
                select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == candidate.source_knowledge_item_id)
            )
            tgt_entity = await self.session.scalar(
                select(KnowledgeEntity).where(KnowledgeEntity.knowledge_item_id == candidate.target_knowledge_item_id)
            )

            if src_entity and tgt_entity:
                rel = KnowledgeRelationship(
                    knowledge_item_id=candidate.source_knowledge_item_id,
                    subject_entity_id=src_entity.id,
                    predicate=candidate.predicate,
                    object_entity_id=tgt_entity.id,
                )
                self.session.add(rel)
                logger.info(
                    "connection_promoted_to_canonical_graph",
                    extra={
                        "candidate_id": str(candidate.id),
                        "subject_entity_id": str(src_entity.id),
                        "object_entity_id": str(tgt_entity.id),
                        "predicate": candidate.predicate,
                    },
                )

    async def list_candidates(
        self,
        user: User,
        source_id: Optional[uuid.UUID] = None,
        lifecycle_state: Optional[str] = None,
        connection_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ConnectionCandidateResponse]:
        """Lists authorized connection candidates with strict filtering."""
        stmt = (
            select(ConnectionCandidate)
            .options(
                selectinload(ConnectionCandidate.evidence_links),
                selectinload(ConnectionCandidate.source_knowledge_item),
                selectinload(ConnectionCandidate.source_memory_item),
                selectinload(ConnectionCandidate.target_knowledge_item),
                selectinload(ConnectionCandidate.target_memory_item),
            )
        )

        if source_id:
            stmt = stmt.where(
                or_(
                    ConnectionCandidate.source_knowledge_item_id == source_id,
                    ConnectionCandidate.source_memory_item_id == source_id,
                )
            )

        if lifecycle_state:
            stmt = stmt.where(ConnectionCandidate.lifecycle_state == lifecycle_state.upper())

        if connection_type:
            stmt = stmt.where(ConnectionCandidate.connection_type == connection_type.upper())

        # Authorization containment
        if user.role != "ADMIN":
            allowed_depts = await get_allowed_departments(self.session, user)
            allowed_ws = await get_allowed_workspaces(self.session, user)

            auth_clauses = [
                ConnectionCandidate.author_user_id == user.id,
            ]
            if allowed_depts is not None and len(allowed_depts) > 0:
                auth_clauses.append(ConnectionCandidate.department_id.in_(allowed_depts))
            if allowed_ws is not None and len(allowed_ws) > 0:
                auth_clauses.append(ConnectionCandidate.workspace_id.in_(allowed_ws))

            stmt = stmt.where(or_(*auth_clauses))

        stmt = stmt.order_by(ConnectionCandidate.ranking_score.desc()).limit(limit).offset(offset)
        rows = (await self.session.scalars(stmt)).all()

        results = []
        for row in rows:
            results.append(await self._format_candidate_response(row))
        return results

    async def get_candidate(
        self,
        candidate_id: uuid.UUID,
        user: User,
    ) -> ConnectionCandidateResponse:
        """Retrieves single authorized candidate."""
        stmt = (
            select(ConnectionCandidate)
            .options(
                selectinload(ConnectionCandidate.evidence_links),
                selectinload(ConnectionCandidate.source_knowledge_item),
                selectinload(ConnectionCandidate.source_memory_item),
                selectinload(ConnectionCandidate.target_knowledge_item),
                selectinload(ConnectionCandidate.target_memory_item),
            )
            .where(ConnectionCandidate.id == candidate_id)
        )
        candidate = await self.session.scalar(stmt)
        if not candidate:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection candidate not found")

        # Authorization check
        if user.role != "ADMIN":
            allowed_depts = await get_allowed_departments(self.session, user)
            allowed_ws = await get_allowed_workspaces(self.session, user)

            is_auth = (
                candidate.author_user_id == user.id
                or (allowed_depts is not None and candidate.department_id in allowed_depts)
                or (allowed_ws is not None and candidate.workspace_id in allowed_ws)
            )
            if not is_auth:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

        return await self._format_candidate_response(candidate)

    async def _format_candidate_response(self, item: ConnectionCandidate) -> ConnectionCandidateResponse:
        src_id = item.source_knowledge_item_id or item.source_memory_item_id
        src_type = "KNOWLEDGE" if item.source_knowledge_item_id else "MEMORY"
        src_title = str(src_id)
        if "source_knowledge_item" in item.__dict__ and item.source_knowledge_item:
            src_title = item.source_knowledge_item.canonical_key
        elif "source_memory_item" in item.__dict__ and item.source_memory_item:
            src_title = item.source_memory_item.canonical_key

        tgt_id = item.target_knowledge_item_id or item.target_memory_item_id
        tgt_type = "KNOWLEDGE" if item.target_knowledge_item_id else "MEMORY"
        tgt_title = str(tgt_id)
        if "target_knowledge_item" in item.__dict__ and item.target_knowledge_item:
            tgt_title = item.target_knowledge_item.canonical_key
        elif "target_memory_item" in item.__dict__ and item.target_memory_item:
            tgt_title = item.target_memory_item.canonical_key

        evidence_responses = []
        if "evidence_links" in item.__dict__ and item.evidence_links:
            for ev in item.evidence_links:
                evidence_responses.append(
                    ConnectionEvidenceResponse(
                        id=ev.id,
                        candidate_id=ev.candidate_id,
                        document_chunk_id=ev.document_chunk_id,
                        chat_message_id=ev.chat_message_id,
                        research_evidence_id=ev.research_evidence_id,
                        claim_excerpt=ev.claim_excerpt,
                        content_hash=ev.content_hash,
                        verification_status=ev.verification_status,
                        created_at=ev.created_at,
                    )
                )

        return ConnectionCandidateResponse(
            id=item.id,
            source_item_id=src_id,
            source_item_type=src_type,
            source_title=src_title,
            target_item_id=tgt_id,
            target_item_type=tgt_type,
            target_title=tgt_title,
            connection_type=item.connection_type,
            predicate=item.predicate,
            lifecycle_state=item.lifecycle_state,
            is_directed=item.is_directed,
            ranking_score=item.ranking_score,
            source_version_hash=item.source_version_hash,
            target_version_hash=item.target_version_hash,
            department_id=item.department_id,
            workspace_id=item.workspace_id,
            author_user_id=item.author_user_id,
            evidence=evidence_responses,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
