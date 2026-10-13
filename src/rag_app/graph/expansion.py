"""
Bounded Relationship Expansion Service for Stage I.

Performs controlled, bounded 1-hop and 2-hop graph expansion starting from a canonical entity.
Strictly enforces:
- Configurable bounded depth (default: 2)
- Evidence budget limit (default: 15)
- Permitted relationship predicates
- Lifecycle filtering (ACTIVE only, never ARCHIVED or SUPERSEDED)
- Access policy filtering (PRIVATE drafts visible only to author, department RBAC, document VIEW authorization)
- Source provenance retention (timestamps, authority, sharing scope, chunk IDs)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Set

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.db.models import (
    Document,
    DocumentChunk,
    KnowledgeEntity,
    KnowledgeEvidence,
    KnowledgeItem,
    KnowledgeRelationship,
    User,
)

logger = logging.getLogger(__name__)

DEFAULT_PERMITTED_PREDICATES: Set[str] = {
    "has_observation",
    "has_commitment",
    "has_decision",
    "has_meeting",
    "has_task",
    "has_contact",
    "works_at",
    "leads",
    "develops",
    "invested_in",
    "works_on",
    "assigned_to",
    "agreed_to",
    "associated_with",
}


@dataclass
class ExpandedEdge:
    hop: int
    subject_id: uuid.UUID
    subject_name: str
    subject_type: str
    predicate: str
    object_id: uuid.UUID
    object_name: str
    object_type: str
    knowledge_item_id: uuid.UUID
    canonical_key: str
    statement: str
    confidence: Optional[float]
    source_type: str
    sharing_scope: str  # PRIVATE, WORKSPACE, SHARED
    approval_status: str  # DRAFT, PENDING, APPROVED, REJECTED
    department_id: Optional[uuid.UUID]
    department_code: Optional[str]
    author_email: Optional[str]
    approved_by_email: Optional[str]
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime]
    evidence_chunks: list[dict] = field(default_factory=list)


@dataclass
class GraphExpansionResult:
    root_entity_id: uuid.UUID
    root_canonical_name: str
    edges: list[ExpandedEdge] = field(default_factory=list)
    depth_reached: int = 0
    budget_reached: bool = False
    total_items: int = 0
    visited_entity_names: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    db_queries_executed: int = 0
    db_rows_fetched: int = 0
    budget_halt_reason: Optional[str] = None


class BoundedGraphExpansionService:
    """
    Service responsible for bounded relationship expansion with strict lifecycle,
    RBAC filtering, and aggregate resource budgets (edges, queries, rows, execution time).
    """

    def __init__(
        self,
        session: AsyncSession,
        max_depth: int = 2,
        max_budget: int = 15,
        max_query_budget: int = 6,
        max_row_budget: int = 60,
        timeout_seconds: float = 3.0,
        permitted_predicates: Optional[Set[str]] = None,
    ):
        self.session = session
        self.max_depth = max(1, min(max_depth, 2))  # Bounded strictly to 1 or 2 hops
        self.max_budget = max(1, max_budget)
        self.max_query_budget = max(1, max_query_budget)
        self.max_row_budget = max(1, max_row_budget)
        self.timeout_seconds = max(0.1, timeout_seconds)
        self.permitted_predicates = permitted_predicates or DEFAULT_PERMITTED_PREDICATES

    async def expand_entity(
        self,
        root_entity: KnowledgeEntity,
        user: User,
        allowed_departments: Optional[list[uuid.UUID]] = None,
        workspace_id: Optional[uuid.UUID] = None,
        predicate_filter: Optional[str] = None,
    ) -> GraphExpansionResult:
        """
        Expands relationships originating from or targeting root_entity up to max_depth hops,
        halting when any aggregate resource budget (edges, queries, rows, time) is met.
        """
        import time
        start_time = time.perf_counter()

        result = GraphExpansionResult(
            root_entity_id=root_entity.id,
            root_canonical_name=root_entity.canonical_name,
        )

        visited_entities: set[uuid.UUID] = {root_entity.id}
        visited_rel_ids: set[uuid.UUID] = set()
        current_frontier: set[uuid.UUID] = {root_entity.id}
        queries_executed = 0
        total_rows_fetched = 0

        for current_hop in range(1, self.max_depth + 1):
            elapsed = time.perf_counter() - start_time
            if elapsed >= self.timeout_seconds:
                result.budget_reached = True
                result.budget_halt_reason = "TIMEOUT_REACHED"
                break

            if not current_frontier or len(result.edges) >= self.max_budget:
                if len(result.edges) >= self.max_budget:
                    result.budget_reached = True
                    result.budget_halt_reason = "MAX_EDGES_REACHED"
                break

            if queries_executed >= self.max_query_budget:
                result.budget_reached = True
                result.budget_halt_reason = "MAX_QUERIES_REACHED"
                break

            if total_rows_fetched >= self.max_row_budget:
                result.budget_reached = True
                result.budget_halt_reason = "MAX_ROWS_REACHED"
                break

            next_frontier: set[uuid.UUID] = set()
            hop_edges, rows_count = await self._traverse_hop(
                frontier_ids=current_frontier,
                user=user,
                allowed_departments=allowed_departments,
                workspace_id=workspace_id,
                predicate_filter=predicate_filter,
                current_hop=current_hop,
                visited_rel_ids=visited_rel_ids,
                remaining_row_budget=max(1, self.max_row_budget - total_rows_fetched),
            )
            queries_executed += 1
            total_rows_fetched += rows_count

            for edge in hop_edges:
                if len(result.edges) >= self.max_budget:
                    result.budget_reached = True
                    result.budget_halt_reason = "MAX_EDGES_REACHED"
                    break

                result.edges.append(edge)
                visited_rel_ids.add(edge.knowledge_item_id)

                # Collect next frontier nodes (the other end of the edge)
                if edge.subject_id not in visited_entities:
                    next_frontier.add(edge.subject_id)
                    visited_entities.add(edge.subject_id)
                if edge.object_id not in visited_entities:
                    next_frontier.add(edge.object_id)
                    visited_entities.add(edge.object_id)

            result.depth_reached = current_hop
            current_frontier = next_frontier

            if len(result.edges) >= self.max_budget:
                result.budget_reached = True
                result.budget_halt_reason = "MAX_EDGES_REACHED"
                break

            if total_rows_fetched >= self.max_row_budget:
                result.budget_reached = True
                result.budget_halt_reason = "MAX_ROWS_REACHED"
                break

            if queries_executed >= self.max_query_budget:
                result.budget_reached = True
                result.budget_halt_reason = "MAX_QUERIES_REACHED"
                break

            if (time.perf_counter() - start_time) >= self.timeout_seconds:
                result.budget_reached = True
                result.budget_halt_reason = "TIMEOUT_REACHED"
                break

        if result.budget_halt_reason is None:
            result.budget_halt_reason = "COMPLETED"

        result.execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        result.db_queries_executed = queries_executed
        result.db_rows_fetched = total_rows_fetched
        result.total_items = len(result.edges)
        result.visited_entity_names = sorted(
            list({e.subject_name for e in result.edges} | {e.object_name for e in result.edges})
        )
        return result

    async def _traverse_hop(
        self,
        frontier_ids: set[uuid.UUID],
        user: User,
        allowed_departments: Optional[list[uuid.UUID]],
        workspace_id: Optional[uuid.UUID],
        predicate_filter: Optional[str],
        current_hop: int,
        visited_rel_ids: set[uuid.UUID],
        remaining_row_budget: int = 60,
    ) -> tuple[list[ExpandedEdge], int]:
        """
        Executes a single hop traversal from frontier_ids with strict lifecycle,
        visibility, and RBAC filters, bounded by database limit.
        """
        stmt = (
            select(
                KnowledgeRelationship,
                KnowledgeItem,
            )
            .join(KnowledgeItem, KnowledgeItem.id == KnowledgeRelationship.knowledge_item_id)
            .options(
                selectinload(KnowledgeRelationship.subject_entity),
                selectinload(KnowledgeRelationship.object_entity),
                selectinload(KnowledgeItem.department),
                selectinload(KnowledgeItem.author),
                selectinload(KnowledgeItem.approved_by),
                selectinload(KnowledgeItem.evidence_links).selectinload(KnowledgeEvidence.document_chunk).selectinload(DocumentChunk.document),
            )
            .where(
                or_(
                    KnowledgeRelationship.subject_entity_id.in_(frontier_ids),
                    KnowledgeRelationship.object_entity_id.in_(frontier_ids),
                ),
                # Lifecycle Rule: ACTIVE status only (exclude ARCHIVED and SUPERSEDED)
                KnowledgeItem.status == "ACTIVE",
                # Never return explicitly REJECTED items
                KnowledgeItem.approval_status != "REJECTED",
            )
        )

        # Department RBAC Filter
        if allowed_departments is not None:
            stmt = stmt.where(
                or_(
                    KnowledgeItem.department_id.is_(None),
                    KnowledgeItem.department_id.in_(allowed_departments),
                )
            )

        # Predicate filtering
        if predicate_filter:
            norm_pf = predicate_filter.strip().lower()
            stmt = stmt.where(KnowledgeRelationship.predicate.ilike(f"%{norm_pf}%"))

        # Database-level budget bounding: limit query work directly in PostgreSQL
        needed_edges = max(1, self.max_budget - len(visited_rel_ids))
        db_limit = min(needed_edges * 3, remaining_row_budget)
        stmt = stmt.limit(db_limit)

        rows = (await self.session.execute(stmt)).all()
        edges: list[ExpandedEdge] = []

        for rel, ki in rows:
            if rel.id in visited_rel_ids or ki.id in visited_rel_ids:
                continue

            # Permitted Predicates Boundary Check
            norm_predicate = (rel.predicate or "").lower().replace("-", "_").replace(" ", "_")
            if norm_predicate not in self.permitted_predicates:
                continue

            # Access Policy Check: Private drafts visible ONLY to their author
            if ki.visibility == "PRIVATE":
                if ki.author_id != user.id and user.role != "ADMIN":
                    continue

            # Workspace Boundary Check (if specified)
            if workspace_id is not None and ki.workspace_id is not None:
                if ki.workspace_id != workspace_id and ki.visibility != "SHARED":
                    continue

            subj = rel.subject_entity
            obj = rel.object_entity
            if not subj or not obj:
                continue

            # Statement extraction: from object entity attributes if note, or canonical key
            statement = ""
            if obj.attributes and isinstance(obj.attributes, dict):
                statement = obj.attributes.get("statement") or obj.canonical_name
            if not statement:
                statement = f"{subj.canonical_name} {rel.predicate} {obj.canonical_name}"

            # Document Evidence Extraction with VIEW Authorization Check
            evidence_chunks: list[dict] = []
            for ev in ki.evidence_links:
                if ev.document_chunk:
                    chunk = ev.document_chunk
                    doc = chunk.document
                    # Authorize document viewing
                    if allowed_departments is not None and doc and doc.department_id:
                        if doc.department_id not in allowed_departments and user.role != "ADMIN":
                            continue  # Skip unauthorized chunk evidence

                    evidence_chunks.append({
                        "document_chunk_id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "file_name": doc.file_name if doc else "Document",
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "extraction_method": ev.extraction_method,
                        "confidence": ev.confidence,
                    })

            dept_code = ki.department.code if ki.department else None
            author_email = ki.author.email if ki.author else None
            approver_email = ki.approved_by.email if ki.approved_by else None

            edges.append(
                ExpandedEdge(
                    hop=current_hop,
                    subject_id=subj.id,
                    subject_name=subj.canonical_name,
                    subject_type=subj.entity_type,
                    predicate=rel.predicate,
                    object_id=obj.id,
                    object_name=obj.canonical_name,
                    object_type=obj.entity_type,
                    knowledge_item_id=ki.id,
                    canonical_key=ki.canonical_key,
                    statement=statement,
                    confidence=ki.confidence,
                    source_type=ki.source_type,
                    sharing_scope=ki.visibility,
                    approval_status=ki.approval_status,
                    department_id=ki.department_id,
                    department_code=dept_code,
                    author_email=author_email,
                    approved_by_email=approver_email,
                    created_at=ki.created_at,
                    updated_at=ki.updated_at,
                    approved_at=ki.approved_at,
                    evidence_chunks=evidence_chunks,
                )
            )

        return edges, len(rows)
