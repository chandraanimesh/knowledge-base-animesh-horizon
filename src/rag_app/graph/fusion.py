"""
Multi-Source Evidence Fusion Service for Stage I.

Fuses authorized evidence across:
1. Structured Stage H tasks and follow-ups.
2. Approved shared Company Notes and Knowledge Graph relationships.
3. Hybrid document chunks and meeting records.
4. Personal memory (strictly scoped to the authenticated user).
5. Explicitly unavailable operational CRM status (preventing CRM fabrication).

Handles deduplication, preserves distinct temporal events, and flags contradictory claims.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.core.config import settings
from rag_app.crm.factory import AnyCRMClient, get_crm_client
from rag_app.crm.twenty_client import TwentyCRMClient
from rag_app.db.models import (
    Document,
    DocumentChunk,
    KnowledgeEntity,
    KnowledgeEvidence,
    KnowledgeItem,
    MemoryItem,
    TaskItem,
    User,
)
from rag_app.graph.expansion import BoundedGraphExpansionService, ExpandedEdge
from rag_app.memory.materializer import MemoryMaterializer

logger = logging.getLogger(__name__)


@dataclass
class FusedEvidenceSource:
    stream: str  # STRUCTURED_TASK, COMPANY_NOTE, GRAPH_RELATIONSHIP, DOCUMENT_CHUNK, PERSONAL_MEMORY, OPERATIONAL_CRM
    source_id: str
    title: str
    content: str
    provenance: dict[str, Any] = field(default_factory=dict)
    is_conflict: bool = False
    conflict_note: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class FusedEvidenceResult:
    company_name: str
    canonical_entity_id: Optional[uuid.UUID] = None
    tasks: list[FusedEvidenceSource] = field(default_factory=list)
    company_notes: list[FusedEvidenceSource] = field(default_factory=list)
    graph_edges: list[ExpandedEdge] = field(default_factory=list)
    document_chunks: list[FusedEvidenceSource] = field(default_factory=list)
    personal_memories: list[FusedEvidenceSource] = field(default_factory=list)
    crm_records: list[FusedEvidenceSource] = field(default_factory=list)
    conflicts_detected: list[str] = field(default_factory=list)
    crm_status_available: bool = False
    crm_notice: str = (
        "Operational CRM integration is not connected or currently unavailable. "
        "Real-time pipeline state, stage transitions, and external CRM data cannot be verified."
    )
    formatted_synthesis: str = ""
    sources_payload: list[dict[str, Any]] = field(default_factory=list)

    def to_retrieved_chunks(self):
        """
        Converts fused evidence items into standard RetrievedChunk objects
        for citations, verification, and grounded generation inspection.
        """
        from rag_app.schemas.rag import RetrievedChunk
        chunks: list[RetrievedChunk] = []

        # 1. Document chunks
        for idx, d in enumerate(self.document_chunks):
            chunks.append(
                RetrievedChunk(
                    chunk_id=d.source_id,
                    document_id=str(d.provenance.get("document_id") or f"doc_{idx}"),
                    department_id=str(d.provenance.get("department_id") or "None"),
                    file_name=d.title,
                    chunk_index=int(d.provenance.get("chunk_index") or idx),
                    text=d.content,
                )
            )

        # 2. Company notes
        for idx, n in enumerate(self.company_notes):
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"shared_kn_{n.source_id}",
                    document_id=f"shared_kn_doc_{n.source_id}",
                    department_id=str(n.provenance.get("department") or "ORG"),
                    file_name=n.title,
                    chunk_index=idx,
                    text=n.content,
                )
            )

        # 3. Tasks
        for idx, t in enumerate(self.tasks):
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"task_{t.source_id}",
                    document_id=f"task_doc_{t.source_id}",
                    department_id=str(t.provenance.get("department") or "None"),
                    file_name=t.title,
                    chunk_index=idx,
                    text=t.content,
                )
            )

        # 4. Operational CRM Records
        crm_dept_str = str(getattr(settings, "twenty_crm_department_id", None) or "SALES_CRM")
        for idx, r in enumerate(self.crm_records):
            chunks.append(
                RetrievedChunk(
                    chunk_id=r.source_id,
                    document_id=f"crm_{r.provenance.get('crm_object', 'record')}_{r.source_id}",
                    department_id=crm_dept_str,
                    file_name=r.title,
                    chunk_index=idx,
                    text=r.content,
                )
            )

        return chunks


class MultiSourceEvidenceFusionService:
    """
    Fuses evidence across multiple authorized organizational data sources.
    """

    def __init__(
        self,
        session: AsyncSession,
        expansion_service: Optional[BoundedGraphExpansionService] = None,
        crm_client: Optional[AnyCRMClient] = None,
    ):
        self.session = session
        self.expansion_service = expansion_service or BoundedGraphExpansionService(session=session)
        self.crm_client = crm_client or get_crm_client()

    async def fuse_company_evidence(
        self,
        user: User,
        company_name: str,
        query: str,
        canonical_entity: Optional[KnowledgeEntity] = None,
        allowed_departments: Optional[list[uuid.UUID]] = None,
        workspace_id: Optional[uuid.UUID] = None,
        max_budget: int = 15,
    ) -> FusedEvidenceResult:
        """
        Executes multi-source evidence fusion for a company query.
        """
        result = FusedEvidenceResult(
            company_name=company_name,
            canonical_entity_id=canonical_entity.id if canonical_entity else None,
        )

        # -------------------------------------------------------------
        # Stream 1: Structured Stage H Tasks & Follow-ups
        # -------------------------------------------------------------
        task_stmt = (
            select(TaskItem)
            .options(
                selectinload(TaskItem.department),
                selectinload(TaskItem.user),
            )
            .where(
                or_(
                    func.lower(TaskItem.company_name) == company_name.lower(),
                    TaskItem.company_name.ilike(f"%{company_name}%"),
                )
            )
        )
        # Apply department or user isolation if not admin
        if user.role != "ADMIN":
            task_stmt = task_stmt.where(
                or_(
                    TaskItem.user_id == user.id,
                    and_clause := (
                        TaskItem.department_id.in_(allowed_departments)
                        if allowed_departments
                        else False
                    ),
                )
            )
        task_stmt = task_stmt.order_by(TaskItem.due_date.asc().nullslast(), TaskItem.created_at.desc())

        tasks = (await self.session.scalars(task_stmt)).all()
        for t in tasks:
            due_str = t.due_date.strftime("%Y-%m-%d %H:%M UTC") if t.due_date else "No due date"
            dept_code = t.department.code if t.department else "None"
            src = FusedEvidenceSource(
                stream="STRUCTURED_TASK",
                source_id=str(t.id),
                title=f"Task: {t.title}",
                content=f"[{t.status}] {t.title} (Due: {due_str}) | Company: {t.company_name}",
                provenance={
                    "task_id": str(t.id),
                    "status": t.status,
                    "due_date": t.due_date.isoformat() if t.due_date else None,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "department": dept_code,
                    "provenance_type": t.provenance_type,
                    "assigned_user": t.user.email if t.user else None,
                },
                created_at=t.created_at,
            )
            result.tasks.append(src)

        # -------------------------------------------------------------
        # Stream 2: Bounded Relationship Expansion (Knowledge Graph & Notes)
        # -------------------------------------------------------------
        if canonical_entity:
            expansion = await self.expansion_service.expand_entity(
                root_entity=canonical_entity,
                user=user,
                allowed_departments=allowed_departments,
                workspace_id=workspace_id,
            )
            result.graph_edges = expansion.edges

            for edge in expansion.edges:
                note_src = FusedEvidenceSource(
                    stream="COMPANY_NOTE" if edge.predicate == "has_observation" else "GRAPH_RELATIONSHIP",
                    source_id=str(edge.knowledge_item_id),
                    title=f"Knowledge: {edge.subject_name} {edge.predicate} {edge.object_name}",
                    content=edge.statement,
                    provenance={
                        "knowledge_item_id": str(edge.knowledge_item_id),
                        "hop": edge.hop,
                        "predicate": edge.predicate,
                        "subject": edge.subject_name,
                        "object": edge.object_name,
                        "sharing_scope": edge.sharing_scope,
                        "approval_status": edge.approval_status,
                        "department": edge.department_code,
                        "author": edge.author_email,
                        "approved_by": edge.approved_by_email,
                        "created_at": edge.created_at.isoformat() if edge.created_at else None,
                        "updated_at": edge.updated_at.isoformat() if edge.updated_at else None,
                        "evidence_chunks": edge.evidence_chunks,
                    },
                    created_at=edge.created_at,
                )
                result.company_notes.append(note_src)

        # Collect search terms: canonical company name, verified aliases, and base name
        from rag_app.graph.entity_resolver import normalize_business_name
        match_terms = {company_name}
        if canonical_entity and canonical_entity.attributes:
            ent_aliases = canonical_entity.attributes.get("aliases") or []
            if isinstance(ent_aliases, str):
                ent_aliases = [ent_aliases]
            for a in ent_aliases:
                if a and len(str(a).strip()) >= 2:
                    match_terms.add(str(a).strip())
        base_norm = normalize_business_name(company_name)
        if base_norm and len(base_norm) >= 3:
            match_terms.add(base_norm)

        doc_clauses = []
        for term in match_terms:
            doc_clauses.append(DocumentChunk.text.ilike(f"%{term}%"))
            doc_clauses.append(Document.file_name.ilike(f"%{term}%"))

        doc_stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(or_(*doc_clauses))
        )
        if allowed_departments is not None:
            doc_stmt = doc_stmt.where(Document.department_id.in_(allowed_departments))
        if workspace_id is not None:
            doc_stmt = doc_stmt.where(Document.workspace_id == workspace_id)

        doc_stmt = doc_stmt.order_by(DocumentChunk.created_at.desc()).limit(10)
        doc_rows = (await self.session.execute(doc_stmt)).all()

        seen_chunk_texts: set[str] = set()
        for chunk, doc in doc_rows:
            norm_c = chunk.text.strip().lower()[:120]
            if norm_c in seen_chunk_texts:
                continue
            seen_chunk_texts.add(norm_c)

            doc_src = FusedEvidenceSource(
                stream="DOCUMENT_CHUNK",
                source_id=str(chunk.id),
                title=f"{doc.file_name} (Page {chunk.page_number or 1})",
                content=chunk.text.strip(),
                provenance={
                    "document_chunk_id": str(chunk.id),
                    "document_id": str(doc.id),
                    "file_name": doc.file_name,
                    "chunk_index": chunk.chunk_index,
                    "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                },
                created_at=chunk.created_at,
            )
            result.document_chunks.append(doc_src)

        # -------------------------------------------------------------
        # Stream 4: Personal User Memory (User-scoped only)
        # -------------------------------------------------------------
        user_mems = await MemoryMaterializer.get_user_memories(
            session=self.session,
            user_id=user.id,
            workspace_id=workspace_id,
            include_global=True,
            limit=20,
        )
        for m in user_mems:
            if company_name.lower() in m.content.lower():
                mem_src = FusedEvidenceSource(
                    stream="PERSONAL_MEMORY",
                    source_id=str(m.id),
                    title="User Personal Memory",
                    content=m.content,
                    provenance={
                        "memory_id": str(m.id),
                        "user_id": str(m.user_id),
                        "memory_type": m.memory_type,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                    },
                    created_at=m.created_at,
                )
                result.personal_memories.append(mem_src)

        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # Stream 5: Live Operational CRM State (Twenty CRM)
        # -------------------------------------------------------------
        # Fail-closed CRM authorization gate using existing authorization primitives
        is_crm_authorized, crm_denial_reason = self._check_crm_authorization(
            user=user,
            allowed_departments=allowed_departments,
        )

        if not is_crm_authorized:
            result.crm_status_available = False
            result.crm_notice = (
                f"Live operational CRM evidence is restricted. "
                f"{crm_denial_reason or 'Access denied.'}"
            )
        else:
            try:
                crm_state = await self.crm_client.get_live_company_state(
                    company_name=company_name,
                    aliases=list(match_terms),
                )
                if crm_state.has_crm_records and crm_state.company:
                    result.crm_status_available = True
                    c = crm_state.company

                    # 1. Company record
                    comp_content = f"Company: {c.name} | Domain: {c.domain or 'N/A'}"
                    if c.address:
                        comp_content += f" | Address: {c.address}"
                    if c.annual_revenue is not None:
                        comp_content += f" | Annual Revenue: ${c.annual_revenue:,.2f}"

                    comp_src = FusedEvidenceSource(
                        stream="OPERATIONAL_CRM",
                        source_id=f"twenty_company_{c.id}",
                        title=f"CRM Company: {c.name}",
                        content=comp_content,
                        provenance={
                            "crm_system": "Twenty CRM",
                            "crm_object": "company",
                            "crm_record_id": c.id,
                            "created_at": c.created_at,
                            "updated_at": c.updated_at,
                            "retrieved_at": crm_state.retrieval_timestamp.isoformat(),
                        },
                        created_at=datetime.fromisoformat(c.updated_at.replace("Z", "+00:00")) if c.updated_at else None,
                    )
                    result.crm_records.append(comp_src)

                    # 2. People / Contacts
                    for p in crm_state.people:
                        p_content = f"Contact: {p.name} ({p.job_title or 'Contact'})"
                        if p.email:
                            p_content += f" | Email: {p.email}"
                        if p.phone:
                            p_content += f" | Phone: {p.phone}"

                        p_src = FusedEvidenceSource(
                            stream="OPERATIONAL_CRM",
                            source_id=f"twenty_person_{p.id}",
                            title=f"CRM Contact: {p.name}",
                            content=p_content,
                            provenance={
                                "crm_system": "Twenty CRM",
                                "crm_object": "person",
                                "crm_record_id": p.id,
                                "company_id": p.company_id,
                                "retrieved_at": crm_state.retrieval_timestamp.isoformat(),
                            },
                        )
                        result.crm_records.append(p_src)

                    # 3. Opportunities / Pipeline Deals
                    for o in crm_state.opportunities:
                        val_str = f"${o.amount:,.2f} {o.currency}" if o.amount is not None else "Unspecified"
                        close_str = f" | Target Close: {o.close_date}" if o.close_date else ""
                        o_content = f"Deal: {o.name} | Stage: {o.stage} | Value: {val_str}{close_str}"

                        o_src = FusedEvidenceSource(
                            stream="OPERATIONAL_CRM",
                            source_id=f"twenty_opportunity_{o.id}",
                            title=f"CRM Deal: {o.name}",
                            content=o_content,
                            provenance={
                                "crm_system": "Twenty CRM",
                                "crm_object": "opportunity",
                                "crm_record_id": o.id,
                                "stage": o.stage,
                                "amount": o.amount,
                                "currency": o.currency,
                                "close_date": o.close_date,
                                "updated_at": o.updated_at,
                                "retrieved_at": crm_state.retrieval_timestamp.isoformat(),
                            },
                            created_at=datetime.fromisoformat(o.updated_at.replace("Z", "+00:00")) if o.updated_at else None,
                        )
                        result.crm_records.append(o_src)

                    # Notice formatting with partial-failure & truncation indicators
                    notice_parts = [
                        f"Live Twenty CRM record verified for '{c.name}' (ID: {c.id})."
                    ]
                    if crm_state.opportunities:
                        trunc_tag = " [truncated to maximum page budget]" if getattr(crm_state, "opportunities_truncated", False) else ""
                        notice_parts.append(f"Retrieved {len(crm_state.opportunities)} active deals{trunc_tag}.")
                    elif getattr(crm_state, "opportunities_error", None):
                        notice_parts.append(f"Deals could not be retrieved due to error: {crm_state.opportunities_error}.")
                    else:
                        notice_parts.append("No active deals recorded.")

                    if crm_state.people:
                        trunc_tag = " [truncated to maximum page budget]" if getattr(crm_state, "people_truncated", False) else ""
                        notice_parts.append(f"Retrieved {len(crm_state.people)} contacts{trunc_tag}.")
                    elif getattr(crm_state, "people_error", None):
                        notice_parts.append(f"Contacts could not be retrieved due to error: {crm_state.people_error}.")
                    else:
                        notice_parts.append("No contacts recorded.")

                    result.crm_notice = " ".join(notice_parts)

                elif getattr(crm_state, "is_ambiguous", False):
                    result.crm_status_available = False
                    candidates = getattr(crm_state, "ambiguous_candidates", [])
                    result.crm_notice = (
                        f"Multiple CRM companies match '{company_name}': {', '.join(candidates)}."
                    )
                elif crm_state.is_connected:
                    result.crm_status_available = False
                    result.crm_notice = (
                        f"Twenty CRM is connected, but no matching company record was found for '{company_name}'."
                    )
                else:
                    result.crm_status_available = False
                    result.crm_notice = (
                        f"Operational CRM integration is currently unavailable ({crm_state.error_message or 'offline'}). "
                        "Real-time pipeline state, stage transitions, and external CRM data cannot be verified."
                    )
            except Exception as crm_ex:
                logger.warning("CRM fusion failed safely: %s", type(crm_ex).__name__)
                result.crm_status_available = False
                result.crm_notice = (
                    f"Operational CRM integration encountered an unexpected error: {type(crm_ex).__name__}."
                )

        # -------------------------------------------------------------
        # Deduplication and Contradiction Detection across Streams
        # -------------------------------------------------------------
        self._detect_conflicts_and_deduplicate(result)

        # -------------------------------------------------------------
        # Synthesize Grounded Multi-Source Answer
        # -------------------------------------------------------------
        result.formatted_synthesis = self._build_synthesis_text(result, query)
        result.sources_payload = self._build_sources_payload(result)

        return result

    def _check_crm_authorization(
        self,
        user: User,
        allowed_departments: Optional[list[uuid.UUID]],
    ) -> tuple[bool, Optional[str]]:
        """
        Fail-closed CRM authorization gate using existing authorization primitives.
        1. Inactive user fails closed.
        2. ADMIN role bypasses (matching repo's existing admin policy).
        3. Non-admin users require their effective allowed_departments to contain
           the configured twenty_crm_department_id.
        4. If twenty_crm_department_id is unconfigured or invalid, non-admin fails closed.
        5. Workspace-scoped allowed_departments are preserved; global scope is not substituted.
        """
        if not getattr(user, "is_active", True):
            return False, "User account is inactive."

        if getattr(user, "role", None) == "ADMIN":
            return True, None

        crm_dept_id_str = getattr(settings, "twenty_crm_department_id", None)
        if not crm_dept_id_str:
            return False, "CRM access policy is not configured (fail-closed)."

        try:
            crm_dept_uuid = uuid.UUID(str(crm_dept_id_str).strip())
        except (ValueError, TypeError, AttributeError):
            return False, "Invalid CRM department configuration (fail-closed)."

        if allowed_departments is None or crm_dept_uuid not in allowed_departments:
            return False, "You do not have authorization to view live operational CRM data."

        return True, None

    def _detect_conflicts_and_deduplicate(self, result: FusedEvidenceResult) -> None:
        """
        Deduplicates identical statements while preserving distinct events,
        and flags temporal contradictions between older historical records
        and newer authoritative knowledge or live CRM state.
        """
        all_sources = result.tasks + result.company_notes + result.document_chunks + result.crm_records

        # Sort sources temporally if dates exist
        sorted_sources = sorted(
            [s for s in all_sources if s.created_at is not None],
            key=lambda x: x.created_at,
        )

        # Check for direct contradictions (e.g. status flip or budget divergence)
        seen_observations: dict[str, FusedEvidenceSource] = {}
        for s in sorted_sources:
            norm_content = s.content.lower()
            for key, prior in list(seen_observations.items()):
                # Conflict pattern: mentions same key topic but opposite status/state
                if ("discontinued" in norm_content and "active" in prior.content.lower()) or (
                    "cancelled" in norm_content and "agreed" in prior.content.lower()
                ) or (
                    "rejected" in norm_content and "accepted" in prior.content.lower()
                ):
                    s.is_conflict = True
                    prior.is_conflict = True
                    c_msg = (
                        f"Temporal Conflict: Historical source '{prior.title}' states '{prior.content}' "
                        f"whereas newer record '{s.title}' states '{s.content}'."
                    )
                    s.conflict_note = c_msg
                    if c_msg not in result.conflicts_detected:
                        result.conflicts_detected.append(c_msg)
            seen_observations[s.title] = s

    def _build_synthesis_text(self, result: FusedEvidenceResult, query: str) -> str:
        """
        Builds a structured, grounded answer representing all fused evidence streams.
        """
        sections: list[str] = []
        q_lower = query.lower()

        # Check if query asks for operational CRM status
        asks_crm = any(term in q_lower for term in ["crm", "deal stage", "pipeline", "billing", "salesforce", "hubspot", "status", "activity", "deal"])

        sections.append(f"## Evidence Overview for **{result.company_name}**\n")

        # 1. Structured Stage H Tasks & Commitments
        if result.tasks:
            sections.append("### 📋 Structured Tasks & Commitments (Stage H)")
            for t in result.tasks:
                sections.append(f"- **{t.title}**: {t.content}")
        else:
            sections.append("### 📋 Structured Tasks & Commitments (Stage H)")
            sections.append(f"- *No pending structured tasks or follow-ups recorded for {result.company_name}.*")

        # 2. Approved Shared Company Notes & Graph Relationships
        if result.company_notes:
            sections.append("\n### 🏢 Approved Company Notes & Relationships (Knowledge Graph)")
            for n in result.company_notes:
                scope_tag = f"[{n.provenance.get('sharing_scope', 'SHARED')}]"
                sections.append(f"- {scope_tag} **{n.title}**: {n.content}")

        # 3. Hybrid Document Records (Meetings & Decisions)
        if result.document_chunks:
            sections.append("\n### 📄 Document Evidence & Meeting Records")
            for d in result.document_chunks[:4]:
                sections.append(f"- **{d.title}**: {d.content}")

        # 4. Personal User Memory
        if result.personal_memories:
            sections.append("\n### 🧠 Authenticated User Personal Memory")
            for m in result.personal_memories:
                sections.append(f"- *[Private to you]*: {m.content}")

        # 5. Live Operational CRM State
        if result.crm_records:
            sections.append("\n### 📊 Live Operational CRM State (Twenty CRM)")
            sections.append(f"- *Status*: {result.crm_notice}")
            for r in result.crm_records:
                sections.append(f"- **{r.title}**: {r.content}")
        elif asks_crm or ("current" in q_lower and "status" in q_lower) or ("activity" in q_lower):
            sections.append(f"\n### ⚠️ Operational CRM State")
            sections.append(f"- {result.crm_notice}")

        # 6. Contradictions & Conflicts
        if result.conflicts_detected:
            sections.append("\n### ⚠️ Notice: Evidence Conflicts Detected")
            for c in result.conflicts_detected:
                sections.append(f"- {c}")

        return "\n".join(sections)

    def _build_sources_payload(self, result: FusedEvidenceResult) -> list[dict[str, Any]]:
        """
        Assembles UI/API sources payload preserving provenance.
        """
        payload: list[dict[str, Any]] = []

        for t in result.tasks:
            payload.append({
                "type": "structured_task",
                "source_id": t.source_id,
                "title": t.title,
                "excerpt": t.content,
                "provenance": t.provenance,
            })

        for n in result.company_notes:
            payload.append({
                "type": "shared_knowledge",
                "source_id": n.source_id,
                "title": n.title,
                "excerpt": n.content,
                "provenance": n.provenance,
            })

        for d in result.document_chunks:
            payload.append({
                "type": "document",
                "chunk_id": d.source_id,
                "title": d.title,
                "excerpt": d.content,
                "provenance": d.provenance,
            })

        for m in result.personal_memories:
            payload.append({
                "type": "personal_memory",
                "source_id": m.source_id,
                "title": m.title,
                "excerpt": m.content,
                "provenance": m.provenance,
            })

        for r in result.crm_records:
            payload.append({
                "type": "operational_crm",
                "source_id": r.source_id,
                "title": r.title,
                "excerpt": r.content,
                "provenance": r.provenance,
            })

        return payload
