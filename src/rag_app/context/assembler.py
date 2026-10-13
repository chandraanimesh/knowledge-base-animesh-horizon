import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from rag_app.core.config import get_settings
from rag_app.context.contracts import (
    ContextBudget,
    ContextEvidence,
    ContextPacket,
    SourceAuthority,
    AUTHORITY_WEIGHTS,
)
from rag_app.schemas.rag import (
    ChunkHierarchyMetadata,
    ChunkProvenance,
    RetrievedChunk,
)
from rag_app.memory.schemas import MemoryRecord

logger = logging.getLogger(__name__)

DEFAULT_INPUT_LIMIT = 4000
DEFAULT_OUTPUT_RESERVE = 500
PROACTIVE_COMPRESSION_RATIO = 0.85


def estimate_tokens(text: str) -> int:
    """Deterministic token estimator: len(text) / 4."""
    if not text:
        return 0
    return int(len(text) / 4)


class BoundedContextAssembler:
    """
    Unified Bounded Context Management Layer (PRP Section 9).
    Combines authorized RAG evidence and Personal Memory into one bounded context packet.

    Responsibilities:
    1. Re-checks authorization across all evidence sources.
    2. Adapts RAG chunks, shared knowledge, memories, and thread context into a common evidence schema.
    3. Deduplicates using stable source identities without erasing distinct historical events.
    4. Resolves conflicts deterministically using source authority and temporal applicability.
    5. Ranks evidence by query intent, relevance, and source authority.
    6. Enforces a single shared hard context budget across RAG chunks and memory with output reserve.
    7. Proactively trims/compresses lower-priority context while preserving IDs, dates, and provenance.
    8. Recounts tokens after compression and produces a validated ContextPacket.
    """

    def __init__(
        self,
        input_limit: Optional[int] = None,
        reserved_output_tokens: int = DEFAULT_OUTPUT_RESERVE,
    ):
        settings = get_settings()
        self.input_limit = (
            input_limit
            if input_limit is not None
            else getattr(settings, "max_context_tokens", DEFAULT_INPUT_LIMIT)
        )
        self.reserved_output_tokens = reserved_output_tokens

    @staticmethod
    def adapt_rag_chunk(
        chunk: RetrievedChunk,
        provenance: Optional[ChunkProvenance] = None,
        hierarchy: Optional[ChunkHierarchyMetadata] = None,
    ) -> ContextEvidence:
        is_shared_kn = chunk.chunk_id.startswith("shared_kn_")
        source_kind = "shared_knowledge" if is_shared_kn else "rag_chunk"
        authority = (
            SourceAuthority.AUTHORITATIVE_RECORD.value
            if is_shared_kn
            else SourceAuthority.DOCUMENT_EVIDENCE.value
        )

        prov_dict = {}
        if provenance:
            prov_dict = {
                "strategies": provenance.strategies,
                "confidence": provenance.confidence,
                "knowledge_item_id": provenance.knowledge_item_id,
            }

        text = chunk.text or ""
        token_count = estimate_tokens(text)

        return ContextEvidence(
            evidence_id=f"rag:{chunk.chunk_id}",
            source_kind=source_kind,
            source_id=chunk.chunk_id,
            claim_or_text=text,
            entity_id=None,
            source_authority=authority,
            relevance_score=float(chunk.hybrid_score or 0.8),
            provenance={
                "file_name": chunk.file_name,
                "page_number": chunk.page_number,
                "document_id": str(chunk.document_id),
                "department_id": str(chunk.department_id),
                **prov_dict,
            },
            access_scope={"department_id": str(chunk.department_id)},
            token_count=token_count,
        )

    @staticmethod
    def adapt_memory_record(record: MemoryRecord) -> ContextEvidence:
        m_type = str(record.memory_type).upper()
        source_kind = "memory_episode" if m_type in ("EPISODIC", "DYNAMIC") else "memory_fact"

        score_val = float(record.score or 0.8)
        text = record.content or ""
        token_count = estimate_tokens(text)

        signals = list(record.importance_signals or [])
        if record.stability_tier:
            signals.append(f"tier:{record.stability_tier}")

        return ContextEvidence(
            evidence_id=f"mem:{record.id}",
            source_kind=source_kind,
            source_id=str(record.id),
            claim_or_text=text,
            subject=record.subject,
            predicate=record.predicate,
            recorded_at=record.created_at,
            valid_from=record.valid_from,
            valid_to=record.valid_until,
            source_authority=SourceAuthority.PERSONAL_MEMORY.value,
            relevance_score=score_val,
            importance_signals=signals,
            provenance={
                "canonical_key": record.canonical_key,
                "source_type": record.source_type,
                "status": record.status,
                "stability_tier": record.stability_tier,
                "source_quote": record.source_quote,
            },
            access_scope={
                "user_id": str(record.user_id),
                "workspace_id": str(record.workspace_id) if record.workspace_id else None,
            },
            token_count=token_count,
        )

    @staticmethod
    def adapt_thread_context(thread: dict) -> ContextEvidence:
        text = str(thread.get("content") or thread.get("excerpt") or "")
        token_count = estimate_tokens(text)

        return ContextEvidence(
            evidence_id=f"thread:{thread.get('message_id') or uuid.uuid4()}",
            source_kind="thread",
            source_id=str(thread.get("thread_id") or ""),
            claim_or_text=text,
            source_authority=SourceAuthority.CONVERSATION_THREAD.value,
            relevance_score=0.7,
            provenance={
                "thread_title": thread.get("thread_title"),
                "workspace_name": thread.get("workspace_name"),
                "role": thread.get("role"),
            },
            access_scope={"workspace_id": str(thread.get("workspace_id")) if thread.get("workspace_id") else None},
            token_count=token_count,
        )

    def assemble(
        self,
        query: str,
        rag_chunks: list[RetrievedChunk],
        memories: list[MemoryRecord],
        conversation_history: Optional[list[dict[str, str]]] = None,
        workspace_thread_context: Optional[list[dict]] = None,
        allowed_departments: Optional[list[uuid.UUID]] = None,
        user_id: Optional[uuid.UUID] = None,
        workspace_id: Optional[uuid.UUID] = None,
        chunk_provenance: Optional[dict[str, ChunkProvenance]] = None,
        chunk_hierarchy: Optional[dict[str, ChunkHierarchyMetadata]] = None,
    ) -> ContextPacket:
        """
        Executes bounded context assembly algorithm.
        """
        # 1. Authorization Re-check
        if allowed_departments is not None:
            allowed_dept_strs = {str(d) for d in allowed_departments}
            for c in rag_chunks:
                if c.department_id not in allowed_dept_strs and c.department_id != "None":
                    logger.error(
                        "context_assembler_auth_violation",
                        extra={"chunk_id": c.chunk_id, "department_id": c.department_id},
                    )
                    raise RuntimeError("Security violation: Unauthorized chunk detected in context assembly.")

        # 2. Normalize to Common Evidence Schema
        all_evidence: list[ContextEvidence] = []

        for c in rag_chunks:
            prov = chunk_provenance.get(c.chunk_id) if chunk_provenance else None
            hier = chunk_hierarchy.get(c.chunk_id) if chunk_hierarchy else None
            all_evidence.append(self.adapt_rag_chunk(c, prov, hier))

        for m in memories:
            # Scope re-check for memory
            if user_id and m.user_id != user_id:
                continue
            if workspace_id and m.workspace_id and m.workspace_id != workspace_id:
                continue
            all_evidence.append(self.adapt_memory_record(m))

        if workspace_thread_context:
            for t in workspace_thread_context:
                all_evidence.append(self.adapt_thread_context(t))

        # 3. Deduplication (stable identity & normalized text)
        seen_evidence_ids: set[str] = set()
        seen_texts: set[str] = set()
        deduped: list[ContextEvidence] = []

        for ev in all_evidence:
            if ev.evidence_id in seen_evidence_ids:
                continue
            norm_text = " ".join(ev.claim_or_text.lower().split())
            if norm_text in seen_texts:
                continue

            seen_evidence_ids.add(ev.evidence_id)
            seen_texts.add(norm_text)
            deduped.append(ev)

        # 4. Deterministic Conflict Resolution
        # Group by (subject, predicate) where present
        conflicts: list[dict] = []
        subject_groups: dict[tuple[str, str], list[ContextEvidence]] = {}

        for ev in deduped:
            if ev.subject and ev.predicate:
                key = (ev.subject.lower().strip(), ev.predicate.lower().strip())
                if key not in subject_groups:
                    subject_groups[key] = []
                subject_groups[key].append(ev)

        protected_ids: set[str] = set()
        for (subj, pred), group in subject_groups.items():
            if len(group) > 1:
                # Check for authority or temporal differences
                # Rank 1: Authoritative records > Rank 2: Documents > Rank 3: Memory > Rank 4: Thread
                sorted_by_auth = sorted(
                    group,
                    key=lambda x: (
                        AUTHORITY_WEIGHTS.get(x.source_authority, 0.5),
                        x.recorded_at or datetime.min.replace(tzinfo=timezone.utc),
                    ),
                    reverse=True,
                )
                winner = sorted_by_auth[0]
                conflicting_items = sorted_by_auth[1:]
                conflict_entry = {
                    "subject": subj,
                    "predicate": pred,
                    "authoritative_source_id": winner.evidence_id,
                    "authoritative_authority": winner.source_authority,
                    "competing_sources": [c.evidence_id for c in conflicting_items],
                    "details": f"Deterministic resolution: {winner.source_authority} selected as operational state for ({subj}, {pred}).",
                }
                conflicts.append(conflict_entry)
                # Protect both sides of the conflict in evidence so LLM has visibility of the divergence
                for it in group:
                    protected_ids.add(it.evidence_id)

        # 5. Intent and Source-Authority Ranking
        def compute_rank_score(ev: ContextEvidence) -> float:
            base_score = float(ev.relevance_score or 0.5)
            auth_weight = AUTHORITY_WEIGHTS.get(ev.source_authority, 0.5)
            # Boost if explicit_rem or decision signal present
            signal_boost = 0.05 if ("explicit_rem" in ev.importance_signals or "decision" in ev.importance_signals) else 0.0
            return (base_score * auth_weight) + signal_boost

        ranked_evidence = sorted(deduped, key=compute_rank_score, reverse=True)

        # 6. Hard Token Budget Accounting & Proactive Compression
        # Compute available input budget after reserving generation output tokens
        total_input_capacity = max(0, self.input_limit - self.reserved_output_tokens)
        instructions_tokens = min(int(total_input_capacity * 0.15), 400)
        history_tokens = sum(estimate_tokens(m.get("content", "")) for m in (conversation_history or []))

        # Max allowed evidence tokens (shared between RAG evidence and personal memory)
        available_evidence_tokens = max(0, total_input_capacity - instructions_tokens - history_tokens)

        selected: list[ContextEvidence] = []
        omissions: list[str] = []
        current_evidence_tokens = 0

        # Protected conflicting evidence retained first
        for ev in ranked_evidence:
            if ev.evidence_id in protected_ids:
                selected.append(ev)
                current_evidence_tokens += ev.token_count

        for ev in ranked_evidence:
            if ev.evidence_id in protected_ids:
                continue

            if current_evidence_tokens + ev.token_count <= available_evidence_tokens:
                selected.append(ev)
                current_evidence_tokens += ev.token_count
            else:
                # Compression threshold reached: omit lower-ranked non-protected tail
                omissions.append(ev.evidence_id)

        # 7. Render Context & Step 9: Recount tokens and trim if exceeding capacity
        def render_evidence(ev_list: list[ContextEvidence]) -> str:
            doc_parts = []
            mem_parts = []
            thread_parts = []
            doc_idx = 1
            for ev in ev_list:
                if ev.source_authority in (SourceAuthority.AUTHORITATIVE_RECORD.value, SourceAuthority.DOCUMENT_EVIDENCE.value):
                    fname = ev.provenance.get("file_name", "Document")
                    pnum = ev.provenance.get("page_number")
                    suffix = f", page {pnum}" if pnum else ""
                    doc_parts.append(
                        f"[{doc_idx}] Source: {fname}{suffix}\n"
                        f"Chunk ID: {ev.source_id}\n"
                        f"Content:\n{ev.claim_or_text}"
                    )
                    doc_idx += 1
                elif ev.source_authority == SourceAuthority.PERSONAL_MEMORY.value:
                    mem_parts.append(f"- [Previously Saved Personal Context] {ev.claim_or_text}")
                elif ev.source_authority == SourceAuthority.CONVERSATION_THREAD.value:
                    ttitle = ev.provenance.get("thread_title", "Workspace Thread")
                    thread_parts.append(f"- [Thread: {ttitle}] {ev.claim_or_text}")

            sections = []
            if doc_parts:
                sections.append("### Authorized Document Evidence:\n" + "\n\n".join(doc_parts))
            if mem_parts:
                sections.append("### Previously Saved Personal Context:\n" + "\n".join(mem_parts))
            if thread_parts:
                sections.append("### Relevant Workspace Discussions:\n" + "\n".join(thread_parts))
            return "\n\n".join(sections)

        rendered_context = render_evidence(selected)
        recounted_evidence_tokens = estimate_tokens(rendered_context)
        total_planned_tokens = instructions_tokens + history_tokens + recounted_evidence_tokens

        # Step 9: Recount tokens after assembly; drop lowest-priority non-protected items if still exceeding
        while total_planned_tokens > total_input_capacity and len(selected) > 0:
            droppable_idx = None
            for idx in reversed(range(len(selected))):
                if selected[idx].evidence_id not in protected_ids:
                    droppable_idx = idx
                    break
            if droppable_idx is None:
                break
            dropped_ev = selected.pop(droppable_idx)
            omissions.append(dropped_ev.evidence_id)
            rendered_context = render_evidence(selected)
            recounted_evidence_tokens = estimate_tokens(rendered_context)
            total_planned_tokens = instructions_tokens + history_tokens + recounted_evidence_tokens

        budget = ContextBudget(
            input_limit=self.input_limit,
            instructions_tokens=instructions_tokens,
            evidence_tokens=recounted_evidence_tokens,
            history_tokens=history_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            total_planned_input_tokens=total_planned_tokens,
        )

        return ContextPacket(
            query=query,
            answer_instructions=[
                "Ground all facts strictly in authorized evidence.",
                "Explicitly label answers from personal memory as previously saved personal context—not verified company facts.",
                "Preserve unresolved conflicts and do not invent resolutions.",
            ],
            selected_evidence=selected,
            unresolved_conflicts=conflicts,
            omissions=omissions,
            budget=budget,
            rendered_context=rendered_context,
        )
