import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from rag_app.core.config import get_settings
from rag_app.context.contracts import (
    ContextBudget,
    ContextEvidence,
    SourceAuthority,
    AUTHORITY_WEIGHTS,
)
from rag_app.schemas.rag import (
    ChunkHierarchyMetadata,
    ChunkProvenance,
    CompiledContext,
    ConflictNotice,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

MAX_CHUNKS_PER_DOCUMENT = 5
DEFAULT_MAX_CONTEXT_TOKENS = 4000
DEFAULT_MODEL_CONTEXT_LIMIT = 8192
DEFAULT_RESERVED_OUTPUT_TOKENS = 500
MAX_MEMORY_CHARS = 1200
DEFAULT_MAX_WEB_EVIDENCE_SHARE = 0.40


from rag_app.core.tokenizer import count_text

# Shared model-aware token counter (replaces legacy len/4 heuristic)
estimate_tokens = count_text


class ContextCompiler:
    """
    Unified ContextCompiler and Token Budgeting Engine between retrieval and generation.

    Responsibilities:
    1. Pre-assembly authorization:
       - Hard filter rejecting cross-user personal memories, unauthorized workspace threads,
         unapproved shared knowledge, and cross-department internal chunks.
       - External web evidence is preserved with distinct provenance and cannot bypass RBAC.
    2. Common evidence normalization:
       - Adapts internal chunks, shared knowledge, memories, workspace threads, and web research.
    3. Final deduplication:
       - Deduplicates using stable identity and normalized text without erasing distinct historical events.
    4. Deterministic relationship divergence detection & protection:
       - Preserves conflicting chunk pairs and opposing claims from truncation.
    5. Global token budget enforcement:
       - available_context_tokens = model_context_limit - reserved_output_tokens - other_prompt_tokens
       - Segmented quota for web evidence (max 40% when mixed with internal documents).
       - Deterministic priority allocation: protected conflicts -> essential recent turns ->
         personal memory -> workspace threads -> ranked evidence chunks.
    6. Synchronize per-chunk provenance and hierarchy:
       - Dropped chunks are excluded from CompiledContext.chunks so they cannot be cited.
    7. Observability:
       - Records token estimates, selected evidence counts by source type, dropped counts,
         budget status, and compilation latency.
    """

    def __init__(
        self,
        max_context_tokens: int | None = None,
        max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
        model_context_limit: int | None = None,
        reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
        max_web_evidence_share: float = DEFAULT_MAX_WEB_EVIDENCE_SHARE,
    ):
        settings = get_settings()
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else getattr(settings, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
        )
        self.max_chunks_per_document = max_chunks_per_document
        self.model_context_limit = (
            model_context_limit
            if model_context_limit is not None
            else getattr(settings, "model_context_limit", DEFAULT_MODEL_CONTEXT_LIMIT)
        )
        self.reserved_output_tokens = reserved_output_tokens
        self.max_web_evidence_share = max_web_evidence_share

    @staticmethod
    def adapt_rag_chunk(
        chunk: RetrievedChunk,
        provenance: Optional[ChunkProvenance] = None,
        hierarchy: Optional[ChunkHierarchyMetadata] = None,
    ) -> ContextEvidence:
        is_shared_kn = chunk.chunk_id.startswith("shared_kn_")
        is_web = chunk.chunk_id.startswith("web_")

        if is_shared_kn:
            source_kind = "shared_knowledge"
            authority = SourceAuthority.AUTHORITATIVE_RECORD.value
            is_ext = False
        elif is_web:
            source_kind = "web_research"
            authority = SourceAuthority.WEB_RESEARCH.value
            is_ext = True
        else:
            source_kind = "rag_chunk"
            authority = SourceAuthority.DOCUMENT_EVIDENCE.value
            is_ext = False

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
                "is_external": is_ext,
                **prov_dict,
            },
            access_scope={"department_id": str(chunk.department_id)},
            token_count=token_count,
            is_external=is_ext,
        )

    @staticmethod
    def adapt_memory_record(record: Any) -> ContextEvidence:
        r_id = getattr(record, "id", None) or (record.get("id") if isinstance(record, dict) else str(uuid.uuid4()))
        content = getattr(record, "content", None) or (record.get("content") if isinstance(record, dict) else str(record))
        m_type = getattr(record, "memory_type", None) or (record.get("memory_type") if isinstance(record, dict) else "FACT")
        m_type_str = getattr(m_type, "value", str(m_type)).upper()

        source_kind = "memory_episode" if m_type_str in ("EPISODIC", "DYNAMIC") else "memory_fact"
        score_val = float(getattr(record, "score", 0.8) if hasattr(record, "score") else (record.get("score", 0.8) if isinstance(record, dict) else 0.8))
        token_count = estimate_tokens(content)

        user_id = str(getattr(record, "user_id", None) or (record.get("user_id") if isinstance(record, dict) else ""))
        ws_id = str(getattr(record, "workspace_id", None) or (record.get("workspace_id") if isinstance(record, dict) else ""))
        subject = getattr(record, "subject", None) or (record.get("subject") if isinstance(record, dict) else None)
        predicate = getattr(record, "predicate", None) or (record.get("predicate") if isinstance(record, dict) else None)
        canonical_key = getattr(record, "canonical_key", None) or (record.get("canonical_key") if isinstance(record, dict) else None)

        signals = list(getattr(record, "importance_signals", []) if hasattr(record, "importance_signals") else (record.get("importance_signals", []) if isinstance(record, dict) else []))
        if getattr(record, "stability_tier", None):
            signals.append(f"tier:{record.stability_tier}")

        return ContextEvidence(
            evidence_id=f"mem:{r_id}",
            source_kind=source_kind,
            source_id=str(r_id),
            claim_or_text=content,
            subject=subject,
            predicate=predicate,
            source_authority=SourceAuthority.PERSONAL_MEMORY.value,
            relevance_score=score_val,
            importance_signals=signals,
            provenance={
                "canonical_key": canonical_key,
                "memory_type": m_type_str,
            },
            access_scope={
                "user_id": user_id,
                "workspace_id": ws_id if ws_id else None,
            },
            token_count=token_count,
            is_external=False,
        )

    @staticmethod
    def adapt_thread_context(thread: dict) -> ContextEvidence:
        text = str(thread.get("content") or thread.get("excerpt") or "")
        token_count = estimate_tokens(text)
        t_id = str(thread.get("thread_id") or "")
        m_id = str(thread.get("message_id") or uuid.uuid4())
        ws_id = str(thread.get("workspace_id")) if thread.get("workspace_id") else None

        return ContextEvidence(
            evidence_id=f"thread:{m_id}",
            source_kind="thread",
            source_id=t_id,
            claim_or_text=text,
            source_authority=SourceAuthority.CONVERSATION_THREAD.value,
            relevance_score=0.7,
            provenance={
                "thread_title": thread.get("thread_title"),
                "workspace_name": thread.get("workspace_name"),
                "role": thread.get("role"),
            },
            access_scope={"workspace_id": ws_id},
            token_count=token_count,
            is_external=False,
        )

    def _detect_relationship_divergence(
        self,
        chunks: list[RetrievedChunk],
        chunk_provenance: dict[str, ChunkProvenance] | None,
    ) -> tuple[list[ConflictNotice], set[str]]:
        """
        Detects deterministic relationship divergence:
        same (subject, predicate) with different object values across active GRAPH provenance.
        Returns a list of ConflictNotice and the set of protected chunk IDs.
        """
        if not chunk_provenance:
            return [], set()

        grouped: dict[tuple[str, str], dict[str, list[tuple[str, str, str, str]]]] = {}

        for chunk in chunks:
            prov = chunk_provenance.get(chunk.chunk_id)
            if not prov or not prov.graph_path:
                continue
            for edge in prov.graph_path:
                s = (edge.subject or "").strip()
                p = (edge.predicate or "").strip()
                o = (edge.object or "").strip()
                if not s or not p or not o:
                    continue
                key = (s.lower(), p.lower())
                o_key = o.lower()
                if key not in grouped:
                    grouped[key] = {}
                if o_key not in grouped[key]:
                    grouped[key][o_key] = []
                grouped[key][o_key].append((s, p, o, chunk.chunk_id))

        conflicts: list[ConflictNotice] = []
        protected_chunk_ids: set[str] = set()

        for (s_norm, p_norm), obj_dict in grouped.items():
            if len(obj_dict) > 1:
                raw_s = ""
                raw_p = ""
                raw_objects: list[str] = []
                conflicting_chunk_ids: list[str] = []
                seen_cids: set[str] = set()

                for o_norm, occurrences in obj_dict.items():
                    for s, p, o, cid in occurrences:
                        if not raw_s:
                            raw_s = s
                        if not raw_p:
                            raw_p = p
                        if o not in raw_objects:
                            raw_objects.append(o)
                        if cid not in seen_cids:
                            seen_cids.add(cid)
                            conflicting_chunk_ids.append(cid)
                            protected_chunk_ids.add(cid)

                notice = ConflictNotice(
                    conflict_type="RELATIONSHIP_DIVERGENCE",
                    subject=raw_s or s_norm,
                    predicate=raw_p or p_norm,
                    conflicting_chunk_ids=conflicting_chunk_ids,
                    conflicting_objects=raw_objects,
                    details=f"Conflicting objects detected for ({raw_s or s_norm}, {raw_p or p_norm}): {', '.join(raw_objects)}",
                )
                conflicts.append(notice)

        return conflicts, protected_chunk_ids

    def compile(
        self,
        chunks: list[RetrievedChunk],
        query: str | None = None,
        allowed_departments: list[uuid.UUID] | None = None,
        is_web_context: bool = False,
        strategy_trace: dict[str, int] | None = None,
        chunk_provenance: dict[str, ChunkProvenance] | None = None,
        chunk_hierarchy: dict[str, ChunkHierarchyMetadata] | None = None,
        max_context_tokens: int | None = None,
        # Unified parameters
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
        personal_memories: list[Any] | None = None,
        user: Any | None = None,
        workspace_id: uuid.UUID | None = None,
        model_context_limit: int | None = None,
        reserved_output_tokens: int | None = None,
        max_web_evidence_share: float | None = None,
    ) -> CompiledContext:
        """
        Compiles candidate evidence from document chunks, shared knowledge, personal memories,
        workspace threads, and web research into a bounded LLM context under a Global Token Budget.
        """
        compilation_start = time.perf_counter()
        dropped_reasons: dict[str, str] = {}
        all_candidate_evidence: list[ContextEvidence] = []

        user_id_str: str | None = None
        if user is not None:
            if isinstance(user, (str, uuid.UUID)):
                user_id_str = str(user)
            else:
                sa_state = getattr(user, "_sa_instance_state", None)
                if sa_state is not None and getattr(sa_state, "expired", False):
                    # Expired ORM instance: retrieve PK from identity key tuple without IO
                    if sa_state.key and len(sa_state.key) > 1 and sa_state.key[1]:
                        user_id_str = str(sa_state.key[1][0])
                    else:
                        logger.warning(
                            "context_compiler_expired_user_orm_skipped_io",
                            extra={"event": "context_compiler_expired_user_orm_skipped_io"},
                        )
                elif hasattr(user, "id"):
                    user_id_str = str(user.id)
                else:
                    user_id_str = str(getattr(user, "id", user))

        ws_id_str = str(workspace_id) if workspace_id else None

        # =====================================================================
        # 1. PRE-ASSEMBLY AUTHORIZATION CHECKS (HARD FILTERS)
        # =====================================================================

        # A. Document Chunks Authorization
        authorized_chunks: list[RetrievedChunk] = []
        allowed_dept_strs = {str(d) for d in allowed_departments} if allowed_departments is not None else None

        for chunk in chunks:
            is_chunk_web = str(chunk.chunk_id).startswith("web_")
            chunk_dept = str(chunk.department_id)

            if not is_chunk_web and allowed_dept_strs is not None:
                if chunk_dept not in allowed_dept_strs and chunk_dept != "None":
                    logger.error(
                        "context_compiler_authorization_violation",
                        extra={
                            "event": "context_compiler_authorization_violation",
                            "chunk_id": chunk.chunk_id,
                            "department_id": chunk.department_id,
                        },
                    )
                    raise RuntimeError("Security violation: Unauthorized chunk detected in final context.")

            # B. Shared Knowledge Lifecycle Validation
            if str(chunk.chunk_id).startswith("shared_kn_"):
                # Unapproved shared knowledge must not be compiled into context
                text_lower = (chunk.text or "").lower()
                if "status: unapproved" in text_lower or "status: rejected" in text_lower:
                    dropped_reasons[chunk.chunk_id] = "UNAPPROVED_SHARED_KNOWLEDGE"
                    logger.warning(
                        "unapproved_shared_knowledge_excluded",
                        extra={"chunk_id": chunk.chunk_id},
                    )
                    continue

            authorized_chunks.append(chunk)

        # C. Personal Memory Pre-Assembly Authorization
        # Extract memories if embedded in conversation_history seam or passed directly
        candidate_memories: list[Any] = []
        if personal_memories:
            candidate_memories.extend(personal_memories)

        if conversation_history:
            for msg in conversation_history:
                if msg.get("name") == "personal_memory" or (
                    msg.get("role") == "system" and "[User Personal Preferences & Context]" in msg.get("content", "")
                ):
                    raw_mems = msg.get("personal_memories")
                    if raw_mems and isinstance(raw_mems, list):
                        candidate_memories.extend(raw_mems)
                    else:
                        # Parse lines from content
                        for line in msg.get("content", "").split("\n"):
                            line_str = line.strip()
                            if line_str.startswith("- ["):
                                c_text = line_str.split("]", 1)[-1].strip()
                                candidate_memories.append({
                                    "id": f"mem-{uuid.uuid5(uuid.NAMESPACE_DNS, c_text)}",
                                    "content": c_text,
                                    "user_id": user_id_str,
                                })

        authorized_memories: list[Any] = []
        for mem in candidate_memories:
            mem_user_id = str(getattr(mem, "user_id", None) or (mem.get("user_id") if isinstance(mem, dict) else ""))
            mem_ws_id = str(getattr(mem, "workspace_id", None) or (mem.get("workspace_id") if isinstance(mem, dict) else ""))
            mem_id = str(getattr(mem, "id", None) or (mem.get("id") if isinstance(mem, dict) else uuid.uuid4()))

            # Cross-user memory containment
            if user_id_str and mem_user_id and mem_user_id != user_id_str:
                dropped_reasons[mem_id] = "UNAUTHORIZED_USER_MEMORY"
                logger.warning(
                    "unauthorized_cross_user_memory_excluded",
                    extra={"memory_id": mem_id, "user_id": user_id_str, "memory_owner": mem_user_id},
                )
                continue

            # Workspace containment for workspace-scoped memory
            if ws_id_str and mem_ws_id and mem_ws_id != ws_id_str:
                dropped_reasons[mem_id] = "CROSS_WORKSPACE_MEMORY_EXCLUDED"
                continue

            authorized_memories.append(mem)

        # D. Workspace Thread Context Authorization
        authorized_threads: list[dict] = []
        if workspace_thread_context:
            for t in workspace_thread_context:
                t_ws_id = str(t.get("workspace_id")) if t.get("workspace_id") else None
                t_id = str(t.get("message_id") or t.get("thread_id") or uuid.uuid4())

                # Global Chat containment: no raw employee threads across workspaces
                if ws_id_str is None:
                    dropped_reasons[t_id] = "GLOBAL_CHAT_THREAD_ISOLATION"
                    continue

                if t_ws_id and t_ws_id != ws_id_str:
                    dropped_reasons[t_id] = "CROSS_WORKSPACE_THREAD_EXCLUDED"
                    continue

                authorized_threads.append(t)

        # =====================================================================
        # 2. EVIDENCE NORMALIZATION & DEDUPLICATION
        # =====================================================================

        # Deduplicate chunks by ID, normalized text, and document diversity cap
        seen_chunk_ids: set[str] = set()
        seen_chunk_texts: set[str] = set()
        document_counts: dict[str, int] = {}
        deduped_chunks: list[RetrievedChunk] = []

        for chunk in authorized_chunks:
            if chunk.chunk_id in seen_chunk_ids:
                dropped_reasons[chunk.chunk_id] = "DUPLICATE_CHUNK_ID"
                continue

            normalized_text = " ".join(chunk.text.lower().split())
            if normalized_text in seen_chunk_texts:
                dropped_reasons[chunk.chunk_id] = "DUPLICATE_TEXT"
                continue

            doc_id = str(chunk.document_id)
            if document_counts.get(doc_id, 0) >= self.max_chunks_per_document:
                dropped_reasons[chunk.chunk_id] = "MAX_CHUNKS_PER_DOC_CAP"
                continue

            seen_chunk_ids.add(chunk.chunk_id)
            seen_chunk_texts.add(normalized_text)
            document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
            deduped_chunks.append(chunk)

        # Adapt chunks to common ContextEvidence schema
        for c in deduped_chunks:
            prov = chunk_provenance.get(c.chunk_id) if chunk_provenance else None
            hier = chunk_hierarchy.get(c.chunk_id) if chunk_hierarchy else None
            all_candidate_evidence.append(self.adapt_rag_chunk(c, prov, hier))

        # Adapt memories to common ContextEvidence schema
        seen_mem_ids: set[str] = set()
        seen_mem_texts: set[str] = set()
        deduped_memories: list[Any] = []

        for m in authorized_memories:
            ev = self.adapt_memory_record(m)
            if ev.evidence_id in seen_mem_ids:
                continue
            norm_m = " ".join(ev.claim_or_text.lower().split())
            if norm_m in seen_mem_texts:
                continue
            seen_mem_ids.add(ev.evidence_id)
            seen_mem_texts.add(norm_m)
            deduped_memories.append(m)
            all_candidate_evidence.append(ev)

        # Adapt workspace threads to common ContextEvidence schema
        seen_thread_ids: set[str] = set()
        deduped_threads: list[dict] = []

        for t in authorized_threads:
            ev = self.adapt_thread_context(t)
            if ev.evidence_id in seen_thread_ids:
                continue
            seen_thread_ids.add(ev.evidence_id)
            deduped_threads.append(t)
            all_candidate_evidence.append(ev)

        # =====================================================================
        # 3. CONFLICT DETECTION & PROTECTION
        # =====================================================================
        conflicts, protected_chunk_ids = self._detect_relationship_divergence(
            deduped_chunks,
            chunk_provenance,
        )

        # =====================================================================
        # 4. GLOBAL TOKEN BUDGET ACCOUNTING
        # available_context_tokens = model_context_limit - reserved_output_tokens - other_prompt_tokens
        # =====================================================================
        model_limit = model_context_limit or self.model_context_limit
        output_reserve = reserved_output_tokens or self.reserved_output_tokens
        total_input_capacity = max(0, model_limit - output_reserve)

        # Prompt overhead calculation
        from rag_app.generation.prompt import RAG_SYSTEM_PROMPT
        system_instruction_tokens = estimate_tokens(RAG_SYSTEM_PROMPT)
        query_tokens = estimate_tokens(query or "")
        prompt_overhead_tokens = 150  # Section headers, conflict notices, web guidance

        other_prompt_tokens = system_instruction_tokens + query_tokens + prompt_overhead_tokens
        calculated_available_context = max(0, total_input_capacity - other_prompt_tokens)

        # If max_context_tokens is configured lower (e.g. 4000), respect it as an additional ceiling
        budget_limit = max_context_tokens if max_context_tokens is not None else self.max_context_tokens
        available_context_tokens = min(calculated_available_context, budget_limit)

        # =====================================================================
        # 5. DETERMINISTIC TOKEN ALLOCATION & TRIMMING
        # Priority:
        # A. Protected conflicting chunks
        # B. Recent conversation turns
        # C. Personal memory (capped ~300 tokens)
        # D. Workspace threads
        # E. Ranked evidence chunks (hierarchically compressed if tight)
        # =====================================================================

        # A. Protected conflicting chunks
        protected_chunks = [c for c in deduped_chunks if c.chunk_id in protected_chunk_ids]
        conflict_tokens = sum(estimate_tokens(c.text) for c in protected_chunks)

        # B. Conversation History Trimming
        # Keep only user/assistant turns, exclude raw memory seams
        raw_history_window = [
            msg for msg in (conversation_history or [])
            if msg.get("role") in ("user", "assistant") and msg.get("name") != "personal_memory"
        ]

        compiled_history: list[dict[str, str]] = []
        history_tokens = 0

        # Preserve latest 2 turns (4 messages) with highest priority, then older turns as budget permits
        turns_reversed = list(reversed(raw_history_window))
        temp_history: list[dict[str, str]] = []

        # Target history budget: up to 25% of available context or max 1000 tokens
        max_history_budget = min(int(available_context_tokens * 0.25), 1000)

        for msg in turns_reversed:
            m_tok = estimate_tokens(msg.get("content", ""))
            if history_tokens + m_tok <= max_history_budget or len(temp_history) < 2:
                temp_history.append(msg)
                history_tokens += m_tok
            else:
                dropped_reasons[f"history_turn_{len(raw_history_window) - len(temp_history)}"] = "HISTORY_BUDGET_TRIMMED"

        compiled_history = list(reversed(temp_history))

        # C. Personal Memory Allocation
        # Capped to 300 tokens / 1200 chars
        compiled_memories: list[Any] = []
        memory_tokens = 0
        max_mem_tokens = 300

        for m in deduped_memories:
            c_text = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else str(m))
            m_tok = estimate_tokens(c_text)
            if memory_tokens + m_tok <= max_mem_tokens:
                compiled_memories.append(m)
                memory_tokens += m_tok
            else:
                m_id = str(getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else uuid.uuid4()))
                dropped_reasons[m_id] = "MEMORY_BUDGET_EXCEEDED"

        # D. Workspace Threads Allocation
        compiled_threads: list[dict] = []
        thread_tokens = 0
        max_thread_tokens = min(int(available_context_tokens * 0.15), 500)

        for t in deduped_threads:
            t_tok = estimate_tokens(str(t.get("content") or t.get("excerpt") or ""))
            if thread_tokens + t_tok <= max_thread_tokens:
                compiled_threads.append(t)
                thread_tokens += t_tok
            else:
                t_id = str(t.get("message_id") or t.get("thread_id") or uuid.uuid4())
                dropped_reasons[t_id] = "THREAD_BUDGET_EXCEEDED"

        # E. Evidence Chunks Allocation (remaining context tokens)
        allocated_so_far = history_tokens + memory_tokens + thread_tokens
        available_evidence_tokens = max(0, available_context_tokens - allocated_so_far)

        compiled_chunks: list[RetrievedChunk] = []
        compiled_hierarchy: dict[str, ChunkHierarchyMetadata] = {}
        seen_overview_nodes: set[str] = set()
        current_evidence_tokens = 0

        # Conflict chunks alone exceed evidence budget condition
        if conflict_tokens > available_evidence_tokens and conflict_tokens > 0:
            # The compiler never exceeds the budget merely to preserve every conflict!
            compiled_chunks = []
            current_evidence_tokens = 0
            for pchunk in protected_chunks:
                p_tok = estimate_tokens(pchunk.text)
                if current_evidence_tokens + p_tok <= available_evidence_tokens:
                    compiled_chunks.append(pchunk)
                    current_evidence_tokens += p_tok
                else:
                    dropped_reasons[pchunk.chunk_id] = "CONFLICT_SET_EXCEEDS_BUDGET"

            for notice in conflicts:
                condition_msg = (
                    f" [Budget condition: conflicting chunks total {conflict_tokens} tokens, "
                    f"exceeding evidence budget {available_evidence_tokens} tokens; surfaced in notice but trimmed to prevent budget overflow]"
                )
                if condition_msg not in notice.details:
                    notice.details += condition_msg

            logger.warning(
                "conflict_pair_exceeds_budget",
                extra={
                    "event": "conflict_pair_exceeds_budget",
                    "conflict_tokens": conflict_tokens,
                    "available_evidence_tokens": available_evidence_tokens,
                    "conflicts_count": len(conflicts),
                },
            )
        else:
            remaining_protected_tokens = conflict_tokens

            # Evaluate web evidence quota:
            # When both internal documents and web evidence are present, web evidence is capped to a
            # maximum share of the available evidence budget (default 40%) so internal evidence is not crowded out.
            # In web-only retrieval, web evidence may utilize up to 100% of the available evidence tokens.
            effective_web_share = (
                max_web_evidence_share
                if max_web_evidence_share is not None
                else self.max_web_evidence_share
            )
            has_internal = any(not str(c.chunk_id).startswith("web_") for c in deduped_chunks)
            has_web = any(str(c.chunk_id).startswith("web_") for c in deduped_chunks)

            if has_internal and has_web:
                max_web_tokens = int(available_evidence_tokens * effective_web_share)
            else:
                max_web_tokens = available_evidence_tokens

            current_web_tokens = 0

            # Sort non-conflict chunks by intent/authority ranking
            def chunk_rank_key(c: RetrievedChunk) -> float:
                is_web = str(c.chunk_id).startswith("web_")
                is_shared = str(c.chunk_id).startswith("shared_kn_")
                if is_shared:
                    auth_weight = AUTHORITY_WEIGHTS[SourceAuthority.AUTHORITATIVE_RECORD.value]
                elif is_web:
                    auth_weight = AUTHORITY_WEIGHTS[SourceAuthority.WEB_RESEARCH.value]
                else:
                    auth_weight = AUTHORITY_WEIGHTS[SourceAuthority.DOCUMENT_EVIDENCE.value]
                return float(c.hybrid_score or 0.8) * auth_weight

            sorted_chunks = sorted(deduped_chunks, key=chunk_rank_key, reverse=True)

            for chunk in sorted_chunks:
                chunk_tokens = estimate_tokens(chunk.text)
                is_protected = chunk.chunk_id in protected_chunk_ids
                is_chunk_web = str(chunk.chunk_id).startswith("web_")

                hier = chunk_hierarchy.get(chunk.chunk_id) if chunk_hierarchy else None
                header_tokens = 0
                overview_tokens = 0
                has_overview = False

                if hier:
                    header_str = f"Section: {hier.title}\nPath: {hier.path_display or hier.ordinal_path}\n"
                    header_tokens = estimate_tokens(header_str)
                    if hier.l1_overview and hier.node_id not in seen_overview_nodes:
                        has_overview = True
                        overview_tokens = estimate_tokens(f"Overview: {hier.l1_overview}\n")

                if is_protected:
                    compiled_chunks.append(chunk)
                    if hier:
                        compiled_hierarchy[chunk.chunk_id] = hier
                        if has_overview:
                            seen_overview_nodes.add(hier.node_id)
                        chunk_total = chunk_tokens + header_tokens + overview_tokens
                    else:
                        chunk_total = chunk_tokens
                    current_evidence_tokens += chunk_total
                    if is_chunk_web:
                        current_web_tokens += chunk_total
                    remaining_protected_tokens -= chunk_tokens
                else:
                    if hier:
                        full_hier_tokens = header_tokens + overview_tokens
                        total_cand_tokens = chunk_tokens + full_hier_tokens

                        # Check web quota for non-protected web chunks
                        if is_chunk_web and (current_web_tokens + total_cand_tokens > max_web_tokens):
                            dropped_reasons[chunk.chunk_id] = "WEB_EVIDENCE_QUOTA_EXCEEDED"
                            continue

                        if current_evidence_tokens + total_cand_tokens + remaining_protected_tokens <= available_evidence_tokens:
                            compiled_chunks.append(chunk)
                            compiled_hierarchy[chunk.chunk_id] = hier
                            if has_overview:
                                seen_overview_nodes.add(hier.node_id)
                            current_evidence_tokens += total_cand_tokens
                            if is_chunk_web:
                                current_web_tokens += total_cand_tokens
                        elif current_evidence_tokens + chunk_tokens + header_tokens + remaining_protected_tokens <= available_evidence_tokens:
                            compiled_chunks.append(chunk)
                            hier_no_overview = hier.model_copy(update={"l1_overview": None})
                            compiled_hierarchy[chunk.chunk_id] = hier_no_overview
                            cand_no_overview = chunk_tokens + header_tokens
                            current_evidence_tokens += cand_no_overview
                            if is_chunk_web:
                                current_web_tokens += cand_no_overview
                        elif current_evidence_tokens + chunk_tokens + remaining_protected_tokens <= available_evidence_tokens:
                            compiled_chunks.append(chunk)
                            current_evidence_tokens += chunk_tokens
                            if is_chunk_web:
                                current_web_tokens += chunk_tokens
                        else:
                            dropped_reasons[chunk.chunk_id] = "TOKEN_BUDGET_EXCEEDED"
                            continue
                    else:
                        # Check web quota for non-protected web chunks
                        if is_chunk_web and (current_web_tokens + chunk_tokens > max_web_tokens):
                            dropped_reasons[chunk.chunk_id] = "WEB_EVIDENCE_QUOTA_EXCEEDED"
                            continue

                        if current_evidence_tokens + chunk_tokens + remaining_protected_tokens <= available_evidence_tokens:
                            compiled_chunks.append(chunk)
                            current_evidence_tokens += chunk_tokens
                            if is_chunk_web:
                                current_web_tokens += chunk_tokens
                        else:
                            dropped_reasons[chunk.chunk_id] = "TOKEN_BUDGET_EXCEEDED"
                            continue

        # Preserve stable retrieval ordering among selected chunks
        final_chunk_ids = {c.chunk_id for c in compiled_chunks}
        compiled_chunks = [c for c in deduped_chunks if c.chunk_id in final_chunk_ids]

        filtered_provenance: dict[str, ChunkProvenance] = {}
        if chunk_provenance:
            for cid, prov in chunk_provenance.items():
                if cid in final_chunk_ids:
                    filtered_provenance[cid] = prov

        filtered_hierarchy: dict[str, ChunkHierarchyMetadata] = {}
        if compiled_hierarchy:
            for cid, meta in compiled_hierarchy.items():
                if cid in final_chunk_ids:
                    filtered_hierarchy[cid] = meta

        selected_node_ids = list(dict.fromkeys(meta.node_id for meta in filtered_hierarchy.values()))

        # Provenance summary
        if strategy_trace is not None:
            provenance_summary = dict(strategy_trace)
        else:
            provenance_summary = {
                chunk.file_name: document_counts.get(str(chunk.document_id), 0)
                for chunk in compiled_chunks
            }

        # Build list of selected ContextEvidence objects
        selected_evidence: list[ContextEvidence] = []
        for c in compiled_chunks:
            prov = filtered_provenance.get(c.chunk_id)
            hier = filtered_hierarchy.get(c.chunk_id)
            selected_evidence.append(self.adapt_rag_chunk(c, prov, hier))

        for m in compiled_memories:
            selected_evidence.append(self.adapt_memory_record(m))

        for t in compiled_threads:
            selected_evidence.append(self.adapt_thread_context(t))

        # Render document text via prompt formatter
        from rag_app.generation.prompt import format_docs
        rendered_context = format_docs(compiled_chunks, chunk_hierarchy=filtered_hierarchy if filtered_hierarchy else None)

        # =====================================================================
        # 7. FINAL RECOUNT & BUDGET STATUS
        # =====================================================================
        total_planned_tokens = (
            other_prompt_tokens
            + history_tokens
            + memory_tokens
            + thread_tokens
            + current_evidence_tokens
        )

        if total_planned_tokens <= int(total_input_capacity * 0.85):
            budget_status = "NORMAL"
        elif total_planned_tokens <= total_input_capacity:
            budget_status = "CONSTRAINED"
        else:
            budget_status = "OVERFLOW"

        dropped_count = len(chunks) - len(compiled_chunks)
        compilation_latency_ms = (time.perf_counter() - compilation_start) * 1000

        try:
            from rag_app.core.observability import (
                record_retrieval_metrics,
                record_context_compilation_metrics,
            )
            record_retrieval_metrics(
                context_chunks=len(compiled_chunks),
                context_tokens=current_evidence_tokens,
                final_context_count=len(compiled_chunks),
            )
            record_context_compilation_metrics(
                context_tokens=total_planned_tokens,
                final_context_count=len(compiled_chunks),
                context_budget_status=budget_status,
                dropped_evidence_count=dropped_count,
                compiler_latency_ms=round(compilation_latency_ms, 2),
            )
        except Exception:
            pass

        logger.info(
            "unified_context_compilation_completed",
            extra={
                "event": "unified_context_compilation_completed",
                "input_chunks": len(chunks),
                "compiled_chunks": len(compiled_chunks),
                "evidence_tokens": current_evidence_tokens,
                "history_tokens": history_tokens,
                "memory_tokens": memory_tokens,
                "thread_tokens": thread_tokens,
                "total_planned_tokens": total_planned_tokens,
                "available_context_tokens": available_context_tokens,
                "budget_status": budget_status,
                "dropped_count": dropped_count,
                "conflicts_count": len(conflicts),
                "compilation_latency_ms": round(compilation_latency_ms, 2),
            },
        )

        return CompiledContext(
            chunks=compiled_chunks,
            token_count=current_evidence_tokens,
            dropped_count=dropped_count,
            provenance_summary=provenance_summary,
            chunk_provenance=filtered_provenance,
            conflicts=conflicts,
            chunk_hierarchy=filtered_hierarchy,
            selected_node_ids=selected_node_ids,
            rendered_context=rendered_context,
            budget_status=budget_status,
            selected_evidence=selected_evidence,
            dropped_reasons=dropped_reasons,
            total_planned_tokens=total_planned_tokens,
            available_context_tokens=available_context_tokens,
            compiled_conversation_history=compiled_history,
            compiled_workspace_threads=compiled_threads,
            compiled_memories=compiled_memories,
        )
