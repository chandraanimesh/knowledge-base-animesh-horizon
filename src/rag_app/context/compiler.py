import logging
import uuid
from rag_app.core.config import get_settings
from rag_app.schemas.rag import ChunkHierarchyMetadata, ChunkProvenance, CompiledContext, ConflictNotice, RetrievedChunk

logger = logging.getLogger(__name__)

MAX_CHUNKS_PER_DOCUMENT = 4
DEFAULT_MAX_CONTEXT_TOKENS = 4000


def estimate_tokens(text: str) -> int:
    """Deterministic token estimator for V1: len(text) / 4."""
    if not text:
        return 0
    return int(len(text) / 4)


class ContextCompiler:
    """
    Stateless ContextCompiler between retrieval and generation.

    Responsibilities:
    1. Final context authorization check (raises on unauthorized chunks).
    2. Final chunk deduplication (chunk_id and normalized text).
    3. Document diversity cap (MAX_CHUNKS_PER_DOCUMENT).
    4. Deterministic relationship conflict detection (same subject+predicate, different object).
    5. Deterministic token-budget enforcement (protects conflicting chunk pairs from truncation).
    6. Preserve provenance/strategy information in CompiledContext.
    7. Synchronize per-chunk provenance sidecar (drops provenance for dropped chunks).
    """

    def __init__(
        self,
        max_context_tokens: int | None = None,
        max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
    ):
        settings = get_settings()
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else getattr(settings, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
        )
        self.max_chunks_per_document = max_chunks_per_document

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
    ) -> CompiledContext:
        """
        Compiles retrieved chunks into a final LLM-ready context.
        """
        # 1. FINAL CONTEXT AUTHORIZATION CHECK
        if not is_web_context and allowed_departments is not None:
            allowed_dept_strs = {str(d) for d in allowed_departments}
            for chunk in chunks:
                if chunk.department_id not in allowed_dept_strs and chunk.department_id != "None":
                    logger.error(
                        "context_compiler_authorization_violation",
                        extra={
                            "event": "context_compiler_authorization_violation",
                            "chunk_id": chunk.chunk_id,
                            "department_id": chunk.department_id,
                        },
                    )
                    raise RuntimeError("Security violation: Unauthorized chunk detected in final context.")

        budget = max_context_tokens if max_context_tokens is not None else self.max_context_tokens

        # 2. FINAL CHUNK DEDUPLICATION & DIVERSITY CAP
        seen_chunk_ids: set[str] = set()
        seen_texts: set[str] = set()
        document_counts: dict[str, int] = {}
        deduped_chunks: list[RetrievedChunk] = []

        for chunk in chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue

            normalized_text = " ".join(chunk.text.lower().split())
            if normalized_text in seen_texts:
                continue

            doc_id = str(chunk.document_id)
            if document_counts.get(doc_id, 0) >= self.max_chunks_per_document:
                continue

            seen_chunk_ids.add(chunk.chunk_id)
            seen_texts.add(normalized_text)
            document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
            deduped_chunks.append(chunk)

        # 3. DETECT CONFLICTS BEFORE TOKEN-BUDGET DROPPING
        conflicts, protected_chunk_ids = self._detect_relationship_divergence(
            deduped_chunks,
            chunk_provenance,
        )

        # 4. DETERMINISTIC TOKEN-BUDGET ENFORCEMENT WITH CONFLICT PROTECTION & HIERARCHY
        protected_chunks = [c for c in deduped_chunks if c.chunk_id in protected_chunk_ids]
        conflict_tokens = sum(estimate_tokens(c.text) for c in protected_chunks)

        compiled_chunks: list[RetrievedChunk] = []
        compiled_hierarchy: dict[str, ChunkHierarchyMetadata] = {}
        seen_overview_nodes: set[str] = set()
        current_tokens = 0

        if budget is not None and conflict_tokens > budget:
            # Conflict pair alone exceeds budget: retain all protected conflict chunks
            # deterministically and record budget condition
            compiled_chunks = protected_chunks
            current_tokens = conflict_tokens
            for notice in conflicts:
                condition_msg = (
                    f" [Budget condition: conflicting chunks total {conflict_tokens} tokens, "
                    f"exceeding configured budget {budget} tokens; retained deterministically to prevent single-sided bias]"
                )
                if condition_msg not in notice.details:
                    notice.details += condition_msg

            logger.warning(
                "conflict_pair_exceeds_budget",
                extra={
                    "event": "conflict_pair_exceeds_budget",
                    "conflict_tokens": conflict_tokens,
                    "budget": budget,
                    "conflicts_count": len(conflicts),
                },
            )
        else:
            remaining_protected_tokens = conflict_tokens

            for chunk in deduped_chunks:
                chunk_tokens = estimate_tokens(chunk.text)
                is_protected = chunk.chunk_id in protected_chunk_ids

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
                        current_tokens += chunk_tokens + header_tokens + overview_tokens
                    else:
                        current_tokens += chunk_tokens
                    remaining_protected_tokens -= chunk_tokens
                else:
                    if hier:
                        full_hier_tokens = header_tokens + overview_tokens
                        if budget is None or (current_tokens + chunk_tokens + full_hier_tokens + remaining_protected_tokens <= budget):
                            # Fits with full hierarchy metadata
                            compiled_chunks.append(chunk)
                            compiled_hierarchy[chunk.chunk_id] = hier
                            if has_overview:
                                seen_overview_nodes.add(hier.node_id)
                            current_tokens += chunk_tokens + full_hier_tokens
                        elif budget is not None and (current_tokens + chunk_tokens + header_tokens + remaining_protected_tokens <= budget):
                            # Budget tight: drop L1 overview first, keep section title/path
                            compiled_chunks.append(chunk)
                            hier_no_overview = hier.model_copy(update={"l1_overview": None})
                            compiled_hierarchy[chunk.chunk_id] = hier_no_overview
                            current_tokens += chunk_tokens + header_tokens
                        elif budget is not None and (current_tokens + chunk_tokens + remaining_protected_tokens <= budget):
                            # Budget very tight: drop all hierarchy metadata to keep evidence
                            compiled_chunks.append(chunk)
                            current_tokens += chunk_tokens
                        else:
                            # Budget exceeded; drop lower-ranked non-conflict tail
                            continue
                    else:
                        if budget is None or (current_tokens + chunk_tokens + remaining_protected_tokens <= budget):
                            compiled_chunks.append(chunk)
                            current_tokens += chunk_tokens
                        else:
                            # Budget exceeded; drop lower-ranked non-conflict tail
                            continue

        # 5. PROVENANCE SUMMARY
        if strategy_trace is not None:
            provenance_summary = dict(strategy_trace)
        else:
            provenance_summary = {
                chunk.file_name: document_counts.get(str(chunk.document_id), 0)
                for chunk in compiled_chunks
            }

        # 6. PER-CHUNK PROVENANCE & HIERARCHY FILTERING (remove entries for dropped chunks)
        final_chunk_ids = {c.chunk_id for c in compiled_chunks}
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

        # 7. RENDERED CONTEXT
        from rag_app.generation.prompt import format_docs
        rendered_context = format_docs(compiled_chunks, chunk_hierarchy=filtered_hierarchy if filtered_hierarchy else None)

        dropped_count = len(chunks) - len(compiled_chunks)

        logger.info(
            "context_compilation_completed",
            extra={
                "event": "context_compilation_completed",
                "input_chunks": len(chunks),
                "compiled_chunks": len(compiled_chunks),
                "token_count": current_tokens,
                "dropped_count": dropped_count,
                "provenance_summary": provenance_summary,
                "provenance_entries": len(filtered_provenance),
                "hierarchy_entries": len(filtered_hierarchy),
                "conflicts_count": len(conflicts),
            },
        )

        return CompiledContext(
            chunks=compiled_chunks,
            token_count=current_tokens,
            dropped_count=dropped_count,
            provenance_summary=provenance_summary,
            chunk_provenance=filtered_provenance,
            conflicts=conflicts,
            chunk_hierarchy=filtered_hierarchy,
            selected_node_ids=selected_node_ids,
            rendered_context=rendered_context,
        )

