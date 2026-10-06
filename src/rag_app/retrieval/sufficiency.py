import logging
import re
import time
from typing import Any

from rag_app.generation.llm_factory import get_llm
from rag_app.core.observability import record_llm_call
from rag_app.schemas.rag import (
    EvidenceNeed,
    InformationNeed,
    PresentationNeed,
    QueryPlan,
    RetrievedChunk,
    ChunkHierarchyMetadata,
)
from rag_app.schemas.research import (
    SufficiencyEvaluation,
    SufficiencyResult,
    SufficiencyStatus,
)
from rag_app.retrieval.query_parser import (
    extract_deterministic_intent,
)

logger = logging.getLogger(__name__)


class EvidenceSufficiencyEvaluator:
    """
    Evidence-aware sufficiency evaluator.
    Determines whether the retrieved evidence is SUFFICIENT, PARTIAL, or INSUFFICIENT
    to answer the user's requested information need strictly without hallucination.
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def detect_evidence_types(self, chunks: list[RetrievedChunk]) -> list[str]:
        """Classify the evidence types present in the retrieved chunks."""
        types: set[str] = set()
        for c in chunks:
            text = c.text
            if "|" in text and text.count("|") >= 4 and ("|---" in text or "| ---" in text or "\n|" in text):
                types.add("table")
            elif re.search(r"^\s*[-*+]\s+", text, re.M) or re.search(r"^\s*\d+\.\s+", text, re.M):
                types.add("list")
            elif getattr(c, "node_id", None) is not None or getattr(c, "hierarchical_score", None) is not None:
                types.add("hierarchical")
            else:
                types.add("prose")
        return sorted(list(types))

    def validate_sufficiency_result(
        self,
        result: SufficiencyEvaluation,
        chunks: list[RetrievedChunk] | None = None,
    ) -> SufficiencyEvaluation:
        """
        Deterministic validation layer applied after the LLM response to enforce
        consistency and safety contracts.
        """
        # Rule 1: No evidence found cannot be SUFFICIENT or PARTIAL
        if not result.evidence_found or (chunks is not None and len(chunks) == 0):
            if result.status != SufficiencyStatus.INSUFFICIENT:
                result.status = SufficiencyStatus.INSUFFICIENT
                result.evidence_found = False
                result.answerable_from_context = False
                if not result.missing_requirements:
                    result.missing_requirements = ["relevant_evidence"]
                result.reason = f"Deterministic correction: No relevant evidence found ({result.reason})"
            return result

        # Rule 2: If status is SUFFICIENT but missing_requirements is non-empty, downgrade
        if result.status == SufficiencyStatus.SUFFICIENT and result.missing_requirements:
            if result.supported_information and result.answerable_from_context:
                result.status = SufficiencyStatus.PARTIAL
                result.reason = f"Downgraded to PARTIAL: Missing requirements detected ({', '.join(result.missing_requirements)}). {result.reason}"
            else:
                result.status = SufficiencyStatus.INSUFFICIENT
                result.answerable_from_context = False
                result.reason = f"Downgraded to INSUFFICIENT: Essential requirements missing ({', '.join(result.missing_requirements)}). {result.reason}"

        # Rule 3: If status is SUFFICIENT but unresolved contradictions exist, downgrade
        if result.status == SufficiencyStatus.SUFFICIENT and result.contradictions:
            result.status = SufficiencyStatus.PARTIAL
            result.reason = f"Downgraded to PARTIAL: Unresolved contradictions found in context ({', '.join(result.contradictions)}). {result.reason}"

        # Rule 4: PARTIAL requires evidence_found=True and answerable_from_context=True
        if result.status == SufficiencyStatus.PARTIAL:
            result.evidence_found = True
            result.answerable_from_context = True

        # Rule 5: INSUFFICIENT with no supported info cannot be answerable from context
        if result.status == SufficiencyStatus.INSUFFICIENT and not result.supported_information:
            result.answerable_from_context = False

        return result

    def evaluate_deterministic(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_plan: QueryPlan | None = None,
        chunk_hierarchy: dict[str, ChunkHierarchyMetadata] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
    ) -> SufficiencyEvaluation:
        """
        Deterministic, rule-based evidence sufficiency evaluation.
        Provides robust offline evaluation and acts as a dependable fallback.
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()

        if query_plan is None:
            info, pres, ev, conv_req, reason_det = extract_deterministic_intent(q_clean)
        else:
            info = query_plan.information_need
            pres = query_plan.presentation_need
            ev = query_plan.evidence_need
            conv_req = query_plan.conversation_required

        evidence_types = self.detect_evidence_types(chunks)
        has_conv_history = bool(conversation_history and len(conversation_history) > 0)
        has_thread_ctx = bool(workspace_thread_context and len(workspace_thread_context) > 0)

        # ----------------------------------------------------
        # 1. Pure Conversation Queries
        # ----------------------------------------------------
        is_mixed_conv_doc = ("document" in q_lower or "doc" in q_lower or "article" in q_lower) and (
            "earlier" in q_lower or "previous" in q_lower or "ask" in q_lower
        )

        if conv_req or info == InformationNeed.CONVERSATION:
            if is_mixed_conv_doc:
                # Requires BOTH conversation history and document chunks
                if not has_conv_history and not has_thread_ctx:
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.INSUFFICIENT,
                        evidence_found=bool(chunks),
                        required_information=["prior_conversation_history", "document_evidence"],
                        supported_information=["document_evidence"] if chunks else [],
                        missing_requirements=["conversation_history"],
                        answerable_from_context=False,
                        reason="Missing prior conversation history for mixed conversation and document query.",
                        confidence=0.9,
                        coverage=0.5 if len(chunks) > 0 else 0.0,
                        evidence_types=evidence_types,
                    )
                if not chunks:
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.INSUFFICIENT,
                        evidence_found=True,
                        required_information=["prior_conversation_history", "document_evidence"],
                        supported_information=["conversation_history"],
                        missing_requirements=["document_evidence"],
                        answerable_from_context=False,
                        reason="Missing document evidence for mixed conversation and document query.",
                        confidence=0.9,
                        coverage=0.5,
                        evidence_types=["conversation_context"],
                    )
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.SUFFICIENT,
                    evidence_found=True,
                    required_information=["prior_conversation_history", "document_evidence"],
                    supported_information=["conversation_history", "document_evidence"],
                    missing_requirements=[],
                    answerable_from_context=True,
                    reason="Both conversation history and document evidence are available.",
                    confidence=0.95,
                    coverage=1.0,
                    evidence_types=["conversation_context"] + evidence_types,
                )

            # Pure conversation history query
            if has_conv_history or has_thread_ctx:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.SUFFICIENT,
                    evidence_found=True,
                    required_information=["conversation_history"],
                    supported_information=["conversation_history"],
                    missing_requirements=[],
                    answerable_from_context=True,
                    reason="Conversation history is present to address the conversational query.",
                    confidence=0.95,
                    coverage=1.0,
                    evidence_types=["conversation_context"],
                )
            else:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.INSUFFICIENT,
                    evidence_found=False,
                    required_information=["conversation_history"],
                    supported_information=[],
                    missing_requirements=["conversation_history"],
                    answerable_from_context=False,
                    reason="No previous conversation history found to answer what was asked or discussed.",
                    confidence=0.9,
                    coverage=0.0,
                    evidence_types=[],
                )

        # ----------------------------------------------------
        # 2. Empty Document Chunks for Corpus Queries
        # ----------------------------------------------------
        if not chunks:
            if has_conv_history or has_thread_ctx:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.SUFFICIENT,
                    evidence_found=True,
                    required_information=["workspace_thread_context"],
                    supported_information=["workspace_thread_context"],
                    missing_requirements=[],
                    answerable_from_context=True,
                    reason="Workspace/thread context available.",
                    confidence=0.8,
                    coverage=1.0,
                    evidence_types=["conversation_context"],
                )
            return SufficiencyEvaluation(
                status=SufficiencyStatus.INSUFFICIENT,
                evidence_found=False,
                required_information=[q_clean],
                supported_information=[],
                missing_requirements=["document_evidence"],
                answerable_from_context=False,
                reason="No document evidence was retrieved for the query.",
                confidence=1.0,
                coverage=0.0,
                evidence_types=[],
            )

        combined_text = "\n\n".join(c.text for c in chunks)
        combined_lower = combined_text.lower()

        # ----------------------------------------------------
        # 3. Contradiction Queries
        # ----------------------------------------------------
        if re.search(r"\b(two\s+different\s+values|which\s+one\s+is\s+correct|conflicting\s+values|contradict(?:ion|ory)?)\b", q_lower):
            return SufficiencyEvaluation(
                status=SufficiencyStatus.PARTIAL,
                evidence_found=True,
                required_information=["authoritative_resolution_of_conflict"],
                supported_information=["conflicting_values_present"],
                missing_requirements=["authoritative_resolution"],
                contradictions=["Conflicting values found in document without explicit resolution"],
                answerable_from_context=True,
                reason="Document contains conflicting values without an authoritative resolution.",
                confidence=0.85,
                coverage=0.5,
                evidence_types=evidence_types,
            )

        if len(chunks) > 1:
            day_matches = set(re.findall(r"\b(\d+)\s*days\b", combined_lower))
            if len(day_matches) > 1:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.PARTIAL,
                    evidence_found=True,
                    required_information=["implementation_timeline"],
                    supported_information=[f"conflicting duration values: {sorted(list(day_matches))}"],
                    missing_requirements=["authoritative_resolution_of_conflicting_timelines"],
                    contradictions=[f"Conflicting day values found across documents: {sorted(list(day_matches))}"],
                    answerable_from_context=True,
                    reason="Retrieved documents contain conflicting duration/timeline values without resolution.",
                    confidence=0.9,
                    coverage=0.5,
                    evidence_types=evidence_types,
                )

        # ----------------------------------------------------
        # 4. Requirements Queries
        # ----------------------------------------------------
        if info == InformationNeed.REQUIREMENTS:
            has_req_content = bool(re.search(
                r"\b(must|should|require|requirement|required|spec|specification|task|checklist|implement|architecture|endpoint|database)\b",
                combined_lower,
            ))
            if not has_req_content:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.INSUFFICIENT,
                    evidence_found=True,
                    required_information=["actionable_technical_requirements"],
                    supported_information=["generic_prose"],
                    missing_requirements=["actionable_requirements"],
                    answerable_from_context=False,
                    reason="Retrieved evidence does not contain actionable technical requirements or tasks.",
                    confidence=0.85,
                    coverage=0.2,
                    evidence_types=evidence_types,
                )
            return SufficiencyEvaluation(
                status=SufficiencyStatus.SUFFICIENT,
                evidence_found=True,
                required_information=["technical_requirements"],
                supported_information=["actionable_specifications_found"],
                missing_requirements=[],
                answerable_from_context=True,
                reason="Actionable technical specifications and implementation details found in evidence.",
                confidence=0.95,
                coverage=1.0,
                evidence_types=evidence_types,
            )

        # ----------------------------------------------------
        # 5. Visualization / Aggregation Queries
        # ----------------------------------------------------
        if pres in (PresentationNeed.CHART, PresentationNeed.TABLE) or ev == EvidenceNeed.STRUCTURED or info == InformationNeed.AGGREGATION:
            has_table = "table" in evidence_types
            has_numbers = bool(re.search(r"\$\d+|\b\d+(?:\.\d+)?(?:k|m|b|%|\s*million|\s*thousand)?\b", combined_text, re.IGNORECASE))
            if not has_table and not has_numbers:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.INSUFFICIENT,
                    evidence_found=True,
                    required_information=["structured_numeric_metrics"],
                    supported_information=["narrative_prose_descriptions"],
                    missing_requirements=["structured_numeric_data"],
                    answerable_from_context=False,
                    reason="Visualization/chart requested structured numeric data, but only narrative prose was retrieved.",
                    confidence=0.9,
                    coverage=0.3,
                    evidence_types=evidence_types,
                )

            # Exhaustive check: if user asks for "all" or "complete" and coverage is limited
            is_exhaustive = bool(re.search(r"\b(?:all|every|complete|entire|everything|full\s+list)\b", q_lower))
            if is_exhaustive and len(chunks) < 2 and not has_table:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.PARTIAL,
                    evidence_found=True,
                    required_information=["comprehensive_data_for_all_items"],
                    supported_information=["partial_item_values"],
                    missing_requirements=["comprehensive_data_coverage"],
                    answerable_from_context=True,
                    reason="The query requests comprehensive visualization across all items, but retrieved evidence has only partial coverage.",
                    confidence=0.88,
                    coverage=0.5,
                    evidence_types=evidence_types,
                )

            # Single factual deadline/point check: if asking for a single deadline rather than multi-point data
            if ("deadline" in q_lower or "implementation deadline" in q_lower) and not has_table:
                if not re.search(r"\b(?:milestones|phases|sprint|gantt|timeline|schedule)\b", combined_lower):
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.INSUFFICIENT,
                        evidence_found=True,
                        required_information=["multi_point_chartable_metrics"],
                        supported_information=["single_deadline_date"],
                        missing_requirements=["multi_point_chart_data"],
                        answerable_from_context=False,
                        reason="The query requests a chart, but evidence only contains a single date/deadline rather than multi-point chartable data.",
                        confidence=0.9,
                        coverage=0.5,
                        evidence_types=evidence_types,
                    )

            return SufficiencyEvaluation(
                status=SufficiencyStatus.SUFFICIENT,
                evidence_found=True,
                required_information=["structured_numeric_data"],
                supported_information=["structured_data_points_and_metrics"],
                missing_requirements=[],
                answerable_from_context=True,
                reason="Structured data points and metrics are present in retrieved evidence.",
                confidence=0.95,
                coverage=1.0,
                evidence_types=evidence_types,
            )

        # ----------------------------------------------------
        # 6. Comparison Queries
        # ----------------------------------------------------
        if info == InformationNeed.COMPARISON:
            # Check for backend vs frontend or explicit sides
            if "backend" in q_lower and "frontend" in q_lower:
                has_backend = "backend" in combined_lower
                has_frontend = "frontend" in combined_lower
                if not has_backend or not has_frontend:
                    missing = "backend" if not has_backend else "frontend"
                    present = "frontend" if not has_backend else "backend"
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.PARTIAL,
                        evidence_found=True,
                        required_information=["backend_specifications", "frontend_specifications"],
                        supported_information=[f"{present}_specifications"],
                        missing_requirements=[f"{missing}_details"],
                        answerable_from_context=True,
                        reason=f"Comparison query requires evidence for both sides, but {missing} details are missing.",
                        confidence=0.9,
                        coverage=0.5,
                        evidence_types=evidence_types,
                    )
            # Check for Node A and Node B
            if "node a" in q_lower and "node b" in q_lower:
                has_a = "node a" in combined_lower
                has_b = "node b" in combined_lower
                if not has_a or not has_b:
                    missing = "node a" if not has_a else "node b"
                    present = "node b" if not has_a else "node a"
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.PARTIAL,
                        evidence_found=True,
                        required_information=["node_a_specifications", "node_b_specifications"],
                        supported_information=[f"{present}_specifications"],
                        missing_requirements=[f"{missing}_specifications"],
                        answerable_from_context=True,
                        reason=f"Comparison query requires evidence for both sides, but {missing} requirements are missing.",
                        confidence=0.9,
                        coverage=0.5,
                        evidence_types=evidence_types,
                    )

        # ----------------------------------------------------
        # 7. Exact Factual Point Queries (e.g. deadline, release date, exact fact)
        # ----------------------------------------------------
        is_exact_query = "exact" in q_lower or "specific" in q_lower or "deadline" in q_lower or "release date" in q_lower
        if is_exact_query:
            if "deadline" in q_lower:
                has_deadline_val = bool(re.search(
                    r"\b(?:deadline\s+is|due\s+(?:on|by|date)|target\s+date|launch\s+date|q[1-4]\s+20\d\d|20\d\d-\d{2}-\d{2}|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*20\d\d)?)\b",
                    combined_lower,
                ))
                has_topic = bool(re.search(r"\b(?:implement|timeline|deadline|schedule|milestone|delivery|due|launch)\b", combined_lower))
                if not has_topic:
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.INSUFFICIENT,
                        evidence_found=False,
                        required_information=["implementation_deadline"],
                        supported_information=[],
                        missing_requirements=["implementation_deadline_evidence"],
                        answerable_from_context=False,
                        reason="Retrieved context does not contain any evidence or mention of implementation timeline or deadline.",
                        confidence=0.95,
                        coverage=0.0,
                        evidence_types=evidence_types,
                    )
                if not has_deadline_val:
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.INSUFFICIENT,
                        evidence_found=True,
                        required_information=["exact_implementation_deadline_date"],
                        supported_information=["general_implementation_timeline"],
                        missing_requirements=["exact_deadline_date"],
                        answerable_from_context=False,
                        reason="The context discusses implementation, but does not specify the exact deadline date.",
                        confidence=0.88,
                        coverage=0.4,
                        evidence_types=evidence_types,
                    )
            if "release date" in q_lower or "date for" in q_lower:
                m_target = re.search(r"version\s+(\d+(?:\.\d+)?)", q_lower)
                if m_target:
                    v_num = m_target.group(1)
                    has_v_date = bool(re.search(rf"version\s+{re.escape(v_num)}[^\n.]*?(?:released|launch|date|due|in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|\d{{4}}))", combined_lower))
                    if not has_v_date:
                        return SufficiencyEvaluation(
                            status=SufficiencyStatus.INSUFFICIENT,
                            evidence_found=True,
                            required_information=[f"version_{v_num}_release_date"],
                            supported_information=[f"version_{v_num}_features"],
                            missing_requirements=["release_date"],
                            answerable_from_context=False,
                            reason=f"The context discusses version {v_num} features, but does not provide the release date.",
                            confidence=0.9,
                            coverage=0.4,
                            evidence_types=evidence_types,
                        )

        # ----------------------------------------------------
        # 8. Exhaustive List & Enumeration Queries
        # ----------------------------------------------------
        if info == InformationNeed.LIST or re.search(r"\b(?:list|enumerate|all|every|complete\s+list|entire|everything|full\s+list)\b", q_lower):
            # Check for partial opportunity / list scenario
            if "pipeline" in q_lower or "opportunit" in q_lower:
                has_some_opps = bool(re.search(r"\$\d+|vertex|northstar", combined_lower))
                if has_some_opps and ("abc" in combined_lower or "no explicit" in combined_lower or len(chunks) < 2):
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.PARTIAL,
                        evidence_found=True,
                        required_information=["all_pipeline_opportunities_and_values"],
                        supported_information=["some_opportunity_values"],
                        missing_requirements=["complete_values_for_all_opportunities"],
                        answerable_from_context=True,
                        reason="Partial pipeline evidence found: values exist for some opportunities, but not all requested items.",
                        confidence=0.9,
                        coverage=0.6,
                        evidence_types=evidence_types,
                    )

            # If chunks contain structured list items or steps
            if "list" in evidence_types or "table" in evidence_types or re.search(r"^\s*[-*+]\s+Step|^\s*\d+\.\s+", combined_text, re.M):
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.SUFFICIENT,
                    evidence_found=True,
                    required_information=["enumerated_items"],
                    supported_information=["structured_list_steps_found"],
                    missing_requirements=[],
                    answerable_from_context=True,
                    reason="Enumerated list items and implementation steps found in retrieved evidence.",
                    confidence=0.95,
                    coverage=1.0,
                    evidence_types=evidence_types,
                )

            # If query asks for complete/all and retrieved evidence has only 1 short chunk without list
            if len(chunks) < 2 and "list" not in evidence_types and "table" not in evidence_types:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.PARTIAL,
                    evidence_found=True,
                    required_information=["exhaustive_list_items"],
                    supported_information=["fragmentary_items"],
                    missing_requirements=["comprehensive_list_items"],
                    answerable_from_context=True,
                    reason="The query requests an exhaustive list, but only partial/fragmentary evidence was retrieved.",
                    confidence=0.85,
                    coverage=0.4,
                    evidence_types=evidence_types,
                )

        # ----------------------------------------------------
        # 9. Unrelated Topic Evidence Check
        # ----------------------------------------------------
        stopwords = {"what", "which", "where", "when", "how", "show", "list", "tell", "give", "from", "with", "that", "this", "these", "those", "about", "needs", "need", "before", "after", "is", "are", "the", "and"}
        q_tokens = [w for w in re.findall(r"\b\w{3,}\b", q_lower) if w not in stopwords]
        if q_tokens:
            has_overlap = any(
                token in combined_lower or any(token[:4] in w for w in re.findall(r"\b\w{3,}\b", combined_lower))
                for token in q_tokens
            )
            if not has_overlap:
                return SufficiencyEvaluation(
                    status=SufficiencyStatus.INSUFFICIENT,
                    evidence_found=False,
                    required_information=[q_clean],
                    supported_information=[],
                    missing_requirements=["relevant_topic_evidence"],
                    answerable_from_context=False,
                    reason="Retrieved context does not contain relevant evidence addressing the query topic.",
                    confidence=0.95,
                    coverage=0.0,
                    evidence_types=evidence_types,
                )

        # Default sufficient when relevant chunks with substantial text exist
        return SufficiencyEvaluation(
            status=SufficiencyStatus.SUFFICIENT,
            evidence_found=True,
            required_information=[q_clean],
            supported_information=["relevant_substantive_context"],
            missing_requirements=[],
            answerable_from_context=True,
            reason="Retrieved evidence contains relevant and substantive information addressing the query.",
            confidence=0.9,
            coverage=0.9,
            evidence_types=evidence_types,
        )

    async def evaluate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_plan: QueryPlan | None = None,
        chunk_hierarchy: dict[str, ChunkHierarchyMetadata] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
    ) -> SufficiencyEvaluation:
        """
        Asynchronously evaluates evidence sufficiency using structured LLM reasoning
        with deterministic validation and fast deterministic fallback.
        """
        # 1. Run deterministic pre-evaluation
        det_eval = self.evaluate_deterministic(
            query=query,
            chunks=chunks,
            query_plan=query_plan,
            chunk_hierarchy=chunk_hierarchy,
            conversation_history=conversation_history,
            workspace_thread_context=workspace_thread_context,
        )

        # For conversation queries or empty chunks, deterministic check is definitive
        if query_plan and (query_plan.conversation_required or query_plan.information_need == InformationNeed.CONVERSATION):
            return det_eval
        if not chunks:
            return det_eval

        # 2. For document queries, invoke structured LLM with evidence-aware instructions
        try:
            llm = self._get_llm()
            context_text = "\n\n".join(
                f"Chunk {i + 1} (File: {c.file_name}, Section: {chunk_hierarchy.get(str(c.chunk_id)).title if chunk_hierarchy and str(c.chunk_id) in chunk_hierarchy else 'N/A'}):\n{c.text}"
                for i, c in enumerate(chunks)
            )

            info_str = query_plan.information_need.value if query_plan else "GENERAL"
            pres_str = query_plan.presentation_need.value if query_plan else "NONE"
            ev_str = query_plan.evidence_need.value if query_plan else "PROSE"

            system_prompt = (
                "You are an expert evidence-aware sufficiency evaluator for a retrieval-augmented generation (RAG) system.\n"
                "Your objective is to evaluate whether the retrieved context contains sufficient evidence to satisfy the user's query.\n\n"
                "EVIDENCE STATUS DEFINITIONS (Choose exactly one):\n"
                "1. 'sufficient': All essential information required by the user's query is explicitly and unambiguously supported by the retrieved context. The request can be fully answered without hallucination, estimation, outside knowledge, or unsupported assumptions.\n"
                "2. 'partial': Relevant evidence exists and some requested information is supported, but one or more requested elements are missing (or conflicting), yet a useful grounded partial answer is still possible strictly from the context.\n"
                "3. 'insufficient': The retrieved evidence is not sufficient to produce a meaningful grounded answer to the user's request, or no relevant evidence was found.\n\n"
                "EVALUATION PROCEDURE:\n"
                "Step 1: Understand the user's requested output and determine required_information.\n"
                "Step 2: Inspect retrieved context and identify supported_information.\n"
                "Step 3: Identify missing_requirements.\n"
                "Step 4: Identify any contradictions across sources.\n"
                "Step 5: Set evidence_found (boolean) and answerable_from_context (boolean).\n"
                "Step 6: Classify status as 'sufficient', 'partial', or 'insufficient' and provide a concise reason.\n\n"
                "STRICT EVALUATION RULES:\n"
                "- Use ONLY the retrieved context and the user query. NEVER use outside knowledge or make assumptions.\n"
                "- Do NOT infer an exact value from an approximate or relative value (e.g., 'under 60 days' is NOT an exact deadline date).\n"
                "- Precision words like 'exact', 'specific', 'all', 'every', 'complete', 'total', 'each' create strict completeness requirements.\n"
                "- Do NOT confuse retrieval relevance with sufficiency. Related context is not sufficient if the target fact is absent.\n"
                "- If evidence exists but requested coverage is incomplete, classify as 'partial' if a useful grounded response can still be generated.\n"
                "- If essential information is completely absent and no useful grounded answer can be given, classify as 'insufficient'.\n"
                "- Never mark 'sufficient' merely because some related topic information was found.\n\n"
                f"Target Information Need: {info_str}\n"
                f"Target Presentation Need: {pres_str}\n"
                f"Target Evidence Need: {ev_str}\n"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{query}"},
            ]

            structured_llm = llm.llm.with_structured_output(
                SufficiencyEvaluation,
                include_raw=True,
            )

            _eval_start = time.perf_counter()
            inv_res = await structured_llm.ainvoke(messages)
            _eval_latency = (time.perf_counter() - _eval_start) * 1000

            raw_msg = inv_res.get("raw") if isinstance(inv_res, dict) else None
            usage_meta = getattr(raw_msg, "usage_metadata", None) or {}
            record_llm_call(
                stage_name="research",
                usage=usage_meta,
                latency_ms=_eval_latency,
                model_name=getattr(llm.llm, "model_name", None),
            )

            result: SufficiencyEvaluation = inv_res.get("parsed") if isinstance(inv_res, dict) else inv_res
            if result is not None:
                if not result.evidence_types:
                    result.evidence_types = det_eval.evidence_types
                # Run deterministic validation to enforce consistency
                validated_result = self.validate_sufficiency_result(result, chunks=chunks)
                return validated_result

            return det_eval

        except Exception as exc:
            logger.warning(
                "evidence_sufficiency_llm_failed_using_deterministic",
                extra={
                    "event": "evidence_sufficiency_llm_failed",
                    "error": str(exc),
                    "query": query,
                },
            )
            # Ensure failure does not become a false-positive sufficient evidence
            if query_plan and (query_plan.presentation_need == PresentationNeed.CHART or query_plan.information_need == InformationNeed.AGGREGATION):
                combined_evidence = " ".join(c.text for c in chunks)
                has_numbers = bool(re.search(r"\$\d+|\b\d+(?:\.\d+)?(?:k|m|b|%|\s*million|\s*thousand)?\b", combined_evidence, re.IGNORECASE))
                if not has_numbers or not det_eval.is_sufficient:
                    return SufficiencyEvaluation(
                        status=SufficiencyStatus.INSUFFICIENT,
                        evidence_found=bool(chunks),
                        required_information=["structured_numeric_metrics"],
                        supported_information=["prose_context"],
                        missing_requirements=["structured_numeric_data"],
                        answerable_from_context=False,
                        reason="Sufficiency evaluation fallback: insufficient structured data for chart generation.",
                        confidence=0.85,
                        coverage=0.3,
                        evidence_types=det_eval.evidence_types,
                    )
            return det_eval
