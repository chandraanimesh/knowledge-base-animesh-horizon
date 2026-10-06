import json
import logging
import re
from rag_app.schemas.visualization import validate_visualization


def _parse_llm_structured_output(raw_text: str) -> tuple[str, dict | None]:
    """
    Extracts answer and visualization from LLM response.
    Supports JSON code blocks, pure JSON, or gracefully falls back to raw text.
    """
    if not raw_text or not raw_text.strip():
        return "", None

    text = raw_text.strip()

    # 1. Strip markdown code block wrappers if present
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if match:
        json_candidate = match.group(1)
    else:
        # Try finding the outer-most JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            json_candidate = text[brace_start : brace_end + 1]
        else:
            json_candidate = None

    if json_candidate:
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                answer = data.get("answer")
                viz_raw = data.get("visualization")
                if answer is not None:
                    parsed_answer = str(answer) if not isinstance(answer, str) else answer
                    validated_viz = validate_visualization(viz_raw)
                    return parsed_answer, validated_viz
        except Exception as e:
            logger.debug(f"Failed to parse LLM structured JSON response: {e}")

    # Fallback to plain text if JSON parsing was not successful
    return text, None

from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential
from rag_app.retrieval.query_parser import is_greeting
from rag_app.core.safety import check_sensitive_request
from rag_app.core.observability import (
    record_stage_start,
    record_stage_end,
    record_sub_stage,
    record_grounding_metrics,
    record_visualization_metrics,
    record_failure_category,
    record_request_context,
)
from rag_app.schemas.rag import (
    ConflictNotice,
    FailureCategory,
    GenerateRequest,
    GenerateResponse,
    RetrievedChunk,
)
from rag_app.providers.base.llm import LLMProvider
from rag_app.generation.llm_factory import get_llm
from rag_app.generation.prompt import build_rag_messages, format_docs
from rag_app.generation.citations import validate_and_render_citations
from rag_app.retrieval.service import RetrievalService
from rag_app.context.compiler import ContextCompiler
from langsmith import traceable
from rag_app.core.tracing import track
from rag_app.db.models import User
from rag_app.security.authorization import get_allowed_departments, get_allowed_document_ids
from typing import Any
import time
import uuid
logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self,
        session: AsyncSession,
        context_compiler: ContextCompiler | None = None,
    ):
        self.retrieval_service = RetrievalService(session)
        self.llm = get_llm()
        self.context_compiler = context_compiler or ContextCompiler()

    @track(name="generate_answer", phase="generation")
    async def generate_answer(
        self,
        request: GenerateRequest,
        user: User,
    ) -> GenerateResponse:

        safety = check_sensitive_request(
            request.query,
            can_view_pii=False,
        )
        logger.info(
            "safety_check_result",
            extra={
                "event": "safety_check_result",
                "query": request.query,
                "blocked": safety.blocked,
                "reason": safety.reason,
        },
)
        if safety.blocked:
            return GenerateResponse(
                answer=safety.response or "This request cannot be completed.",
                citations=[],
                retrieved_chunks=[],
            )
       
        allowed_departments = await get_allowed_departments(self.retrieval_service.session, user)
        allowed_documents = await get_allowed_document_ids(self.retrieval_service.session, user, "VIEW")

        if allowed_departments is not None and len(allowed_departments) == 0 and (allowed_documents is not None and len(allowed_documents) == 0):
            return GenerateResponse(
                answer="You do not have access to any departments to fulfill this request.",
                citations=[],
                retrieved_chunks=[],
            )
            

        retrieval_response = await self.retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            rerank=request.rerank,
            allowed_departments=allowed_departments,
            allowed_documents=allowed_documents,
        )

        compiled_context = self.context_compiler.compile(
            chunks=retrieval_response.results,
            query=request.query,
            allowed_departments=allowed_departments,
            is_web_context=False,
            strategy_trace=retrieval_response.strategy_trace,
            chunk_provenance=retrieval_response.chunk_provenance,
            chunk_hierarchy=retrieval_response.chunk_hierarchy,
        )

        return await self.generate_from_context(
            query=request.query,
            chunks=compiled_context.chunks,
            allowed_departments=allowed_departments,
            conflicts=compiled_context.conflicts,
            chunk_hierarchy=compiled_context.chunk_hierarchy,
            rendered_context=compiled_context.rendered_context,
            workspace_id=request.workspace_id,
            provider_overrides=request.provider_overrides,
        )

    @track(name="generate_from_context", phase="generation")
    async def generate_from_context(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        allowed_departments: list[uuid.UUID] | None = None,
        is_web_context: bool = False,
        conflicts: list[ConflictNotice] | None = None,
        chunk_hierarchy: dict[str, "ChunkHierarchyMetadata"] | None = None,
        rendered_context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
        workspace_id: uuid.UUID | None = None,
        llm: LLMProvider | None = None,
        provider_overrides: dict[str, str] | None = None,
        sufficiency_evaluation: Any | None = None,
    ) -> GenerateResponse:
        
        if is_greeting(query):
            logger.info(
                "conversational_query_detected_in_context_generation",
                extra={
                    "event": (
                        "conversational_query_detected_in_context_generation"
                    ),
                    "query": query,
                },
            )

            return GenerateResponse(
                answer=(
                    "Hello! I'm Doot — a research assistant "
                    "grounded in your document library. "
                    "How can I help you?"
                ),
                citations=[],
                retrieved_chunks=[],
            )

        record_stage_start("generation")
        try:
            # Final context safety check for internal chunks
            if not is_web_context and allowed_departments is not None:
                allowed_dept_strs = {str(d) for d in allowed_departments}
                for chunk in chunks:
                    if chunk.department_id not in allowed_dept_strs and chunk.department_id != "None":
                        logger.error(
                            "context_safety_violation",
                            extra={
                                "event": "context_safety_violation",
                                "chunk_id": chunk.chunk_id,
                                "department_id": chunk.department_id,
                            }
                        )
                        # Defense in depth: mathematically guarantee it cannot reach LLM
                        raise RuntimeError("Security violation: Unauthorized chunk detected in final context.")

            context = rendered_context if rendered_context is not None else format_docs(chunks, chunk_hierarchy=chunk_hierarchy)
            messages = build_rag_messages(
                context=context,
                question=query,
                conflicts=conflicts,
                chunks=chunks,
                conversation_history=conversation_history,
                workspace_thread_context=workspace_thread_context,
            )

            # Resolve active LLM provider for the given Workspace (supporting per-request experiment overrides)
            llm_to_use = llm
            if llm_to_use is None:
                if workspace_id is not None or provider_overrides:
                    from rag_app.providers.resolver import resolve_llm_for_workspace
                    override_llm = provider_overrides.get("llm") if provider_overrides else None
                    llm_to_use = await resolve_llm_for_workspace(
                        workspace_id,
                        self.retrieval_service.session,
                        override_provider=override_llm,
                    )
                else:
                    llm_to_use = self.llm

            # Context info
            provider_name = getattr(llm_to_use, "provider_name", None) or type(llm_to_use).__name__
            model_name = getattr(llm_to_use, "model_name", None) or "default"
            record_request_context(
                workspace_id=str(workspace_id) if workspace_id else None,
                provider=provider_name,
                model=model_name,
            )

            is_viz_requested = bool(re.search(r"\b(chart|graph|plot|visualize|bar|line|pie|scatter)\b", query.lower()))
            repair_triggered = False
            regeneration_triggered = False
            safe_limitation_triggered = False
            viz_repaired = False
            viz_dropped = False
            viz_failure_reason = None

            # GENERATE GROUNDED ANSWER
            _init_gen_start = time.perf_counter()
            raw_response_text, token_usage = await _invoke_llm_with_retry(
                llm_to_use,
                messages,
                stage="generation",
            )
            record_sub_stage("generation", "initial_generation", (time.perf_counter() - _init_gen_start) * 1000)

            # PARSE STRUCTURED ANSWER AND VISUALIZATION
            text_answer, validated_viz = _parse_llm_structured_output(raw_response_text)
            viz_generated = validated_viz is not None

            # VALIDATE AND RENDER CITATIONS
            answer, citations = validate_and_render_citations(
                raw_answer=text_answer,
                chunks=chunks,
                is_conflict=bool(conflicts),
            )

            # VALIDATE GROUNDING AGAINST AUTHORIZED EVIDENCE
            from rag_app.generation.grounding import AnswerGroundingEvaluator
            grounding_evaluator = AnswerGroundingEvaluator(llm=llm_to_use)
            _ground_start = time.perf_counter()
            grounding_res = await grounding_evaluator.evaluate(
                query=query,
                answer=answer,
                chunks=chunks,
                visualization=validated_viz,
                conflicts=conflicts,
                conversation_history=conversation_history,
                workspace_thread_context=workspace_thread_context,
            )
            record_sub_stage("generation", "grounding", (time.perf_counter() - _ground_start) * 1000)

            # ENFORCE GROUNDING: PREVENT UNGROUNDED/FABRICATED CLAIMS FROM BEING DELIVERED AS FACT
            if not grounding_res.is_grounded:
                record_failure_category(FailureCategory.GROUNDING)
                logger.warning(
                    "grounding_violation_detected_enforcing_repair_or_regeneration",
                    extra={
                        "event": "grounding_violation_detected",
                        "unsupported_claims": grounding_res.unsupported_claims,
                        "contradictions": grounding_res.contradictions,
                        "query": query,
                    },
                )

                # 1. Deterministic Visualization Repair / Pruning
                if validated_viz:
                    _repair_start = time.perf_counter()
                    repaired_viz = grounding_evaluator.repair_visualization(
                        visualization=validated_viz,
                        query=query,
                        chunks=chunks,
                        conversation_history=conversation_history,
                        workspace_thread_context=workspace_thread_context,
                    )
                    record_sub_stage("generation", "repair", (time.perf_counter() - _repair_start) * 1000)
                    repair_triggered = True
                    viz_repaired = True
                    validated_viz = repaired_viz

                # 2. Constrained Regeneration for Answer if unsupported claims or contradictions exist
                if grounding_res.unsupported_claims or grounding_res.contradictions:
                    regeneration_triggered = True
                    correction_instructions = (
                        "You are an expert answer grounding repair assistant.\n"
                        "The previous answer contained unsupported factual claims, wrong dates/numbers, or unresolved contradictions:\n"
                        + "\n".join(f"- {c}" for c in grounding_res.unsupported_claims)
                        + ("\nContradictions:\n" + "\n".join(f"- {c}" for c in grounding_res.contradictions) if grounding_res.contradictions else "")
                        + "\n\nCRITICAL ENFORCEMENT RULES:\n"
                        "1. Answer strictly and only from the provided Context and Conversation History.\n"
                        "2. NEVER invent, assume, or guess missing dates, numbers, requirements, or entities.\n"
                        "3. If a requested detail cannot be verified from the provided documents, explicitly state that it is not available in the authorized evidence.\n"
                        "4. If there is conflicting evidence, present the conflicting values and cite both sources rather than choosing one.\n"
                        "5. Format output as JSON with keys 'answer' and 'visualization'."
                    )

                    correction_messages = [
                        {"role": "system", "content": correction_instructions},
                        *messages[1:],
                        {"role": "assistant", "content": raw_response_text},
                        {
                            "role": "user",
                            "content": (
                                "Please regenerate the answer strictly complying with the authorized evidence. "
                                "Remove all unsupported claims and accurately state any document limitations."
                            ),
                        },
                    ]

                    try:
                        _regen_start = time.perf_counter()
                        regen_raw, regen_tokens = await _invoke_llm_with_retry(
                            llm_to_use,
                            correction_messages,
                            stage="generation",
                        )
                        record_sub_stage("generation", "regeneration", (time.perf_counter() - _regen_start) * 1000)
                        regen_text, regen_viz = _parse_llm_structured_output(regen_raw)
                        regen_answer, regen_citations = validate_and_render_citations(
                            raw_answer=regen_text,
                            chunks=chunks,
                            is_conflict=bool(conflicts),
                        )

                        # Re-evaluate grounding on regenerated response
                        regen_grounding = await grounding_evaluator.evaluate(
                            query=query,
                            answer=regen_answer,
                            chunks=chunks,
                            visualization=regen_viz or validated_viz,
                            conflicts=conflicts,
                            conversation_history=conversation_history,
                            workspace_thread_context=workspace_thread_context,
                        )

                        if regen_grounding.is_grounded:
                            answer = regen_answer
                            citations = regen_citations
                            if regen_viz is not None:
                                validated_viz = regen_viz
                            grounding_res = regen_grounding
                        else:
                            # 3. Grounded Limitation Fallback
                            safe_limitation_triggered = True
                            claim_summary = ", ".join(f'"{c}"' for c in regen_grounding.unsupported_claims[:2])
                            disclaimer = (
                                f"\n\n*Note: Some specific details ({claim_summary}) "
                                f"could not be verified from the authorized documents.*"
                            )
                            answer = regen_answer + disclaimer
                            grounding_res = regen_grounding
                    except Exception as regen_err:
                        safe_limitation_triggered = True
                        logger.warning(f"Constrained regeneration skipped or failed: {regen_err}")
                        claim_summary = ", ".join(f'"{c}"' for c in grounding_res.unsupported_claims[:2])
                        disclaimer = f"\n\n*Note: Specific details ({claim_summary}) could not be verified from the authorized evidence.*"
                        answer = answer + disclaimer

            # 4. Visualization Completeness Check for Exhaustive Queries (Phases 5-8)
            is_exhaustive_query = bool(re.search(r"\b(all|every|complete|entire|full\s+list|everything)\b", query.lower()))
            if validated_viz and is_exhaustive_query and sufficiency_evaluation is not None:
                if not getattr(sufficiency_evaluation, "is_sufficient", True):
                    logger.warning(
                        "visualization_dropped_due_to_incomplete_evidence_for_exhaustive_query",
                        extra={"query": query, "event": "visualization_incomplete_evidence_dropped"},
                    )
                    validated_viz = None
                    viz_dropped = True
                    viz_failure_reason = "insufficient_coverage"
                    record_failure_category(FailureCategory.VISUALIZATION)
                    if "partial" not in answer.lower() and "limited" not in answer.lower() and "incomplete" not in answer.lower():
                        answer = f"{answer}\n\n*Note: The retrieved evidence only contains partial records and does not cover the complete set.*"

            # Record Grounding and Visualization Observability
            record_grounding_metrics(
                grounding=grounding_res.is_grounded,
                confidence=grounding_res.confidence,
                unsupported_claim_count=len(grounding_res.unsupported_claims),
                evidence_ref_count=len(grounding_res.evidence_refs),
                repair_triggered=repair_triggered,
                regeneration_triggered=regeneration_triggered,
                safe_limitation_triggered=safe_limitation_triggered,
            )
            record_visualization_metrics(
                requested=is_viz_requested,
                generated=viz_generated,
                validated=validated_viz is not None,
                repaired=viz_repaired,
                dropped=viz_dropped,
                failure_reason=viz_failure_reason,
            )

            logger.info(
                "generation_completed",
                extra={
                    "event": "generation_completed",
                    "citations": len(citations),
                    "has_visualization": validated_viz is not None,
                    "visualization_type": validated_viz.get("type") if validated_viz else None,
                    "is_grounded": grounding_res.is_grounded,
                    "unsupported_claims": len(grounding_res.unsupported_claims),
                    "token_usage": token_usage,
                    "is_web_context": is_web_context,
                },
            )

            return GenerateResponse(
                answer=answer,
                visualization=validated_viz,
                citations=citations,
                retrieved_chunks=chunks,
                grounding=grounding_res,
            )
        finally:
            record_stage_end("generation")

    async def stream_from_context(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        allowed_departments: list[uuid.UUID] | None = None,
        is_web_context: bool = False,
        conflicts: list[ConflictNotice] | None = None,
        chunk_hierarchy: dict[str, "ChunkHierarchyMetadata"] | None = None,
        rendered_context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
        workspace_id: uuid.UUID | None = None,
        llm: LLMProvider | None = None,
        provider_overrides: dict[str, str] | None = None,
        sufficiency_evaluation: Any | None = None,
    ):
        """
        Asynchronously streams provisional tokens to the caller, then executes grounding
        and enforcement on the completed output before emitting the final authoritative response.
        Ensures intermediate ungrounded tokens are superseded by the validated grounded answer.
        """
        record_request_context(streaming=True)
        if is_greeting(query):
            final_resp = GenerateResponse(
                answer=(
                    "Hello! I'm Doot — a research assistant "
                    "grounded in your document library. "
                    "How can I help you?"
                ),
                citations=[],
                retrieved_chunks=[],
            )
            yield {"type": "token", "content": final_resp.answer}
            yield {"type": "complete", "response": final_resp}
            return

        final_response = await self.generate_from_context(
            query=query,
            chunks=chunks,
            allowed_departments=allowed_departments,
            is_web_context=is_web_context,
            conflicts=conflicts,
            chunk_hierarchy=chunk_hierarchy,
            rendered_context=rendered_context,
            conversation_history=conversation_history,
            workspace_thread_context=workspace_thread_context,
            workspace_id=workspace_id,
            llm=llm,
            provider_overrides=provider_overrides,
            sufficiency_evaluation=sufficiency_evaluation,
        )

        # Stream provisional tokens
        words = final_response.answer.split(" ")
        for i, word in enumerate(words):
            yield {"type": "token", "content": word + (" " if i < len(words) - 1 else "")}

        # Emit final authoritative grounded response
        yield {"type": "complete", "response": final_response}


@track(name="invoke_llm", phase="generation")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=10,
    ),
)
async def _invoke_llm_with_retry(
    llm: LLMProvider,
    messages: list[dict[str, str]],
    stage: str | None = None,
) -> tuple[str, dict]:
    return await llm.generate(messages, stage=stage)