import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from rag_app.schemas.rag import (
    ExecutionMetrics,
    ExecutionStages,
    StageMetrics,
    SubStageMetric,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_request_metrics_ctx: ContextVar["RequestContextMetrics | None"] = ContextVar(
    "request_metrics_ctx", default=None
)


def _extract_int(val: Any) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def parse_usage_dict(usage: dict | None) -> TokenUsage:
    """
    Extracts standardized TokenUsage from provider usage dicts.
    Handles LangChain usage_metadata, OpenAI/Groq response formats.
    """
    if not usage or not isinstance(usage, dict):
        return TokenUsage()

    input_tokens = _extract_int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("inputTokens")
    )
    output_tokens = _extract_int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("outputTokens")
    )
    total_tokens = _extract_int(
        usage.get("total_tokens")
        or usage.get("totalTokens")
    )
    if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
        total_tokens = input_tokens + output_tokens

    # Reasoning tokens
    out_details = usage.get("output_token_details") or {}
    if not isinstance(out_details, dict):
        out_details = {}
    reasoning_tokens = _extract_int(
        out_details.get("reasoning_tokens")
        or usage.get("reasoning_tokens")
        or usage.get("reasoningTokens")
    )

    # Cache read tokens
    in_details = usage.get("input_token_details") or {}
    if not isinstance(in_details, dict):
        in_details = {}
    cache_read_tokens = _extract_int(
        in_details.get("cache_read")
        or in_details.get("cache_read_tokens")
        or usage.get("cache_read_tokens")
        or usage.get("cacheReadTokens")
    )

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
    )


@dataclass
class LLMCallRecord:
    stage: str
    usage: TokenUsage
    latency_ms: float = 0.0
    model_name: str | None = None


@dataclass
class StageRecord:
    name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    explicit_latency_ms: float | None = None
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    sub_stages: list[SubStageMetric] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        if self.explicit_latency_ms is not None:
            return round(self.explicit_latency_ms, 2)
        if self.end_time is not None:
            return round((self.end_time - self.start_time) * 1000, 2)
        return round((time.perf_counter() - self.start_time) * 1000, 2)

    @property
    def aggregated_usage(self) -> TokenUsage:
        in_tok = sum(c.usage.input_tokens for c in self.llm_calls)
        out_tok = sum(c.usage.output_tokens for c in self.llm_calls)
        tot_tok = sum(c.usage.total_tokens for c in self.llm_calls)
        reas_tok = sum(c.usage.reasoning_tokens for c in self.llm_calls)
        cache_tok = sum(c.usage.cache_read_tokens for c in self.llm_calls)
        return TokenUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            reasoning_tokens=reas_tok,
            cache_read_tokens=cache_tok,
        )


@dataclass
class RequestContextMetrics:
    request_id: str
    trace_id: str | None = None
    workspace_id: str | None = None
    thread_id: str | None = None
    provider: str | None = None
    model: str | None = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    stages: dict[str, StageRecord] = field(default_factory=dict)
    current_stage: str | None = None

    # Granular Latency Spans
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    grounding_latency_ms: float = 0.0
    repair_latency_ms: float = 0.0
    initial_generation_latency_ms: float = 0.0
    regeneration_latency_ms: float = 0.0

    # Retrieval Quality & Strategy
    planned_strategies: list[str] = field(default_factory=list)
    executed_strategies: list[str] = field(default_factory=list)
    vector_candidates: int = 0
    keyword_candidates: int = 0
    knowledge_candidates: int = 0
    hierarchical_candidates: int = 0
    candidate_count: int = 0
    reranked_candidates: int = 0
    accepted_results: int = 0
    accepted_count: int = 0
    final_context_count: int = 0
    context_chunks: int = 0
    context_tokens: int = 0
    vector_top_k: int | None = None
    keyword_top_k: int | None = None
    hierarchical_count: int = 0
    reranker_top_score: float | None = None
    reranker_bottom_score: float | None = None
    structured_evidence_count: int = 0
    hierarchical_evidence_count: int = 0
    conversation_evidence_count: int = 0

    # Sufficiency Quality
    sufficiency: bool | None = None
    sufficiency_confidence: float | None = None
    sufficiency_coverage: float | None = None
    sufficiency_reason: str | None = None
    missing_requirements_count: int = 0
    contradiction_count: int = 0
    evidence_types: list[str] = field(default_factory=list)

    # Grounding Quality & Enforcement
    grounding: bool | None = None
    grounding_confidence: float | None = None
    unsupported_claim_count: int = 0
    evidence_ref_count: int = 0
    repair_triggered: bool = False
    regeneration_triggered: bool = False
    safe_limitation_triggered: bool = False

    # Visualization
    visualization_requested: bool = False
    visualization_generated: bool = False
    visualization_validated: bool = False
    visualization_repaired: bool = False
    visualization_dropped: bool = False
    visualization_failure_reason: str | None = None

    # Streaming & Taxonomy
    streaming: bool = False
    usage_available: bool = True
    failure_categories: list[Any] = field(default_factory=list)

    def get_or_create_stage(self, stage_name: str) -> StageRecord:
        if stage_name not in self.stages:
            self.stages[stage_name] = StageRecord(name=stage_name)
        return self.stages[stage_name]

    @property
    def total_latency_ms(self) -> float:
        if self.end_time is not None:
            return round((self.end_time - self.start_time) * 1000, 2)
        return round((time.perf_counter() - self.start_time) * 1000, 2)


def init_request_metrics(request_id: str, trace_id: str | None = None) -> RequestContextMetrics:
    metrics = RequestContextMetrics(request_id=request_id, trace_id=trace_id or request_id)
    _request_metrics_ctx.set(metrics)
    return metrics


def get_request_metrics() -> RequestContextMetrics | None:
    return _request_metrics_ctx.get()


def record_request_context(
    workspace_id: str | None = None,
    thread_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    trace_id: str | None = None,
    streaming: bool = False,
) -> None:
    ctx = get_request_metrics()
    if not ctx:
        return
    if workspace_id is not None:
        ctx.workspace_id = str(workspace_id)
    if thread_id is not None:
        ctx.thread_id = str(thread_id)
    if provider is not None:
        ctx.provider = provider
    if model is not None:
        ctx.model = model
    if trace_id is not None:
        ctx.trace_id = trace_id
    if streaming:
        ctx.streaming = True


def record_retrieval_metrics(
    planned_strategies: list[str] | None = None,
    executed_strategies: list[str] | None = None,
    vector_candidates: int | None = None,
    keyword_candidates: int | None = None,
    knowledge_candidates: int | None = None,
    hierarchical_candidates: int | None = None,
    candidate_count: int | None = None,
    reranked_candidates: int | None = None,
    accepted_results: int | None = None,
    final_context_count: int | None = None,
    context_chunks: int | None = None,
    context_tokens: int | None = None,
    vector_top_k: int | None = None,
    keyword_top_k: int | None = None,
    hierarchical_count: int | None = None,
    reranker_top_score: float | None = None,
    reranker_bottom_score: float | None = None,
    structured_evidence_count: int | None = None,
    hierarchical_evidence_count: int | None = None,
    conversation_evidence_count: int | None = None,
) -> None:
    ctx = get_request_metrics()
    if not ctx:
        return
    if planned_strategies is not None:
        ctx.planned_strategies = planned_strategies
    if executed_strategies is not None:
        ctx.executed_strategies = executed_strategies
    if vector_candidates is not None:
        ctx.vector_candidates = vector_candidates
    if keyword_candidates is not None:
        ctx.keyword_candidates = keyword_candidates
    if knowledge_candidates is not None:
        ctx.knowledge_candidates = knowledge_candidates
    if hierarchical_candidates is not None:
        ctx.hierarchical_candidates = hierarchical_candidates
    if candidate_count is not None:
        ctx.candidate_count = candidate_count
    if reranked_candidates is not None:
        ctx.reranked_candidates = reranked_candidates
    if accepted_results is not None:
        ctx.accepted_results = accepted_results
        ctx.accepted_count = accepted_results
    if final_context_count is not None:
        ctx.final_context_count = final_context_count
    if context_chunks is not None:
        ctx.context_chunks = context_chunks
    if context_tokens is not None:
        ctx.context_tokens = context_tokens
    if vector_top_k is not None:
        ctx.vector_top_k = vector_top_k
    if keyword_top_k is not None:
        ctx.keyword_top_k = keyword_top_k
    if hierarchical_count is not None:
        ctx.hierarchical_count = hierarchical_count
    if reranker_top_score is not None:
        ctx.reranker_top_score = reranker_top_score
    if reranker_bottom_score is not None:
        ctx.reranker_bottom_score = reranker_bottom_score
    if structured_evidence_count is not None:
        ctx.structured_evidence_count = structured_evidence_count
    if hierarchical_evidence_count is not None:
        ctx.hierarchical_evidence_count = hierarchical_evidence_count
    if conversation_evidence_count is not None:
        ctx.conversation_evidence_count = conversation_evidence_count


def record_sufficiency_metrics(
    sufficiency: bool | None = None,
    status: str | None = None,
    evidence_found: bool | None = None,
    confidence: float | None = None,
    coverage: float | None = None,
    reason: str | None = None,
    missing_requirements_count: int | None = None,
    contradiction_count: int | None = None,
    evidence_types: list[str] | None = None,
) -> None:
    ctx = get_request_metrics()
    if not ctx:
        return
    if sufficiency is not None:
        ctx.sufficiency = sufficiency
    if status is not None:
        ctx.sufficiency_status = status
    if evidence_found is not None:
        ctx.evidence_found = evidence_found
    if confidence is not None:
        ctx.sufficiency_confidence = confidence
    if coverage is not None:
        ctx.sufficiency_coverage = coverage
    if reason is not None:
        ctx.sufficiency_reason = reason
    if missing_requirements_count is not None:
        ctx.missing_requirements_count = missing_requirements_count
    if contradiction_count is not None:
        ctx.contradiction_count = contradiction_count
    if evidence_types is not None:
        ctx.evidence_types = evidence_types


def record_grounding_metrics(
    grounding: bool | None = None,
    confidence: float | None = None,
    unsupported_claim_count: int | None = None,
    evidence_ref_count: int | None = None,
    repair_triggered: bool | None = None,
    regeneration_triggered: bool | None = None,
    safe_limitation_triggered: bool | None = None,
) -> None:
    ctx = get_request_metrics()
    if not ctx:
        return
    if grounding is not None:
        ctx.grounding = grounding
    if confidence is not None:
        ctx.grounding_confidence = confidence
    if unsupported_claim_count is not None:
        ctx.unsupported_claim_count = unsupported_claim_count
    if evidence_ref_count is not None:
        ctx.evidence_ref_count = evidence_ref_count
    if repair_triggered is not None:
        ctx.repair_triggered = repair_triggered
    if regeneration_triggered is not None:
        ctx.regeneration_triggered = regeneration_triggered
    if safe_limitation_triggered is not None:
        ctx.safe_limitation_triggered = safe_limitation_triggered


def record_visualization_metrics(
    requested: bool | None = None,
    generated: bool | None = None,
    validated: bool | None = None,
    repaired: bool | None = None,
    dropped: bool | None = None,
    failure_reason: str | None = None,
) -> None:
    ctx = get_request_metrics()
    if not ctx:
        return
    if requested is not None:
        ctx.visualization_requested = requested
    if generated is not None:
        ctx.visualization_generated = generated
    if validated is not None:
        ctx.visualization_validated = validated
    if repaired is not None:
        ctx.visualization_repaired = repaired
    if dropped is not None:
        ctx.visualization_dropped = dropped
    if failure_reason is not None:
        ctx.visualization_failure_reason = failure_reason


def record_failure_category(category: Any) -> None:
    ctx = get_request_metrics()
    if ctx and category not in ctx.failure_categories:
        ctx.failure_categories.append(category)


def record_stage_start(stage_name: str) -> None:
    ctx = get_request_metrics()
    if ctx:
        stage = ctx.get_or_create_stage(stage_name)
        stage.start_time = time.perf_counter()
        ctx.current_stage = stage_name


def record_stage_end(stage_name: str, latency_ms: float | None = None) -> None:
    ctx = get_request_metrics()
    if ctx and stage_name in ctx.stages:
        stage = ctx.stages[stage_name]
        stage.end_time = time.perf_counter()
        if latency_ms is not None:
            stage.explicit_latency_ms = latency_ms


def record_sub_stage(stage_name: str, sub_stage_name: str, latency_ms: float) -> None:
    ctx = get_request_metrics()
    if ctx:
        stage = ctx.get_or_create_stage(stage_name)
        stage.sub_stages.append(
            SubStageMetric(name=sub_stage_name, latency_ms=round(latency_ms, 2))
        )


def record_llm_call(
    stage_name: str | None = None,
    usage: dict | None = None,
    latency_ms: float = 0.0,
    model_name: str | None = None,
) -> None:
    ctx = get_request_metrics()
    if not ctx:
        return

    target_stage = stage_name or ctx.current_stage or "generation"
    stage = ctx.get_or_create_stage(target_stage)
    parsed_usage = parse_usage_dict(usage)
    stage.llm_calls.append(
        LLMCallRecord(
            stage=target_stage,
            usage=parsed_usage,
            latency_ms=round(latency_ms, 2),
            model_name=model_name,
        )
    )


def finalize_request_metrics() -> ExecutionMetrics | None:
    ctx = get_request_metrics()
    if not ctx:
        return None

    ctx.end_time = time.perf_counter()
    total_latency = ctx.total_latency_ms

    # Aggregate across all stages
    total_input = 0
    total_output = 0
    total_tokens = 0
    total_reasoning = 0
    total_cache = 0

    stage_metrics_map: dict[str, StageMetrics] = {}

    for name, stage_rec in ctx.stages.items():
        stg_usage = stage_rec.aggregated_usage
        total_input += stg_usage.input_tokens
        total_output += stg_usage.output_tokens
        total_tokens += stg_usage.total_tokens
        total_reasoning += stg_usage.reasoning_tokens
        total_cache += stg_usage.cache_read_tokens

        stage_metrics_map[name] = StageMetrics(
            latency_ms=stage_rec.latency_ms,
            llm_calls=len(stage_rec.llm_calls),
            llm_usage=stg_usage,
            sub_stages=stage_rec.sub_stages,
        )

    llm_usage_total = TokenUsage(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_tokens,
        reasoning_tokens=total_reasoning,
        cache_read_tokens=total_cache,
    )

    stages_obj = ExecutionStages(
        retrieval=stage_metrics_map.get("retrieval"),
        generation=stage_metrics_map.get("generation"),
        research=stage_metrics_map.get("research"),
    )

    retrieval_lat = stage_metrics_map["retrieval"].latency_ms if "retrieval" in stage_metrics_map else 0.0
    gen_lat = stage_metrics_map["generation"].latency_ms if "generation" in stage_metrics_map else 0.0

    # Extract sub-stage latencies
    grounding_lat = 0.0
    repair_lat = 0.0
    init_gen_lat = 0.0
    regen_lat = 0.0

    if "generation" in stage_metrics_map:
        for sub in stage_metrics_map["generation"].sub_stages:
            if "grounding" in sub.name.lower():
                grounding_lat += sub.latency_ms
            elif "repair" in sub.name.lower():
                repair_lat += sub.latency_ms
            elif "initial" in sub.name.lower():
                init_gen_lat += sub.latency_ms
            elif "regen" in sub.name.lower():
                regen_lat += sub.latency_ms

    return ExecutionMetrics(
        request_id=ctx.request_id,
        trace_id=ctx.trace_id or ctx.request_id,
        workspace_id=ctx.workspace_id,
        thread_id=ctx.thread_id,
        provider=ctx.provider,
        model=ctx.model,
        total_latency_ms=total_latency,
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_tokens,
        usage_available=ctx.usage_available,
        llm_usage=llm_usage_total,
        retrieval_latency_ms=retrieval_lat or ctx.retrieval_latency_ms,
        generation_latency_ms=gen_lat or ctx.generation_latency_ms,
        grounding_latency_ms=grounding_lat or ctx.grounding_latency_ms,
        repair_latency_ms=repair_lat or ctx.repair_latency_ms,
        initial_generation_latency_ms=init_gen_lat or ctx.initial_generation_latency_ms,
        regeneration_latency_ms=regen_lat or ctx.regeneration_latency_ms,
        planned_strategies=ctx.planned_strategies,
        executed_strategies=ctx.executed_strategies,
        vector_candidates=ctx.vector_candidates,
        keyword_candidates=ctx.keyword_candidates,
        knowledge_candidates=ctx.knowledge_candidates,
        hierarchical_candidates=ctx.hierarchical_candidates,
        candidate_count=ctx.candidate_count or (ctx.vector_candidates + ctx.keyword_candidates + ctx.knowledge_candidates + ctx.hierarchical_candidates),
        reranked_candidates=ctx.reranked_candidates,
        accepted_results=ctx.accepted_results,
        accepted_count=ctx.accepted_count or ctx.accepted_results,
        final_context_count=ctx.final_context_count or ctx.accepted_results,
        context_chunks=ctx.context_chunks or ctx.accepted_results,
        context_tokens=ctx.context_tokens,
        vector_top_k=ctx.vector_top_k,
        keyword_top_k=ctx.keyword_top_k,
        hierarchical_count=ctx.hierarchical_count,
        reranker_top_score=ctx.reranker_top_score,
        reranker_bottom_score=ctx.reranker_bottom_score,
        structured_evidence_count=ctx.structured_evidence_count,
        hierarchical_evidence_count=ctx.hierarchical_evidence_count,
        conversation_evidence_count=ctx.conversation_evidence_count,
        sufficiency=ctx.sufficiency,
        sufficiency_confidence=ctx.sufficiency_confidence,
        sufficiency_coverage=ctx.sufficiency_coverage,
        sufficiency_reason=ctx.sufficiency_reason,
        missing_requirements_count=ctx.missing_requirements_count,
        contradiction_count=ctx.contradiction_count,
        evidence_types=ctx.evidence_types,
        grounding=ctx.grounding,
        grounding_confidence=ctx.grounding_confidence,
        unsupported_claim_count=ctx.unsupported_claim_count,
        evidence_ref_count=ctx.evidence_ref_count,
        repair_triggered=ctx.repair_triggered,
        regeneration_triggered=ctx.regeneration_triggered,
        safe_limitation_triggered=ctx.safe_limitation_triggered,
        visualization_requested=ctx.visualization_requested,
        visualization_generated=ctx.visualization_generated,
        visualization_validated=ctx.visualization_validated,
        visualization_repaired=ctx.visualization_repaired,
        visualization_dropped=ctx.visualization_dropped,
        visualization_failure_reason=ctx.visualization_failure_reason,
        streaming=ctx.streaming,
        failure_categories=ctx.failure_categories,
        stages=stages_obj,
    )

