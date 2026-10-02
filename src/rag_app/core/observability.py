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
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    stages: dict[str, StageRecord] = field(default_factory=dict)
    current_stage: str | None = None

    def get_or_create_stage(self, stage_name: str) -> StageRecord:
        if stage_name not in self.stages:
            self.stages[stage_name] = StageRecord(name=stage_name)
        return self.stages[stage_name]

    @property
    def total_latency_ms(self) -> float:
        if self.end_time is not None:
            return round((self.end_time - self.start_time) * 1000, 2)
        return round((time.perf_counter() - self.start_time) * 1000, 2)


def init_request_metrics(request_id: str) -> RequestContextMetrics:
    metrics = RequestContextMetrics(request_id=request_id)
    _request_metrics_ctx.set(metrics)
    return metrics


def get_request_metrics() -> RequestContextMetrics | None:
    return _request_metrics_ctx.get()


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

    return ExecutionMetrics(
        request_id=ctx.request_id,
        total_latency_ms=total_latency,
        llm_usage=llm_usage_total,
        stages=stages_obj,
    )
