from rag_app.schemas.rag import ExecutionMetrics
from rag_app.core.observability import (
    init_request_metrics,
    record_request_context,
    finalize_request_metrics,
    record_retrieval_metrics,
    record_sufficiency_metrics,
    record_grounding_metrics,
)


def test_execution_metrics_schema_total_llm_calls():
    """ExecutionMetrics schema includes total_llm_calls with safe defaults."""
    metrics = ExecutionMetrics(
        request_id="req-123",
        input_tokens=500,
        output_tokens=100,
        total_tokens=600,
        total_llm_calls=2,
        provider="Groq",
        model="llama-3.3-70b-versatile",
    )
    dumped = metrics.model_dump()
    assert dumped["total_llm_calls"] == 2
    assert dumped["total_tokens"] == 600
    assert dumped["provider"] == "Groq"


def test_finalize_request_metrics_computes_llm_calls_and_checkpoints():
    """finalize_request_metrics computes total_llm_calls and preserves all retrieval checkpoints."""
    req_id = "test-inspect-req-456"
    init_request_metrics(req_id)
    record_request_context(provider="Groq", model="llama-3.3-70b")

    record_retrieval_metrics(
        planned_strategies=["hybrid_vector", "bm25_keyword", "knowledge_graph"],
        executed_strategies=["hybrid_vector", "bm25_keyword"],
        vector_candidates=10,
        keyword_candidates=8,
        knowledge_candidates=2,
        reranked_candidates=20,
        accepted_results=4,
    )
    record_sufficiency_metrics(
        sufficiency=True,
        confidence=0.95,
        reason="Evidence covers required information.",
    )
    record_grounding_metrics(
        grounding=True,
        confidence=0.98,
    )

    metrics = finalize_request_metrics()
    assert metrics is not None
    assert metrics.request_id == req_id
    assert metrics.total_llm_calls >= 1
    assert metrics.planned_strategies == ["hybrid_vector", "bm25_keyword", "knowledge_graph"]
    assert metrics.vector_candidates == 10
    assert metrics.keyword_candidates == 8
    assert metrics.accepted_count == 4
    assert metrics.sufficiency is True
    assert metrics.grounding is True
    assert metrics.provider == "Groq"
    assert metrics.model == "llama-3.3-70b"


def test_execution_metrics_serialization_integrity():
    """All checkpoints serialize into JSON dict matching frontend inspector expectations."""
    metrics = ExecutionMetrics(
        request_id="req-789",
        provider="Ollama",
        model="qwen2.5",
        total_latency_ms=1840.0,
        input_tokens=1024,
        output_tokens=128,
        total_tokens=1152,
        total_llm_calls=1,
        retrieval_latency_ms=320.0,
        generation_latency_ms=1520.0,
        planned_strategies=["vector", "keyword"],
        vector_candidates=12,
        keyword_candidates=6,
        accepted_count=5,
        sufficiency=True,
        sufficiency_reason="Sufficient authorized documents retrieved.",
        grounding=True,
        grounding_confidence=0.99,
    )
    data = metrics.model_dump()
    assert data["total_llm_calls"] == 1
    assert data["retrieval_latency_ms"] == 320.0
    assert data["generation_latency_ms"] == 1520.0
    assert data["sufficiency"] is True
    assert data["grounding"] is True
