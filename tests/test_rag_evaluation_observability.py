import pytest
import uuid
import time
from rag_app.schemas.rag import (
    AnswerGroundingResult,
    ExecutionMetrics,
    FailureCategory,
    GenerateResponse,
    RetrievedChunk,
    TokenUsage,
)
from rag_app.core.observability import (
    init_request_metrics,
    finalize_request_metrics,
    record_request_context,
    record_retrieval_metrics,
    record_sufficiency_metrics,
    record_grounding_metrics,
    record_visualization_metrics,
    record_failure_category,
    record_stage_start,
    record_stage_end,
    record_sub_stage,
)
from evaluation.dataset import EVALUATION_DATASET, EvalCase
from evaluation.runner import EvaluationRunner


@pytest.mark.asyncio
async def test_normalized_execution_metrics_contract():
    req_id = str(uuid.uuid4())
    init_request_metrics(req_id, trace_id=f"trace_{req_id[:8]}")

    record_request_context(
        workspace_id="ws-prod-1",
        thread_id="thread-42",
        provider="groq",
        model="llama-3.3-70b",
        streaming=False,
    )

    record_stage_start("retrieval")
    record_sub_stage("retrieval", "query_parsing", 12.5)
    record_sub_stage("retrieval", "vector_search", 45.0)
    record_sub_stage("retrieval", "keyword_search", 20.0)
    record_sub_stage("retrieval", "reranking", 85.0)
    record_retrieval_metrics(
        planned_strategies=["SEMANTIC", "HIERARCHICAL"],
        executed_strategies=["SEMANTIC", "HIERARCHICAL"],
        vector_candidates=20,
        keyword_candidates=15,
        knowledge_candidates=5,
        hierarchical_candidates=8,
        candidate_count=48,
        reranked_candidates=40,
        accepted_results=5,
        final_context_count=5,
        context_chunks=5,
        context_tokens=1200,
        reranker_top_score=0.92,
        reranker_bottom_score=0.74,
        structured_evidence_count=2,
        hierarchical_evidence_count=1,
    )
    record_stage_end("retrieval")

    record_stage_start("research")
    record_sufficiency_metrics(
        sufficiency=True,
        confidence=0.95,
        coverage=1.0,
        reason="Evidence directly contains requested configuration values.",
        missing_requirements_count=0,
        contradiction_count=0,
        evidence_types=["structured", "prose"],
    )
    record_stage_end("research")

    record_stage_start("generation")
    record_sub_stage("generation", "initial_generation", 320.0)
    record_sub_stage("generation", "grounding", 80.0)
    record_grounding_metrics(
        grounding=True,
        confidence=0.98,
        unsupported_claim_count=0,
        evidence_ref_count=4,
        repair_triggered=False,
        regeneration_triggered=False,
        safe_limitation_triggered=False,
    )
    record_visualization_metrics(
        requested=True,
        generated=True,
        validated=True,
        repaired=False,
        dropped=False,
    )
    record_stage_end("generation")

    metrics = finalize_request_metrics()
    assert metrics is not None
    assert metrics.request_id == req_id
    assert metrics.trace_id == f"trace_{req_id[:8]}"
    assert metrics.workspace_id == "ws-prod-1"
    assert metrics.thread_id == "thread-42"
    assert metrics.provider == "groq"
    assert metrics.model == "llama-3.3-70b"

    # Verify retrieval metrics
    assert metrics.planned_strategies == ["SEMANTIC", "HIERARCHICAL"]
    assert metrics.executed_strategies == ["SEMANTIC", "HIERARCHICAL"]
    assert metrics.candidate_count == 48
    assert metrics.accepted_results == 5
    assert metrics.context_chunks == 5
    assert metrics.context_tokens == 1200
    assert metrics.reranker_top_score == 0.92
    assert metrics.reranker_bottom_score == 0.74
    assert metrics.structured_evidence_count == 2
    assert metrics.hierarchical_evidence_count == 1

    # Verify sufficiency metrics
    assert metrics.sufficiency is True
    assert metrics.sufficiency_confidence == 0.95
    assert metrics.sufficiency_coverage == 1.0
    assert metrics.missing_requirements_count == 0

    # Verify grounding metrics
    assert metrics.grounding is True
    assert metrics.grounding_confidence == 0.98
    assert metrics.unsupported_claim_count == 0
    assert metrics.evidence_ref_count == 4
    assert metrics.repair_triggered is False

    # Verify visualization metrics
    assert metrics.visualization_requested is True
    assert metrics.visualization_generated is True
    assert metrics.visualization_validated is True
    assert metrics.visualization_repaired is False
    assert metrics.visualization_dropped is False


@pytest.mark.asyncio
async def test_failure_taxonomy_recording():
    req_id = str(uuid.uuid4())
    init_request_metrics(req_id)

    record_failure_category(FailureCategory.QUERY_UNDERSTANDING)
    record_failure_category(FailureCategory.GROUNDING)
    record_failure_category(FailureCategory.VISUALIZATION)

    metrics = finalize_request_metrics()
    assert metrics is not None
    assert FailureCategory.QUERY_UNDERSTANDING in metrics.failure_categories
    assert FailureCategory.GROUNDING in metrics.failure_categories
    assert FailureCategory.VISUALIZATION in metrics.failure_categories


@pytest.mark.asyncio
async def test_evaluation_runner_benchmark():
    runner = EvaluationRunner()
    report = await runner.run_all(EVALUATION_DATASET)

    assert report.total_cases == len(EVALUATION_DATASET)
    assert report.total_cases >= 20
    assert report.passed_cases > 0
    assert report.accuracy >= 0.85
    assert report.retrieval_success_rate == 1.0
    assert report.evidence_recall >= 0.90
    assert report.sufficiency_precision >= 0.80
    assert report.sufficiency_recall >= 0.80
    assert report.grounding_pass_rate >= 0.80
    assert report.unsupported_claim_detection_rate >= 0.80
    assert report.valid_visualization_rate >= 0.80
    assert report.security_leakage == 0.0  # Zero leakage security requirement
    assert report.median_latency_ms >= 0.0
    assert len(report.traces) == len(EVALUATION_DATASET)
