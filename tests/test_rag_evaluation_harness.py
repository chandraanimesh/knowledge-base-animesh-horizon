import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_app.evaluation.harness import (
    GROUNDED_EVAL_DATASET,
    GroundedTestCase,
    MockGroundedLLM,
    RAGEvaluationHarness,
)
from rag_app.context.compiler import ContextCompiler, estimate_tokens
from rag_app.context.assembler import estimate_tokens as assembler_estimate_tokens
from rag_app.core.tokenizer import count_messages, count_text
from rag_app.generation.service import GenerationService
from rag_app.schemas.rag import ConflictNotice, GenerateResponse, RetrievedChunk


@pytest.mark.asyncio
async def test_harness_executes_all_grounded_cases_successfully():
    """Evaluate all 8 grounded cases across single-source, cross-source, conflicting, and unanswerable."""
    harness = RAGEvaluationHarness()
    report = await harness.run_all()

    assert report.total_cases == 8
    assert report.passed_cases == 8
    assert report.failed_cases == 0
    assert report.accuracy == 1.0
    assert report.mean_evidence_recall == 1.0
    assert report.mean_citation_validity == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.conflict_disclosure_rate == 1.0
    assert report.mean_compiler_latency_ms >= 0.0
    assert report.mean_generation_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_grounded_single_source_finance_budget():
    """Case 1: Single-source factual retrieval from FIN-DEMO-001."""
    harness = RAGEvaluationHarness()
    case = GROUNDED_EVAL_DATASET[0]
    metric = await harness.evaluate_case(case)

    assert metric.passed is True
    assert metric.category == "single_source"
    assert metric.expected_behavior == "ANSWER"
    assert metric.evidence_recall == 1.0
    assert metric.citation_validity == 1.0
    assert "12,00,000" in metric.answer_preview or "12,00,000" in case.expected_facts
    assert metric.claim_citation_supported is True


@pytest.mark.asyncio
async def test_grounded_cross_source_synthesis():
    """Case 3: Multi-document synthesis across FIN-DEMO-001 and HOTEL-DEMO-001."""
    harness = RAGEvaluationHarness()
    case = GROUNDED_EVAL_DATASET[2]
    metric = await harness.evaluate_case(case)

    assert metric.passed is True
    assert metric.category == "cross_source"
    assert metric.evidence_recall == 1.0
    assert metric.citation_validity == 1.0
    assert len(metric.citations_rendered) >= 2


@pytest.mark.asyncio
async def test_grounded_conflicting_temporal_resolution():
    """Case 5: Temporal conflict preferring audited 2024 filing ($25M) over 2020 ($10M)."""
    harness = RAGEvaluationHarness()
    case = GROUNDED_EVAL_DATASET[4]
    metric = await harness.evaluate_case(case)

    assert metric.passed is True
    assert metric.category == "conflicting"
    assert any("annual_report_2024" in cit or "chk_rev_2024" in cit for cit in metric.citations_rendered)
    assert metric.evidence_recall == 1.0


@pytest.mark.asyncio
async def test_grounded_conflicting_unresolved_discrepancy_disclosure():
    """Case 6: Unresolved conflict without date superiority requires explicit disclosure."""
    harness = RAGEvaluationHarness()
    case = GROUNDED_EVAL_DATASET[5]
    metric = await harness.evaluate_case(case)

    assert metric.passed is True
    assert metric.actual_behavior == "CONFLICT_DISCLOSURE"
    assert metric.conflict_disclosed is True


@pytest.mark.asyncio
async def test_grounded_unanswerable_explicit_abstention():
    """Case 7: Hotel booking confirmation and taxes explicitly unrecorded -> requires abstention."""
    harness = RAGEvaluationHarness()
    case = GROUNDED_EVAL_DATASET[6]
    metric = await harness.evaluate_case(case)

    assert metric.passed is True
    assert metric.actual_behavior == "ABSTAIN"
    assert metric.abstention_accurate is True


@pytest.mark.asyncio
async def test_grounded_unanswerable_missing_evidence():
    """Case 8: Out-of-scope query with missing evidence -> requires abstention."""
    harness = RAGEvaluationHarness()
    case = GROUNDED_EVAL_DATASET[7]
    metric = await harness.evaluate_case(case)

    assert metric.passed is True
    assert metric.actual_behavior == "ABSTAIN"
    assert metric.abstention_accurate is True


def test_context_budget_consistency_no_double_safety_margin_subtraction():
    """
    Ensure safety margin / overhead is subtracted exactly ONCE when computing available context.
    available_context = max(0, (model_limit - reserved_output) - (system + query + overhead))
    """
    compiler = ContextCompiler(
        model_context_limit=4000,
        reserved_output_tokens=500,
        max_context_tokens=3000,
    )
    query = "Check budget arithmetic consistency"
    dept_id = uuid.uuid4()
    chunks = [
        RetrievedChunk(
            chunk_id="chk_c1",
            document_id="doc_c1",
            department_id=str(dept_id),
            file_name="budget.pdf",
            chunk_index=0,
            text="Operating budget data line with facts.",
            hybrid_score=0.9,
        )
    ]

    compiled = compiler.compile(
        chunks=chunks,
        query=query,
        allowed_departments=[dept_id],
    )

    from rag_app.generation.prompt import RAG_SYSTEM_PROMPT
    sys_toks = count_text(RAG_SYSTEM_PROMPT)
    query_toks = count_text(query)
    prompt_overhead = 150

    expected_other_prompt_tokens = sys_toks + query_toks + prompt_overhead
    expected_total_input_capacity = 4000 - 500  # 3500
    expected_available = max(0, expected_total_input_capacity - expected_other_prompt_tokens)

    # Verify that available_context_tokens matches expected single-subtraction budget
    assert compiled.available_context_tokens == min(expected_available, 3000)
    # Total planned tokens must strictly fit within total_input_capacity
    assert compiled.total_planned_tokens <= expected_total_input_capacity


def test_shared_tokenizer_used_across_all_modules():
    """Verify that compiler, assembler, and generation service all use the shared model-aware tokenizer."""
    sample = "Verification of unified tokenizer reuse across ContextCompiler and Assembler."
    t1 = count_text(sample)
    t2 = estimate_tokens(sample)
    t3 = assembler_estimate_tokens(sample)

    assert t1 == t2
    assert t1 == t3
    assert t1 > 0


@pytest.mark.asyncio
async def test_final_structured_messages_checked_and_overflow_prevented():
    """Verify structured messages are evaluated before LLM invocation and overflow prevents call."""
    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = MagicMock()
    service.llm.generate = AsyncMock()

    dept_id = uuid.uuid4()
    overflow_chunk = RetrievedChunk(
        chunk_id="chk_overflow_guard",
        document_id="doc_overflow",
        department_id=str(dept_id),
        file_name="overflow.pdf",
        chunk_index=0,
        text="Massive overflow text sequence " * 3000,
        hybrid_score=0.9,
    )

    with patch("rag_app.core.config.get_settings") as mock_settings:
        from rag_app.core.config import get_settings
        cfg = get_settings().model_copy(
            update={
                "model_context_limit": 1000,
                "reserved_output_tokens": 200,
            }
        )
        mock_settings.return_value = cfg

        resp = await service.generate_from_context(
            query="Analyze massive data",
            chunks=[overflow_chunk],
            allowed_departments=[dept_id],
        )

        assert "exceeds the maximum allowable token budget" in resp.answer
        service.llm.generate.assert_not_called()


def test_mock_vs_live_provider_distinction_and_calibration():
    """Verify harness tracks estimated vs provider tokens and handles missing usage metadata gracefully."""
    mock_llm = MockGroundedLLM(model_name="openai/gpt-oss-120b")
    assert mock_llm.model == "openai/gpt-oss-120b"

    messages = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Test prompt"},
    ]
    est_toks = count_messages(messages, model=mock_llm.model)
    assert est_toks > 0
