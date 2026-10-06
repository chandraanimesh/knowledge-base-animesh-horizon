import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock

from rag_app.retrieval.sufficiency import EvidenceSufficiencyEvaluator
from rag_app.schemas.rag import (
    EvidenceNeed,
    InformationNeed,
    PresentationNeed,
    QueryPlan,
    RetrievedChunk,
)
from rag_app.schemas.research import (
    SufficiencyEvaluation,
    SufficiencyResult,
    SufficiencyStatus,
)


def make_chunk(text: str, file_name: str = "doc.md", chunk_index: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        department_id=str(uuid.uuid4()),
        text=text,
        file_name=file_name,
        chunk_index=chunk_index,
        hybrid_score=0.85,
    )


# ===========================================================================
# 1. CORE SPECIFICATION TESTS (User Required Cases 1 - 5)
# ===========================================================================

def test_core_case_1_sufficient():
    """
    Test 1 — Sufficient:
    Query: 'What is the implementation timeline?'
    Context: 'Implementation timeline under 60 days.'
    Expected: status = SufficiencyStatus.SUFFICIENT, evidence_found = True
    """
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What is the implementation timeline?"
    chunks = [
        make_chunk("The product implementation timeline is under 60 days from signing.")
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.SUFFICIENT
    assert res.is_sufficient is True
    assert res.evidence_found is True
    assert res.answerable_from_context is True
    assert len(res.missing_requirements) == 0


def test_core_case_2_exact_date_missing():
    """
    Test 2 — Exact date missing:
    Query: 'What is the exact implementation deadline?'
    Context: 'Implementation timeline under 60 days.'
    Expected: status = SufficiencyStatus.INSUFFICIENT, missing_requirements contains exact deadline date
    """
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What is the exact implementation deadline?"
    chunks = [
        make_chunk("The product implementation timeline is under 60 days from contract signing.")
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.INSUFFICIENT
    assert res.is_sufficient is False
    assert res.evidence_found is True
    assert any("deadline" in req.lower() for req in res.missing_requirements)


def test_core_case_3_partial_structured_evidence():
    """
    Test 3 — Partial structured evidence:
    Query: 'Show all Q3 pipeline opportunities with their values.'
    Context: Vertex = $920K, Northstar = $650K, ABC = no explicit opportunity value
    Expected: status = SufficiencyStatus.PARTIAL, evidence exists but coverage is incomplete
    """
    evaluator = EvidenceSufficiencyEvaluator()
    query = "Show all Q3 pipeline opportunities with their values."
    chunks = [
        make_chunk("Q3 Pipeline: Vertex has value $920K, Northstar has value $650K, and ABC has no explicit opportunity value recorded.")
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.PARTIAL
    assert res.evidence_found is True
    assert res.answerable_from_context is True
    assert len(res.missing_requirements) > 0


def test_core_case_4_no_relevant_evidence():
    """
    Test 4 — No relevant evidence:
    Query: 'What is the implementation deadline?'
    Context: unrelated sales information with no implementation timeline.
    Expected: status = SufficiencyStatus.INSUFFICIENT, evidence_found = False
    """
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What is the implementation deadline?"
    chunks = [
        make_chunk("Q3 regional sales numbers exceeded quarterly targets by fifteen percent.")
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.INSUFFICIENT
    assert res.evidence_found is False
    assert res.answerable_from_context is False


def test_core_case_5_contradiction():
    """
    Test 5 — Contradiction:
    Query: 'What is the implementation timeline?'
    Context: Doc 1: 'Implementation will take 60 days.' Doc 2: 'Implementation will take 90 days.'
    Expected: status = PARTIAL (or INSUFFICIENT), contradictions is non-empty
    """
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What is the implementation timeline?"
    chunks = [
        make_chunk("Policy A states that implementation will take 60 days.", file_name="doc_a.md"),
        make_chunk("Policy B states that implementation will take 90 days.", file_name="doc_b.md"),
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status in (SufficiencyStatus.PARTIAL, SufficiencyStatus.INSUFFICIENT)
    assert len(res.contradictions) > 0
    assert "60" in str(res.contradictions[0]) and "90" in str(res.contradictions[0])


# ===========================================================================
# 2. DETERMINISTIC POST-LLM VALIDATOR TESTS
# ===========================================================================

def test_deterministic_validator_downgrades_sufficient_with_missing_requirements():
    """Verify validator downgrades SUFFICIENT to PARTIAL/INSUFFICIENT if missing requirements exist."""
    evaluator = EvidenceSufficiencyEvaluator()
    raw_llm_eval = SufficiencyEvaluation(
        status=SufficiencyStatus.SUFFICIENT,
        evidence_found=True,
        required_information=["total_cost", "deadline"],
        supported_information=["total_cost"],
        missing_requirements=["deadline"],
        answerable_from_context=True,
        reason="Found cost but deadline was missing.",
    )
    validated = evaluator.validate_sufficiency_result(raw_llm_eval)
    assert validated.status == SufficiencyStatus.PARTIAL
    assert "Downgraded to PARTIAL" in validated.reason


def test_deterministic_validator_enforces_no_evidence_cannot_be_sufficient():
    """Verify validator forces INSUFFICIENT when evidence_found is False."""
    evaluator = EvidenceSufficiencyEvaluator()
    raw_llm_eval = SufficiencyEvaluation(
        status=SufficiencyStatus.SUFFICIENT,
        evidence_found=False,
        reason="Claimed sufficient even though no evidence was found.",
    )
    validated = evaluator.validate_sufficiency_result(raw_llm_eval)
    assert validated.status == SufficiencyStatus.INSUFFICIENT
    assert validated.evidence_found is False
    assert validated.answerable_from_context is False


def test_deterministic_validator_downgrades_on_unresolved_contradictions():
    """Verify validator downgrades SUFFICIENT to PARTIAL when contradictions exist."""
    evaluator = EvidenceSufficiencyEvaluator()
    raw_llm_eval = SufficiencyEvaluation(
        status=SufficiencyStatus.SUFFICIENT,
        evidence_found=True,
        supported_information=["timeline"],
        contradictions=["Doc A says 60 days, Doc B says 90 days"],
        reason="Claimed sufficient despite contradiction.",
    )
    validated = evaluator.validate_sufficiency_result(raw_llm_eval)
    assert validated.status == SufficiencyStatus.PARTIAL
    assert "Downgraded to PARTIAL" in validated.reason


# ===========================================================================
# 3. Exact Factual Tests
# ===========================================================================

def test_exact_factual_with_explicit_deadline():
    """Verify factual query is sufficient when explicit deadline date is present."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What is the exact implementation deadline?"
    chunks = [
        make_chunk("The project timeline specifies that the implementation deadline is November 15, 2026.")
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.SUFFICIENT
    assert res.is_sufficient is True
    assert "exact_deadline_date" not in res.missing_requirements


def test_exact_factual_when_deadline_value_is_absent():
    """Verify factual query is insufficient when topic is discussed but deadline date is absent."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What is the exact implementation deadline?"
    chunks = [
        make_chunk("We discussed the implementation timeline and schedule in the previous meeting.")
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.INSUFFICIENT
    assert res.is_sufficient is False
    assert "exact_deadline_date" in res.missing_requirements


# ===========================================================================
# 4. Requirements Tests
# ===========================================================================

def test_requirements_with_actionable_specifications():
    """Verify requirements query is sufficient with actionable specifications."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What needs to be implemented before launch?"
    chunks = [
        make_chunk("5.1 Backend Requirements: The backend must implement PostgreSQL indexing and async Celery workers.")
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.SUFFICIENT
    assert res.is_sufficient is True


def test_requirements_with_generic_prose_only():
    """Verify requirements query is insufficient with purely non-technical generic prose."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What needs to be implemented before launch?"
    chunks = [
        make_chunk("Our company culture embraces high quality and innovation in all endeavors.")
    ]
    plan = QueryPlan(
        semantic_query="what needs to be implemented before launch",
        information_need=InformationNeed.REQUIREMENTS,
        evidence_need=EvidenceNeed.HIERARCHICAL,
    )
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks, query_plan=plan)
    assert res.status == SufficiencyStatus.INSUFFICIENT
    assert res.is_sufficient is False
    assert "actionable_requirements" in res.missing_requirements


# ===========================================================================
# 5. Exhaustive List & Comparison Tests
# ===========================================================================

def test_exhaustive_list_with_complete_items():
    """Verify exhaustive query is sufficient when list items are present."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "List all implementation requirements."
    chunks = [
        make_chunk("- Step 1: Database setup\n- Step 2: API routes\n- Step 3: Auth middleware"),
        make_chunk("- Step 4: Testing\n- Step 5: Deployment"),
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.SUFFICIENT
    assert res.is_sufficient is True


def test_comparison_with_both_sides_present():
    """Verify comparison query is sufficient when both sides are described."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What work is required on the backend versus the frontend?"
    chunks = [
        make_chunk("Backend requires FastAPI endpoints and PostgreSQL migrations."),
        make_chunk("Frontend requires React UI components and Tailwind styling."),
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.SUFFICIENT
    assert res.is_sufficient is True


def test_comparison_when_one_side_is_missing():
    """Verify comparison query is PARTIAL when one side is missing but grounded partial answer possible."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "What work is required on the backend versus the frontend?"
    chunks = [
        make_chunk("Backend requires FastAPI endpoints, JWT auth, and PostgreSQL migrations.")
    ]
    plan = QueryPlan(
        semantic_query="backend versus frontend requirements",
        information_need=InformationNeed.COMPARISON,
    )
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks, query_plan=plan)
    assert res.status == SufficiencyStatus.PARTIAL
    assert "frontend_details" in res.missing_requirements


# ===========================================================================
# 6. Aggregation & Visualization Tests
# ===========================================================================

def test_visualization_with_complete_table_evidence():
    """Verify visualization query is sufficient when structured table data is present."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "Visualize the Q3 sales pipeline opportunities by value as a bar chart."
    chunks = [
        make_chunk("| Opportunity | Value | Stage |\n|---|---|---|\n| Deal Alpha | $600K | Closed |\n| Deal Beta | $450K | Negotiation |")
    ]
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks)
    assert res.status == SufficiencyStatus.SUFFICIENT
    assert "table" in res.evidence_types


def test_visualization_when_only_prose_is_retrieved():
    """Verify visualization query is insufficient when only qualitative prose is retrieved."""
    evaluator = EvidenceSufficiencyEvaluator()
    query = "Visualize the Q3 sales pipeline opportunities by value as a bar chart."
    chunks = [
        make_chunk("Sales performance this quarter was strong, with the team focusing on enterprise accounts.")
    ]
    plan = QueryPlan(
        semantic_query="visualize Q3 sales pipeline opportunities",
        information_need=InformationNeed.AGGREGATION,
        presentation_need=PresentationNeed.CHART,
        evidence_need=EvidenceNeed.STRUCTURED,
    )
    res = evaluator.evaluate_deterministic(query=query, chunks=chunks, query_plan=plan)
    assert res.status == SufficiencyStatus.INSUFFICIENT
    assert "structured_numeric_data" in res.missing_requirements


# ===========================================================================
# 7. Asynchronous Structured LLM Invocation Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_async_evaluate_with_structured_llm():
    """Verify async evaluate properly invokes LLM with structured output and validates result."""
    evaluator = EvidenceSufficiencyEvaluator()
    evaluator._get_llm = MagicMock()
    mock_llm = MagicMock()
    evaluator._get_llm.return_value = mock_llm

    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value={
        "parsed": SufficiencyEvaluation(
            status=SufficiencyStatus.SUFFICIENT,
            evidence_found=True,
            required_information=["implementation timeline"],
            supported_information=["under 60 days"],
            missing_requirements=[],
            answerable_from_context=True,
            reason="Timeline is clearly documented.",
            confidence=0.95,
            coverage=1.0,
        ),
        "raw": MagicMock(usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}),
    })
    mock_llm.llm.with_structured_output.return_value = mock_structured

    chunk = make_chunk("Implementation timeline is under 60 days.")
    res = await evaluator.evaluate(
        query="What is the implementation timeline?",
        chunks=[chunk],
    )
    assert res.status == SufficiencyStatus.SUFFICIENT
    assert res.is_sufficient is True
    assert res.evidence_found is True
