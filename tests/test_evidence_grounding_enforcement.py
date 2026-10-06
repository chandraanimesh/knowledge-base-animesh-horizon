import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from rag_app.generation.grounding import AnswerGroundingEvaluator
from rag_app.generation.service import GenerationService
from rag_app.schemas.rag import ConflictNotice, RetrievedChunk


def _make_chunk(text: str, chunk_id: str = "chk-1", file_name: str = "doc.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=str(uuid.uuid4()),
        department_id="dept-1",
        chunk_index=0,
        file_name=file_name,
        text=text,
    )


@pytest.mark.asyncio
async def test_enforcement_correct_factual_answer_delivered_normally():
    """Valid grounded answer is returned unchanged without disclaimer."""
    chunks = [
        _make_chunk(
            text="The project deadline is October 15 and revenue target is $8.42M.",
            file_name="plan.pdf",
        )
    ]
    query = "What is the deadline and revenue target?"
    raw_answer = "The project deadline is October 15 and the revenue target is $8.42M [1]."

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=(f'{{"answer": "{raw_answer}", "visualization": null}}', {"total_tokens": 100})
    )

    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = mock_llm

    response = await service.generate_from_context(
        query=query,
        chunks=chunks,
        llm=mock_llm,
    )

    assert response.grounding is not None
    assert response.grounding.is_grounded is True
    assert "October 15" in response.answer
    assert "$8.42M" in response.answer
    assert "Note:" not in response.answer


@pytest.mark.asyncio
async def test_enforcement_wrong_date_not_delivered_as_fact():
    """When LLM hallucinates an unsupported date, the final output must not remain unchanged as fact."""
    chunks = [
        _make_chunk(
            text="The project deadline is October 15.",
            file_name="plan.pdf",
        )
    ]
    query = "What is the project deadline?"
    hallucinated_answer = "The project deadline is October 20 [1]."

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        side_effect=[
            # 1. Initial generation (hallucinated)
            (f'{{"answer": "{hallucinated_answer}", "visualization": null}}', {"total_tokens": 100}),
            # 2. Constrained regeneration (grounded)
            ('{"answer": "The project deadline is October 15 [1].", "visualization": null}', {"total_tokens": 120}),
        ]
    )

    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = mock_llm

    response = await service.generate_from_context(
        query=query,
        chunks=chunks,
        llm=mock_llm,
    )

    assert response.grounding is not None
    assert response.grounding.is_grounded is True
    assert "October 15" in response.answer
    assert "October 20" not in response.answer


@pytest.mark.asyncio
async def test_enforcement_unrepairable_hallucination_returns_grounded_limitation():
    """When regeneration also fails or repeats hallucination, safe limitation disclaimer is enforced."""
    chunks = [
        _make_chunk(
            text="The project deadline is October 15.",
            file_name="plan.pdf",
        )
    ]
    query = "What is the project deadline?"
    hallucinated_answer = "The project deadline is November 30 [1]."

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=(f'{{"answer": "{hallucinated_answer}", "visualization": null}}', {"total_tokens": 100})
    )

    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = mock_llm

    response = await service.generate_from_context(
        query=query,
        chunks=chunks,
        llm=mock_llm,
    )

    assert response.grounding is not None
    assert response.grounding.is_grounded is False
    assert "could not be verified" in response.answer or "Note:" in response.answer


@pytest.mark.asyncio
async def test_enforcement_fabricated_chart_row_pruned():
    """Fabricated visualization row is deterministically pruned before delivery."""
    chunks = [
        _make_chunk(
            text="Opportunities: Vertex ($920K) and Northstar ($650K).",
            file_name="deals.pdf",
        )
    ]
    query = "Visualize opportunities as a bar chart."
    
    chart_with_fabrication = {
        "type": "bar",
        "title": "Pipeline Opportunities",
        "x_axis": {"label": "Opportunity", "unit": None},
        "y_axis": {"label": "Value", "unit": "USD"},
        "series": [
            {
                "name": "Deals",
                "data": [
                    {"label": "Vertex", "value": 920000},
                    {"label": "Northstar", "value": 650000},
                    {"label": "Starlight", "value": 400000},  # Fabricated
                ],
            }
        ],
    }

    evaluator = AnswerGroundingEvaluator()
    repaired_chart = evaluator.repair_visualization(
        visualization=chart_with_fabrication,
        query=query,
        chunks=chunks,
    )

    assert repaired_chart is not None
    repaired_data = repaired_chart["series"][0]["data"]
    labels = [pt["label"] for pt in repaired_data]
    assert "Vertex" in labels
    assert "Northstar" in labels
    assert "Starlight" not in labels


@pytest.mark.asyncio
async def test_enforcement_chart_filter_violation_pruned():
    """Visualization row that violates the query filter is pruned or chart dropped."""
    chunks = [
        _make_chunk(
            text="Opportunities: Vertex ($920K), Northstar ($650K), and SmallCo ($300K).",
            file_name="deals.pdf",
        )
    ]
    query = "Show only opportunities above $500K as a bar chart."

    chart_with_violating_row = {
        "type": "bar",
        "title": "Opportunities above $500K",
        "x_axis": {"label": "Opportunity", "unit": None},
        "y_axis": {"label": "Value", "unit": "USD"},
        "series": [
            {
                "name": "Deals",
                "data": [
                    {"label": "Vertex", "value": 920000},
                    {"label": "Northstar", "value": 650000},
                    {"label": "SmallCo", "value": 300000},  # Violates > $500K
                ],
            }
        ],
    }

    evaluator = AnswerGroundingEvaluator()
    repaired_chart = evaluator.repair_visualization(
        visualization=chart_with_violating_row,
        query=query,
        chunks=chunks,
    )

    assert repaired_chart is not None
    repaired_data = repaired_chart["series"][0]["data"]
    labels = [pt["label"] for pt in repaired_data]
    assert "Vertex" in labels
    assert "Northstar" in labels
    assert "SmallCo" not in labels


@pytest.mark.asyncio
async def test_enforcement_contradictory_evidence_flagged():
    """Contradictory evidence without uncertainty qualification is flagged and qualified."""
    chunks = [
        _make_chunk(
            text="Initial release deadline is October 15.",
            chunk_id="chk-1",
            file_name="spec_v1.pdf",
        ),
        _make_chunk(
            text="Updated release deadline is October 20.",
            chunk_id="chk-2",
            file_name="spec_v2.pdf",
        ),
    ]
    conflicts = [
        ConflictNotice(
            subject="release deadline",
            predicate="target_date",
            details="Document spec_v1 says October 15 whereas spec_v2 says October 20.",
            conflicting_chunk_ids=["chk-1", "chk-2"],
        )
    ]
    query = "Are there conflicting values for the deadline?"
    one_sided_answer = "The release deadline is October 15 [1]."

    evaluator = AnswerGroundingEvaluator()
    result = evaluator.evaluate_deterministic(
        query=query,
        answer=one_sided_answer,
        chunks=chunks,
        conflicts=conflicts,
    )

    assert result.is_grounded is False
    assert len(result.contradictions) > 0
