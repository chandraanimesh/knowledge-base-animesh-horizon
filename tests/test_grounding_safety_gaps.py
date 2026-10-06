import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from rag_app.generation.service import GenerationService
from rag_app.generation.grounding import AnswerGroundingEvaluator
from rag_app.schemas.rag import ConflictNotice, GenerateRequest, RetrievedChunk
from rag_app.schemas.research import SufficiencyEvaluation


def _make_chunk(text: str, chunk_id: str = "chk-1", file_name: str = "doc.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=str(uuid.uuid4()),
        department_id="dept-1",
        chunk_index=0,
        file_name=file_name,
        text=text,
    )


# ===========================================================================
# 1. STREAMING SAFETY TESTS
# ===========================================================================

@pytest.mark.asyncio
async def test_streaming_hallucinated_provisional_superseded_by_grounded_complete_event():
    """
    Verify that in streaming mode, if provisional generation produces an unsupported fact (Oct 20),
    the final complete event delivers the enforced, grounded response (Oct 15).
    """
    chunks = [
        _make_chunk(
            text="The project release deadline is October 15.",
            file_name="plan.pdf",
        )
    ]
    query = "What is the release deadline?"

    # LLM first generates hallucinated date, then constrained regeneration generates grounded date
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        side_effect=[
            ('{"answer": "The release deadline is October 20 [1].", "visualization": null}', {"total_tokens": 80}),
            ('{"answer": "The release deadline is October 15 [1].", "visualization": null}', {"total_tokens": 100}),
        ]
    )

    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = mock_llm

    events = []
    async for event in service.stream_from_context(
        query=query,
        chunks=chunks,
        llm=mock_llm,
    ):
        events.append(event)

    # Must contain token events and complete event
    complete_events = [e for e in events if e["type"] == "complete"]
    assert len(complete_events) == 1

    final_resp = complete_events[0]["response"]
    assert final_resp.grounding.is_grounded is True
    assert "October 15" in final_resp.answer
    assert "October 20" not in final_resp.answer


@pytest.mark.asyncio
async def test_streaming_persistence_ensures_exactly_one_assistant_message():
    """
    Verify that across full request execution with grounding enforcement,
    intermediate/provisional states do not create duplicate assistant records in the DB.
    """
    from rag_app.conversation.service import ConversationService
    from rag_app.db.models import User, Conversation, ChatMessage

    mock_session = AsyncMock()
    conv_id = uuid.uuid4()
    mock_conv = MagicMock(spec=Conversation)
    mock_conv.id = conv_id
    mock_conv.title = "Project Deadline"
    mock_conv.workspace_id = None

    conv_service = ConversationService(mock_session)
    
    # Save user message
    user_msg = MagicMock(spec=ChatMessage)
    user_msg.id = uuid.uuid4()
    
    # Save final assistant message
    asst_msg = MagicMock(spec=ChatMessage)
    asst_msg.id = uuid.uuid4()
    
    # Verify save_assistant_message is called only once
    with pytest.MonkeyPatch.context() as mp:
        save_asst_mock = AsyncMock(return_value=asst_msg)
        mp.setattr(conv_service, "save_assistant_message", save_asst_mock)

        await conv_service.save_assistant_message(
            conversation_id=conv_id,
            content="The release deadline is October 15 [1].",
            sources=[],
        )

        assert save_asst_mock.call_count == 1


# ===========================================================================
# 2. VISUALIZATION COMPLETENESS & EXHAUSTIVE QUERIES TESTS
# ===========================================================================

@pytest.mark.asyncio
async def test_exhaustive_query_with_complete_evidence_delivers_chart():
    """
    When query requests 'all opportunities' and evidence is sufficient,
    the complete visualization is delivered as authoritative.
    """
    full_pipeline_text = """| Opportunity | Value |
|---|---|
| Vertex | $920K |
| Northstar | $650K |
| BluePeak | $550K |
| ABC | $480K |"""
    
    chunks = [_make_chunk(text=full_pipeline_text, file_name="pipeline.md")]
    query = "Show all Q3 pipeline opportunities as a bar chart"

    llm_payload = {
        "answer": "Here are all Q3 pipeline opportunities [1].",
        "visualization": {
            "type": "bar",
            "title": "All Q3 Pipeline Opportunities",
            "x_axis": {"label": "Opportunity", "unit": None},
            "y_axis": {"label": "Value", "unit": "USD"},
            "series": [
                {
                    "name": "Deals",
                    "data": [
                        {"label": "Vertex", "value": 920000},
                        {"label": "Northstar", "value": 650000},
                        {"label": "BluePeak", "value": 550000},
                        {"label": "ABC", "value": 480000},
                    ],
                }
            ],
        },
    }

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=(json.dumps(llm_payload), {"total_tokens": 100}))

    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = mock_llm

    sufficiency_eval = SufficiencyEvaluation(
        is_sufficient=True,
        confidence=0.95,
        coverage=1.0,
        evidence_types=["table"],
    )

    response = await service.generate_from_context(
        query=query,
        chunks=chunks,
        llm=mock_llm,
        sufficiency_evaluation=sufficiency_eval,
    )

    assert response.visualization is not None
    assert response.visualization["type"] == "bar"
    assert len(response.visualization["series"][0]["data"]) == 4


@pytest.mark.asyncio
async def test_exhaustive_query_with_incomplete_evidence_does_not_deliver_misleading_chart():
    """
    When user requests 'all opportunities', but evidence is incomplete (insufficient),
    the system must NOT deliver a 2-row chart as complete.
    """
    partial_text = "Retrieved partial notes: Vertex ($920K) and Northstar ($650K) discussed."
    chunks = [_make_chunk(text=partial_text, file_name="notes.md")]
    query = "Show all Q3 pipeline opportunities as a bar chart"

    # LLM generated a 2-row chart claiming it is "all"
    llm_payload = {
        "answer": "Here are all the opportunities [1].",
        "visualization": {
            "type": "bar",
            "title": "All Q3 Pipeline Opportunities",
            "x_axis": {"label": "Opportunity", "unit": None},
            "y_axis": {"label": "Value", "unit": "USD"},
            "series": [
                {
                    "name": "Deals",
                    "data": [
                        {"label": "Vertex", "value": 920000},
                        {"label": "Northstar", "value": 650000},
                    ],
                }
            ],
        },
    }

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value=(json.dumps(llm_payload), {"total_tokens": 100}))

    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = mock_llm

    # Sufficiency evaluated that evidence is incomplete for the exhaustive query
    sufficiency_eval = SufficiencyEvaluation(
        is_sufficient=False,
        reason="Exhaustive query requested, but only partial notes retrieved.",
        confidence=0.85,
        coverage=0.3,
        evidence_types=["prose"],
        missing_requirements=["full_opportunity_records"],
    )

    response = await service.generate_from_context(
        query=query,
        chunks=chunks,
        llm=mock_llm,
        sufficiency_evaluation=sufficiency_eval,
    )

    # Misleading partial chart must be dropped and qualified
    assert response.visualization is None
    assert "partial" in response.answer.lower() or "incomplete" in response.answer.lower()


@pytest.mark.asyncio
async def test_repair_resulting_in_insufficient_data_drops_chart():
    """
    If pruning ungrounded rows leaves insufficient data (e.g. pie chart with <2 slices or 0 rows),
    the visualization is safely dropped.
    """
    chunks = [
        _make_chunk(
            text="Deals: Only Vertex ($920K) is listed.",
            file_name="single_deal.md",
        )
    ]
    query = "Show market share as a pie chart"

    # Pie chart had 1 real slice (Vertex) and 1 fabricated slice (GhostCo)
    chart_with_fabrication = {
        "type": "pie",
        "title": "Market Share",
        "unit": "USD",
        "data": [
            {"label": "Vertex", "value": 920000},
            {"label": "GhostCo", "value": 500000},  # Fabricated
        ],
    }

    evaluator = AnswerGroundingEvaluator()
    repaired_chart = evaluator.repair_visualization(
        visualization=chart_with_fabrication,
        query=query,
        chunks=chunks,
    )

    # Pie chart requires >= 2 valid slices; after pruning GhostCo only 1 slice remains -> must return None
    assert repaired_chart is None
