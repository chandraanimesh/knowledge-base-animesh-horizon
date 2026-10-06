import uuid
import pytest

from rag_app.generation.grounding import AnswerGroundingEvaluator
from rag_app.schemas.rag import RetrievedChunk


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
# 1. Exact Factual Values (Numbers & Dates)
# ===========================================================================

def test_grounding_with_correct_date():
    """Verify answer with exact supported date is grounded."""
    evaluator = AnswerGroundingEvaluator()
    chunks = [make_chunk("The release deadline is October 15, 2026 for the system.")]
    query = "What is the release deadline?"
    answer = "The release deadline is October 15, 2026."
    res = evaluator.evaluate_deterministic(query=query, answer=answer, chunks=chunks)
    assert res.is_grounded is True
    assert len(res.unsupported_claims) == 0


def test_grounding_with_wrong_date():
    """Verify answer with hallucinated date is flagged as unsupported."""
    evaluator = AnswerGroundingEvaluator()
    chunks = [make_chunk("The release deadline is October 15, 2026 for the system.")]
    query = "What is the release deadline?"
    answer = "The release deadline is October 20, 2026."
    res = evaluator.evaluate_deterministic(query=query, answer=answer, chunks=chunks)
    assert res.is_grounded is False
    assert any("October 20" in claim for claim in res.unsupported_claims)


def test_grounding_with_correct_numeric_metric():
    """Verify answer with exact supported revenue is grounded."""
    evaluator = AnswerGroundingEvaluator()
    chunks = [make_chunk("Q3 revenue reached $8.42M across all regions.")]
    query = "What was the Q3 revenue?"
    answer = "The Q3 revenue was $8.42M."
    res = evaluator.evaluate_deterministic(query=query, answer=answer, chunks=chunks)
    assert res.is_grounded is True
    assert len(res.unsupported_claims) == 0


def test_grounding_with_wrong_numeric_metric():
    """Verify answer with hallucinated revenue value is flagged as unsupported."""
    evaluator = AnswerGroundingEvaluator()
    chunks = [make_chunk("Q3 revenue reached $8.42M across all regions.")]
    query = "What was the Q3 revenue?"
    answer = "The Q3 revenue was $9.42M."
    res = evaluator.evaluate_deterministic(query=query, answer=answer, chunks=chunks)
    assert res.is_grounded is False
    assert any("9.42" in claim for claim in res.unsupported_claims)


# ===========================================================================
# 2. Visualization Grounding & Filter Validation
# ===========================================================================

def test_visualization_grounding_with_supported_rows():
    """Verify chart with supported data points is grounded."""
    evaluator = AnswerGroundingEvaluator()
    chunks = [make_chunk("| Opportunity | Value |\n|---|---|\n| Deal Alpha | $600K |\n| Deal Beta | $750K |")]
    query = "Show pipeline opportunities as a chart"
    answer = "Here are the opportunities."
    viz = {
        "type": "bar",
        "data": [
            {"Opportunity": "Deal Alpha", "Value": 600000},
            {"Opportunity": "Deal Beta", "Value": 750000},
        ],
    }
    res = evaluator.evaluate_deterministic(query=query, answer=answer, chunks=chunks, visualization=viz)
    assert res.is_grounded is True


def test_visualization_grounding_with_fabricated_row():
    """Verify chart with fabricated opportunity row is flagged as unsupported."""
    evaluator = AnswerGroundingEvaluator()
    chunks = [make_chunk("| Opportunity | Value |\n|---|---|\n| Deal Alpha | $600K |\n| Deal Beta | $750K |")]
    query = "Show pipeline opportunities as a chart"
    answer = "Here are the opportunities."
    viz = {
        "type": "bar",
        "data": [
            {"Opportunity": "Deal Alpha", "Value": 600000},
            {"Opportunity": "Deal Fabricated Corp", "Value": 990000},
        ],
    }
    res = evaluator.evaluate_deterministic(query=query, answer=answer, chunks=chunks, visualization=viz)
    assert res.is_grounded is False
    assert any("Deal Fabricated Corp" in claim for claim in res.unsupported_claims)


def test_visualization_grounding_filter_violation():
    """Verify chart violating requested filter (e.g. >$500K with a $480K row) is flagged."""
    evaluator = AnswerGroundingEvaluator()
    chunks = [make_chunk("| Opportunity | Value |\n|---|---|\n| Deal Alpha | $600K |\n| Deal Small | $480K |")]
    query = "Show only opportunities above $500K as a chart"
    answer = "Here are the opportunities above $500K."
    viz = {
        "type": "bar",
        "data": [
            {"Opportunity": "Deal Alpha", "Value": 600000},
            {"Opportunity": "Deal Small", "Value": 480000},
        ],
    }
    res = evaluator.evaluate_deterministic(query=query, answer=answer, chunks=chunks, visualization=viz)
    assert res.is_grounded is False
    assert any("violates query filter" in claim for claim in res.unsupported_claims)


# ===========================================================================
# 3. Conversation & History Grounding
# ===========================================================================

def test_conversation_grounding_with_history():
    """Verify conversational answer matching conversation history is grounded."""
    evaluator = AnswerGroundingEvaluator()
    conv_history = [
        {"role": "user", "content": "How do we implement authentication?"},
        {"role": "assistant", "content": "We use JWT authentication."},
    ]
    query = "What did I ask earlier?"
    answer = "Earlier you asked how to implement authentication."
    res = evaluator.evaluate_deterministic(
        query=query,
        answer=answer,
        chunks=[],
        conversation_history=conv_history,
    )
    assert res.is_grounded is True


def test_conversation_grounding_with_unsupported_history_claim():
    """Verify conversational answer claiming unmentioned discussion is flagged."""
    evaluator = AnswerGroundingEvaluator()
    conv_history = [
        {"role": "user", "content": "How do we implement authentication?"},
        {"role": "assistant", "content": "We use JWT authentication."},
    ]
    query = "What did I ask earlier?"
    answer = "Earlier you asked about the $9.42M budget allocation."
    res = evaluator.evaluate_deterministic(
        query=query,
        answer=answer,
        chunks=[],
        conversation_history=conv_history,
    )
    assert res.is_grounded is False
    assert any("9.42" in claim for claim in res.unsupported_claims)


# ===========================================================================
# 4. Contradiction Handling
# ===========================================================================

def test_contradiction_grounding_when_unresolved_claim_made():
    """Verify answer asserting certainty on conflicting document values is flagged."""
    evaluator = AnswerGroundingEvaluator()
    chunks = [
        make_chunk("Section A mentions latency is 50ms, while Section B mentions latency is 200ms.")
    ]
    query = "The document mentions two different values. Which one is correct?"
    answer = "The latency is definitely 50ms and Section B is outdated."
    res = evaluator.evaluate_deterministic(query=query, answer=answer, chunks=chunks)
    assert res.is_grounded is False
    assert len(res.contradictions) > 0


# ===========================================================================
# 5. Mixed Conversation + Document Grounding
# ===========================================================================

def test_mixed_grounding_both_supported():
    """Verify answer with both conversation and document evidence is grounded."""
    evaluator = AnswerGroundingEvaluator()
    conv_history = [{"role": "user", "content": "What was the previous question about JWT?"}]
    chunks = [make_chunk("The document specifies token expiration is 60 minutes.")]
    query = "What did I ask earlier, and what does the document say about it?"
    answer = "Earlier you asked about JWT, and the document specifies token expiration is 60 minutes."
    res = evaluator.evaluate_deterministic(
        query=query,
        answer=answer,
        chunks=chunks,
        conversation_history=conv_history,
    )
    assert res.is_grounded is True


def test_mixed_grounding_with_unsupported_document_claim():
    """Verify answer with hallucinated document claim is flagged."""
    evaluator = AnswerGroundingEvaluator()
    conv_history = [{"role": "user", "content": "What was the previous question about JWT?"}]
    chunks = [make_chunk("The document specifies token expiration is 60 minutes.")]
    query = "What did I ask earlier, and what does the document say about it?"
    answer = "Earlier you asked about JWT, and the document specifies token expiration is 120 minutes."
    res = evaluator.evaluate_deterministic(
        query=query,
        answer=answer,
        chunks=chunks,
        conversation_history=conv_history,
    )
    assert res.is_grounded is False
    assert any("120" in claim for claim in res.unsupported_claims)

