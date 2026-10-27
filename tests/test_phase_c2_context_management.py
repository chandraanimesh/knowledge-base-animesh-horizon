import uuid
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_app.context.compiler import ContextCompiler, estimate_tokens
from rag_app.context.contracts import SourceAuthority, ContextEvidence
from rag_app.schemas.rag import (
    RetrievedChunk,
    ChunkProvenance,
    GraphEdgeProvenance,
    GenerateResponse,
    ConflictNotice,
)
from rag_app.schemas.research import CleanWebPage
from rag_app.research.provider import WebPageResult
from rag_app.research.cleaner import clean_web_page
from rag_app.research.extractor import extract_knowledge
from rag_app.generation.service import (
    GenerationService,
    is_retryable_llm_error,
    _wait_strategy_with_retry_after,
)
from rag_app.generation.citations import validate_and_render_citations
from rag_app.db.models import User


# =============================================================================
# PATCH C2.1 — GROUNDING REGENERATION BUDGET GUARD TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_grounding_regeneration_in_budget_invokes_provider():
    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = MagicMock()
    service.llm.generate = AsyncMock(
        side_effect=[
            # Initial generation: ungrounded statement to trigger repair
            ('{"answer": "Company was founded in 1999 [chunk: chk1].", "visualization": null}', {"input_tokens": 50, "output_tokens": 20}),
            # Regeneration: grounded response
            ('{"answer": "Company was founded in 2020 according to records [chunk: chk1].", "visualization": null}', {"input_tokens": 80, "output_tokens": 20}),
        ]
    )

    dept_id = uuid.uuid4()
    chunk = RetrievedChunk(
        chunk_id="chk1",
        document_id="doc1",
        department_id=str(dept_id),
        file_name="about.pdf",
        chunk_index=0,
        text="Acme Corp was founded in 2020 in California.",
        hybrid_score=0.9,
    )

    with patch("rag_app.generation.grounding.AnswerGroundingEvaluator.evaluate") as mock_eval:
        from rag_app.generation.grounding import AnswerGroundingResult
        mock_eval.side_effect = [
            # 1st eval: ungrounded (triggers regeneration)
            AnswerGroundingResult(is_grounded=False, unsupported_claims=["founded in 1999"]),
            # 2nd eval: grounded
            AnswerGroundingResult(is_grounded=True),
        ]

        resp = await service.generate_from_context(
            query="When was Acme founded?",
            chunks=[chunk],
            allowed_departments=[dept_id],
        )

        # Provider must be invoked twice: once for initial, once for regeneration
        assert service.llm.generate.call_count == 2
        assert "2020" in resp.answer


@pytest.mark.asyncio
async def test_grounding_regeneration_oversized_payload_never_invokes_provider():
    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = MagicMock()
    service.llm.generate = AsyncMock(
        return_value=('{"answer": "Fabricated claim unsupported [chunk: chk1].", "visualization": null}', {"input_tokens": 50, "output_tokens": 20})
    )

    dept_id = uuid.uuid4()
    # Chunk with 250 words
    chunk = RetrievedChunk(
        chunk_id="chk1",
        document_id="doc1",
        department_id=str(dept_id),
        file_name="doc.pdf",
        chunk_index=0,
        text="Official operational record with verified evidence details " * 20,
        hybrid_score=0.9,
    )

    from rag_app.core.config import get_settings
    base_settings = get_settings()

    with patch("rag_app.generation.grounding.AnswerGroundingEvaluator.evaluate") as mock_eval, \
         patch("rag_app.core.config.get_settings") as mock_settings:
        from rag_app.generation.grounding import AnswerGroundingResult

        # Trigger ungrounded check with large unsupported claims list that pushes regeneration prompt over budget (>5000 tokens)
        large_claims = [f"Unsupported fabricated claim detail {i}: " + ("evidence mismatch " * 100) for i in range(25)]
        mock_eval.return_value = AnswerGroundingResult(is_grounded=False, unsupported_claims=large_claims)

        # Configure budget: initial prompt (~2122 tokens) fits under 2500, but regeneration payload (>3000 tokens) exceeds
        cfg = base_settings.model_copy(
            update={
                "model_context_limit": 3000,
                "reserved_output_tokens": 500,  # max capacity = 2500 tokens
            }
        )
        mock_settings.return_value = cfg

        resp = await service.generate_from_context(
            query="Summarize operations",
            chunks=[chunk],
            allowed_departments=[dept_id],
        )

        # Provider MUST be called exactly once (initial generation), and NEVER for regeneration!
        assert service.llm.generate.call_count == 1
        # Safe limitation fallback must be returned
        assert "cannot confirm" in resp.answer or "not available" in resp.answer
        assert "Unsupported fabricated claim" in resp.answer or "evidence mismatch" in resp.answer


@pytest.mark.asyncio
async def test_grounding_regeneration_prunes_optional_history_to_fit_budget():
    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = MagicMock()
    service.llm.generate = AsyncMock(
        side_effect=[
            ('{"answer": "Ungrounded output [chunk: chk1].", "visualization": null}', {}),
            ('{"answer": "Regenerated grounded output [chunk: chk1].", "visualization": null}', {}),
        ]
    )

    dept_id = uuid.uuid4()
    chunk = RetrievedChunk(
        chunk_id="chk1",
        document_id="doc1",
        department_id=str(dept_id),
        file_name="doc.pdf",
        chunk_index=0,
        text="Official policy detail.",
        hybrid_score=0.9,
    )

    # Supply several historical conversation turns
    conv_history = [
        {"role": "user", "content": "Prior question 1 " * 15},
        {"role": "assistant", "content": "Prior answer 1 " * 15},
        {"role": "user", "content": "Prior question 2 " * 15},
        {"role": "assistant", "content": "Prior answer 2 " * 15},
    ]

    with patch("rag_app.generation.grounding.AnswerGroundingEvaluator.evaluate") as mock_eval:
        from rag_app.generation.grounding import AnswerGroundingResult
        mock_eval.side_effect = [
            AnswerGroundingResult(is_grounded=False, unsupported_claims=["bad detail"]),
            AnswerGroundingResult(is_grounded=True),
        ]

        resp = await service.generate_from_context(
            query="Current policy?",
            chunks=[chunk],
            allowed_departments=[dept_id],
            conversation_history=conv_history,
        )

        # Regeneration succeeds after trimming optional history
        assert service.llm.generate.call_count == 2
        assert "Regenerated grounded output" in resp.answer


# =============================================================================
# PATCH C2.2 — BOUNDED WEB EXTRACTION TESTS
# =============================================================================

def test_web_cleaner_bounds_very_large_page():
    large_markdown = "# Document Header\n\n" + ("This is detailed informational content. " * 1000)
    page = WebPageResult(
        url="https://example.com/huge-doc",
        title="Huge Doc",
        markdown=large_markdown,
    )
    clean = clean_web_page(page, max_chars=5000)

    assert len(clean.content) <= 5000
    assert clean.metadata.get("is_truncated") is True
    assert clean.metadata.get("truncated_from_chars") == len(large_markdown)


@pytest.mark.asyncio
async def test_web_extractor_bounds_tokens_and_preflights_budget():
    huge_page_content = ("Key corporation facts: Revenue $50M, Founded 2010. " * 300)
    clean_page = CleanWebPage(
        url="https://example.com/facts",
        title="Corporation Facts",
        content=huge_page_content,
        metadata={},
    )

    with patch("rag_app.research.extractor.get_llm") as mock_get_llm:
        mock_provider = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=MagicMock(facts=[]))
        mock_provider.llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_provider

        candidate = await extract_knowledge(clean_page)

        # Must have truncated the content before building the request
        assert clean_page.metadata.get("is_truncated") is True
        assert clean_page.metadata.get("bounded_tokens") <= 3000

        # LLM must have been invoked with bounded payload
        assert mock_structured.ainvoke.call_count == 1
        invoked_messages = mock_structured.ainvoke.call_args[0][0]
        user_content = invoked_messages[1].content
        assert len(user_content) < len(huge_page_content)


@pytest.mark.asyncio
async def test_web_extractor_hindi_unicode_content_bounded():
    hindi_content = "यह कंपनी का आधिकारिक वित्तीय डेटा है: लाभ $10M, स्थान नई दिल्ली। " * 250
    clean_page = CleanWebPage(
        url="https://example.com/hindi-facts",
        title="वित्तीय रिपोर्ट",
        content=hindi_content,
        metadata={},
    )

    with patch("rag_app.research.extractor.get_llm") as mock_get_llm:
        mock_provider = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=MagicMock(facts=[]))
        mock_provider.llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_provider

        candidate = await extract_knowledge(clean_page)
        assert mock_structured.ainvoke.call_count == 1
        assert clean_page.metadata.get("bounded_tokens") <= 3000


@pytest.mark.asyncio
async def test_web_extractor_oversized_payload_fails_closed_without_provider_call():
    clean_page = CleanWebPage(
        url="https://example.com/page",
        title="Page",
        content="Normal text content that cannot fit in a 100 token capacity.",
        metadata={},
    )

    from rag_app.core.config import get_settings
    base_settings = get_settings()

    with patch("rag_app.research.extractor.get_llm") as mock_get_llm, \
         patch("rag_app.core.config.get_settings") as mock_settings:
        # Tiny capacity
        cfg = base_settings.model_copy(
            update={
                "model_context_limit": 150,
                "reserved_output_tokens": 100,  # max capacity = 50 tokens
            }
        )
        mock_settings.return_value = cfg

        mock_provider = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock()
        mock_provider.llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_provider

        candidate = await extract_knowledge(clean_page)

        # Provider must NOT be invoked when total prompt exceeds capacity
        mock_structured.ainvoke.assert_not_called()
        assert candidate.facts == []


# =============================================================================
# PATCH C2.3 — CORRECT PROVIDER RETRY CLASSIFICATION TESTS
# =============================================================================

def test_is_retryable_llm_error_classification():
    # 1. Non-retryable: 413 Payload Too Large
    err_413 = Exception("413 Request Entity Too Large: maximum context length exceeded")
    assert is_retryable_llm_error(err_413) is False

    err_413_status = MagicMock()
    err_413_status.status_code = 413
    assert is_retryable_llm_error(err_413_status) is False

    # 2. Non-retryable: 400 Bad Request / 401 Auth
    err_400 = MagicMock()
    err_400.status_code = 400
    assert is_retryable_llm_error(err_400) is False

    err_401 = MagicMock()
    err_401.status_code = 401
    assert is_retryable_llm_error(err_401) is False

    # 3. Non-retryable: Cancellation
    assert is_retryable_llm_error(asyncio.CancelledError()) is False

    # 4. Retryable: 429 Rate Limit
    err_429 = MagicMock()
    err_429.status_code = 429
    assert is_retryable_llm_error(err_429) is True

    err_429_msg = Exception("Rate limit reached for model groq. Please try again in 2s.")
    assert is_retryable_llm_error(err_429_msg) is True

    # 5. Retryable: 503 / 502 / Network Timeout
    err_503 = MagicMock()
    err_503.status_code = 503
    assert is_retryable_llm_error(err_503) is True

    assert is_retryable_llm_error(TimeoutError("Connection timed out")) is True
    assert is_retryable_llm_error(ConnectionError("Connection reset by peer")) is True


def test_wait_strategy_respects_and_caps_retry_after():
    class MockOutcome:
        def __init__(self, exc):
            self._exc = exc
        def exception(self):
            return self._exc

    class MockRetryState:
        def __init__(self, exc, attempt_number=1):
            self.outcome = MockOutcome(exc)
            self.attempt_number = attempt_number

    # 429 error with Retry-After: 3.5s
    err_429 = MagicMock()
    err_429.retry_after = 3.5
    state = MockRetryState(err_429)
    wait_time = _wait_strategy_with_retry_after(state)
    assert wait_time == 3.5

    # 429 error with huge Retry-After: 120s (capped at 10.0s)
    err_huge = MagicMock()
    err_huge.retry_after = 120.0
    state_huge = MockRetryState(err_huge)
    wait_time_huge = _wait_strategy_with_retry_after(state_huge)
    assert wait_time_huge == 10.0


@pytest.mark.asyncio
async def test_invoke_llm_fails_fast_on_413_without_retries():
    from rag_app.generation.service import _invoke_llm_with_retry

    mock_llm = MagicMock()
    err_413 = Exception("413 Payload Too Large: context_length_exceeded")
    mock_llm.generate = AsyncMock(side_effect=err_413)

    with pytest.raises(Exception, match="413 Payload Too Large"):
        await _invoke_llm_with_retry(mock_llm, [{"role": "user", "content": "test"}])

    # Must be called exactly 1 time; no retries on 413!
    assert mock_llm.generate.call_count == 1


# =============================================================================
# PATCH C2.4 — SEPARATE WEB EVIDENCE QUOTA TESTS
# =============================================================================

def test_web_evidence_quota_mixed_capped_at_40_percent():
    """When both internal and web chunks exist, web chunks cannot exceed 40% of evidence tokens."""
    compiler = ContextCompiler(max_context_tokens=1000)
    dept_id = uuid.uuid4()

    # Internal chunks: 2 chunks of 150 tokens = 300 tokens
    internal_chunks = [
        RetrievedChunk(
            chunk_id=f"int_{i}",
            document_id=f"doc_{i}",
            department_id=str(dept_id),
            file_name="internal_policy.pdf",
            chunk_index=i,
            text=f"Internal corporate policy details paragraph {i}. " * 15,
            hybrid_score=0.95 - (i * 0.05),
        )
        for i in range(2)
    ]

    # Web chunks: 5 chunks of 150 tokens = 750 tokens (would flood without quota!)
    web_chunks = [
        RetrievedChunk(
            chunk_id=f"web_search_{i}",
            document_id=f"https://external.com/page_{i}",
            department_id="None",
            file_name=f"https://external.com/page_{i}",
            chunk_index=i,
            text=f"External web search result information {i}. " * 15,
            hybrid_score=0.99,  # Higher score than internal!
        )
        for i in range(5)
    ]

    combined = internal_chunks + web_chunks

    compiled = compiler.compile(
        chunks=combined,
        query="What is the policy?",
        allowed_departments=[dept_id],
    )

    selected_ids = [c.chunk_id for c in compiled.chunks]
    # Internal chunks MUST survive and not be crowded out
    assert any(c.startswith("int_") for c in selected_ids)

    # Some web chunks must be dropped with WEB_EVIDENCE_QUOTA_EXCEEDED
    assert "WEB_EVIDENCE_QUOTA_EXCEEDED" in compiled.dropped_reasons.values()

    # Web token share must not exceed 40% of the available evidence budget
    web_tokens_selected = sum(estimate_tokens(c.text) for c in compiled.chunks if c.chunk_id.startswith("web_"))
    total_evidence_budget = compiled.available_context_tokens
    assert web_tokens_selected <= int(total_evidence_budget * 0.40) + 10  # small margin for rounding


def test_web_evidence_quota_web_only_uses_full_budget():
    """When only web chunks exist (web-only research), web chunks can use the full evidence budget."""
    compiler = ContextCompiler(max_context_tokens=500)

    web_chunks = [
        RetrievedChunk(
            chunk_id=f"web_only_{i}",
            document_id=f"https://external.com/page_{i}",
            department_id="None",
            file_name=f"https://external.com/page_{i}",
            chunk_index=i,
            text=f"External web only research chunk {i}. " * 8,
            hybrid_score=0.90,
        )
        for i in range(3)
    ]

    compiled = compiler.compile(
        chunks=web_chunks,
        query="What happened externally?",
        allowed_departments=[],
        is_web_context=True,
    )

    # Web chunks can consume beyond 40% when web-only
    assert len(compiled.chunks) >= 2
    assert "WEB_EVIDENCE_QUOTA_EXCEEDED" not in compiled.dropped_reasons.values()


def test_web_evidence_quota_conflict_chunks_always_preserved():
    """Protected conflicting chunks (graph edge divergence) are preserved regardless of web quota."""
    compiler = ContextCompiler(max_context_tokens=400)
    dept_id = uuid.uuid4()

    internal_chunk = RetrievedChunk(
        chunk_id="int_conf_1",
        document_id="doc_internal",
        department_id=str(dept_id),
        file_name="report.pdf",
        chunk_index=0,
        text="Acme valuation was reported as 100M.",
        hybrid_score=0.9,
    )

    # Web chunk conflicting on (Acme, valuation)
    web_chunk_conflict = RetrievedChunk(
        chunk_id="web_conf_2",
        document_id="https://news.com/acme",
        department_id="None",
        file_name="https://news.com/acme",
        chunk_index=0,
        text="Acme valuation was reported as 250M.",
        hybrid_score=0.85,
    )

    prov = {
        "int_conf_1": ChunkProvenance(
            strategies=["graph"],
            graph_path=[GraphEdgeProvenance(subject="Acme", predicate="valuation", object="100M", score=0.9)],
        ),
        "web_conf_2": ChunkProvenance(
            strategies=["graph"],
            graph_path=[GraphEdgeProvenance(subject="Acme", predicate="valuation", object="250M", score=0.85)],
        ),
    }

    compiled = compiler.compile(
        chunks=[internal_chunk, web_chunk_conflict],
        query="What is Acme valuation?",
        allowed_departments=[dept_id],
        chunk_provenance=prov,
    )

    # Both conflicting chunks MUST be preserved in compiled.chunks
    selected_cids = {c.chunk_id for c in compiled.chunks}
    assert "int_conf_1" in selected_cids
    assert "web_conf_2" in selected_cids
    assert len(compiled.conflicts) == 1
