import pytest
import asyncio
from unittest.mock import AsyncMock

from rag_app.ingestion.resilient_embedder import (
    resilient_batch_embed,
    EmbeddingErrorClassification,
    UnrecoverableEmbeddingError,
    classify_embedding_exception,
)


@pytest.mark.asyncio
async def test_resilient_embed_clean_batch():
    """Verify that a normal clean batch passes directly without bisection."""
    texts = ["doc 1", "doc 2", "doc 3"]
    call_count = 0

    async def mock_embed(batch: list[str]) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        return [[float(len(t))] * 384 for t in batch]

    res = await resilient_batch_embed(texts, mock_embed)
    assert call_count == 1
    assert res.success_count == 3
    assert res.failed_count == 0
    assert len(res.embeddings) == 3
    assert res.embeddings[0] == [5.0] * 384
    assert res.embeddings[1] == [5.0] * 384


@pytest.mark.asyncio
async def test_resilient_embed_single_toxic_chunk():
    """Verify single toxic chunk isolation via binary search and valid chunk rescue."""
    texts = ["good 0", "good 1", "POISON CHUNK", "good 3", "good 4"]

    async def mock_embed(batch: list[str]) -> list[list[float]]:
        for t in batch:
            if "POISON" in t:
                raise ValueError("Toxic malformed token in text")
        return [[float(len(t))] * 4 for t in batch]

    res = await resilient_batch_embed(texts, mock_embed)
    assert res.success_count == 4
    assert res.failed_count == 1
    assert len(res.embeddings) == 5
    # Indexes 0, 1, 3, 4 must be valid
    assert res.embeddings[0] is not None
    assert res.embeddings[1] is not None
    assert res.embeddings[2] is None  # Poison dropped
    assert res.embeddings[3] is not None
    assert res.embeddings[4] is not None
    # Verify failed chunk record
    assert len(res.failed_chunks) == 1
    assert res.failed_chunks[0].chunk_index == 2
    assert "POISON" in res.failed_chunks[0].text_preview
    assert res.failed_chunks[0].error_classification == EmbeddingErrorClassification.TOXIC_INPUT


@pytest.mark.asyncio
async def test_resilient_embed_multiple_toxic_chunks():
    """Verify multiple toxic chunks are all isolated and all valid chunks rescued."""
    texts = ["POISON 0", "good 1", "POISON 2", "good 3"]

    async def mock_embed(batch: list[str]) -> list[list[float]]:
        for t in batch:
            if "POISON" in t:
                raise ValueError("Toxic chunk detected")
        return [[1.0] * 4 for _ in batch]

    res = await resilient_batch_embed(texts, mock_embed)
    assert res.success_count == 2
    assert res.failed_count == 2
    assert res.embeddings[0] is None
    assert res.embeddings[1] is not None
    assert res.embeddings[2] is None
    assert res.embeddings[3] is not None
    assert {fc.chunk_index for fc in res.failed_chunks} == {0, 2}


@pytest.mark.asyncio
async def test_resilient_embed_rate_limit_backoff():
    """Verify 429 rate limit triggers bounded exponential backoff and recovers."""
    texts = ["t1", "t2"]
    attempts = 0

    async def mock_embed(batch: list[str]) -> list[list[float]]:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("Rate limit exceeded 429: Too many requests")
        return [[1.0] * 4 for _ in batch]

    res = await resilient_batch_embed(texts, mock_embed, max_retries=3, base_backoff=0.01)
    assert attempts == 2
    assert res.success_count == 2
    assert res.failed_count == 0


@pytest.mark.asyncio
async def test_resilient_embed_auth_fail_fast():
    """Verify 401/Unauthorized raises UnrecoverableEmbeddingError immediately without wasteful bisection."""
    texts = ["t1", "t2", "t3"]
    attempts = 0

    async def mock_embed(batch: list[str]) -> list[list[float]]:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("401 Unauthorized: Invalid API key")

    with pytest.raises(UnrecoverableEmbeddingError) as exc_info:
        await resilient_batch_embed(texts, mock_embed)

    assert "authentication failed" in str(exc_info.value).lower()
    assert attempts == 1  # No wasteful recursive slicing on bad auth


@pytest.mark.asyncio
async def test_resilient_embed_empty_batch():
    """Verify empty batch returns 0 items safely."""
    res = await resilient_batch_embed([], AsyncMock())
    assert res.embeddings == []
    assert res.success_count == 0
    assert res.failed_count == 0

