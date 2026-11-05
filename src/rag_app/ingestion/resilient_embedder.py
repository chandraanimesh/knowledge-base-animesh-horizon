"""Resilient Batch Embedding & Toxic Chunk Recovery

Provides recursive binary-search bisection to isolate poison/toxic chunks in an embedding batch,
rescuing all valid chunks while recording structured provenance for failed chunks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Coroutine, Any

logger = logging.getLogger(__name__)


class EmbeddingErrorClassification:
    TOXIC_INPUT = "TOXIC_INPUT"             # Malformed string, encoding, or poison token
    TOKEN_LIMIT = "TOKEN_LIMIT"             # Exceeds maximum token context
    RATE_LIMIT = "RATE_LIMIT"               # Provider 429 rate limit
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"  # Connection error, 502/503/504
    AUTH_ERROR = "AUTH_ERROR"               # Invalid credentials / 401 / 403
    UNKNOWN = "UNKNOWN"                     # Unclassified error


class UnrecoverableEmbeddingError(Exception):
    """Raised when the error cannot be resolved by bisection (e.g. invalid auth)."""
    pass


@dataclass
class FailedChunkInfo:
    chunk_index: int
    text_preview: str
    error_classification: str
    error_message: str


@dataclass
class ResilientBatchResult:
    embeddings: list[list[float] | None]
    failed_chunks: list[FailedChunkInfo] = field(default_factory=list)
    success_count: int = 0
    failed_count: int = 0


def classify_embedding_exception(exc: Exception) -> str:
    """Classifies an embedding exception to decide whether to bisect or fail fast."""
    msg = str(exc).lower()
    
    # 1. Authentication errors — fail fast
    if any(k in msg for k in ("unauthorized", "401", "forbidden", "403", "invalid api key", "authentication")):
        return EmbeddingErrorClassification.AUTH_ERROR
        
    # 2. Rate limits — retryable with backoff
    if any(k in msg for k in ("rate limit", "429", "too many requests", "quota exceeded")):
        return EmbeddingErrorClassification.RATE_LIMIT
        
    # 3. Provider availability — transient
    if any(k in msg for k in ("connection refused", "timeout", "timed out", "502", "503", "504", "unavailable", "server error")):
        return EmbeddingErrorClassification.PROVIDER_UNAVAILABLE
        
    # 4. Toxic / Malformed / Input
    if any(k in msg for k in ("toxic", "poison", "malformed", "null byte", "utf-8", "codec", "decode", "dimension", "shape", "tensor", "index out of range")):
        return EmbeddingErrorClassification.TOXIC_INPUT

    # 5. Token limit / oversized
    if any(k in msg for k in ("token limit", "maximum context length", "sequence length", "exceeds maximum", "too many tokens", "input is too long")):
        return EmbeddingErrorClassification.TOKEN_LIMIT
        
    # Default to TOXIC_INPUT if error is during batch processing of data
    return EmbeddingErrorClassification.TOXIC_INPUT


async def resilient_batch_embed(
    texts: list[str],
    embed_fn: Callable[[list[str]], Coroutine[Any, Any, list[list[float]]]],
    max_retries: int = 3,
    base_backoff: float = 0.5,
) -> ResilientBatchResult:
    """
    Attempts to embed a batch of texts.
    If the batch succeeds, returns all embeddings directly.
    If a permanent or input-related error occurs, recursively bisects the batch to isolate
    poison chunks and rescues all valid chunks.
    Preserves original text ordering and index identity.
    """
    if not texts:
        return ResilientBatchResult(embeddings=[], failed_chunks=[], success_count=0, failed_count=0)

    n = len(texts)
    embeddings: list[list[float] | None] = [None] * n
    failed_chunks: list[FailedChunkInfo] = []

    # Indexed pairs to preserve global indices during recursive bisection
    indexed_batch = list(enumerate(texts))

    async def _embed_slice(slice_items: list[tuple[int, str]]) -> None:
        if not slice_items:
            return

        slice_texts = [text for _, text in slice_items]
        
        # Try with bounded retries for transient rate limits or provider hiccups
        for attempt in range(max_retries):
            try:
                res = await embed_fn(slice_texts)
                for (orig_idx, _), emb in zip(slice_items, res, strict=True):
                    embeddings[orig_idx] = emb
                return
            except Exception as exc:
                classification = classify_embedding_exception(exc)
                
                # Unrecoverable auth error — immediately fail fast
                if classification == EmbeddingErrorClassification.AUTH_ERROR:
                    logger.critical(f"Unrecoverable authentication error in embedding provider: {exc}")
                    raise UnrecoverableEmbeddingError(f"Embedding authentication failed: {exc}") from exc
                
                # Rate limit — apply exponential backoff
                if classification == EmbeddingErrorClassification.RATE_LIMIT and attempt < max_retries - 1:
                    sleep_time = base_backoff * (2 ** attempt)
                    logger.warning(f"Embedding rate limited. Backing off for {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(sleep_time)
                    continue

                # Provider unavailable — retry if attempts remain
                if classification == EmbeddingErrorClassification.PROVIDER_UNAVAILABLE and attempt < max_retries - 1:
                    sleep_time = base_backoff * (2 ** attempt)
                    logger.warning(f"Embedding provider unavailable. Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                    continue

                # If single chunk slice failed, we've isolated the poison chunk!
                if len(slice_items) == 1:
                    orig_idx, bad_text = slice_items[0]
                    preview = bad_text[:80] + "..." if len(bad_text) > 80 else bad_text
                    logger.error(
                        f"Isolated poison chunk at index {orig_idx} ({classification}): {exc}. "
                        f"Dropping chunk and rescuing remaining valid chunks."
                    )
                    failed_chunks.append(
                        FailedChunkInfo(
                            chunk_index=orig_idx,
                            text_preview=preview,
                            error_classification=classification,
                            error_message=str(exc),
                        )
                    )
                    embeddings[orig_idx] = None
                    return

                # Otherwise, bisect slice recursively
                mid = len(slice_items) // 2
                left_slice = slice_items[:mid]
                right_slice = slice_items[mid:]
                logger.info(
                    f"Embedding batch error on slice size {len(slice_items)} ({classification}). "
                    f"Bisecting into left={len(left_slice)} and right={len(right_slice)}..."
                )
                await _embed_slice(left_slice)
                await _embed_slice(right_slice)
                return

    await _embed_slice(indexed_batch)

    successes = sum(1 for e in embeddings if e is not None)
    failures = len(failed_chunks)

    return ResilientBatchResult(
        embeddings=embeddings,
        failed_chunks=failed_chunks,
        success_count=successes,
        failed_count=failures,
    )
