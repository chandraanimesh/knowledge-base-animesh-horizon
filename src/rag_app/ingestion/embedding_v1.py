import asyncio
from functools import lru_cache

from rag_app.core.tracing import track
from rag_app.providers.base.embedding import EmbeddingProvider as BaseEmbeddingProvider

class EmbeddingProvider(BaseEmbeddingProvider):
    """Base embedding provider interface for backwards compatibility."""
    pass


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Free local embedding provider.

    Model:
        sentence-transformers/all-MiniLM-L6-v2

    Output:
        384-dimensional vector
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        import time
        import logging
        logger = logging.getLogger(__name__)

        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        initialization_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "embedding_model_initialized",
            extra={
                "event": "embedding_model_initialized",
                "model": model_name,
                "initialization_ms": initialization_ms,
            },
        )

    @track(name="embed_texts", phase="ingestion")
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.astype("float32").tolist()


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    from rag_app.providers.registry import get_provider
    return get_provider("embedding")

