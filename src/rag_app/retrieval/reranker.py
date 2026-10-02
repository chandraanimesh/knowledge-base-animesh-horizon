import asyncio
import time
import logging
from sentence_transformers import CrossEncoder
from functools import lru_cache
from rag_app.core.tracing import track
from rag_app.providers.base.reranker import RerankerProvider

logger = logging.getLogger(__name__)


class AsyncCrossEncoderProvider(RerankerProvider):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        t0 = time.perf_counter()
        # Load the model once into memory during app startup
        self.model = CrossEncoder(model_name)
        initialization_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "reranker_initialized",
            extra={
                "event": "reranker_initialized",
                "model": model_name,
                "initialization_ms": initialization_ms,
            },
        )

    @track(name="rerank_documents", phase="retrieval")
    async def rerank(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        if not texts:
            return []
        # Prepare [Query, Document] pairs required by Cross-Encoders
        pairs = [[query, text] for text in texts]

        # OFFLOAD to a separate thread so we don't block the async event loop
        scores = await asyncio.to_thread(self.model.predict, pairs)

        return scores.tolist()


@lru_cache(maxsize=1)
def get_reranker_provider() -> RerankerProvider:
    from rag_app.providers.registry import get_provider
    return get_provider("reranker")