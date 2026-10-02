from rag_app.core.config import get_settings
from rag_app.providers.base.embedding import EmbeddingProvider


class SentenceTransformerEmbeddingAdapter(EmbeddingProvider):
    """
    Adapter wrapping SentenceTransformerEmbeddingProvider.
    """

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model

        from rag_app.ingestion.embedding_v1 import SentenceTransformerEmbeddingProvider
        self._provider = SentenceTransformerEmbeddingProvider(
            model_name=self.model_name,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.embed_texts(texts)

    async def embed_query(self, query: str) -> list[float]:
        return await self._provider.embed_query(query)
