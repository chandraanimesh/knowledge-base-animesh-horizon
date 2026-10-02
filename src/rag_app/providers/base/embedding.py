from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Abstract interface for text embedding providers.
    Describes the capability to compute dense vector embeddings for documents and queries.
    """

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Compute embeddings for a batch of text strings.
        """
        raise NotImplementedError

    async def embed_query(self, query: str) -> list[float]:
        """
        Compute embedding for a single query string.
        Default implementation embeds query as a batch of 1.
        """
        embeddings = await self.embed_texts([query])
        return embeddings[0]
