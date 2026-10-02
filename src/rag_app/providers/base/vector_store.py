from abc import ABC, abstractmethod
from typing import Any


class VectorStoreProvider(ABC):
    """
    Abstract interface for vector database storage and similarity search providers.
    Describes the capability to index and query dense embeddings.
    """

    @abstractmethod
    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute cosine/inner-product similarity search against stored vectors.
        """
        raise NotImplementedError
