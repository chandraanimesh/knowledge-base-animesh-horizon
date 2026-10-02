from abc import ABC, abstractmethod
from typing import Any


class SearchProvider(ABC):
    """
    Abstract interface for full-text search (BM25 / lexical / inverted index) providers.
    Describes the capability to execute text search queries against indexed document tokens.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute full-text lexical search against stored documents.
        """
        raise NotImplementedError
