from abc import ABC, abstractmethod


class RerankerProvider(ABC):
    """
    Abstract interface for cross-encoder/reranker providers.
    Describes the capability to score relevance between a query and a set of candidate texts.
    """

    @abstractmethod
    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        """
        Score and rank candidate texts against query.
        Returns a list of float relevance scores matching the input texts.
        """
        raise NotImplementedError
