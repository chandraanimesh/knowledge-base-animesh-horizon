from rag_app.providers.base.reranker import RerankerProvider


class CrossEncoderRerankerAdapter(RerankerProvider):
    """
    Adapter wrapping AsyncCrossEncoderProvider.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from rag_app.retrieval.reranker import AsyncCrossEncoderProvider
        self._provider = AsyncCrossEncoderProvider(model_name=model_name)

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        return await self._provider.rerank(query=query, texts=texts)
