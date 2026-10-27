from typing import Any
from rag_app.providers.base.web import WebProvider, WebSearchResponse, WebPageResult, WebCrawlResult


class TinyFishWebAdapter(WebProvider):
    """
    Adapter wrapping TinyFishProvider for pluggable web research.
    Conforms to the internal WebProvider interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        search_url: str | None = None,
        fetch_url: str | None = None,
        timeout_seconds: float = 15.0,
    ):
        self._api_key = api_key
        self._search_url = search_url
        self._fetch_url = fetch_url
        self._timeout_seconds = timeout_seconds
        self._provider = None

    def _get_provider(self):
        if self._provider is None:
            from rag_app.research.providers.tinyfish import TinyFishProvider
            self._provider = TinyFishProvider(
                api_key=self._api_key,
                search_url=self._search_url,
                fetch_url=self._fetch_url,
                timeout_seconds=self._timeout_seconds,
            )
        return self._provider

    async def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        provider = self._get_provider()
        res = await provider.search(query=query, limit=limit)
        return WebSearchResponse(results=[r.model_dump() for r in res.results])

    async def scrape(self, url: str) -> WebPageResult:
        provider = self._get_provider()
        res = await provider.scrape(url=url)
        return WebPageResult(**res.model_dump())

    async def crawl(self, url: str, max_pages: int = 5) -> WebCrawlResult:
        provider = self._get_provider()
        res = await provider.crawl(url=url, max_pages=max_pages)
        return WebCrawlResult(pages=[p.model_dump() for p in res.pages])
