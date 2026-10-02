from typing import Any
from rag_app.providers.base.web import WebProvider, WebSearchResponse, WebPageResult, WebCrawlResult


class FirecrawlWebAdapter(WebProvider):
    """
    Adapter wrapping FirecrawlProvider with lazy initialization.
    """

    def __init__(self, api_key: str | None = None, api_url: str | None = None):
        self._api_key = api_key
        self._api_url = api_url
        self._provider = None

    def _get_provider(self):
        if self._provider is None:
            from rag_app.research.providers.firecrawl import FirecrawlProvider
            self._provider = FirecrawlProvider(api_key=self._api_key, api_url=self._api_url)
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

