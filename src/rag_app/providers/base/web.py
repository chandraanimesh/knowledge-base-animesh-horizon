from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class WebSearchResult(BaseModel):
    url: str
    title: str | None = None
    snippet: str | None = None


class WebSearchResponse(BaseModel):
    results: list[WebSearchResult] = Field(default_factory=list)


class WebPageResult(BaseModel):
    url: str
    title: str | None = None
    markdown: str
    metadata: dict[str, Any] | None = None


class WebCrawlResult(BaseModel):
    pages: list[WebPageResult] = Field(default_factory=list)


class WebProvider(ABC):
    """
    Abstract interface for web acquisition and research providers (crawlers, search engines, scrapers).
    """

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        """
        Execute a web search query.
        """
        raise NotImplementedError

    @abstractmethod
    async def scrape(self, url: str) -> WebPageResult:
        """
        Extract page content as markdown.
        """
        raise NotImplementedError

    @abstractmethod
    async def crawl(self, url: str, max_pages: int = 5) -> WebCrawlResult:
        """
        Crawl a URL path or domain.
        """
        raise NotImplementedError
