from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, HttpUrl, Field

# --- Exceptions ---

class WebProviderError(Exception):
    """Base exception for all WebProvider-related errors."""
    pass

class ProviderUnavailableError(WebProviderError):
    """Raised when the underlying provider is unreachable or times out."""
    pass

class ProviderRequestError(WebProviderError):
    """Raised when the provider rejects the request (e.g., bad auth, rate limit, invalid url)."""
    pass

class UnsupportedCapabilityError(WebProviderError):
    """Raised when the chosen provider does not support the requested operation."""
    pass

# --- Normalized Result Models ---

class WebSearchResult(BaseModel):
    """A single search result from a web search operation."""
    url: str = Field(..., description="The URL of the search result")
    title: str | None = Field(None, description="The title of the page")
    snippet: str | None = Field(None, description="A brief snippet or summary from the search result")

class WebSearchResponse(BaseModel):
    """Normalized response for a web search capability."""
    results: list[WebSearchResult] = Field(default_factory=list)

class WebPageResult(BaseModel):
    """Normalized result of scraping a single web page."""
    url: str = Field(..., description="The source URL of the page")
    title: str | None = Field(None, description="The extracted title of the page")
    markdown: str = Field(..., description="The main content of the page converted to markdown")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata extracted (e.g., author, date)")

class WebCrawlResult(BaseModel):
    """Normalized result of crawling a domain or URL path."""
    pages: list[WebPageResult] = Field(default_factory=list)

# --- Abstract Provider Interface ---

class WebProvider(ABC):
    """
    Abstract interface for web acquisition providers.
    The Research Engine depends on this contract, not on specific SDKs.
    """
    
    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        """
        Execute a web search query.
        Raises UnsupportedCapabilityError if the provider cannot search.
        """
        pass

    @abstractmethod
    async def scrape(self, url: str) -> WebPageResult:
        """
        Scrape a single URL and return normalized markdown content.
        Raises UnsupportedCapabilityError if the provider cannot scrape.
        """
        pass

    @abstractmethod
    async def crawl(self, url: str, max_pages: int = 10) -> WebCrawlResult:
        """
        Crawl a URL and its sub-paths.
        Raises UnsupportedCapabilityError if the provider cannot crawl.
        """
        pass
