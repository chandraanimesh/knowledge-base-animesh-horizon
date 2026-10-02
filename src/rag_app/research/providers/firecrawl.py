import logging
from typing import Any
import asyncio

from rag_app.core.config import get_settings
from rag_app.research.provider import (
    WebProvider,
    WebSearchResponse,
    WebSearchResult,
    WebPageResult,
    WebCrawlResult,
    ProviderUnavailableError,
    ProviderRequestError,
    UnsupportedCapabilityError,
)

logger = logging.getLogger(__name__)


class FirecrawlProvider(WebProvider):
    """
    Firecrawl implementation of the WebProvider interface.
    Normalizes Firecrawl SDK responses into our domain contracts.
    """

    def __init__(self, api_key: str | None = None, api_url: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.firecrawl_api_key
        self.api_url = api_url or settings.firecrawl_base_url

        if not self.api_key:
            raise ProviderRequestError("FIRECRAWL_API_KEY is not configured.")

        try:
            from firecrawl import AsyncFirecrawlApp
            self.app = AsyncFirecrawlApp(api_key=self.api_key, api_url=self.api_url or "https://api.firecrawl.dev")
        except ImportError:
            raise ProviderUnavailableError("firecrawl-py SDK is not installed.")
        except Exception as e:
            logger.error("Failed to initialize Firecrawl SDK", exc_info=True)
            raise ProviderUnavailableError(f"Failed to initialize Firecrawl: {str(e)}")

    def _map_exception(self, e: Exception) -> Exception:
        """Map SDK/HTTP exceptions to provider-independent exceptions."""
        error_msg = str(e).lower()
        if "timeout" in error_msg or "unavailable" in error_msg or "connection" in error_msg:
            return ProviderUnavailableError(f"Firecrawl service unavailable: {str(e)}")
        if "unauthorized" in error_msg or "rate limit" in error_msg or "invalid" in error_msg:
            return ProviderRequestError(f"Firecrawl request rejected: {str(e)}")
        return ProviderRequestError(f"Firecrawl operation failed: {str(e)}")

    async def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        try:
            logger.info("Executing Firecrawl search", extra={"query": query, "limit": limit})
            # Wait for search response.
            response = await self.app.search(query, params={"limit": limit})
            
            data = response.get("data", []) if isinstance(response, dict) else response
            
            results = []
            for item in data:
                if not isinstance(item, dict):
                    if hasattr(item, 'model_dump'):
                        item = item.model_dump()
                    elif hasattr(item, 'dict'):
                        item = item.dict()
                    else:
                        item = {"url": getattr(item, "url", None), "title": getattr(item, "title", None), "description": getattr(item, "description", None)}

                url = item.get("url")
                if not url:
                    continue

                results.append(
                    WebSearchResult(
                        url=url,
                        title=item.get("title"),
                        snippet=item.get("description") or item.get("snippet")
                    )
                )

            return WebSearchResponse(results=results)
        except Exception as e:
            raise self._map_exception(e)

    async def scrape(self, url: str) -> WebPageResult:
        try:
            logger.info("Executing Firecrawl scrape", extra={"url": url})
            # Attempt to scrape directly to markdown format
            response = await self.app.scrape_url(url, params={'formats': ['markdown']})

            data = response.get("data", {}) if isinstance(response, dict) and "data" in response else response
            
            if not isinstance(data, dict):
                if hasattr(data, 'model_dump'):
                    data = data.model_dump()
                elif hasattr(data, 'dict'):
                    data = data.dict()
                else:
                    data = {"markdown": getattr(data, "markdown", ""), "metadata": getattr(data, "metadata", {})}

            markdown = data.get("markdown") or ""
            metadata = data.get("metadata", {})
            title = metadata.get("title")

            return WebPageResult(
                url=url,
                title=title,
                markdown=markdown,
                metadata=metadata
            )
        except Exception as e:
            raise self._map_exception(e)

    async def crawl(self, url: str, max_pages: int = 10) -> WebCrawlResult:
        try:
            logger.info("Executing Firecrawl crawl", extra={"url": url, "max_pages": max_pages})
            
            # v2 provides async_crawl_url with poll_interval natively for awaiting completion
            response = await self.app.async_crawl_url(
                url, 
                params={"limit": max_pages, "scrapeOptions": {"formats": ["markdown"]}},
                wait_until_done=True,
                poll_interval=2
            )

            data = response.get("data", []) if isinstance(response, dict) else response

            pages = []
            for item in data:
                if not isinstance(item, dict):
                    if hasattr(item, 'model_dump'):
                        item = item.model_dump()
                    elif hasattr(item, 'dict'):
                        item = item.dict()
                    else:
                        item = {"url": getattr(item, "url", None), "markdown": getattr(item, "markdown", ""), "metadata": getattr(item, "metadata", {})}

                page_url = item.get("url") or url
                markdown = item.get("markdown") or ""
                metadata = item.get("metadata", {})
                title = metadata.get("title")

                pages.append(
                    WebPageResult(
                        url=page_url,
                        title=title,
                        markdown=markdown,
                        metadata=metadata
                    )
                )

            return WebCrawlResult(pages=pages)
        except Exception as e:
            raise self._map_exception(e)
