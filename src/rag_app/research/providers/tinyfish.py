import logging
from typing import Any
import httpx

import os
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


class TinyFishProvider(WebProvider):
    """
    TinyFish implementation of the WebProvider interface.
    Uses TinyFish Search API for live web search and Fetch API for Markdown extraction.
    """

    def __init__(
        self,
        api_key: str | None = None,
        search_url: str | None = None,
        fetch_url: str | None = None,
        timeout_seconds: float = 15.0,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.tinyfish_api_key
        self.search_url = search_url or getattr(settings, "tinyfish_search_url", "https://api.search.tinyfish.ai")
        self.fetch_url = fetch_url or getattr(settings, "tinyfish_fetch_url", "https://api.fetch.tinyfish.ai")
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ProviderRequestError("TINYFISH_API_KEY is not configured.")

    def _map_http_error(self, e: Exception, operation: str) -> Exception:
        """Maps HTTP/network exceptions to provider-independent exceptions without leaking keys."""
        if isinstance(e, (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout)):
            return ProviderUnavailableError(f"TinyFish {operation} timed out after {self.timeout_seconds}s")
        if isinstance(e, httpx.ConnectError):
            return ProviderUnavailableError(f"TinyFish {operation} connection failed: server unreachable")
        if isinstance(e, httpx.HTTPStatusError):
            status = e.response.status_code
            if status in (401, 403):
                return ProviderRequestError(f"TinyFish {operation} authentication failed: invalid API key")
            if status == 429:
                return ProviderRequestError(f"TinyFish {operation} rate limit exceeded")
            if status >= 500:
                return ProviderUnavailableError(f"TinyFish {operation} server error (HTTP {status})")
            return ProviderRequestError(f"TinyFish {operation} failed with status HTTP {status}")
        return ProviderRequestError(f"TinyFish {operation} failed: {type(e).__name__}")

    async def search(self, query: str, limit: int = 5) -> WebSearchResponse:
        """
        Execute a web search query via TinyFish Search API.
        """
        if not query or not query.strip():
            return WebSearchResponse(results=[])

        headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        }
        # Note: According to the official TinyFish API reference (https://docs.tinyfish.ai/search-api/reference),
        # the search endpoint accepts 'query', 'purpose', 'page', etc., but does NOT support a 'limit' parameter.
        # Upstream returns full-page result sets. We enforce the requested result cap client-side.
        params = {
            "query": query.strip(),
            "purpose": f"RAG web research for query: {query.strip()[:200]}",
        }

        logger.info(
            "executing_tinyfish_search",
            extra={"event": "executing_tinyfish_search", "query": query, "limit": limit},
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(self.search_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            mapped_err = self._map_http_error(exc, "search")
            logger.error(
                "tinyfish_search_error",
                extra={"event": "tinyfish_search_error", "error": str(mapped_err)},
            )
            raise mapped_err

        results: list[WebSearchResult] = []
        raw_results = data.get("results", []) if isinstance(data, dict) else []
        raw_upstream_count = len(raw_results)
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not url:
                continue
            results.append(
                WebSearchResult(
                    url=str(url),
                    title=item.get("title") or item.get("site_name"),
                    snippet=item.get("snippet"),
                )
            )

        # Enforce application-side maximum result count even when upstream returns more results
        results = results[:max(1, limit)]
        returned_count = len(results)

        logger.info(
            "tinyfish_search_completed",
            extra={
                "event": "tinyfish_search_completed",
                "raw_upstream_count": raw_upstream_count,
                "returned_count": returned_count,
            },
        )
        return WebSearchResponse(results=results)

    async def scrape(self, url: str) -> WebPageResult:
        """
        Scrape a single URL and return normalized Markdown content via TinyFish Fetch API.
        """
        if not url or not url.strip():
            raise ProviderRequestError("URL must not be empty.")

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "urls": [url.strip()],
            "format": "markdown",
            "purpose": f"RAG evidence scrape for: {url.strip()[:200]}",
        }

        logger.info(
            "executing_tinyfish_fetch",
            extra={"event": "executing_tinyfish_fetch", "url": url},
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self.fetch_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            mapped_err = self._map_http_error(exc, "scrape")
            logger.error(
                "tinyfish_fetch_error",
                extra={"event": "tinyfish_fetch_error", "url": url, "error": str(mapped_err)},
            )
            raise mapped_err

        raw_results = data.get("results", []) if isinstance(data, dict) else []
        raw_errors = data.get("errors", []) if isinstance(data, dict) else []

        if not raw_results and raw_errors:
            err_msg = str(raw_errors[0]) if raw_errors else "Fetch returned errors"
            raise ProviderRequestError(f"TinyFish scrape failed for {url}: {err_msg}")

        if raw_results and isinstance(raw_results[0], dict):
            page_data = raw_results[0]
            markdown_content = page_data.get("text") or ""
            title = page_data.get("title")
            metadata: dict[str, Any] = {
                k: v
                for k, v in page_data.items()
                if k not in ("text", "url", "title") and v is not None
            }
            return WebPageResult(
                url=page_data.get("url") or url,
                title=title,
                markdown=markdown_content,
                metadata=metadata if metadata else None,
            )

        return WebPageResult(
            url=url,
            title=None,
            markdown="",
            metadata=None,
        )

    async def crawl(self, url: str, max_pages: int = 10) -> WebCrawlResult:
        """
        Crawl a URL. TinyFish Fetch is single/batch URL-based rather than recursive domain crawler;
        we fetch the seed URL and return it as a single-page crawl result.
        """
        page = await self.scrape(url)
        return WebCrawlResult(pages=[page] if page.markdown else [])
