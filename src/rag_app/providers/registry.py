import logging
import os
from typing import Any, Callable, Type

from rag_app.providers.exceptions import ProviderNotFoundError
from rag_app.providers import config

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Centralized registry and factory for all pluggable RAG infrastructure providers.
    Supports lazy instantiation and runtime resolution via configuration.
    """

    def __init__(self):
        self._factories: dict[str, dict[str, Callable[[], Any]]] = {}
        self._instances: dict[str, dict[str, Any]] = {}
        self._defaults: dict[str, str] = {}
        self._explicit_defaults: dict[str, str] = {}
        self._initialized_defaults = False

    def register(
        self,
        capability: str,
        name: str,
        factory: Callable[[], Any] | Type[Any],
        default: bool = False,
    ) -> None:
        """
        Register a provider factory for a capability.

        Args:
            capability: Capability category ('llm', 'embedding', 'document_reader', etc.)
            name: Unique name of this provider implementation (e.g. 'groq', 'sentence_transformer')
            factory: Callable (function or class) returning an instance implementing the capability interface.
            default: If True, marks this provider as the fallback default for this capability.
        """
        norm_cap = capability.strip().lower()
        norm_name = name.strip().lower()

        if norm_cap not in self._factories:
            self._factories[norm_cap] = {}
            self._instances[norm_cap] = {}

        self._factories[norm_cap][norm_name] = factory

        if default:
            self._explicit_defaults[norm_cap] = norm_name
            self._defaults[norm_cap] = norm_name
        elif norm_cap not in self._defaults:
            self._defaults[norm_cap] = norm_name

        logger.debug(
            "provider_registered",
            extra={"capability": norm_cap, "provider": norm_name, "is_default": default},
        )

    def _ensure_default_registrations(self) -> None:
        """
        Lazily registers standard built-in providers on first lookup.
        """
        if self._initialized_defaults:
            return

        self._initialized_defaults = True

        # 1. LLM
        def _groq_factory():
            from rag_app.providers.adapters.llm.groq import GroqLLMAdapter
            return GroqLLMAdapter()

        if "llm" not in self._factories or "groq" not in self._factories["llm"]:
            self.register("llm", "groq", _groq_factory)
            self._defaults["llm"] = "groq"

        # 2. Embedding
        def _st_factory():
            from rag_app.providers.adapters.embedding.sentence_transformer import SentenceTransformerEmbeddingAdapter
            return SentenceTransformerEmbeddingAdapter()

        if "embedding" not in self._factories or "sentence_transformer" not in self._factories["embedding"]:
            self.register("embedding", "sentence_transformer", _st_factory)
            self.register("embedding", "sentence_transformers", _st_factory)
            self._defaults["embedding"] = "sentence_transformer"

        # 3. Document Reader
        def _doc_reader_factory():
            from rag_app.providers.adapters.document_reader.standard import StandardDocumentReaderAdapter
            return StandardDocumentReaderAdapter()

        if "document_reader" not in self._factories or "standard" not in self._factories["document_reader"]:
            self.register("document_reader", "standard", _doc_reader_factory)
            self.register("document_reader", "default", _doc_reader_factory)
            self.register("document_reader", "docling", _doc_reader_factory)
            self._defaults["document_reader"] = "standard"

        # 4. Reranker
        def _reranker_factory():
            from rag_app.providers.adapters.reranker.cross_encoder import CrossEncoderRerankerAdapter
            return CrossEncoderRerankerAdapter()

        if "reranker" not in self._factories or "cross_encoder" not in self._factories["reranker"]:
            self.register("reranker", "cross_encoder", _reranker_factory)
            self.register("reranker", "ms_marco", _reranker_factory)
            self._defaults["reranker"] = "cross_encoder"

        # 5. OCR
        def _ocr_factory():
            from rag_app.providers.adapters.ocr.paddleocr import PaddleOCRAdapter
            return PaddleOCRAdapter()

        if "ocr" not in self._factories or "paddleocr" not in self._factories["ocr"]:
            self.register("ocr", "paddleocr", _ocr_factory)
            self.register("ocr", "paddle_ocr", _ocr_factory)
            self._defaults["ocr"] = "paddleocr"

        # 6. Web
        def _web_factory():
            from rag_app.providers.adapters.web.firecrawl import FirecrawlWebAdapter
            return FirecrawlWebAdapter()

        if "web" not in self._factories or "firecrawl" not in self._factories["web"]:
            self.register("web", "firecrawl", _web_factory)
            self._defaults["web"] = "firecrawl"

    def _get_configured_name(self, capability: str) -> str:
        """
        Retrieves the provider name specified by environment or settings for the capability.
        """
        cap = capability.strip().lower()

        # Check environment/settings first
        configured = ""
        if cap == "llm":
            configured = config.get_llm_provider_name()
        elif cap == "embedding":
            configured = config.get_embedding_provider_name()
        elif cap == "document_reader":
            configured = config.get_document_reader_provider_name()
        elif cap == "reranker":
            configured = config.get_reranker_provider_name()
        elif cap == "ocr":
            configured = config.get_ocr_provider_name()
        elif cap == "web":
            configured = config.get_web_provider_name()
        elif cap == "vector_store":
            configured = config.get_vector_store_provider_name()
        elif cap == "search":
            configured = config.get_search_provider_name()

        # If an explicit default was programmatically registered, use it if no env var was set
        if cap in self._explicit_defaults and not os.getenv(f"{cap.upper()}_PROVIDER"):
            return self._explicit_defaults[cap]

        if configured:
            return configured

        return self._defaults.get(cap, "")

    def get(self, capability: str, name: str | None = None) -> Any:
        """
        Resolves and returns a provider instance for the given capability.

        Args:
            capability: Capability category ('llm', 'embedding', 'document_reader', etc.)
            name: Optional provider name. If None, resolves from centralized configuration.

        Returns:
            An instance implementing the capability interface.

        Raises:
            ProviderNotFoundError: If no provider matching the requested name is registered.
        """
        self._ensure_default_registrations()
        norm_cap = capability.strip().lower()

        target_name = (name or self._get_configured_name(norm_cap)).strip().lower()

        if norm_cap not in self._factories or target_name not in self._factories[norm_cap]:
            available = self.list_providers(norm_cap)
            raise ProviderNotFoundError(
                f"Unsupported or unregistered provider '{target_name}' for capability '{norm_cap}'. "
                f"Available providers: {available}"
            )

        # Cache singleton instance
        if target_name not in self._instances[norm_cap]:
            factory = self._factories[norm_cap][target_name]
            self._instances[norm_cap][target_name] = factory()

        return self._instances[norm_cap][target_name]

    def list_providers(self, capability: str) -> list[str]:
        """
        List all registered provider names for a capability.
        """
        self._ensure_default_registrations()
        norm_cap = capability.strip().lower()
        return list(self._factories.get(norm_cap, {}).keys())

    def clear_cache(self) -> None:
        """
        Clear all cached provider singleton instances.
        """
        for cap in self._instances:
            self._instances[cap].clear()


# Global Singleton Registry
provider_registry = ProviderRegistry()


def get_provider(capability: str, name: str | None = None) -> Any:
    """Convenience access function for provider_registry.get"""
    return provider_registry.get(capability=capability, name=name)
