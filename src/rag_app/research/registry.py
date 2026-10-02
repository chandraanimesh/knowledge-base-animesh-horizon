from typing import Dict
from rag_app.research.provider import WebProvider, WebProviderError

class UnknownProviderError(WebProviderError):
    """Raised when a requested provider is not found in the registry."""
    pass

class DuplicateProviderError(WebProviderError):
    """Raised when attempting to register a provider name that already exists."""
    pass

class ProviderRegistry:
    """
    A registry for WebProvider implementations.
    Allows runtime registration and lookup of providers without hardcoding
    dependencies on specific implementations (like Firecrawl).
    """
    def __init__(self):
        self._providers: Dict[str, WebProvider] = {}

    def register(self, name: str, provider: WebProvider) -> None:
        """
        Registers a WebProvider under the given name.
        Rejects duplicates to ensure deterministic behavior.
        """
        if name in self._providers:
            raise DuplicateProviderError(f"Provider '{name}' is already registered.")
        self._providers[name] = provider

    def get(self, name: str) -> WebProvider:
        """
        Retrieves a WebProvider by name.
        Raises UnknownProviderError if not found.
        """
        provider = self._providers.get(name)
        if not provider:
            raise UnknownProviderError(f"Provider '{name}' is not registered.")
        return provider

# Global instance following the repository's singleton/factory pattern
_global_registry = ProviderRegistry()

def get_provider_registry() -> ProviderRegistry:
    """
    Factory function to retrieve the global provider registry.
    Can be easily overridden in tests or used via FastAPI Depends.
    """
    return _global_registry
