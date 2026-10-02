"""
Provider architecture exceptions.
"""


class ProviderError(Exception):
    """Base exception for all provider-related errors."""
    pass


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered for a capability."""
    pass


class InvalidProviderConfigError(ProviderError):
    """Raised when provider configuration or initialization fails."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when an external provider service is unreachable or rate limited."""
    pass


class UnsupportedCapabilityError(ProviderError):
    """Raised when a provider does not support an optional capability."""
    pass
