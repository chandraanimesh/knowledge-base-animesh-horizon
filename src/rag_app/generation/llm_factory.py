from functools import lru_cache
from rag_app.providers.base.llm import LLMProvider
from rag_app.providers.adapters.llm.groq import GroqLLMAdapter

# Backwards-compatibility aliases
ChatProvider = LLMProvider
LangChainGroqChatProvider = GroqLLMAdapter


@lru_cache
def get_llm() -> LLMProvider:
    """
    Compatibility convenience function that delegates to the centralized provider registry.
    """
    from rag_app.providers.registry import get_provider
    return get_provider("llm")
