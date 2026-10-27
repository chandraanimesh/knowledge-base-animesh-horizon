"""
Deterministic Mock LLM Provider Adapter.
Provides deterministic, repeatable completions, token accounting, and structured outputs
for testing without live external API dependencies or network credentials.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel

from rag_app.core.observability import record_llm_call
from rag_app.providers.base.llm import LLMProvider

logger = logging.getLogger(__name__)


class MockStructuredRunnable:
    """
    Lightweight runnable simulating LangChain's structured output wrapper for tests.
    """

    def __init__(self, schema: Type[BaseModel] | Any, default_instance: Any = None):
        self.schema = schema
        self.default_instance = default_instance

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        if self.default_instance is not None:
            return self.default_instance

        # If schema is a Pydantic model class, instantiate default minimal instance
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            try:
                # Attempt default initialization
                return self.schema.model_validate({})
            except Exception:
                # If required fields exist, create mock fields
                fields = getattr(self.schema, "model_fields", {})
                from enum import Enum
                mock_data = {}
                for f_name, f_info in fields.items():
                    if f_info.is_required():
                        annotation = f_info.annotation
                        if isinstance(annotation, type) and issubclass(annotation, Enum):
                            mock_data[f_name] = list(annotation)[0]
                        elif annotation is str:
                            mock_data[f_name] = f"mock_{f_name}"
                        elif annotation is int:
                            mock_data[f_name] = 1
                        elif annotation is bool:
                            mock_data[f_name] = True
                        elif annotation is list or getattr(annotation, "__origin__", None) is list:
                            mock_data[f_name] = []
                        elif annotation is dict or getattr(annotation, "__origin__", None) is dict:
                            mock_data[f_name] = {}
                        else:
                            mock_data[f_name] = None
                return self.schema.model_validate(mock_data)

        # Fallback to empty dict or raw schema default
        return {}

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        import asyncio
        return asyncio.run(self.ainvoke(messages, **kwargs))


class MockLLMAdapter(LLMProvider):
    """
    Deterministic mock implementation of LLMProvider.
    Conforms strictly to LLMProvider contract for testability and provider-swap verification.
    """

    provider_name: str = "mock"
    model: str = "mock-deterministic-v1"

    def __init__(
        self,
        default_response: Optional[str] = None,
        canned_responses: Optional[dict[str, str]] = None,
        default_usage: Optional[dict] = None,
        error_trigger: Optional[Exception] = None,
        structured_output_factory: Optional[Callable[[Any], Any]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model = model or "mock-deterministic-v1"
        self.default_response = default_response or "Mock deterministic response grounded in evidence [source: doc_1]."
        self.canned_responses = canned_responses or {}
        self.default_usage = default_usage or {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        }
        self.error_trigger = error_trigger
        self.structured_output_factory = structured_output_factory
        self.extra_kwargs = kwargs
        self.invocation_count = 0
        self.last_messages: list[dict[str, str]] = []

    @property
    def llm(self) -> Any:
        """Compatibility property for consumers referencing provider.llm."""
        return self

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Deterministic token count approximation (4 chars/token)."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return max(1, total_chars // 4)

    async def generate(
        self,
        messages: list[dict[str, str]],
        stage: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """
        Generates deterministic text completion without calling external networks.
        """
        self.invocation_count += 1
        self.last_messages = messages

        if self.error_trigger is not None:
            raise self.error_trigger

        start = time.perf_counter()

        # Check canned responses by keyword match in messages
        combined_text = " ".join(str(m.get("content", "")) for m in messages).lower()
        selected_text = self.default_response

        for kw, resp in self.canned_responses.items():
            if kw.lower() in combined_text:
                selected_text = resp
                break

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        usage = dict(self.default_usage)

        record_llm_call(
            stage_name=stage,
            usage=usage,
            latency_ms=latency_ms,
            model_name=self.model,
            estimated_prompt_tokens=self.count_tokens(messages),
        )

        return selected_text, usage

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """
        Returns a mock structured runnable matching the requested schema.
        """
        if self.structured_output_factory:
            return self.structured_output_factory(schema)
        return MockStructuredRunnable(schema)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """Direct ainvoke support simulating LangChain chat model response."""
        content, usage = await self.generate(messages if isinstance(messages, list) else [{"content": str(messages)}])
        
        class MockAIMessage:
            def __init__(self, c: str, u: dict):
                self.content = c
                self.usage_metadata = u
        return MockAIMessage(content, usage)
