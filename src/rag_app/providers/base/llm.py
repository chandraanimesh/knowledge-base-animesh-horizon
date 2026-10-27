from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """
    Abstract interface for Large Language Model generation providers.
    Describes the capability to generate text from structured messages,
    count tokens, and produce structured outputs.
    """

    provider_name: str = "base"
    model: str | None = None

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        stage: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """
        Generate text completion from input messages.

        Returns:
            tuple[str, dict]: (generated_text, usage_metadata)
        """
        raise NotImplementedError

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for structured input messages."""
        from rag_app.core.tokenizer import count_messages
        return count_messages(messages, model=getattr(self, "model", None))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """
        Returns a runnable or model configured to return structured output matching the schema.
        """
        if hasattr(self, "llm") and hasattr(self.llm, "with_structured_output") and self.llm is not self:
            return self.llm.with_structured_output(schema, **kwargs)
        raise NotImplementedError(
            f"Provider '{self.provider_name}' does not implement structured output for schema {schema}."
        )
