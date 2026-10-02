from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """
    Abstract interface for Large Language Model generation providers.
    Describes the capability to generate text from structured messages.
    """

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
