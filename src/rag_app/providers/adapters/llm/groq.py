import time
from typing import Any
from rag_app.core.config import get_settings
from rag_app.core.observability import record_llm_call
from rag_app.providers.base.llm import LLMProvider
from rag_app.providers.exceptions import InvalidProviderConfigError


class GroqLLMAdapter(LLMProvider):
    """
    Concrete implementation of LLMProvider for Groq Cloud.
    Wraps ChatGroq for text generation and records observability metrics.
    """

    provider_name: str = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model = model or settings.groq_model
        self.temperature = temperature
        self.extra_kwargs = kwargs

        if not self.api_key or not str(self.api_key).strip():
            raise InvalidProviderConfigError(
                "GROQ_API_KEY is required to initialize GroqLLMAdapter."
            )

        from langchain_groq import ChatGroq

        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            **self.extra_kwargs,
        )

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        from rag_app.core.tokenizer import count_messages
        return count_messages(messages, model=self.model)

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
        start = time.perf_counter()
        response = await self.llm.ainvoke(messages, **kwargs)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        usage = response.usage_metadata or {}
        estimated_input = None
        try:
            estimated_input = self.count_tokens(messages)
        except Exception:
            estimated_input = None

        record_llm_call(
            stage_name=stage,
            usage=usage,
            latency_ms=latency_ms,
            model_name=getattr(self.llm, "model_name", self.model),
            estimated_prompt_tokens=estimated_input,
        )

        if isinstance(response.content, str):
            return response.content, usage
        return str(response.content) if response.content else "", usage
