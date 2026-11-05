import os
import time
from typing import Any
from pydantic import BaseModel

from rag_app.core.config import get_settings
from rag_app.core.observability import record_llm_call
from rag_app.providers.base.llm import LLMProvider
from rag_app.providers.exceptions import InvalidProviderConfigError


class AIMessageProxy:
    """Lightweight proxy matching LangChain AIMessage shape for downstream consumers."""

    def __init__(self, content: str, usage_metadata: dict[str, Any] | None = None):
        self.content = content
        self.usage_metadata = usage_metadata or {}


class StructuredOutputRunnable:
    """Runnable wrapper executing structured parsing for any OpenAI-compatible provider."""

    def __init__(self, adapter: "OpenAILLMAdapter", schema: type[BaseModel]):
        self.adapter = adapter
        self.schema = schema

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        client = self.adapter._get_client()
        clean_msgs = self.adapter._format_messages(messages)

        # 1. Try native beta parse (standard OpenAI, DeepSeek, vLLM modern endpoints)
        try:
            response = await client.beta.chat.completions.parse(
                model=self.adapter.model,
                messages=clean_msgs,
                response_format=self.schema,
                temperature=self.adapter.temperature,
                **self.adapter.extra_kwargs,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception:
            pass

        # 2. Fallback: prompt schema reinforcement and validate JSON
        schema_json = self.schema.model_json_schema()
        system_instruction = (
            f"You must strictly output valid JSON matching this schema: {schema_json}. "
            "Do not output markdown code blocks, preamble, or commentary."
        )
        augmented_msgs = [{"role": "system", "content": system_instruction}] + clean_msgs
        raw_text, _ = await self.adapter.generate(augmented_msgs)
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        return self.schema.model_validate_json(cleaned)


class OpenAILLMAdapter(LLMProvider):
    """
    Universal OpenAI-Compatible LLM Adapter.
    Works seamlessly with:
    - Official OpenAI (gpt-4o, gpt-4o-mini, o1, o3-mini)
    - DeepSeek (deepseek-chat, deepseek-reasoner)
    - OpenRouter (all models)
    - Together AI, Mistral, Perplexity
    - Local LLMs: Ollama (http://localhost:11434/v1), vLLM, LM Studio
    """

    provider_name: str = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ):
        settings = get_settings()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or getattr(settings, "openai_api_key", None)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or getattr(settings, "openai_base_url", None)
        self.model = model or getattr(settings, "openai_model", "gpt-4o")
        self.temperature = temperature
        self.extra_kwargs = kwargs

        # Allow consumer code accessing .llm to find methods directly
        self.llm = self

    def _get_client(self):
        from openai import AsyncOpenAI

        effective_key = self.api_key or "placeholder_key"
        kwargs: dict[str, Any] = {"api_key": effective_key}
        if self.base_url and str(self.base_url).strip():
            kwargs["base_url"] = str(self.base_url).strip()
        return AsyncOpenAI(**kwargs)

    def _format_messages(self, messages: Any) -> list[dict[str, str]]:
        """Normalize messages from dicts or LangChain BaseMessage objects."""
        if not isinstance(messages, list):
            messages = [messages]

        clean: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = str(m.get("content", ""))
                clean.append({"role": role, "content": content})
            elif hasattr(m, "type") or hasattr(m, "content"):
                msg_type = getattr(m, "type", "user")
                role = "system" if msg_type == "system" else "assistant" if msg_type in ("ai", "assistant") else "user"
                content = str(getattr(m, "content", ""))
                clean.append({"role": role, "content": content})
            else:
                clean.append({"role": "user", "content": str(m)})
        return clean

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        from rag_app.core.tokenizer import count_messages
        try:
            return count_messages(messages, model=self.model)
        except Exception:
            # Fallback estimation
            total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
            return max(1, total_chars // 4)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessageProxy:
        """LangChain compatibility invoke."""
        content, usage = await self.generate(messages, **kwargs)
        return AIMessageProxy(content=content, usage_metadata=usage)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredOutputRunnable:
        """Returns structured runnable conforming to schema."""
        return StructuredOutputRunnable(self, schema)

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
        if not self.api_key or not str(self.api_key).strip():
            # If base_url is a local endpoint like Ollama/vLLM, dummy key is acceptable
            if not (self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url)):
                raise InvalidProviderConfigError(
                    f"API key is required for provider '{self.provider_name}' (model: {self.model})."
                )

        client = self._get_client()
        clean_msgs = self._format_messages(messages)

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=self.model,
            messages=clean_msgs,
            temperature=self.temperature,
            **self.extra_kwargs,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        estimated_input = None
        try:
            estimated_input = self.count_tokens(clean_msgs)
        except Exception:
            estimated_input = None

        record_llm_call(
            stage_name=stage,
            usage=usage,
            latency_ms=latency_ms,
            model_name=self.model,
            estimated_prompt_tokens=estimated_input,
        )

        return content, usage

