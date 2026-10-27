import logging
from functools import lru_cache
from typing import Any
import tiktoken
from rag_app.core.config import get_settings

logger = logging.getLogger(__name__)


class TokenizerConfigurationError(Exception):
    """Raised when tokenizer configuration or initialization fails."""
    pass


class UnsupportedTokenizerModelError(TokenizerConfigurationError):
    """Raised when an explicit model identifier cannot be resolved to a known tokenizer encoding."""
    pass


# Explicit encoding overrides for model families
_MODEL_TO_ENCODING: dict[str, str] = {
    # OpenAI GPT-OSS open weights models
    "openai/gpt-oss-120b": "o200k_harmony",
    "openai/gpt-oss-20b": "o200k_harmony",
    "gpt-oss-120b": "o200k_harmony",
    "gpt-oss-20b": "o200k_harmony",
    # OpenAI Flagship models
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Llama 3 models on Groq (calibrated 128k BPE approximation via o200k_base)
    "llama-3.3-70b-versatile": "o200k_base",
    "llama-3.1-8b-instant": "o200k_base",
    "llama3-70b-8192": "o200k_base",
    "llama3-8b-8192": "o200k_base",
}


@lru_cache(maxsize=16)
def get_encoding_for_model(model_name: str | None = None) -> tiktoken.Encoding:
    """
    Resolve the matching tiktoken.Encoding instance for a given model identifier.

    Resolution Strategy:
    1. If model_name is None or empty, fall back to configured settings.groq_model.
    2. Normalize model identifier (stripped, lowercase).
    3. Check explicit mapping table (_MODEL_TO_ENCODING).
    4. Handle 'gpt-oss' family specifically: prefer 'o200k_harmony' if available in tiktoken,
       otherwise 'o200k_base'.
    5. Attempt standard tiktoken.encoding_for_model().
    6. If all resolution attempts fail, raise UnsupportedTokenizerModelError.
       We NEVER silently fall back to GPT-2 or character heuristics when resolution fails.
    """
    settings = get_settings()
    raw_name = model_name or getattr(settings, "groq_model", "openai/gpt-oss-120b")
    normalized_name = (raw_name or "").strip().lower()

    if not normalized_name:
        raise TokenizerConfigurationError("Model name cannot be empty when resolving tokenizer encoding.")

    # 1. Explicit mapping lookup
    if normalized_name in _MODEL_TO_ENCODING:
        enc_name = _MODEL_TO_ENCODING[normalized_name]
        try:
            return tiktoken.get_encoding(enc_name)
        except Exception:
            # If o200k_harmony is not available on older tiktoken builds, gracefully try o200k_base
            if enc_name == "o200k_harmony":
                return tiktoken.get_encoding("o200k_base")
            raise

    # 2. Heuristic mapping for gpt-oss family
    if "gpt-oss" in normalized_name:
        available = tiktoken.list_encoding_names()
        if "o200k_harmony" in available:
            return tiktoken.get_encoding("o200k_harmony")
        return tiktoken.get_encoding("o200k_base")

    # 3. Llama / Meta open model family mapping
    if "llama" in normalized_name:
        return tiktoken.get_encoding("o200k_base")

    # 4. Standard tiktoken model lookup
    try:
        return tiktoken.encoding_for_model(normalized_name)
    except KeyError:
        pass

    # 5. Direct encoding name check (e.g. user passed 'o200k_base' directly)
    if normalized_name in tiktoken.list_encoding_names():
        return tiktoken.get_encoding(normalized_name)

    raise UnsupportedTokenizerModelError(
        f"Cannot resolve tokenizer encoding for model '{raw_name}'. "
        f"Available encodings: {tiktoken.list_encoding_names()}"
    )


def count_text(text: str, model: str | None = None) -> int:
    """
    Count input tokens for arbitrary plain text strings using the resolved model encoding.

    Guarantees:
    - Empty strings return 0.
    - Disables disallowed special token checks (disallowed_special=()) so text containing
      literals like '<|endoftext|>' or '<|im_start|>' is safely encoded as normal text
      without crashing.
    """
    if not text:
        return 0

    encoding = get_encoding_for_model(model)
    return len(encoding.encode(text, disallowed_special=()))


def count_messages(
    messages: list[dict[str, str]],
    model: str | None = None,
) -> int:
    """
    Count tokens for structured chat messages formatted for OpenAI/Groq Chat Completions API.

    Wire Serialization Note:
    Groq Cloud serves open models through an OpenAI-compatible Chat Completions REST API.
    Messages are formatted using standard chat message envelopes:
      <|im_start|>{role}\n{content}<|im_end|>\n
    Framing Overhead Calculation:
    - 3 tokens per message for delimiter framing: <|im_start|>, role delimiter, <|im_end|>.
    - Plus role string tokens.
    - Plus content string tokens.
    - Plus name tokens (if provided).
    - Plus 3 priming tokens for the assistant response prompt: <|im_start|>assistant\\n.

    Disclaimer:
    Exact provider-side serialization (system turn wrappers, whitespace trimming, templating)
    may vary slightly (typically within 1-3 tokens) depending on the provider's server backend.
    This function provides a deterministic, calibrated estimate. Actual usage metadata
    returned by the provider (usage_metadata['prompt_tokens']) should be recorded for empirical
    calibration.
    """
    if not messages:
        return 0

    encoding = get_encoding_for_model(model)
    num_tokens = 0
    tokens_per_message = 3
    tokens_per_name = 1

    for message in messages:
        num_tokens += tokens_per_message
        role = message.get("role", "")
        if role:
            num_tokens += len(encoding.encode(role, disallowed_special=()))

        content = message.get("content", "")
        if content:
            num_tokens += len(encoding.encode(content, disallowed_special=()))

        if "name" in message:
            num_tokens += tokens_per_name
            name = message["name"]
            if name:
                num_tokens += len(encoding.encode(name, disallowed_special=()))

    # Every reply is primed with <|im_start|>assistant\n (3 tokens)
    num_tokens += 3
    return num_tokens


def get_effective_model_info() -> dict[str, Any]:
    """
    Return effective runtime model and tokenizer diagnostics.
    Never exposes API keys or secrets.
    """
    settings = get_settings()
    model_name = getattr(settings, "groq_model", "openai/gpt-oss-120b")
    provider_name = getattr(settings, "llm_provider", "groq")
    try:
        enc = get_encoding_for_model(model_name)
        enc_name = getattr(enc, "name", "unknown")
    except Exception as exc:
        enc_name = f"resolution_error: {exc}"

    return {
        "llm_provider": provider_name,
        "effective_model": model_name,
        "tokenizer_encoding": enc_name,
        "model_context_limit": getattr(settings, "model_context_limit", 8192),
        "reserved_output_tokens": getattr(settings, "reserved_output_tokens", 500),
    }


def log_effective_model_diagnostics() -> None:
    """Safe diagnostic logging of runtime model and tokenizer without secrets."""
    info = get_effective_model_info()
    logger.info(
        "effective_llm_runtime_diagnostics",
        extra={
            "event": "effective_llm_runtime_diagnostics",
            "provider": info["llm_provider"],
            "model": info["effective_model"],
            "encoding": info["tokenizer_encoding"],
            "model_context_limit": info["model_context_limit"],
            "reserved_output_tokens": info["reserved_output_tokens"],
        },
    )
