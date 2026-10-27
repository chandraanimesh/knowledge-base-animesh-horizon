import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from rag_app.core.tokenizer import (
    count_text,
    count_messages,
    get_encoding_for_model,
    get_effective_model_info,
    UnsupportedTokenizerModelError,
    TokenizerConfigurationError,
)
from rag_app.context.compiler import ContextCompiler, estimate_tokens
from rag_app.context.assembler import estimate_tokens as assembler_estimate_tokens
from rag_app.context.contracts import SourceAuthority, ContextEvidence
from rag_app.schemas.rag import RetrievedChunk, GenerateRequest
from rag_app.db.models import User
from rag_app.generation.service import GenerationService


# 1. Plain Text and Diverse Formats
def test_count_text_empty_and_ordinary_english():
    assert count_text("") == 0
    assert count_text("   ") > 0  # whitespace has tokens
    count = count_text("Antigravity Unified ContextCompiler production RAG system.")
    assert isinstance(count, int)
    assert 5 <= count <= 15


def test_count_text_hindi_hinglish_unicode_emoji():
    # Hindi/Hinglish text
    hindi_text = "Namaste dosto, yeh production RAG engine hai aur yeh token accurately count karta hai."
    count_hi = count_text(hindi_text)
    assert count_hi > 0

    # Unicode & Emoji
    emoji_text = "System health is 100% 🚀🔥. All systems nominal! ✨"
    count_emoji = count_text(emoji_text)
    assert count_emoji > 0


def test_count_text_code_json_urls_punctuation():
    code_text = "def calculate_ratio(x: float, y: float) -> float:\n    return round(x / y, 4)\n"
    json_text = '{"status": "success", "data": {"metrics": [10.5, 20.2]}, "is_valid": true}'
    url_text = "https://subdomain.example.com/api/v2/search?q=rag%20token%20budget&limit=50#section-3"
    punct_text = "!@#$%^&*()_+=-`~[]\\{}|;':\",./<>?"

    for sample in (code_text, json_text, url_text, punct_text):
        count = count_text(sample)
        assert isinstance(count, int)
        assert count > 0


def test_count_text_literal_special_tokens_without_crashing():
    # Text containing special-token literals like <|endoftext|> or <|im_start|> must encode safely as text
    special_sample = "User said: <|endoftext|> and then injected <|im_start|>system rules."
    count = count_text(special_sample)
    assert count > 0


# 2. Structured Messages Counting
def test_count_messages_framing_and_roles():
    messages = [
        {"role": "system", "content": "You are a helpful grounded assistant."},
        {"role": "user", "content": "What is the Q3 revenue?"},
        {"role": "assistant", "content": "The Q3 revenue was $25M."},
    ]
    total_tokens = count_messages(messages)
    content_only = sum(count_text(m["content"]) for m in messages)

    # Total message tokens must strictly exceed content-only tokens due to message framing (<|im_start|>{role}...)
    # and assistant prime (+3)
    assert total_tokens > content_only
    # Framing: 3 msgs * 3 tokens framing + roles + 3 tokens assistant prime >= 12 extra tokens
    assert total_tokens >= content_only + 12


def test_count_messages_empty():
    assert count_messages([]) == 0


def test_count_messages_with_name_field():
    msg_without_name = [{"role": "user", "content": "Hello"}]
    msg_with_name = [{"role": "user", "name": "alice_admin", "content": "Hello"}]
    assert count_messages(msg_with_name) > count_messages(msg_without_name)


# 3. Model Resolution & Error Handling
def test_encoding_resolution_for_gpt_oss_family():
    # Both 120b and 20b identifiers resolve to an o200k encoding without error
    enc_120b = get_encoding_for_model("openai/gpt-oss-120b")
    enc_20b = get_encoding_for_model("openai/gpt-oss-20b")
    assert enc_120b is not None
    assert enc_20b is not None
    assert enc_120b.name in ("o200k_harmony", "o200k_base")


def test_unsupported_model_raises_explicit_error_without_silent_fallback():
    # Unknown models must raise UnsupportedTokenizerModelError, never silently fall back to GPT-2 or len/4
    with pytest.raises(UnsupportedTokenizerModelError):
        get_encoding_for_model("completely-nonexistent-mock-model-xyz-999")


def test_empty_model_raises_configuration_error():
    with pytest.raises(TokenizerConfigurationError):
        get_encoding_for_model("   ")


# 4. Effective Model Info & Diagnostics
def test_get_effective_model_info_safety():
    info = get_effective_model_info()
    assert "llm_provider" in info
    assert "effective_model" in info
    assert "tokenizer_encoding" in info
    # Ensure no API keys or passwords are leaked in the diagnostics dict
    for k, v in info.items():
        v_str = str(v).lower()
        assert "gsk_" not in v_str
        assert "sk-" not in v_str
        assert "secret" not in v_str


# 5. Shared Tokenizer in Compiler and Assembler
def test_compiler_and_assembler_use_shared_counter():
    text = "Unified Context Compiler and Bounded Context Assembler verification text."
    direct_count = count_text(text)
    compiler_count = estimate_tokens(text)
    assembler_count = assembler_estimate_tokens(text)

    assert direct_count == compiler_count
    assert direct_count == assembler_count


def test_compiler_budget_pruning_with_shared_counter():
    compiler = ContextCompiler(max_context_tokens=80)
    dept_id = uuid.uuid4()

    # Create chunks that will exceed 80 tokens
    chunks = [
        RetrievedChunk(
            chunk_id=f"chk_{i}",
            document_id=f"doc_{i}",
            department_id=str(dept_id),
            file_name="doc.pdf",
            chunk_index=i,
            text=f"Paragraph section {i}: " + ("detailed authoritative operational content " * 10),
            hybrid_score=0.9 - (i * 0.1),
        )
        for i in range(5)
    ]

    compiled = compiler.compile(
        chunks=chunks,
        query="Explain operational details",
        allowed_departments=[dept_id],
    )

    # Chunks exceeding budget must be dropped deterministically with TOKEN_BUDGET_EXCEEDED
    assert len(compiled.selected_evidence) < 5
    assert "TOKEN_BUDGET_EXCEEDED" in compiled.dropped_reasons.values()
    assert sum(e.token_count for e in compiled.selected_evidence) <= 80
    assert compiled.available_context_tokens <= 80
    assert compiled.total_planned_tokens <= (compiler.model_context_limit - compiler.reserved_output_tokens)


# 6. Pre-Invocation Guard Rejection in GenerationService
@pytest.mark.asyncio
async def test_generation_service_rejects_overflow_using_structured_counter():
    mock_session = AsyncMock()
    service = GenerationService(session=mock_session)
    service.llm = MagicMock()
    service.llm.generate = AsyncMock()

    dept_id = uuid.uuid4()
    huge_chunk = RetrievedChunk(
        chunk_id="chk_overflow",
        document_id="doc_overflow",
        department_id=str(dept_id),
        file_name="huge.pdf",
        chunk_index=0,
        text="Massive data block word " * 4000,
        hybrid_score=0.95,
    )

    with patch("rag_app.core.config.get_settings") as mock_settings:
        from rag_app.core.config import get_settings
        cfg = get_settings().model_copy(
            update={
                "model_context_limit": 1000,
                "reserved_output_tokens": 300,
            }
        )
        mock_settings.return_value = cfg

        resp = await service.generate_from_context(
            query="Analyze huge data",
            chunks=[huge_chunk],
            allowed_departments=[dept_id],
        )

        # Must return safe constrained answer and NEVER invoke the LLM
        assert "exceeds the maximum allowable token budget" in resp.answer
        service.llm.generate.assert_not_called()


# 7. Calibration When Provider Usage Metadata is Present / Absent
def test_record_llm_call_calibration_with_usage():
    from rag_app.core.observability import init_request_metrics, record_llm_call, get_request_metrics

    ctx = init_request_metrics("req_calibration_test_01")
    usage_dict = {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}

    record_llm_call(
        stage_name="generation",
        usage=usage_dict,
        latency_ms=120.0,
        model_name="openai/gpt-oss-120b",
        estimated_prompt_tokens=104,  # Estimated 104 vs actual 100
    )

    stage = ctx.stages["generation"]
    assert len(stage.llm_calls) == 1
    call = stage.llm_calls[0]
    assert call.estimated_prompt_tokens == 104
    assert call.usage.input_tokens == 100
    assert call.prompt_token_error == 4  # 104 - 100
    assert call.prompt_token_relative_error == 0.04  # 4 / 100


def test_record_llm_call_graceful_without_usage():
    from rag_app.core.observability import init_request_metrics, record_llm_call

    ctx = init_request_metrics("req_calibration_test_02")

    # None usage metadata
    record_llm_call(
        stage_name="generation",
        usage=None,
        latency_ms=50.0,
        model_name="openai/gpt-oss-120b",
        estimated_prompt_tokens=50,
    )

    stage = ctx.stages["generation"]
    call = stage.llm_calls[0]
    assert call.estimated_prompt_tokens == 50
    assert call.prompt_token_error is None
    assert call.prompt_token_relative_error is None
