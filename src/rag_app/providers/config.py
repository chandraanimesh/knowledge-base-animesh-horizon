import os
from rag_app.core.config import get_settings


def get_llm_provider_name() -> str:
    settings = get_settings()
    return (os.getenv("LLM_PROVIDER") or getattr(settings, "llm_provider", "groq") or "groq").strip().lower()


def get_embedding_provider_name() -> str:
    settings = get_settings()
    name = (os.getenv("EMBEDDING_PROVIDER") or getattr(settings, "embedding_provider", "sentence_transformers") or "sentence_transformers").strip().lower()
    if name in ("sentence_transformer", "sentence_transformers", "st"):
        return "sentence_transformer"
    return name


def get_document_reader_provider_name() -> str:
    settings = get_settings()
    name = (os.getenv("DOCUMENT_READER_PROVIDER") or getattr(settings, "document_reader_provider", "standard") or "standard").strip().lower()
    if name in ("standard", "default", "builtin"):
        return "standard"
    return name


def get_reranker_provider_name() -> str:
    settings = get_settings()
    name = (os.getenv("RERANKER_PROVIDER") or getattr(settings, "reranker_provider", "cross_encoder") or "cross_encoder").strip().lower()
    if name in ("cross_encoder", "crossencoder", "ms-marco"):
        return "cross_encoder"
    return name


def get_ocr_provider_name() -> str:
    settings = get_settings()
    name = (os.getenv("OCR_PROVIDER") or getattr(settings, "ocr_provider", "paddleocr") or "paddleocr").strip().lower()
    return name


def get_web_provider_name() -> str:
    settings = get_settings()
    env_name = os.getenv("WEB_PROVIDER") or os.getenv("WEB_SEARCH_PROVIDER")
    if env_name:
        return env_name.strip().lower()
    if getattr(settings, "web_provider", None):
        return str(settings.web_provider).strip().lower()
    if getattr(settings, "web_search_provider", None):
        val = str(settings.web_search_provider).strip().lower()
        if val == "firecrawl" and not settings.firecrawl_api_key and settings.tinyfish_api_key:
            return "tinyfish"
        return val
    if settings.tinyfish_api_key and not settings.firecrawl_api_key:
        return "tinyfish"
    return "firecrawl"


def get_vector_store_provider_name() -> str:
    settings = get_settings()
    name = (os.getenv("VECTOR_STORE_PROVIDER") or getattr(settings, "vector_store_provider", "pgvector") or "pgvector").strip().lower()
    return name


def get_search_provider_name() -> str:
    settings = get_settings()
    name = (os.getenv("SEARCH_PROVIDER") or getattr(settings, "search_provider", "hybrid_pg") or "hybrid_pg").strip().lower()
    return name
