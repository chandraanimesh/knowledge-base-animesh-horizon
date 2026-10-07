from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "production-rag-backend"
    env: str = "dev"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/ragdb"

    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"

    firecrawl_api_key: str | None = None
    firecrawl_base_url: str | None = None

    chunk_size: int = 900
    chunk_overlap: int = 150
    default_top_k: int = 5
    max_context_tokens: int = 4000

    min_vector_similarity: float = 0.25
    retrieval_candidate_k: int = 20
    rrf_k: int = 60
    vector_search_weight: float = 1.0
    keyword_search_weight: float = 1.2
    structured_search_weight: float = 2.0
    hierarchical_search_weight: float = 0.8
    hierarchical_max_descendant_depth: int = 3
    enable_web_research: bool = False
    unsafe_bypass_security_preprocessing: bool = False
    stirling_pdf_url: str | None = None
    stirling_api_key: str | None = None
    preprocessing_timeout: int = 60
    max_upload_size: int = 50 * 1024 * 1024 # 50 MB

    # OV-1: Hierarchical resource model — L0/L1 generation bounds
    node_l0_max_chars: int = 300   # max chars for L0 section abstract
    node_l1_max_chars: int = 600   # max chars for L1 section overview
    
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    
    admin_email: str | None = None
    admin_password: str | None = None

    langsmith_api_key: str | None = None
    langsmith_tracing: str | None = None
    langsmith_project: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()


class _LazySettings:
    """Lazy proxy for Settings that defers get_settings() resolution until attribute access."""

    def __getattr__(self, name: str):
        return getattr(get_settings(), name)


settings = _LazySettings()

