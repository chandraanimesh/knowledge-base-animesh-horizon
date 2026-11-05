from functools import lru_cache
from typing import Any

from pydantic import Field, AliasChoices
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

    firecrawl_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("firecrawl_api_key", "FIRECRAWL_API_KEY"),
    )
    firecrawl_base_url: str | None = None

    web_search_provider: str = Field(
        default="tinyfish",
        validation_alias=AliasChoices("web_search_provider", "WEB_SEARCH_PROVIDER", "web_provider", "WEB_PROVIDER"),
    )
    tinyfish_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("tinyfish_api_key", "TINYFISH_API_KEY", "TINY_FISH", "tiny_fish"),
    )
    tinyfish_search_url: str = "https://api.search.tinyfish.ai"
    tinyfish_fetch_url: str = "https://api.fetch.tinyfish.ai"

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
    enable_web_research: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable_web_research", "ENABLE_WEB_RESEARCH"),
    )
    web_research_target_pages: int = 2
    web_research_max_scrapes: int = 4
    web_research_max_candidates_per_domain: int = 1
    enable_conversational_memory_extraction: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable_conversational_memory_extraction", "ENABLE_CONVERSATIONAL_MEMORY_EXTRACTION"),
    )
    dreaming_consolidation_threshold: int = 5
    enable_self_registration: bool = True
    unsafe_bypass_security_preprocessing: bool = False
    stirling_pdf_url: str | None = None
    stirling_api_key: str | None = None
    preprocessing_timeout: int = 60
    max_upload_size: int = 50 * 1024 * 1024 # 50 MB
    max_chunks_per_document: int = 4

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
    # Twenty CRM Integration Settings
    twenty_base_url: str = "http://localhost:3000"
    twenty_api_key: str | None = None
    twenty_timeout_seconds: float = 5.0
    enable_crm_integration: bool = True
    enable_connection_discovery: bool = True
    twenty_mcp_enabled: bool = False
    twenty_mcp_url: str | None = None
    twenty_crm_department_id: str | None = None
    twenty_webhook_secret: str | None = None
    twenty_webhook_max_bytes: int = 1024 * 1024  # 1 MB safe default
    twenty_webhook_mappings: str | dict[str, Any] | None = None

    def get_twenty_webhook_mappings(self) -> dict[str, Any]:
        if not self.twenty_webhook_mappings:
            return {}
        if isinstance(self.twenty_webhook_mappings, dict):
            return self.twenty_webhook_mappings
        try:
            import json
            return json.loads(str(self.twenty_webhook_mappings).strip())
        except Exception:
            return {}

    # Multi-format artifact storage and retention settings
    artifact_storage_dir: str = "storage/artifacts"
    artifact_retention_hours: int = 48
    artifact_max_bytes: int = 25 * 1024 * 1024
    artifact_max_rows: int = 500

    # Dedicated encryption key for workspace custom credentials (Fernet / MultiFernet)
    app_encryption_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("app_encryption_key", "APP_ENCRYPTION_KEY"),
        description="Dedicated Fernet key or comma-separated keys for MultiFernet rotation. Never derived from JWT_SECRET.",
    )

    # Durable background scheduler interval in seconds (default: 60s for minute-level cron responsiveness)
    scheduler_interval_seconds: int = Field(
        default=60,
        ge=5,
        le=3600,
        validation_alias=AliasChoices("scheduler_interval_seconds", "SCHEDULER_INTERVAL_SECONDS"),
        description="Background reconciliation and workflow trigger scheduler loop interval in seconds.",
    )

    # Adaptive Agent Runtime Settings (Milestone 10)
    adaptive_max_planner_calls: int = 3
    adaptive_max_plan_iterations: int = 3
    adaptive_max_tasks: int = 15
    adaptive_max_parallel_tasks: int = 5
    adaptive_max_execution_time_seconds: int = 180
    adaptive_max_tool_calls: int = 20
    adaptive_max_tokens: int = 25000
    adaptive_max_retries: int = 3
    adaptive_artifact_char_threshold: int = 4000
    adaptive_artifact_token_threshold: int = 1000

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

    def __setattr__(self, name: str, value: Any):
        setattr(get_settings(), name, value)


settings = _LazySettings()
