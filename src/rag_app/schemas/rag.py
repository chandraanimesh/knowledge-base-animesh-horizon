from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    file_name: str
    skipped: bool
    document_id: str | None = None
    chunks_created: int = 0
    reason: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    department_id: str
    file_name: str
    chunk_index: int
    text: str
    distance: float | None = None
    vector_similarity: float | None = None
    keyword_rank: float | None = None
    hybrid_score: float | None = None
    reranker_score: float | None = None
    hierarchical_score: float | None = None
    hierarchical_position: int | None = None
    node_id: str | None = None
    page_number: int | None = None


class GraphEdgeProvenance(BaseModel):
    subject: str
    predicate: str
    object: str
    hop: int = 1


class ChunkHierarchyMetadata(BaseModel):
    chunk_id: str
    node_id: str
    title: str
    parent_titles: list[str] = Field(default_factory=list)
    ordinal_path: str = ""
    level: int = 0
    l1_overview: str | None = None
    l0_abstract: str | None = None
    path_display: str | None = None


class ChunkProvenance(BaseModel):
    chunk_id: str | None = None
    strategies: list[str] = Field(default_factory=list)
    knowledge_item_id: str | None = None
    entity_name: str | None = None
    entity_type: str | None = None
    confidence: float | None = None
    graph_path: list[GraphEdgeProvenance] = Field(default_factory=list)
    chunk_created_at: str | None = None
    item_updated_at: str | None = None


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    strategy_trace: dict[str, int] = Field(default_factory=dict)
    chunk_provenance: dict[str, ChunkProvenance] = Field(default_factory=dict)
    chunk_hierarchy: dict[str, ChunkHierarchyMetadata] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0


class SubStageMetric(BaseModel):
    name: str
    latency_ms: float = 0.0


class StageMetrics(BaseModel):
    latency_ms: float = 0.0
    llm_calls: int = 0
    llm_usage: TokenUsage = Field(default_factory=TokenUsage)
    sub_stages: list[SubStageMetric] = Field(default_factory=list)


class ExecutionStages(BaseModel):
    retrieval: StageMetrics | None = None
    generation: StageMetrics | None = None
    research: StageMetrics | None = None


class ExecutionMetrics(BaseModel):
    request_id: str
    total_latency_ms: float = 0.0
    llm_usage: TokenUsage = Field(default_factory=TokenUsage)
    stages: ExecutionStages = Field(default_factory=ExecutionStages)


import uuid
from datetime import datetime


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = False
    conversation_id: uuid.UUID | None = None
    workspace_id: uuid.UUID | None = None
    provider_overrides: dict[str, str] | None = Field(
        default=None,
        description="Optional runtime provider overrides for experiment runs (e.g. {'llm': 'groq', 'embedding': 'sentence_transformers'}). Does not alter stored workspace defaults.",
    )



class GenerateResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved_chunks: list[RetrievedChunk]
    execution_metrics: ExecutionMetrics | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    conversation_title: str | None = None



class ChatMessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources: list[dict] | None = None
    execution_metrics: ExecutionMetrics | dict | None = None
    created_at: datetime


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    workspace_id: uuid.UUID | None = None
    workspace_name: str | None = None


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    workspace_id: uuid.UUID | None = None
    messages: list[ChatMessageResponse] = Field(default_factory=list)


class ConversationCreateRequest(BaseModel):
    title: str | None = None
    workspace_id: uuid.UUID | None = None



class ConflictNotice(BaseModel):
    conflict_type: str = "RELATIONSHIP_DIVERGENCE"
    subject: str
    predicate: str
    conflicting_chunk_ids: list[str] = Field(default_factory=list)
    conflicting_objects: list[str] = Field(default_factory=list)
    details: str = ""


class CompiledContext(BaseModel):
    chunks: list[RetrievedChunk]
    token_count: int
    dropped_count: int
    provenance_summary: dict[str, int] = Field(default_factory=dict)
    chunk_provenance: dict[str, ChunkProvenance] = Field(default_factory=dict)
    conflicts: list[ConflictNotice] = Field(default_factory=list)
    chunk_hierarchy: dict[str, ChunkHierarchyMetadata] = Field(default_factory=dict)
    selected_node_ids: list[str] = Field(default_factory=list)
    rendered_context: str | None = None

class CompanyFilters(BaseModel):
    company_id: str | None = None
    company_name_contains: str | None = None
    funding_status: str | None = None
    city: str | None = None
    country: str | None = None

class RelationshipQuery(BaseModel):
    subject_entity: str | None = None
    predicate: str | None = None
    object_entity: str | None = None
    direction: str = "out"  # "out", "in", "both"
    hop_limit: int = 1


class QueryPlan(BaseModel):
    semantic_query: str | None = None
    filters: CompanyFilters = Field(
        default_factory=CompanyFilters
    )
    relationship_query: RelationshipQuery | None = None
    entity_search_term: str | None = None