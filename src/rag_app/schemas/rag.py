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


from enum import StrEnum


class FailureCategory(StrEnum):
    QUERY_UNDERSTANDING = "QUERY_UNDERSTANDING"
    RETRIEVAL = "RETRIEVAL"
    RERANKING = "RERANKING"
    EVIDENCE_SELECTION = "EVIDENCE_SELECTION"
    SUFFICIENCY = "SUFFICIENCY"
    GENERATION = "GENERATION"
    GROUNDING = "GROUNDING"
    VISUALIZATION = "VISUALIZATION"
    AUTHORIZATION = "AUTHORIZATION"
    PERSISTENCE = "PERSISTENCE"
    STREAMING = "STREAMING"
    PROVIDER = "PROVIDER"


class ExecutionStages(BaseModel):
    retrieval: StageMetrics | None = None
    generation: StageMetrics | None = None
    research: StageMetrics | None = None


class ExecutionMetrics(BaseModel):
    # Identifiers & Context
    request_id: str
    trace_id: str | None = None
    workspace_id: str | None = None
    thread_id: str | None = None
    provider: str | None = None
    model: str | None = None

    # Overall Token & Latency
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_available: bool = True
    llm_usage: TokenUsage = Field(default_factory=TokenUsage)

    # Detailed Latency Split (ms)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    grounding_latency_ms: float = 0.0
    repair_latency_ms: float = 0.0
    initial_generation_latency_ms: float = 0.0
    regeneration_latency_ms: float = 0.0

    # Retrieval Strategies & Candidates
    planned_strategies: list[str] = Field(default_factory=list)
    executed_strategies: list[str] = Field(default_factory=list)
    vector_candidates: int = 0
    keyword_candidates: int = 0
    knowledge_candidates: int = 0
    hierarchical_candidates: int = 0
    candidate_count: int = 0
    reranked_candidates: int = 0
    accepted_results: int = 0
    accepted_count: int = 0
    final_context_count: int = 0
    context_chunks: int = 0
    context_tokens: int = 0

    # Retrieval Quality & Breakdown
    vector_top_k: int | None = None
    keyword_top_k: int | None = None
    hierarchical_count: int = 0
    reranker_top_score: float | None = None
    reranker_bottom_score: float | None = None
    structured_evidence_count: int = 0
    hierarchical_evidence_count: int = 0
    conversation_evidence_count: int = 0

    # Sufficiency Quality
    sufficiency: bool | None = None
    sufficiency_status: str | None = None
    evidence_found: bool | None = None
    sufficiency_confidence: float | None = None
    sufficiency_coverage: float | None = None
    sufficiency_reason: str | None = None
    missing_requirements_count: int = 0
    contradiction_count: int = 0
    evidence_types: list[str] = Field(default_factory=list)

    # Grounding Quality & Enforcement
    grounding: bool | None = None
    grounding_confidence: float | None = None
    unsupported_claim_count: int = 0
    evidence_ref_count: int = 0
    repair_triggered: bool = False
    regeneration_triggered: bool = False
    safe_limitation_triggered: bool = False

    # Visualization Observability
    visualization_requested: bool = False
    visualization_generated: bool = False
    visualization_validated: bool = False
    visualization_repaired: bool = False
    visualization_dropped: bool = False
    visualization_failure_reason: str | None = None

    # Streaming & Taxonomy
    streaming: bool = False
    failure_categories: list[FailureCategory] = Field(default_factory=list)

    # Legacy & Stage Metrics
    stages: ExecutionStages = Field(default_factory=ExecutionStages)


import uuid
from datetime import datetime


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True
    conversation_id: uuid.UUID | None = None
    workspace_id: uuid.UUID | None = None
    provider_overrides: dict[str, str] | None = Field(
        default=None,
        description="Optional runtime provider overrides for experiment runs (e.g. {'llm': 'groq', 'embedding': 'sentence_transformers'}). Does not alter stored workspace defaults.",
    )
    action: str | None = Field(
        default=None,
        description="Optional typed action intent, e.g. 'visualize'.",
    )
    target_message_id: uuid.UUID | None = Field(
        default=None,
        description="Optional target assistant message ID to visualize or follow up on.",
    )




class AnswerGroundingResult(BaseModel):
    """Evaluation of whether the generated answer and visualization are grounded in authorized evidence."""
    is_grounded: bool = Field(description="True if all claims and data points in the answer are supported by authorized evidence.")
    confidence: float | None = Field(default=None, description="Confidence score in the grounding evaluation (0.0 to 1.0).")
    unsupported_claims: list[str] = Field(default_factory=list, description="List of factual claims, numbers, dates, or rows not supported by evidence.")
    evidence_refs: list[str] = Field(default_factory=list, description="Traceable chunk/source references supporting the grounded claims.")
    contradictions: list[str] = Field(default_factory=list, description="Detected contradictions or conflicting statements between answer and evidence.")
    reason: str | None = Field(default=None, description="Detailed explanation of the grounding validation.")


class GenerateResponse(BaseModel):
    answer: str
    visualization: dict | None = None
    citations: list[str]
    retrieved_chunks: list[RetrievedChunk]
    grounding: AnswerGroundingResult | None = None
    execution_metrics: ExecutionMetrics | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    conversation_title: str | None = None



class ChatMessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    visualization: dict | None = None
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


from enum import StrEnum


class InformationNeed(StrEnum):
    OVERVIEW = "OVERVIEW"
    FACTUAL = "FACTUAL"
    REQUIREMENTS = "REQUIREMENTS"
    LIST = "LIST"
    COMPARISON = "COMPARISON"
    AGGREGATION = "AGGREGATION"
    EXPLANATION = "EXPLANATION"
    CONVERSATION = "CONVERSATION"
    GENERAL = "GENERAL"


class PresentationNeed(StrEnum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    CHART = "CHART"
    TIMELINE = "TIMELINE"
    NETWORK = "NETWORK"
    NONE = "NONE"


class EvidenceNeed(StrEnum):
    PROSE = "PROSE"
    HIERARCHICAL = "HIERARCHICAL"
    STRUCTURED = "STRUCTURED"
    MIXED = "MIXED"
    CONVERSATION = "CONVERSATION"


class QueryPlan(BaseModel):
    semantic_query: str | None = None
    filters: CompanyFilters = Field(
        default_factory=CompanyFilters
    )
    relationship_query: RelationshipQuery | None = None
    entity_search_term: str | None = None
    information_need: InformationNeed = InformationNeed.GENERAL
    presentation_need: PresentationNeed = PresentationNeed.NONE
    evidence_need: EvidenceNeed = EvidenceNeed.PROSE
    conversation_required: bool = False
    reason: str | None = None