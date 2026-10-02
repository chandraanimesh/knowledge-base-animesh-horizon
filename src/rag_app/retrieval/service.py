
import logging
from dataclasses import dataclass
import uuid
import time
from rag_app.core.observability import (
    record_stage_start,
    record_stage_end,
    record_sub_stage,
)
from rag_app.retrieval.filter_normalise import normalize_company_filters
from sqlalchemy import func, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from rag_app.retrieval.bm25 import GlobalBM25Index
from rag_app.retrieval.reranker import AsyncCrossEncoderProvider, get_reranker_provider
from rag_app.core.config import get_settings
from rag_app.core.tracing import track
from rag_app.db.models import (
    Document,
    DocumentChunk,
    DocumentSection,
    KnowledgeItem,
    KnowledgeEntity,
    KnowledgeRelationship,
    KnowledgeEvidence,
)
from rag_app.ingestion.embedding_v1 import get_embedding_provider
from rag_app.ingestion.normaliser import normalize_company_id
from rag_app.retrieval.query_parser import QueryParser, is_greeting, extract_entity_search_term
from rag_app.retrieval.router import RetrievalMode, route_query
from rag_app.retrieval.controller import RetrievalController, RetrievalStrategy
from rag_app.knowledge.materializer import normalize_string


from rag_app.schemas.rag import (
    ChunkHierarchyMetadata,
    ChunkProvenance,
    CompanyFilters,
    GraphEdgeProvenance,
    QueryPlan,
    RelationshipQuery,
    RetrievedChunk,
    RetrievalResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def _format_timestamp(val: object) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if hasattr(val, "isoformat") and callable(val.isoformat):
        try:
            res = val.isoformat()
            if isinstance(res, str):
                return res
        except Exception:
            pass
    return None


@dataclass
class HybridCandidate:
    chunk: DocumentChunk
    document: Document

    distance: float | None = None
    vector_similarity: float | None = None
    keyword_rank: float | None = None

    vector_position: int | None = None
    keyword_position: int | None = None
    knowledge_position: int | None = None
    knowledge_score: float | None = None
    hierarchical_position: int | None = None
    hierarchical_score: float | None = None

    hybrid_score: float = 0.0
    reranker_score: float | None = None


class RetrievalService:

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        self.session = session
        self.embedding_provider = (
            get_embedding_provider()
        )
        self.query_parser = QueryParser()
        self.reranker_provider = get_reranker_provider()
        self.controller = RetrievalController()
    @track(
        name="retrieve_chunks",
        phase="retrieval",
    )
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool = False,
        allowed_departments: list[uuid.UUID] | None = None,
        allowed_documents: list[uuid.UUID] | None = None,
        workspace_id: uuid.UUID | None = None,
    ) -> RetrievalResponse:
        record_stage_start("retrieval")
        try:
            return await self._do_retrieve(
                query=query,
                top_k=top_k,
                rerank=rerank,
                allowed_departments=allowed_departments,
                allowed_documents=allowed_documents,
                workspace_id=workspace_id,
            )
        finally:
            record_stage_end("retrieval")

    async def _do_retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool = False,
        allowed_departments: list[uuid.UUID] | None = None,
        allowed_documents: list[uuid.UUID] | None = None,
        workspace_id: uuid.UUID | None = None,
    ) -> RetrievalResponse:

        top_k = (
            top_k
            or settings.default_top_k
        )

        processed_query = (
            self._preprocess_query(query)
        )
        
        if not processed_query:
            return RetrievalResponse(
                query=query,
                results=[],
            )
        # =========================================================
        # CONVERSATIONAL QUERY
        # =========================================================

        if is_greeting(processed_query):
            logger.info(
                "conversational_query_detected",
                extra={
                    "event": "conversational_query_detected",
                    "query": processed_query,
                },
            )

            return RetrievalResponse(
                query=query,
                results=[],
    )
        # =========================================================
        # 1. QUERY PARSING
        # =========================================================

        _qp_start = time.perf_counter()
        query_plan = (
            await self.query_parser.parse(
                processed_query
                )
        )
        record_sub_stage(
            "retrieval",
            "query_parsing",
            (time.perf_counter() - _qp_start) * 1000,
        )
        # normalisation 
        entity_term = (
            query_plan.entity_search_term
            if query_plan.entity_search_term is not None
            else self._extract_entity_search_term(query_plan)
        )
        query_plan = query_plan.model_copy(
            update={
                "filters": normalize_company_filters(
                    query_plan.filters
                ),
                "entity_search_term": entity_term,
            }
        )
        
        route = self.controller.plan_route(query_plan)
        is_compound = self.controller.is_compound_query(query_plan)
        retrieval_mode = route_query(
            query_plan
        )

        logger.info(
            "retrieval_route_selected",
            extra={
                "event": "retrieval_route_selected",
                "query": processed_query,
                "mode": retrieval_mode.value,
                "route": [s.value for s in route],
                "is_compound": is_compound,
                "semantic_query": (
                    query_plan.semantic_query
                ),
                "filters": (
                    query_plan.filters.model_dump()
                ),
            },
        )

    
        # =========================================================
        # 2. SEQUENTIAL STRATEGY EXECUTION VIA CONTROLLER
        # =========================================================
        vector_rows: list[tuple[DocumentChunk, Document, float]] = []
        keyword_rows: list[tuple[DocumentChunk, Document, float]] = []
        knowledge_rows: list[tuple[DocumentChunk, Document, float]] = []
        hierarchical_rows: list[tuple[DocumentChunk, Document, float]] = []
        seen_knowledge_chunk_ids: set[uuid.UUID] = set()
        executed_strategies: list[str] = []
        strategy_candidate_counts: dict[str, int] = {}
        stopping_strategy: str | None = None
        provenance_collector: dict[str, ChunkProvenance] = {}

        for strategy in route:
            executed_strategies.append(strategy.value)
            if strategy == RetrievalStrategy.GRAPH:
                if query_plan.relationship_query:
                    graph_rows = await self._relationship_search(
                        rel_query=query_plan.relationship_query,
                        allowed_departments=allowed_departments,
                        provenance_collector=provenance_collector,
                    )
                    for chunk, doc, score in graph_rows:
                        chunk_id_str = str(chunk.id)
                        c_time = _format_timestamp(getattr(chunk, "created_at", None))
                        if chunk_id_str not in provenance_collector:
                            provenance_collector[chunk_id_str] = ChunkProvenance(
                                chunk_id=chunk_id_str,
                                strategies=["GRAPH"],
                                chunk_created_at=c_time,
                            )
                        else:
                            if "GRAPH" not in provenance_collector[chunk_id_str].strategies:
                                provenance_collector[chunk_id_str].strategies.append("GRAPH")
                            if provenance_collector[chunk_id_str].chunk_created_at is None:
                                provenance_collector[chunk_id_str].chunk_created_at = c_time

                        if chunk.id not in seen_knowledge_chunk_ids:
                            seen_knowledge_chunk_ids.add(chunk.id)
                            knowledge_rows.append((chunk, doc, score))
                    strategy_candidate_counts[strategy.value] = len(graph_rows)

                    if not is_compound and self.controller.is_sufficient(len(knowledge_rows)):
                        stopping_strategy = strategy.value
                        logger.info(
                            "strategy_execution_sufficient",
                            extra={
                                "event": "strategy_execution_sufficient",
                                "strategy": strategy.value,
                                "candidates": len(knowledge_rows),
                                "planned_strategies": [s.value for s in route],
                                "executed_strategies": executed_strategies,
                                "stopping_strategy": stopping_strategy,
                                "candidate_count_per_strategy": strategy_candidate_counts,
                                "is_compound": is_compound,
                            },
                        )
                        break
                else:
                    strategy_candidate_counts[strategy.value] = 0

            elif strategy == RetrievalStrategy.KNOWLEDGE:
                search_term = self._extract_entity_search_term(query_plan)
                if search_term:
                    entity_rows = await self._entity_search(
                        search_term=search_term,
                        allowed_departments=allowed_departments,
                        provenance_collector=provenance_collector,
                    )
                    for chunk, doc, score in entity_rows:
                        chunk_id_str = str(chunk.id)
                        c_time = _format_timestamp(getattr(chunk, "created_at", None))
                        if chunk_id_str not in provenance_collector:
                            provenance_collector[chunk_id_str] = ChunkProvenance(
                                chunk_id=chunk_id_str,
                                strategies=["KNOWLEDGE"],
                                chunk_created_at=c_time,
                            )
                        else:
                            if "KNOWLEDGE" not in provenance_collector[chunk_id_str].strategies:
                                provenance_collector[chunk_id_str].strategies.append("KNOWLEDGE")
                            if provenance_collector[chunk_id_str].chunk_created_at is None:
                                provenance_collector[chunk_id_str].chunk_created_at = c_time

                        if chunk.id not in seen_knowledge_chunk_ids:
                            seen_knowledge_chunk_ids.add(chunk.id)
                            knowledge_rows.append((chunk, doc, score))
                    strategy_candidate_counts[strategy.value] = len(entity_rows)

                    if not is_compound and self.controller.is_sufficient(len(knowledge_rows)):
                        stopping_strategy = strategy.value
                        logger.info(
                            "strategy_execution_sufficient",
                            extra={
                                "event": "strategy_execution_sufficient",
                                "strategy": strategy.value,
                                "candidates": len(knowledge_rows),
                                "planned_strategies": [s.value for s in route],
                                "executed_strategies": executed_strategies,
                                "stopping_strategy": stopping_strategy,
                                "candidate_count_per_strategy": strategy_candidate_counts,
                                "is_compound": is_compound,
                            },
                        )
                        break
                else:
                    strategy_candidate_counts[strategy.value] = 0

            elif strategy == RetrievalStrategy.STRUCTURED:
                structured_rows = await self._structured_search(
                    filters=query_plan.filters,
                    candidate_k=settings.retrieval_candidate_k,
                    allowed_departments=allowed_departments,
                )
                rows_to_use = structured_rows if structured_rows else knowledge_rows
                strategy_candidate_counts[strategy.value] = len(rows_to_use)
                if self.controller.is_sufficient(len(rows_to_use)):
                    stopping_strategy = strategy.value
                    for chunk, doc, _ in rows_to_use:
                        chunk_id_str = str(chunk.id)
                        c_time = _format_timestamp(getattr(chunk, "created_at", None))
                        if chunk_id_str not in provenance_collector:
                            provenance_collector[chunk_id_str] = ChunkProvenance(
                                chunk_id=chunk_id_str,
                                strategies=["STRUCTURED"],
                                chunk_created_at=c_time,
                            )
                        else:
                            if "STRUCTURED" not in provenance_collector[chunk_id_str].strategies:
                                provenance_collector[chunk_id_str].strategies.append("STRUCTURED")
                            if provenance_collector[chunk_id_str].chunk_created_at is None:
                                provenance_collector[chunk_id_str].chunk_created_at = c_time

                    results = [
                        RetrievedChunk(
                            chunk_id=str(chunk.id),
                            document_id=str(document.id),
                            department_id=str(document.department_id),
                            file_name=document.file_name,
                            chunk_index=chunk.chunk_index,
                            text=chunk.text,
                            page_number=chunk.page_number,
                            node_id=str(chunk.node_id) if getattr(chunk, "node_id", None) else None,
                            hybrid_score=1.0,
                        )
                        for chunk, document, _ in rows_to_use
                    ]
                    selected = results[:top_k]
                    logger.info(
                        "structured_retrieval_completed",
                        extra={
                            "event": "structured_retrieval_completed",
                            "query": processed_query,
                            "results": len(selected),
                            "planned_strategies": [s.value for s in route],
                            "executed_strategies": executed_strategies,
                            "stopping_strategy": stopping_strategy,
                            "candidate_count_per_strategy": strategy_candidate_counts,
                            "is_compound": is_compound,
                        },
                    )
                    final_chunk_ids = {r.chunk_id for r in selected}
                    filtered_prov = {
                        cid: prov for cid, prov in provenance_collector.items()
                        if cid in final_chunk_ids
                    }
                    chunk_hierarchy = await self._resolve_chunk_hierarchies(
                        chunks=selected,
                        allowed_departments=allowed_departments,
                    )
                    return RetrievalResponse(
                        query=query,
                        results=selected,
                        strategy_trace=strategy_candidate_counts,
                        chunk_provenance=filtered_prov,
                        chunk_hierarchy=chunk_hierarchy,
                    )

            elif strategy == RetrievalStrategy.SEMANTIC:
                retrieval_query = (
                    query_plan.semantic_query
                    or processed_query
                )

                query_embedding = (
                    await self.embedding_provider.embed_query(
                        retrieval_query
                    )
                )

                _vec_start = time.perf_counter()
                vector_rows = (
                    await self._vector_search(
                        query_embedding=query_embedding,
                        candidate_k=settings.retrieval_candidate_k,
                        filters=(
                            query_plan.filters
                            if retrieval_mode == RetrievalMode.HYBRID
                            else None
                        ),
                        allowed_departments=allowed_departments,
                        allowed_documents=allowed_documents,
                        workspace_id=workspace_id,
                    )
                )
                record_sub_stage(
                    "retrieval",
                    "vector_search",
                    (time.perf_counter() - _vec_start) * 1000,
                )

                _kw_start = time.perf_counter()
                keyword_rows = (
                    await self._keyword_search(
                        query=retrieval_query,
                        candidate_k=settings.retrieval_candidate_k,
                        filters=(
                            query_plan.filters
                            if retrieval_mode == RetrievalMode.HYBRID
                            else None
                        ),
                        allowed_departments=allowed_departments,
                        allowed_documents=allowed_documents,
                        workspace_id=workspace_id,
                    )
                )
                record_sub_stage(
                    "retrieval",
                    "keyword_search",
                    (time.perf_counter() - _kw_start) * 1000,
                )
                for chunk, doc, score in vector_rows:
                    chunk_id_str = str(chunk.id)
                    c_time = _format_timestamp(getattr(chunk, "created_at", None))
                    if chunk_id_str not in provenance_collector:
                        provenance_collector[chunk_id_str] = ChunkProvenance(
                            chunk_id=chunk_id_str,
                            strategies=["SEMANTIC"],
                            chunk_created_at=c_time,
                        )
                    else:
                        if "SEMANTIC" not in provenance_collector[chunk_id_str].strategies:
                            provenance_collector[chunk_id_str].strategies.append("SEMANTIC")
                        if provenance_collector[chunk_id_str].chunk_created_at is None:
                            provenance_collector[chunk_id_str].chunk_created_at = c_time

                for chunk, doc, score in keyword_rows:
                    chunk_id_str = str(chunk.id)
                    c_time = _format_timestamp(getattr(chunk, "created_at", None))
                    if chunk_id_str not in provenance_collector:
                        provenance_collector[chunk_id_str] = ChunkProvenance(
                            chunk_id=chunk_id_str,
                            strategies=["SEMANTIC"],
                            chunk_created_at=c_time,
                        )
                    else:
                        if "SEMANTIC" not in provenance_collector[chunk_id_str].strategies:
                            provenance_collector[chunk_id_str].strategies.append("SEMANTIC")
                        if provenance_collector[chunk_id_str].chunk_created_at is None:
                            provenance_collector[chunk_id_str].chunk_created_at = c_time

                strategy_candidate_counts[strategy.value] = len(vector_rows) + len(keyword_rows)

            elif strategy == RetrievalStrategy.HIERARCHICAL:
                retrieval_query = (
                    query_plan.semantic_query
                    or processed_query
                )
                _hier_start = time.perf_counter()
                hierarchical_rows = await self._hierarchical_search(
                    query=retrieval_query,
                    candidate_k=settings.retrieval_candidate_k,
                    filters=(
                        query_plan.filters
                        if retrieval_mode == RetrievalMode.HYBRID
                        else None
                    ),
                    allowed_departments=allowed_departments,
                    allowed_documents=allowed_documents,
                    provenance_collector=provenance_collector,
                    workspace_id=workspace_id,
                )
                record_sub_stage(
                    "retrieval",
                    "hierarchical_retrieval",
                    (time.perf_counter() - _hier_start) * 1000,
                )
                strategy_candidate_counts[strategy.value] = len(hierarchical_rows)

        if stopping_strategy is None and executed_strategies:
            stopping_strategy = executed_strategies[-1]

        # =========================================================
        # 3. PURE STRUCTURED RETRIEVAL FALLBACK
        # =========================================================
        if retrieval_mode == RetrievalMode.STRUCTURED:
            structured_rows = (
                await self._structured_search(
                    filters=query_plan.filters,
                    candidate_k=settings.retrieval_candidate_k,
                    allowed_departments=allowed_departments,
                )
            )
            rows_to_use = structured_rows if structured_rows else knowledge_rows
            for chunk, document, _ in rows_to_use:
                chunk_id_str = str(chunk.id)
                c_time = _format_timestamp(getattr(chunk, "created_at", None))
                if chunk_id_str not in provenance_collector:
                    provenance_collector[chunk_id_str] = ChunkProvenance(
                        chunk_id=chunk_id_str,
                        strategies=["STRUCTURED"],
                        chunk_created_at=c_time,
                    )
                else:
                    if "STRUCTURED" not in provenance_collector[chunk_id_str].strategies:
                        provenance_collector[chunk_id_str].strategies.append("STRUCTURED")
                    if provenance_collector[chunk_id_str].chunk_created_at is None:
                        provenance_collector[chunk_id_str].chunk_created_at = c_time

            results = [
                RetrievedChunk(
                    chunk_id=str(chunk.id),
                    document_id=str(document.id),
                    department_id=str(document.department_id),
                    file_name=document.file_name,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    node_id=str(chunk.node_id) if getattr(chunk, "node_id", None) else None,
                    hybrid_score=1.0,
                )
                for chunk, document, _ in rows_to_use
            ]
            selected = results[:top_k]
            final_chunk_ids = {r.chunk_id for r in selected}
            filtered_prov = {
                cid: prov for cid, prov in provenance_collector.items()
                if cid in final_chunk_ids
            }
            chunk_hierarchy = await self._resolve_chunk_hierarchies(
                chunks=selected,
                allowed_departments=allowed_departments,
            )
            return RetrievalResponse(
                query=query,
                results=selected,
                strategy_trace=strategy_candidate_counts,
                chunk_provenance=filtered_prov,
                chunk_hierarchy=chunk_hierarchy,
            )

        # =========================================================
        # 4. RRF (Vector + Keyword + Knowledge + Hierarchical)
        # =========================================================
        candidates = self._merge_with_rrf(
            vector_rows=vector_rows,
            keyword_rows=keyword_rows,
            knowledge_rows=knowledge_rows,
            hierarchical_rows=hierarchical_rows,
        )

        # EXACT DUPLICATE SUPPRESSION (by chunk_id and text)
        unique_candidates = []
        seen_texts = set()
        seen_chunk_ids = set()
        for candidate in candidates:
            if candidate.chunk.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(candidate.chunk.id)
            normalized_text = " ".join(
                candidate.chunk.text.lower().split()
            )
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_candidates.append(candidate)
        candidates = unique_candidates

        logger.info(    
            "retrieval_candidates_before_rerank",
            extra={
                "event": "retrieval_candidates_before_rerank",
                "query": processed_query,
                "candidates": [
                {
                    "file_name": candidate.document.file_name,
                    "chunk_id": str(candidate.chunk.id),
                    "vector_similarity": candidate.vector_similarity,
                    "keyword_rank": candidate.keyword_rank,
                    "vector_position": candidate.vector_position,
                    "keyword_position": candidate.keyword_position,
                    "rrf_score": candidate.hybrid_score,
                    "text_preview": candidate.chunk.text[:200],
                }
                for candidate in candidates
            ],
        },
    )
        #=========================================================
        # CROSS-ENCODER RERANKING
        #=========================================================
        retrieval_query = (
            query_plan.semantic_query
            or processed_query
        )
        
        if rerank and candidates:
            candidate_texts = [
                c.chunk.metadata_['embedding_text']
                if c.chunk.metadata_ and 'embedding_text' in c.chunk.metadata_
                else (f"{c.chunk.metadata_.get('search_context')}\n\n{c.chunk.text}"
                      if c.chunk.metadata_ and c.chunk.metadata_.get('search_context')
                      else c.chunk.text)
                for c in candidates
            ]
            _rerank_start = time.perf_counter()
            rerank_scores = await self.reranker_provider.rerank(
                query=retrieval_query,
                texts=candidate_texts,
            )
            record_sub_stage(
                "retrieval",
                "reranking",
                (time.perf_counter() - _rerank_start) * 1000,
            )
            # Store reranker score separately, preserving RRF hybrid_score
            for candidate, score in zip(candidates, rerank_scores):
                candidate.reranker_score = float(score)

        #=========================================================
        # RELEVANCE FILTER
        #=========================================================

        relevant_candidates = [
            candidate
            for candidate in candidates
            if self._is_relevant(candidate)
        ]

        relevant_candidates.sort(
            key=lambda item: (
                item.reranker_score if item.reranker_score is not None else item.hybrid_score
            ),
            reverse=True,
        )
        for candidate in relevant_candidates:
            logger.info(
                "post_rerank_candidate",
                extra={
            "file_name": candidate.document.file_name,
            "reranker_score": candidate.reranker_score,
            "hybrid_score": candidate.hybrid_score,
            "vector_similarity": candidate.vector_similarity,
            "keyword_rank": candidate.keyword_rank,
        },
    )
        selected = []
        seen_texts = set()
        document_counts = {}

        MAX_CHUNKS_PER_DOCUMENT = 4

        for candidate in relevant_candidates:
            normalized_text = " ".join(
                candidate.chunk.text.lower().split()
            )

            document_id = str(candidate.document.id)

            if normalized_text in seen_texts:
                continue

            if document_counts.get(document_id, 0) >= MAX_CHUNKS_PER_DOCUMENT:
                continue

            seen_texts.add(normalized_text)
            document_counts[document_id] = (
                document_counts.get(document_id, 0) + 1
            )

            selected.append(candidate)

            if len(selected) >= top_k:
                break

        # =========================================================
        # 8. RESPONSE
        # =========================================================

        results = [
            RetrievedChunk(
                chunk_id=str(
                    candidate.chunk.id
                ),
                document_id=str(
                    candidate.document.id
                ),
                department_id=str(
                    candidate.document.department_id
                ),
                file_name=(
                    candidate.document.file_name
                ),
                chunk_index=(
                    candidate.chunk.chunk_index
                ),
                text=candidate.chunk.text,
                page_number=(
                    candidate.chunk.page_number
                ),
                distance=(
                    candidate.distance
                ),
                vector_similarity=(
                    candidate.vector_similarity
                ),
                keyword_rank=(
                    candidate.keyword_rank
                ),
                hybrid_score=(
                    candidate.hybrid_score
                ),
                reranker_score=(
                    candidate.reranker_score
                ),
                hierarchical_score=(
                    candidate.hierarchical_score
                ),
                hierarchical_position=(
                    candidate.hierarchical_position
                ),
                node_id=(
                    str(candidate.chunk.node_id)
                    if getattr(candidate.chunk, "node_id", None)
                    else None
                ),
            )
            for candidate in selected
        ]

        logger.info(
            "hybrid_retrieval_completed",
            extra={
                "event": (
                    "hybrid_retrieval_completed"
                ),
                "query": processed_query,
                "mode": retrieval_mode.value,
                "planned_strategies": [s.value for s in route],
                "executed_strategies": executed_strategies,
                "stopping_strategy": stopping_strategy,
                "vector_candidates": (
                    len(vector_rows)
                ),
                "keyword_candidates": (
                    len(keyword_rows)
                ),
                "knowledge_candidates": (
                    len(knowledge_rows)
                ),
                "candidate_count_per_strategy": strategy_candidate_counts,
                "is_compound": is_compound,
                "accepted_results": (
                    len(results)
                ),
            },
        )

        final_chunk_ids = {r.chunk_id for r in results}
        filtered_prov = {
            cid: prov for cid, prov in provenance_collector.items()
            if cid in final_chunk_ids
        }
        chunk_hierarchy = await self._resolve_chunk_hierarchies(
            chunks=results,
            allowed_departments=allowed_departments,
        )
        return RetrievalResponse(
            query=query,
            results=results,
            strategy_trace=strategy_candidate_counts,
            chunk_provenance=filtered_prov,
            chunk_hierarchy=chunk_hierarchy,
        )

    # =============================================================
    # STRUCTURED FILTER CONDITIONS
    # =============================================================

    def _structured_conditions(
        self,
        filters: CompanyFilters,
    ):

        conditions = []

        if filters.company_id:

            company_id = (
                normalize_company_id(
                    filters.company_id
                )
            )

            conditions.append(
                DocumentChunk.metadata_[
                    "company_id"
                ].astext == company_id
            )

        if filters.company_name_contains:
            conditions.append(
                DocumentChunk.metadata_[
                    "company_name"
                ].astext.ilike(
                    f"%{filters.company_name_contains}%"
                )
            )

        if filters.funding_status:
            conditions.append(
                DocumentChunk.metadata_[
                    "funding_status"
                ].astext == filters.funding_status
            )

        if filters.city:
            conditions.append(
                DocumentChunk.metadata_[
                    "city"
                ].astext.ilike(
                    filters.city
                )
            )

        if filters.country:
            conditions.append(
                DocumentChunk.metadata_[
                    "country"
                ].astext.ilike(
                    filters.country
                )
            )

        return conditions

    # =============================================================
    # VECTOR SEARCH
    # =============================================================

    @track(name="vector_search", phase="retrieval")
    async def _vector_search(
        self,
        query_embedding: list[float],
        candidate_k: int,
        filters: CompanyFilters | None = None,
        allowed_departments: list[uuid.UUID] | None = None,
        allowed_documents: list[uuid.UUID] | None = None,
        workspace_id: uuid.UUID | None = None,
    ):

        distance_expression = (
            DocumentChunk.embedding
            .cosine_distance(
                query_embedding
            )
            .label("distance")
        )

        statement = (
            select(
                DocumentChunk,
                Document,
                distance_expression,
            )
            .join(
                Document,
                Document.id
                == DocumentChunk.document_id,
            )
        )

        # HARD FILTER
        if filters:

            conditions = (
                self._structured_conditions(
                    filters
                )
            )

            if conditions:
                statement = statement.where(
                    *conditions
                )

        if allowed_departments is not None or allowed_documents is not None:
            from sqlalchemy import or_
            doc_conds = []
            if allowed_departments is not None:
                doc_conds.append(Document.department_id.in_(allowed_departments))
                doc_conds.append(Document.department_id.is_(None))
            if allowed_documents:
                doc_conds.append(Document.id.in_(allowed_documents))
            if doc_conds:
                statement = statement.where(or_(*doc_conds))

        if workspace_id is not None:
            statement = statement.where(Document.workspace_id == workspace_id)

        statement = (
            statement
            .order_by(
                distance_expression
            )
            .limit(candidate_k)
        )

        rows = await self.session.execute(
            statement
        )

        results = [
            (
                chunk,
                document,
                float(distance),
            )
            for chunk, document, distance
            in rows.all()
        ]

        for chunk, document, distance in results:
            logger.info(
                "vector_candidate_debug",   
                extra={
                    "file_name": document.file_name,
                    "distance": distance,
                },
            )

        return results

    # =============================================================
    # KEYWORD SEARCH
    # =============================================================
    @track(name="keyword_search", phase="retrieval")
    async def _keyword_search(
        self,
        query: str,
        candidate_k: int,
        filters: CompanyFilters | None = None,
        allowed_departments: list[uuid.UUID] | None = None,
        allowed_documents: list[uuid.UUID] | None = None,
        workspace_id: uuid.UUID | None = None,
    ):
        logger.info(
            "keyword_search_debug",
            extra={
                "event": "keyword_search_debug",
                "query": repr(query),
                "candidate_k": candidate_k,
                "filters": (
                    filters.model_dump()
                    if filters
                    else None
                ),
            },
        )
        
        search_index = GlobalBM25Index.get_instance()
        chunk_scores = await search_index.get_scores(query)
        
        if not chunk_scores:
            return []
            
        # Get top IDs with a generous limit in case some are filtered out later
        max_chunks_to_fetch = candidate_k * 5
        best_chunk_ids = sorted(
            chunk_scores.keys(), 
            key=lambda chunk_id: chunk_scores[chunk_id], 
            reverse=True
        )[:max_chunks_to_fetch]
        
        if not best_chunk_ids:
            return []
            
        statement = (
            select(
                DocumentChunk,
                Document,
            )
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
            .where(
                DocumentChunk.id.in_(best_chunk_ids)
            )
        )

        # HARD FILTER
        if filters:
            conditions = self._structured_conditions(filters)
            if conditions:
                statement = statement.where(*conditions)
                
        if allowed_departments is not None or allowed_documents is not None:
            from sqlalchemy import or_
            doc_conds = []
            if allowed_departments is not None:
                doc_conds.append(Document.department_id.in_(allowed_departments))
                doc_conds.append(Document.department_id.is_(None))
            if allowed_documents:
                doc_conds.append(Document.id.in_(allowed_documents))
            if doc_conds:
                statement = statement.where(or_(*doc_conds))

        if workspace_id is not None:
            statement = statement.where(Document.workspace_id == workspace_id)


        rows = await self.session.execute(statement)

        candidates = []
        seen_chunk_ids = set()

        for chunk, document in rows.all():
            if chunk.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.id)
            
            final_score = chunk_scores[chunk.id]
            candidates.append((chunk, document, final_score))

        # Sort by best score first
        candidates.sort(key=lambda item: item[2], reverse=True)
        final_candidates = candidates[:candidate_k]

        logger.info(
            "keyword_search_completed",
            extra={
                "event": "keyword_search_completed",
                "query": query,
                "final_candidates": len(final_candidates),
            },
        )

        return final_candidates

    @track(name="hierarchical_search", phase="retrieval")
    async def _hierarchical_search(
        self,
        query: str,
        candidate_k: int,
        filters: CompanyFilters | None = None,
        allowed_departments: list[uuid.UUID] | None = None,
        allowed_documents: list[uuid.UUID] | None = None,
        provenance_collector: dict[str, ChunkProvenance] | None = None,
        max_descendant_depth: int | None = None,
        workspace_id: uuid.UUID | None = None,
    ) -> list[tuple[DocumentChunk, Document, float]]:
        """
        OV-2 Native Hierarchical Retrieval:
        1. L0 FTS retrieval on DocumentSection.title + l0_abstract with deterministic tie-breaking.
        2. Bounded descendant traversal (L0 -> L1 descendant nodes) up to max_descendant_depth.
        3. L2 chunk expansion via DocumentChunk.node_id with full authorization enforcement.
        4. Preserves existing chunk models, deterministic scoring, provenance, and auth boundaries.
        """
        if not query or not query.strip():
            return []

        if max_descendant_depth is None:
            max_descendant_depth = settings.hierarchical_max_descendant_depth

        # -------------------------------------------------------------
        # 1. L0 Section Search via PostgreSQL FTS
        # -------------------------------------------------------------
        text_target = (
            func.coalesce(DocumentSection.title, "")
            + " "
            + func.coalesce(DocumentSection.l0_abstract, "")
        )
        ts_vector = func.to_tsvector("english", text_target)
        ts_query = func.plainto_tsquery("english", query)
        fts_score = func.ts_rank_cd(ts_vector, ts_query).label("fts_score")

        sec_stmt = (
            select(
                DocumentSection,
                Document,
                fts_score,
            )
            .join(
                Document,
                Document.id == DocumentSection.document_id,
            )
            .where(
                ts_vector.op("@@")(ts_query)
            )
            .where(
                fts_score > 0
            )
        )

        if filters:
            conditions = self._structured_conditions(filters)
            if conditions:
                sec_stmt = sec_stmt.where(*conditions)

        if allowed_departments is not None or allowed_documents is not None:
            from sqlalchemy import or_
            doc_conds = []
            if allowed_departments is not None:
                doc_conds.append(Document.department_id.in_(allowed_departments))
                doc_conds.append(Document.department_id.is_(None))
            if allowed_documents:
                doc_conds.append(Document.id.in_(allowed_documents))
            if doc_conds:
                sec_stmt = sec_stmt.where(or_(*doc_conds))

        if workspace_id is not None:
            sec_stmt = sec_stmt.where(Document.workspace_id == workspace_id)


        sec_stmt = (
            sec_stmt
            .order_by(
                fts_score.desc(),
                DocumentSection.level.asc(),
                DocumentSection.ordinal_path.asc(),
                DocumentSection.id.asc(),
            )
            .limit(candidate_k)
        )

        try:
            exec_res = await self.session.execute(sec_stmt)
            sec_rows = exec_res.all() if hasattr(exec_res, "all") else []
            if hasattr(sec_rows, "__await__"):
                sec_rows = await sec_rows
            if not isinstance(sec_rows, (list, tuple)):
                sec_rows = []
        except Exception as e:
            logger.warning("hierarchical_fts_query_failed", extra={"error": str(e), "query": query})
            return []

        if not sec_rows:
            return []

        # -------------------------------------------------------------
        # 2. Bounded Descendant Traversal (L0 -> L1 descendants)
        # -------------------------------------------------------------
        matched_sections: list[DocumentSection] = []
        section_score_map: dict[uuid.UUID, float] = {}
        section_doc_map: dict[uuid.UUID, Document] = {}
        all_expanded_section_ids: set[uuid.UUID] = set()

        current_level_ids: list[uuid.UUID] = []
        for row in sec_rows:
            try:
                sec, doc, score = row[0], row[1], row[2]
            except Exception:
                continue
            matched_sections.append(sec)
            score_val = float(score)
            section_score_map[sec.id] = score_val
            section_doc_map[sec.id] = doc
            all_expanded_section_ids.add(sec.id)
            current_level_ids.append(sec.id)

        if not matched_sections:
            return []

        # Bounded BFS descendant traversal
        for _ in range(max_descendant_depth):
            if not current_level_ids:
                break
            child_stmt = (
                select(DocumentSection, Document)
                .join(Document, Document.id == DocumentSection.document_id)
                .where(DocumentSection.parent_id.in_(current_level_ids))
            )
            if allowed_departments is not None or allowed_documents is not None:
                from sqlalchemy import or_
                doc_conds = []
                if allowed_departments is not None:
                    doc_conds.append(Document.department_id.in_(allowed_departments))
                    doc_conds.append(Document.department_id.is_(None))
                if allowed_documents:
                    doc_conds.append(Document.id.in_(allowed_documents))
                if doc_conds:
                    child_stmt = child_stmt.where(or_(*doc_conds))
            if workspace_id is not None:
                child_stmt = child_stmt.where(Document.workspace_id == workspace_id)
            try:
                exec_child = await self.session.execute(child_stmt)
                child_rows = exec_child.all() if hasattr(exec_child, "all") else []
                if hasattr(child_rows, "__await__"):
                    child_rows = await child_rows
            except Exception:
                child_rows = []

            next_level_ids: list[uuid.UUID] = []
            for child_row in child_rows:
                try:
                    child_sec, doc = child_row[0], child_row[1]
                except Exception:
                    continue
                if child_sec.id not in all_expanded_section_ids:
                    all_expanded_section_ids.add(child_sec.id)
                    next_level_ids.append(child_sec.id)
                    parent_score = section_score_map.get(child_sec.parent_id, 0.0)
                    section_score_map[child_sec.id] = parent_score
                    section_doc_map[child_sec.id] = doc
            current_level_ids = next_level_ids

        # -------------------------------------------------------------
        # 3. L2 Chunk Expansion via DocumentChunk.node_id
        # -------------------------------------------------------------
        chunk_stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.node_id.in_(all_expanded_section_ids))
        )
        if allowed_departments is not None or allowed_documents is not None:
            from sqlalchemy import or_
            doc_conds = []
            if allowed_departments is not None:
                doc_conds.append(Document.department_id.in_(allowed_departments))
                doc_conds.append(Document.department_id.is_(None))
            if allowed_documents:
                doc_conds.append(Document.id.in_(allowed_documents))
            if doc_conds:
                chunk_stmt = chunk_stmt.where(or_(*doc_conds))
        if workspace_id is not None:
            chunk_stmt = chunk_stmt.where(Document.workspace_id == workspace_id)
        if filters:
            conditions = self._structured_conditions(filters)
            if conditions:
                chunk_stmt = chunk_stmt.where(*conditions)

        chunk_stmt = chunk_stmt.order_by(
            DocumentChunk.document_id.asc(),
            DocumentChunk.chunk_index.asc(),
        )

        try:
            exec_chunk = await self.session.execute(chunk_stmt)
            chunk_rows = exec_chunk.all() if hasattr(exec_chunk, "all") else []
            if hasattr(chunk_rows, "__await__"):
                chunk_rows = await chunk_rows
        except Exception:
            chunk_rows = []

        results: list[tuple[DocumentChunk, Document, float]] = []
        seen_chunk_ids: set[uuid.UUID] = set()

        for chunk_row in chunk_rows:
            try:
                chunk, doc = chunk_row[0], chunk_row[1]
            except Exception:
                continue
            if chunk.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.id)

            h_score = section_score_map.get(chunk.node_id, 0.0)
            if h_score <= 0:
                continue

            results.append((chunk, doc, h_score))

            if provenance_collector is not None:
                chunk_id_str = str(chunk.id)
                c_time = _format_timestamp(getattr(chunk, "created_at", None))
                if chunk_id_str not in provenance_collector:
                    provenance_collector[chunk_id_str] = ChunkProvenance(
                        chunk_id=chunk_id_str,
                        strategies=["HIERARCHICAL"],
                        chunk_created_at=c_time,
                    )
                else:
                    if "HIERARCHICAL" not in provenance_collector[chunk_id_str].strategies:
                        provenance_collector[chunk_id_str].strategies.append("HIERARCHICAL")
                    if provenance_collector[chunk_id_str].chunk_created_at is None:
                        provenance_collector[chunk_id_str].chunk_created_at = c_time

        # Sort candidate chunks deterministically: score DESC, chunk_index ASC, id ASC
        results.sort(key=lambda x: (-x[2], x[0].chunk_index, str(x[0].id)))

        logger.info(
            "hierarchical_retrieval_completed",
            extra={
                "event": "hierarchical_retrieval_completed",
                "matched_sections_count": len(matched_sections),
                "expanded_nodes_count": len(all_expanded_section_ids),
                "retrieved_chunks_count": len(results),
            },
        )

        return results[:candidate_k]

    def _extract_entity_search_term(self, query_plan: QueryPlan) -> str | None:
        """
        Extract an entity name lookup term from the query plan if applicable.
        Skips generic procedural, conceptual, or long questions.
        """
        if query_plan.entity_search_term is not None:
            return query_plan.entity_search_term
        return extract_entity_search_term(query_plan)

    @track(name="knowledge_search", phase="retrieval")
    async def _knowledge_search(
        self,
        query_plan: QueryPlan,
        allowed_departments: list[uuid.UUID] | None = None,
        provenance_collector: dict[str, ChunkProvenance] | None = None,
    ) -> list[tuple[DocumentChunk, Document, float]]:
        """
        Perform a knowledge lookup based on entity name matching and/or relationship traversal.
        Returns document chunks linked via KnowledgeEvidence.
        """
        results: list[tuple[DocumentChunk, Document, float]] = []
        seen_chunk_ids: set[uuid.UUID] = set()

        # 1. Relationship traversal if relationship_query is present
        if query_plan.relationship_query:
            rel_results = await self._relationship_search(
                rel_query=query_plan.relationship_query,
                allowed_departments=allowed_departments,
                provenance_collector=provenance_collector,
            )
            for chunk, doc, score in rel_results:
                if chunk.id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk.id)
                    results.append((chunk, doc, score))

        # 2. Entity-based lookup
        search_term = self._extract_entity_search_term(query_plan)
        if search_term:
            entity_results = await self._entity_search(
                search_term=search_term,
                allowed_departments=allowed_departments,
                provenance_collector=provenance_collector,
            )
            for chunk, doc, score in entity_results:
                if chunk.id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk.id)
                    results.append((chunk, doc, score))

        logger.info(
            "knowledge_search_completed",
            extra={
                "event": "knowledge_search_completed",
                "has_relationship_query": query_plan.relationship_query is not None,
                "search_term": search_term,
                "candidates": len(results),
            },
        )
        return results

    async def _entity_search(
        self,
        search_term: str,
        allowed_departments: list[uuid.UUID] | None = None,
        provenance_collector: dict[str, ChunkProvenance] | None = None,
    ) -> list[tuple[DocumentChunk, Document, float]]:
        """
        Match entities in KnowledgeEntity and resolve to DocumentChunks via KnowledgeEvidence.
        Filters KnowledgeItem.status == 'ACTIVE'.
        """
        from sqlalchemy import or_

        stmt = (
            select(
                DocumentChunk,
                Document,
                func.coalesce(1.0, literal_column("1.0")).label("score"),
                KnowledgeItem.id.label("knowledge_item_id"),
                KnowledgeItem.confidence.label("confidence"),
                KnowledgeEntity.canonical_name.label("entity_name"),
                KnowledgeEntity.entity_type.label("entity_type"),
                DocumentChunk.created_at.label("chunk_created_at"),
                KnowledgeItem.updated_at.label("item_updated_at"),
            )
            .join(KnowledgeEvidence, KnowledgeEvidence.document_chunk_id == DocumentChunk.id)
            .join(KnowledgeItem, KnowledgeItem.id == KnowledgeEvidence.knowledge_item_id)
            .join(KnowledgeEntity, KnowledgeEntity.knowledge_item_id == KnowledgeItem.id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                KnowledgeItem.status == "ACTIVE",
                or_(
                    KnowledgeEntity.canonical_name.ilike(f"%{search_term}%"),
                    KnowledgeEntity.entity_type.ilike(f"%{search_term}%"),
                )
            )
        )

        if allowed_departments is not None:
            stmt = stmt.where(
                or_(
                    KnowledgeItem.department_id.in_(allowed_departments),
                    KnowledgeItem.department_id.is_(None),
                ),
                or_(
                    Document.department_id.in_(allowed_departments),
                    Document.department_id.is_(None),
                ),
            )

        stmt = stmt.order_by(
            KnowledgeItem.confidence.desc().nullslast(),
            DocumentChunk.chunk_index.asc(),
            DocumentChunk.id.asc(),
        )

        rows = await self.session.execute(stmt)
        seen_chunk_ids = set()
        entity_results = []
        for row in rows.all():
            chunk = row[0]
            doc = row[1]
            score = row[2]
            k_item_id = row[3] if len(row) > 3 else None
            conf = row[4] if len(row) > 4 else None
            ent_name = row[5] if len(row) > 5 else None
            ent_type = row[6] if len(row) > 6 else None
            chunk_created_at = row[7] if len(row) > 7 else getattr(chunk, "created_at", None)
            item_updated_at = row[8] if len(row) > 8 else None

            if chunk.id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.id)
                entity_results.append((chunk, doc, float(score)))

            if provenance_collector is not None:
                chunk_id_str = str(chunk.id)
                prov = provenance_collector.get(chunk_id_str)
                if not prov:
                    prov = ChunkProvenance(chunk_id=chunk_id_str, strategies=[])
                    provenance_collector[chunk_id_str] = prov
                if "KNOWLEDGE" not in prov.strategies:
                    prov.strategies.append("KNOWLEDGE")
                if prov.knowledge_item_id is None and k_item_id is not None:
                    prov.knowledge_item_id = str(k_item_id)
                if prov.confidence is None and conf is not None:
                    prov.confidence = float(conf)
                if prov.entity_name is None and ent_name is not None:
                    prov.entity_name = str(ent_name)
                if prov.entity_type is None and ent_type is not None:
                    prov.entity_type = str(ent_type)
                if prov.chunk_created_at is None and chunk_created_at is not None:
                    prov.chunk_created_at = _format_timestamp(chunk_created_at)
                if prov.item_updated_at is None and item_updated_at is not None:
                    prov.item_updated_at = _format_timestamp(item_updated_at)

        return entity_results

    async def _relationship_search(
        self,
        rel_query: RelationshipQuery,
        allowed_departments: list[uuid.UUID] | None = None,
        provenance_collector: dict[str, ChunkProvenance] | None = None,
    ) -> list[tuple[DocumentChunk, Document, float]]:
        """
        Perform 1-hop or 2-hop PostgreSQL graph traversal for relationship queries.
        Pre-filters authorization on KnowledgeItem and Document before returning chunks.
        Filters KnowledgeItem.status == 'ACTIVE'.
        """
        from sqlalchemy.orm import aliased
        from sqlalchemy import or_

        # -------------------------------------------------------------
        # 2-HOP TRAVERSAL
        # -------------------------------------------------------------
        if rel_query.hop_limit == 2:
            Subj1 = aliased(KnowledgeEntity, name="subj1")
            Middle = aliased(KnowledgeEntity, name="middle")
            Obj2 = aliased(KnowledgeEntity, name="obj2")

            Rel1 = aliased(KnowledgeRelationship, name="rel1")
            Rel2 = aliased(KnowledgeRelationship, name="rel2")

            Item1 = aliased(KnowledgeItem, name="item1")
            Item2 = aliased(KnowledgeItem, name="item2")

            Evidence = aliased(KnowledgeEvidence, name="evidence")

            stmt_2hop = (
                select(
                    DocumentChunk,
                    Document,
                    func.coalesce(1.0, literal_column("1.0")).label("score"),
                    Item2.id.label("knowledge_item_id"),
                    Item2.confidence.label("confidence"),
                    Subj1.canonical_name.label("subj1_name"),
                    Rel1.predicate.label("rel1_pred"),
                    Middle.canonical_name.label("middle_name"),
                    Rel2.predicate.label("rel2_pred"),
                    Obj2.canonical_name.label("obj2_name"),
                    DocumentChunk.created_at.label("chunk_created_at"),
                    Item2.updated_at.label("item_updated_at"),
                )
                .join(Evidence, Evidence.document_chunk_id == DocumentChunk.id)
                .join(Item2, Item2.id == Evidence.knowledge_item_id)
                .join(Rel2, Rel2.knowledge_item_id == Item2.id)
                .join(Middle, Middle.id == Rel2.subject_entity_id)
                .join(Rel1, Rel1.object_entity_id == Middle.id)
                .join(Item1, Item1.id == Rel1.knowledge_item_id)
                .join(Subj1, Subj1.id == Rel1.subject_entity_id)
                .join(Obj2, Obj2.id == Rel2.object_entity_id)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    Item1.status == "ACTIVE",
                    Item2.status == "ACTIVE",
                )
            )

            # Strict pre-filter authorization
            if allowed_departments is not None:
                stmt_2hop = stmt_2hop.where(
                    or_(Item1.department_id.in_(allowed_departments), Item1.department_id.is_(None)),
                    or_(Item2.department_id.in_(allowed_departments), Item2.department_id.is_(None)),
                    or_(Document.department_id.in_(allowed_departments), Document.department_id.is_(None)),
                )

            if rel_query.subject_entity:
                stmt_2hop = stmt_2hop.where(Subj1.canonical_name.ilike(f"%{rel_query.subject_entity.strip()}%"))
            if rel_query.object_entity:
                stmt_2hop = stmt_2hop.where(Obj2.canonical_name.ilike(f"%{rel_query.object_entity.strip()}%"))
            if rel_query.predicate:
                norm_pred = normalize_string(rel_query.predicate).replace("-", "_").replace(" ", "_")
                stmt_2hop = stmt_2hop.where(
                    or_(
                        Rel1.predicate == norm_pred,
                        Rel1.predicate.ilike(f"%{norm_pred}%"),
                        Rel2.predicate == norm_pred,
                        Rel2.predicate.ilike(f"%{norm_pred}%"),
                    )
                )

            stmt_2hop = stmt_2hop.order_by(
                Item2.confidence.desc().nullslast(),
                Item1.confidence.desc().nullslast(),
                DocumentChunk.chunk_index.asc(),
                DocumentChunk.id.asc(),
            )

            rows_2hop = await self.session.execute(stmt_2hop)
            seen_chunk_ids = set()
            results = []
            for row in rows_2hop.all():
                chunk = row[0]
                doc = row[1]
                score = row[2]
                k_item_id = row[3] if len(row) > 3 else None
                conf = row[4] if len(row) > 4 else None
                subj1 = row[5] if len(row) > 5 else None
                rel1 = row[6] if len(row) > 6 else None
                middle = row[7] if len(row) > 7 else None
                rel2 = row[8] if len(row) > 8 else None
                obj2 = row[9] if len(row) > 9 else None
                chunk_created_at = row[10] if len(row) > 10 else getattr(chunk, "created_at", None)
                item_updated_at = row[11] if len(row) > 11 else None

                if chunk.id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk.id)
                    results.append((chunk, doc, float(score)))

                if provenance_collector is not None:
                    chunk_id_str = str(chunk.id)
                    prov = provenance_collector.get(chunk_id_str)
                    if not prov:
                        prov = ChunkProvenance(chunk_id=chunk_id_str, strategies=[])
                        provenance_collector[chunk_id_str] = prov
                    if "GRAPH" not in prov.strategies:
                        prov.strategies.append("GRAPH")
                    if prov.knowledge_item_id is None and k_item_id is not None:
                        prov.knowledge_item_id = str(k_item_id)
                    if prov.confidence is None and conf is not None:
                        prov.confidence = float(conf)
                    if prov.entity_name is None and subj1 is not None:
                        prov.entity_name = str(subj1)
                    if prov.chunk_created_at is None and chunk_created_at is not None:
                        prov.chunk_created_at = _format_timestamp(chunk_created_at)
                    if prov.item_updated_at is None and item_updated_at is not None:
                        prov.item_updated_at = _format_timestamp(item_updated_at)
                    if not prov.graph_path and (subj1 or rel1 or middle or rel2 or obj2):
                        edge1 = GraphEdgeProvenance(
                            subject=str(subj1 or ""),
                            predicate=str(rel1 or ""),
                            object=str(middle or ""),
                            hop=1,
                        )
                        edge2 = GraphEdgeProvenance(
                            subject=str(middle or ""),
                            predicate=str(rel2 or ""),
                            object=str(obj2 or ""),
                            hop=2,
                        )
                        prov.graph_path = [edge1, edge2]

            if results:
                return results

        # -------------------------------------------------------------
        # 1-HOP TRAVERSAL
        # -------------------------------------------------------------
        Subj = aliased(KnowledgeEntity, name="subj_ent")
        Obj = aliased(KnowledgeEntity, name="obj_ent")

        stmt = (
            select(
                DocumentChunk,
                Document,
                func.coalesce(1.0, literal_column("1.0")).label("score"),
                KnowledgeItem.id.label("knowledge_item_id"),
                KnowledgeItem.confidence.label("confidence"),
                Subj.canonical_name.label("subj_name"),
                KnowledgeRelationship.predicate.label("predicate"),
                Obj.canonical_name.label("obj_name"),
                DocumentChunk.created_at.label("chunk_created_at"),
                KnowledgeItem.updated_at.label("item_updated_at"),
            )
            .join(KnowledgeEvidence, KnowledgeEvidence.document_chunk_id == DocumentChunk.id)
            .join(KnowledgeItem, KnowledgeItem.id == KnowledgeEvidence.knowledge_item_id)
            .join(KnowledgeRelationship, KnowledgeRelationship.knowledge_item_id == KnowledgeItem.id)
            .join(Subj, Subj.id == KnowledgeRelationship.subject_entity_id)
            .join(Obj, Obj.id == KnowledgeRelationship.object_entity_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                KnowledgeItem.status == "ACTIVE"
            )
        )

        # Strict pre-filter authorization
        if allowed_departments is not None:
            stmt = stmt.where(
                or_(
                    KnowledgeItem.department_id.in_(allowed_departments),
                    KnowledgeItem.department_id.is_(None),
                ),
                or_(
                    Document.department_id.in_(allowed_departments),
                    Document.department_id.is_(None),
                ),
            )

        # Predicate filtering
        if rel_query.predicate:
            norm_pred = normalize_string(rel_query.predicate).replace("-", "_").replace(" ", "_")
            stmt = stmt.where(
                or_(
                    KnowledgeRelationship.predicate == norm_pred,
                    KnowledgeRelationship.predicate.ilike(f"%{norm_pred}%"),
                )
            )

        # Subject / Object / Direction resolution
        subj_name = rel_query.subject_entity.strip() if rel_query.subject_entity else None
        obj_name = rel_query.object_entity.strip() if rel_query.object_entity else None
        direction = rel_query.direction or "out"

        if subj_name and obj_name:
            stmt = stmt.where(
                Subj.canonical_name.ilike(f"%{subj_name}%"),
                Obj.canonical_name.ilike(f"%{obj_name}%"),
            )
        elif subj_name and not obj_name:
            if direction == "in":
                stmt = stmt.where(Obj.canonical_name.ilike(f"%{subj_name}%"))
            elif direction == "both":
                stmt = stmt.where(
                    or_(
                        Subj.canonical_name.ilike(f"%{subj_name}%"),
                        Obj.canonical_name.ilike(f"%{subj_name}%"),
                    )
                )
            else:  # "out"
                stmt = stmt.where(Subj.canonical_name.ilike(f"%{subj_name}%"))
        elif obj_name and not subj_name:
            if direction == "out":
                stmt = stmt.where(Subj.canonical_name.ilike(f"%{obj_name}%"))
            elif direction == "both":
                stmt = stmt.where(
                    or_(
                        Subj.canonical_name.ilike(f"%{obj_name}%"),
                        Obj.canonical_name.ilike(f"%{obj_name}%"),
                    )
                )
            else:  # "in"
                stmt = stmt.where(Obj.canonical_name.ilike(f"%{obj_name}%"))

        stmt = stmt.order_by(
            KnowledgeItem.confidence.desc().nullslast(),
            DocumentChunk.chunk_index.asc(),
            DocumentChunk.id.asc(),
        )

        rows = await self.session.execute(stmt)
        seen_chunk_ids = set()
        results = []
        for row in rows.all():
            chunk = row[0]
            doc = row[1]
            score = row[2]
            k_item_id = row[3] if len(row) > 3 else None
            conf = row[4] if len(row) > 4 else None
            subj = row[5] if len(row) > 5 else None
            pred = row[6] if len(row) > 6 else None
            obj = row[7] if len(row) > 7 else None
            chunk_created_at = row[8] if len(row) > 8 else getattr(chunk, "created_at", None)
            item_updated_at = row[9] if len(row) > 9 else None

            if chunk.id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.id)
                results.append((chunk, doc, float(score)))

            if provenance_collector is not None:
                chunk_id_str = str(chunk.id)
                prov = provenance_collector.get(chunk_id_str)
                if not prov:
                    prov = ChunkProvenance(chunk_id=chunk_id_str, strategies=[])
                    provenance_collector[chunk_id_str] = prov
                if "GRAPH" not in prov.strategies:
                    prov.strategies.append("GRAPH")
                if prov.knowledge_item_id is None and k_item_id is not None:
                    prov.knowledge_item_id = str(k_item_id)
                if prov.confidence is None and conf is not None:
                    prov.confidence = float(conf)
                if prov.entity_name is None and subj is not None:
                    prov.entity_name = str(subj)
                if prov.chunk_created_at is None and chunk_created_at is not None:
                    prov.chunk_created_at = _format_timestamp(chunk_created_at)
                if prov.item_updated_at is None and item_updated_at is not None:
                    prov.item_updated_at = _format_timestamp(item_updated_at)
                if not prov.graph_path and (subj or pred or obj):
                    edge1 = GraphEdgeProvenance(
                        subject=str(subj or ""),
                        predicate=str(pred or ""),
                        object=str(obj or ""),
                        hop=1,
                    )
                    prov.graph_path = [edge1]

        return results
        # PURE STRUCTURED SEARCH
    # =============================================================

    @track(name="structured_search", phase="retrieval")
    async def _structured_search(
        self,
        filters: CompanyFilters,
        candidate_k: int,
        allowed_departments: list[uuid.UUID] | None = None,
    ):

        conditions = (
            self._structured_conditions(
                filters
            )
        )

        if not conditions:
            return []

        statement = (
            select(
                DocumentChunk,
                Document,
            )
            .join(
                Document,
                Document.id
                == DocumentChunk.document_id,
            )
            .where(*conditions)
        )
        
        if allowed_departments is not None:
            from sqlalchemy import or_
            statement = statement.where(
                or_(
                    Document.department_id.in_(allowed_departments),
                    Document.department_id.is_(None)
                )
            )
            
        statement = statement.limit(candidate_k)

        rows = await self.session.execute(
            statement
        )

        return [
            (
                chunk,
                document,
                1.0,
            )
            for chunk, document in rows.all()
        ]

    # =============================================================
    # RRF
    # =============================================================

    def _merge_with_rrf(
        self,
        vector_rows: list[tuple[DocumentChunk, Document, float]],
        keyword_rows: list[tuple[DocumentChunk, Document, float]],
        knowledge_rows: list[tuple[DocumentChunk, Document, float]] | None = None,
        hierarchical_rows: list[tuple[DocumentChunk, Document, float]] | None = None,
    ) -> list[HybridCandidate]:
        candidates: dict[str, HybridCandidate] = {}

        for position, (chunk, document, distance) in enumerate(vector_rows, start=1):
            chunk_id = str(chunk.id)
            similarity = 1.0 - distance
            candidate = candidates.setdefault(
                chunk_id,
                HybridCandidate(chunk=chunk, document=document),
            )
            candidate.distance = distance
            candidate.vector_similarity = similarity
            candidate.vector_position = position
            candidate.hybrid_score += settings.vector_search_weight / (
                settings.rrf_k + position
            )

        for position, (chunk, document, keyword_rank) in enumerate(keyword_rows, start=1):
            chunk_id = str(chunk.id)
            candidate = candidates.setdefault(
                chunk_id,
                HybridCandidate(chunk=chunk, document=document),
            )
            candidate.keyword_rank = keyword_rank
            candidate.keyword_position = position
            candidate.hybrid_score += settings.keyword_search_weight / (
                settings.rrf_k + position
            )

        if knowledge_rows:
            for position, (chunk, document, knowledge_score) in enumerate(knowledge_rows, start=1):
                chunk_id = str(chunk.id)
                candidate = candidates.setdefault(
                    chunk_id,
                    HybridCandidate(chunk=chunk, document=document),
                )
                candidate.knowledge_position = position
                candidate.knowledge_score = knowledge_score
                candidate.hybrid_score += settings.vector_search_weight / (
                    settings.rrf_k + position
                )

        if hierarchical_rows:
            for position, (chunk, document, hierarchical_score) in enumerate(hierarchical_rows, start=1):
                chunk_id = str(chunk.id)
                candidate = candidates.setdefault(
                    chunk_id,
                    HybridCandidate(chunk=chunk, document=document),
                )
                candidate.hierarchical_position = position
                candidate.hierarchical_score = hierarchical_score
                candidate.hybrid_score += settings.hierarchical_search_weight / (
                    settings.rrf_k + position
                )

        return list(candidates.values())


    # =============================================================
    # RELEVANCE
    # =============================================================

    def _is_relevant(
        self,
        candidate: HybridCandidate,
    ) -> bool:

        vector_passed = (
            candidate.vector_similarity is not None
            and candidate.vector_similarity >= settings.min_vector_similarity
        )

        keyword_passed = (
            candidate.keyword_position is not None
            and candidate.keyword_rank is not None
            and candidate.keyword_rank > 0
        )

        knowledge_passed = (
            candidate.knowledge_position is not None
        )

        hierarchical_passed = (
            candidate.hierarchical_position is not None
            and candidate.hierarchical_score is not None
            and candidate.hierarchical_score > 0
        )

        return (
            vector_passed
            or keyword_passed
            or knowledge_passed
            or hierarchical_passed
        )

    # =============================================================
    # QUERY NORMALIZATION
    # =============================================================

    def _preprocess_query(
        self,
        query: str,
    ) -> str:

        return " ".join(
            query.strip().split()
        )

    # =============================================================
    # HIERARCHY METADATA RESOLUTION (OV-3)
    # =============================================================

    async def _resolve_chunk_hierarchies(
        self,
        chunks: list[RetrievedChunk],
        allowed_departments: list[uuid.UUID] | None = None,
    ) -> dict[str, ChunkHierarchyMetadata]:
        """
        Resolves hierarchy metadata for chunks with DocumentChunk.node_id.
        Walks parent sections to build parent_titles and path_display.
        Enforces department authorization.
        """
        if not chunks:
            return {}

        node_id_to_chunks: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            if chunk.node_id:
                node_id_to_chunks.setdefault(str(chunk.node_id), []).append(chunk)

        if not node_id_to_chunks:
            return {}

        node_uuids = []
        for nid in node_id_to_chunks:
            try:
                node_uuids.append(uuid.UUID(nid))
            except Exception:
                pass

        if not node_uuids:
            return {}

        stmt = (
            select(DocumentSection, Document)
            .join(Document, DocumentSection.document_id == Document.id)
            .where(DocumentSection.id.in_(node_uuids))
        )
        if allowed_departments is not None:
            stmt = stmt.where(Document.department_id.in_(allowed_departments))

        result = await self.session.execute(stmt)
        section_doc_pairs = result.all()

        if not section_doc_pairs:
            return {}

        doc_ids = list({
            (doc.id if hasattr(doc, "id") else doc[1].id if isinstance(doc, (tuple, list)) else pair[1].id)
            for pair in section_doc_pairs
            for doc in ([pair[1]] if isinstance(pair, (tuple, list)) or hasattr(pair, "__getitem__") else [pair.Document])
        }) if section_doc_pairs else []

        # Load all sections for relevant documents to resolve ancestor titles efficiently
        all_sections_stmt = (
            select(DocumentSection)
            .where(DocumentSection.document_id.in_(doc_ids))
            .order_by(DocumentSection.level.asc(), DocumentSection.ordinal_path.asc())
        )
        all_sec_result = await self.session.execute(all_sections_stmt)
        all_sections = all_sec_result.scalars().all()
        section_by_id: dict[uuid.UUID, DocumentSection] = {s.id: s for s in all_sections}

        chunk_hierarchy: dict[str, ChunkHierarchyMetadata] = {}

        for row in section_doc_pairs:
            sec = row[0] if isinstance(row, (tuple, list)) or hasattr(row, "__getitem__") else getattr(row, "DocumentSection", row)
            if not sec:
                continue

            ancestors: list[DocumentSection] = []
            curr_parent_id = sec.parent_id
            seen_ancestor_ids = set()
            while curr_parent_id and curr_parent_id in section_by_id and curr_parent_id not in seen_ancestor_ids:
                seen_ancestor_ids.add(curr_parent_id)
                p_sec = section_by_id[curr_parent_id]
                ancestors.insert(0, p_sec)
                curr_parent_id = p_sec.parent_id

            parent_titles = [a.title for a in ancestors if a.title]

            path_segments = [a.ordinal_path for a in ancestors if a.ordinal_path]
            if sec.ordinal_path:
                path_segments.append(sec.ordinal_path)

            path_display = " → ".join(path_segments) if path_segments else sec.ordinal_path

            matching_chunks = node_id_to_chunks.get(str(sec.id), [])
            for chunk in matching_chunks:
                chunk_hierarchy[chunk.chunk_id] = ChunkHierarchyMetadata(
                    chunk_id=chunk.chunk_id,
                    node_id=str(sec.id),
                    title=sec.title,
                    parent_titles=parent_titles,
                    ordinal_path=sec.ordinal_path,
                    level=sec.level,
                    l1_overview=sec.l1_overview,
                    l0_abstract=sec.l0_abstract,
                    path_display=path_display,
                )

        return chunk_hierarchy