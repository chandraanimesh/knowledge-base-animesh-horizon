from enum import StrEnum
from rag_app.schemas.rag import QueryPlan
from rag_app.retrieval.controller import RetrievalController, RetrievalStrategy


class RetrievalMode(StrEnum):
    STRUCTURED = "structured"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


def route_query(
    query_plan: QueryPlan,
) -> RetrievalMode:

    has_filters = any(
        value is not None
        for value in query_plan.filters.model_dump().values()
    )

    has_semantic_query = bool(
        query_plan.semantic_query
        and query_plan.semantic_query.strip()
    )

    if has_filters and has_semantic_query:
        return RetrievalMode.HYBRID

    if has_filters:
        return RetrievalMode.STRUCTURED

    return RetrievalMode.SEMANTIC