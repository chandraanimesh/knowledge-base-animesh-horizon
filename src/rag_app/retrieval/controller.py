import logging
from enum import StrEnum
from rag_app.schemas.rag import (
    EvidenceNeed,
    InformationNeed,
    PresentationNeed,
    QueryPlan,
    RelationshipQuery,
)
from rag_app.retrieval.query_parser import extract_deterministic_relationship

logger = logging.getLogger(__name__)


class RetrievalStrategy(StrEnum):
    GRAPH = "graph"
    KNOWLEDGE = "knowledge"
    SEMANTIC = "semantic"
    STRUCTURED = "structured"
    HIERARCHICAL = "hierarchical"


class RetrievalController:
    """
    Centralized controller for determining and managing retrieval strategy routes.
    Deterministic, zero external LLM calls, and strictly preserves existing
    retrieval algorithms, authorization boundaries, and fusion ranking.
    """

    def plan_route(self, query_plan: QueryPlan) -> list[RetrievalStrategy]:
        """
        Determines the priority order of retrieval strategies based on query signals
        and generic query understanding (information_need, presentation_need, evidence_need):
        
        - CONVERSATION -> [] (Conversation/greeting routed via conversation path, skips doc retrieval)
        - relationship_query -> GRAPH
        - entity_search_term -> KNOWLEDGE
        - Pure structured filters only -> STRUCTURED
        - OVERVIEW -> SEMANTIC + HIERARCHICAL
        - REQUIREMENTS -> SEMANTIC + HIERARCHICAL
        - LIST -> SEMANTIC + HIERARCHICAL
        - COMPARISON -> SEMANTIC + (STRUCTURED if filters) + HIERARCHICAL
        - AGGREGATION / CHART / TABLE -> SEMANTIC + (STRUCTURED if filters) + HIERARCHICAL
        - GENERAL / FACTUAL / EXPLANATION -> SEMANTIC + (HIERARCHICAL if evidence warrants)
        """
        if query_plan.conversation_required or query_plan.information_need == InformationNeed.CONVERSATION or query_plan.evidence_need == EvidenceNeed.CONVERSATION:
            return []

        strategies: list[RetrievalStrategy] = []

        has_filters = any(
            v is not None
            for v in query_plan.filters.model_dump().values()
        )
        has_semantic = bool(
            query_plan.semantic_query
            and query_plan.semantic_query.strip()
        )

        # Graph traversal priority
        if query_plan.relationship_query is not None:
            strategies.append(RetrievalStrategy.GRAPH)

        # Entity/knowledge priority
        if query_plan.entity_search_term:
            strategies.append(RetrievalStrategy.KNOWLEDGE)

        # Pure structured query check
        if has_filters and not has_semantic and not query_plan.relationship_query and not query_plan.entity_search_term:
            strategies.append(RetrievalStrategy.STRUCTURED)
        else:
            # Semantic search is the primary anchor for text queries
            strategies.append(RetrievalStrategy.SEMANTIC)

            # Include structured search if structured filters exist in compound/hybrid query
            if has_filters:
                strategies.append(RetrievalStrategy.STRUCTURED)

            # Hierarchical strategy planning based on generic information need and evidence need
            include_hierarchical = False
            if query_plan.information_need in (
                InformationNeed.OVERVIEW,
                InformationNeed.REQUIREMENTS,
                InformationNeed.LIST,
                InformationNeed.COMPARISON,
                InformationNeed.AGGREGATION,
            ):
                include_hierarchical = True
            elif query_plan.evidence_need in (
                EvidenceNeed.HIERARCHICAL,
                EvidenceNeed.MIXED,
                EvidenceNeed.STRUCTURED,
            ):
                include_hierarchical = True
            elif query_plan.presentation_need in (
                PresentationNeed.CHART,
                PresentationNeed.TABLE,
                PresentationNeed.TIMELINE,
            ):
                include_hierarchical = True
            else:
                # Default for general/factual/explanation queries: allow hierarchy to contribute
                include_hierarchical = True

            if include_hierarchical:
                strategies.append(RetrievalStrategy.HIERARCHICAL)

        # Deduplicate while preserving insertion order
        seen = set()
        deduped_route: list[RetrievalStrategy] = []
        for strat in strategies:
            if strat not in seen:
                seen.add(strat)
                deduped_route.append(strat)

        logger.info(
            "retrieval_route_planned",
            extra={
                "event": "retrieval_route_planned",
                "route": [s.value for s in deduped_route],
                "information_need": query_plan.information_need.value if hasattr(query_plan.information_need, "value") else str(query_plan.information_need),
                "presentation_need": query_plan.presentation_need.value if hasattr(query_plan.presentation_need, "value") else str(query_plan.presentation_need),
                "evidence_need": query_plan.evidence_need.value if hasattr(query_plan.evidence_need, "value") else str(query_plan.evidence_need),
                "has_relationship": query_plan.relationship_query is not None,
                "has_entity": bool(query_plan.entity_search_term),
                "has_filters": has_filters,
                "has_semantic": has_semantic,
            },
        )

        return deduped_route

    def is_compound_query(self, query_plan: QueryPlan) -> bool:
        """
        Deterministically checks if a QueryPlan contains multiple retrieval signals
        requiring all planned strategies to execute without early stopping.
        """
        sq = (query_plan.semantic_query or "").strip().lower()

        # 1. Check for compound conjunctions / multi-clause delimiters
        compound_delimiters = (
            " and ", " & ", " as well as ", " also ",
            ";", " plus ", " along with ", " in addition to "
        )
        if any(delim in sq for delim in compound_delimiters):
            return True

        has_rel = query_plan.relationship_query is not None
        has_entity = bool(query_plan.entity_search_term)

        # 2. Both explicit relationship and entity signals present
        if has_rel and has_entity:
            return True

        # 3. Entity signal with non-pure/compound query phrasing
        if has_entity and sq:
            entity_term = query_plan.entity_search_term.strip().lower()
            if not self._is_pure_entity_phrasing(sq, entity_term):
                return True

        # 4. Relationship signal with non-pure/compound query phrasing
        if has_rel and sq:
            if not self._is_pure_relationship_phrasing(sq, query_plan.relationship_query):
                return True

        return False

    @staticmethod
    def _is_pure_entity_phrasing(sq: str, entity_term: str) -> bool:
        sq_clean = sq.rstrip("?.,!\"' ")
        term_clean = entity_term.rstrip("?.,!\"' ")
        if not sq_clean or sq_clean == term_clean:
            return True
        for prefix in ("what is ", "who is ", "who was ", "tell me about ", "about "):
            if sq_clean.startswith(prefix):
                remainder = sq_clean[len(prefix):].strip("?.,!\"' ")
                if remainder == term_clean:
                    return True
        return False

    @staticmethod
    def _is_pure_relationship_phrasing(sq: str, rel_query: RelationshipQuery) -> bool:
        det_rel = extract_deterministic_relationship(sq)
        if det_rel is not None and det_rel.predicate == rel_query.predicate:
            target = det_rel.object_entity or det_rel.subject_entity
            if target and len(target.split()) <= 4:
                return True
        return False

    def is_sufficient(self, candidate_count: int) -> bool:
        """
        Reuses existing binary sufficiency check (presence of candidates).
        No arbitrary numeric thresholds are introduced.
        """
        return candidate_count > 0

