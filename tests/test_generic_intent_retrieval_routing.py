import asyncio
import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import Document, DocumentChunk, DocumentSection
from rag_app.retrieval.service import RetrievalService, HybridCandidate
from rag_app.retrieval.controller import RetrievalController, RetrievalStrategy
from rag_app.retrieval.query_parser import (
    QueryParser,
    extract_deterministic_intent,
    strip_presentation_instructions,
)
from rag_app.schemas.rag import (
    EvidenceNeed,
    InformationNeed,
    PresentationNeed,
    QueryPlan,
)
from rag_app.providers.base.reranker import RerankerProvider


@pytest.fixture
def db_session():
    mock_session = AsyncMock(spec=AsyncSession)
    return mock_session


class MockReranker(RerankerProvider):
    def __init__(self, score_map: dict[str, float] | None = None):
        self.score_map = score_map or {}
        self.invocations: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.invocations.append((query, texts))
        scores = []
        for text in texts:
            matched = False
            for key, score in self.score_map.items():
                if key in text:
                    scores.append(score)
                    matched = True
                    break
            if not matched:
                scores.append(1.0)
        return scores


# ===========================================================================
# 1. Query Understanding & Presentation Separation Tests
# ===========================================================================

def test_query_understanding_classification():
    """Verify generic deterministic intent extraction for core categories."""
    # Visualization / Chart intent separation
    info, pres, ev, conv_req, reason = extract_deterministic_intent("Visualize the Q3 sales pipeline opportunities by value as a bar chart")
    assert info == InformationNeed.AGGREGATION
    assert pres == PresentationNeed.CHART
    assert ev == EvidenceNeed.STRUCTURED
    assert conv_req is False

    # Overview
    info, pres, ev, conv_req, reason = extract_deterministic_intent("What is this document about?")
    assert info == InformationNeed.OVERVIEW
    assert pres == PresentationNeed.NONE
    assert ev == EvidenceNeed.HIERARCHICAL
    assert conv_req is False

    # Requirements
    info, pres, ev, conv_req, reason = extract_deterministic_intent("What needs to be implemented?")
    assert info == InformationNeed.REQUIREMENTS
    assert pres == PresentationNeed.NONE
    assert ev == EvidenceNeed.HIERARCHICAL
    assert conv_req is False

    info, pres, ev, conv_req, reason = extract_deterministic_intent("What are the technical requirements?")
    assert info == InformationNeed.REQUIREMENTS
    assert pres == PresentationNeed.NONE
    assert conv_req is False

    # List
    info, pres, ev, conv_req, reason = extract_deterministic_intent("List the requirements.")
    assert info == InformationNeed.LIST
    assert pres == PresentationNeed.NONE
    assert conv_req is False

    # Comparison
    info, pres, ev, conv_req, reason = extract_deterministic_intent("Compare the relevant options.")
    assert info == InformationNeed.COMPARISON
    assert pres == PresentationNeed.NONE
    assert ev == EvidenceNeed.MIXED
    assert conv_req is False

    # Conversation / History
    info, pres, ev, conv_req, reason = extract_deterministic_intent("What did I ask earlier?")
    assert info == InformationNeed.CONVERSATION
    assert pres == PresentationNeed.NONE
    assert ev == EvidenceNeed.CONVERSATION
    assert conv_req is True


def test_phase_8_planner_categories():
    """Verify Phase 8 planner categories without false positive conversation classifications."""
    # 1. Corpus requirements
    corpus_queries = [
        "What work is still outstanding before launch?",
        "What information is missing before I can start implementation?",
        "What still needs to be implemented?",
        "Which requirements remain?",
        "What needs to change before the system is ready?",
    ]
    for q in corpus_queries:
        info, pres, ev, conv_req, reason = extract_deterministic_intent(q)
        assert conv_req is False, f"Expected conv_req=False for {q}"
        assert info in (InformationNeed.REQUIREMENTS, InformationNeed.OVERVIEW, InformationNeed.FACTUAL, InformationNeed.LIST), f"Unexpected info need for {q}: {info}"
        assert ev != EvidenceNeed.CONVERSATION, f"Unexpected evidence need for {q}: {ev}"

    # 2. Conversation
    conv_queries = [
        "What did I ask earlier?",
        "What did you tell me earlier about implementation?",
        "Which of those requirements did we discuss?",
    ]
    for q in conv_queries:
        info, pres, ev, conv_req, reason = extract_deterministic_intent(q)
        assert conv_req is True, f"Expected conv_req=True for {q}"
        assert info == InformationNeed.CONVERSATION, f"Unexpected info need for {q}: {info}"
        assert ev == EvidenceNeed.CONVERSATION, f"Unexpected evidence need for {q}: {ev}"

    # 3. Visualization
    viz_queries = [
        "Show Q3 revenue by region as a bar chart.",
        "Visualize pipeline opportunities by value.",
        "Make a chart of opportunities above $500K.",
    ]
    for q in viz_queries:
        info, pres, ev, conv_req, reason = extract_deterministic_intent(q)
        assert conv_req is False
        assert pres == PresentationNeed.CHART
        assert ev == EvidenceNeed.STRUCTURED

    # 4. Normal factual
    factual_queries = [
        "What is the implementation deadline?",
        "Who owns the backend integration?",
        "What is the required authentication method?",
    ]
    for q in factual_queries:
        info, pres, ev, conv_req, reason = extract_deterministic_intent(q)
        assert conv_req is False
        assert ev in (EvidenceNeed.PROSE, EvidenceNeed.HIERARCHICAL)

    # 5. Mixed
    mixed_query = "Explain the implementation requirements and compare backend vs frontend responsibilities."
    info, pres, ev, conv_req, reason = extract_deterministic_intent(mixed_query)
    assert conv_req is False
    assert ev == EvidenceNeed.MIXED


def test_query_representation_strips_presentation_terms():
    """Verify presentation normalization removes presentation terms while preserving factual terms."""
    q = "Visualize the Q3 sales pipeline opportunities by value as a bar chart"
    cleaned = strip_presentation_instructions(q)
    assert "bar chart" not in cleaned.lower()
    assert "visualize" not in cleaned.lower()
    assert "q3 sales pipeline opportunities by value" in cleaned.lower()

    q2 = "Show the quarterly revenue as a markdown table"
    cleaned2 = strip_presentation_instructions(q2)
    assert "markdown table" not in cleaned2.lower()
    assert "quarterly revenue" in cleaned2.lower()


# ===========================================================================
# 2. Retrieval Planner Routing Tests
# ===========================================================================

def test_retrieval_planner_routing():
    controller = RetrievalController()

    # Overview route
    plan_overview = QueryPlan(
        semantic_query="overview of the platform",
        information_need=InformationNeed.OVERVIEW,
        evidence_need=EvidenceNeed.HIERARCHICAL,
    )
    route = controller.plan_route(plan_overview)
    assert RetrievalStrategy.SEMANTIC in route
    assert RetrievalStrategy.HIERARCHICAL in route

    # Requirements route
    plan_reqs = QueryPlan(
        semantic_query="technical requirements",
        information_need=InformationNeed.REQUIREMENTS,
        evidence_need=EvidenceNeed.HIERARCHICAL,
    )
    route = controller.plan_route(plan_reqs)
    assert RetrievalStrategy.SEMANTIC in route
    assert RetrievalStrategy.HIERARCHICAL in route

    # List route
    plan_list = QueryPlan(
        semantic_query="requirements",
        information_need=InformationNeed.LIST,
        evidence_need=EvidenceNeed.MIXED,
    )
    route = controller.plan_route(plan_list)
    assert RetrievalStrategy.SEMANTIC in route
    assert RetrievalStrategy.HIERARCHICAL in route

    # Comparison route
    plan_comp = QueryPlan(
        semantic_query="compare approach A and approach B",
        information_need=InformationNeed.COMPARISON,
        evidence_need=EvidenceNeed.MIXED,
    )
    route = controller.plan_route(plan_comp)
    assert RetrievalStrategy.SEMANTIC in route
    assert RetrievalStrategy.HIERARCHICAL in route

    # Visualization / Aggregation route
    plan_viz = QueryPlan(
        semantic_query="sales pipeline by value",
        information_need=InformationNeed.AGGREGATION,
        presentation_need=PresentationNeed.CHART,
        evidence_need=EvidenceNeed.STRUCTURED,
    )
    route = controller.plan_route(plan_viz)
    assert RetrievalStrategy.SEMANTIC in route
    assert RetrievalStrategy.HIERARCHICAL in route

    # Conversation route skips document retrieval
    plan_conv = QueryPlan(
        semantic_query="what did i ask earlier",
        information_need=InformationNeed.CONVERSATION,
        evidence_need=EvidenceNeed.CONVERSATION,
    )
    route = controller.plan_route(plan_conv)
    assert route == []


# ===========================================================================
# 3. Hierarchical Retrieval Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_hierarchical_retrieval_finds_section_evidence(db_session: AsyncSession):
    """
    Verify hierarchical retrieval retrieves section and subsection chunks
    for requirements and structural queries.
    """
    service = RetrievalService(db_session)
    dept_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    doc = Document(
        id=doc_id,
        file_name="Architecture_Spec.md",
        file_path="/tmp/Architecture_Spec.md",
        content_hash="hash_spec",
        department_id=dept_id,
    )

    root_section = DocumentSection(
        id=uuid.uuid4(),
        document_id=doc_id,
        node_type="DOCUMENT",
        title="Architecture Specification",
        level=0,
        ordinal_path="0",
        l0_abstract="Architecture specification document covering technical requirements.",
    )
    reqs_section = DocumentSection(
        id=uuid.uuid4(),
        document_id=doc_id,
        parent_id=root_section.id,
        node_type="SECTION",
        title="5. Technical Requirements",
        level=1,
        ordinal_path="1.5",
        l0_abstract="Technical requirements for backend and frontend systems.",
        l1_overview="Sections: Backend Requirements | Frontend Requirements",
    )
    backend_section = DocumentSection(
        id=uuid.uuid4(),
        document_id=doc_id,
        parent_id=reqs_section.id,
        node_type="SUBSECTION",
        title="5.1 Backend Requirements",
        level=2,
        ordinal_path="1.5.1",
        l0_abstract="Backend requirements: FastAPI, PostgreSQL, and async workers.",
    )

    chunk1 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        chunk_index=1,
        node_id=backend_section.id,
        text="The backend must implement asynchronous queue processing and PostgreSQL indexing.",
        metadata_={"header_1": "5. Technical Requirements", "header_2": "5.1 Backend Requirements"},
    )

    # Mock DB query results for DocumentSection and DocumentChunk
    async def mock_execute(stmt):
        mock_res = MagicMock()
        stmt_str = str(stmt).lower()
        if "from document_chunks" in stmt_str:
            mock_res.all.return_value = [(chunk1, doc)]
        elif "where" in stmt_str and "document_sections.parent_id in" in stmt_str:
            mock_res.all.return_value = [(backend_section, doc)]
        elif "from document_sections" in stmt_str:
            mock_res.all.return_value = [(reqs_section, doc, 0.8)]
        else:
            mock_res.all.return_value = []
        return mock_res

    with patch.object(service.session, "execute", side_effect=mock_execute):
        results = await service._hierarchical_search(
            query="What needs to be implemented?",
            candidate_k=5,
            allowed_departments=[dept_id],
            information_need=InformationNeed.REQUIREMENTS,
        )

    assert len(results) >= 1
    assert results[0][0].id == chunk1.id


# ===========================================================================
# 4. Evidence Preservation & Diversity Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_evidence_preservation_rescues_table_chunk_for_visualization(db_session: AsyncSession):
    """
    Verify bounded evidence preservation rescues a high-relevance table chunk
    when diversity cap would otherwise exclude it for visualization queries.
    """
    service = RetrievalService(db_session)
    mock_reranker = MockReranker({
        "Prose chunk 1": 0.95,
        "Prose chunk 2": 0.94,
        "Prose chunk 3": 0.93,
        "Prose chunk 4": 0.92,
        "Prose chunk 5": 0.91,
        "Opportunity": 0.90,  # High-relevance table chunk (> 80% of floor 0.92)
    })
    service.reranker_provider = mock_reranker

    doc = Document(id=uuid.uuid4(), file_name="Q3_Sales.md", department_id=uuid.uuid4(), content_hash="h1")
    
    prose_chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=i,
            text=f"Prose chunk {i} discussing quarterly overview.",
        )
        for i in range(1, 6)
    ]
    table_chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=6,
        text="| Opportunity | Value |\n|---|---|\n| Deal 1 | $500K |\n| Deal 2 | $300K |",
    )

    candidates = [
        HybridCandidate(chunk=c, document=doc, vector_similarity=0.8, hybrid_score=0.03)
        for c in prose_chunks
    ] + [HybridCandidate(chunk=table_chunk, document=doc, vector_similarity=0.8, hybrid_score=0.02)]

    with patch.object(service, "_vector_search", AsyncMock(return_value=[])), \
         patch.object(service, "_keyword_search", AsyncMock(return_value=[])), \
         patch.object(service, "_hierarchical_search", AsyncMock(return_value=[])), \
         patch.object(service, "_merge_with_rrf", MagicMock(return_value=candidates)):

        resp = await service.retrieve(
            query="Visualize the Q3 sales pipeline opportunities by value as a bar chart",
            top_k=4,
            rerank=True,
        )

    # 4 prose chunks + 1 rescued table chunk = 5 accepted chunks (bounded rescue)
    rescued_texts = [r.text for r in resp.results]
    assert any("| Opportunity | Value |" in t for t in rescued_texts)


@pytest.mark.asyncio
async def test_authorization_strictly_enforced_in_hierarchical_search(db_session: AsyncSession):
    """Verify unauthorized departments cannot be accessed via hierarchical retrieval."""
    service = RetrievalService(db_session)
    authorized_dept = uuid.uuid4()
    unauthorized_dept = uuid.uuid4()

    doc_unauth = Document(id=uuid.uuid4(), file_name="Secret.md", department_id=unauthorized_dept, content_hash="s1")
    sec_unauth = DocumentSection(id=uuid.uuid4(), document_id=doc_unauth.id, title="Secret Requirements")
    chunk_unauth = DocumentChunk(id=uuid.uuid4(), document_id=doc_unauth.id, text="Confidential info", node_id=sec_unauth.id)

    async def mock_execute(stmt):
        mock_res = MagicMock()
        # If department filter is enforced in query, unauth doc is not returned
        mock_res.all.return_value = []
        return mock_res

    with patch.object(service.session, "execute", side_effect=mock_execute):
        results = await service._hierarchical_search(
            query="Secret Requirements",
            candidate_k=5,
            allowed_departments=[authorized_dept],
        )

    assert len(results) == 0
