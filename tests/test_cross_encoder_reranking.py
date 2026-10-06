import asyncio
import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import Document, DocumentChunk
from rag_app.retrieval.service import RetrievalService, HybridCandidate
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


@pytest.mark.asyncio
async def test_cross_encoder_invoked_and_assigns_numeric_scores(db_session: AsyncSession):
    """
    Requirements 1 & 2:
    - CrossEncoder is invoked when rerank=True.
    - reranker_score is numeric (float) after reranking for all accepted candidates.
    """
    service = RetrievalService(db_session)
    mock_reranker = MockReranker()
    service.reranker_provider = mock_reranker

    doc = Document(
        id=uuid.uuid4(),
        file_name="Q3_Report.md",
        file_path="/tmp/Q3_Report.md",
        content_hash="hash1",
        department_id=uuid.uuid4(),
    )
    chunk1 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=1,
        text="Prose overview of sales performance.",
        metadata_={"header_1": "Executive Summary"},
    )
    chunk2 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=2,
        text="| Opportunity | Value |\n|---|---|\n| Deal A | $100K |",
        metadata_={"header_1": "Pipeline Opportunities"},
    )

    candidate1 = HybridCandidate(chunk=chunk1, document=doc, vector_similarity=0.8, hybrid_score=0.030)
    candidate2 = HybridCandidate(chunk=chunk2, document=doc, vector_similarity=0.7, hybrid_score=0.020)

    with patch.object(service, "_vector_search", AsyncMock(return_value=[])), \
         patch.object(service, "_keyword_search", AsyncMock(return_value=[])), \
         patch.object(service, "_merge_with_rrf", MagicMock(return_value=[candidate1, candidate2])):

        resp = await service.retrieve(query="sales pipeline opportunities", top_k=5, rerank=True)

    # 1. CrossEncoder was invoked
    assert len(mock_reranker.invocations) == 1
    query_used, texts_used = mock_reranker.invocations[0]
    assert len(texts_used) == 2

    # 2. Results have numeric reranker_scores
    assert len(resp.results) == 2
    for r in resp.results:
        assert isinstance(r.reranker_score, float)
        assert r.reranker_score is not None


@pytest.mark.asyncio
async def test_top_k_selected_after_reranking_and_table_outranks_prose(db_session: AsyncSession):
    """
    Requirements 3 & 4:
    - Final top_k is selected AFTER reranking.
    - Relevant table chunk with lower RRF can outrank generic prose chunks with higher RRF.
    """
    service = RetrievalService(db_session)
    # Configure mock reranker so the table chunk receives a higher score than the prose chunk
    mock_reranker = MockReranker(score_map={
        "Pipeline Opportunities": 8.5,
        "Executive Summary": 2.1,
    })
    service.reranker_provider = mock_reranker

    doc = Document(
        id=uuid.uuid4(),
        file_name="Q3_Report.md",
        file_path="/tmp/Q3_Report.md",
        content_hash="hash2",
        department_id=uuid.uuid4(),
    )
    chunk_prose = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=1,
        text="General corporate commentary and prose text.",
        metadata_={"header_1": "Executive Summary"},
    )
    chunk_table = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=2,
        text="| Opportunity | Value |\n| Vertex | $920K |",
        metadata_={"header_1": "Pipeline Opportunities"},
    )

    # Pre-rerank: prose has higher RRF (0.040) vs table (0.015)
    cand_prose = HybridCandidate(chunk=chunk_prose, document=doc, vector_similarity=0.9, hybrid_score=0.040)
    cand_table = HybridCandidate(chunk=chunk_table, document=doc, vector_similarity=0.5, hybrid_score=0.015)

    with patch.object(service, "_vector_search", AsyncMock(return_value=[])), \
         patch.object(service, "_keyword_search", AsyncMock(return_value=[])), \
         patch.object(service, "_merge_with_rrf", MagicMock(return_value=[cand_prose, cand_table])):

        # Retrieve top_k=1 to prove table wins the single accepted slot after reranking
        resp = await service.retrieve(query="Q3 sales pipeline opportunities", top_k=1, rerank=True)

    assert len(resp.results) == 1
    # The table chunk outranked prose and won slot #1!
    assert resp.results[0].chunk_index == 2
    assert "Vertex" in resp.results[0].text
    assert resp.results[0].reranker_score == 8.5


def test_structural_metadata_in_reranker_text():
    """
    Requirement 5:
    - section/search_context and document name metadata are included in reranker input.
    """
    session = MagicMock()
    service = RetrievalService(session)

    doc = Document(
        id=uuid.uuid4(),
        file_name="0a54d58489d34ebca39bba76ae6aade3_Q3_Sales_Report.md",
        file_path="/tmp/doc.md",
        content_hash="hash3",
        department_id=uuid.uuid4(),
    )
    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=10,
        text="| Opportunity | Value |\n| Vertex | $920K |",
        metadata_={
            "header_1": "Q3 Report",
            "header_2": "Sales Pipeline",
            "search_context": "Quarterly opportunities summary",
        },
    )
    candidate = HybridCandidate(chunk=chunk, document=doc)

    formatted = service._get_candidate_rerank_text(candidate)

    assert "Document: Q3_Sales_Report.md" in formatted
    assert "Section: Q3 Report > Sales Pipeline" in formatted
    assert "Search context: Quarterly opportunities summary" in formatted
    assert "Content: | Opportunity | Value |\n| Vertex | $920K |" in formatted


@pytest.mark.asyncio
async def test_normal_semantic_retrieval_when_rerank_disabled(db_session: AsyncSession):
    """
    Requirement 6:
    - Existing normal semantic retrieval still works when rerank=False.
    """
    service = RetrievalService(db_session)
    mock_reranker = MockReranker()
    service.reranker_provider = mock_reranker

    doc = Document(
        id=uuid.uuid4(),
        file_name="Q3_Report.md",
        file_path="/tmp/Q3_Report.md",
        content_hash="hash4",
        department_id=uuid.uuid4(),
    )
    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=1,
        text="Normal semantic text.",
        metadata_={},
    )
    cand = HybridCandidate(chunk=chunk, document=doc, vector_similarity=0.8, hybrid_score=0.035)

    with patch.object(service, "_vector_search", AsyncMock(return_value=[])), \
         patch.object(service, "_keyword_search", AsyncMock(return_value=[])), \
         patch.object(service, "_merge_with_rrf", MagicMock(return_value=[cand])):

        resp = await service.retrieve(query="test query", top_k=5, rerank=False)

    # Reranker should NOT have been invoked
    assert len(mock_reranker.invocations) == 0
    assert len(resp.results) == 1
    assert resp.results[0].reranker_score is None
    assert resp.results[0].hybrid_score == 0.035
