import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.schemas.rag import (
    GenerateRequest,
    GenerateResponse,
    RetrievedChunk,
    FailureCategory,
)
from rag_app.schemas.visualization import validate_visualization
from rag_app.db.models import User, Department, UserDepartment, Workspace, WorkspaceMember, Conversation, ChatMessage
from rag_app.generation.service import GenerationService
from rag_app.providers.base.llm import LLMProvider
from rag_app.api.routes import ask_question


import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from rag_app.core.config import get_settings


@pytest_asyncio.fixture
async def db_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session
    await engine.dispose()


class MockLLM(LLMProvider):
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.provider_name = "mock"
        self.model_name = "mock-model"

    async def generate(self, messages, stage=None, **kwargs):
        return self.response_text, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}

    async def astream(self, messages, **kwargs):
        yield self.response_text

    async def ainvoke(self, messages, **kwargs):
        class MockAIMessage:
            def __init__(self, content):
                self.content = content
                self.usage_metadata = {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
        return MockAIMessage(self.response_text)


# ===========================================================================
# 1. Markdown bold rendering parser unit test
# ===========================================================================

def test_markdown_bold_parsing():
    """Verify that **bold** tokens are correctly separated for bold rendering."""
    import re
    inline_regex = re.compile(r"(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\[(?:chunk:\s*)?[a-zA-Z0-9_\-]+\]|\*[^*]+\*|_[^_]+_)")
    text = "This is **bold text** and `code`."
    parts = inline_regex.split(text)
    assert "**bold text**" in parts
    assert "`code`" in parts
    assert parts[parts.index("**bold text**")][2:-2] == "bold text"


# ===========================================================================
# 2. Schema Validation & Rejection Tests
# ===========================================================================

def test_valid_bar_visualization_passes():
    """Valid bar visualization passes strict schema validation."""
    valid_bar = {
        "type": "bar",
        "title": "Pipeline Opportunities",
        "x_axis": {"label": "Opportunity", "unit": None},
        "y_axis": {"label": "Value", "unit": "USD"},
        "series": [
            {
                "name": "Value",
                "data": [
                    {"label": "Vertex", "value": 920.0},
                    {"label": "Northstar", "value": 650.0},
                ],
            }
        ],
    }
    validated = validate_visualization(valid_bar)
    assert validated is not None
    assert validated["type"] == "bar"
    assert len(validated["series"][0]["data"]) == 2


def test_missing_axes_and_series_rejected():
    """Bar chart missing title or with empty/unusable structure is rejected."""
    invalid_bar = {
        "type": "bar",
        # missing title, x_axis, y_axis, series, data
    }
    assert validate_visualization(invalid_bar) is None

    empty_bar = {
        "type": "bar",
        "title": "Empty Bar",
        "data": [],
    }
    assert validate_visualization(empty_bar) is None


def test_fabricated_chart_row_rejected():
    """Grounding evaluator rejects fabricated chart rows not present in authorized evidence."""
    from rag_app.generation.grounding import AnswerGroundingEvaluator
    evaluator = AnswerGroundingEvaluator()
    chunks = [
        RetrievedChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=str(uuid.uuid4()),
            department_id=str(uuid.uuid4()),
            file_name="report.md",
            chunk_index=1,
            text="Real Opportunity Alpha has value $500,000.",
        )
    ]
    fabricated_viz = {
        "type": "bar",
        "title": "Pipeline",
        "x_axis": {"label": "Opportunity"},
        "y_axis": {"label": "Value"},
        "series": [
            {
                "name": "Deals",
                "data": [
                    {"label": "Real Opportunity Alpha", "value": 500000.0},
                    {"label": "Fabricated Phantom Corp", "value": 9999999.0},
                ],
            }
        ],
    }
    repaired = evaluator.repair_visualization(
        visualization=fabricated_viz,
        query="Visualize pipeline",
        chunks=chunks,
    )
    # The fabricated row is pruned
    assert repaired is not None
    points = repaired["series"][0]["data"]
    assert len(points) == 1
    assert points[0]["label"] == "Real Opportunity Alpha"


# ===========================================================================
# 3. Scoped "Visualize This" Flow Tests (No second retrieval, authorized evidence)
# ===========================================================================

@pytest.mark.asyncio
async def test_visualize_this_uses_parent_evidence_no_second_retrieval(db_session: AsyncSession):
    """
    Verify that when Visualize This is called with target_message_id:
    - Zero calls are made to RetrievalService.retrieve
    - The visualization is constructed from target message's authorized sources
    """
    user_id = uuid.uuid4()
    dept_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    user = User(id=user_id, email=f"analyst_{uuid.uuid4().hex[:6]}@corp.local", role="USER", hashed_password="fake")
    dept = Department(id=dept_id, name="Sales", code=f"SALES_{uuid.uuid4().hex[:6]}", is_active=True)
    user_dept = UserDepartment(user_id=user_id, department_id=dept_id)
    conv = Conversation(id=conv_id, user_id=user_id, title="Sales Discussion")

    parent_sources = [
        {
            "type": "document",
            "chunk_id": "chunk_parent_1",
            "document_id": str(uuid.uuid4()),
            "department_id": str(dept_id),
            "file_name": "Pipeline.md",
            "chunk_index": 0,
            "title": "Pipeline.md",
            "excerpt": "Opportunity Vertex has value $920K. Opportunity Northstar has value $650K.",
        }
    ]
    parent_msg = ChatMessage(
        conversation_id=conv_id,
        role="assistant",
        content="Here is the pipeline: Vertex ($920K) and Northstar ($650K).",
        sources=parent_sources,
    )

    db_session.add_all([user, dept, user_dept, conv, parent_msg])
    await db_session.commit()

    viz_payload = {
        "answer": "Here is the visual pipeline chart.",
        "visualization": {
            "type": "bar",
            "title": "Pipeline Opportunities",
            "x_axis": {"label": "Opportunity"},
            "y_axis": {"label": "Value"},
            "series": [
                {
                    "name": "Value",
                    "data": [
                        {"label": "Vertex", "value": 920.0},
                        {"label": "Northstar", "value": 650.0},
                    ],
                }
            ],
        },
    }
    mock_llm = MockLLM(json.dumps(viz_payload))

    gen_service = GenerationService(db_session)
    gen_service.llm = mock_llm

    mock_retrieval_service = MagicMock()
    mock_retrieval_service.retrieve = AsyncMock()

    request = GenerateRequest(
        query="Visualize this",
        action="visualize",
        target_message_id=parent_msg.id,
        conversation_id=conv_id,
    )

    with patch("rag_app.security.authorization.get_allowed_departments", return_value=[dept_id]):
        resp = await ask_question(
            request=request,
            service=gen_service,
            retrieval_service=mock_retrieval_service,
            session=db_session,
            user=user,
        )

    # 1. RetrievalService.retrieve MUST NOT be called!
    mock_retrieval_service.retrieve.assert_not_called()

    # 2. Response has valid visualization based on parent evidence
    assert resp.visualization is not None
    assert resp.visualization["type"] == "bar"
    assert len(resp.retrieved_chunks) == 1
    assert resp.retrieved_chunks[0].chunk_id == "chunk_parent_1"


@pytest.mark.asyncio
async def test_unauthorized_evidence_cannot_enter_visualization(db_session: AsyncSession):
    """
    Verify that if a parent message somehow contained sources from an unauthorized department,
    those chunks are stripped and cannot enter the visualization context.
    """
    user_id = uuid.uuid4()
    dept_allowed = uuid.uuid4()
    dept_forbidden = uuid.uuid4()
    conv_id = uuid.uuid4()

    user = User(id=user_id, email=f"finance_{uuid.uuid4().hex[:6]}@corp.local", role="USER", hashed_password="fake")
    dept1 = Department(id=dept_allowed, name="Finance", code=f"FIN_{uuid.uuid4().hex[:6]}", is_active=True)
    dept2 = Department(id=dept_forbidden, name="Confidential HR", code=f"HR_{uuid.uuid4().hex[:6]}", is_active=True)
    user_dept = UserDepartment(user_id=user_id, department_id=dept_allowed)
    conv = Conversation(id=conv_id, user_id=user_id, title="Finance Review")

    parent_sources = [
        {
            "type": "document",
            "chunk_id": "chunk_hr_forbidden",
            "document_id": str(uuid.uuid4()),
            "department_id": str(dept_forbidden),
            "file_name": "Executive_Salaries.md",
            "chunk_index": 0,
            "title": "Executive_Salaries.md",
            "excerpt": "CEO Salary is $5,000,000.",
        }
    ]
    parent_msg = ChatMessage(
        conversation_id=conv_id,
        role="assistant",
        content="Confidential salary data was previously discussed.",
        sources=parent_sources,
    )

    db_session.add_all([user, dept1, dept2, user_dept, conv, parent_msg])
    await db_session.commit()

    mock_retrieval_service = MagicMock()
    mock_retrieval_service.retrieve = AsyncMock()
    gen_service = GenerationService(db_session)

    request = GenerateRequest(
        query="Visualize this",
        action="visualize",
        target_message_id=parent_msg.id,
        conversation_id=conv_id,
    )

    with patch("rag_app.security.authorization.get_allowed_departments", return_value=[dept_allowed]):
        resp = await ask_question(
            request=request,
            service=gen_service,
            retrieval_service=mock_retrieval_service,
            session=db_session,
            user=user,
        )

    # Forbidden chunk was filtered out, leaving no authorized structured data -> fails safely
    assert resp.visualization is None
    assert "unable" in resp.answer.lower() or "cannot" in resp.answer.lower()
    if resp.execution_metrics and resp.execution_metrics.failure_categories:
        assert FailureCategory.VISUALIZATION in resp.execution_metrics.failure_categories

