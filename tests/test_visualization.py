import pytest
import pytest_asyncio
import uuid
import json
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.schemas.visualization import (
    validate_visualization,
    BarVisualization,
    LineVisualization,
    PieVisualization,
    AreaVisualization,
    TableVisualization,
    NetworkVisualization,
    TimelineVisualization,
)
from rag_app.generation.service import _parse_llm_structured_output, GenerationService
from rag_app.schemas.rag import GenerateResponse, RetrievedChunk
from rag_app.generation.citations import validate_and_render_citations
from rag_app.providers.base.llm import LLMProvider
from rag_app.db.models import (
    User,
    Department,
    Permission,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
    Conversation,
    ChatMessage,
)
from rag_app.conversation.service import ConversationService


# ============================================================
# 1. UNIT TESTS: SCHEMA & PARSING
# ============================================================

def test_bar_chart_validation():
    data = {
        "type": "bar",
        "title": "Q2 vs Q3 Sales by Region",
        "x_axis": {"label": "Region", "unit": None},
        "y_axis": {"label": "Revenue", "unit": "USD"},
        "series": [
            {
                "name": "Q2",
                "data": [
                    {"label": "North America", "value": 2.4},
                    {"label": "Europe", "value": 2.0},
                ],
            },
            {
                "name": "Q3",
                "data": [
                    {"label": "North America", "value": 3.28},
                    {"label": "Europe", "value": 2.31},
                ],
            },
        ],
    }
    validated = validate_visualization(data)
    assert validated is not None
    assert validated["type"] == "bar"
    assert validated["title"] == "Q2 vs Q3 Sales by Region"
    assert len(validated["series"]) == 2
    assert validated["series"][0]["data"][0]["value"] == 2.4


def test_line_chart_validation():
    data = {
        "type": "line",
        "title": "Monthly Revenue",
        "x_axis": {"label": "Month", "unit": None},
        "y_axis": {"label": "Revenue", "unit": "USD"},
        "series": [
            {
                "name": "Revenue",
                "data": [
                    {"label": "July", "value": 2.6},
                    {"label": "August", "value": 2.8},
                    {"label": "September", "value": 3.02},
                ],
            }
        ],
    }
    validated = validate_visualization(data)
    assert validated is not None
    assert validated["type"] == "line"
    assert validated["title"] == "Monthly Revenue"
    assert len(validated["series"][0]["data"]) == 3


def test_pie_chart_validation():
    data = {
        "type": "pie",
        "title": "Revenue by Region",
        "unit": "USD",
        "data": [
            {"label": "North America", "value": 3.28},
            {"label": "Europe", "value": 2.31},
            {"label": "APAC", "value": 1.94},
        ],
    }
    validated = validate_visualization(data)
    assert validated is not None
    assert validated["type"] == "pie"
    assert len(validated["data"]) == 3


def test_area_chart_validation():
    data = {
        "type": "area",
        "title": "Quarterly Revenue Trend",
        "x_axis": {"label": "Quarter", "unit": None},
        "y_axis": {"label": "Revenue", "unit": "USD"},
        "series": [
            {
                "name": "Revenue",
                "data": [
                    {"label": "Q1", "value": 7.1},
                    {"label": "Q2", "value": 7.57},
                    {"label": "Q3", "value": 8.42},
                ],
            }
        ],
    }
    validated = validate_visualization(data)
    assert validated is not None
    assert validated["type"] == "area"


def test_table_validation():
    data = {
        "type": "table",
        "title": "Sales Pipeline",
        "columns": [
            {"key": "account", "label": "Account", "type": "string"},
            {"key": "value", "label": "Pipeline Value", "type": "number", "unit": "USD"},
            {"key": "probability", "label": "Probability", "type": "percent"},
        ],
        "rows": [
            {"account": "Vertex Retail Group", "value": 920000, "probability": 70},
            {"account": "ABC Technologies", "value": 480000, "probability": 65},
        ],
    }
    validated = validate_visualization(data)
    assert validated is not None
    assert validated["type"] == "table"
    assert len(validated["rows"]) == 2


def test_network_graph_validation():
    data = {
        "type": "network",
        "title": "ABC Company Sales Relationships",
        "nodes": [
            {"id": "abc-company", "label": "ABC Company", "type": "company"},
            {"id": "sales-team", "label": "Sales", "type": "department"},
            {"id": "pricing-discussion", "label": "Pricing Discussion", "type": "topic"},
        ],
        "edges": [
            {"source": "abc-company", "target": "sales-team", "label": "worked_with"},
            {"source": "abc-company", "target": "pricing-discussion", "label": "discussed"},
        ],
    }
    validated = validate_visualization(data)
    assert validated is not None
    assert validated["type"] == "network"
    assert len(validated["nodes"]) == 3
    assert len(validated["edges"]) == 2


def test_timeline_validation():
    data = {
        "type": "timeline",
        "title": "ABC Company Sales Timeline",
        "events": [
            {"date": "2026-07-10", "label": "Initial discussion", "description": "ABC Company requested pricing information."},
            {"date": "2026-08-02", "label": "Commercial negotiation", "description": "Sales discussed discount terms."},
        ],
    }
    validated = validate_visualization(data)
    assert validated is not None
    assert validated["type"] == "timeline"
    assert len(validated["events"]) == 2


def test_invalid_visualizations_graceful_null():
    # 1. Invalid type
    assert validate_visualization({"type": "scatter", "title": "Scatter"}) is None

    # 2. Malformed numeric values (string instead of number)
    malformed_bar = {
        "type": "bar",
        "title": "Bar",
        "x_axis": {"label": "X"},
        "y_axis": {"label": "Y"},
        "series": [{"name": "S1", "data": [{"label": "A", "value": "$100"}]}],
    }
    assert validate_visualization(malformed_bar) is None

    # 3. Pie chart with fewer than 2 items
    malformed_pie = {
        "type": "pie",
        "title": "Pie",
        "data": [{"label": "Only 1", "value": 50}],
    }
    assert validate_visualization(malformed_pie) is None

    # 4. Pie chart with negative value
    negative_pie = {
        "type": "pie",
        "title": "Pie",
        "data": [{"label": "A", "value": -10}, {"label": "B", "value": 20}],
    }
    assert validate_visualization(negative_pie) is None

    # 5. Network with duplicate node IDs
    duplicate_nodes_net = {
        "type": "network",
        "title": "Net",
        "nodes": [
            {"id": "node-1", "label": "N1", "type": "company"},
            {"id": "node-1", "label": "N1 dup", "type": "person"},
        ],
        "edges": [],
    }
    assert validate_visualization(duplicate_nodes_net) is None

    # 6. Network with edge pointing to non-existent node
    invalid_edge_net = {
        "type": "network",
        "title": "Net",
        "nodes": [
            {"id": "node-1", "label": "N1", "type": "company"},
        ],
        "edges": [{"source": "node-1", "target": "missing-node", "label": "link"}],
    }
    assert validate_visualization(invalid_edge_net) is None

    # 7. Table missing required column key in row
    malformed_table = {
        "type": "table",
        "title": "Table",
        "columns": [
            {"key": "col1", "label": "Col 1", "type": "string"},
            {"key": "col2", "label": "Col 2", "type": "number"},
        ],
        "rows": [{"col1": "val1"}],  # missing col2
    }
    assert validate_visualization(malformed_table) is None

    # 8. None input returns None
    assert validate_visualization(None) is None


def test_parse_llm_structured_output():
    # 1. JSON in markdown block
    llm_output_markdown = """```json
{
  "answer": "Q3 revenue grew by 15% across all regions [1].",
  "visualization": {
    "type": "bar",
    "title": "Q3 Revenue Comparison",
    "x_axis": {"label": "Region"},
    "y_axis": {"label": "Revenue", "unit": "USD"},
    "series": [
      {"name": "Q3", "data": [{"label": "North America", "value": 3.28}]}
    ]
  }
}
```"""
    ans, viz = _parse_llm_structured_output(llm_output_markdown)
    assert "Q3 revenue grew by 15%" in ans
    assert viz is not None
    assert viz["type"] == "bar"

    # 2. JSON with null visualization
    llm_output_null_viz = """{
  "answer": "Q3 revenue was 3.28 million USD [1].",
  "visualization": null
}"""
    ans, viz = _parse_llm_structured_output(llm_output_null_viz)
    assert ans == "Q3 revenue was 3.28 million USD [1]."
    assert viz is None

    # 3. Plain text fallback when LLM does not return JSON
    raw_prose = "Q3 revenue was $3.28M according to [1]."
    ans, viz = _parse_llm_structured_output(raw_prose)
    assert ans == raw_prose
    assert viz is None

    # 4. JSON with invalid visualization degrades gracefully
    invalid_viz_json = """{
  "answer": "Here is the sales analysis [1].",
  "visualization": {
    "type": "unknown_chart_type",
    "foo": "bar"
  }
}"""
    ans, viz = _parse_llm_structured_output(invalid_viz_json)
    assert ans == "Here is the sales analysis [1]."
    assert viz is None


def test_citations_preserved_with_structured_answer():
    chunk = RetrievedChunk(
        chunk_id="chunk-123",
        document_id="doc-456",
        department_id="dept-789",
        file_name="Q3_Sales_Report.pdf",
        chunk_index=0,
        text="North America generated 3.28M in Q3 revenue.",
        page_number=4,
    )
    raw_answer = "North America generated 3.28M in Q3 revenue [1]."
    rendered_ans, citations = validate_and_render_citations(
        raw_answer=raw_answer,
        chunks=[chunk],
        is_conflict=False,
    )
    assert "[1]" in rendered_ans
    assert len(citations) == 1
    assert "Q3_Sales_Report.pdf" in citations[0]


# ============================================================
# 2. MOCK LLM & GENERATION SERVICE INTEGRATION TESTS
# ============================================================

class MockLLMProvider(LLMProvider):
    def __init__(self, response_text: str):
        self.response_text = response_text

    async def generate(self, messages, stage=None, **kwargs):
        return self.response_text, {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


@pytest_asyncio.fixture
async def db_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_generation_service_numeric_query_bar_chart(db_session: AsyncSession):
    llm_payload = {
        "answer": "In Q3, North America generated 3.28M and Europe generated 2.31M [1].",
        "visualization": {
            "type": "bar",
            "title": "Q2 vs Q3 Sales by Region",
            "x_axis": {"label": "Region"},
            "y_axis": {"label": "Revenue", "unit": "USD"},
            "series": [
                {
                    "name": "Q3",
                    "data": [
                        {"label": "North America", "value": 3.28},
                        {"label": "Europe", "value": 2.31},
                    ],
                }
            ],
        },
    }
    mock_llm = MockLLMProvider(json.dumps(llm_payload))
    service = GenerationService(db_session)
    dept_id = uuid.uuid4()

    chunk = RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        department_id=str(dept_id),
        file_name="Sales_Report.md",
        chunk_index=0,
        text="North America generated 3.28M and Europe generated 2.31M in Q3.",
    )

    resp = await service.generate_from_context(
        query="Compare Q2 and Q3 sales by region",
        chunks=[chunk],
        allowed_departments=[dept_id],
        llm=mock_llm,
    )

    assert "North America generated 3.28M" in resp.answer
    assert "[1]" in resp.answer
    assert len(resp.citations) == 1
    assert resp.visualization is not None
    assert resp.visualization["type"] == "bar"
    assert resp.visualization["title"] == "Q2 vs Q3 Sales by Region"


@pytest.mark.asyncio
async def test_generation_service_relationship_query_network_graph(db_session: AsyncSession):
    llm_payload = {
        "answer": "ABC Technologies worked with the Sales team regarding pricing [1].",
        "visualization": {
            "type": "network",
            "title": "ABC Technologies Sales Relationships",
            "nodes": [
                {"id": "abc-tech", "label": "ABC Technologies", "type": "company"},
                {"id": "sales-team", "label": "Sales", "type": "department"},
                {"id": "pricing", "label": "Pricing Discussion", "type": "topic"},
            ],
            "edges": [
                {"source": "abc-tech", "target": "sales-team", "label": "worked_with"},
                {"source": "abc-tech", "target": "pricing", "label": "discussed"},
            ],
        },
    }
    mock_llm = MockLLMProvider(json.dumps(llm_payload))
    service = GenerationService(db_session)
    dept_id = uuid.uuid4()

    chunk = RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        department_id=str(dept_id),
        file_name="Deals.md",
        chunk_index=0,
        text="ABC Technologies worked with the Sales team regarding pricing.",
    )

    resp = await service.generate_from_context(
        query="Show the relationship between ABC Technologies, Sales, and pricing",
        chunks=[chunk],
        allowed_departments=[dept_id],
        llm=mock_llm,
    )

    assert resp.visualization is not None
    assert resp.visualization["type"] == "network"
    assert len(resp.visualization["nodes"]) == 3
    assert len(resp.visualization["edges"]) == 2


@pytest.mark.asyncio
async def test_generation_service_timeline_query(db_session: AsyncSession):
    llm_payload = {
        "answer": "ABC Technologies initial discussion was on 2026-07-10 followed by negotiation on 2026-08-02 [1].",
        "visualization": {
            "type": "timeline",
            "title": "ABC Technologies Discussion Timeline",
            "events": [
                {"date": "2026-07-10", "label": "Initial discussion", "description": "Pricing discussion started"},
                {"date": "2026-08-02", "label": "Negotiation", "description": "Discount terms agreed"},
            ],
        },
    }
    mock_llm = MockLLMProvider(json.dumps(llm_payload))
    service = GenerationService(db_session)
    dept_id = uuid.uuid4()

    chunk = RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        department_id=str(dept_id),
        file_name="Timeline.md",
        chunk_index=0,
        text="Timeline: 2026-07-10 initial discussion, 2026-08-02 negotiation.",
    )

    resp = await service.generate_from_context(
        query="What is the timeline of ABC Technologies discussions?",
        chunks=[chunk],
        allowed_departments=[dept_id],
        llm=mock_llm,
    )

    assert resp.visualization is not None
    assert resp.visualization["type"] == "timeline"
    assert len(resp.visualization["events"]) == 2


@pytest.mark.asyncio
async def test_unauthorized_chunk_security_barrier(db_session: AsyncSession):
    mock_llm = MockLLMProvider("Normal response")
    service = GenerationService(db_session)

    authorized_dept = uuid.uuid4()
    unauthorized_dept = uuid.uuid4()

    unauthorized_chunk = RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        department_id=str(unauthorized_dept),  # Unauthorized!
        file_name="Secret.md",
        chunk_index=0,
        text="Confidential revenue data.",
    )

    with pytest.raises(RuntimeError, match="Security violation: Unauthorized chunk detected in final context"):
        await service.generate_from_context(
            query="Show secret revenue",
            chunks=[unauthorized_chunk],
            allowed_departments=[authorized_dept],
            llm=mock_llm,
        )


# ============================================================
# 3. DATABASE PERSISTENCE & HISTORY TESTS
# ============================================================

@pytest.mark.asyncio
async def test_conversation_persistence_with_visualization(db_session: AsyncSession):
    session = db_session
    now = datetime.now(timezone.utc)

    # 1. Setup user
    test_user = User(
        id=uuid.uuid4(),
        email=f"viz_test_{uuid.uuid4().hex[:6]}@example.com",
        hashed_password="hash",
        role="USER",
        is_active=True,
        created_at=now,
    )
    session.add(test_user)
    await session.flush()

    conv_service = ConversationService(session)

    # 2. Create conversation
    conv = await conv_service.get_or_create_conversation(
        user=test_user,
        conversation_id=None,
        initial_query="Compare Q2 and Q3 sales by region",
        workspace_id=None,
    )

    # 3. Save User message
    await conv_service.save_user_message(
        conversation_id=conv.id,
        content="Compare Q2 and Q3 sales by region",
    )

    # 4. Save Assistant message with visualization
    viz_payload = {
        "type": "bar",
        "title": "Q2 vs Q3 Sales",
        "x_axis": {"label": "Region"},
        "y_axis": {"label": "Revenue", "unit": "USD"},
        "series": [{"name": "Q3", "data": [{"label": "NA", "value": 3.28}]}],
    }
    sources_payload = [{"title": "Sales.md", "excerpt": "Sales summary"}]

    asst_msg = await conv_service.save_assistant_message(
        conversation_id=conv.id,
        content="Q3 outperformed Q2 [1].",
        sources=sources_payload,
        execution_metrics={"total_latency_ms": 120.5},
        visualization=viz_payload,
    )

    assert asst_msg.id is not None
    assert asst_msg.visualization == viz_payload

    # 5. Fetch detail
    detail = await conv_service.get_conversation_detail(conv.id, test_user)
    assert len(detail.messages) == 2
    asst_resp = next(m for m in detail.messages if m.role == "assistant")
    assert asst_resp.content == "Q3 outperformed Q2 [1]."
    assert asst_resp.visualization == viz_payload

    # 6. Check context window maintains 10-message max turns without losing DB history
    context_window = await conv_service.get_conversation_context_window(conv.id, max_turns=5)
    assert len(context_window) == 2
    assert context_window[0]["role"] == "user"
    assert context_window[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_regression_q3_2026_revenue_by_region_bar_chart(db_session: AsyncSession):
    """
    Regression Test for User Query:
    'Show me Q3 2026 revenue by region as a bar chart.'
    Must include all 4 regions:
    - North America: 3.28M
    - Europe: 2.31M
    - APAC: 1.94M
    - Latin America: 0.89M
    """
    llm_payload = {
        "answer": "Here is the Q3 2026 revenue by region:\n- North America: $3.28M [1]\n- Europe: $2.31M [1]\n- APAC: $1.94M [1]\n- Latin America: $0.89M [1]",
        "visualization": {
            "type": "bar",
            "title": "Q3 2026 Revenue by Region",
            "x_axis": {"label": "Region", "unit": None},
            "y_axis": {"label": "Revenue", "unit": "USD"},
            "series": [
                {
                    "name": "Q3 2026 Revenue",
                    "data": [
                        {"label": "North America", "value": 3.28},
                        {"label": "Europe", "value": 2.31},
                        {"label": "APAC", "value": 1.94},
                        {"label": "Latin America", "value": 0.89},
                    ],
                }
            ],
        },
    }
    mock_llm = MockLLMProvider(json.dumps(llm_payload))
    service = GenerationService(db_session)
    dept_id = uuid.uuid4()

    chunk = RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        department_id=str(dept_id),
        file_name="Q3_Sales_Performance_Pipeline_Report.md",
        chunk_index=0,
        text="North America: $3.28M, Europe: $2.31M, APAC: $1.94M, Latin America: $0.89M.",
    )

    resp = await service.generate_from_context(
        query="Show me Q3 2026 revenue by region as a bar chart.",
        chunks=[chunk],
        allowed_departments=[dept_id],
        llm=mock_llm,
    )

    assert resp.visualization is not None
    assert resp.visualization["type"] == "bar"
    assert resp.visualization["title"] == "Q3 2026 Revenue by Region"

    series_data = resp.visualization["series"][0]["data"]
    region_map = {item["label"]: item["value"] for item in series_data}

    assert "North America" in region_map
    assert "Europe" in region_map
    assert "APAC" in region_map
    assert "Latin America" in region_map

    assert region_map["North America"] == 3.28
    assert region_map["Europe"] == 2.31
    assert region_map["APAC"] == 1.94
    assert region_map["Latin America"] == 0.89

    assert "Latin America: $0.89M" in resp.answer


@pytest.mark.asyncio
async def test_regression_q3_sales_pipeline_opportunities_bar_chart(db_session: AsyncSession):
    """
    Regression Test for User Query:
    'Visualize the Q3 sales pipeline opportunities by value as a bar chart.'
    Must include all 6 opportunities and their values:
    1. Vertex Enterprise Platform — $920K (920)
    2. Northstar Analytics Expansion — $650K (650)
    3. BluePeak Fintech Platform — $550K (550)
    4. ABC Automation Expansion — $480K (480)
    5. Meridian Healthcare Upgrade — $440K (440)
    6. Orion Logistics Renewal — $390K (390)
    """
    llm_payload = {
        "answer": "Here is the breakdown of the Q3 sales pipeline opportunities by value:\n- Vertex Enterprise Platform: $920K [1]\n- Northstar Analytics Expansion: $650K [1]\n- BluePeak Fintech Platform: $550K [1]\n- ABC Automation Expansion: $480K [1]\n- Meridian Healthcare Upgrade: $440K [1]\n- Orion Logistics Renewal: $390K [1]",
        "visualization": {
            "type": "bar",
            "title": "Q3 Sales Pipeline Opportunities by Value",
            "x_axis": {"label": "Opportunity", "unit": None},
            "y_axis": {"label": "Value", "unit": "USD (Thousands)"},
            "series": [
                {
                    "name": "Opportunity Value",
                    "data": [
                        {"label": "Vertex Enterprise Platform", "value": 920},
                        {"label": "Northstar Analytics Expansion", "value": 650},
                        {"label": "BluePeak Fintech Platform", "value": 550},
                        {"label": "ABC Automation Expansion", "value": 480},
                        {"label": "Meridian Healthcare Upgrade", "value": 440},
                        {"label": "Orion Logistics Renewal", "value": 390},
                    ],
                }
            ],
        },
    }
    mock_llm = MockLLMProvider(json.dumps(llm_payload))
    service = GenerationService(db_session)
    dept_id = uuid.uuid4()

    pipeline_table_text = """| Opportunity | Account | Region | Stage | Value | Probability | Expected Close |
|---|---|---|---|---:|---:|---|
| Vertex Enterprise Platform | Vertex Retail Group | Europe | Negotiation | $920K | 70% | Oct 2026 |
| ABC Automation Expansion | ABC Technologies | North America | Proposal | $480K | 65% | Nov 2026 |
| Northstar Analytics Expansion | Northstar Manufacturing | North America | Discovery | $650K | 40% | Dec 2026 |
| Orion Logistics Renewal | Orion Logistics | APAC | Negotiation | $390K | 80% | Oct 2026 |
| BluePeak Fintech Platform | BluePeak Financial | APAC | Qualification | $550K | 30% | Dec 2026 |
| Meridian Healthcare Upgrade | Meridian Health | Europe | Proposal | $440K | 55% | Nov 2026 | 
**Total pipeline:** $3.43M
**Weighted pipeline:** approximately **$1.95M**"""

    chunk = RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        department_id=str(dept_id),
        file_name="Q3_Sales_Performance_Pipeline_Report.md",
        chunk_index=10,
        text=pipeline_table_text,
    )

    resp = await service.generate_from_context(
        query="Visualize the Q3 sales pipeline opportunities by value as a bar chart.",
        chunks=[chunk],
        allowed_departments=[dept_id],
        llm=mock_llm,
    )

    assert resp.visualization is not None
    assert resp.visualization["type"] == "bar"
    assert resp.visualization["title"] == "Q3 Sales Pipeline Opportunities by Value"

    series_data = resp.visualization["series"][0]["data"]
    assert len(series_data) == 6

    opp_map = {item["label"]: item["value"] for item in series_data}
    assert "Vertex Enterprise Platform" in opp_map
    assert "Northstar Analytics Expansion" in opp_map
    assert "BluePeak Fintech Platform" in opp_map
    assert "ABC Automation Expansion" in opp_map
    assert "Meridian Healthcare Upgrade" in opp_map
    assert "Orion Logistics Renewal" in opp_map

    assert opp_map["Vertex Enterprise Platform"] in [920, 920000, 0.92]
    assert opp_map["Northstar Analytics Expansion"] in [650, 650000, 0.65]
    assert opp_map["BluePeak Fintech Platform"] in [550, 550000, 0.55]
    assert opp_map["ABC Automation Expansion"] in [480, 480000, 0.48]
    assert opp_map["Meridian Healthcare Upgrade"] in [440, 440000, 0.44]
    assert opp_map["Orion Logistics Renewal"] in [390, 390000, 0.39]

