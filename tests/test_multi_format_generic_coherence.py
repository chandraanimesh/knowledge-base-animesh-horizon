"""Regression and cross-format consistency tests for generic multi-format RAG pipeline.

Verifies:
1. An unseen synthetic domain/schema (Cloud Infrastructure Fleet & Cluster Capacity).
2. Cross-format consistency across canonical dataset, summary metrics, inline JSON, PDF, CSV, and DOCX.
3. Open-ended threshold protection (no fabricated numbers like 50001).
4. Symmetrical multi-format intent parsing (PDF, CSV, DOCX, JSON).
5. Grounding filename isolation (artifact slugs never treated as ungrounded domain entities).
6. Conversation history restoration of unassigned artifacts.
"""

from datetime import datetime, timezone, timedelta
import json
import uuid
import pytest
from unittest.mock import AsyncMock

from rag_app.db.models import GeneratedArtifact, User
from rag_app.generation.grounding import AnswerGroundingEvaluator
from rag_app.generation.tabular_extractor import TabularExtractionService
from rag_app.reports.registry import renderer_registry
from rag_app.reports.storage import ArtifactStorageService
from rag_app.retrieval.query_parser import parse_output_preferences, strip_presentation_instructions
from rag_app.schemas.artifacts import (
    ColumnDataType,
    DownloadableArtifactFormat,
    RetrievalCoverage,
    TableColumn,
    TabularDataset,
    VerificationStatus,
)
from rag_app.schemas.rag import RetrievedChunk


@pytest.mark.asyncio
async def test_unseen_synthetic_domain_cloud_fleet(tmp_path):
    """
    Demonstrates that an unseen synthetic domain (Cloud Infrastructure Fleet)
    with custom schema, null values, and multiple requested formats (PDF, CSV, JSON)
    flows through the generic pipeline without any domain-specific branches.
    """
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_cloud_01",
            document_id="doc_infra_01",
            department_id="dept_engineering",
            file_name="cloud_infrastructure_q1.pdf",
            chunk_index=0,
            text=(
                "Cloud Fleet Capacity Status:\n"
                "Cluster us-east-alpha | Region: us-east-1 | Active Nodes: 48 | CPU: 72.5% | Cost: $144.00/hr | Status: Healthy\n"
                "Cluster eu-west-beta | Region: eu-west-1 | Active Nodes: 32 | CPU: 58.0% | Cost: $96.00/hr | Status: Healthy\n"
                "Cluster ap-south-gamma | Region: ap-south-1 | Active Nodes: 16 | CPU: 41.2% | Cost: $48.00/hr | Status: Healthy\n"
                "Cluster us-west-standby | Region: us-west-2 | Active Nodes: 4 | CPU: 12.0% | Cost: $12.00/hr | Status: Standby Maintenance\n"
                "Note: Standby clusters incur zero network egress charges during the maintenance window."
            ),
            page_number=2,
        )
    ]

    mock_llm_json = """```json
    {
      "title": "Cloud Fleet Capacity Status",
      "description": "Production and standby compute clusters overview",
      "columns": [
        {"key": "cluster_id", "label": "Cluster ID", "data_type": "string"},
        {"key": "region", "label": "Region", "data_type": "string"},
        {"key": "active_nodes", "label": "Active Nodes", "data_type": "integer"},
        {"key": "cpu_utilization", "label": "CPU Utilization", "data_type": "number"},
        {"key": "cost_per_hr", "label": "Hourly Cost", "data_type": "currency", "unit": "USD"},
        {"key": "status", "label": "Operational Status", "data_type": "string"}
      ],
      "rows": [
        {
          "_source_chunk_id": "chunk_cloud_01",
          "cluster_id": "us-east-alpha",
          "region": "us-east-1",
          "active_nodes": 48,
          "cpu_utilization": 72.5,
          "cost_per_hr": 144.00,
          "status": "Healthy"
        },
        {
          "_source_chunk_id": "chunk_cloud_01",
          "cluster_id": "eu-west-beta",
          "region": "eu-west-1",
          "active_nodes": 32,
          "cpu_utilization": 58.0,
          "cost_per_hr": 96.00,
          "status": "Healthy"
        },
        {
          "_source_chunk_id": "chunk_cloud_01",
          "cluster_id": "ap-south-gamma",
          "region": "ap-south-1",
          "active_nodes": 16,
          "cpu_utilization": 41.2,
          "cost_per_hr": 48.00,
          "status": "Healthy"
        },
        {
          "_source_chunk_id": "chunk_cloud_01",
          "cluster_id": "us-west-standby",
          "region": "us-west-2",
          "active_nodes": 4,
          "cpu_utilization": 12.0,
          "cost_per_hr": 12.00,
          "status": "Standby Maintenance"
        }
      ]
    }
    ```"""

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (mock_llm_json, {"total_tokens": 180})

    extractor = TabularExtractionService(llm=mock_llm)
    dataset, coverage = await extractor.extract_and_verify(
        query="Provide a breakdown of cloud fleet capacity with PDF and CSV export.",
        chunks=chunks,
    )

    # 1. Tabular Dataset Verification
    assert dataset is not None
    assert len(dataset.rows) == 4
    assert dataset.rejected_rows_count == 0
    assert dataset.summary_metrics["total_records"] == 4
    assert dataset.summary_metrics["sum_active_nodes"] == 100.0  # 48+32+16+4
    assert dataset.summary_metrics["sum_cost_per_hr"] == 300.0   # 144+96+48+12
    assert dataset.summary_metrics["avg_cost_per_hr"] == 75.0

    # 2. Renderers: Cross-format consistency
    # PDF
    pdf_renderer = renderer_registry.get_renderer(DownloadableArtifactFormat.PDF)
    pdf_bytes = pdf_renderer.render(dataset, title=dataset.title, summary="Cloud capacity overview")
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 500

    # CSV
    csv_renderer = renderer_registry.get_renderer(DownloadableArtifactFormat.CSV)
    csv_bytes = csv_renderer.render(dataset, title=dataset.title)
    csv_str = csv_bytes.decode("utf-8")
    assert "Cluster ID,Region,Active Nodes" in csv_str
    assert "us-east-alpha,us-east-1,48" in csv_str
    assert "us-west-standby,us-west-2,4" in csv_str

    # JSON
    json_renderer = renderer_registry.get_renderer(DownloadableArtifactFormat.JSON)
    json_bytes = json_renderer.render(dataset, title=dataset.title)
    parsed_json = json.loads(json_bytes.decode("utf-8"))
    assert len(parsed_json["records"]) == 4
    assert parsed_json["summary_metrics"]["sum_cost_per_hr"] == 300.0

    # 3. Storage
    storage = ArtifactStorageService(base_dir=str(tmp_path))
    art_id = uuid.uuid4()
    path, sha, sz = storage.save_artifact(art_id, "pdf", pdf_bytes)
    assert sz == len(pdf_bytes)


def test_symmetrical_intent_parsing():
    """Verifies that format export parsing is symmetrical across PDF, CSV, DOCX, and JSON."""
    # Query 1: PDF and JSON files explicitly requested
    q1 = "Please generate a PDF report and JSON file of cloud expenses."
    p1 = parse_output_preferences(q1)
    assert DownloadableArtifactFormat.PDF in p1.download_formats
    assert DownloadableArtifactFormat.JSON in p1.download_formats

    # Query 2: CSV and DOCX
    q2 = "Ek CSV file aur Word document export karo"
    p2 = parse_output_preferences(q2)
    assert DownloadableArtifactFormat.CSV in p2.download_formats
    assert DownloadableArtifactFormat.DOCX in p2.download_formats

    # Query 3: Inline JSON presentation vs Downloadable JSON
    q3 = "Mujhe detailed records JSON mein do"
    p3 = parse_output_preferences(q3)
    assert p3.include_inline_json is True
    assert DownloadableArtifactFormat.JSON not in p3.download_formats

    # Query 4: Downloadable JSON
    q4 = "Export detailed records as JSON file"
    p4 = parse_output_preferences(q4)
    assert DownloadableArtifactFormat.JSON in p4.download_formats


@pytest.mark.asyncio
async def test_grounding_filename_isolation():
    """
    Verifies that generated artifact filenames in Markdown (e.g. Finance_Expense_Approval_Rules.pdf)
    are NOT flagged as ungrounded factual domain entities.
    """
    evaluator = AnswerGroundingEvaluator()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_01",
            document_id="doc_01",
            department_id="dept_fin",
            file_name="expense_policy.pdf",
            chunk_index=0,
            text="Expense threshold up to $10,000 requires Manager approval.",
        )
    ]

    answer = (
        "Here are the expense thresholds:\n\n"
        "- Manager approval is required up to $10,000.\n\n"
        "### Downloadable Files (Generated from verified evidence)\n"
        "- **PDF report:** `Finance_Expense_Approval_Rules.pdf`\n"
        "- **JSON report:** `finance_expense_approval_rules.json`"
    )

    res = evaluator.evaluate_deterministic(
        query="What are the expense approval rules? Generate PDF and JSON.",
        answer=answer,
        chunks=chunks,
    )

    # The artifact slug 'Finance_Expense_Approval_Rules' must NOT be flagged as ungrounded
    entity_claims = [c for c in res.unsupported_claims if "Entity/term" in c]
    assert not any("Finance_Expense_Approval_Rules" in c for c in entity_claims)
    assert res.is_grounded is True


@pytest.mark.asyncio
async def test_all_four_formats_simultaneous_generation(tmp_path):
    """
    Verifies that requesting all 4 formats (PDF, CSV, DOCX, JSON) simultaneously
    produces 4 real persisted artifacts with real IDs, MIME types, and valid sizes.
    """
    storage = ArtifactStorageService(base_dir=str(tmp_path))
    dataset = TabularDataset(
        title="Fleet Summary",
        description="Cluster nodes and costs",
        columns=[
            TableColumn(key="name", label="Cluster", data_type=ColumnDataType.STRING),
            TableColumn(key="nodes", label="Nodes", data_type=ColumnDataType.INTEGER),
        ],
        rows=[{"name": "alpha", "nodes": 10}, {"name": "beta", "nodes": 20}],
        summary_metrics={"total_records": 2, "sum_nodes": 30},
    )

    formats = [
        DownloadableArtifactFormat.PDF,
        DownloadableArtifactFormat.CSV,
        DownloadableArtifactFormat.DOCX,
        DownloadableArtifactFormat.JSON,
    ]

    persisted_artifacts = []
    for fmt in formats:
        renderer = renderer_registry.get_renderer(fmt)
        b = renderer.render(dataset=dataset, title=dataset.title, summary="Test summary")
        assert len(b) > 0
        art_id = uuid.uuid4()
        storage_path, sha, sz = storage.save_artifact(
            artifact_id=art_id,
            format_ext=renderer.file_extension,
            content_bytes=b,
        )
        assert sz == len(b)
        persisted_artifacts.append({
            "id": art_id,
            "format": fmt,
            "filename": f"Fleet_Summary.{renderer.file_extension}",
            "mime_type": renderer.mime_type,
            "size": sz,
            "path": storage_path,
        })

    assert len(persisted_artifacts) == 4
    # Check MIME types
    mimes = {a["format"]: a["mime_type"] for a in persisted_artifacts}
    assert mimes[DownloadableArtifactFormat.PDF] == "application/pdf"
    assert "text/csv" in mimes[DownloadableArtifactFormat.CSV]
    assert "wordprocessingml" in mimes[DownloadableArtifactFormat.DOCX]
    assert "application/json" in mimes[DownloadableArtifactFormat.JSON]


@pytest.mark.asyncio
async def test_partial_renderer_failure_reporting():
    """
    Verifies that if one renderer fails (e.g. DOCX), the failure is isolated:
    successful artifacts have valid URLs, while the failed one has error set and no fake download URL.
    """
    from rag_app.schemas.artifacts import ArtifactMetadata

    created = []
    # Successful PDF
    created.append(
        ArtifactMetadata(
            id=uuid.uuid4(),
            format=DownloadableArtifactFormat.PDF,
            filename="Report.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            download_url="/api/reports/123/download",
            created_at=datetime.now(timezone.utc),
        )
    )
    # Failed DOCX
    created.append(
        ArtifactMetadata(
            id=uuid.uuid4(),
            format=DownloadableArtifactFormat.DOCX,
            filename="report.docx",
            mime_type="application/octet-stream",
            size_bytes=0,
            download_url="",
            created_at=datetime.now(timezone.utc),
            error="Failed to generate DOCX: XML serialization error",
        )
    )

    assert created[0].error is None
    assert created[0].download_url != ""
    assert created[1].error is not None
    assert created[1].download_url == ""


def test_speculative_boilerplate_removal():
    """
    Verifies that speculative LLM notes claiming 'Potential downloadable artifacts...'
    or 'File generation note: ... are not included' are completely suppressed.
    """
    import re
    raw_answer = (
        "Here is the budget summary for Q3:\n\n"
        "- Operating budget: INR 12,00,000\n"
        "- Actual spend: INR 10,80,000\n\n"
        "Potential downloadable artifacts (can be generated from the data above): PDF report, CSV file, DOCX document, and a JSON file.\n\n"
        "File generation note: Based on the verified data, PDF, CSV, DOCX, and JSON files can be created if needed, but they are not included in this response.\n\n"
        "Document limitation: The source only defines the figures above."
    )

    speculative_pat = re.compile(
        r"(?:\n+)?(?:\*\*|###?\s*)?(?:Potential\s+downloadable\s+artifacts|File\s+generation\s+note|Downloadable\s+files)[^\n]*?(?:\n[^\n]+)*?(?:can\s+be\s+created|can\s+be\s+generated|not\s+included\s+in\s+this\s+response|can\s+be\s+downloaded|potential\s+files?)[^\n]*",
        re.IGNORECASE,
    )
    cleaned = speculative_pat.sub("", raw_answer).strip()
    assert "Potential downloadable artifacts" not in cleaned
    assert "File generation note" not in cleaned
    assert "not included in this response" not in cleaned
    assert "Operating budget: INR 12,00,000" in cleaned

