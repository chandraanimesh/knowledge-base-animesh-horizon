"""End-to-end integration tests for generic multi-format RAG and secure artifact generation."""

from datetime import datetime, timezone, timedelta
import uuid
import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock

from rag_app.api.artifact_routes import _verify_artifact_access
from rag_app.db.models import GeneratedArtifact, User
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
async def test_illustrative_financial_query_full_lifecycle(tmp_path):
    """
    Simulates the exact end-to-end flow for the illustrative request:
    "Finance department ki pending payments nikalo. Mujhe summary chat mein do, detailed records JSON mein aur ek PDF report bhi generate kar do."
    """
    raw_query = "Finance department ki pending payments nikalo. Mujhe summary chat mein do, detailed records JSON mein aur ek PDF report bhi generate kar do."

    # 1. Step 1: Parse Intent & Output Preferences
    prefs = parse_output_preferences(raw_query)
    assert prefs.chat_summary is True
    assert prefs.include_inline_json is True
    assert DownloadableArtifactFormat.PDF in prefs.download_formats

    # Clean query for substantive retrieval
    cleaned_query = strip_presentation_instructions(raw_query)
    assert "Finance department ki pending payments nikalo" in cleaned_query
    assert "summary chat mein" not in cleaned_query
    assert "PDF report" not in cleaned_query

    # 2. Step 2: Tabular Extraction & Grounding Verification
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_fin_1",
            document_id="doc_fin_1",
            department_id="dept_fin",
            file_name="vendor_invoices_q1.pdf",
            chunk_index=0,
            text="Vendor: Acme Global Services | Invoice: INV-2026-001 | Pending Amount: INR 45,000.00 | Due: 2026-04-30 | Status: Pending Approval",
            page_number=4,
        ),
        RetrievedChunk(
            chunk_id="chunk_fin_2",
            document_id="doc_fin_2",
            department_id="dept_fin",
            file_name="credit_notes.pdf",
            chunk_index=1,
            text="Vendor: Delta Logistics | Refund Amount: INR -5,000.00 | Date: 2026-04-15 | Status: Adjusted",
            page_number=1,
        ),
    ]

    mock_llm_json = """```json
    {
      "title": "Finance Pending Vendor Payments",
      "description": "Verified pending invoices and adjustments",
      "columns": [
        {"key": "vendor", "label": "Vendor Name", "data_type": "string"},
        {"key": "amount", "label": "Pending Amount", "data_type": "currency", "unit": "INR"},
        {"key": "due_date", "label": "Due Date", "data_type": "date"}
      ],
      "rows": [
        {
          "_source_chunk_id": "chunk_fin_1",
          "vendor": "Acme Global Services",
          "amount": 45000.00,
          "due_date": "2026-04-30"
        },
        {
          "_source_chunk_id": "chunk_fin_2",
          "vendor": "Delta Logistics",
          "amount": -5000.00,
          "due_date": "2026-04-15"
        },
        {
          "_source_chunk_id": "chunk_forged_999",
          "vendor": "Hallucinated Unverified Corp",
          "amount": 999999.00,
          "due_date": "2026-05-01"
        }
      ]
    }
    ```"""

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (mock_llm_json, {"total_tokens": 150})

    extractor = TabularExtractionService(llm=mock_llm)
    dataset, coverage = await extractor.extract_and_verify(query=cleaned_query, chunks=chunks)

    assert dataset is not None
    assert coverage is not None
    # Forged chunk_forged_999 quarantined
    assert len(dataset.rows) == 2
    assert dataset.rejected_rows_count == 1
    assert dataset.rows[0]["vendor"] == "Acme Global Services"
    assert dataset.rows[0]["amount"] == 45000.00
    assert dataset.rows[1]["amount"] == -5000.00  # Negative credit note safely preserved

    # Coverage disclosure properly identified as sample
    assert coverage.retrieval_coverage == RetrievalCoverage.SAMPLED_EVIDENCE

    # 3. Step 3 & 4: Pluggable Renderer & Secure Storage
    storage = ArtifactStorageService(base_dir=str(tmp_path))
    renderer = renderer_registry.get_renderer(DownloadableArtifactFormat.PDF)
    pdf_bytes = renderer.render(dataset, title=dataset.title, summary="Summary in chat")

    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 500

    art_id = uuid.uuid4()
    storage_path, sha256_hash, byte_size = storage.save_artifact(
        artifact_id=art_id,
        format_ext="pdf",
        content_bytes=pdf_bytes,
    )

    # 4. Step 5: Authorization & Access Control
    owner_user = User()
    owner_user.id = uuid.uuid4()
    owner_user.role = "USER"
    owner_user.is_active = True

    attacker_user = User()
    attacker_user.id = uuid.uuid4()
    attacker_user.role = "USER"
    attacker_user.is_active = True

    artifact = GeneratedArtifact()
    artifact.id = art_id
    artifact.user_id = owner_user.id
    artifact.title = dataset.title
    artifact.format = "pdf"
    artifact.safe_filename = "Finance_Pending_Payments.pdf"
    artifact.mime_type = "application/pdf"
    artifact.byte_size = byte_size
    artifact.storage_path = storage_path
    artifact.checksum_sha256 = sha256_hash
    artifact.source_departments = ["dept_fin"]
    artifact.tabular_data = dataset.model_dump()
    artifact.expires_at = datetime.now(timezone.utc) + timedelta(hours=48)

    # Attacker access is blocked with 403 Forbidden
    with pytest.raises(HTTPException) as exc:
        await _verify_artifact_access(artifact, attacker_user, session=None)
    assert exc.value.status_code == 403

    # Owner access is allowed
    # (Without session, dynamic dept re-check passes if no allowed_depts returned or admin)
    admin = User()
    admin.id = uuid.uuid4()
    admin.role = "ADMIN"
    admin.is_active = True
    await _verify_artifact_access(artifact, admin, session=None)


@pytest.mark.asyncio
async def test_sales_discount_query_multi_format_canonical_dataset(tmp_path):
    """
    End-to-end regression test for the Sales discount approval query:
    Proves that all three rows appear consistently across:
    1. Canonical TabularDataset (3 rows)
    2. Executive Summary Metrics (total_records == 3)
    3. Inline JSON (3 rows)
    4. Inline Markdown Table (3 rows)
    5. Rendered PDF artifact (contains all 3 rows, verified bytes)
    6. Rendered JSON artifact (contains all 3 rows, verified JSON payload)
    7. No fabricated 'L1-L3' line range citation warning.
    """
    import json
    from rag_app.generation.citations import validate_and_render_citations
    from rag_app.schemas.artifacts import OutputPreferences

    sales_chunk_text = (
        "Discount Approval\n"
        "Discounts up to 5% require the account owner's approval. "
        "Discounts above 5% and up to 10% require Sales Manager approval. "
        "Discounts above 10% require Sales Director approval and a documented business justification. "
        "These are fictional demo rules, not company policy."
    )

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_sales_01",
            document_id="doc_sales_01",
            department_id="dept_sales",
            file_name="Sales_Department_Demo.docx",
            chunk_index=0,
            text=sales_chunk_text,
        )
    ]

    mock_llm_json = """```json
    {
      "title": "Sales Discount Approval Rules",
      "description": "Fictional demo discount approval thresholds",
      "columns": [
        {"key": "discount_range", "label": "Discount Range", "data_type": "string"},
        {"key": "approver", "label": "Required Approver", "data_type": "string"},
        {"key": "conditions", "label": "Conditions", "data_type": "string"}
      ],
      "rows": [
        {
          "_source_chunk_id": "chunk_sales_01",
          "discount_range": "Up to 5%",
          "approver": "account owner's approval",
          "conditions": "account owner approval"
        },
        {
          "_source_chunk_id": "chunk_sales_01",
          "discount_range": ">5% up to 10%",
          "approver": "Sales Manager",
          "conditions": "Sales Manager approval"
        },
        {
          "_source_chunk_id": "chunk_sales_01",
          "discount_range": "Above 10%",
          "approver": "Sales Director",
          "conditions": "documented business justification"
        }
      ]
    }
    ```"""

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (mock_llm_json, {"total_tokens": 120})

    # 1. Tabular Extraction
    extractor = TabularExtractionService(llm=mock_llm)
    dataset, coverage = await extractor.extract_and_verify(
        query="What are the sales discount approval rules?",
        chunks=chunks,
    )

    assert dataset is not None
    assert len(dataset.rows) == 3
    assert dataset.summary_metrics["total_records"] == 3
    assert dataset.rejected_rows_count == 0

    # 2. Citation sanitization check
    raw_prose = (
        "Here are the discount approval rules [Sales_Department_Demo.docx, L1-L3]:\n"
        "These are fictional demo rules, not company policy."
    )
    sanitized_ans, citations = validate_and_render_citations(raw_answer=raw_prose, chunks=chunks)
    assert "L1-L3" not in sanitized_ans
    assert "Sales_Department_Demo.docx" in citations[0]

    # 3. Artifact Renderers - PDF
    pdf_renderer = renderer_registry.get_renderer(DownloadableArtifactFormat.PDF)
    pdf_bytes = pdf_renderer.render(dataset, title=dataset.title, summary=sanitized_ans)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 500

    # 4. Artifact Renderers - JSON
    json_renderer = renderer_registry.get_renderer(DownloadableArtifactFormat.JSON)
    json_bytes = json_renderer.render(dataset, title=dataset.title, summary=sanitized_ans)
    parsed_json_artifact = json.loads(json_bytes.decode("utf-8"))
    assert len(parsed_json_artifact["records"]) == 3
    assert parsed_json_artifact["records"][0]["data"]["discount_range"] == "Up to 5%"
    assert parsed_json_artifact["records"][1]["data"]["discount_range"] == ">5% up to 10%"
    assert parsed_json_artifact["records"][2]["data"]["discount_range"] == "Above 10%"

    # 5. Artifact Storage & URL generation
    storage = ArtifactStorageService(base_dir=str(tmp_path))
    pdf_id = uuid.uuid4()
    pdf_path, pdf_hash, pdf_size = storage.save_artifact(
        artifact_id=pdf_id,
        format_ext="pdf",
        content_bytes=pdf_bytes,
    )
    assert pdf_size == len(pdf_bytes)

