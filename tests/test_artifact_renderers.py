"""Unit tests for multi-format artifact renderers."""

import json
from decimal import Decimal
import pytest

from rag_app.reports.csv_renderer import CSVRenderer
from rag_app.reports.docx_renderer import DOCXRenderer
from rag_app.reports.json_renderer import JSONRenderer
from rag_app.reports.pdf_renderer import PDFRenderer
from rag_app.schemas.artifacts import (
    ColumnDataType,
    CoverageReport,
    ExportTruncation,
    RetrievalCoverage,
    RowProvenance,
    TableColumn,
    TabularDataset,
    VerificationStatus,
)


@pytest.fixture
def sample_dataset() -> TabularDataset:
    columns = [
        TableColumn(key="vendor", label="Vendor", data_type=ColumnDataType.STRING),
        TableColumn(key="amount", label="Amount", data_type=ColumnDataType.CURRENCY),
        TableColumn(key="date", label="Date", data_type=ColumnDataType.DATE),
        TableColumn(key="notes", label="Notes", data_type=ColumnDataType.STRING),
    ]
    rows = [
        {
            "vendor": "Acme Corp",
            "amount": 12500.50,
            "date": "2026-03-15",
            "notes": "Q1 Cloud Infrastructure",
        },
        {
            "vendor": "=CMD|' /C calc'!A0",  # Potential formula injection
            "amount": -500.00,  # Negative credit note
            "date": "2026-03-20",
            "notes": "Credit refund, with comma and \"quotes\"",
        },
    ]
    provenance = {
        "row_0": RowProvenance(
            source_chunk_id="chunk-1",
            document_id="doc-1",
            document_name="invoices_q1.pdf",
            page_number=1,
            verification_status=VerificationStatus.VERIFIED,
            is_grounded=True,
        ),
        "row_1": RowProvenance(
            source_chunk_id="chunk-2",
            document_id="doc-2",
            document_name="credit_notes.pdf",
            page_number=2,
            verification_status=VerificationStatus.VERIFIED,
            is_grounded=True,
        ),
    }
    coverage = CoverageReport(
        retrieval_coverage=RetrievalCoverage.SAMPLED_EVIDENCE,
        export_truncation=ExportTruncation(
            is_truncated=False, total_extracted_rows=2, exported_rows_count=2
        ),
        disclaimer="Sampled evidence disclaimer notice.",
    )
    return TabularDataset(
        title="Finance Pending Payments",
        description="Verified Q1 vendor payments",
        columns=columns,
        rows=rows,
        row_provenance=provenance,
        summary_metrics={"total_records": 2, "sum_amount": 12000.50},
        coverage=coverage,
    )


def test_json_renderer(sample_dataset: TabularDataset):
    renderer = JSONRenderer()
    raw_bytes = renderer.render(sample_dataset, title="Finance Pending Payments")
    
    assert isinstance(raw_bytes, bytes)
    parsed = json.loads(raw_bytes.decode("utf-8"))
    
    assert parsed["title"] == "Finance Pending Payments"
    assert parsed["total_verified_records"] == 2
    assert len(parsed["records"]) == 2
    assert parsed["records"][0]["data"]["vendor"] == "Acme Corp"
    assert parsed["records"][0]["data"]["amount"] == 12500.50
    assert parsed["records"][0]["provenance"]["source_chunk_id"] == "chunk-1"
    assert parsed["coverage"]["retrieval_coverage"] == "sampled_evidence"


def test_csv_renderer_formula_safe(sample_dataset: TabularDataset):
    renderer = CSVRenderer()
    raw_bytes = renderer.render(sample_dataset, title="Finance Pending Payments")
    
    csv_text = raw_bytes.decode("utf-8-sig")
    lines = csv_text.strip().split("\r\n")
    
    assert len(lines) == 3  # Header + 2 rows
    # Header uses col.label
    assert "Vendor,Amount,Date,Notes" in lines[0]
    # Row 1
    assert "Acme Corp,12500.5,2026-03-15,Q1 Cloud Infrastructure" in lines[1]
    # Row 2: Formula injection escaped with leading single quote!
    assert "'=CMD|' /C calc'!A0,-500.0,2026-03-20,\"Credit refund, with comma and \"\"quotes\"\"\"" in lines[2]


def test_pdf_renderer(sample_dataset: TabularDataset):
    renderer = PDFRenderer()
    raw_bytes = renderer.render(sample_dataset, title="Finance Pending Payments")
    
    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) > 500
    # PDF magic bytes
    assert raw_bytes.startswith(b"%PDF-")


def test_docx_renderer(sample_dataset: TabularDataset):
    renderer = DOCXRenderer()
    raw_bytes = renderer.render(sample_dataset, title="Finance Pending Payments")
    
    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) > 1000
    # ZIP / DOCX magic bytes: PK\x03\x04
    assert raw_bytes.startswith(b"PK\x03\x04")


def test_renderers_empty_dataset():
    empty_dataset = TabularDataset(
        title="Empty Dataset",
        columns=[],
        rows=[],
    )
    
    json_bytes = JSONRenderer().render(empty_dataset, title="Empty")
    assert b"\"records\": []" in json_bytes
    
    csv_bytes = CSVRenderer().render(empty_dataset, title="Empty")
    assert len(csv_bytes) >= 3  # UTF-8 BOM
    
    pdf_bytes = PDFRenderer().render(empty_dataset, title="Empty")
    assert pdf_bytes.startswith(b"%PDF-")
    
    docx_bytes = DOCXRenderer().render(empty_dataset, title="Empty")
    assert docx_bytes.startswith(b"PK\x03\x04")


def test_csv_renderer_preserves_negative_and_decimal_numbers():
    import csv
    import io

    renderer = CSVRenderer()
    columns = [
        TableColumn(key="item", label="Item", data_type=ColumnDataType.STRING),
        TableColumn(key="amount", label="Amount", data_type=ColumnDataType.CURRENCY),
        TableColumn(key="discount", label="Discount", data_type=ColumnDataType.NUMBER),
    ]
    rows = [
        {"item": "Refund", "amount": Decimal("-12500.50"), "discount": -250.75},
        {"item": "Adjustment", "amount": "-500.00", "discount": "  -10.00"},
        {"item": "Credit", "amount": "+1200", "discount": "+0.5"},
    ]
    dataset = TabularDataset(title="Finance Adjustments", columns=columns, rows=rows)
    raw_bytes = renderer.render(dataset, title=dataset.title)
    csv_text = raw_bytes.decode("utf-8-sig")

    reader = list(csv.reader(io.StringIO(csv_text)))
    assert reader[0] == ["Item", "Amount", "Discount"]
    # Row 1: Decimal and float negative numbers must NOT have leading single quotes
    assert reader[1] == ["Refund", "-12500.50", "-250.75"]
    # Row 2: String negative numbers must NOT have leading single quotes
    assert reader[2][0] == "Adjustment"
    assert reader[2][1] == "-500.00"
    assert reader[2][2] == "  -10.00"
    # Row 3: Positive numbers with '+' sign
    assert reader[3] == ["Credit", "+1200", "+0.5"]


def test_csv_renderer_escapes_dangerous_prefixes_and_whitespace():
    import csv
    import io

    renderer = CSVRenderer()
    columns = [
        TableColumn(key="payload", label="Payload", data_type=ColumnDataType.STRING),
    ]
    rows = [
        {"payload": "=1+1"},
        {"payload": "  =SUM(A1:B2)"},
        {"payload": "-cmd|' /C calc'!A0"},
        {"payload": "+cmd|' /C calc'!A0"},
        {"payload": "@SUM(A1)"},
        {"payload": "\t@HYPERLINK('http://evil.com')"},
        {"payload": "\r=cmd"},
    ]
    dataset = TabularDataset(title="Dangerous Formulas", columns=columns, rows=rows)
    raw_bytes = renderer.render(dataset, title=dataset.title)
    csv_text = raw_bytes.decode("utf-8-sig")

    reader = list(csv.reader(io.StringIO(csv_text)))
    assert reader[0] == ["Payload"]
    # All dangerous payloads must be escaped with a leading single quote (')
    assert reader[1][0] == "'=1+1"
    assert reader[2][0] == "'  =SUM(A1:B2)"
    assert reader[3][0] == "'-cmd|' /C calc'!A0"
    assert reader[4][0] == "'+cmd|' /C calc'!A0"
    assert reader[5][0] == "'@SUM(A1)"
    assert reader[6][0] == "'\t@HYPERLINK('http://evil.com')"
    assert reader[7][0] == "'\r=cmd"
