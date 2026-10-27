import pytest
from unittest.mock import AsyncMock
from rag_app.generation.tabular_extractor import (
    TabularExtractionService,
    normalize_date_cell,
    normalize_numeric_cell,
    verify_cell_grounding,
)
from rag_app.schemas.artifacts import (
    ColumnDataType,
    RetrievalCoverage,
    TableColumn,
    VerificationStatus,
)
from rag_app.schemas.rag import RetrievedChunk


def test_date_and_numeric_normalization():
    """Verify ISO-8601 date coercion and negative/decimal currency handling."""
    # Date normalizations
    assert normalize_date_cell("2026-11-20") == "2026-11-20"
    assert normalize_date_cell("20/11/2026") == "2026-11-20"
    assert normalize_date_cell("20 Nov 2026") == "2026-11-20"
    assert normalize_date_cell("November 20, 2026") == "2026-11-20"
    assert normalize_date_cell("invalid-date-string") is None

    # Numeric normalizations (including negative amounts e.g. refunds / discounts)
    assert normalize_numeric_cell(150000.50) == 150000.50
    assert normalize_numeric_cell("$250,000.00") == 250000.00
    assert normalize_numeric_cell("₹ 1,50,000.00") == 150000.00
    assert normalize_numeric_cell("-250.75") == -250.75
    assert normalize_numeric_cell("(500.00)") == -500.00  # Accounting negative
    assert normalize_numeric_cell("not-a-number") is None


def test_cell_grounding_verification():
    """Verify cell value grounding against chunk text."""
    chunk_text = "Vendor Acme Corp submitted Invoice INV-902 for amount $250,000 due on Nov 20, 2026."
    assert verify_cell_grounding("INV-902", chunk_text) is True
    assert verify_cell_grounding("Acme Corp", chunk_text) is True
    assert verify_cell_grounding(250000.00, chunk_text) is True
    assert verify_cell_grounding("2026-11-20", chunk_text) is True

    # Hallucinated / ungrounded values
    assert verify_cell_grounding("INV-9999", chunk_text) is False
    assert verify_cell_grounding("Stark Industries", chunk_text) is False
    assert verify_cell_grounding(999999.00, chunk_text) is False


@pytest.mark.asyncio
async def test_tabular_extractor_forged_chunk_and_ungrounded_rejection():
    """Verify that forged chunk IDs and ungrounded values are quarantined."""
    mock_llm = AsyncMock()
    # LLM returns 3 rows:
    # 1. Valid grounded row
    # 2. Row with forged chunk ID
    # 3. Row with hallucinated amount
    mock_llm.generate.return_value = (
        """{
            "title": "Pending Invoices",
            "columns": [
                {"key": "invoice_id", "label": "Invoice ID", "data_type": "string"},
                {"key": "amount", "label": "Amount", "data_type": "currency", "unit": "USD"}
            ],
            "rows": [
                {"_source_chunk_id": "chunk_auth_1", "invoice_id": "INV-101", "amount": 5000.00},
                {"_source_chunk_id": "forged_chunk_999", "invoice_id": "INV-102", "amount": 6000.00},
                {"_source_chunk_id": "chunk_auth_1", "invoice_id": "INV-103", "amount": 999999.00}
            ]
        }""",
        {},
    )

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_auth_1",
            document_id="doc_1",
            department_id="dept_1",
            file_name="Invoices.pdf",
            chunk_index=0,
            text="Invoice INV-101 is pending with amount $5,000.00.",
        )
    ]

    service = TabularExtractionService(llm=mock_llm)
    dataset, coverage = await service.extract_tabular_dataset(
        query="List pending invoices",
        chunks=chunks,
        max_rows=10,
    )

    assert dataset is not None
    assert coverage is not None
    # Only 1 row was truly valid
    assert len(dataset.rows) == 1
    assert dataset.rows[0]["invoice_id"] == "INV-101"
    assert dataset.rows[0]["amount"] == 5000.00

    # 2 rows were rejected
    assert dataset.rejected_rows_count == 2
    assert dataset.row_provenance["row_1"].verification_status == VerificationStatus.REJECTED_FORGED_CHUNK
    assert dataset.row_provenance["row_2"].verification_status == VerificationStatus.REJECTED_UNGROUNDED_CELL

    # Coverage report must be SAMPLED_EVIDENCE with truthful disclaimers
    assert coverage.retrieval_coverage == RetrievalCoverage.SAMPLED_EVIDENCE
    assert coverage.export_truncation.is_truncated is False
    assert dataset.summary_metrics["sum_amount"] == 5000.00


@pytest.mark.asyncio
async def test_tabular_extractor_truncation():
    """Verify that exceeding max_rows triggers export truncation."""
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        """{
            "title": "Items",
            "columns": [{"key": "id", "label": "ID", "data_type": "string"}],
            "rows": [
                {"_source_chunk_id": "c1", "id": "A1"},
                {"_source_chunk_id": "c1", "id": "A2"},
                {"_source_chunk_id": "c1", "id": "A3"}
            ]
        }""",
        {},
    )

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="doc_1",
            department_id="dept_1",
            file_name="Items.pdf",
            chunk_index=0,
            text="Found items A1, A2, and A3.",
        )
    ]

    service = TabularExtractionService(llm=mock_llm)
    dataset, coverage = await service.extract_tabular_dataset(
        query="List items",
        chunks=chunks,
        max_rows=2,  # Limit to 2
    )

    assert dataset is not None
    assert len(dataset.rows) == 2
    assert coverage.export_truncation.is_truncated is True
    assert coverage.export_truncation.total_extracted_rows == 3
    assert coverage.export_truncation.exported_rows_count == 2


@pytest.mark.asyncio
async def test_sales_discount_approval_rules_canonical_extraction():
    """
    Regression test: Proves that all 3 source-supported discount ranges appear consistently
    in tabular extraction without dropping the '>5% up to 10%' Sales Manager approval row.
    Also proves that summary metrics 'total_records' is 3, and any hallucinated row is rejected.
    """
    sales_chunk_text = (
        "Discount Approval\n"
        "Discounts up to 5% require the account owner's approval. "
        "Discounts above 5% and up to 10% require Sales Manager approval. "
        "Discounts above 10% require Sales Director approval and a documented business justification. "
        "These are fictional demo rules, not company policy."
    )

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_sales_demo_01",
            document_id="doc_sales_01",
            department_id="dept_sales",
            file_name="Sales_Department_Demo.docx",
            chunk_index=0,
            text=sales_chunk_text,
        )
    ]

    mock_llm = AsyncMock()
    # Mock LLM returning 3 valid rows + 1 hallucinated row
    mock_llm.generate.return_value = (
        """{
            "title": "Sales Discount Approval Rules",
            "description": "Rules governing sales discount approval thresholds",
            "columns": [
                {"key": "discount_range", "label": "Discount Range", "data_type": "string"},
                {"key": "approver", "label": "Approver Required", "data_type": "string"},
                {"key": "notes", "label": "Conditions / Documentation", "data_type": "string"}
            ],
            "rows": [
                {
                    "_source_chunk_id": "chunk_sales_demo_01",
                    "discount_range": "Up to 5%",
                    "approver": "account owner's approval",
                    "notes": "account owner's approval"
                },
                {
                    "_source_chunk_id": "chunk_sales_demo_01",
                    "discount_range": ">5% up to 10%",
                    "approver": "Sales Manager",
                    "notes": "Sales Manager approval"
                },
                {
                    "_source_chunk_id": "chunk_sales_demo_01",
                    "discount_range": "Above 10%",
                    "approver": "Sales Director",
                    "notes": "documented business justification"
                },
                {
                    "_source_chunk_id": "chunk_sales_demo_01",
                    "discount_range": ">25%",
                    "approver": "CFO approval",
                    "notes": "Board resolution"
                }
            ]
        }""",
        {},
    )

    service = TabularExtractionService(llm=mock_llm)
    dataset, coverage = await service.extract_tabular_dataset(
        query="What are the sales discount approval rules?",
        chunks=chunks,
    )

    assert dataset is not None
    assert coverage is not None

    # Exactly 3 source-supported rows are verified; the 4th hallucinated row is rejected
    assert len(dataset.rows) == 3
    assert dataset.rejected_rows_count == 1
    assert dataset.summary_metrics["total_records"] == 3

    # Verify all 3 discount ranges are preserved
    ranges = [r["discount_range"] for r in dataset.rows]
    approvers = [r["approver"] for r in dataset.rows]

    assert "Up to 5%" in ranges
    assert ">5% up to 10%" in ranges
    assert "Above 10%" in ranges

    assert "account owner's approval" in approvers
    assert "Sales Manager" in approvers
    assert "Sales Director" in approvers

    # Provenance records check
    assert dataset.row_provenance["row_0"].verification_status == VerificationStatus.VERIFIED
    assert dataset.row_provenance["row_1"].verification_status == VerificationStatus.VERIFIED
    assert dataset.row_provenance["row_2"].verification_status == VerificationStatus.VERIFIED
    assert dataset.row_provenance["row_3"].verification_status == VerificationStatus.REJECTED_UNGROUNDED_CELL


def test_fabricated_line_range_citation_sanitization():
    """
    Regression test: Proves that fabricated line ranges like 'L1-L3' are sanitized
    from answer citations when not substantiated by retrieved evidence chunks,
    preventing spurious grounding entity violations.
    """
    from rag_app.generation.citations import validate_and_render_citations

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_sales_1",
            document_id="doc_1",
            department_id="dept_1",
            file_name="Sales_Department_Demo.docx",
            chunk_index=0,
            text="Discounts up to 5% require account owner approval. These are fictional demo rules, not company policy.",
        )
    ]

    # Model emits fabricated line range L1-L3 in text and citation
    raw_answer = (
        "According to the policy [Sales_Department_Demo.docx, L1-L3], "
        "discounts up to 5% require account owner approval. "
        "These are fictional demo rules, not company policy."
    )

    sanitized, citations = validate_and_render_citations(raw_answer=raw_answer, chunks=chunks)

    # L1-L3 must be cleanly stripped
    assert "L1-L3" not in sanitized
    assert "Sales_Department_Demo.docx" in citations[0]
    assert "fictional demo rules, not company policy" in sanitized


def test_grounding_uuid_masking_no_spurious_numeric_claims():
    """
    Proves that UUID chunk references such as 5182cdad-7582-4786-9074-d4fdf67ed45c
    do NOT extract hex numeric sub-segments (7582, 4786) as business metrics.
    """
    from rag_app.generation.grounding import _extract_numbers

    text = (
        "Source: Finance Department - Demo Knowledge Document, Chunk ID "
        "5182cdad-7582-4786-9074-d4fdf67ed45c (the only chunk containing the Q3 budget details).\n"
        "Operating budget is ₹12,00,000 and actual spend is ₹10,80,000."
    )

    extracted = _extract_numbers(text)
    # Business metrics must be extracted
    assert "₹12,00,000" in extracted or "12,00,000" in extracted
    assert "₹10,80,000" in extracted or "10,80,000" in extracted

    # Hex subsegments of chunk UUID must NOT be extracted as standalone numbers
    assert "7582" not in extracted
    assert "4786" not in extracted
    assert "9074" not in extracted


def test_indian_and_international_number_grounding():
    """
    Proves that Indian lakh/crore comma grouping (₹12,00,000, ₹3,60,000)
    and European/international formats are grounded without false rejection.
    """
    from rag_app.generation.grounding import _is_number_in_evidence

    evidence = "The budget for Q3 is ₹12,00,000. Software spend is ₹3,60,000 and Training is ₹60,000."

    assert _is_number_in_evidence("₹12,00,000", evidence) is True
    assert _is_number_in_evidence("12,00,000", evidence) is True
    assert _is_number_in_evidence(1200000, evidence) is True
    assert _is_number_in_evidence("360000", evidence) is True
    assert _is_number_in_evidence(360000.0, evidence) is True
    assert _is_number_in_evidence(60000, evidence) is True

    # Cell grounding check
    assert verify_cell_grounding(360000.0, evidence) is True
    assert verify_cell_grounding(1200000.0, evidence) is True
    assert verify_cell_grounding("₹3,60,000", evidence) is True


@pytest.mark.asyncio
async def test_nested_dict_and_list_row_expansion():
    """
    Proves that nested dictionaries or repeated object arrays inside extracted rows
    are generically flattened into individual canonical rows.
    """
    mock_llm = AsyncMock()
    # LLM returns a single object containing a nested dict of spending categories
    mock_llm.generate.return_value = (
        """{
            "title": "Q3 Department Budget",
            "columns": [
                {"key": "operating_budget", "label": "Operating Budget", "data_type": "currency"},
                {"key": "spending_categories", "label": "Spending Categories", "data_type": "string"}
            ],
            "rows": [
                {
                    "_source_chunk_id": "chunk_budget_01",
                    "operating_budget": 1200000.0,
                    "spending_categories": {
                        "software": 360000.0,
                        "contractors": 300000.0,
                        "travel": 120000.0,
                        "training": 60000.0,
                        "other_operating_costs": 240000.0
                    }
                }
            ]
        }""",
        {},
    )

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_budget_01",
            document_id="doc_budget",
            department_id="dept_fin",
            file_name="Budget_Q3.pdf",
            chunk_index=0,
            text=(
                "Q3 Operating budget is ₹12,00,000. "
                "Spending categories: software ₹3,60,000, contractors ₹3,00,000, "
                "travel ₹1,20,000, training ₹60,000, other operating costs ₹2,40,000."
            ),
        )
    ]

    service = TabularExtractionService(llm=mock_llm)
    dataset, coverage = await service.extract_tabular_dataset(
        query="List budget and spending categories",
        chunks=chunks,
    )

    assert dataset is not None
    # All 5 categories must be expanded into distinct canonical rows
    assert len(dataset.rows) == 5
    assert dataset.summary_metrics["total_records"] == 5

    items = [r.get("spending_categories_item") for r in dataset.rows]
    assert "software" in items
    assert "contractors" in items
    assert "travel" in items
    assert "training" in items
    assert "other_operating_costs" in items


@pytest.mark.asyncio
async def test_arbitrary_synthetic_domain_schema_driven_extraction():
    """
    Proves that structured extraction works for completely arbitrary domains
    (e.g. IoT Sensor Metrics) with negative numbers, ranges, and nullability,
    without any domain hardcoding.
    """
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        """{
            "title": "Cryogenic Sensor Telemetry",
            "columns": [
                {"key": "sensor_id", "label": "Sensor ID", "data_type": "string"},
                {"key": "temp_celsius", "label": "Temperature (°C)", "data_type": "number"},
                {"key": "operational_range", "label": "Operational Range", "data_type": "string"},
                {"key": "calibrated_date", "label": "Calibration Date", "data_type": "date"},
                {"key": "is_alert", "label": "Alert State", "data_type": "boolean"}
            ],
            "rows": [
                {
                    "_source_chunk_id": "sensor_chunk_88",
                    "sensor_id": "SNS-CRYO-01",
                    "temp_celsius": -185.5,
                    "operational_range": "-190°C up to -150°C",
                    "calibrated_date": "2026-08-15",
                    "is_alert": false
                },
                {
                    "_source_chunk_id": "sensor_chunk_88",
                    "sensor_id": "SNS-CRYO-02",
                    "temp_celsius": -40.0,
                    "operational_range": "above -50°C",
                    "calibrated_date": "2026-09-01",
                    "is_alert": true
                }
            ]
        }""",
        {},
    )

    chunks = [
        RetrievedChunk(
            chunk_id="sensor_chunk_88",
            document_id="doc_iot",
            department_id="dept_eng",
            file_name="Cryo_Telemetry.txt",
            chunk_index=0,
            text=(
                "Sensor SNS-CRYO-01 reading: -185.5 °C, range -190°C up to -150°C, calibrated 2026-08-15. "
                "Sensor SNS-CRYO-02 reading: -40.0 °C, range above -50°C, calibrated 2026-09-01, trigger alert."
            ),
        )
    ]

    service = TabularExtractionService(llm=mock_llm)
    dataset, coverage = await service.extract_tabular_dataset(
        query="Report cryogenic sensor readings",
        chunks=chunks,
    )

    assert dataset is not None
    assert len(dataset.rows) == 2
    assert dataset.summary_metrics["total_records"] == 2

    # Negative number preservation
    assert dataset.rows[0]["temp_celsius"] == -185.5
    assert dataset.rows[1]["temp_celsius"] == -40.0

    # Range and condition expression preservation
    assert dataset.rows[0]["operational_range"] == "-190°C up to -150°C"
    assert dataset.rows[1]["operational_range"] == "above -50°C"

    # Boolean and Date types
    assert dataset.rows[0]["is_alert"] is False
    assert dataset.rows[1]["is_alert"] is True
    assert dataset.rows[0]["calibrated_date"] == "2026-08-15"


