import pytest
from rag_app.reports.md_renderer import MDRenderer
from rag_app.reports.registry import renderer_registry
from rag_app.schemas.artifacts import DownloadableArtifactFormat, TabularDataset, TableColumn, ColumnDataType


def test_md_renderer_registration():
    renderer = renderer_registry.get_renderer(DownloadableArtifactFormat.MD)
    assert isinstance(renderer, MDRenderer)
    assert renderer.file_extension == "md"
    assert "text/markdown" in renderer.mime_type


def test_md_renderer_renders_frontmatter_table_and_sources():
    renderer = MDRenderer()

    dataset = TabularDataset(
        title="Quarterly Financial Overview",
        columns=[
            TableColumn(key="quarter", label="Quarter", data_type=ColumnDataType.STRING),
            TableColumn(key="revenue", label="Revenue ($M)", data_type=ColumnDataType.NUMBER),
            TableColumn(key="notes", label="Notes & Exceptions", data_type=ColumnDataType.STRING),
        ],
        rows=[
            {"quarter": "Q1 2026", "revenue": "12.5", "notes": "On track | verified"},
            {"quarter": "Q2 2026", "revenue": "14.2", "notes": "New contract"},
        ],
    )

    metadata = {
        "department": "Finance",
        "verification_status": "VERIFIED_ACCURATE",
        "sources": [
            {"title": "Q2 Earnings Report", "url": "https://intranet.local/q2"},
            "Internal Ledger CSV",
        ],
    }

    raw_bytes = renderer.render(
        dataset=dataset,
        title="Quarterly Financial Overview",
        summary="Executive summary highlighting revenue growth across quarters.",
        metadata=metadata,
    )

    md_text = raw_bytes.decode("utf-8")

    # 1. Frontmatter
    assert "---" in md_text
    assert 'title: "Quarterly Financial Overview"' in md_text
    assert 'department: "Finance"' in md_text
    assert 'verification_status: "VERIFIED_ACCURATE"' in md_text

    # 2. Main Title and Summary
    assert "# Quarterly Financial Overview" in md_text
    assert "## Executive Summary" in md_text
    assert "revenue growth across quarters" in md_text

    # 3. GFM Table with safe pipe escaping
    assert "| Quarter | Revenue ($M) | Notes & Exceptions |" in md_text
    assert "| :--- | ---: | :--- |" in md_text
    # Cell with pipe "On track | verified" must be escaped to "On track \| verified"
    assert r"On track \| verified" in md_text
    assert "12.5" in md_text

    # 4. Sources & Footnotes
    assert "## Sources & Provenance" in md_text
    assert "[Q2 Earnings Report](https://intranet.local/q2)" in md_text
    assert "Internal Ledger CSV" in md_text
