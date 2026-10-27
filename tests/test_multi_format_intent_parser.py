import pytest
from rag_app.retrieval.query_parser import (
    parse_output_preferences,
    strip_presentation_instructions,
    is_substantive_format_topic,
)
from rag_app.schemas.artifacts import DownloadableArtifactFormat, OutputPreferences


def test_compound_hinglish_request():
    """Verify that compound Hinglish request accurately detects summary, inline JSON, and PDF download."""
    query = (
        "Finance department ki pending payments nikalo. "
        "Mujhe summary chat mein do, detailed records JSON mein aur ek PDF report bhi generate kar do."
    )
    prefs = parse_output_preferences(query)
    assert prefs.include_chat_summary is True
    assert prefs.include_inline_json is True
    assert DownloadableArtifactFormat.PDF in prefs.download_formats

    # Verify that substantive query is preserved while presentation clauses are stripped
    cleaned = strip_presentation_instructions(query)
    assert "Finance department ki pending payments nikalo" in cleaned
    assert "pdf report" not in cleaned.lower()
    assert "json" not in cleaned.lower()
    assert "summary" not in cleaned.lower()


def test_each_supported_download_format():
    """Verify detection of PDF, CSV, DOCX, and JSON downloadable files."""
    # PDF
    p1 = parse_output_preferences("Export Q3 sales report as PDF")
    assert DownloadableArtifactFormat.PDF in p1.download_formats

    # CSV
    p2 = parse_output_preferences("Download vendor commitments as a CSV file")
    assert DownloadableArtifactFormat.CSV in p2.download_formats

    # DOCX
    p3 = parse_output_preferences("Generate compliance audit findings as a Word docx report")
    assert DownloadableArtifactFormat.DOCX in p3.download_formats

    # JSON File
    p4 = parse_output_preferences("Export employee roster as a JSON file download")
    assert DownloadableArtifactFormat.JSON in p4.download_formats


def test_substantive_format_keywords_not_stripped_or_triggered():
    """
    Format words used as substantive query topics must not trigger downloads
    and must not be stripped from semantic search queries.
    """
    query = "What is our company policy on PDF file retention and JSON schema validation?"

    assert is_substantive_format_topic(query, "pdf") is True
    assert is_substantive_format_topic(query, "json") is True

    prefs = parse_output_preferences(query)
    assert prefs.download_formats == []
    assert prefs.include_inline_json is False

    cleaned = strip_presentation_instructions(query)
    # The substantive query should retain PDF and JSON
    assert "PDF" in cleaned
    assert "JSON" in cleaned
    assert "retention" in cleaned
    assert "schema validation" in cleaned


def test_explicit_preferences_override_inferred():
    """Explicit preferences passed from API/client take precedence over inferred text."""
    query = "Give me summary in chat and export as PDF"
    explicit = OutputPreferences(
        include_chat_summary=False,
        include_inline_table=True,
        download_formats=[DownloadableArtifactFormat.CSV],
    )
    prefs = parse_output_preferences(query, explicit=explicit)
    assert prefs.include_chat_summary is False
    assert prefs.include_inline_table is True
    assert prefs.download_formats == [DownloadableArtifactFormat.CSV]


def test_normal_prose_query_regression():
    """Normal prose queries default cleanly to standard chat without exports."""
    query = "What are the rules for travel reimbursement?"
    prefs = parse_output_preferences(query)
    assert prefs.include_chat_summary is True
    assert prefs.include_inline_json is False
    assert prefs.include_inline_table is False
    assert prefs.include_chart is False
    assert prefs.download_formats == []

    cleaned = strip_presentation_instructions(query)
    assert cleaned == "What are the rules for travel reimbursement"
