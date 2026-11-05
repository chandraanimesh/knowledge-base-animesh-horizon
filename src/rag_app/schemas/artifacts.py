from datetime import datetime
from enum import StrEnum
from typing import Any
import uuid
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Output Modality & Preference Contracts
# ---------------------------------------------------------------------------

class DownloadableArtifactFormat(StrEnum):
    """Supported V1 downloadable binary/structured file formats."""
    PDF = "pdf"    # Branded enterprise document (PyMuPDF)
    CSV = "csv"    # RFC 4180 compliant comma-separated values
    DOCX = "docx"  # Formatted Word document (python-docx)
    JSON = "json"  # Standalone raw JSON file
    MD = "md"      # GitHub Flavored Markdown file


class OutputPreferences(BaseModel):
    """
    Decoupled output preferences. Allows simultaneous chat presentations
    and multiple downloadable artifact formats.
    """
    include_chat_summary: bool = True       # Conversational narrative / executive summary in chat
    include_inline_json: bool = False       # Fenced ```json code block in chat message
    include_inline_table: bool = False      # Formatted Markdown table in chat message
    include_chart: bool = False             # Interactive visual chart widget
    download_formats: list[DownloadableArtifactFormat] = Field(default_factory=list)

    @property
    def chat_summary(self) -> bool:
        return self.include_chat_summary

    @chat_summary.setter
    def chat_summary(self, val: bool) -> None:
        self.include_chat_summary = val



# ---------------------------------------------------------------------------
# 2. Coverage & Completeness Contracts
# ---------------------------------------------------------------------------

class RetrievalCoverage(StrEnum):
    """Factual completeness boundary of the underlying retrieved evidence."""
    SAMPLED_EVIDENCE = "sampled_evidence"    # Top-K semantic retrieval; partial evidence sample
    EXHAUSTIVE_CENSUS = "exhaustive_census"  # Verified complete via authoritative structured DB or un-pruned scan


class ExportTruncation(BaseModel):
    """Details regarding export size limitations and row capping."""
    is_truncated: bool = False
    total_extracted_rows: int = 0
    exported_rows_count: int = 0
    truncation_reason: str | None = None


class CoverageReport(BaseModel):
    """Attached to responses ONLY when structured extraction or tabulation occurs."""
    retrieval_coverage: RetrievalCoverage = RetrievalCoverage.SAMPLED_EVIDENCE
    export_truncation: ExportTruncation = Field(default_factory=ExportTruncation)
    disclaimer: str = (
        "Sample metrics represent only the documents retrieved in active context. "
        "This does NOT represent an exhaustive census of all department records."
    )


# ---------------------------------------------------------------------------
# 3. Dynamic Tabular Schema & Provenance Contracts
# ---------------------------------------------------------------------------

class ColumnDataType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    CURRENCY = "currency"
    DATE = "date"
    BOOLEAN = "boolean"


class TableColumn(BaseModel):
    key: str
    label: str
    data_type: ColumnDataType = ColumnDataType.STRING
    unit: str | None = None  # e.g. "INR", "USD", "days", "%"


class VerificationStatus(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED_FORGED_CHUNK = "rejected_forged_chunk"
    REJECTED_UNGROUNDED_CELL = "rejected_ungrounded_cell"
    REJECTED_TYPE_ERROR = "rejected_type_error"


class RowProvenance(BaseModel):
    """Internal provenance envelope. Kept separate from user-facing columns."""
    source_chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    is_grounded: bool = False
    rejection_reason: str | None = None


class TabularDataset(BaseModel):
    """Canonical validated structured dataset."""
    title: str
    description: str | None = None
    columns: list[TableColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)  # Clean business fields
    row_provenance: dict[str, RowProvenance] = Field(default_factory=dict)  # Keyed by row_id e.g. "row_0"
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    rejected_rows_count: int = 0
    coverage: CoverageReport | None = None


# ---------------------------------------------------------------------------
# 4. Client-Facing Artifact Metadata
# ---------------------------------------------------------------------------

class ArtifactMetadata(BaseModel):
    """Client-facing metadata for generated downloadable files."""
    id: uuid.UUID
    format: DownloadableArtifactFormat
    filename: str
    mime_type: str
    size_bytes: int
    download_url: str
    created_at: datetime
    expires_at: datetime | None = None
    is_expired: bool = False
    error: str | None = None


class ArtifactPreviewResponse(BaseModel):
    """Authenticated preview payload for execution and report artifacts."""
    artifact_id: uuid.UUID
    title: str
    format: str
    mime_type: str
    content: str
    byte_size: int
    checksum_sha256: str
    is_truncated: bool = False
    is_binary: bool = False
    expires_at: datetime | None = None



class ArtifactPreviewResponse(BaseModel):
    """Authenticated preview payload for execution and report artifacts."""
    artifact_id: uuid.UUID
    title: str
    format: str
    mime_type: str
    content: str
    byte_size: int
    checksum_sha256: str
    is_truncated: bool = False
    is_binary: bool = False
    expires_at: datetime | None = None

