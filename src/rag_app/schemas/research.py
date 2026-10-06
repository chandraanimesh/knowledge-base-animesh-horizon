from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field, model_validator


class SufficiencyStatus(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class SufficiencyEvaluation(BaseModel):
    """
    Structured evidence sufficiency evaluation result.
    Evaluates whether retrieved evidence is sufficient, partial, or insufficient
    to answer the user's information need without hallucination or estimation.
    """
    status: SufficiencyStatus = Field(
        ...,
        description="Strict sufficiency status: 'sufficient' (all essential facts supported), 'partial' (some supported, some missing, useful grounded answer possible), or 'insufficient' (missing essential facts, cannot produce useful grounded answer)."
    )
    evidence_found: bool = Field(
        default=True,
        description="Whether any relevant evidence addressing the query was found in retrieved context."
    )
    required_information: list[str] = Field(
        default_factory=list,
        description="Key facts, dimensions, entities, measures, or tasks required by the query."
    )
    supported_information: list[str] = Field(
        default_factory=list,
        description="Required information items that are explicitly supported by the retrieved context."
    )
    missing_requirements: list[str] = Field(
        default_factory=list,
        description="Required information items that are absent from the retrieved context."
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Contradictory statements or conflicting values found across retrieved sources."
    )
    answerable_from_context: bool = Field(
        default=True,
        description="Whether a useful, fully or partially grounded answer can be produced strictly from context."
    )
    reason: str = Field(
        ...,
        description="Concise factual explanation for the sufficiency status decision."
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence score in the sufficiency evaluation (0.0 to 1.0)."
    )
    coverage: float | None = Field(
        default=None,
        description="Estimated coverage score of requested information (0.0 to 1.0)."
    )
    evidence_types: list[str] = Field(
        default_factory=list,
        description="Detected evidence types (e.g. 'prose', 'table', 'list', 'hierarchical', 'conversation')."
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "status" not in data and "is_sufficient" in data:
                data["status"] = SufficiencyStatus.SUFFICIENT if data["is_sufficient"] else SufficiencyStatus.INSUFFICIENT
            if "reason" not in data or data["reason"] is None:
                data["reason"] = "No reason provided."
            # If answerable_from_context not explicitly set
            if "answerable_from_context" not in data:
                if data.get("status") == SufficiencyStatus.INSUFFICIENT and not data.get("supported_information"):
                    data["answerable_from_context"] = False
                else:
                    data["answerable_from_context"] = True
        return data

    @property
    def is_sufficient(self) -> bool:
        """Backward-compatible boolean property."""
        return self.status == SufficiencyStatus.SUFFICIENT


# Canonical alias for compatibility with new and existing code
SufficiencyResult = SufficiencyEvaluation

class SourceSelection(BaseModel):
    """Selection of relevant URLs to crawl from a list of search results."""
    selected_urls: list[str] = Field(description="List of URLs that are most likely to contain the answer to the user's query.")
    reason: str | None = Field(default=None, description="Explanation for the selection.")

class CleanWebPage(BaseModel):
    """Clean, normalized content of a web page."""
    url: str = Field(..., description="The source URL of the page")
    title: str | None = Field(None, description="The extracted title of the page")
    content: str = Field(..., description="The normalized and cleaned markdown/text content")
    metadata: dict | None = Field(None, description="Additional provenance metadata")

class ExtractedFact(BaseModel):
    """A single structured fact extracted from a web page."""
    entity_name: str = Field(..., description="Name of the entity (e.g., Company Name, Person)")
    entity_type: str = Field(..., description="Type of the entity (e.g., ORGANIZATION, PERSON, LOCATION)")
    attributes: dict = Field(..., description="Structured attributes extracted for the entity (e.g., {'email': 'x@y.com'})")
    department_code: str = Field(..., description="Department category (e.g., SALES, FINANCE, HR, UNCLASSIFIED, ALL). Must be exactly one of these.")
    source_url: str = Field(..., description="The URL of the clean web page where this fact was found.")
    evidence_text: str = Field(..., description="The exact verbatim text span from the clean web page that proves this fact.")

class StructuredKnowledgeCandidate(BaseModel):
    """A collection of extracted facts from a web page, ready for validation."""
    facts: list[ExtractedFact] = Field(default_factory=list, description="List of structured facts extracted.")
