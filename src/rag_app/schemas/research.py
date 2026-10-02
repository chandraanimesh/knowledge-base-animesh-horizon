from pydantic import BaseModel, Field

class SufficiencyEvaluation(BaseModel):
    """Evaluation of whether the provided context is sufficient to answer the query."""
    is_sufficient: bool = Field(description="True if the context contains relevant and factual information to answer or address the query, False otherwise.")
    reason: str | None = Field(default=None, description="Explanation for why the context is or isn't sufficient.")

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
