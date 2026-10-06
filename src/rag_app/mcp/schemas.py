"""
MCP Request and Response Schemas for Production RAG.
Exposes data models strictly for the 5 MCP capabilities without leaking internal secrets.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class IngestDocumentInput(BaseModel):
    filename: str = Field(..., description="Name of the file to ingest including extension (e.g. report.pdf, notes.txt)")
    content_base64: str | None = Field(default=None, description="Base64 encoded binary content of the file")
    content_text: str | None = Field(default=None, description="Plain text content (alternative to content_base64 for text/md files)")
    department_code: str | None = Field(default=None, description="Target department code (defaults to active department if not specified)")
    workspace_id: str | None = Field(default=None, description="Optional UUID string of workspace to associate the document with")
    auth_token: str | None = Field(default=None, description="Optional JWT bearer token for caller authentication")


class IngestDocumentResult(BaseModel):
    success: bool = Field(description="Whether the document was successfully ingested")
    file_name: str = Field(description="Name of the ingested file")
    document_id: str | None = Field(default=None, description="UUID string of the created document")
    chunks_created: int = Field(default=0, description="Number of text chunks created and indexed")
    skipped: bool = Field(default=False, description="True if document was skipped (e.g. identical content hash already exists)")
    reason: str | None = Field(default=None, description="Explanation if skipped or failed")


class RagAskInput(BaseModel):
    query: str = Field(..., min_length=1, description="The user question or query to answer using RAG")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top chunks to retrieve for generation")
    rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking to retrieved chunks")
    workspace_id: str | None = Field(default=None, description="Optional workspace UUID to scope retrieval and context")
    auth_token: str | None = Field(default=None, description="Optional JWT bearer token for caller authentication")


class RagAskSource(BaseModel):
    type: str = Field(default="document", description="Source type (document or thread)")
    document_id: str | None = Field(default=None, description="UUID of source document")
    file_name: str | None = Field(default=None, description="File name of the source document")
    chunk_index: int | None = Field(default=None, description="Index of chunk in source document")
    page_number: int | None = Field(default=None, description="Page number if applicable")
    excerpt: str | None = Field(default=None, description="Snippet / excerpt text of the evidence")


class RagAskResult(BaseModel):
    query: str = Field(description="The original user query")
    answer: str = Field(description="The grounded answer generated from authorized evidence")
    citations: list[str] = Field(default_factory=list, description="Citation references found in answer")
    sources: list[RagAskSource] = Field(default_factory=list, description="Structured sources and excerpts used")
    conversation_id: str | None = Field(default=None, description="Conversation ID if part of a thread")


class RagSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="The search query to retrieve evidence for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top chunks to retrieve")
    rerank: bool = Field(default=True, description="Whether to apply cross-encoder reranking")
    workspace_id: str | None = Field(default=None, description="Optional workspace UUID to scope search")
    auth_token: str | None = Field(default=None, description="Optional JWT bearer token for caller authentication")


class RagSearchResultItem(BaseModel):
    chunk_id: str = Field(description="Unique identifier of the chunk")
    document_id: str = Field(description="UUID of the parent document")
    document_name: str = Field(description="File name of the parent document")
    text: str = Field(description="Chunk text content")
    score: float | None = Field(default=None, description="Relevance score (reranker_score or hybrid_score)")
    page_number: int | None = Field(default=None, description="Page number if available")
    chunk_index: int = Field(default=0, description="Position index of chunk in document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional non-sensitive chunk metadata")


class RagSearchResult(BaseModel):
    query: str = Field(description="The original search query")
    total_results: int = Field(description="Number of chunks retrieved")
    results: list[RagSearchResultItem] = Field(default_factory=list, description="List of retrieved evidence chunks")


class ListDocumentsInput(BaseModel):
    department_code: str | None = Field(default=None, description="Optional department code to filter by")
    workspace_id: str | None = Field(default=None, description="Optional workspace UUID to filter by")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum number of documents to return")
    auth_token: str | None = Field(default=None, description="Optional JWT bearer token for caller authentication")


class DocumentSummaryItem(BaseModel):
    id: str = Field(description="UUID of the document")
    name: str = Field(description="File name of the document")
    status: str = Field(description="Status of the document (e.g. READY, PROCESSING)")
    chunk_count: int = Field(description="Total chunks created for this document")
    created_at: str | None = Field(default=None, description="ISO timestamp of creation")
    department_code: str | None = Field(default=None, description="Department code of the document")
    mime_type: str | None = Field(default=None, description="MIME type of the document")


class ListDocumentsResult(BaseModel):
    total_count: int = Field(description="Total documents found")
    documents: list[DocumentSummaryItem] = Field(default_factory=list, description="List of authorized documents")


class DiagnosticsResult(BaseModel):
    status: str = Field(description="Overall system status: healthy, degraded, or unhealthy")
    timestamp: str = Field(description="ISO timestamp of diagnostic run")
    runtime: dict[str, Any] = Field(description="Runtime environment details")
    imports: dict[str, bool] = Field(description="Validation of required Python packages")
    database: dict[str, Any] = Field(description="Database connectivity, pgvector extension, and table presence")
    models: dict[str, Any] = Field(description="Model providers status (embedding, reranker, LLM)")
    rag: dict[str, Any] = Field(description="RAG service health and index status")
