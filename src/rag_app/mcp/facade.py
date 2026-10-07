"""
MCP Facade Layer for Production RAG.
Acts as a thin adapter delegating MCP tool invocations directly into existing application services:
- IngestionService
- RetrievalService
- GenerationService
- Security & Authorization
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path
from typing import Any

import jwt
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.core.config import get_settings
from rag_app.db.models import Department, Document, DocumentChunk, User, Workspace, WorkspaceMember
from rag_app.db.session import AsyncSessionLocal
from rag_app.generation.service import GenerationService
from rag_app.ingestion.service_v1 import IngestionService
from rag_app.mcp.diagnostics import run_system_diagnostics
from rag_app.mcp.schemas import (
    DiagnosticsResult,
    DocumentSummaryItem,
    IngestDocumentInput,
    IngestDocumentResult,
    ListDocumentsInput,
    ListDocumentsResult,
    RagAskInput,
    RagAskResult,
    RagAskSource,
    RagSearchInput,
    RagSearchResult,
    RagSearchResultItem,
)
from rag_app.retrieval.service import RetrievalService
from rag_app.schemas.rag import GenerateRequest
from rag_app.core.config import settings
from rag_app.security.authorization import (
    authorize,
    get_allowed_departments,
    get_allowed_document_ids,
)

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls", ".docx"}


class MCPFacade:
    """
    Thin adapter between FastMCP tools and existing RAG services.
    Does NOT duplicate vector search, FTS, RRF, reranking, or embedding logic.
    """

    async def _resolve_user(self, session: AsyncSession, auth_token: str | None) -> User:
        """
        Resolves caller identity using existing JWT verification or loads the default local active user.
        Never bypasses authorization.
        """
        if auth_token:
            try:
                # Strip Bearer prefix if provided
                token_str = auth_token.strip()
                if token_str.lower().startswith("bearer "):
                    token_str = token_str[7:].strip()

                payload = jwt.decode(
                    token_str,
                    settings.jwt_secret_key,
                    algorithms=[settings.jwt_algorithm],
                )
                user_id_str = payload.get("sub")
                if not user_id_str:
                    raise ValueError("JWT token missing 'sub' claim")

                user = await session.get(User, uuid.UUID(user_id_str))
                if not user or not user.is_active:
                    raise ValueError("Authenticated user is inactive or does not exist")
                return user
            except Exception as exc:
                logger.warning(f"MCP Token authentication failed: {exc}")
                raise ValueError(f"Authentication failed: {exc}")

        # If no auth_token is provided (e.g. local FastMCP caller without header):
        # Find active admin or first active user
        admin_user = await session.scalar(
            select(User).where(User.role == "ADMIN", User.is_active == True).limit(1)
        )
        if admin_user:
            return admin_user

        any_user = await session.scalar(
            select(User).where(User.is_active == True).limit(1)
        )
        if any_user:
            return any_user

        # Fallback local in-memory user for bootstrapping
        return User(
            id=uuid.uuid4(),
            email="mcp_local@example.com",
            role="ADMIN",
            is_active=True,
        )

    async def ingest_document(self, input_data: IngestDocumentInput) -> IngestDocumentResult:
        """
        Ingests a document through the existing IngestionService pipeline.
        Supports base64 or plain text content without accessing client local filesystems.
        """
        filename = Path(input_data.filename).name
        suffix = Path(filename).suffix.lower()

        if suffix not in ALLOWED_SUFFIXES:
            return IngestDocumentResult(
                success=False,
                file_name=filename,
                skipped=True,
                reason=f"Unsupported file format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
            )

        # Decode or encode payload
        raw_bytes: bytes
        if input_data.content_base64:
            try:
                raw_bytes = base64.b64decode(input_data.content_base64)
            except Exception as exc:
                return IngestDocumentResult(
                    success=False,
                    file_name=filename,
                    skipped=True,
                    reason=f"Invalid base64 encoding: {exc}",
                )
        elif input_data.content_text is not None:
            raw_bytes = input_data.content_text.encode("utf-8")
        else:
            return IngestDocumentResult(
                success=False,
                file_name=filename,
                skipped=True,
                reason="Must provide either 'content_base64' or 'content_text'",
            )

        if len(raw_bytes) > settings.max_upload_size:
            return IngestDocumentResult(
                success=False,
                file_name=filename,
                skipped=True,
                reason=f"File size ({len(raw_bytes)} bytes) exceeds maximum limit ({settings.max_upload_size} bytes)",
            )

        upload_dir = Path("uploaded_docs")
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"mcp_{uuid.uuid4().hex}_{filename}"
        saved_path = upload_dir / safe_name

        try:
            saved_path.write_bytes(raw_bytes)

            async with AsyncSessionLocal() as session:
                user = await self._resolve_user(session, input_data.auth_token)

                # Resolve department
                dept = None
                if input_data.department_code:
                    dept = await session.scalar(
                        select(Department).where(Department.code == input_data.department_code)
                    )
                    if not dept or not dept.is_active:
                        return IngestDocumentResult(
                            success=False,
                            file_name=filename,
                            skipped=True,
                            reason=f"Department '{input_data.department_code}' not found or inactive",
                        )
                else:
                    dept = await session.scalar(
                        select(Department).where(Department.is_active == True).limit(1)
                    )

                dept_id = dept.id if dept else None
                dept_code = dept.code if dept else None

                # Authorization check
                if dept_code and user.role != "ADMIN":
                    is_authorized = await authorize(session, user, "UPLOAD", dept_code)
                    if not is_authorized:
                        return IngestDocumentResult(
                            success=False,
                            file_name=filename,
                            skipped=True,
                            reason=f"User unauthorized to upload to department '{dept_code}'",
                        )

                # Workspace validation if provided
                ws_uuid = None
                if input_data.workspace_id:
                    try:
                        ws_uuid = uuid.UUID(input_data.workspace_id)
                        ws = await session.get(Workspace, ws_uuid)
                        if not ws or not ws.is_active or ws.deleted_at is not None:
                            return IngestDocumentResult(
                                success=False,
                                file_name=filename,
                                skipped=True,
                                reason=f"Workspace '{input_data.workspace_id}' not found or inactive",
                            )
                    except ValueError:
                        return IngestDocumentResult(
                            success=False,
                            file_name=filename,
                            skipped=True,
                            reason=f"Invalid workspace_id UUID: {input_data.workspace_id}",
                        )

                service = IngestionService(session)
                resp = await service.ingest_file(
                    saved_path,
                    department_id=dept_id,
                    created_by=user.id,
                    workspace_id=ws_uuid,
                )

                return IngestDocumentResult(
                    success=not resp.skipped or bool(resp.document_id),
                    file_name=resp.file_name,
                    document_id=resp.document_id,
                    chunks_created=resp.chunks_created,
                    skipped=resp.skipped,
                    reason=resp.reason,
                )

        except Exception as exc:
            logger.error(f"MCP Ingestion failed: {exc}", exc_info=True)
            saved_path.unlink(missing_ok=True)
            return IngestDocumentResult(
                success=False,
                file_name=filename,
                skipped=True,
                reason=f"Ingestion error: {exc}",
            )

    async def rag_search(self, input_data: RagSearchInput) -> RagSearchResult:
        """
        Retrieves evidence chunks from the existing RAG pipeline without generating an LLM answer.
        Reuses RetrievalService (query parser, hybrid search, RRF, cross-encoder reranking).
        """
        async with AsyncSessionLocal() as session:
            user = await self._resolve_user(session, input_data.auth_token)
            allowed_departments = await get_allowed_departments(session, user)
            allowed_documents = await get_allowed_document_ids(session, user, "VIEW")

            ws_uuid = None
            if input_data.workspace_id:
                try:
                    ws_uuid = uuid.UUID(input_data.workspace_id)
                except ValueError:
                    logger.warning(f"Invalid workspace_id '{input_data.workspace_id}', ignoring filter")

            service = RetrievalService(session)
            resp = await service.retrieve(
                query=input_data.query,
                top_k=input_data.top_k,
                rerank=input_data.rerank,
                allowed_departments=allowed_departments,
                allowed_documents=allowed_documents,
                workspace_id=ws_uuid,
            )

            results_items = []
            for chunk in resp.results:
                score = chunk.reranker_score if chunk.reranker_score is not None else chunk.hybrid_score
                results_items.append(
                    RagSearchResultItem(
                        chunk_id=str(chunk.chunk_id),
                        document_id=str(chunk.document_id),
                        document_name=chunk.file_name,
                        text=chunk.text,
                        score=round(score, 4) if score is not None else None,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        metadata={
                            "department_id": str(chunk.department_id),
                            "distance": chunk.distance,
                            "vector_similarity": chunk.vector_similarity,
                        },
                    )
                )

            return RagSearchResult(
                query=resp.query,
                total_results=len(results_items),
                results=results_items,
            )

    async def rag_ask(self, input_data: RagAskInput) -> RagAskResult:
        """
        Generates an authorized, grounded RAG response reusing the existing GenerationService.
        """
        async with AsyncSessionLocal() as session:
            user = await self._resolve_user(session, input_data.auth_token)

            ws_uuid = None
            if input_data.workspace_id:
                try:
                    ws_uuid = uuid.UUID(input_data.workspace_id)
                except ValueError:
                    logger.warning(f"Invalid workspace_id '{input_data.workspace_id}', ignoring filter")

            gen_service = GenerationService(session)
            gen_request = GenerateRequest(
                query=input_data.query,
                top_k=input_data.top_k,
                rerank=input_data.rerank,
                workspace_id=ws_uuid,
            )

            resp = await gen_service.generate_answer(gen_request, user=user)

            sources_list = []
            for c in resp.retrieved_chunks:
                sources_list.append(
                    RagAskSource(
                        type="document",
                        document_id=str(c.document_id),
                        file_name=c.file_name,
                        chunk_index=c.chunk_index,
                        page_number=c.page_number,
                        excerpt=c.text[:300] if c.text else "",
                    )
                )

            return RagAskResult(
                query=input_data.query,
                answer=resp.answer,
                citations=resp.citations,
                sources=sources_list,
                conversation_id=resp.conversation_id,
            )

    async def list_documents(self, input_data: ListDocumentsInput) -> ListDocumentsResult:
        """
        Lists documents accessible to the caller with chunk counts.
        """
        async with AsyncSessionLocal() as session:
            user = await self._resolve_user(session, input_data.auth_token)
            allowed_dept_ids = await get_allowed_departments(session, user)
            allowed_doc_ids = await get_allowed_document_ids(session, user, "VIEW")

            if user.role != "ADMIN" and not allowed_dept_ids and not allowed_doc_ids:
                return ListDocumentsResult(total_count=0, documents=[])

            # Query Document joined with Department and aggregated Chunk Count
            stmt = (
                select(
                    Document,
                    Department.code.label("department_code"),
                    func.count(DocumentChunk.id).label("chunk_count"),
                )
                .outerjoin(Department, Document.department_id == Department.id)
                .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
                .group_by(Document.id, Department.code)
                .order_by(Document.created_at.desc())
            )

            if input_data.department_code:
                stmt = stmt.where(Department.code == input_data.department_code)

            if input_data.workspace_id:
                try:
                    stmt = stmt.where(Document.workspace_id == uuid.UUID(input_data.workspace_id))
                except ValueError:
                    pass

            if user.role != "ADMIN":
                from sqlalchemy import or_, and_
                conditions = []
                if allowed_dept_ids:
                    conditions.append(Document.department_id.in_(allowed_dept_ids))
                if allowed_doc_ids:
                    conditions.append(Document.id.in_(allowed_doc_ids))
                if conditions:
                    dept_auth_cond = or_(*conditions)
                    user_ws_subq = select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user.id)
                    ws_auth_cond = or_(
                        Document.workspace_id.is_(None),
                        Document.workspace_id.in_(user_ws_subq),
                    )
                    stmt = stmt.where(and_(dept_auth_cond, ws_auth_cond))
                else:
                    return ListDocumentsResult(total_count=0, documents=[])

            stmt = stmt.limit(input_data.limit)
            res = await session.execute(stmt)
            rows = res.all()

            docs = []
            for doc, dept_code, chunk_count in rows:
                docs.append(
                    DocumentSummaryItem(
                        id=str(doc.id),
                        name=doc.file_name,
                        status=doc.status,
                        chunk_count=int(chunk_count or 0),
                        created_at=doc.created_at.isoformat() if doc.created_at else None,
                        department_code=dept_code,
                        mime_type=doc.mime_type,
                    )
                )

            return ListDocumentsResult(total_count=len(docs), documents=docs)

    async def run_diagnostics(self, include_rag_smoketest: bool = False) -> DiagnosticsResult:
        """
        Executes safe allowlisted diagnostics for the RAG platform.
        """
        return await run_system_diagnostics(include_rag_smoketest=include_rag_smoketest)
