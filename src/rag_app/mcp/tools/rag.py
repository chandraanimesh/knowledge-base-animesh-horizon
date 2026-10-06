"""
RAG Domain Tools for FastMCP.
Exposes:
- list_documents
- preview_document
- ingest_document
- rag_search
- rag_ask
- list_active_departments

Reuses existing IngestionService, RetrievalService, GenerationService, and RBAC authorization.
"""

from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.core.config import get_settings
from rag_app.db.models import Department, Document, DocumentChunk, User, Workspace, WorkspaceMember
from rag_app.generation.service import GenerationService
from rag_app.ingestion.service_v1 import IngestionService
from rag_app.mcp.gateway import mcp_gateway
from rag_app.retrieval.service import RetrievalService
from rag_app.schemas.rag import GenerateRequest
from rag_app.security.authorization import (
    authorize,
    authorize_document,
    get_allowed_departments,
    get_allowed_document_ids,
)

logger = logging.getLogger(__name__)
settings = get_settings()

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls", ".docx"}


# ── Underlying Handlers ───────────────────────────────────────────────────────

async def handle_list_documents(
    session: AsyncSession,
    user: User,
    department_code: Optional[str] = None,
    workspace_id: Optional[str] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List documents authorized for the caller."""
    allowed_dept_ids = await get_allowed_departments(session, user)
    allowed_doc_ids = await get_allowed_document_ids(session, user, "VIEW")

    if user.role != "ADMIN" and not allowed_dept_ids and not allowed_doc_ids:
        return {"documents": [], "total_count": 0}

    stmt = (
        select(
            Document,
            Department.code.label("department_code"),
            func.count(DocumentChunk.id).label("chunk_count"),
        )
        .outerjoin(Department, Document.department_id == Department.id)
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .group_by(Document.id, Department.code)
    )

    if user.role != "ADMIN":
        from sqlalchemy import and_, or_

        conditions = []
        if allowed_dept_ids:
            conditions.append(Document.department_id.in_(allowed_dept_ids))
        if allowed_doc_ids:
            conditions.append(Document.id.in_(allowed_doc_ids))

        if conditions:
            user_ws_subq = select(WorkspaceMember.workspace_id).where(
                WorkspaceMember.user_id == user.id
            )
            ws_auth_cond = or_(
                Document.workspace_id.is_(None),
                Document.workspace_id.in_(user_ws_subq),
            )
            stmt = stmt.where(and_(or_(*conditions), ws_auth_cond))
        else:
            return {"documents": [], "total_count": 0}

    if department_code:
        dept = await session.scalar(
            select(Department).where(Department.code == department_code.upper())
        )
        if dept:
            stmt = stmt.where(Document.department_id == dept.id)
        else:
            return {"documents": [], "total_count": 0}

    if workspace_id:
        try:
            ws_uuid = uuid.UUID(workspace_id)
            stmt = stmt.where(Document.workspace_id == ws_uuid)
        except ValueError:
            return {"documents": [], "total_count": 0}

    stmt = stmt.order_by(Document.created_at.desc()).limit(min(max(1, limit), 200))
    result = await session.execute(stmt)
    rows = result.all()

    documents = [
        {
            "id": str(doc.id),
            "file_name": doc.file_name,
            "department_code": dept_code or "GLOBAL",
            "status": doc.status,
            "chunk_count": chunk_count,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "workspace_id": str(doc.workspace_id) if doc.workspace_id else None,
        }
        for doc, dept_code, chunk_count in rows
    ]
    return {"documents": documents, "total_count": len(documents)}


async def handle_preview_document(
    document_id: str,
    session: AsyncSession,
    user: User,
) -> dict[str, Any]:
    """Read-only preview of document content for authorized caller."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid document UUID: {document_id}")

    doc = await session.get(Document, doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    is_allowed = await authorize_document(session, user, "VIEW", doc_uuid)
    if not is_allowed:
        raise HTTPException(status_code=403, detail="Not authorized to access this document")

    if doc.workspace_id and user.role != "ADMIN":
        ws_member = await session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == doc.workspace_id,
                WorkspaceMember.user_id == user.id,
            )
        )
        if not ws_member:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to access this workspace document",
            )

    dept = await session.get(Department, doc.department_id) if doc.department_id else None

    chunk_stmt = (
        select(DocumentChunk.text)
        .where(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index.asc())
        .limit(20)
    )
    chunks_res = await session.execute(chunk_stmt)
    chunks = chunks_res.scalars().all()
    preview_content = "\n\n".join(chunks) if chunks else "No chunk text indexed."

    return {
        "document_id": str(doc.id),
        "file_name": doc.file_name,
        "department_code": dept.code if dept else "GLOBAL",
        "mime_type": doc.mime_type,
        "status": doc.status,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "preview_content": preview_content,
    }


async def handle_ingest_document(
    filename: str,
    session: AsyncSession,
    user: User,
    content_base64: Optional[str] = None,
    content_text: Optional[str] = None,
    department_code: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> dict[str, Any]:
    """Ingests a document through IngestionService."""
    clean_filename = Path(filename).name
    suffix = Path(clean_filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        return {
            "success": False,
            "document_id": None,
            "file_name": clean_filename,
            "chunk_count": 0,
            "skipped": True,
            "reason": f"Unsupported file format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        }

    raw_bytes: bytes
    if content_base64:
        try:
            raw_bytes = base64.b64decode(content_base64)
        except Exception as exc:
            return {
                "success": False,
                "document_id": None,
                "file_name": clean_filename,
                "chunk_count": 0,
                "skipped": True,
                "reason": f"Invalid base64 encoding: {exc}",
            }
    elif content_text is not None:
        raw_bytes = content_text.encode("utf-8")
    else:
        return {
            "success": False,
            "document_id": None,
            "file_name": clean_filename,
            "chunk_count": 0,
            "skipped": True,
            "reason": "Must provide either 'content_base64' or 'content_text'",
        }

    if len(raw_bytes) > settings.max_upload_size:
        return {
            "success": False,
            "document_id": None,
            "file_name": clean_filename,
            "chunk_count": 0,
            "skipped": True,
            "reason": f"File size ({len(raw_bytes)} bytes) exceeds maximum limit ({settings.max_upload_size} bytes)",
        }

    # Department validation and permission check
    target_dept_id: Optional[uuid.UUID] = None
    if department_code:
        dept = await session.scalar(
            select(Department).where(Department.code == department_code.upper())
        )
        if not dept:
            raise HTTPException(status_code=404, detail=f"Department '{department_code}' not found.")
        if not dept.is_active:
            raise HTTPException(status_code=400, detail=f"Department '{department_code}' is inactive.")
        
        if user.role != "ADMIN":
            has_upload = await authorize(session, user, "UPLOAD", dept.code)
            if not has_upload:
                raise HTTPException(
                    status_code=403,
                    detail=f"User lacks UPLOAD permission for department '{department_code}'.",
                )
        target_dept_id = dept.id
    else:
        if user.role == "ADMIN":
            dept = await session.scalar(
                select(Department).where(Department.is_active == True).limit(1)
            )
            target_dept_id = dept.id if dept else None
        else:
            user_depts = await get_allowed_departments(session, user)
            if not user_depts:
                raise HTTPException(
                    status_code=403,
                    detail="User does not have access to any department for upload.",
                )
            target_dept_id = user_depts[0]

    # Workspace validation
    ws_uuid: Optional[uuid.UUID] = None
    if workspace_id:
        try:
            ws_uuid = uuid.UUID(workspace_id)
            ws = await session.scalar(
                select(Workspace).where(
                    Workspace.id == ws_uuid,
                    Workspace.deleted_at.is_(None),
                    Workspace.is_active == True,
                )
            )
            if not ws:
                raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")
            if user.role != "ADMIN":
                ws_member = await session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == ws.id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not ws_member or ws_member.role not in ("OWNER", "EDITOR"):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Forbidden: Workspace role does not permit content upload.",
                    )
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    upload_dir = Path("uploaded_docs")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"mcp_{uuid.uuid4().hex}_{clean_filename}"
    saved_path = upload_dir / safe_name

    try:
        saved_path.write_bytes(raw_bytes)
        ingestion_service = IngestionService(session)
        resp = await ingestion_service.ingest_file(
            file_path=saved_path,
            department_id=target_dept_id,
            created_by=user.id,
            workspace_id=ws_uuid,
        )
        return {
            "success": not resp.skipped,
            "document_id": str(resp.document_id) if resp.document_id else None,
            "file_name": clean_filename,
            "chunk_count": getattr(resp, "chunk_count", getattr(resp, "chunks_created", 0)),
            "skipped": resp.skipped,
            "reason": resp.reason,
        }
    finally:
        saved_path.unlink(missing_ok=True)


async def handle_rag_search(
    query: str,
    session: AsyncSession,
    user: User,
    top_k: int = 5,
    rerank: bool = True,
    workspace_id: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve evidence chunks authorized for the caller."""
    ws_uuid: Optional[uuid.UUID] = None
    if workspace_id:
        try:
            ws_uuid = uuid.UUID(workspace_id)
            ws = await session.scalar(
                select(Workspace).where(
                    Workspace.id == ws_uuid,
                    Workspace.deleted_at.is_(None),
                    Workspace.is_active == True,
                )
            )
            if not ws:
                raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")
            if user.role != "ADMIN":
                dept = await session.get(Department, ws.department_id)
                if not dept or not (await authorize(session, user, "VIEW", dept.code)):
                    raise HTTPException(status_code=403, detail="Forbidden department access.")
                ws_member = await session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == ws.id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not ws_member:
                    raise HTTPException(status_code=403, detail="Forbidden workspace access.")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    allowed_departments = await get_allowed_departments(session, user)
    allowed_documents = await get_allowed_document_ids(session, user, "VIEW")

    if user.role != "ADMIN" and not allowed_departments and not allowed_documents:
        return {"query": query, "total_results": 0, "results": []}

    retrieval_service = RetrievalService(session)
    resp = await retrieval_service.retrieve(
        query=query,
        top_k=min(max(1, top_k), 20),
        rerank=rerank,
        allowed_departments=allowed_departments,
        allowed_documents=allowed_documents,
        workspace_id=ws_uuid,
    )

    results = [
        {
            "chunk_id": str(c.chunk_id),
            "document_id": str(c.document_id),
            "department_id": str(c.department_id) if c.department_id else None,
            "file_name": c.file_name,
            "chunk_index": c.chunk_index,
            "score": round(
                float(
                    c.reranker_score
                    if c.reranker_score is not None
                    else (c.hybrid_score if c.hybrid_score is not None else (c.vector_similarity or 0.0))
                ),
                4,
            ),
            "text": c.text,
            "page_number": c.page_number,
        }
        for c in resp.results
    ]
    return {"query": query, "total_results": len(results), "results": results}


async def handle_rag_ask(
    query: str,
    session: AsyncSession,
    user: User,
    top_k: int = 5,
    rerank: bool = True,
    workspace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a grounded, authorized answer with citations."""
    ws_uuid: Optional[uuid.UUID] = None
    if workspace_id:
        try:
            ws_uuid = uuid.UUID(workspace_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid workspace UUID: {workspace_id}")

    conv_uuid: Optional[uuid.UUID] = None
    if conversation_id:
        try:
            conv_uuid = uuid.UUID(conversation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid conversation UUID: {conversation_id}")

    allowed_departments = await get_allowed_departments(session, user)
    allowed_documents = await get_allowed_document_ids(session, user, "VIEW")

    retrieval_service = RetrievalService(session)
    generation_service = GenerationService(session)

    retrieval_resp = await retrieval_service.retrieve(
        query=query,
        top_k=min(max(1, top_k), 20),
        rerank=rerank,
        allowed_departments=allowed_departments,
        allowed_documents=allowed_documents,
        workspace_id=ws_uuid,
    )

    gen_req = GenerateRequest(
        query=query,
        top_k=min(max(1, top_k), 20),
        rerank=rerank,
        workspace_id=ws_uuid,
        conversation_id=conv_uuid,
    )

    gen_resp = await generation_service.generate_answer(gen_req, user=user)

    sources = [
        {
            "chunk_id": str(c.chunk_id),
            "document_id": str(c.document_id),
            "file_name": c.file_name,
            "page_number": c.page_number,
            "excerpt": c.text[:300] if c.text else "",
        }
        for c in (gen_resp.retrieved_chunks or [])
    ]

    return {
        "query": query,
        "answer": gen_resp.answer,
        "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in (gen_resp.citations or [])],
        "sources": sources,
        "confidence_score": getattr(gen_resp, "confidence_score", 1.0),
        "has_conflicts": bool(getattr(gen_resp, "conflicts", None)),
        "is_grounded": bool(gen_resp.citations),
        "conversation_id": getattr(gen_resp, "conversation_id", None),
    }


async def handle_list_active_departments(
    session: AsyncSession,
    user: User,
) -> list[dict[str, Any]]:
    """Return active departments for authenticated users."""
    result = await session.scalars(
        select(Department).where(Department.is_active == True).order_by(Department.code.asc())
    )
    return [
        {
            "id": str(d.id),
            "code": d.code,
            "name": d.name,
            "is_active": d.is_active,
            "created_at": d.created_at.isoformat() if hasattr(d, "created_at") and d.created_at else None,
        }
        for d in result.all()
    ]


# ── MCP Tool Callers ─────────────────────────────────────────────────────────

async def list_documents(
    department_code: Optional[str] = None,
    workspace_id: Optional[str] = None,
    limit: int = 50,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """List authorized documents available in the knowledge base."""
    return await mcp_gateway.execute(
        "list_documents",
        handle_list_documents,
        department_code=department_code,
        workspace_id=workspace_id,
        limit=limit,
        auth_token=auth_token,
    )


async def preview_document(
    document_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Preview authorized document content."""
    return await mcp_gateway.execute(
        "preview_document",
        handle_preview_document,
        document_id=document_id,
        auth_token=auth_token,
    )


async def ingest_document(
    filename: str,
    content_base64: Optional[str] = None,
    content_text: Optional[str] = None,
    department_code: Optional[str] = None,
    workspace_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Ingest a document into PostgreSQL / pgvector and BM25 index."""
    return await mcp_gateway.execute(
        "ingest_document",
        handle_ingest_document,
        filename=filename,
        content_base64=content_base64,
        content_text=content_text,
        department_code=department_code,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


async def rag_search(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    workspace_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve ranked evidence chunks from hybrid search."""
    return await mcp_gateway.execute(
        "rag_search",
        handle_rag_search,
        query=query,
        top_k=top_k,
        rerank=rerank,
        workspace_id=workspace_id,
        auth_token=auth_token,
    )


async def rag_ask(
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    workspace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Answer a question using the end-to-end RAG application service."""
    return await mcp_gateway.execute(
        "rag_ask",
        handle_rag_ask,
        query=query,
        top_k=top_k,
        rerank=rerank,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        auth_token=auth_token,
    )


async def list_active_departments(
    auth_token: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List all active departments."""
    return await mcp_gateway.execute(
        "list_active_departments",
        handle_list_active_departments,
        auth_token=auth_token,
    )
