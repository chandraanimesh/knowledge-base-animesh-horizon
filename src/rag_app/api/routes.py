from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.session import get_db_session
from rag_app.generation.service import GenerationService
from rag_app.ingestion.service_v1 import IngestionService
from rag_app.retrieval.service import RetrievalService
from rag_app.schemas.rag import (
    GenerateRequest,
    GenerateResponse,
    IngestResponse,
    RetrievalResponse,
)
from rag_app.schemas.users import DepartmentResponse, DocumentResponse
from rag_app.api.dependencies.auth import RequirePermission, get_current_user
from rag_app.db.models import User, Department, Document, DocumentChunk
from rag_app.security.authorization import get_allowed_departments, authorize_document, get_allowed_document_ids

router = APIRouter(tags=["rag"])

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
TopK = Annotated[int, Query(ge=1, le=20)]


def get_generation_service(session: DbSession) -> GenerationService:
    return GenerationService(session)


def get_retrieval_service(session: DbSession) -> RetrievalService:
    return RetrievalService(session)


def get_ingestion_service(session: DbSession) -> IngestionService:
    return IngestionService(session)


GenerationDep = Annotated[GenerationService, Depends(get_generation_service)]
RetrievalDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    session: DbSession,
    user: User = Depends(get_current_user),
) -> list[DocumentResponse]:
    """
    List documents authorized for the current user.
    ADMIN sees all documents globally with uploader audit identity and workspace info.
    USER sees documents where they have active department VIEW or explicit document-level VIEW,
    AND for workspace documents, where they are a workspace member.
    """
    from rag_app.db.models import Workspace, WorkspaceMember

    allowed_dept_ids = await get_allowed_departments(session, user)
    allowed_doc_ids = await get_allowed_document_ids(session, user, "VIEW")

    # If non-admin has neither allowed departments nor allowed documents, return empty list immediately
    if user.role != "ADMIN" and not allowed_dept_ids and not allowed_doc_ids:
        return []

    stmt = (
        select(
            Document,
            Department.code.label("department_code"),
            Department.name.label("department_name"),
            User.email.label("uploader_email"),
            Workspace.code.label("workspace_code"),
            Workspace.name.label("workspace_name"),
        )
        .outerjoin(Department, Document.department_id == Department.id)
        .outerjoin(User, Document.created_by == User.id)
        .outerjoin(Workspace, Document.workspace_id == Workspace.id)
    )

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
            return []

    stmt = stmt.order_by(Document.created_at.desc())

    results = await session.execute(stmt)
    rows = results.all()

    doc_responses = []
    for doc, dept_code, dept_name, uploader_email, ws_code, ws_name in rows:
        uploader_name = None
        if uploader_email:
            local = uploader_email.split("@")[0]
            uploader_name = local.capitalize()

        doc_responses.append(
            DocumentResponse(
                id=doc.id,
                file_name=doc.file_name,
                department_id=doc.department_id,
                department_code=dept_code,
                department_name=dept_name,
                workspace_id=doc.workspace_id,
                workspace_code=ws_code,
                workspace_name=ws_name,
                uploaded_by_id=doc.created_by,
                uploaded_by_email=uploader_email,
                uploaded_by_name=uploader_name,
                mime_type=doc.mime_type,
                status=doc.status,
                created_at=doc.created_at,
                preview_url=f"/documents/{doc.id}/preview",
            )
        )
    return doc_responses


@router.get("/documents/{document_id}/preview")
async def preview_document(
    document_id: UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
):
    """
    Read-only document preview / download reference without re-ingesting.
    Authorized user-scoped check: ADMIN can access globally, USER via department VIEW or document VIEW
    and workspace membership where applicable.
    """
    from rag_app.db.models import WorkspaceMember

    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    is_allowed = await authorize_document(session, user, "VIEW", document_id)
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
            raise HTTPException(status_code=403, detail="Not authorized to access this workspace document")

    # Check if original physical file exists on disk
    if doc.file_path and Path(doc.file_path).exists():
        return FileResponse(
            path=doc.file_path,
            filename=doc.file_name,
            media_type=doc.mime_type or "application/octet-stream",
        )

    # Fallback to chunk texts if raw file was not retained
    chunk_stmt = (
        select(DocumentChunk.text)
        .where(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    chunks_res = await session.execute(chunk_stmt)
    chunks = chunks_res.scalars().all()
    if chunks:
        combined_text = "\n\n".join(chunks)
        return PlainTextResponse(content=combined_text, media_type="text/plain; charset=utf-8")

    raise HTTPException(status_code=404, detail="Document content not available for preview")


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(
    department_code: str,
    service: IngestionDep,
    session: DbSession,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    workspace_id: Optional[UUID] = Query(None, description="Optional Workspace UUID to assign document to"),
    user: User = Depends(RequirePermission("UPLOAD")),
) -> IngestResponse:
    from rag_app.knowledge.worker import run_document_knowledge_extraction
    from rag_app.db.models import Workspace, WorkspaceMember
    
    allowed_suffixes = {".pdf", ".txt", ".md",".csv",".xlsx",".xls", ".docx"}

    original_name = Path(file.filename or "uploaded_file").name
    suffix = Path(original_name).suffix.lower()

    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format.",
        )

    from rag_app.core.config import get_settings
    settings = get_settings()

    upload_dir = Path("uploaded_docs")
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid4().hex}_{original_name}"
    saved_path = upload_dir / safe_name
    processed_path = None
    ingest_success = False

    try:
        # Stream upload and enforce size limit
        total_bytes = 0
        CHUNK_SIZE = 1024 * 1024  # 1MB
        with open(saved_path, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_size:
                    raise HTTPException(
                        status_code=413,
                        detail="File size exceeds maximum allowed limit."
                    )
                f.write(chunk)

        # Get department ID
        from sqlalchemy import select
        from rag_app.db.models import Department
        dept = await session.scalar(select(Department).where(Department.code == department_code))
        if not dept:
            raise HTTPException(status_code=404, detail=f"Department '{department_code}' not found.")
        if not dept.is_active:
            raise HTTPException(status_code=400, detail=f"Department '{department_code}' is inactive.")
        
        # Validate workspace if provided
        if workspace_id:
            from sqlalchemy import or_
            ws = await session.scalar(
                select(Workspace).where(
                    or_(Workspace.id == workspace_id, Workspace.department_id == workspace_id),
                    Workspace.deleted_at.is_(None),
                    Workspace.is_active == True,
                )
            )
            if not ws:
                raise HTTPException(
                    status_code=404,
                    detail=f"Workspace '{workspace_id}' not found or inactive.",
                )
            if ws.department_id != dept.id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Workspace '{ws.name}' belongs to department '{ws.department_id}', not '{dept.code}'.",
                )
            if user.role != "ADMIN":
                ws_member = await session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == ws.id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not ws_member:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Forbidden: You are not a member of workspace '{ws.name}'.",
                    )
                if ws_member.role not in ("OWNER", "EDITOR"):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Forbidden: Workspace role '{ws_member.role}' does not permit content upload.",
                    )

        # Ingestion Layer
        try:
            resp = await service.ingest_file(
                saved_path,
                department_id=dept.id,
                created_by=user.id,
                workspace_id=workspace_id,
            )
            # Only schedule the background task if a document was actually processed/created
            if not resp.skipped and resp.document_id:
                from uuid import UUID
                background_tasks.add_task(run_document_knowledge_extraction, UUID(resp.document_id))
            ingest_success = True
            return resp
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ingestion failed for {safe_name}: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=500,
                detail="Document ingestion failed."
            )

    finally:
        # If ingestion was not successful, clean up temporary file
        if not ingest_success:
            saved_path.unlink(missing_ok=True)
        if processed_path and processed_path != saved_path:
            processed_path.unlink(missing_ok=True)


@router.post("/chat/ask", response_model=GenerateResponse)
async def ask_question(
    request: GenerateRequest,
    service: GenerationDep,
    retrieval_service: RetrievalDep,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> GenerateResponse:
    from rag_app.core.safety import check_sensitive_request
    from rag_app.research.graph import research_graph
    from rag_app.core.observability import init_request_metrics, finalize_request_metrics
    from rag_app.conversation.service import ConversationService
    from rag_app.db.models import Workspace, WorkspaceMember
    from rag_app.security.authorization import authorize
    
    request_id = str(uuid4())
    init_request_metrics(request_id)
    
    conv_service = ConversationService(session)
    # Get or create conversation (with deterministic title from first query if new, and workspace validation)
    conv = await conv_service.get_or_create_conversation(
        user=user,
        conversation_id=request.conversation_id,
        initial_query=request.query,
        workspace_id=request.workspace_id,
    )
    effective_workspace_id = conv.workspace_id or request.workspace_id

    # 1. Global Admin Chat check & authorization (Requirement 1)
    if effective_workspace_id is None:
        if user.role != "ADMIN":
            raise HTTPException(
                status_code=403,
                detail="Forbidden: Global chat across departments is restricted to administrators.",
            )
        from rag_app.db.models import Department
        active_depts = (await session.scalars(select(Department.id).where(Department.is_active == True))).all()
        allowed_departments = list(active_depts)
        workspace_thread_context = []
    else:
        # Department workspace chat: scoped strictly to this workspace's department
        ws = await conv_service._validate_workspace_access(user, effective_workspace_id)
        effective_workspace_id = ws.id
        allowed_departments = [ws.department_id]
        
        # Retrieve relevant messages from prior threads in the SAME workspace
        workspace_thread_context = await conv_service.search_workspace_threads(
            user=user,
            workspace_id=effective_workspace_id,
            query=request.query,
            current_conversation_id=conv.id,
            limit=5,
        )
    
    # Retrieve prior conversation history (latest 5 turns = max 10 messages) before adding current query
    conv_history = await conv_service.get_conversation_context_window(
        conversation_id=conv.id,
        max_turns=5,
    )
    
    # Persist user's incoming message
    await conv_service.save_user_message(
        conversation_id=conv.id,
        content=request.query,
    )
    
    safety = check_sensitive_request(
        request.query,
        can_view_pii=False,
    )
    if safety.blocked:
        metrics = finalize_request_metrics()
        blocked_resp = GenerateResponse(
            answer=safety.response or "This request cannot be completed.",
            citations=[],
            retrieved_chunks=[],
            execution_metrics=metrics,
            conversation_id=str(conv.id),
            conversation_title=conv.title,
        )
        await conv_service.save_assistant_message(
            conversation_id=conv.id,
            content=blocked_resp.answer,
            sources=[],
            execution_metrics=metrics.model_dump() if metrics else None,
        )
        return blocked_resp

    if allowed_departments is not None and len(allowed_departments) == 0:
        metrics = finalize_request_metrics()
        auth_resp = GenerateResponse(
            answer="You do not have access to any departments to fulfill this request.",
            citations=[],
            retrieved_chunks=[],
            execution_metrics=metrics,
            conversation_id=str(conv.id),
            conversation_title=conv.title,
        )
        await conv_service.save_assistant_message(
            conversation_id=conv.id,
            content=auth_resp.answer,
            sources=[],
            execution_metrics=metrics.model_dump() if metrics else None,
        )
        return auth_resp
        
    initial_state = {
        "query": request.query,
        "top_k": request.top_k,
        "rerank": request.rerank,
        "user": user,
        "allowed_departments": allowed_departments,
        "workspace_id": effective_workspace_id,
        "conversation_history": conv_history,
        "workspace_thread_context": workspace_thread_context,
    }
    
    config = {
        "configurable": {
            "retrieval_service": retrieval_service,
            "generation_service": service,
            "db_session": session,
        }
    }
    
    final_state = await research_graph.ainvoke(initial_state, config)
    metrics = finalize_request_metrics()
    
    resp: GenerateResponse = final_state["final_response"]
    if resp is not None:
        resp.execution_metrics = metrics
        resp.conversation_id = str(conv.id)
        
        # Persist assistant message in DB (including document chunks and retrieved workspace thread sources)
        sources_payload = [
            {
                "type": "document",
                "title": f"{c.file_name}{f' (Page {c.page_number})' if c.page_number else ''}",
                "excerpt": c.text,
            }
            for c in resp.retrieved_chunks
        ]
        for t in workspace_thread_context:
            sources_payload.append({
                "type": "thread",
                "title": f"Thread: \"{t['thread_title']}\"",
                "thread_id": t["thread_id"],
                "thread_title": t["thread_title"],
                "message_id": t["message_id"],
                "role": t["role"],
                "excerpt": t["content"],
                "created_at": t.get("created_at"),
            })
        metrics_payload = metrics.model_dump() if metrics else None
        asst_msg = await conv_service.save_assistant_message(
            conversation_id=conv.id,
            content=resp.answer,
            sources=sources_payload,
            execution_metrics=metrics_payload,
        )
        resp.message_id = str(asst_msg.id)

        # Automatic Thread Naming (Requirement 3)
        # If thread has default/placeholder title, generate descriptive title and persist
        if conv_service.is_placeholder_title(conv.title):
            try:
                new_title = await conv_service.generate_llm_title(
                    query=request.query,
                    assistant_response=resp.answer,
                    llm=getattr(service, "llm", None),
                )
                if new_title and new_title != conv.title:
                    conv.title = new_title
                    conv.updated_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception as title_err:
                import logging
                logging.getLogger(__name__).warning(f"Non-fatal title generation error: {title_err}")

        resp.conversation_title = conv.title
        
    return resp


@router.get("/retrieve", response_model=RetrievalResponse)
async def retrieve_only(
    query: str,
    service: RetrievalDep,
    session: DbSession,
    user: User = Depends(get_current_user),
    top_k: TopK = 5,
    rerank: bool = False,
    workspace_id: Optional[UUID] = Query(None, description="Scope retrieval to a specific Workspace"),
) -> RetrievalResponse:
    from rag_app.db.models import Workspace, WorkspaceMember
    from rag_app.security.authorization import authorize

    if workspace_id:
        from sqlalchemy import or_
        ws = await session.scalar(
            select(Workspace).where(
                or_(Workspace.id == workspace_id, Workspace.department_id == workspace_id),
                Workspace.deleted_at.is_(None),
                Workspace.is_active == True,
            )
        )
        if not ws:
            raise HTTPException(
                status_code=404,
                detail=f"Workspace '{workspace_id}' not found or inactive.",
            )
        if user.role != "ADMIN":
            dept = await session.get(Department, ws.department_id)
            if not dept or not (await authorize(session, user, "VIEW", dept.code)):
                raise HTTPException(
                    status_code=403,
                    detail=f"Forbidden: You do not have access to department '{dept.code if dept else ''}'.",
                )
            ws_member = await session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == ws.id,
                    WorkspaceMember.user_id == user.id,
                )
            )
            if not ws_member:
                raise HTTPException(
                    status_code=403,
                    detail=f"Forbidden: You are not a member of workspace '{ws.name}'.",
                )

    allowed_departments = await get_allowed_departments(session, user)
    allowed_documents = await get_allowed_document_ids(session, user, "VIEW")

    if (
        allowed_departments is not None
        and len(allowed_departments) == 0
        and (allowed_documents is not None and len(allowed_documents) == 0)
    ):
        return RetrievalResponse(query=query, results=[])

    return await service.retrieve(
        query=query,
        top_k=top_k,
        rerank=rerank,
        allowed_departments=allowed_departments,
        allowed_documents=allowed_documents,
        workspace_id=workspace_id,
    )



@router.get("/departments", response_model=list[DepartmentResponse])
async def list_active_departments(
    session: DbSession,
    _: User = Depends(get_current_user),
):
    """Return active departments for authenticated users (e.g. for Request Access dropdown)."""
    result = await session.scalars(
        select(Department).where(Department.is_active == True).order_by(Department.code.asc())
    )
    return result.all()


@router.get("/departments/{department_code}/status")
async def get_department_status(
    department_code: str,
    user: User = Depends(RequirePermission("VIEW")),
):
    """
    Test endpoint for Phase 3 Authorization.
    Requires an authenticated user with VIEW permission on the department.
    """
    return {"status": "ok", "department": department_code, "user_id": str(user.id)}