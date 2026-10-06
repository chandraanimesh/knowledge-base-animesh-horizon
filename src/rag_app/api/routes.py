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
    RetrievedChunk,
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
    from rag_app.core.observability import (
        init_request_metrics,
        finalize_request_metrics,
        record_request_context,
        record_failure_category,
    )
    from rag_app.schemas.rag import FailureCategory
    from rag_app.conversation.service import ConversationService
    from rag_app.db.models import ChatMessage, Conversation, Department, Workspace, WorkspaceMember
    from rag_app.security.authorization import authorize, get_allowed_departments
    
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
    record_request_context(
        workspace_id=str(effective_workspace_id) if effective_workspace_id else None,
        thread_id=str(conv.id),
    )

    # 1. Global vs Workspace Chat authorization & scoping
    if effective_workspace_id is None:
        # Global Chat: search across all authorized workspaces
        if user.role == "ADMIN":
            from rag_app.db.models import Department
            active_depts = (await session.scalars(select(Department.id).where(Department.is_active == True, Department.deleted_at.is_(None)))).all()
            allowed_departments = list(active_depts)
        else:
            allowed_departments = await get_allowed_departments(session, user)

        # Cross-workspace conversation retrieval across ALL authorized workspaces
        workspace_thread_context = await conv_service.search_workspace_threads(
            user=user,
            workspace_id=None,
            query=request.query,
            current_conversation_id=conv.id,
            limit=5,
        )
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
        record_failure_category(FailureCategory.QUERY_UNDERSTANDING)
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
        record_failure_category(FailureCategory.AUTHORIZATION)
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
        
    # Check for "Visualize this" action targeting the current/previous assistant response
    is_visualize_action = (
        (request.action == "visualize")
        or (request.target_message_id is not None)
        or (request.query.strip().lower() in ["visualize this", "visulaize this", "visualize", "visulaize"])
    )

    if is_visualize_action:
        # 1. Fetch the target assistant message
        if request.target_message_id:
            target_msg = await session.get(ChatMessage, request.target_message_id)
        else:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conv.id, ChatMessage.role == "assistant")
                .order_by(ChatMessage.created_at.desc())
                .limit(1)
            )
            target_msg = await session.scalar(stmt)

        if not target_msg or target_msg.conversation_id != conv.id or target_msg.role != "assistant":
            record_failure_category(FailureCategory.VISUALIZATION)
            metrics = finalize_request_metrics()
            safe_resp = GenerateResponse(
                answer="No prior assistant response found in this conversation to visualize.",
                citations=[],
                retrieved_chunks=[],
                execution_metrics=metrics,
                conversation_id=str(conv.id),
                conversation_title=conv.title,
            )
            asst_msg = await conv_service.save_assistant_message(
                conversation_id=conv.id,
                content=safe_resp.answer,
                sources=[],
                execution_metrics=metrics.model_dump() if metrics else None,
            )
            safe_resp.message_id = str(asst_msg.id)
            return safe_resp

        # 2. Reconstruct authorized chunks from target message's sources without second global retrieval
        reconstructed_chunks: list[RetrievedChunk] = []
        allowed_dept_strs = {str(d) for d in allowed_departments} if allowed_departments is not None else None

        for idx, s in enumerate(target_msg.sources or []):
            if s.get("type") == "document" or "excerpt" in s or "text" in s:
                chunk_dept = s.get("department_id")
                if allowed_dept_strs is not None and chunk_dept and chunk_dept != "None" and chunk_dept not in allowed_dept_strs:
                    continue  # Strict department boundary check
                reconstructed_chunks.append(
                    RetrievedChunk(
                        chunk_id=str(s.get("chunk_id") or f"src_{idx}"),
                        document_id=str(s.get("document_id") or "doc_prior"),
                        department_id=str(chunk_dept or (str(allowed_departments[0]) if allowed_departments else "None")),
                        file_name=str(s.get("file_name") or s.get("title") or "Document"),
                        chunk_index=int(s.get("chunk_index") or idx),
                        text=str(s.get("excerpt") or s.get("text") or ""),
                        page_number=s.get("page_number"),
                    )
                )

        if not reconstructed_chunks and target_msg.content:
            reconstructed_chunks.append(
                RetrievedChunk(
                    chunk_id="prior_answer_context",
                    document_id="doc_prior",
                    department_id=str(allowed_departments[0]) if allowed_departments else "None",
                    file_name="Previous Assistant Answer",
                    chunk_index=0,
                    text=target_msg.content,
                )
            )

        # 3. Explicit QueryPlan with CHART presentation intent (bypasses generic PROSE interpretation)
        from rag_app.schemas.rag import QueryPlan, InformationNeed, PresentationNeed, EvidenceNeed
        viz_plan = QueryPlan(
            semantic_query="visualize key metrics and opportunities",
            information_need=InformationNeed.AGGREGATION,
            presentation_need=PresentationNeed.CHART,
            evidence_need=EvidenceNeed.STRUCTURED,
        )

        from rag_app.retrieval.sufficiency import EvidenceSufficiencyEvaluator
        evaluator = EvidenceSufficiencyEvaluator()
        suff_eval = await evaluator.evaluate(
            query=request.query,
            chunks=reconstructed_chunks,
            query_plan=viz_plan,
            conversation_history=conv_history,
            workspace_thread_context=workspace_thread_context,
        )

        if not suff_eval.is_sufficient:
            record_failure_category(FailureCategory.VISUALIZATION)
            metrics = finalize_request_metrics()
            reason_text = suff_eval.reason or "The authorized evidence does not contain sufficient structured numeric data to generate a valid chart."
            safe_resp = GenerateResponse(
                answer=f"Unable to generate visualization: {reason_text}",
                visualization=None,
                citations=[],
                retrieved_chunks=reconstructed_chunks,
                execution_metrics=metrics,
                conversation_id=str(conv.id),
                conversation_title=conv.title,
            )
            asst_msg = await conv_service.save_assistant_message(
                conversation_id=conv.id,
                content=safe_resp.answer,
                sources=target_msg.sources,
                execution_metrics=metrics.model_dump() if metrics else None,
                visualization=None,
            )
            safe_resp.message_id = str(asst_msg.id)
            return safe_resp

        # 4. Generate grounded chart from context
        viz_prompt = (
            "Generate a chart visualizing the key metrics, opportunities, or structured comparison data from the authorized evidence. "
            "Output valid JSON with keys 'answer' and 'visualization'. Ensure all chart values are pure numbers and match the authorized evidence."
        )
        rendered_context = (
            f"Prior Answer:\n{target_msg.content}\n\nAuthorized Evidence:\n"
            + "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(reconstructed_chunks))
        )
        resp = await service.generate_from_context(
            query=viz_prompt,
            chunks=reconstructed_chunks,
            allowed_departments=allowed_departments,
            rendered_context=rendered_context,
            conversation_history=conv_history,
            workspace_thread_context=workspace_thread_context,
            workspace_id=effective_workspace_id,
            provider_overrides=request.provider_overrides,
            sufficiency_evaluation=suff_eval,
        )

        if not resp.visualization:
            record_failure_category(FailureCategory.VISUALIZATION)
            if "unable" not in resp.answer.lower() and "could not" not in resp.answer.lower():
                resp.answer = f"Unable to generate a valid chart from the authorized evidence.\n\n{resp.answer}"

        metrics = finalize_request_metrics()
        resp.execution_metrics = metrics
        resp.conversation_id = str(conv.id)
        resp.conversation_title = conv.title

        asst_msg = await conv_service.save_assistant_message(
            conversation_id=conv.id,
            content=resp.answer,
            sources=target_msg.sources,
            execution_metrics=metrics.model_dump() if metrics else None,
            visualization=resp.visualization,
        )
        resp.message_id = str(asst_msg.id)
        return resp

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
        
        # Persist assistant message in DB with complete chunk metadata
        sources_payload = [
            {
                "type": "document",
                "chunk_id": c.chunk_id,
                "document_id": str(c.document_id),
                "department_id": str(c.department_id),
                "file_name": c.file_name,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "title": f"{c.file_name}{f' (Page {c.page_number})' if c.page_number else ''}",
                "excerpt": c.text,
            }
            for c in resp.retrieved_chunks
        ]
        for t in workspace_thread_context:
            title_suffix = f" ({t['workspace_name']})" if t.get("workspace_name") else ""
            sources_payload.append({
                "type": "thread",
                "title": f"Thread: \"{t['thread_title']}\"{title_suffix}",
                "thread_id": t["thread_id"],
                "thread_title": t["thread_title"],
                "workspace_id": t.get("workspace_id"),
                "workspace_name": t.get("workspace_name"),
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
            visualization=resp.visualization,
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