from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy import select, or_, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.session import get_db_session
from rag_app.generation.service import GenerationService
from rag_app.ingestion.service_v1 import IngestionService
from rag_app.retrieval.service import RetrievalService
from rag_app.schemas.rag import (
    DocumentChunkInfo,
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
from rag_app.core.filenames import derive_original_filename
from sqlalchemy.orm import selectinload

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
                original_file_name=derive_original_filename(doc.file_name, doc.file_path),
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


@router.get("/documents/{document_id}/chunks", response_model=list[DocumentChunkInfo])
async def list_document_chunks(
    document_id: UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> list[DocumentChunkInfo]:
    """
    List chunks for an authorized document for evidence discovery and selection.
    Strictly enforces document VIEW permission and workspace authorization.
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

    stmt = (
        select(DocumentChunk)
        .options(selectinload(DocumentChunk.node))
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    chunks = (await session.scalars(stmt)).all()

    original_fname = derive_original_filename(doc.file_name, doc.file_path)

    return [
        DocumentChunkInfo(
            id=str(c.id),
            document_id=str(c.document_id),
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            section_title=c.node.title if c.node else None,
            section_ordinal=c.node.ordinal_path if c.node else None,
            excerpt=c.text[:400] if c.text else "",
            file_name=doc.file_name,
            original_file_name=original_fname,
        )
        for c in chunks
    ]


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(
    department_code: str,
    service: IngestionDep,
    session: DbSession,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Query(None, description="Optional Workspace UUID or Code to assign document to"),
    user: User = Depends(RequirePermission("UPLOAD")),
) -> IngestResponse:
    from rag_app.knowledge.worker import run_document_knowledge_extraction
    from rag_app.db.models import Workspace, WorkspaceMember
    from rag_app.conversation.service import ConversationService
    
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
        target_ws_id: Optional[UUID] = None
        if workspace_id and str(workspace_id).strip():
            conv_service = ConversationService(session)
            ws = await conv_service._validate_workspace_access(user, workspace_id)
            target_ws_id = ws.id
            if ws.department_id != dept.id and user.role != "ADMIN":
                # Align department if needed or validate
                pass

        # Ingestion Layer
        try:
            resp = await service.ingest_file(
                saved_path,
                department_id=dept.id,
                created_by=user.id,
                workspace_id=target_ws_id,
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
    ws = None
    if effective_workspace_id is None:
        # Global Chat: search across all authorized workspaces
        if user.role == "ADMIN":
            from rag_app.db.models import Department
            active_depts = (await session.scalars(select(Department.id).where(Department.is_active == True, Department.deleted_at.is_(None)))).all()
            allowed_departments = list(active_depts)
        else:
            allowed_departments = await get_allowed_departments(session, user)

        # Policy containment: Global Chat does not retrieve raw employee conversation threads across workspaces.
        # Shared organizational knowledge requires explicit sharing/approval policies, not unapproved raw ChatMessage retrieval.
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
    
    # Retrieve prior conversation history (latest 5 turns = max 10 messages) with memory seam retrieval
    conv_history = await conv_service.get_conversation_context_window(
        conversation_id=conv.id,
        max_turns=5,
        query=request.query,
    )
    
    # Persist user's incoming message
    user_msg = await conv_service.save_user_message(
        conversation_id=conv.id,
        content=request.query,
    )

    # 2. Deterministic routing for Personal Memory slash commands
    query_stripped = request.query.strip()
    cmd_lower = query_stripped.split(maxsplit=1)[0].lower() if query_stripped.startswith("/") else ""

    if cmd_lower in ("/rem", "/remember"):
        from rag_app.memory.jobs import create_memory_job, execute_memory_job
        from rag_app.memory.schemas import JobStatus

        # Capture durable job referencing user message before execution
        job = await create_memory_job(
            session=session,
            owner_user_id=user.id,
            source_text=request.query,
            trigger_kind="explicit_rem",
            workspace_id=effective_workspace_id,
            source_message_id=user_msg.id,
            conversation_id=conv.id,
        )

        job_status, materialized_items, reason = await execute_memory_job(
            session=session,
            job_id=job.id,
            auto_embed=True,
        )

        if job_status == JobStatus.SUCCEEDED and materialized_items:
            record = materialized_items[0]
            scope_desc = f"workspace '{ws.name}'" if (effective_workspace_id and ws) else "global profile"
            ans = f"🧠 **Memory saved** ({scope_desc}):\n> \"{record.content}\""
        elif job_status == JobStatus.SKIPPED_NOT_MEMORABLE:
            ans = reason or "⚠️ Malformed command. Usage: `/rem <text to remember>`\n*Example:* `/rem I prefer responses in concise bullet points.`"
        else:
            # RETRY_PENDING or DEAD_LETTER: never falsely report saved
            ans = "⏳ **Memory capture pending**: Your statement has been securely captured and queued for background processing."

        asst_msg = await conv_service.save_assistant_message(
            conversation_id=conv.id,
            content=ans,
            sources=[],
        )
        metrics = finalize_request_metrics()
        return GenerateResponse(
            answer=ans,
            citations=[],
            retrieved_chunks=[],
            execution_metrics=metrics,
            conversation_id=str(conv.id),
            conversation_title=conv.title,
            message_id=str(asst_msg.id),
        )

    elif cmd_lower in ("/memory", "/memories"):
        from rag_app.memory.materializer import MemoryMaterializer
        records = await MemoryMaterializer.get_user_memories(
            session=session,
            user_id=user.id,
            workspace_id=effective_workspace_id,
            include_global=True,
            limit=50,
        )
        if not records:
            ans = "🧠 You currently have no active personal memories saved."
        else:
            lines = ["### 🧠 Active Personal Memories\n"]
            for r in records:
                scope_tag = "🌐 Global" if r.workspace_id is None else "📁 Workspace"
                type_tag = r.memory_type.value if hasattr(r.memory_type, "value") else str(r.memory_type)
                lines.append(f"- [{scope_tag} | {type_tag}] {r.content}")
            lines.append("\n*To remove a memory, use `/forget <keyword or phrase>`.*")
            ans = "\n".join(lines)

        asst_msg = await conv_service.save_assistant_message(
            conversation_id=conv.id,
            content=ans,
            sources=[],
        )
        metrics = finalize_request_metrics()
        return GenerateResponse(
            answer=ans,
            citations=[],
            retrieved_chunks=[],
            execution_metrics=metrics,
            conversation_id=str(conv.id),
            conversation_title=conv.title,
            message_id=str(asst_msg.id),
        )

    elif cmd_lower in ("/forget", "/forget_memory"):
        arg_text = query_stripped[len(cmd_lower):].strip()
        if not arg_text:
            ans = "⚠️ Malformed command. Usage: `/forget <keyword or phrase>`\n*Example:* `/forget bullet points`"
        else:
            from rag_app.memory.materializer import MemoryMaterializer
            target_uuid = None
            try:
                target_uuid = UUID(arg_text)
            except (ValueError, TypeError, AttributeError):
                target_uuid = None

            if target_uuid:
                archived = await MemoryMaterializer.archive_memory(
                    session=session,
                    memory_id=target_uuid,
                    user_id=user.id,
                )
                if archived:
                    ans = "🗑️ Memory has been archived."
                else:
                    ans = "⚠️ Memory not found or you are not authorized to remove it."
            else:
                all_user_mems = await MemoryMaterializer.get_user_memories(
                    session=session,
                    user_id=user.id,
                    workspace_id=effective_workspace_id,
                    include_global=True,
                    limit=100,
                )
                matches = [m for m in all_user_mems if arg_text.lower() in m.content.lower()]
                if not matches:
                    ans = f"⚠️ No active memories found matching '\"{arg_text}\"'."
                elif len(matches) == 1:
                    await MemoryMaterializer.archive_memory(
                        session=session,
                        memory_id=matches[0].id,
                        user_id=user.id,
                    )
                    ans = f"🗑️ Archived memory matching '\"{arg_text}\"':\n> \"{matches[0].content}\""
                else:
                    match_lines = [f"- {m.content}" for m in matches[:5]]
                    ans = f"⚠️ Multiple memories matched '\"{arg_text}\"'. Please be more specific with the text to forget:\n" + "\n".join(match_lines)

        asst_msg = await conv_service.save_assistant_message(
            conversation_id=conv.id,
            content=ans,
            sources=[],
        )
        metrics = finalize_request_metrics()
        return GenerateResponse(
            answer=ans,
            citations=[],
            retrieved_chunks=[],
            execution_metrics=metrics,
            conversation_id=str(conv.id),
            conversation_title=conv.title,
            message_id=str(asst_msg.id),
        )

    elif cmd_lower in ("/task", "/tasks"):
        from rag_app.tasks.service import TaskService
        from rag_app.schemas.tasks import TaskCreateRequest
        task_service = TaskService(session)
        sub_arg = query_stripped[len(cmd_lower):].strip()

        if not sub_arg or sub_arg.lower() in ("list", "all"):
            tasks, total = await task_service.query_tasks(user=user, filter_type="ALL", workspace_id=effective_workspace_id, limit=20)
            if not tasks:
                ans = "📋 You currently have no tasks or follow-ups recorded."
            else:
                lines = [f"### 📋 Follow-up Tasks ({total} total):\n"]
                for t in tasks:
                    due_str = f" | Due: {t.due_date.strftime('%Y-%m-%d %H:%M UTC')}" if t.due_date else ""
                    comp_str = f" | [{t.company_name}]" if t.company_name else ""
                    lines.append(f"- **`{t.id}`** [{t.status}{comp_str}{due_str}]: {t.title}")
                ans = "\n".join(lines)
        elif sub_arg.lower() == "today":
            tasks, total = await task_service.query_tasks(user=user, filter_type="DUE_TODAY", workspace_id=effective_workspace_id)
            if not tasks:
                ans = "📋 You have no follow-ups due today."
            else:
                lines = [f"### 📋 Follow-ups Due Today ({len(tasks)}):\n"]
                for t in tasks:
                    due_str = f" | Due: {t.due_date.strftime('%H:%M UTC')}" if t.due_date else ""
                    comp_str = f" | [{t.company_name}]" if t.company_name else ""
                    lines.append(f"- **`{t.id}`** [{t.status}{comp_str}{due_str}]: {t.title}")
                ans = "\n".join(lines)
        elif sub_arg.lower() == "overdue":
            tasks, total = await task_service.query_tasks(user=user, filter_type="OVERDUE", workspace_id=effective_workspace_id)
            if not tasks:
                ans = "✅ You have no overdue follow-ups."
            else:
                lines = [f"### ⚠️ Overdue Follow-ups ({len(tasks)}):\n"]
                for t in tasks:
                    due_str = f" | Due: {t.due_date.strftime('%Y-%m-%d')}" if t.due_date else ""
                    comp_str = f" | [{t.company_name}]" if t.company_name else ""
                    lines.append(f"- **`{t.id}`** [{t.status}{comp_str}{due_str}]: {t.title}")
                ans = "\n".join(lines)
        elif sub_arg.lower().startswith("add "):
            title_text = sub_arg[4:].strip()
            if not title_text:
                ans = "⚠️ Please specify a task description: `/task add <description>`"
            else:
                t = await task_service.create_task(user=user, request=TaskCreateRequest(title=title_text, workspace_id=effective_workspace_id, conversation_id=conv.id))
                ans = f"✅ **Task created**:\n> \"{t.title}\"\n\n*ID:* `{t.id}`\n\n*Note: Task has been persisted to your follow-up list. Automated push notification delivery is not configured.*"
        elif sub_arg.lower().startswith("done "):
            target_id = sub_arg[5:].strip()
            try:
                t = await task_service.complete_task(user=user, task_id=UUID(target_id))
                ans = f"✅ Task `{t.id}` marked as COMPLETED."
            except Exception as e:
                ans = f"⚠️ Could not complete task: {e}"
        else:
            ans = "⚠️ Usage:\n- `/tasks` or `/tasks today` or `/tasks overdue`\n- `/task add <description>`\n- `/task done <task_id>`"

        asst_msg = await conv_service.save_assistant_message(
            conversation_id=conv.id,
            content=ans,
            sources=[],
        )
        metrics = finalize_request_metrics()
        return GenerateResponse(
            answer=ans,
            citations=[],
            retrieved_chunks=[],
            execution_metrics=metrics,
            conversation_id=str(conv.id),
            conversation_title=conv.title,
            message_id=str(asst_msg.id),
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

    # Check natural language structured task and follow-up intent
    from rag_app.tasks.intent import detect_task_intent
    task_intent = detect_task_intent(request.query, user_timezone="UTC")
    if task_intent.is_task_intent:
        from rag_app.tasks.service import TaskService
        from rag_app.schemas.tasks import TaskCreateRequest
        task_service = TaskService(session)

        if task_intent.is_ambiguous:
            ans = task_intent.clarification_message or "Could you clarify what task or follow-up you would like to manage?"
            asst_msg = await conv_service.save_assistant_message(conversation_id=conv.id, content=ans, sources=[])
            metrics = finalize_request_metrics()
            return GenerateResponse(answer=ans, citations=[], retrieved_chunks=[], execution_metrics=metrics, conversation_id=str(conv.id), conversation_title=conv.title, message_id=str(asst_msg.id))

        if task_intent.intent_type == "CREATE_TASK":
            t = await task_service.create_task(
                user=user,
                request=TaskCreateRequest(
                    title=task_intent.action_text or "Follow-up",
                    due_date=task_intent.target_date,
                    company_name=task_intent.company_name,
                    workspace_id=effective_workspace_id,
                    conversation_id=conv.id,
                ),
            )
            due_fmt = f"\n- **Due:** {t.due_date.strftime('%Y-%m-%d %H:%M UTC')}" if t.due_date else ""
            comp_fmt = f"\n- **Company:** {t.company_name}" if t.company_name else ""
            ans = (
                f"✅ **Follow-up task created**:\n"
                f"> \"{t.title}\"{comp_fmt}{due_fmt}\n\n"
                f"*ID:* `{t.id}`\n\n"
                f"*Note: Task has been persisted to your follow-up list. Automated push notification delivery is not configured.*"
            )
            asst_msg = await conv_service.save_assistant_message(conversation_id=conv.id, content=ans, sources=[])
            metrics = finalize_request_metrics()
            return GenerateResponse(answer=ans, citations=[], retrieved_chunks=[], execution_metrics=metrics, conversation_id=str(conv.id), conversation_title=conv.title, message_id=str(asst_msg.id))

        elif task_intent.intent_type == "QUERY_DUE_TODAY":
            tasks, total = await task_service.query_tasks(user=user, filter_type="DUE_TODAY", workspace_id=effective_workspace_id)
            if not tasks:
                ans = "📋 You have no follow-up tasks due today."
            else:
                lines = [f"### 📋 Follow-ups Due Today ({len(tasks)}):\n"]
                for t in tasks:
                    comp_tag = f" [{t.company_name}]" if t.company_name else ""
                    time_tag = f" (at {t.due_date.strftime('%H:%M UTC')})" if t.due_date else ""
                    lines.append(f"- **{t.title}**{comp_tag}{time_tag} `[ID: {t.id}]`")
                ans = "\n".join(lines)
            asst_msg = await conv_service.save_assistant_message(conversation_id=conv.id, content=ans, sources=[])
            metrics = finalize_request_metrics()
            return GenerateResponse(answer=ans, citations=[], retrieved_chunks=[], execution_metrics=metrics, conversation_id=str(conv.id), conversation_title=conv.title, message_id=str(asst_msg.id))

        elif task_intent.intent_type == "QUERY_OVERDUE":
            tasks, total = await task_service.query_tasks(user=user, filter_type="OVERDUE", workspace_id=effective_workspace_id)
            if not tasks:
                ans = "✅ You have no overdue follow-ups."
            else:
                lines = [f"### ⚠️ Overdue Follow-ups ({len(tasks)}):\n"]
                for t in tasks:
                    comp_tag = f" [{t.company_name}]" if t.company_name else ""
                    due_tag = f" (was due {t.due_date.strftime('%Y-%m-%d %H:%M UTC')})" if t.due_date else ""
                    lines.append(f"- **{t.title}**{comp_tag}{due_tag} `[ID: {t.id}]`")
                ans = "\n".join(lines)
        elif task_intent.intent_type == "QUERY_UPCOMING":
            tasks, total = await task_service.query_tasks(user=user, filter_type="UPCOMING", workspace_id=effective_workspace_id)
            if task_intent.action_text == "Sales":
                tasks = [t for t in tasks if ("sales" in t.title.lower()) or (t.company_name and "sales" in t.company_name.lower())] or tasks
            if not tasks:
                ans = "📋 You have no open follow-up tasks recorded."
            else:
                lines = [f"### 📋 Upcoming Follow-ups and Commitments ({len(tasks)}):\n"]
                for t in tasks:
                    comp_tag = f" [{t.company_name}]" if t.company_name else ""
                    due_tag = f" (due {t.due_date.strftime('%Y-%m-%d %H:%M UTC')})" if t.due_date else ""
                    lines.append(f"- **{t.title}**{comp_tag}{due_tag} `[ID: {t.id}]`")
                ans = "\n".join(lines)
            asst_msg = await conv_service.save_assistant_message(conversation_id=conv.id, content=ans, sources=[])
            metrics = finalize_request_metrics()
            return GenerateResponse(answer=ans, citations=[], retrieved_chunks=[], execution_metrics=metrics, conversation_id=str(conv.id), conversation_title=conv.title, message_id=str(asst_msg.id))

        elif task_intent.intent_type == "QUERY_COMPANY":
            from rag_app.graph.entity_resolver import CanonicalEntityResolver
            from rag_app.graph.fusion import MultiSourceEvidenceFusionService

            resolver = CanonicalEntityResolver(session)
            resolution = await resolver.resolve_company(
                name_or_query=task_intent.company_name,
                allowed_departments=allowed_departments,
            )

            # Phase 2 Ambiguity Guard: Ask for clarification if entity cannot be safely resolved
            if resolution.is_ambiguous:
                ans = resolution.clarification_prompt or f"Multiple companies matched '{task_intent.company_name}'. Please clarify which company you meant."
                asst_msg = await conv_service.save_assistant_message(conversation_id=conv.id, content=ans, sources=[])
                metrics = finalize_request_metrics()
                return GenerateResponse(
                    answer=ans,
                    citations=[],
                    retrieved_chunks=[],
                    execution_metrics=metrics,
                    conversation_id=str(conv.id),
                    conversation_title=conv.title,
                    message_id=str(asst_msg.id),
                )

            target_company_name = resolution.canonical_name or task_intent.company_name

            # Phase 4 Multi-Source Evidence Fusion across tasks, notes, graph, documents, memory
            fusion_service = MultiSourceEvidenceFusionService(session)
            fused = await fusion_service.fuse_company_evidence(
                user=user,
                company_name=target_company_name,
                query=request.query,
                canonical_entity=resolution.entity,
                allowed_departments=allowed_departments,
                workspace_id=effective_workspace_id,
            )

            asst_msg = await conv_service.save_assistant_message(
                conversation_id=conv.id,
                content=fused.formatted_synthesis,
                sources=fused.sources_payload,
            )
            metrics = finalize_request_metrics()
            fused_chunks = fused.to_retrieved_chunks()
            citations = [c.file_name for c in fused_chunks]
            return GenerateResponse(
                answer=fused.formatted_synthesis,
                citations=citations,
                retrieved_chunks=fused_chunks,
                execution_metrics=metrics,
                conversation_id=str(conv.id),
                conversation_title=conv.title,
                message_id=str(asst_msg.id),
            )
        
    # Check for "Visualize this" action targeting the current/previous assistant response
    is_visualize_action = (
        (request.action == "visualize")
        or (request.target_message_id is not None)
        or (request.query.strip().lower() in ["visualize this", "visulaize this", "visualize", "visulaize"])
    )

    if is_visualize_action:
        # 1. Fetch the target assistant message
        if request.target_message_id:
            target_uuid = None
            try:
                target_uuid = UUID(str(request.target_message_id))
            except (ValueError, TypeError, AttributeError):
                target_uuid = None
            target_msg = await session.get(ChatMessage, target_uuid) if target_uuid else None
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
        
        from rag_app.context.sources import (
            deduplicate_sources_payload,
            build_personal_memory_source_item,
        )
        from rag_app.retrieval.sufficiency import is_personal_memory_relevant

        # Persist assistant message in DB with complete chunk metadata
        sources_payload = [
            {
                "type": "shared_knowledge" if c.chunk_id.startswith("shared_kn_") else "document",
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

        # Extract personal memories that were provided in conv_history and used as evidence
        personal_mems_candidates = []
        for msg in (conv_history or []):
            if msg.get("name") == "personal_memory" or (
                msg.get("role") == "system" and "[User Personal Preferences & Context]" in msg.get("content", "")
            ):
                if msg.get("personal_memories"):
                    personal_mems_candidates.extend(msg["personal_memories"])
                else:
                    for line in msg.get("content", "").split("\n"):
                        if line.startswith("- ["):
                            raw_text = line.split("]", 1)[-1].strip()
                            personal_mems_candidates.append({
                                "id": f"mem-{uuid.uuid5(uuid.NAMESPACE_DNS, raw_text)}",
                                "content": raw_text,
                                "canonical_key": raw_text[:32],
                            })

        for pmem in personal_mems_candidates:
            mem_content = pmem.get("content", "")
            # Only label evidence that actually came from personal memory and is relevant to the query/answer
            if is_personal_memory_relevant(request.query, mem_content):
                sources_payload.append(
                    build_personal_memory_source_item(
                        memory_id=pmem.get("id"),
                        memory_text=mem_content,
                        user_id=pmem.get("user_id", user.id),
                        workspace_id=pmem.get("workspace_id", effective_workspace_id),
                        canonical_key=pmem.get("canonical_key"),
                        created_at=pmem.get("created_at"),
                    )
                )

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

        # Deduplicate repeated sources (e.g. repeated thread cards for the same thread)
        sources_payload = deduplicate_sources_payload(sources_payload)
        resp.sources = sources_payload

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
    workspace_id: Optional[str] = Query(None, description="Scope retrieval to a specific Workspace"),
) -> RetrievalResponse:
    from rag_app.db.models import Workspace, WorkspaceMember
    from rag_app.security.authorization import authorize
    from rag_app.conversation.service import ConversationService

    target_ws_id: Optional[UUID] = None
    if workspace_id and str(workspace_id).strip():
        conv_service = ConversationService(session)
        ws = await conv_service._validate_workspace_access(user, workspace_id)
        target_ws_id = ws.id

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
        workspace_id=target_ws_id,
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