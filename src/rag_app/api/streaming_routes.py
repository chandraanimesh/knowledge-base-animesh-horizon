"""Real-time SSE Streaming Endpoint

Provides /chat/stream using Server-Sent Events (SSE) with structured event typing:
start, token, tool_start, tool_result, trace, usage, error, done.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.conversation.service import ConversationService
from rag_app.core.config import get_settings
from rag_app.core.observability import (
    finalize_request_metrics,
    init_request_metrics,
    record_request_context,
)
from rag_app.core.stream_cleaner import StreamToolCleaner
from rag_app.db.models import Department, User
from rag_app.db.session import get_db_session
from rag_app.generation.service import GenerationService
from rag_app.retrieval.service import RetrievalService
from rag_app.schemas.rag import GenerateRequest
from rag_app.schemas.users import UserContext
from rag_app.security.authorization import get_allowed_departments

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat-streaming"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


def sse_event(event_name: str, payload: dict) -> str:
    """Formats payload as standard SSE event."""
    return f"event: {event_name}\ndata: {json.dumps(payload, default=str)}\n\n"


@router.post("/stream")
async def stream_chat(
    request: GenerateRequest,
    session: DbSession,
    user: CurrentUser,
) -> StreamingResponse:
    """
    Real-time Server-Sent Events (SSE) streaming endpoint for RAG chat.
    Emits structured events: start, token, tool_start, tool_result, trace, done, error.
    Isolates tool-call payloads from user text tokens.
    """
    request_id = str(uuid.uuid4())
    init_request_metrics(request_id)
    user_context = UserContext.from_user(user)

    conv_service = ConversationService(session)
    conv = await conv_service.get_or_create_conversation(
        user=user_context,
        conversation_id=request.conversation_id,
        initial_query=request.query,
        workspace_id=request.workspace_id,
    )
    conv_id = conv.id
    effective_workspace_id = getattr(conv, "workspace_id", None) or request.workspace_id

    # Resolve allowed departments
    if effective_workspace_id is None:
        if user_context.role == "ADMIN":
            active_depts = (await session.scalars(select(Department.id).where(Department.is_active == True))).all()
            allowed_departments = list(active_depts)
        else:
            allowed_departments = await get_allowed_departments(session, user_context)
        workspace_thread_context = []
    else:
        ws = await conv_service._validate_workspace_access(user_context, effective_workspace_id)
        effective_workspace_id = ws.id
        allowed_departments = [ws.department_id]
        workspace_thread_context = await conv_service.search_workspace_threads(
            user=user_context,
            workspace_id=effective_workspace_id,
            query=request.query,
            current_conversation_id=conv_id,
            limit=5,
        )

    # Context window & user message persistence
    conv_history = await conv_service.get_conversation_context_window(
        conversation_id=conv_id,
        max_turns=5,
        query=request.query,
    )
    user_msg = await conv_service.save_user_message(
        conversation_id=conv_id,
        content=request.query,
        commit=True,
    )

    # Conversational memory candidate capture
    try:
        query_stripped = (request.query or "").strip()
        from rag_app.memory.command_parser import detect_memory_candidate_rules
        from rag_app.core.safety import check_sensitive_request
        from rag_app.memory.jobs import create_memory_job

        is_candidate, is_transient, _ = detect_memory_candidate_rules(query_stripped)
        safety = check_sensitive_request(query_stripped, can_view_pii=False)

        if not is_transient and not safety.blocked:
            await create_memory_job(
                session=session,
                owner_user_id=user_context.id,
                source_text=request.query,
                trigger_kind="conversational_turn",
                workspace_id=effective_workspace_id,
                source_message_id=user_msg.id,
                conversation_id=conv_id,
            )
            await session.commit()
    except Exception as m_err:
        logger.warning(f"Conversational memory capture skipped: {m_err}")

    retrieval_service = RetrievalService(session)
    generation_service = GenerationService(session)

    from rag_app.orchestration.router import AdaptiveRequestRouter, AdaptiveRouteMode
    adaptive_router = AdaptiveRequestRouter()
    classification = await adaptive_router.classify(request.query)
    routing_mode = classification.route.value

    async def event_generator():
        yield sse_event("start", {
            "request_id": request_id,
            "conversation_id": str(conv_id),
            "query": request.query,
            "routing_mode": routing_mode,
            "routing_reason": classification.reason,
        })

        cleaner = StreamToolCleaner()
        accumulated_answer_parts = []
        retrieved_chunks = []
        citations = []

        try:
            # 1. Retrieve knowledge
            retrieval_resp = await retrieval_service.retrieve(
                query=request.query,
                top_k=request.top_k or 5,
                rerank=request.rerank if request.rerank is not None else True,
                allowed_departments=allowed_departments,
                workspace_id=effective_workspace_id,
            )
            retrieved_chunks = list(retrieval_resp.results)
            citations = [c.file_name for c in retrieved_chunks if c.file_name]

            # Emit trace event with retrieved chunks count
            yield sse_event("trace", {
                "event": "retrieval_complete",
                "chunks_count": len(retrieved_chunks),
                "strategy_trace": retrieval_resp.strategy_trace or {},
            })

            # 2. Stream from context
            stream_gen = generation_service.stream_from_context(
                query=request.query,
                chunks=retrieved_chunks,
                allowed_departments=allowed_departments,
                conversation_history=conv_history,
                workspace_thread_context=workspace_thread_context,
                workspace_id=effective_workspace_id,
                provider_overrides=request.provider_overrides,
            )

            async for item in stream_gen:
                item_type = item.get("type")
                if item_type == "token":
                    raw_token = item.get("content", "")
                    clean_text, tool_events = cleaner.process_chunk(raw_token)

                    # Emit any detected tool events
                    for te in tool_events:
                        yield sse_event("tool_start", te)

                    # Emit clean prose token delta
                    if clean_text:
                        accumulated_answer_parts.append(clean_text)
                        yield sse_event("token", {"delta": clean_text})

                elif item_type == "complete":
                    final_gen_resp = item.get("response")
                    if final_gen_resp and final_gen_resp.citations:
                        citations = final_gen_resp.citations

            # Flush cleaner buffer
            leftover = cleaner.flush()
            if leftover:
                accumulated_answer_parts.append(leftover)
                yield sse_event("token", {"delta": leftover})

            full_answer = "".join(accumulated_answer_parts).strip()
            if not full_answer:
                full_answer = "I could not find sufficient information to answer your query."

            # Save assistant message to database
            sources_payload = [
                {
                    "type": "document",
                    "chunk_id": c.chunk_id,
                    "document_id": str(c.document_id),
                    "file_name": c.file_name,
                    "excerpt": c.text,
                    "page_number": c.page_number,
                }
                for c in retrieved_chunks
            ]
            asst_msg = await conv_service.save_assistant_message(
                conversation_id=conv_id,
                content=full_answer,
                sources=sources_payload,
            )

            metrics = finalize_request_metrics()
            yield sse_event("done", {
                "message_id": str(asst_msg.id),
                "conversation_id": str(conv_id),
                "answer": full_answer,
                "citations": citations,
                "retrieved_chunks": [c.model_dump() for c in retrieved_chunks],
                "execution_metrics": metrics.model_dump() if metrics else None,
                "routing_mode": routing_mode,
                "routing_reason": classification.reason,
            })

        except asyncio.CancelledError:
            logger.info(f"Client disconnected from stream for request {request_id}")
            raise
        except Exception as exc:
            logger.error(f"Streaming error on request {request_id}: {exc}", exc_info=True)
            yield sse_event("error", {
                "message": str(exc),
                "code": 500,
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

