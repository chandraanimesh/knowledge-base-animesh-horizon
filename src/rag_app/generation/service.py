import logging

from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential
from rag_app.retrieval.query_parser import is_greeting
from rag_app.core.safety import check_sensitive_request
from rag_app.core.tracing import track
from rag_app.core.observability import record_stage_start, record_stage_end
from rag_app.providers.base.llm import LLMProvider
from rag_app.generation.llm_factory import get_llm
from rag_app.generation.prompt import build_rag_messages, format_docs
from rag_app.generation.citations import validate_and_render_citations
from rag_app.retrieval.service import RetrievalService
from rag_app.context.compiler import ContextCompiler
from rag_app.schemas.rag import ConflictNotice, GenerateRequest, GenerateResponse, RetrievedChunk
from langsmith import traceable
from rag_app.db.models import User
from rag_app.security.authorization import get_allowed_departments, get_allowed_document_ids
import uuid
logger = logging.getLogger(__name__)


class GenerationService:
    def __init__(
        self,
        session: AsyncSession,
        context_compiler: ContextCompiler | None = None,
    ):
        self.retrieval_service = RetrievalService(session)
        self.llm = get_llm()
        self.context_compiler = context_compiler or ContextCompiler()

    @track(name="generate_answer", phase="generation")
    async def generate_answer(
        self,
        request: GenerateRequest,
        user: User,
    ) -> GenerateResponse:

        safety = check_sensitive_request(
            request.query,
            can_view_pii=False,
        )
        logger.info(
            "safety_check_result",
            extra={
                "event": "safety_check_result",
                "query": request.query,
                "blocked": safety.blocked,
                "reason": safety.reason,
        },
)
        if safety.blocked:
            return GenerateResponse(
                answer=safety.response or "This request cannot be completed.",
                citations=[],
                retrieved_chunks=[],
            )
       
        allowed_departments = await get_allowed_departments(self.retrieval_service.session, user)
        allowed_documents = await get_allowed_document_ids(self.retrieval_service.session, user, "VIEW")

        if allowed_departments is not None and len(allowed_departments) == 0 and (allowed_documents is not None and len(allowed_documents) == 0):
            return GenerateResponse(
                answer="You do not have access to any departments to fulfill this request.",
                citations=[],
                retrieved_chunks=[],
            )
            

        retrieval_response = await self.retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            rerank=request.rerank,
            allowed_departments=allowed_departments,
            allowed_documents=allowed_documents,
        )

        compiled_context = self.context_compiler.compile(
            chunks=retrieval_response.results,
            query=request.query,
            allowed_departments=allowed_departments,
            is_web_context=False,
            strategy_trace=retrieval_response.strategy_trace,
            chunk_provenance=retrieval_response.chunk_provenance,
            chunk_hierarchy=retrieval_response.chunk_hierarchy,
        )

        return await self.generate_from_context(
            query=request.query,
            chunks=compiled_context.chunks,
            allowed_departments=allowed_departments,
            conflicts=compiled_context.conflicts,
            chunk_hierarchy=compiled_context.chunk_hierarchy,
            rendered_context=compiled_context.rendered_context,
            workspace_id=request.workspace_id,
            provider_overrides=request.provider_overrides,
        )

    @track(name="generate_from_context", phase="generation")
    async def generate_from_context(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        allowed_departments: list[uuid.UUID] | None = None,
        is_web_context: bool = False,
        conflicts: list[ConflictNotice] | None = None,
        chunk_hierarchy: dict[str, "ChunkHierarchyMetadata"] | None = None,
        rendered_context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        workspace_thread_context: list[dict] | None = None,
        workspace_id: uuid.UUID | None = None,
        llm: LLMProvider | None = None,
        provider_overrides: dict[str, str] | None = None,
    ) -> GenerateResponse:
        
        if is_greeting(query):
            logger.info(
                "conversational_query_detected_in_context_generation",
                extra={
                    "event": (
                        "conversational_query_detected_in_context_generation"
                    ),
                    "query": query,
                },
            )

            return GenerateResponse(
                answer=(
                    "Hello! I'm Doot — a research assistant "
                    "grounded in your document library. "
                    "How can I help you?"
                ),
                citations=[],
                retrieved_chunks=[],
            )

        record_stage_start("generation")
        try:
            # Final context safety check for internal chunks
            if not is_web_context and allowed_departments is not None:
                allowed_dept_strs = {str(d) for d in allowed_departments}
                for chunk in chunks:
                    if chunk.department_id not in allowed_dept_strs and chunk.department_id != "None":
                        logger.error(
                            "context_safety_violation",
                            extra={
                                "event": "context_safety_violation",
                                "chunk_id": chunk.chunk_id,
                                "department_id": chunk.department_id,
                            }
                        )
                        # Defense in depth: mathematically guarantee it cannot reach LLM
                        raise RuntimeError("Security violation: Unauthorized chunk detected in final context.")

            context = rendered_context if rendered_context is not None else format_docs(chunks, chunk_hierarchy=chunk_hierarchy)
            messages = build_rag_messages(
                context=context,
                question=query,
                conflicts=conflicts,
                chunks=chunks,
                conversation_history=conversation_history,
                workspace_thread_context=workspace_thread_context,
            )

            # Resolve active LLM provider for the given Workspace (supporting per-request experiment overrides)
            llm_to_use = llm
            if llm_to_use is None:
                if workspace_id is not None or provider_overrides:
                    from rag_app.providers.resolver import resolve_llm_for_workspace
                    override_llm = provider_overrides.get("llm") if provider_overrides else None
                    llm_to_use = await resolve_llm_for_workspace(
                        workspace_id,
                        self.retrieval_service.session,
                        override_provider=override_llm,
                    )
                else:
                    llm_to_use = self.llm

            # GENERATE GROUNDED ANSWER
            raw_answer, token_usage = await _invoke_llm_with_retry(
                llm_to_use,
                messages,
                stage="generation",
            )

            # VALIDATE AND RENDER CITATIONS
            answer, citations = validate_and_render_citations(
                raw_answer=raw_answer,
                chunks=chunks,
                is_conflict=bool(conflicts),
            )

            logger.info(
                "generation_completed",
                extra={
                    "event": "generation_completed",
                    "citations": len(citations),
                    "token_usage": token_usage,
                    "is_web_context": is_web_context,
                },
            )

            return GenerateResponse(
                answer=answer,
                citations=citations,
                retrieved_chunks=chunks,
            )
        finally:
            record_stage_end("generation")


@track(name="invoke_llm", phase="generation")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=10,
    ),
)
async def _invoke_llm_with_retry(
    llm: LLMProvider,
    messages: list[dict[str, str]],
    stage: str | None = None,
) -> tuple[str, dict]:
    return await llm.generate(messages, stage=stage)