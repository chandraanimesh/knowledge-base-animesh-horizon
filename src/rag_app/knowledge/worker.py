import logging
import uuid
import time
from sqlalchemy import select

from rag_app.db.session import AsyncSessionLocal
from rag_app.db.models import Document, DocumentChunk
from rag_app.knowledge.extractor import KnowledgeExtractorService
from rag_app.knowledge.materializer import materialize_knowledge

logger = logging.getLogger(__name__)

async def run_document_knowledge_extraction(document_id: uuid.UUID) -> None:
    """
    Background worker job to extract knowledge from a committed Document.
    
    LIMITATION WARNING: Currently executed via FastAPI BackgroundTasks (asyncio/memory). 
    This is NOT a durable queue (like Celery/Redis). If the application server restarts 
    or crashes during extraction, the job is lost and will not be automatically retried.
    For production, this should be replaced with a robust message broker.
    """
    start_time = time.perf_counter()
    logger.info(
        "job_start",
        extra={
            "event": "knowledge_extraction_job_start",
            "document_id": str(document_id)
        }
    )
    
    async with AsyncSessionLocal() as session:
        # Load the document
        doc = await session.scalar(select(Document).where(Document.id == document_id))
        if not doc:
            logger.error(
                "job_failure",
                extra={
                    "event": "knowledge_extraction_job_failure",
                    "document_id": str(document_id),
                    "error": "Document not found"
                }
            )
            return
            
        # Load the chunks
        result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index)
        )
        chunks = result.scalars().all()
        
        if not chunks:
            logger.info(f"No chunks found for document {document_id}. Skipping extraction.")
            return
            
        extractor = KnowledgeExtractorService()
        
        success_count = 0
        failure_count = 0
        candidate_count = 0
        
        for chunk in chunks:
            try:
                # 1. Extract candidates for the chunk (transient failures handled inside extractor)
                candidates = await extractor.extract_chunk(chunk, doc)
                
                # 2. Materialize candidates. We commit after each chunk so partial failures 
                # do not destroy successful materializations of previous chunks.
                if candidates:
                    for candidate in candidates:
                        await materialize_knowledge(session, candidate)
                        candidate_count += 1
                    
                    await session.commit()
                    
                success_count += 1
            except Exception as e:
                # Permanent materialization or DB failure for this specific chunk
                await session.rollback()
                logger.error(
                    "chunk_failure",
                    extra={
                        "event": "knowledge_extraction_chunk_failure",
                        "document_id": str(document_id),
                        "chunk_id": str(chunk.id),
                        "error": str(e)
                    }
                )
                failure_count += 1
                
        duration = round(time.perf_counter() - start_time, 2)
        logger.info(
            "job_complete",
            extra={
                "event": "knowledge_extraction_job_complete",
                "document_id": str(document_id),
                "total_chunks": len(chunks),
                "success_count": success_count,
                "failure_count": failure_count,
                "candidate_count": candidate_count,
                "duration_seconds": duration
            }
        )
