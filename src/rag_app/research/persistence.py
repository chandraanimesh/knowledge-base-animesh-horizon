import logging
import uuid
import json
import hashlib
from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from rag_app.db.models import Department, ResearchKnowledge, ResearchEvidence, Document, DocumentChunk
from rag_app.schemas.research import StructuredKnowledgeCandidate, ExtractedFact

logger = logging.getLogger(__name__)

async def persist_knowledge(
    session: AsyncSession,
    research_run_id: str | None = None,
    candidate: StructuredKnowledgeCandidate | None = None,
    run_id_str: str | None = None,
) -> list[ResearchKnowledge]:
    """
    Persists validated structured knowledge to the database.
    """
    effective_run_id = research_run_id or run_id_str
    if not candidate or not candidate.facts or not effective_run_id:
        return []
        
    persisted_records = []
    
    # Pre-fetch departments to map code -> id
    dept_result = await session.execute(select(Department))
    departments = {d.code: d for d in dept_result.scalars().all()}
    
    for fact in candidate.facts:
        dept = departments.get(fact.department_code)
        dept_id = dept.id if dept else None
        
        # 1. Create Knowledge
        knowledge = ResearchKnowledge(
            research_run_id=uuid.UUID(effective_run_id),
            department_id=dept_id,
            entity_name=fact.entity_name,
            entity_type=fact.entity_type,
            attributes=fact.attributes
        )
        
        # 2. Add to session (we try to persist and handle duplicates)
        session.add(knowledge)
        try:
            await session.flush()
        except IntegrityError:
            # Duplicate knowledge for this run
            await session.rollback()
            logger.warning(f"Duplicate knowledge ignored: {fact.entity_name} ({fact.entity_type})")
            continue
            
        # 3. Create Evidence
        evidence = ResearchEvidence(
            knowledge_id=knowledge.id,
            source_url=fact.source_url,
            evidence_text=fact.evidence_text
        )
        session.add(evidence)
        await session.flush()
        
        persisted_records.append(knowledge)
        
        # Phase A Guard: Do not silently promote unverified external web results to approved shared knowledge
        # await adapt_to_canonical(session, knowledge, evidence)
        
    return persisted_records

from rag_app.schemas.rag import RetrievedChunk
from rag_app.schemas.research import CleanWebPage

def create_transient_web_chunks(
    candidate: StructuredKnowledgeCandidate | None = None,
    clean_pages: list[CleanWebPage] | None = None,
    research_run_id: str | None = None,
    run_id_str: str | None = None,
    max_chunks_per_page: int = 3,
    chunk_char_size: int = 800,
) -> list[RetrievedChunk]:
    """
    Phase A: Transform clean web pages and extracted facts into ephemeral RetrievedChunk
    evidence objects for the ContextCompiler and GenerationService, WITHOUT persisting
    them into internal Document or DocumentChunk database tables.
    """
    web_chunks: list[RetrievedChunk] = []
    seen_texts: set[str] = set()
    effective_run_id = research_run_id or run_id_str
    run_id_short = effective_run_id[:8] if effective_run_id else "web"
    clean_pages = clean_pages or []

    # 1. Chunks from extracted facts (structured entity-attribute evidence)
    if candidate and candidate.facts:
        for idx, fact in enumerate(candidate.facts):
            attr_lines = [f"{k}: {v}" for k, v in (fact.attributes or {}).items()]
            attr_block = "\n".join(attr_lines)
            content_parts = [
                f"Web Source: {fact.source_url}",
                f"Entity: {fact.entity_name} ({fact.entity_type})",
            ]
            if attr_block:
                content_parts.append(f"Attributes:\n{attr_block}")
            if fact.evidence_text:
                content_parts.append(f"Supporting Evidence: {fact.evidence_text}")
            content = "\n".join(content_parts).strip()

            norm_content = " ".join(content.lower().split())
            if norm_content in seen_texts:
                continue
            seen_texts.add(norm_content)

            chunk = RetrievedChunk(
                chunk_id=f"web_{run_id_short}_fact_{idx}",
                document_id=fact.source_url,
                department_id="None",
                file_name=fact.source_url,
                chunk_index=idx,
                text=content,
                hybrid_score=1.0,
            )
            web_chunks.append(chunk)

    # 2. Chunks from clean web page content (comprehensive text excerpts)
    for p_idx, page in enumerate(clean_pages):
        raw_text = (page.content or "").strip()
        if not raw_text:
            continue

        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        current_chunk: list[str] = []
        current_len = 0
        page_chunk_idx = 0

        for para in paragraphs:
            if current_len + len(para) > chunk_char_size and current_chunk:
                chunk_body = "\n\n".join(current_chunk)
                norm_body = " ".join(chunk_body.lower().split())
                if norm_body not in seen_texts:
                    seen_texts.add(norm_body)
                    web_chunks.append(
                        RetrievedChunk(
                            chunk_id=f"web_{run_id_short}_{p_idx}_{page_chunk_idx}",
                            document_id=page.url,
                            department_id="None",
                            file_name=page.title or page.url,
                            chunk_index=page_chunk_idx,
                            text=chunk_body,
                            hybrid_score=0.9,
                        )
                    )
                    page_chunk_idx += 1
                    if page_chunk_idx >= max_chunks_per_page:
                        break
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para)

        if current_chunk and page_chunk_idx < max_chunks_per_page:
            chunk_body = "\n\n".join(current_chunk)
            norm_body = " ".join(chunk_body.lower().split())
            if norm_body not in seen_texts:
                seen_texts.add(norm_body)
                web_chunks.append(
                    RetrievedChunk(
                        chunk_id=f"web_{run_id_short}_{p_idx}_{page_chunk_idx}",
                        document_id=page.url,
                        department_id="None",
                        file_name=page.title or page.url,
                        chunk_index=page_chunk_idx,
                        text=chunk_body,
                        hybrid_score=0.9,
                    )
                )

    return web_chunks

async def adapt_to_canonical(session: AsyncSession, rk: ResearchKnowledge, evidence: ResearchEvidence):
    """
    Adapter pattern to bridge legacy extraction output to the canonical Knowledge layer.
    """
    from rag_app.knowledge.materializer import materialize_knowledge, MaterializationCandidate
    
    candidate = MaterializationCandidate(
        kind="ENTITY",
        entity_type=rk.entity_type,
        entity_name=rk.entity_name,
        attributes=rk.attributes,
        department_id=rk.department_id,
        confidence=0.8, # Default for legacy web research
        research_evidence_id=evidence.id,
        extraction_run_id=rk.research_run_id,
        extraction_method="web_research"
    )
    await materialize_knowledge(session, candidate)

import hashlib
from rag_app.core.config import get_settings
from rag_app.ingestion.embedding_v1 import get_embedding_provider
from rag_app.ingestion.element import DocumentElement
from rag_app.ingestion.chunker import chunk_elements
from rag_app.ingestion.service_v1 import get_text_hash, sanitize_nul_bytes
from sqlalchemy import insert

settings = get_settings()

async def ingest_knowledge_to_rag(
    session: AsyncSession,
    research_run_id: str,
    knowledge_records: list[ResearchKnowledge]
) -> Document | None:
    """
    Ingests validated ResearchKnowledge into the existing RAG vector database.
    Reuses existing Document and DocumentChunk tables.
    """
    if not knowledge_records:
        return None
        
    # Group by department to create a document for each department?
    # Or just create one document per run.
    # The Document table has a single department_id. So if knowledge records have different departments,
    # we need separate Documents.
    
    docs_created = []
    
    # Group by department
    dept_groups = {}
    for k in knowledge_records:
        dept_groups.setdefault(k.department_id, []).append(k)
        
    embedding_provider = get_embedding_provider()
    
    for dept_id, records in dept_groups.items():
        # Build text representation
        lines = [f"# Research Knowledge for Run {research_run_id}"]
        for r in records:
            lines.append(f"## {r.entity_name} ({r.entity_type})")
            lines.append(f"Department: {r.department_id}")
            lines.append("Attributes:")
            for k, v in r.attributes.items():
                lines.append(f"- {k}: {v}")
            # Get evidence URLs
            urls = {e.source_url for e in r.evidence}
            lines.append(f"Sources: {', '.join(urls)}")
            lines.append("")
            
        full_text = "\n".join(lines)
        content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        
        # Check if already exists
        existing = await session.scalar(select(Document).where(Document.content_hash == content_hash))
        if existing:
            docs_created.append(existing)
            continue
            
        # Create Document
        doc = Document(
            file_name=f"research_{research_run_id}_{dept_id}.md",
            file_path=f"research://{research_run_id}/{dept_id}",
            content_hash=content_hash,
            mime_type="text/markdown",
            department_id=dept_id,
            created_by=None,
            status="READY"
        )
        session.add(doc)
        await session.flush()
        docs_created.append(doc)
        
        # Chunk
        element = DocumentElement(
            text=full_text,
            source_path=doc.file_path,
            file_name=doc.file_name,
            file_type="markdown",
            element_type="text",
            metadata={
                "source_type": "web_research",
                "research_run_id": str(research_run_id)
            }
        )
        
        chunks = chunk_elements([element], chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        
        if not chunks:
            continue
            
        # Embed
        texts_to_embed = [c.text for c in chunks]
        embeddings = await embedding_provider.embed_texts(texts_to_embed)
        
        # Insert Chunks
        chunk_data = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = dict(chunk.metadata)
            metadata = sanitize_nul_bytes(metadata)
            
            c_hash = get_text_hash(chunk.text)
            
            chunk_data.append({
                "document_id": doc.id,
                "chunk_index": chunk.chunk_index,
                "text": sanitize_nul_bytes(chunk.text),
                "embedding": embedding,
                "chunk_hash": c_hash,
                "page_number": None,
                "metadata_": metadata
            })
            
        if chunk_data:
            await session.execute(insert(DocumentChunk), chunk_data)
            await session.flush()
            
    # Update BM25
    from rag_app.retrieval.bm25 import GlobalBM25Index
    bm25 = GlobalBM25Index.get_instance()
    await bm25.build(session)
    
    return docs_created[0] if docs_created else None
