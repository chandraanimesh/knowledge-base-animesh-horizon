from __future__ import annotations
from langsmith import traceable
import asyncio
import hashlib
import logging
import time
import uuid
from pathlib import Path

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.core.config import get_settings
from rag_app.core.tracing import track
from rag_app.db.models import Document, DocumentChunk, DocumentSection
from rag_app.ingestion.chunker import chunk_elements
from rag_app.ingestion.embedding_v1 import get_embedding_provider
from rag_app.ingestion.hash import hash_file
from rag_app.ingestion.hierarchy import HierarchyBuilder, section_nodes_to_db_rows
from rag_app.ingestion.loader import load_document
from rag_app.core.config import settings
from rag_app.schemas.rag import IngestResponse


logger = logging.getLogger(__name__)

def sanitize_nul_bytes(obj):
    if isinstance(obj, str):
        return obj.replace('\x00', '')
    elif isinstance(obj, dict):
        return {sanitize_nul_bytes(k): sanitize_nul_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nul_bytes(i) for i in obj]
    return obj



def get_text_hash(text: str) -> str:
    """
    Create a deterministic SHA-256 hash for chunk content.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

def get_chunk_search_text(chunk) -> str:
    if chunk.metadata and "embedding_text" in chunk.metadata:
        return chunk.metadata["embedding_text"]
    ctx = chunk.metadata.get("search_context") if chunk.metadata else None
    return f"{ctx}\n\n{chunk.text}" if ctx else chunk.text

class IngestionService:

    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session
        self.embedding_provider = (
            get_embedding_provider()
        )

    @track(
        name="ingest_file",
        phase="ingestion",
    )
    @traceable(run_type = "chain")
    async def ingest_file(
        self,
        file_path: str | Path,
        department_id: uuid.UUID | None = None,
        created_by: uuid.UUID | None = None,
        force: bool = False,
        workspace_id: uuid.UUID | None = None,
    ) -> IngestResponse:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        # ======================================================
        # 1. FILE HASH
        # ======================================================

        file_hash = await asyncio.to_thread(
            hash_file,
            path,
        )

        # ======================================================
        # 2. DOCUMENT-LEVEL DEDUPLICATION
        # ======================================================

        if not force:
            existing = await self.session.scalar(
                select(Document).where(
                    Document.content_hash == file_hash
                )
            )

            if existing:

                logger.info(
                    "document_skip_duplicate",
                    extra={
                        "event": "document_skip_duplicate",
                        "file_name": path.name,
                    },
                )

                return IngestResponse(
                    file_name=path.name,
                    skipped=True,
                    document_id=str(existing.id),
                    chunks_created=0,
                    reason=(
                        "same content_hash already exists"
                    ),
                )

        # ======================================================
        # 3. REMOVE OLD VERSION OF SAME FILE
        # ======================================================

        old_doc = await self.session.scalar(
            select(Document).where(
                Document.file_path == str(path)
            )
        )

        if old_doc:

            logger.info(
                "replacing_existing_document",
                extra={
                    "event": "replacing_existing_document",
                    "file_name": path.name,
                    "document_id": str(old_doc.id),
                },
            )

            await self.session.execute(
                delete(Document).where(
                    Document.id == old_doc.id
                )
            )

            await self.session.flush()

        # ======================================================
        # 4. LOAD
        # ======================================================

        logger.info(
            "document_loading",
            extra={
                "event": "document_loading",
                "file_name": path.name,
                "suffix": path.suffix.lower(),
            },
        )

        t0 = time.perf_counter()
        elements = await asyncio.to_thread(
            load_document,
            path,
        )
        pdf_load_ms = round((time.perf_counter() - t0) * 1000, 2)

        if not elements:

            return IngestResponse(
                file_name=path.name,
                skipped=True,
                chunks_created=0,
                reason="no document elements created",
            )

        # ======================================================
        # 5. CHUNK
        # ======================================================

        t0 = time.perf_counter()
        chunks = await asyncio.to_thread(
            chunk_elements,
            elements=elements,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunking_ms = round((time.perf_counter() - t0) * 1000, 2)

        if not chunks:

            return IngestResponse(
                file_name=path.name,
                skipped=True,
                chunks_created=0,
                reason="no text chunks created",
            )

        logger.info(
            "document_chunked",
            extra={
                "event": "document_chunked",
                "file_name": path.name,
                "elements": len(elements),
                "chunks": len(chunks),
            },
        )

        # ======================================================
        # 6. CHUNK HASHING
        # ======================================================

        chunk_hashes = [
            get_text_hash(get_chunk_search_text(chunk))
            for chunk in chunks
        ]

        # ======================================================
        # 7. FIND REUSABLE EMBEDDINGS
        # ======================================================

        existing_chunks = await self.session.execute(
            select(
                DocumentChunk.chunk_hash,
                DocumentChunk.embedding,
            ).where(
                DocumentChunk.chunk_hash.in_(
                    chunk_hashes
                )
            )
        )

        existing_embeddings_map = {
            row.chunk_hash: row.embedding
            for row in existing_chunks
        }

        # ======================================================
        # 8. PREPARE EMBEDDING WORK
        # ======================================================

        texts_to_embed: list[str] = []
        indices_to_embed: list[int] = []

        final_embeddings = [
            None
        ] * len(chunks)

        for index, (chunk, chunk_hash) in enumerate(
            zip(
                chunks,
                chunk_hashes,
                strict=True,
            )
        ):

            existing_embedding = (
                existing_embeddings_map.get(
                    chunk_hash
                )
            )

            if existing_embedding is not None:

                final_embeddings[index] = (
                    existing_embedding
                )

            else:

                texts_to_embed.append(
                    get_chunk_search_text(chunk)
                )

                indices_to_embed.append(
                    index
                )

        # ======================================================
        # 9. BATCH EMBEDDING
        # ======================================================

        embedding_ms = 0.0
        if texts_to_embed:
            t0 = time.perf_counter()

            batch_size = 100

            new_embeddings = []

            for start in range(
                0,
                len(texts_to_embed),
                batch_size,
            ):

                batch = texts_to_embed[
                    start:start + batch_size
                ]

                batch_embeddings = (
                    await self.embedding_provider.embed_texts(
                        batch
                    )
                )

                new_embeddings.extend(
                    batch_embeddings
                )

            for index, embedding in zip(
                indices_to_embed,
                new_embeddings,
                strict=True,
            ):

                final_embeddings[index] = (
                    embedding
                )
            
            embedding_ms = round((time.perf_counter() - t0) * 1000, 2)

        # ======================================================
        # 10. CREATE DOCUMENT
        # ======================================================

        document = Document(
            file_name=path.name,
            file_path=str(path),
            content_hash=file_hash,
            mime_type=self._get_mime_type(
                path
            ),
            department_id=department_id,
            created_by=created_by,
            workspace_id=workspace_id,
        )

        self.session.add(document)

        await self.session.flush()

        # ======================================================
        # 11. BULK CHUNK INSERT
        # ======================================================

        chunk_data = []

        for chunk, embedding, chunk_hash in zip(
            chunks,
            final_embeddings,
            chunk_hashes,
            strict=True,
        ):

            if embedding is None:
                raise RuntimeError(
                    "Missing embedding for chunk"
                )

            metadata = dict(
                chunk.metadata
            )
            metadata = sanitize_nul_bytes(metadata)

            chunk_data.append(
                {
                    "document_id": document.id,
                    "chunk_index": chunk.chunk_index,
                    "text": sanitize_nul_bytes(chunk.text),
                    "embedding": embedding,
                    "chunk_hash": chunk_hash,
                    "page_number": metadata.get(
                        "page_number"
                    ),
                    "metadata_": metadata,
                }
            )

        if chunk_data:

            await self.session.execute(
                insert(DocumentChunk),
                chunk_data,
            )

        # ======================================================
        # 11B. BUILD DOCUMENT HIERARCHY (OV-1)
        # ======================================================

        nodes_created = 0
        chunks_linked = 0
        try:
            builder = HierarchyBuilder(
                document_id=document.id,
                file_name=path.stem,
            )
            result = builder.build(chunks)

            # Persist DocumentSection rows
            section_rows = section_nodes_to_db_rows(result.all_nodes)
            if section_rows:
                await self.session.execute(
                    insert(DocumentSection),
                    section_rows,
                )
                await self.session.flush()
                nodes_created = len(section_rows)

            # Link chunks to their section nodes
            # We need DB chunk IDs — fetch them by (document_id, chunk_index)
            if result.chunk_to_node:
                fetched = (await self.session.execute(
                    select(DocumentChunk.id, DocumentChunk.chunk_index).where(
                        DocumentChunk.document_id == document.id
                    )
                )).all()
                db_chunk_map = {row[1]: row[0] for row in fetched}

                for chunk_index, section_node in result.chunk_to_node.items():
                    db_chunk_id = db_chunk_map.get(chunk_index)
                    if db_chunk_id is None:
                        continue
                    await self.session.execute(
                        update(DocumentChunk)
                        .where(DocumentChunk.id == db_chunk_id)
                        .values(node_id=section_node.id)
                    )
                    chunks_linked += 1

            logger.info(
                "document_hierarchy_built",
                extra={
                    "event": "document_hierarchy_built",
                    "file_name": path.name,
                    "nodes_created": nodes_created,
                    "chunks_linked": chunks_linked,
                },
            )

        except Exception as exc:
            # Hierarchy building must never abort a successful ingestion.
            logger.warning(
                "document_hierarchy_build_failed",
                extra={
                    "event": "document_hierarchy_build_failed",
                    "file_name": path.name,
                    "error": str(exc),
                },
            )

        # ======================================================
        # 12. COMMIT
        # ======================================================

        t0 = time.perf_counter()
        await self.session.commit()
        db_insert_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "document_ingested",
            extra={
                "event": "document_ingested",
                "file_name": path.name,
                "total_elements": len(elements),
                "total_chunks": len(chunks),
                "new_embeddings_generated": len(
                    texts_to_embed
                ),
                "reused_embeddings": (
                    len(chunks) - len(texts_to_embed)
                ),
                "pdf_load_ms": pdf_load_ms,
                "chunking_ms": chunking_ms,
                "embedding_ms": embedding_ms,
                "db_insert_ms": db_insert_ms,
            },
        )

        from rag_app.retrieval.bm25 import GlobalBM25Index
        bm25 = GlobalBM25Index.get_instance()
        await bm25.build(self.session)

        return IngestResponse(
            file_name=path.name,
            skipped=False,
            document_id=str(document.id),
            chunks_created=len(chunks),
        )

    async def ingest_files(
        self,
        file_paths: list[str | Path],
        force: bool = False,
    ) -> list[IngestResponse]:
        results: list[IngestResponse] = []

        for file_path in file_paths:
            result = await self.ingest_file(
                file_path,
                force=force,
            )

            results.append(result)

        from rag_app.retrieval.bm25 import GlobalBM25Index
        bm25 = GlobalBM25Index.get_instance()
        await bm25.build(self.session)

        return results
        
    @staticmethod
    def _get_mime_type(
        path: Path,
    ) -> str:

        mime_types = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".xlsx": (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            ".xls": "application/vnd.ms-excel",
        }

        return mime_types.get(
            path.suffix.lower(),
            "application/octet-stream",
        )