"""
OV-1 Backfill Script — scripts/backfill_hierarchy.py

Rebuilds DocumentSection trees for existing documents that have no
section nodes yet (i.e., documents ingested before OV-1 migration).

Usage:
    uv run python scripts/backfill_hierarchy.py [--dry-run]

Options:
    --dry-run   Print what would be done without writing to DB.
    --doc-id    Backfill a specific document UUID only.

Safety:
- Only processes documents with zero existing DocumentSection rows.
- Does NOT delete or re-embed any DocumentChunk.
- Re-runnable (idempotent): already-processed documents are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# Add src to sys.path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import Document, DocumentChunk, DocumentSection
from rag_app.ingestion.hierarchy import HierarchyBuilder, section_nodes_to_db_rows
from rag_app.ingestion.chunker import TextChunk

settings = get_settings()


async def _backfill_document(
    session: AsyncSession,
    doc: Document,
    dry_run: bool,
) -> dict:
    """Backfill hierarchy for a single document. Returns stats dict."""

    # Load all chunks for this document
    chunk_rows = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index)
    )
    db_chunks = chunk_rows.scalars().all()

    if not db_chunks:
        return {"doc_id": str(doc.id), "status": "skipped", "reason": "no_chunks"}

    # Convert DB chunks to TextChunk objects (for HierarchyBuilder)
    text_chunks = [
        TextChunk(
            text=c.text,
            chunk_index=c.chunk_index,
            metadata=c.metadata_ or {},
        )
        for c in db_chunks
    ]

    builder = HierarchyBuilder(document_id=doc.id, file_name=Path(doc.file_name).stem)
    result = builder.build(text_chunks)

    if dry_run:
        print(f"  [DRY RUN] doc={doc.file_name} -> {len(result.all_nodes)} nodes, "
              f"{len(result.chunk_to_node)} chunks would be linked")
        return {
            "doc_id": str(doc.id),
            "status": "dry_run",
            "nodes": len(result.all_nodes),
            "chunks": len(result.chunk_to_node),
        }

    # Persist section rows
    section_rows = section_nodes_to_db_rows(result.all_nodes)
    await session.execute(insert(DocumentSection), section_rows)
    await session.flush()

    # Link chunks to nodes
    chunks_linked = 0
    db_chunk_map = {c.chunk_index: c.id for c in db_chunks}
    for chunk_index, section_node in result.chunk_to_node.items():
        chunk_db_id = db_chunk_map.get(chunk_index)
        if chunk_db_id is None:
            continue
        await session.execute(
            update(DocumentChunk)
            .where(DocumentChunk.id == chunk_db_id)
            .values(node_id=section_node.id)
        )
        chunks_linked += 1

    await session.commit()
    print(f"  doc={doc.file_name} -> {len(result.all_nodes)} nodes created, "
          f"{chunks_linked}/{len(db_chunks)} chunks linked")

    return {
        "doc_id": str(doc.id),
        "status": "backfilled",
        "nodes": len(result.all_nodes),
        "chunks": chunks_linked,
    }


async def run(dry_run: bool = False, doc_id: str | None = None) -> None:
    engine = create_async_engine(settings.database_url, echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with SessionLocal() as session:
        if doc_id:
            # Single document mode
            target_id = uuid.UUID(doc_id)
            doc = await session.get(Document, target_id)
            if doc is None:
                print(f"Document {doc_id} not found.")
                return
            # Check if already has sections
            existing_count = await session.scalar(
                select(DocumentSection)
                .where(DocumentSection.document_id == target_id)
                .limit(1)
            )
            if existing_count is not None:
                print(f"Document {doc.file_name} already has sections. Skipping.")
                return
            stats = await _backfill_document(session, doc, dry_run)
            print(stats)
        else:
            # All documents without sections
            docs_result = await session.execute(
                select(Document)
                .where(
                    ~Document.id.in_(
                        select(DocumentSection.document_id).distinct()
                    )
                )
                .order_by(Document.created_at)
            )
            docs = docs_result.scalars().all()

            print(f"Found {len(docs)} document(s) without hierarchy nodes.")
            total_nodes = 0
            total_chunks = 0

            for doc in docs:
                stats = await _backfill_document(session, doc, dry_run)
                if stats.get("status") in ("backfilled", "dry_run"):
                    total_nodes += stats.get("nodes", 0)
                    total_chunks += stats.get("chunks", 0)

            print(f"\nBackfill complete: {total_nodes} nodes, {total_chunks} chunks linked.")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OV-1 hierarchy backfill")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--doc-id", default=None, help="Backfill specific document UUID")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run, doc_id=args.doc_id))
