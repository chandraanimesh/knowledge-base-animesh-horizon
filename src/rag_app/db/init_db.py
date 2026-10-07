import asyncio
import logging

from sqlalchemy import text

from rag_app.core.config import get_settings
from rag_app.core.logging import setup_logging
from rag_app.db.models import Base
from rag_app.db.session import engine

logger = logging.getLogger(__name__)


async def init_db() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )
        )

        await conn.run_sync(
            Base.metadata.create_all
        )

        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                ix_document_chunks_embedding_hnsw
                ON document_chunks
                USING hnsw (
                    embedding vector_cosine_ops
                )
                """
            )
        )

        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                ix_document_chunks_text_fts
                ON document_chunks
                USING gin (
                    to_tsvector('simple', text)
                )
                """
            )
        )

        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                ix_document_sections_fts
                ON document_sections
                USING gin (
                    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(l0_abstract, ''))
                )
                """
            )
        )

        await conn.execute(
            text(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS visualization JSONB"
            )
        )

    logger.info(
        "db_initialized",
        extra={"event": "db_initialized"},
    )

if __name__ == "__main__":
    asyncio.run(init_db())
