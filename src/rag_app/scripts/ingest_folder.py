import asyncio
import logging
import sys
from pathlib import Path

from rag_app.core.config import get_settings
from rag_app.core.logging import setup_logging
from rag_app.db.session import AsyncSessionLocal
from rag_app.ingestion.service import IngestionService

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


async def main(folder: str) -> None:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    input_files = [
        path for path in folder_path.rglob("*")
        if path.suffix.lower() in {".pdf", ".txt", ".md"}
    ]

    if not input_files:
        logger.warning(
            "no_files_found",
            extra={"event": "no_files_found", "folder": str(folder_path)},
        )
        return

    async with AsyncSessionLocal() as session:
        service = IngestionService(session)

        for file_path in input_files:
            result = await service.ingest_file(file_path)
            logger.info(
                "ingest_result",
                extra={
                    "event": "ingest_result",
                    "file_name": result.file_name,
                    "skipped": result.skipped,
                    "chunks_created": result.chunks_created,
                },
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: uv run python -m rag_app.scripts.ingest_folder sample_docs")

    asyncio.run(main(sys.argv[1]))
