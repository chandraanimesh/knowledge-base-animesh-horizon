"""DOCX Embedded Media Extractor

Safely inspects and extracts embedded images, diagrams, and comic strips from DOCX
packages (word/media/) with zip-slip and zip-bomb protection.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".svg"}
MAX_EXTRACTED_MEDIA_FILES = 50
MAX_UNCOMPRESSED_MEDIA_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_COMPRESSION_RATIO = 100.0


@dataclass
class DocxMediaItem:
    file_name: str
    media_path: str
    format: str
    byte_size: int
    sha256: str
    data: bytes
    width: int | None = None
    height: int | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_representation(self) -> str:
        """Formatted text context to inject into document chunking."""
        dims = f" ({self.width}x{self.height})" if self.width and self.height else ""
        desc = f": {self.caption}" if self.caption else ""
        return f"[Embedded Image: {self.file_name}{dims}{desc}]"


def _is_safe_zip_entry(name: str) -> bool:
    """Guards against directory traversal (zip-slip) attacks."""
    if not name:
        return False
    norm = name.replace("\\", "/")
    # Reject absolute paths or paths with parent traversal
    if norm.startswith("/") or ".." in norm.split("/"):
        return False
    return True


def extract_docx_media(
    source: Path | str | bytes,
    max_files: int = MAX_EXTRACTED_MEDIA_FILES,
    max_total_bytes: int = MAX_UNCOMPRESSED_MEDIA_BYTES,
) -> list[DocxMediaItem]:
    """
    Inspects DOCX package and safely extracts embedded images from word/media/.
    Enforces security guards:
      - Path traversal / directory slip rejection
      - File count limit
      - Accumulated uncompressed size cap
      - Compression ratio cap (zip-bomb guard)
    """
    media_items: list[DocxMediaItem] = []

    try:
        if isinstance(source, bytes):
            zip_target = io.BytesIO(source)
        else:
            p = Path(source)
            if not p.exists() or not p.is_file():
                return []
            zip_target = p

        if not zipfile.is_zipfile(zip_target):
            return []

        with zipfile.ZipFile(zip_target, "r") as zf:
            total_uncompressed = 0
            image_members: list[zipfile.ZipInfo] = []

            for info in zf.infolist():
                norm_name = info.filename.replace("\\", "/")

                # Only examine word/media/
                if not norm_name.lower().startswith("word/media/"):
                    continue

                ext = os.path.splitext(norm_name)[1].lower()
                if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue

                # Security check 1: Directory traversal
                if not _is_safe_zip_entry(info.filename):
                    logger.warning(
                        "suspicious_docx_zip_entry_skipped",
                        extra={"event": "suspicious_docx_zip_entry_skipped", "entry": info.filename},
                    )
                    continue

                # Security check 2: Zip bomb ratio check
                if info.compress_size > 0 and (info.file_size / info.compress_size) > MAX_COMPRESSION_RATIO:
                    logger.warning(
                        "docx_media_compression_ratio_exceeded",
                        extra={"event": "docx_media_compression_ratio_exceeded", "entry": info.filename},
                    )
                    continue

                total_uncompressed += info.file_size
                # Security check 3: Accumulated bytes cap
                if total_uncompressed > max_total_bytes:
                    logger.warning(
                        "docx_media_total_bytes_cap_exceeded",
                        extra={"event": "docx_media_total_bytes_cap_exceeded", "limit": max_total_bytes},
                    )
                    break

                image_members.append(info)
                # Security check 4: File count cap
                if len(image_members) >= max_files:
                    logger.warning(
                        "docx_media_max_files_cap_reached",
                        extra={"event": "docx_media_max_files_cap_reached", "limit": max_files},
                    )
                    break

            # Safely extract image payloads and metadata
            for info in image_members:
                try:
                    data = zf.read(info)
                    base_name = os.path.basename(info.filename.replace("\\", "/"))
                    ext = os.path.splitext(base_name)[1].lower().lstrip(".")
                    sha = hashlib.sha256(data).hexdigest()

                    width, height = None, None
                    try:
                        from PIL import Image
                        with Image.open(io.BytesIO(data)) as img:
                            width, height = img.size
                    except Exception:
                        pass

                    media_items.append(
                        DocxMediaItem(
                            file_name=base_name,
                            media_path=info.filename,
                            format=ext,
                            byte_size=len(data),
                            sha256=sha,
                            data=data,
                            width=width,
                            height=height,
                            caption=None,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "failed_to_extract_single_docx_image",
                        extra={"event": "failed_to_extract_single_docx_image", "entry": info.filename, "error": str(exc)},
                    )

    except Exception as exc:
        logger.error(f"Error inspecting DOCX media: {exc}", exc_info=True)

    return media_items

