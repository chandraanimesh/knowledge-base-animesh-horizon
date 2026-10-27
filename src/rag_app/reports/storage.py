"""Secure storage service for generated RAG artifacts."""

import hashlib
import os
from pathlib import Path
from typing import BinaryIO
import uuid

from rag_app.core.config import settings


class ArtifactStorageService:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.artifact_storage_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_or_absolute_path: str) -> Path:
        target = Path(relative_or_absolute_path)
        if not target.is_absolute():
            target = (self.base_dir / target).resolve()
        else:
            target = target.resolve()

        # Strict containment check against path traversal
        if not str(target).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected: {relative_or_absolute_path}")
        return target

    def save_artifact(
        self,
        artifact_id: uuid.UUID,
        format_ext: str,
        content_bytes: bytes,
    ) -> tuple[str, str, int]:
        """
        Saves artifact bytes to a server-controlled path.
        Returns:
            (relative_storage_path, sha256_checksum, byte_size)
        """
        if len(content_bytes) > settings.artifact_max_bytes:
            raise ValueError(
                f"Artifact size {len(content_bytes)} exceeds maximum configured limit of {settings.artifact_max_bytes} bytes"
            )

        clean_ext = format_ext.lower().lstrip(".")
        filename = f"{artifact_id}.{clean_ext}"
        target_path = self.base_dir / filename

        # Compute SHA-256
        sha256 = hashlib.sha256(content_bytes).hexdigest()
        byte_size = len(content_bytes)

        # Write safely
        with open(target_path, "wb") as f:
            f.write(content_bytes)

        return filename, sha256, byte_size

    def read_artifact(self, storage_path: str) -> bytes:
        """Reads artifact bytes verifying path containment."""
        safe_path = self._resolve_safe_path(storage_path)
        if not safe_path.exists() or not safe_path.is_file():
            raise FileNotFoundError(f"Artifact file not found at {storage_path}")
        return safe_path.read_bytes()

    def get_absolute_path(self, storage_path: str) -> Path:
        """Gets safe verified absolute path for streaming file responses."""
        safe_path = self._resolve_safe_path(storage_path)
        if not safe_path.exists() or not safe_path.is_file():
            raise FileNotFoundError(f"Artifact file not found at {storage_path}")
        return safe_path

    def delete_artifact(self, storage_path: str) -> bool:
        """Safely removes an artifact file from disk."""
        try:
            safe_path = self._resolve_safe_path(storage_path)
            if safe_path.exists():
                safe_path.unlink()
                return True
        except Exception:
            return False
        return False
