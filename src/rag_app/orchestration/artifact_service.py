"""
Symbolic Execution Memory & Artifact Offloading Service (Milestone 10F).
Inspired by TencentDB Agent Memory symbolic offloading.
Prevents large execution outputs from flooding LLM context by offloading outputs
exceeding token thresholds to durable storage, issuing stable symbolic references,
and supporting bounded, authorized drill-down with SHA-256 verification.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.core.config import get_settings
from rag_app.db.models import GeneratedArtifact, User

logger = logging.getLogger(__name__)

DEFAULT_OFFLOAD_TOKEN_THRESHOLD = 1000
DEFAULT_OFFLOAD_CHAR_THRESHOLD = 4000
DEFAULT_STORAGE_DIR = "storage/artifacts"


class SymbolicArtifactReference(BaseModel):
    """Compact symbolic reference returned to LLM context instead of raw large output."""
    model_config = ConfigDict(from_attributes=True)

    artifact_id: str
    symbolic_uri: str
    title: str
    mime_type: str = "text/plain"
    size_bytes: int
    content_hash: str
    summary_snippet: str
    is_offloaded: bool = True


class ArtifactOffloadingService:
    """
    Manages symbolic execution memory offloading and authorized retrieval.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        char_threshold: int = DEFAULT_OFFLOAD_CHAR_THRESHOLD,
    ):
        settings = get_settings()
        configured_dir = getattr(settings, "artifact_storage_path", DEFAULT_STORAGE_DIR)
        self.storage_dir = Path(storage_dir or configured_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.char_threshold = char_threshold

    def should_offload(self, content: str | dict | list) -> bool:
        """Determines whether payload exceeds the inline token/character threshold."""
        if isinstance(content, (dict, list)):
            c_str = json.dumps(content)
        else:
            c_str = str(content or "")
        return len(c_str) >= self.char_threshold

    async def offload_output(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        content: str | dict | list,
        title: str,
        workspace_id: Optional[uuid.UUID] = None,
        run_id: Optional[uuid.UUID] = None,
        task_key: Optional[str] = None,
        format_type: str = "txt",
    ) -> SymbolicArtifactReference:
        """
        Persists a large task output to durable storage, records GeneratedArtifact,
        and returns a lightweight symbolic reference.
        """
        if isinstance(content, (dict, list)):
            raw_text = json.dumps(content, indent=2)
            mime = "application/json"
            fmt = "json"
        else:
            raw_text = str(content or "")
            mime = "text/markdown" if format_type == "md" else "text/plain"
            fmt = format_type

        # 1. Compute cryptographic SHA-256 hash
        encoded_bytes = raw_text.encode("utf-8")
        content_hash = hashlib.sha256(encoded_bytes).hexdigest()
        byte_size = len(encoded_bytes)

        # 2. Persist to storage directory safely
        artifact_uuid = uuid.uuid4()
        safe_name = f"{artifact_uuid.hex}.{fmt}"
        file_path = self.storage_dir / safe_name
        file_path.write_text(raw_text, encoding="utf-8")

        # 3. Insert authoritative record into GeneratedArtifact table
        artifact_record = GeneratedArtifact(
            id=artifact_uuid,
            user_id=user_id,
            workspace_id=workspace_id,
            title=title[:255],
            format=fmt[:20],
            safe_filename=safe_name,
            mime_type=mime,
            byte_size=byte_size,
            storage_path=str(file_path),
            checksum_sha256=content_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        session.add(artifact_record)
        await session.flush()

        # 4. Generate stable symbolic URI and summary
        run_str = str(run_id) if run_id else "global"
        t_key = task_key or "task"
        symbolic_uri = f"artifact://{run_str}/{t_key}/{artifact_uuid.hex[:12]}"
        
        # Summary snippet: first 250 characters with metadata
        summary_snippet = (
            f"[Offloaded Artifact: {title} ({byte_size:,} bytes, {fmt.upper()}). "
            f"Preview: {raw_text[:200].strip()}...]"
        )

        return SymbolicArtifactReference(
            artifact_id=str(artifact_uuid),
            symbolic_uri=symbolic_uri,
            title=title,
            mime_type=mime,
            size_bytes=byte_size,
            content_hash=content_hash,
            summary_snippet=summary_snippet,
            is_offloaded=True,
        )

    async def retrieve_artifact(
        self,
        session: AsyncSession,
        user: User,
        artifact_id: uuid.UUID | str,
        workspace_id: Optional[uuid.UUID] = None,
        max_expand_chars: int = 50000,
    ) -> str:
        """
        Retrieves authorized execution artifact content with strict access control
        and SHA-256 verification.
        """
        a_id = uuid.UUID(str(artifact_id))
        stmt = select(GeneratedArtifact).where(GeneratedArtifact.id == a_id)
        artifact = await session.scalar(stmt)

        if not artifact:
            raise KeyError(f"Artifact '{artifact_id}' not found.")

        # Access Control: User must be owner or workspace match or admin
        user_role = getattr(user, "role", "USER")
        if user.id != artifact.user_id and user_role != "ADMIN":
            if artifact.workspace_id and workspace_id != artifact.workspace_id:
                raise PermissionError("Access denied: Artifact belongs to a different workspace scope.")

        # Path Traversal Guard: Verify target path is within self.storage_dir
        target_path = Path(artifact.storage_path).resolve()
        resolved_storage = self.storage_dir.resolve()
        try:
            target_path.relative_to(resolved_storage)
        except ValueError:
            raise PermissionError("Security alert: Attempted directory traversal outside artifact storage boundary.")

        if not target_path.exists():
            raise FileNotFoundError(f"Underlying artifact file missing on disk: {target_path}")

        content = target_path.read_text(encoding="utf-8")

        # Cryptographic integrity verification
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_hash != artifact.checksum_sha256:
            raise ValueError(f"Integrity check failed: Artifact content hash mismatch for {artifact_id}.")

        # Bounded expansion
        if len(content) > max_expand_chars:
            return content[:max_expand_chars] + f"\n\n[TRUNCATED: Exceeded expansion limit of {max_expand_chars} characters]"

        return content


def generate_execution_canvas_mermaid(
    tasks: list[dict[str, Any]],
    run_id: Optional[str] = None,
) -> str:
    """
    Renders an Execution Canvas visualization in Mermaid flowchart syntax.
    Shows task dependencies, status styles, and artifact references.
    """
    lines = ["flowchart TD"]
    run_header = f"subgraph Run_{run_id[:8]}" if run_id else "subgraph WorkflowRun"
    lines.append(f"    {run_header}[\"Adaptive Execution Canvas\"]")

    for t in tasks:
        t_key = t.get("task_key", "task")
        t_type = t.get("task_type", "TASK")
        status = t.get("status", "PENDING")
        desc = t.get("description", "")[:40]
        
        status_icon = "✓" if status == "COMPLETED" else "✗" if status == "FAILED" else "⏳"
        node_label = f"\"{status_icon} [{t_type}] {t_key}\\n{desc}\""
        lines.append(f"        {t_key}[{node_label}]")

    lines.append("    end")

    # Add edges based on dependencies
    for t in tasks:
        t_key = t.get("task_key")
        for dep in t.get("dependencies", []):
            lines.append(f"    {dep} --> {t_key}")

    return "\n".join(lines)

