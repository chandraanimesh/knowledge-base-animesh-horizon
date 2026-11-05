"""Markdown Artifact Renderer

Generates clean, GitHub Flavored Markdown (GFM) documents with YAML frontmatter,
properly aligned tables, safe pipe escaping, and provenance footnotes.
"""

from __future__ import annotations

import datetime
from typing import Any

from rag_app.reports.base import BaseArtifactRenderer
from rag_app.schemas.artifacts import DownloadableArtifactFormat, TabularDataset


def _escape_md_cell(val: Any) -> str:
    """Escapes pipes and cleans newlines so Markdown tables do not break."""
    if val is None:
        return ""
    s = str(val).replace("|", "\\|").replace("\r\n", " ").replace("\n", " ")
    return s.strip()


class MDRenderer(BaseArtifactRenderer):
    format = DownloadableArtifactFormat.MD
    mime_type = "text/markdown; charset=utf-8"
    file_extension = "md"

    def render(
        self,
        dataset: TabularDataset,
        title: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        meta = metadata or {}
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        lines: list[str] = []

        # 1. Frontmatter
        lines.append("---")
        lines.append(f"title: \"{title}\"")
        lines.append(f"generated_at: \"{now_iso}\"")
        if "department" in meta:
            lines.append(f"department: \"{meta['department']}\"")
        if "workspace_id" in meta:
            lines.append(f"workspace_id: \"{meta['workspace_id']}\"")
        if "verification_status" in meta:
            lines.append(f"verification_status: \"{meta['verification_status']}\"")
        lines.append("---")
        lines.append("")

        # 2. Main Title
        lines.append(f"# {title}")
        lines.append("")

        # 3. Verification Badge / Callout
        if meta.get("verification_status"):
            lines.append(f"> **Verification Status:** `{meta['verification_status']}`")
            lines.append("")

        # 4. Summary / Narrative
        if summary and summary.strip():
            lines.append("## Executive Summary")
            lines.append(summary.strip())
            lines.append("")

        # 5. Tabular Dataset
        if dataset and dataset.columns:
            lines.append("## Data")
            lines.append("")

            # Header row
            headers = [_escape_md_cell(col.label) for col in dataset.columns]
            lines.append(f"| {' | '.join(headers)} |")

            # Separator row
            separators = []
            for col in dataset.columns:
                col_type = str(getattr(col, "data_type", getattr(col, "type", ""))).lower()
                if any(t in col_type for t in ("number", "integer", "currency", "numeric")):
                    separators.append("---:")
                else:
                    separators.append(":---")
            lines.append(f"| {' | '.join(separators)} |")

            # Data rows
            for row in dataset.rows:
                cells = [_escape_md_cell(row.get(col.key, "")) for col in dataset.columns]
                lines.append(f"| {' | '.join(cells)} |")
            lines.append("")

        # 6. Provenance & Footnotes
        sources = meta.get("sources") or meta.get("citations")
        if sources and isinstance(sources, list):
            lines.append("## Sources & Provenance")
            for i, src in enumerate(sources, 1):
                if isinstance(src, dict):
                    src_title = src.get("title") or src.get("file_name") or f"Source {i}"
                    src_url = src.get("url") or src.get("source_uri")
                    if src_url:
                        lines.append(f"{i}. [{src_title}]({src_url})")
                    else:
                        lines.append(f"{i}. {src_title}")
                else:
                    lines.append(f"{i}. {src}")
            lines.append("")

        content = "\n".join(lines)
        return content.encode("utf-8")
