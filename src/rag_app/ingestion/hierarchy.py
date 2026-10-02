"""
OV-1: Native Hierarchical Resource Model — HierarchyBuilder

Constructs a DocumentSection tree from ingested TextChunks and links
each chunk to the deepest enclosing section node.

Design principles:
- Entirely deterministic; no external LLM required.
- Uses heading metadata (header_1 / header_2 / header_3) already produced by
  the MarkdownHeaderTextSplitter in chunker.py.
- L0: bounded short abstract from section title + first N characters of content.
- L1: section title + child section listing (bounded).
- Additive only — existing chunks with NULL node_id remain fully functional.
- Authorization inherited from Document.department_id (no second auth table).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from rag_app.core.config import get_settings
from rag_app.ingestion.chunker import TextChunk

settings = get_settings()

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class SectionNode:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    document_id: uuid.UUID = field(default_factory=uuid.uuid4)
    parent_id: uuid.UUID | None = None
    node_type: str = "SECTION"
    title: str = ""
    level: int = 0
    ordinal_path: str = ""
    l0_abstract: str | None = None
    l1_overview: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_indices: list[int] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)


@dataclass
class HierarchyResult:
    root: SectionNode
    all_nodes: list[SectionNode]
    chunk_to_node: dict[int, SectionNode]


# ---------------------------------------------------------------------------
# Heading metadata helpers
# ---------------------------------------------------------------------------

_HEADING_KEYS = ("header_1", "header_2", "header_3")
_HEADING_LEVELS = {"header_1": 1, "header_2": 2, "header_3": 3}


def _heading_signature(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(metadata.get(k, "")).strip() for k in _HEADING_KEYS)


# ---------------------------------------------------------------------------
# L0 / L1 generation — deterministic, no LLM
# ---------------------------------------------------------------------------

def _generate_l0(title: str, content_text: str) -> str:
    limit = settings.node_l0_max_chars
    content = content_text.strip().replace("\n", " ")
    if title:
        prefix = f"{title}: "
        avail = limit - len(prefix)
        if avail <= 0:
            return title[:limit]
        snippet = content[:avail]
        if len(content) > avail:
            snippet = snippet.rstrip() + "..."
        return prefix + snippet
    snippet = content[:limit]
    if len(content) > limit:
        snippet = snippet.rstrip() + "..."
    return snippet


def _generate_l1(node: SectionNode, content_text: str) -> str:
    limit = settings.node_l1_max_chars
    parts: list[str] = []
    if node.title:
        parts.append(f"**{node.title}**")
    if node.children:
        child_line = "Sections: " + " | ".join(c.title for c in node.children if c.title)
        parts.append(child_line)
    parts.append(content_text.strip().replace("\n", " "))
    overview = "\n".join(parts)
    if len(overview) > limit:
        overview = overview[:limit].rstrip() + "..."
    return overview


# ---------------------------------------------------------------------------
# HierarchyBuilder
# ---------------------------------------------------------------------------

class HierarchyBuilder:
    def __init__(self, document_id: uuid.UUID, file_name: str = "") -> None:
        self.document_id = document_id
        self.file_name = file_name

    def build(self, chunks: list[TextChunk]) -> HierarchyResult:
        root = SectionNode(
            document_id=self.document_id,
            parent_id=None,
            node_type="DOCUMENT",
            title=self.file_name or "Document",
            level=0,
            ordinal_path="0",
        )
        all_nodes: list[SectionNode] = [root]
        chunk_to_node: dict[int, SectionNode] = {}

        has_headings = any(
            any(k in chunk.metadata for k in _HEADING_KEYS)
            for chunk in chunks
        )

        if not has_headings:
            for chunk in chunks:
                chunk_to_node[chunk.chunk_index] = root
                root.chunk_indices.append(chunk.chunk_index)
            self._fill_l0_l1(root, chunks)
            return HierarchyResult(root=root, all_nodes=all_nodes, chunk_to_node=chunk_to_node)

        # current_path: level -> SectionNode
        current_path: dict[int, SectionNode] = {0: root}
        ordinal_counters: dict[uuid.UUID, dict[int, int]] = {}
        prev_sig: tuple[str, ...] | None = None

        for chunk in chunks:
            sig = _heading_signature(chunk.metadata)
            if sig != prev_sig:
                changed_at = self._first_changed_level(sig, prev_sig)
                self._update_path(sig, changed_at, current_path, ordinal_counters, all_nodes)
                prev_sig = sig

            deepest = current_path[max(current_path.keys())]
            chunk_to_node[chunk.chunk_index] = deepest
            deepest.chunk_indices.append(chunk.chunk_index)

        chunk_map = {c.chunk_index: c for c in chunks}
        for node in all_nodes:
            content = " ".join(chunk_map[i].text for i in node.chunk_indices if i in chunk_map)
            node.l0_abstract = _generate_l0(node.title, content)
            node.l1_overview = _generate_l1(node, content)

        return HierarchyResult(root=root, all_nodes=all_nodes, chunk_to_node=chunk_to_node)

    @staticmethod
    def _first_changed_level(sig: tuple[str, ...], prev: tuple[str, ...] | None) -> int:
        if prev is None:
            return 1
        for i, (a, b) in enumerate(zip(sig, prev)):
            if a != b:
                return i + 1
        return len(_HEADING_KEYS)

    def _update_path(
        self,
        sig: tuple[str, ...],
        changed_at: int,
        current_path: dict[int, SectionNode],
        ordinal_counters: dict[uuid.UUID, dict[int, int]],
        all_nodes: list[SectionNode],
    ) -> None:
        for level_idx in range(1, len(_HEADING_KEYS) + 1):
            if level_idx < changed_at:
                continue
            heading_title = sig[level_idx - 1]
            if not heading_title:
                current_path.pop(level_idx, None)
                continue

            # find parent
            parent_level = level_idx - 1
            while parent_level > 0 and parent_level not in current_path:
                parent_level -= 1
            parent = current_path[parent_level]

            pid = parent.id
            if pid not in ordinal_counters:
                ordinal_counters[pid] = {}
            counters = ordinal_counters[pid]
            counters[level_idx] = counters.get(level_idx, 0) + 1

            parent_ord = parent.ordinal_path
            ordinal = (
                f"{parent_ord}.{counters[level_idx]}"
                if parent_ord and parent_ord != "0"
                else str(counters[level_idx])
            )

            node_type = "SECTION" if level_idx == 1 else "SUBSECTION"
            node = SectionNode(
                document_id=self.document_id,
                parent_id=parent.id,
                node_type=node_type,
                title=heading_title,
                level=level_idx,
                ordinal_path=ordinal,
            )
            parent.children.append(node)
            all_nodes.append(node)
            current_path[level_idx] = node
            # clear deeper levels
            for deeper in range(level_idx + 1, len(_HEADING_KEYS) + 1):
                current_path.pop(deeper, None)

    def _fill_l0_l1(self, node: SectionNode, chunks: list[TextChunk]) -> None:
        chunk_map = {c.chunk_index: c for c in chunks}
        content = " ".join(chunk_map[i].text for i in node.chunk_indices if i in chunk_map)
        node.l0_abstract = _generate_l0(node.title, content)
        node.l1_overview = _generate_l1(node, content)


# ---------------------------------------------------------------------------
# DB row conversion helper
# ---------------------------------------------------------------------------

def section_nodes_to_db_rows(nodes: list[SectionNode]) -> list[dict[str, Any]]:
    rows = []
    for node in nodes:
        rows.append({
            "id": node.id,
            "document_id": node.document_id,
            "parent_id": node.parent_id,
            "node_type": node.node_type,
            "title": node.title,
            "level": node.level,
            "ordinal_path": node.ordinal_path,
            "l0_abstract": node.l0_abstract,
            "l1_overview": node.l1_overview,
            "metadata_": node.metadata or None,
        })
    return rows
