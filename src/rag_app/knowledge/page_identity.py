"""
Milestone 7B: Page Identity Resolver and Change Classifier.
Provides stable page identity tracking beyond simple page numbers:
- Normalized content fingerprints (preserving critical tokens: numbers, dates, currency, percentages, negations, headings)
- Structural fingerprinting (headings, lines, tables)
- Robust multi-pass page alignment (exact matches, shifted/moved pages, fuzzy modified, split, merged, added, deleted)
- Mapping confidence scoring and reliability assessment
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence
import uuid

from rag_app.ingestion.element import DocumentElement
from rag_app.ingestion.loader import load_document


class PageChangeStatus(str, Enum):
    UNCHANGED = "UNCHANGED"
    MODIFIED = "MODIFIED"
    ADDED = "ADDED"
    DELETED = "DELETED"
    MOVED = "MOVED"
    SPLIT = "SPLIT"
    MERGED = "MERGED"
    UNKNOWN = "UNKNOWN"


@dataclass
class PageData:
    """Represents extracted page-level content and deterministic fingerprints."""
    page_number: int
    raw_text: str
    normalized_text: str
    raw_content_hash: str
    normalized_content_hash: str
    structural_hash: str
    page_fingerprint: str
    headings: list[str] = field(default_factory=list)
    has_tables: bool = False
    elements: list[DocumentElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageDiffResult:
    """Structured result of comparing a single page against a prior version."""
    page_number: int
    page_fingerprint: str
    raw_content_hash: str
    structural_hash: str
    status: PageChangeStatus
    reused_from_page_id: uuid.UUID | None = None
    reused_from_page_number: int | None = None
    mapping_confidence: float = 1.0
    raw_text: str = ""
    reason: str = ""
    elements: list[DocumentElement] = field(default_factory=list)


@dataclass
class DocumentDiffSummary:
    """Document-level diff summary containing all page diffs and overall confidence."""
    total_pages_old: int
    total_pages_new: int
    unchanged_pages: list[PageDiffResult] = field(default_factory=list)
    modified_pages: list[PageDiffResult] = field(default_factory=list)
    added_pages: list[PageDiffResult] = field(default_factory=list)
    deleted_pages: list[PageDiffResult] = field(default_factory=list)
    moved_pages: list[PageDiffResult] = field(default_factory=list)
    split_pages: list[PageDiffResult] = field(default_factory=list)
    merged_pages: list[PageDiffResult] = field(default_factory=list)
    unknown_pages: list[PageDiffResult] = field(default_factory=list)
    all_new_pages: list[PageDiffResult] = field(default_factory=list)
    overall_confidence: float = 1.0
    is_reliable: bool = True
    escalate_to_full_reingest: bool = False


class PageIdentityResolver:
    """
    Deterministic page identity resolution, normalization, and change detection.
    Enforces that meaningful semantics (numbers, currency, percentages, headings)
    are strictly preserved during fingerprinting.
    """

    @staticmethod
    def normalize_text_safe(text: str) -> str:
        """
        Normalize trivial formatting variations without stripping numbers,
        dates, currency, policy limits, or negations.
        """
        if not text:
            return ""

        # Normalize line endings to \n
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        # Strip null bytes
        t = t.replace("\x00", "")
        # Remove trailing whitespace per line
        lines = [line.rstrip() for line in t.split("\n")]
        # Remove standalone page footer/header artifacts like "Page 1 of 10" or "--- Page 1 ---"
        cleaned_lines = []
        for line in lines:
            if re.match(r"^[-—\s]*Page\s+\d+(\s+of\s+\d+)?[-—\s]*$", line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)

        t = "\n".join(cleaned_lines)
        # Collapse multiple blank lines into at most two
        t = re.sub(r"\n{3,}", "\n\n", t)
        # Collapse consecutive horizontal whitespace (spaces/tabs) to a single space
        t = re.sub(r"[ \t]+", " ", t)
        return t.strip()

    @staticmethod
    def compute_hashes(page_number: int, text: str, elements: list[DocumentElement] | None = None) -> PageData:
        """Computes deterministic raw, normalized, and structural hashes for a page."""
        raw_text = text or ""
        norm_text = PageIdentityResolver.normalize_text_safe(raw_text)

        raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        norm_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

        # Structural extraction: extract markdown headings or lines
        headings = [line.strip() for line in raw_text.split("\n") if line.strip().startswith("#")]
        has_tables = "|" in raw_text and "-|-" in raw_text

        # Structural signature: line count, heading count, table presence, first 40 chars, last 40 chars
        non_empty_lines = [l for l in norm_text.split("\n") if l.strip()]
        line_count = len(non_empty_lines)
        head_sample = norm_text[:40].strip()
        tail_sample = norm_text[-40:].strip()
        struct_repr = f"L:{line_count}|H:{len(headings)}|T:{has_tables}|HS:{head_sample}|TS:{tail_sample}"
        struct_hash = hashlib.sha256(struct_repr.encode("utf-8")).hexdigest()

        # Composite page fingerprint: combines normalized content + structural hash
        composite = f"{norm_hash}:{struct_hash}"
        page_fingerprint = hashlib.sha256(composite.encode("utf-8")).hexdigest()

        return PageData(
            page_number=page_number,
            raw_text=raw_text,
            normalized_text=norm_text,
            raw_content_hash=raw_hash,
            normalized_content_hash=norm_hash,
            structural_hash=struct_hash,
            page_fingerprint=page_fingerprint,
            headings=headings,
            has_tables=has_tables,
            elements=elements or [],
        )

    @classmethod
    def extract_pages_from_elements(cls, elements: list[DocumentElement]) -> list[PageData]:
        """Groups DocumentElements by page_number and builds PageData objects."""
        if not elements:
            return []

        # If page_number is not set on elements, check for page markers or form feed
        has_page_numbers = any(el.page_number is not None for el in elements)
        pages_dict: dict[int, list[DocumentElement]] = {}

        if has_page_numbers:
            for el in elements:
                p_num = el.page_number if el.page_number is not None else 1
                pages_dict.setdefault(p_num, []).append(el)
        else:
            # Check if any element text contains form-feed '\x0c' or '--- PAGE X ---'
            current_page = 1
            for el in elements:
                if "\x0c" in el.text:
                    parts = [p.strip() for p in el.text.split("\x0c") if p.strip()]
                elif re.search(r"---+\s*PAGE\s+\d+\s*---+", el.text):
                    parts = [p.strip() for p in re.split(r"---+\s*PAGE\s+\d+\s*---+", el.text) if p.strip()]
                else:
                    parts = [el.text]

                if len(parts) > 1:
                    for part in parts:
                        cleaned_part = re.sub(r"^---+\s*PAGE\s+\d+\s*---+\s*", "", part, flags=re.MULTILINE).strip()
                        if cleaned_part:
                            sub_el = DocumentElement(
                                text=cleaned_part,
                                source_path=el.source_path,
                                file_name=el.file_name,
                                file_type=el.file_type,
                                element_type=el.element_type,
                                page_number=current_page,
                                metadata=dict(el.metadata),
                            )
                            pages_dict.setdefault(current_page, []).append(sub_el)
                            current_page += 1
                else:
                    pages_dict.setdefault(current_page, []).append(el)
                    current_page += 1

        # Build ordered list of PageData
        result: list[PageData] = []
        for p_num in sorted(pages_dict.keys()):
            p_elements = pages_dict[p_num]
            combined_text = "\n\n".join(el.text for el in p_elements if el.text)
            page_data = cls.compute_hashes(p_num, combined_text, p_elements)
            result.append(page_data)

        return result

    @classmethod
    def extract_pages_from_file(cls, file_path: str | Path) -> list[PageData]:
        """Loads a document using existing loader and returns parsed PageData items."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # TXT / MD: inspect form feeds or explicit page markers before standard load
        if path.suffix.lower() in {".txt", ".md"}:
            raw_content = path.read_text(encoding="utf-8", errors="ignore")
            if "\x0c" in raw_content:
                raw_pages = raw_content.split("\x0c")
            elif re.search(r"(?:^|\n)---+\s*PAGE\s+\d+\s*---+(?:\n|$)", raw_content):
                raw_pages = re.split(r"(?:^|\n)---+\s*PAGE\s+\d+\s*---+(?:\n|$)", raw_content)
            else:
                raw_pages = [raw_content]

            non_empty_blocks = [p.strip() for p in raw_pages if p and p.strip()]
            if len(non_empty_blocks) > 1:
                result = []
                for idx, text_block in enumerate(non_empty_blocks, start=1):
                    cleaned_block = re.sub(r"^---+\s*PAGE\s+\d+\s*---+\s*", "", text_block, flags=re.MULTILINE).strip()
                    if not cleaned_block:
                        continue
                    el = DocumentElement(
                        text=cleaned_block,
                        source_path=str(path),
                        file_name=path.name,
                        file_type=path.suffix.lower(),
                        element_type="text",
                        page_number=idx,
                        metadata={"source": str(path)},
                    )
                    result.append(cls.compute_hashes(idx, cleaned_block, [el]))
                if result:
                    return result

        # Standard loader fallback
        elements = load_document(path)
        return cls.extract_pages_from_elements(elements)

    @classmethod
    def calculate_text_similarity(cls, text_a: str, text_b: str) -> float:
        """Token-based Jaccard / overlap similarity between two normalized texts."""
        if not text_a and not text_b:
            return 1.0
        if not text_a or not text_b:
            return 0.0

        tokens_a = set(re.findall(r"\w+", text_a.lower()))
        tokens_b = set(re.findall(r"\w+", text_b.lower()))

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = intersection / union if union > 0 else 0.0

        # Substring/length containment bonus
        min_len = min(len(text_a), len(text_b))
        max_len = max(len(text_a), len(text_b))
        len_ratio = min_len / max_len if max_len > 0 else 1.0

        return 0.7 * jaccard + 0.3 * len_ratio

    @classmethod
    def compare_pages(
        cls,
        old_pages: Sequence[Any],  # DocumentPageVersion objects or PageData
        new_pages: Sequence[PageData],
    ) -> DocumentDiffSummary:
        """
        Executes robust multi-pass alignment:
        Pass 1: Exact fingerprint or content matches (UNCHANGED or MOVED)
        Pass 2: High token similarity / modified matches (MODIFIED)
        Pass 3: Split and merge detection (SPLIT, MERGED)
        Pass 4: Remaining unmatched items (ADDED, DELETED)
        Pass 5: Overall confidence calculation and reliability fallback trigger
        """
        summary = DocumentDiffSummary(
            total_pages_old=len(old_pages),
            total_pages_new=len(new_pages),
        )

        if not old_pages and not new_pages:
            summary.is_reliable = True
            return summary

        # Normalized lookup maps for old pages
        old_unmatched = list(range(len(old_pages)))
        new_unmatched = list(range(len(new_pages)))

        matched_new_results: dict[int, PageDiffResult] = {}

        # -------------------------------------------------------------
        # PASS 1: Exact matches (fingerprint or raw_content_hash)
        # -------------------------------------------------------------
        for n_idx in list(new_unmatched):
            n_page = new_pages[n_idx]
            for o_idx in list(old_unmatched):
                o_page = old_pages[o_idx]
                o_fp = getattr(o_page, "page_fingerprint", None)
                o_raw = getattr(o_page, "raw_content_hash", None)
                o_id = getattr(o_page, "id", None)
                o_num = getattr(o_page, "page_number", o_idx + 1)

                is_exact = (
                    (o_fp and o_fp == n_page.page_fingerprint)
                    or (o_raw and o_raw == n_page.raw_content_hash)
                    or (n_page.normalized_content_hash and getattr(o_page, "normalized_content_hash", None) == n_page.normalized_content_hash)
                )

                if is_exact:
                    if o_num == n_page.page_number:
                        res = PageDiffResult(
                            page_number=n_page.page_number,
                            page_fingerprint=n_page.page_fingerprint,
                            raw_content_hash=n_page.raw_content_hash,
                            structural_hash=n_page.structural_hash,
                            status=PageChangeStatus.UNCHANGED,
                            reused_from_page_id=o_id,
                            reused_from_page_number=o_num,
                            mapping_confidence=1.0,
                            raw_text=n_page.raw_text,
                            reason="Exact content and fingerprint match at same position",
                            elements=n_page.elements,
                        )
                        summary.unchanged_pages.append(res)
                    else:
                        res = PageDiffResult(
                            page_number=n_page.page_number,
                            page_fingerprint=n_page.page_fingerprint,
                            raw_content_hash=n_page.raw_content_hash,
                            structural_hash=n_page.structural_hash,
                            status=PageChangeStatus.MOVED,
                            reused_from_page_id=o_id,
                            reused_from_page_number=o_num,
                            mapping_confidence=0.98,
                            raw_text=n_page.raw_text,
                            reason=f"Exact match moved from page {o_num} to {n_page.page_number}",
                            elements=n_page.elements,
                        )
                        summary.moved_pages.append(res)

                    matched_new_results[n_idx] = res
                    new_unmatched.remove(n_idx)
                    old_unmatched.remove(o_idx)
                    break

        # -------------------------------------------------------------
        # PASS 2: Split and Merge Detection
        # -------------------------------------------------------------
        # Check Split: 1 old page split across 2 adjacent new pages
        for o_idx in list(old_unmatched):
            o_page = old_pages[o_idx]
            o_meta = getattr(o_page, "metadata_", {}) or {}
            if not isinstance(o_meta, dict):
                o_meta = {}
            o_text = getattr(o_page, "raw_text", "") or getattr(o_page, "normalized_text", "") or o_meta.get("raw_text", "") or o_meta.get("raw_text_snippet", "")
            o_id = getattr(o_page, "id", None)
            o_num = getattr(o_page, "page_number", o_idx + 1)

            for i in range(len(new_unmatched) - 1):
                n_idx1 = new_unmatched[i]
                n_idx2 = new_unmatched[i + 1]
                if new_pages[n_idx2].page_number == new_pages[n_idx1].page_number + 1:
                    comb_text = new_pages[n_idx1].normalized_text + " " + new_pages[n_idx2].normalized_text
                    sim = cls.calculate_text_similarity(o_text, comb_text)
                    if sim >= 0.75:
                        for n_idx, p_data in [(n_idx1, new_pages[n_idx1]), (n_idx2, new_pages[n_idx2])]:
                            res = PageDiffResult(
                                page_number=p_data.page_number,
                                page_fingerprint=p_data.page_fingerprint,
                                raw_content_hash=p_data.raw_content_hash,
                                structural_hash=p_data.structural_hash,
                                status=PageChangeStatus.SPLIT,
                                reused_from_page_id=o_id,
                                reused_from_page_number=o_num,
                                mapping_confidence=round(sim, 3),
                                raw_text=p_data.raw_text,
                                reason=f"Page split from old page {o_num}",
                                elements=p_data.elements,
                            )
                            summary.split_pages.append(res)
                            matched_new_results[n_idx] = res
                        new_unmatched.remove(n_idx1)
                        new_unmatched.remove(n_idx2)
                        old_unmatched.remove(o_idx)
                        break

        # Check Merge: 2 old pages merged into 1 new page
        for n_idx in list(new_unmatched):
            n_page = new_pages[n_idx]
            for i in range(len(old_unmatched) - 1):
                o_idx1 = old_unmatched[i]
                o_idx2 = old_unmatched[i + 1]
                o1 = old_pages[o_idx1]
                o2 = old_pages[o_idx2]
                o_num1 = getattr(o1, "page_number", o_idx1 + 1)
                o_num2 = getattr(o2, "page_number", o_idx2 + 1)
                if o_num2 == o_num1 + 1:
                    comb_text = (getattr(o1, "raw_text", "") or "") + " " + (getattr(o2, "raw_text", "") or "")
                    sim = cls.calculate_text_similarity(comb_text, n_page.normalized_text)
                    if sim >= 0.75:
                        res = PageDiffResult(
                            page_number=n_page.page_number,
                            page_fingerprint=n_page.page_fingerprint,
                            raw_content_hash=n_page.raw_content_hash,
                            structural_hash=n_page.structural_hash,
                            status=PageChangeStatus.MERGED,
                            reused_from_page_id=getattr(o1, "id", None),
                            reused_from_page_number=o_num1,
                            mapping_confidence=round(sim, 3),
                            raw_text=n_page.raw_text,
                            reason=f"Page merged from old pages {o_num1} and {o_num2}",
                            elements=n_page.elements,
                        )
                        summary.merged_pages.append(res)
                        matched_new_results[n_idx] = res
                        new_unmatched.remove(n_idx)
                        old_unmatched.remove(o_idx1)
                        old_unmatched.remove(o_idx2)
                        break

        # -------------------------------------------------------------
        # PASS 3: Content Similarity / Modification Detection
        # -------------------------------------------------------------
        for n_idx in list(new_unmatched):
            n_page = new_pages[n_idx]
            best_sim = 0.0
            best_o_idx = None

            for o_idx in old_unmatched:
                o_page = old_pages[o_idx]
                o_meta = getattr(o_page, "metadata_", {}) or {}
                if not isinstance(o_meta, dict):
                    o_meta = {}
                o_text = (
                    getattr(o_page, "raw_text", "")
                    or getattr(o_page, "normalized_text", "")
                    or o_meta.get("raw_text", "")
                    or o_meta.get("raw_text_snippet", "")
                )
                sim = cls.calculate_text_similarity(o_text, n_page.normalized_text)
                o_num = getattr(o_page, "page_number", o_idx + 1)
                if o_num == n_page.page_number:
                    sim = min(1.0, sim + 0.35)
                if sim > best_sim:
                    best_sim = sim
                    best_o_idx = o_idx

            # If similarity >= 0.40 and best_o_idx is not None, classify as MODIFIED
            if best_sim >= 0.40 and best_o_idx is not None:
                o_match = old_pages[best_o_idx]
                o_id = getattr(o_match, "id", None)
                o_num = getattr(o_match, "page_number", best_o_idx + 1)

                res = PageDiffResult(
                    page_number=n_page.page_number,
                    page_fingerprint=n_page.page_fingerprint,
                    raw_content_hash=n_page.raw_content_hash,
                    structural_hash=n_page.structural_hash,
                    status=PageChangeStatus.MODIFIED,
                    reused_from_page_id=o_id,
                    reused_from_page_number=o_num,
                    mapping_confidence=round(best_sim, 3),
                    raw_text=n_page.raw_text,
                    reason=f"Modified version of page {o_num} (similarity {best_sim:.2f})",
                    elements=n_page.elements,
                )
                summary.modified_pages.append(res)
                matched_new_results[n_idx] = res
                new_unmatched.remove(n_idx)
                old_unmatched.remove(best_o_idx)

        # -------------------------------------------------------------
        # PASS 4: Remaining Added and Deleted Pages
        # -------------------------------------------------------------
        for n_idx in list(new_unmatched):
            n_page = new_pages[n_idx]
            res = PageDiffResult(
                page_number=n_page.page_number,
                page_fingerprint=n_page.page_fingerprint,
                raw_content_hash=n_page.raw_content_hash,
                structural_hash=n_page.structural_hash,
                status=PageChangeStatus.ADDED,
                mapping_confidence=1.0,
                raw_text=n_page.raw_text,
                reason="New page inserted with no historical counterpart",
                elements=n_page.elements,
            )
            summary.added_pages.append(res)
            matched_new_results[n_idx] = res

        for o_idx in list(old_unmatched):
            o_page = old_pages[o_idx]
            o_num = getattr(o_page, "page_number", o_idx + 1)
            o_id = getattr(o_page, "id", None)
            res = PageDiffResult(
                page_number=o_num,
                page_fingerprint=getattr(o_page, "page_fingerprint", ""),
                raw_content_hash=getattr(o_page, "raw_content_hash", ""),
                structural_hash=getattr(o_page, "structural_hash", None) or "",
                status=PageChangeStatus.DELETED,
                reused_from_page_id=o_id,
                reused_from_page_number=o_num,
                mapping_confidence=1.0,
                raw_text=getattr(o_page, "raw_text", "") or "",
                reason=f"Page {o_num} removed in new document version",
            )
            summary.deleted_pages.append(res)

        # Assemble all_new_pages sorted by page_number
        sorted_indices = sorted(matched_new_results.keys())
        summary.all_new_pages = [matched_new_results[idx] for idx in sorted_indices]

        # -------------------------------------------------------------
        # PASS 5: Reliability & Fallback Assessment
        # -------------------------------------------------------------
        if summary.all_new_pages:
            conf_sum = sum(p.mapping_confidence for p in summary.all_new_pages)
            summary.overall_confidence = round(conf_sum / len(summary.all_new_pages), 3)
        else:
            summary.overall_confidence = 1.0

        # Check if mapping is unreliable (drastic layout rewrite, low confidence)
        # If > 50% of pages are low confidence (< 0.5) or overall confidence < 0.60
        low_conf_count = sum(1 for p in summary.all_new_pages if p.mapping_confidence < 0.5)
        if len(summary.all_new_pages) > 0 and (low_conf_count / len(summary.all_new_pages) > 0.4 or summary.overall_confidence < 0.60):
            summary.is_reliable = False
            summary.escalate_to_full_reingest = True
            for p in summary.all_new_pages:
                if p.mapping_confidence < 0.6:
                    p.status = PageChangeStatus.UNKNOWN

        return summary
