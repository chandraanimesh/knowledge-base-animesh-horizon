import logging
import re
from rag_app.schemas.rag import RetrievedChunk

logger = logging.getLogger(__name__)

# Matches [chunk: <id>] case-insensitively (only accepted citation reference)
CHUNK_REF_PATTERN = re.compile(r"\[chunk:\s*([a-zA-Z0-9_\-]+)\]", re.IGNORECASE)


def build_chunk_reference(chunk: RetrievedChunk) -> str:
    """
    Builds a deterministic, traceable citation reference representation for a chunk.
    Example:
    [chunk: abc123]
    source=annual_report.pdf
    page=12
    """
    lines = [f"[chunk: {chunk.chunk_id}]"]
    file_name = (chunk.file_name or "").strip()
    if file_name and file_name.lower() != "none":
        lines.append(f"source={file_name}")
    if chunk.page_number is not None:
        lines.append(f"page={chunk.page_number}")
    return "\n".join(lines)


def render_citation_label(chunk: RetrievedChunk) -> str:
    """
    Renders a human-readable citation label for a validated chunk.
    - When document & page exist: "annual_report.pdf, page 12"
    - When only document exists: "annual_report.pdf"
    - When only chunk ID exists: "chunk: abc123"
    """
    file_name = (chunk.file_name or "").strip()
    has_file = bool(file_name and file_name.lower() != "none")
    has_page = chunk.page_number is not None

    if has_file and has_page:
        return f"{file_name}, page {chunk.page_number}"
    elif has_file:
        return file_name
    else:
        return f"chunk: {chunk.chunk_id}"


def normalize_citation_brackets(text: str) -> str:
    """
    Normalizes full-width / lenticular citation brackets (【1】, 【chunk: id】)
    produced by some LLMs to standard square brackets ([1], [chunk: id]).
    """
    if not text:
        return ""
    # Normalize 【1】 -> [1], 【12】 -> [12]
    text = re.sub(r"【\s*(\d+)\s*】", r"[\1]", text)
    # Normalize 【chunk: ...】 -> [chunk: ...]
    text = re.sub(r"【\s*chunk:\s*([^】]+)\s*】", r"[chunk: \1]", text, flags=re.IGNORECASE)
    # Normalize other fullwidth brackets ［1］, 〔1〕
    text = re.sub(r"［\s*(\d+)\s*］", r"[\1]", text)
    text = re.sub(r"〔\s*(\d+)\s*〕", r"[\1]", text)
    return text


def validate_and_render_citations(
    raw_answer: str,
    chunks: list[RetrievedChunk],
    is_conflict: bool = False,
) -> tuple[str, list[str]]:
    """
    Validates citation references in the LLM output against the supplied chunks.
    - Unknown/fabricated citation IDs are stripped from the answer and logged.
    - Valid citation references are preserved in the text and rendered into final unique citations.
    - If no explicit citations are found in normal non-conflict mode, falls back to unique source file names.
    """
    raw_answer = normalize_citation_brackets(raw_answer)
    chunk_map: dict[str, RetrievedChunk] = {str(c.chunk_id): c for c in chunks}

    sanitized_answer = raw_answer
    valid_chunks_cited: list[RetrievedChunk] = []
    seen_valid_ids: set[str] = set()
    invalid_citations: list[str] = []

    # 1. Process explicit [chunk: <id>] references
    for match in CHUNK_REF_PATTERN.finditer(raw_answer):
        ref_id = match.group(1).strip()
        full_match = match.group(0)
        if ref_id in chunk_map:
            chunk = chunk_map[ref_id]
            if ref_id not in seen_valid_ids:
                seen_valid_ids.add(ref_id)
                valid_chunks_cited.append(chunk)
        else:
            invalid_citations.append(ref_id)
            logger.warning(
                "invalid_citation_reference_rejected",
                extra={
                    "event": "invalid_citation_reference_rejected",
                    "invalid_citation_id": ref_id,
                    "raw_token": full_match,
                },
            )
            # Deterministically strip fabricated citation from output text
            sanitized_answer = sanitized_answer.replace(full_match, "")

    # Clean up any leftover double spaces or spacing before punctuation from stripped tokens
    sanitized_answer = re.sub(r" +([.,!?;])", r"\1", sanitized_answer)
    sanitized_answer = re.sub(r"  +", " ", sanitized_answer).strip()

    # 2. Check and strip fabricated line ranges (e.g. "L1-L3", "L1", "lines L1-L3")
    # unless the retrieved evidence explicitly provides and substantiates those line numbers.
    has_real_line_numbers = any(
        bool(re.search(r"\bL\d+(?:[-–—]L?\d+)?\b", c.text, re.IGNORECASE))
        for c in chunks
    )
    if not has_real_line_numbers:
        # Strip trailing or inline fabricated line coordinates in citations, e.g.:
        # "[Doc.pdf, L1-L3]" -> "[Doc.pdf]"
        # "[1, L1-L3]" -> "[1]"
        # "[chunk: abc, L1-L3]" -> "[chunk: abc]"
        # "(lines L1-L3)" -> ""
        # "[L1-L3]" -> ""
        sanitized_answer = re.sub(
            r"([\[(][^\])]*?)(?:,\s*|:\s*|\s+)?(?:lines?\s+)?L\d+(?:[-–—]L?\d+)?([^\])]*?[\])])",
            r"\1\2",
            sanitized_answer,
            flags=re.IGNORECASE,
        )
        # Remove standalone empty or line-only brackets/parentheses e.g. "[]", "()", "[L1-L3]"
        sanitized_answer = re.sub(r"\[\s*\]", "", sanitized_answer)
        sanitized_answer = re.sub(r"\(\s*\)", "", sanitized_answer)
        sanitized_answer = re.sub(r"\b(?:lines?\s+)?L\d+[-–—]L\d+\b", "", sanitized_answer, flags=re.IGNORECASE)
        sanitized_answer = re.sub(r" +([.,!?;])", r"\1", sanitized_answer)
        sanitized_answer = re.sub(r"  +", " ", sanitized_answer).strip()

    # 3. Render final citations
    if valid_chunks_cited:
        # Render labels and deduplicate preserving appearance order
        rendered = [render_citation_label(c) for c in valid_chunks_cited]
        final_citations = list(dict.fromkeys(rendered))
    else:
        # Fallback for non-conflict generation without explicit inline tags
        final_citations = sorted({c.file_name for c in chunks if c.file_name and c.file_name.lower() != "none"})

    return sanitized_answer, final_citations
