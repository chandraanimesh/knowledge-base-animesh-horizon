"""
Deterministic Knowledge-Quality Diagnostics (Milestone 7A.3).

Inexpensive, deterministic checks supported by ingestion metadata and text structure:
1. Empty or minimal extraction (parser noise or empty sections).
2. Malformed encoding / unresolved control or replacement characters (e.g. \\ufffd, \\x00).
3. High symbol / non-alphanumeric noise (garbled OCR or corrupted tables).
4. Missing required source location (e.g. missing page number on PDF).
5. Curated content drift (active chunk differs from original ingested text).
6. External CRM management locks.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional
from rag_app.schemas.chunk_curation import QualityCheckFlag


def run_deterministic_quality_checks(
    text: str,
    original_text: Optional[str] = None,
    current_version: int = 1,
    mime_type: Optional[str] = None,
    file_name: Optional[str] = None,
    page_number: Optional[int] = None,
    is_managed_external: bool = False,
    source_available: bool = True,
) -> List[QualityCheckFlag]:
    """
    Executes fast, deterministic quality checks on a chunk.
    Only returns substantiated flags with severity and remediation guidance.
    """
    flags: List[QualityCheckFlag] = []
    stripped = (text or "").strip()

    # 1. Empty or minimal extraction
    if len(stripped) == 0:
        flags.append(
            QualityCheckFlag(
                flag_id="EMPTY_EXTRACTION",
                severity="ERROR",
                message="Extracted text is empty. Parser produced zero content.",
                suggested_remediation="REINGEST_DOCUMENT",
            )
        )
    elif len(stripped) < 25:
        flags.append(
            QualityCheckFlag(
                flag_id="MINIMAL_EXTRACTION",
                severity="WARNING",
                message=f"Extracted text is very short ({len(stripped)} chars). May be page number or parser artifact.",
                suggested_remediation="MANUAL_CURATE",
            )
        )

    # 2. Malformed encoding / replacement characters
    if "\x00" in text or "\ufffd" in text:
        flags.append(
            QualityCheckFlag(
                flag_id="MALFORMED_ENCODING",
                severity="ERROR",
                message="Contains null bytes or unicode replacement characters (\\ufffd), indicating decoding failure.",
                suggested_remediation="REINGEST_DOCUMENT",
            )
        )
    else:
        # Check for control characters (excluding newline, carriage return, tab)
        ctrl_chars = [c for c in text if ord(c) < 32 and c not in "\n\r\t"]
        if ctrl_chars:
            flags.append(
                QualityCheckFlag(
                    flag_id="UNESCAPED_CONTROL_CHARS",
                    severity="WARNING",
                    message=f"Contains {len(ctrl_chars)} unescaped control character(s) from document parser.",
                    suggested_remediation="REINGEST_DOCUMENT",
                )
            )

    # 3. High symbol noise / low alphanumeric ratio
    non_ws = re.sub(r"\s+", "", text)
    if len(non_ws) >= 40:
        # Check if table or code block
        is_markdown_table = text.count("|") >= 4 and "-+-" in text or ("|---" in text)
        if not is_markdown_table:
            alnum_count = sum(1 for c in non_ws if c.isalnum())
            ratio = alnum_count / len(non_ws)
            if ratio < 0.40:
                flags.append(
                    QualityCheckFlag(
                        flag_id="HIGH_SYMBOL_NOISE",
                        severity="WARNING",
                        message=f"Low alphanumeric ratio ({ratio:.1%}). Content may be distorted OCR noise or corrupted table.",
                        suggested_remediation="REINGEST_DOCUMENT",
                    )
                )

    # 4. Missing required source location
    is_pdf = (
        (mime_type and "pdf" in mime_type.lower())
        or (file_name and file_name.lower().endswith(".pdf"))
    )
    if is_pdf and page_number is None:
        flags.append(
            QualityCheckFlag(
                flag_id="MISSING_PAGE_LOCATION",
                severity="INFO",
                message="PDF source is missing page number location metadata.",
                suggested_remediation="REINGEST_DOCUMENT",
            )
        )

    # 5. Curated content drift
    if current_version > 1 or (original_text is not None and original_text != text):
        flags.append(
            QualityCheckFlag(
                flag_id="CURATED_CONTENT_DRIFT",
                severity="INFO",
                message=f"Active text has been curated (version {current_version}) and differs from originally ingested content.",
                suggested_remediation="MANUAL_CURATE",
            )
        )

    # 6. External CRM managed chunk
    if is_managed_external:
        flags.append(
            QualityCheckFlag(
                flag_id="MANAGED_EXTERNAL_SYNC",
                severity="INFO",
                message="This chunk is synchronized from external Twenty CRM. Manual override is locked.",
                suggested_remediation="SOURCE_CORRECTION",
            )
        )

    # 7. Unavailable source on disk
    if not source_available:
        flags.append(
            QualityCheckFlag(
                flag_id="SOURCE_FILE_UNAVAILABLE",
                severity="WARNING",
                message="Original source file is not accessible on the host storage volume.",
                suggested_remediation="SOURCE_CORRECTION",
            )
        )

    return flags
