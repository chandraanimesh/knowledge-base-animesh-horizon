"""CSV artifact renderer with formula-safe escaping and RFC 4180 compliance."""

import csv
import io
import json
from typing import Any

from decimal import Decimal, InvalidOperation

from rag_app.reports.base import BaseArtifactRenderer
from rag_app.schemas.artifacts import DownloadableArtifactFormat, TabularDataset, VerificationStatus


def _is_safe_numeric(text: str) -> bool:
    """
    Determines if a string represents a legitimate, safe numeric value.
    Disallows formula operators and shell injection delimiters like '(', ')', '|', '!', '&', '$', ';', '='.
    """
    cleaned = text.strip()
    if not cleaned:
        return False
    # If any command/formula invocation character is present, it's unsafe
    if any(ch in cleaned for ch in "()|!&$=;`"):
        return False
    try:
        # Normalize commas (e.g. -1,250.50 -> -1250.50)
        norm = cleaned.replace(",", "")
        d = Decimal(norm)
        return d.is_finite()
    except (InvalidOperation, ValueError, TypeError):
        return False


class CSVRenderer(BaseArtifactRenderer):
    format = DownloadableArtifactFormat.CSV
    mime_type = "text/csv; charset=utf-8"
    file_extension = "csv"

    @staticmethod
    def _sanitize_cell_value(val: Any) -> Any:
        """
        Protects against spreadsheet formula injection (=, +, -, @, \\t, \\r).
        Preserves typed numeric values including Decimal and float, as well as safe stringified numbers.
        """
        if val is None:
            return ""
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        # Directly allow typed numbers (int, float, Decimal)
        if isinstance(val, (int, float, Decimal)):
            return val

        str_val = str(val)
        if not str_val:
            return ""

        lstripped = str_val.lstrip(" \t\r\n")
        raw_prefix = str_val[0] in ("=", "+", "-", "@", "\t", "\r")
        stripped_prefix = bool(lstripped and lstripped[0] in ("=", "+", "-", "@"))

        if raw_prefix or stripped_prefix:
            # If the value is a safe legitimate number (e.g. -500.00, +1250, -1,200.50), do not escape
            check_candidate = lstripped if stripped_prefix else str_val
            if check_candidate and check_candidate[0] in ("+", "-") and _is_safe_numeric(check_candidate):
                return str_val
            return f"'{str_val}"

        return str_val

    def render(
        self,
        dataset: TabularDataset,
        title: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        # Header row
        headers = [col.label for col in dataset.columns]
        writer.writerow(headers)

        # Verified data rows
        for row in dataset.rows:
            row_data = [
                self._sanitize_cell_value(row.get(col.key, ""))
                for col in dataset.columns
            ]
            writer.writerow(row_data)

        # Encode to UTF-8 with BOM for Excel compatibility
        return output.getvalue().encode("utf-8-sig")

