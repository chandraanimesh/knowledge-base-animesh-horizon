import json
import logging
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from rag_app.generation.llm_factory import get_llm
from rag_app.providers.base.llm import LLMProvider
from rag_app.schemas.artifacts import (
    ColumnDataType,
    CoverageReport,
    ExportTruncation,
    OutputPreferences,
    RetrievalCoverage,
    RowProvenance,
    TableColumn,
    TabularDataset,
    VerificationStatus,
)
from rag_app.schemas.rag import RetrievedChunk

logger = logging.getLogger(__name__)


TABULAR_EXTRACTION_SYSTEM_PROMPT = """You are a strict, factual structured data extraction assistant for an enterprise RAG system.
Your task is to extract structured tabular records strictly and only from the provided Authorized Evidence Chunks.

CRITICAL INSTRUCTIONS:
1. ONLY extract information that is explicitly stated in the provided chunks.
2. NEVER invent, hallucinate, extrapolate, or estimate missing rows, entities, identifiers, dates, or numeric values.
3. Every row MUST reference the exact `_source_chunk_id` of the chunk from which it was extracted.
4. If the evidence contains no structured records matching the query, return an empty rows list.
5. Column names and keys must be clean, lowercase alphanumeric with underscores (e.g. record_id, category, amount, date, status, condition, owner).
6. Assign appropriate data_type for each column: "string", "number", "currency", "date", "integer", or "boolean". For ranges, bands, descriptive conditions, or open-ended thresholds (e.g. ">5% up to 10%", "up to 5%", "> $50,000", "over 50,000"), assign data_type "string". NEVER convert an open-ended threshold into an artificial incremented number (e.g. NEVER convert '> $50,000' into 50001).
7. For currency or number columns, provide pure numbers (e.g. 150000.00, -250.50). Do NOT write "$150k" or "150,000 INR" in the numeric value.
8. When evidence contains itemized lists, breakdowns, spending categories, transactions, or repeated entities, output EACH item/category as its own row in the `rows` array. Do NOT nest arrays or objects inside a single row.
9. If an attribute or amount does not apply to a row (e.g. general conditions, notes, policy restrictions), set its value explicitly to null. Do NOT coerce null to 0 or invent placeholder numbers.
10. Output MUST be a single valid JSON object adhering strictly to the schema:
{
  "title": "Title of the Dataset",
  "description": "Short description of the extracted records",
  "columns": [
    {"key": "col_key", "label": "Column Label", "data_type": "string|number|currency|date|integer|boolean", "unit": null}
  ],
  "rows": [
    {
      "_source_chunk_id": "chunk_id_here",
      "col_key": "value"
    }
  ]
}
""".strip()


def normalize_date_cell(value: Any) -> str | None:
    """Safely normalizes supported date formats to ISO-8601 (YYYY-MM-DD)."""
    if value is None:
        return None
    val_str = str(value).strip()
    if not val_str:
        return None

    # Already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
        try:
            dt = datetime.strptime(val_str, "%Y-%m-%d")
            if 1900 <= dt.year <= 2100:
                return val_str
        except ValueError:
            return None

    date_formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
        "%Y/%m/%d", "%Y.%m.%d",
    ]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(val_str, fmt)
            if 1900 <= dt.year <= 2100:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_numeric_cell(value: Any) -> float | None:
    """
    Normalizes numeric and currency values safely.
    Supports negative numbers (e.g. refunds / credit notes / discounts).
    Rejects NaN and Infinity.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        if val != val or val == float("inf") or val == float("-inf"):
            return None
        return val

    val_str = str(value).strip()
    if not val_str:
        return None

    # Remove common currency symbols and thousands separators
    cleaned = re.sub(r"[₹$,\s€£]", "", val_str)
    # Check parenthesized negative e.g. (500.00) -> -500.00
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    try:
        dec = Decimal(cleaned)
        val = float(dec)
        if val != val or val == float("inf") or val == float("-inf"):
            return None
        return val
    except (InvalidOperation, ValueError, TypeError):
        return None


def verify_cell_grounding(cell_val: Any, chunk_text: str) -> bool:
    """
    Verifies that a critical cell value appears in or is substantiated by the chunk text.
    Handles numeric formatting variations, date representations, inequalities (> / < / <= / >=),
    percentage ranges (e.g. '>5% up to 10%'), and textual conditions.
    """
    if cell_val is None:
        return True

    text_lower = chunk_text.lower()
    val_str = str(cell_val).strip()
    if not val_str:
        return True

    val_lower = val_str.lower()

    # 1. Direct string containment
    if val_lower in text_lower:
        return True

    # 2. Number variations (with/without commas, decimals, or international/Indian digit grouping)
    num = normalize_numeric_cell(cell_val)
    if num is not None:
        int_repr = str(int(num))
        # Strip commas, spaces, currency symbols from evidence for robust numerical comparison
        clean_evidence = re.sub(r"[₹$,\s€£¥]", "", text_lower)
        if int_repr in clean_evidence or int_repr in text_lower:
            return True
        fmt_2dp = f"{num:.2f}"
        if fmt_2dp in clean_evidence or fmt_2dp in text_lower:
            return True
        comma_std = f"{num:,.2f}".replace(".00", "")
        if comma_std.lower() in text_lower:
            return True

    # 3. Normalized date string check (e.g. 2026-11-20 vs Nov 20, 2026)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
        try:
            dt = datetime.strptime(val_str, "%Y-%m-%d")
            tokens = [
                str(dt.day),
                str(dt.year),
                dt.strftime("%b").lower(),
            ]
            if all(t in text_lower for t in tokens):
                return True
        except ValueError:
            pass

    # 4. Inequality and symbol expansion (e.g. > -> above/greater, < -> below/under, <= -> up to)
    expanded = val_lower
    expanded = re.sub(r">=\s*", "at least ", expanded)
    expanded = re.sub(r"<=\s*", "up to ", expanded)
    expanded = re.sub(r">\s*", "above ", expanded)
    expanded = re.sub(r"<\s*", "below ", expanded)
    if expanded in text_lower:
        return True

    # 5. Token-level grounding for composite range and condition expressions (e.g. '>5% up to 10%')
    cell_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", val_str)
    if cell_numbers:
        clean_evidence = re.sub(r"[₹$,\s€£¥]", "", text_lower)
        for n in cell_numbers:
            if not (re.search(rf"\b{re.escape(n)}\b", text_lower) or n in clean_evidence):
                return False
        # Significant non-numeric tokens must be substantiated
        tokens = re.findall(r"[a-zA-Z]+", expanded)
        meaningful = [t for t in tokens if t not in {"and", "or", "the", "a", "an", "to", "in", "of", "for"}]
        symbol_map = {
            "above": ["above", "over", "greater", "exceeding", ">"],
            "below": ["below", "under", "less", "<"],
            "up": ["up", "max", "maximum", "capped"],
        }
        for t in meaningful:
            equivalents = symbol_map.get(t, [t])
            if not any(eq in text_lower for eq in equivalents):
                return False
        return True

    # 6. Textual condition phrase check (all significant words must appear in evidence)
    words = re.findall(r"[a-zA-Z]{3,}", val_lower)
    if words:
        stop_words = {"the", "and", "for", "with", "from", "that", "this", "are", "have", "has"}
        sig_words = [w for w in words if w not in stop_words]
        if sig_words and all(w in text_lower for w in sig_words):
            return True

    return False


def flatten_and_expand_raw_rows(raw_rows: list[Any]) -> list[dict[str, Any]]:
    """
    Generically expands nested objects, dictionaries, or lists within rows
    so that each itemized record or repeated entity becomes its own tabular row.
    """
    expanded: list[dict[str, Any]] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            continue

        source_chunk = item.get("_source_chunk_id") or item.get("source_chunk_id")
        nested_collections: dict[str, Any] = {}
        scalar_fields: dict[str, Any] = {}

        for k, v in item.items():
            if k in ("_source_chunk_id", "source_chunk_id"):
                continue
            if isinstance(v, dict) and v:
                nested_collections[k] = v
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                nested_collections[k] = v
            else:
                scalar_fields[k] = v

        if not nested_collections:
            expanded.append(item)
        else:
            expanded_from_item = False
            for coll_key, coll_val in nested_collections.items():
                if isinstance(coll_val, dict):
                    for sub_k, sub_v in coll_val.items():
                        new_row = {
                            "_source_chunk_id": source_chunk,
                            **scalar_fields,
                            coll_key + "_item": sub_k,
                            coll_key + "_amount": sub_v,
                        }
                        expanded.append(new_row)
                        expanded_from_item = True
                elif isinstance(coll_val, list):
                    for sub_dict in coll_val:
                        new_row = {
                            "_source_chunk_id": source_chunk,
                            **scalar_fields,
                            **sub_dict,
                        }
                        expanded.append(new_row)
                        expanded_from_item = True
            if not expanded_from_item:
                expanded.append(item)
    return expanded


class TabularExtractionService:
    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or get_llm()

    async def extract_tabular_dataset(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        max_rows: int = 100,
    ) -> tuple[TabularDataset | None, CoverageReport | None]:
        """
        Extracts, server-side validates, and grounds a canonical TabularDataset from chunks.
        Treats model-generated rows as untrusted until validated against authorized evidence.
        """
        if not chunks:
            return None, None

        # Build authorized chunk lookup map (LLM CANNOT forge arbitrary chunk IDs)
        authorized_chunk_map: dict[str, RetrievedChunk] = {
            c.chunk_id: c for c in chunks if c.chunk_id
        }

        # Format evidence prompt
        evidence_blocks = []
        for i, c in enumerate(chunks):
            evidence_blocks.append(
                f"[Chunk ID: {c.chunk_id} | Document: {c.file_name} | Page: {c.page_number or 'N/A'}]\n{c.text}"
            )
        evidence_text = "\n\n---\n\n".join(evidence_blocks)

        user_content = (
            f"User Request: {query}\n\n"
            f"Authorized Evidence Chunks:\n{evidence_text}\n\n"
            f"Extract the structured records as valid JSON adhering to the schema."
        )

        messages = [
            {"role": "system", "content": TABULAR_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw_response, _usage = await self.llm.generate(messages, stage="generation")
        except Exception as exc:
            logger.error(f"Tabular extraction LLM call failed: {exc}")
            return None, None

        # Parse LLM JSON candidate
        cleaned = raw_response.strip()
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            b_start = cleaned.find("{")
            b_end = cleaned.rfind("}")
            json_text = cleaned[b_start : b_end + 1] if b_start != -1 and b_end > b_start else None

        if not json_text:
            logger.warning("No JSON object candidate found in tabular extraction response.")
            return None, None

        try:
            payload = json.loads(json_text)
        except Exception as exc:
            logger.warning(f"Failed to parse tabular extraction JSON: {exc}")
            return None, None

        raw_title = str(payload.get("title") or "Extracted Records")
        raw_desc = payload.get("description")
        raw_cols = payload.get("columns") or []
        raw_rows = payload.get("rows") or []
        if isinstance(raw_rows, list):
            raw_rows = flatten_and_expand_raw_rows(raw_rows)
        elif isinstance(payload, list):
            raw_rows = flatten_and_expand_raw_rows(payload)

        # Validate Column Definitions
        columns: list[TableColumn] = []
        for c in raw_cols:
            if isinstance(c, dict) and c.get("key"):
                dtype_str = str(c.get("data_type") or "string").lower()
                try:
                    dtype = ColumnDataType(dtype_str)
                except ValueError:
                    dtype = ColumnDataType.STRING
                columns.append(
                    TableColumn(
                        key=str(c.get("key")),
                        label=str(c.get("label") or c.get("key")),
                        data_type=dtype,
                        unit=c.get("unit"),
                    )
                )

        # Schema discovery / enrichment: Ensure all keys present in raw_rows exist in columns
        existing_keys = {c.key for c in columns}
        for r in raw_rows:
            if isinstance(r, dict):
                for k, v in r.items():
                    if k not in ("_source_chunk_id", "source_chunk_id") and k not in existing_keys:
                        dtype = ColumnDataType.STRING
                        if isinstance(v, (int, float)):
                            dtype = ColumnDataType.NUMBER
                        elif isinstance(v, str) and normalize_numeric_cell(v) is not None and any(c.isdigit() for c in v):
                            dtype = ColumnDataType.NUMBER
                        elif isinstance(v, bool):
                            dtype = ColumnDataType.BOOLEAN
                        columns.append(TableColumn(key=k, label=k.replace("_", " ").title(), data_type=dtype))
                        existing_keys.add(k)

        # Server-Side Row Validation & Grounding
        validated_rows: list[dict[str, Any]] = []
        row_provenance: dict[str, RowProvenance] = {}
        rejected_count = 0

        for idx, r in enumerate(raw_rows):
            if not isinstance(r, dict):
                rejected_count += 1
                continue

            row_id = f"row_{idx}"
            claimed_chunk_id = str(r.get("_source_chunk_id") or r.get("source_chunk_id") or "").strip()

            # 1. Verify Claimed Chunk ID against Authorized Context
            if claimed_chunk_id not in authorized_chunk_map:
                row_provenance[row_id] = RowProvenance(
                    source_chunk_id=claimed_chunk_id,
                    document_id="unknown",
                    document_name="unknown",
                    verification_status=VerificationStatus.REJECTED_FORGED_CHUNK,
                    is_grounded=False,
                    rejection_reason="Claimed _source_chunk_id is not in authorized context chunks.",
                )
                rejected_count += 1
                continue

            chunk = authorized_chunk_map[claimed_chunk_id]

            # 2. Dynamic Column Coercion & Type Validation
            clean_row: dict[str, Any] = {}
            row_has_type_error = False
            type_error_reason = ""

            for col in columns:
                raw_val = r.get(col.key)
                if raw_val is None:
                    clean_row[col.key] = None
                    continue

                if col.data_type == ColumnDataType.DATE:
                    norm_date = normalize_date_cell(raw_val)
                    if norm_date is None:
                        row_has_type_error = True
                        type_error_reason = f"Invalid date value '{raw_val}' for column '{col.key}'."
                        break
                    clean_row[col.key] = norm_date

                elif col.data_type in (ColumnDataType.NUMBER, ColumnDataType.CURRENCY):
                    norm_num = normalize_numeric_cell(raw_val)
                    if norm_num is None:
                        # If value is a grounded textual range or phrase (e.g. '>5% up to 10%'), gracefully adapt column to string
                        if verify_cell_grounding(raw_val, chunk.text):
                            col.data_type = ColumnDataType.STRING
                            clean_row[col.key] = str(raw_val)
                        else:
                            row_has_type_error = True
                            type_error_reason = f"Invalid numeric value '{raw_val}' for column '{col.key}'."
                            break
                    else:
                        clean_row[col.key] = norm_num

                elif col.data_type == ColumnDataType.INTEGER:
                    norm_num = normalize_numeric_cell(raw_val)
                    if norm_num is None:
                        if verify_cell_grounding(raw_val, chunk.text):
                            col.data_type = ColumnDataType.STRING
                            clean_row[col.key] = str(raw_val)
                        else:
                            row_has_type_error = True
                            type_error_reason = f"Invalid integer value '{raw_val}' for column '{col.key}'."
                            break
                    else:
                        clean_row[col.key] = int(norm_num)

                elif col.data_type == ColumnDataType.BOOLEAN:
                    clean_row[col.key] = bool(raw_val)
                else:
                    clean_row[col.key] = str(raw_val)

            if row_has_type_error:
                row_provenance[row_id] = RowProvenance(
                    source_chunk_id=claimed_chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.file_name,
                    page_number=chunk.page_number,
                    verification_status=VerificationStatus.REJECTED_TYPE_ERROR,
                    is_grounded=False,
                    rejection_reason=type_error_reason,
                )
                rejected_count += 1
                continue

            # 3. Field Grounding Verification against Source Chunk Text
            ungrounded_fields = []
            for col in columns:
                val = clean_row.get(col.key)
                if val is not None and not verify_cell_grounding(val, chunk.text):
                    ungrounded_fields.append(f"{col.key}={val}")

            if ungrounded_fields:
                row_provenance[row_id] = RowProvenance(
                    source_chunk_id=claimed_chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.file_name,
                    page_number=chunk.page_number,
                    verification_status=VerificationStatus.REJECTED_UNGROUNDED_CELL,
                    is_grounded=False,
                    rejection_reason=f"Cell value not grounded in chunk: {', '.join(ungrounded_fields)}",
                )
                rejected_count += 1
                continue

            # Row Passed All Verification Checks
            row_provenance[row_id] = RowProvenance(
                source_chunk_id=claimed_chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.file_name,
                page_number=chunk.page_number,
                verification_status=VerificationStatus.VERIFIED,
                is_grounded=True,
            )
            validated_rows.append(clean_row)

        if not validated_rows:
            return None, None

        # 4. Deterministic Metric Calculations
        summary_metrics: dict[str, Any] = {"total_records": len(validated_rows)}
        for col in columns:
            if col.data_type in (ColumnDataType.NUMBER, ColumnDataType.CURRENCY, ColumnDataType.INTEGER):
                vals = [r[col.key] for r in validated_rows if r.get(col.key) is not None and isinstance(r.get(col.key), (int, float))]
                if vals:
                    col_sum = sum(vals)
                    summary_metrics[f"sum_{col.key}"] = round(col_sum, 2)
                    summary_metrics[f"avg_{col.key}"] = round(col_sum / len(vals), 2)

        # 5. Export Truncation Handling
        total_extracted = len(validated_rows)
        if total_extracted > max_rows:
            final_rows = validated_rows[:max_rows]
            truncation = ExportTruncation(
                is_truncated=True,
                total_extracted_rows=total_extracted,
                exported_rows_count=max_rows,
                truncation_reason=f"Export capped at top {max_rows} records.",
            )
        else:
            final_rows = validated_rows
            truncation = ExportTruncation(
                is_truncated=False,
                total_extracted_rows=total_extracted,
                exported_rows_count=total_extracted,
            )

        coverage_report = CoverageReport(
            retrieval_coverage=RetrievalCoverage.SAMPLED_EVIDENCE,
            export_truncation=truncation,
            disclaimer=(
                f"Sample metrics represent {len(final_rows)} verified records retrieved in active context. "
                "This does NOT represent an exhaustive census for the entire department."
            ),
        )

        dataset = TabularDataset(
            title=raw_title,
            description=raw_desc,
            columns=columns,
            rows=final_rows,
            row_provenance=row_provenance,
            summary_metrics=summary_metrics,
            rejected_rows_count=rejected_count,
            coverage=coverage_report,
        )

        return dataset, coverage_report

    extract_and_verify = extract_tabular_dataset

