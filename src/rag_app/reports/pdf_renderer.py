"""Enterprise PDF artifact renderer using PyMuPDF (fitz) with pagination and disclosures."""

from datetime import datetime, timezone
from typing import Any
import fitz

from rag_app.reports.base import BaseArtifactRenderer
from rag_app.schemas.artifacts import (
    DownloadableArtifactFormat,
    RetrievalCoverage,
    TabularDataset,
    VerificationStatus,
)


class PDFRenderer(BaseArtifactRenderer):
    format = DownloadableArtifactFormat.PDF
    mime_type = "application/pdf"
    file_extension = "pdf"

    def render(
        self,
        dataset: TabularDataset,
        title: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        doc = fitz.open()
        
        # Dimensions for A4
        page_width = 595.0
        page_height = 842.0
        margin_x = 40.0
        margin_top = 40.0
        margin_bottom = 50.0
        content_width = page_width - (2 * margin_x)  # 515.0

        def create_page():
            page = doc.new_page(width=page_width, height=page_height)
            return page

        current_page = create_page()
        y = margin_top

        # 1. Header Banner
        banner_rect = fitz.Rect(margin_x, y, margin_x + content_width, y + 45)
        current_page.draw_rect(banner_rect, color=None, fill=(0.11, 0.23, 0.36))  # Navy
        current_page.insert_text(
            (margin_x + 12, y + 22),
            (title or dataset.title or "Enterprise Intelligence Report")[:50],
            fontsize=13,
            color=(1, 1, 1),
        )
        subtitle = f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Multi-Format RAG Platform"
        current_page.insert_text(
            (margin_x + 12, y + 36),
            subtitle,
            fontsize=8,
            color=(0.85, 0.85, 0.85),
        )
        y += 55

        # 2. Coverage & Truthfulness Disclosure
        is_sampled = (
            dataset.coverage
            and dataset.coverage.retrieval_coverage == RetrievalCoverage.SAMPLED_EVIDENCE
        )
        is_truncated = (
            dataset.coverage
            and dataset.coverage.export_truncation.is_truncated
        )

        disclosure_text = "AUDIT NOTICE: "
        if is_sampled:
            disclosure_text += (
                "Generated from sampled RAG evidence (top-k semantic search). "
                "Does NOT represent an exhaustive department-wide census. Figures reflect verified retrieved evidence only. "
            )
        if is_truncated:
            disclosure_text += (
                f"Export truncated to {dataset.coverage.export_truncation.exported_rows_count} records. "
            )
        if not is_sampled and not is_truncated:
            disclosure_text += "Authoritative record verification completed across all matched evidence."

        disc_rect = fitz.Rect(margin_x, y, margin_x + content_width, y + 28)
        current_page.draw_rect(disc_rect, color=(0.85, 0.65, 0.13), fill=(0.99, 0.98, 0.92))
        current_page.insert_textbox(
            fitz.Rect(margin_x + 8, y + 4, margin_x + content_width - 8, y + 26),
            disclosure_text,
            fontsize=7,
            color=(0.4, 0.25, 0.05),
        )
        y += 36

        # 3. Summary / Metrics Section (if present)
        if dataset.summary_metrics:
            metrics_str = "Key Metrics: " + "  |  ".join(
                f"{k}: {v}" for k, v in dataset.summary_metrics.items()
            )
            current_page.insert_text(
                (margin_x + 4, y + 10),
                metrics_str[:120],
                fontsize=8,
                color=(0.15, 0.2, 0.3),
            )
            y += 18

        # 4. Table Construction
        columns = dataset.columns or []

        if not columns:
            current_page.insert_text((margin_x, y + 20), "No structured data columns available.", fontsize=9)
            return doc.tobytes()

        # Compute column widths
        col_count = len(columns)
        base_col_width = content_width / col_count
        col_widths = [base_col_width] * col_count

        def draw_table_header(page, cur_y):
            hdr_rect = fitz.Rect(margin_x, cur_y, margin_x + content_width, cur_y + 18)
            page.draw_rect(hdr_rect, color=None, fill=(0.2, 0.25, 0.32))
            cx = margin_x
            for idx, col in enumerate(columns):
                w = col_widths[idx]
                col_title = col.label[:18]
                page.insert_text((cx + 5, cur_y + 12), col_title, fontsize=8, color=(1, 1, 1))
                cx += w
            return cur_y + 18

        y = draw_table_header(current_page, y)

        row_height = 16.0
        for r_idx, row in enumerate(dataset.rows):
            if y + row_height > (page_height - margin_bottom):
                # New Page
                current_page = create_page()
                y = margin_top
                y = draw_table_header(current_page, y)

            # Alternate row background
            bg_color = (0.96, 0.97, 0.98) if r_idx % 2 == 1 else (1.0, 1.0, 1.0)
            row_rect = fitz.Rect(margin_x, y, margin_x + content_width, y + row_height)
            current_page.draw_rect(row_rect, color=(0.88, 0.88, 0.88), fill=bg_color)

            # Draw cells
            cx = margin_x
            for idx, col in enumerate(columns):
                w = col_widths[idx]
                raw_val = row.get(col.key, "")
                val_str = str(raw_val) if raw_val is not None else ""
                # Clip text to fit column
                max_chars = max(6, int(w / 5.5))
                if len(val_str) > max_chars:
                    val_str = val_str[: max_chars - 2] + ".."
                current_page.insert_text((cx + 5, y + 11), val_str, fontsize=7.5, color=(0.1, 0.1, 0.1))
                cx += w

            y += row_height

        # Footer on all pages
        total_pages = len(doc)
        for p_idx, page in enumerate(doc):
            footer_text = f"Page {p_idx + 1} of {total_pages}  |  CONFIDENTIAL  |  Verified RAG Extraction"
            page.insert_text(
                (margin_x, page_height - 25),
                footer_text,
                fontsize=7,
                color=(0.5, 0.5, 0.5),
            )

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
