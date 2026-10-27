"""DOCX artifact renderer using python-docx with formatted tables and disclosures."""

from datetime import datetime, timezone
import io
from typing import Any
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from rag_app.reports.base import BaseArtifactRenderer
from rag_app.schemas.artifacts import (
    DownloadableArtifactFormat,
    RetrievalCoverage,
    TabularDataset,
    VerificationStatus,
)


class DOCXRenderer(BaseArtifactRenderer):
    format = DownloadableArtifactFormat.DOCX
    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    file_extension = "docx"

    def render(
        self,
        dataset: TabularDataset,
        title: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        doc = docx.Document()

        # Document Title
        heading = doc.add_heading(title or dataset.title or "Enterprise Intelligence Report", level=1)
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

        # Metadata subtitle
        p_sub = doc.add_paragraph()
        run_sub = p_sub.add_run(
            f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | Multi-Format RAG Platform"
        )
        run_sub.font.size = Pt(9)
        run_sub.font.color.rgb = RGBColor(120, 120, 120)

        # Audit & Disclosure Box
        p_disc = doc.add_paragraph()
        p_disc.paragraph_format.space_before = Pt(8)
        p_disc.paragraph_format.space_after = Pt(8)
        run_disc_label = p_disc.add_run("AUDIT NOTICE: ")
        run_disc_label.bold = True
        run_disc_label.font.size = Pt(8.5)
        run_disc_label.font.color.rgb = RGBColor(180, 100, 0)

        is_sampled = (
            dataset.coverage
            and dataset.coverage.retrieval_coverage == RetrievalCoverage.SAMPLED_EVIDENCE
        )
        is_truncated = (
            dataset.coverage
            and dataset.coverage.export_truncation.is_truncated
        )

        disc_text = ""
        if is_sampled:
            disc_text += (
                "Generated from sampled RAG evidence (top-k semantic search). "
                "Does NOT represent an exhaustive department-wide census. Figures reflect verified retrieved evidence only. "
            )
        if is_truncated:
            disc_text += f"Export truncated to {dataset.coverage.export_truncation.exported_rows_count} records. "
        if not is_sampled and not is_truncated:
            disc_text += "Authoritative record verification completed across all matched evidence."

        run_disc_body = p_disc.add_run(disc_text)
        run_disc_body.font.size = Pt(8.5)
        run_disc_body.font.italic = True

        # Summary narrative if available
        if summary or dataset.description:
            p_desc = doc.add_paragraph()
            p_desc.add_run(summary or dataset.description)

        # Summary Metrics
        if dataset.summary_metrics:
            doc.add_heading("Key Metrics", level=2)
            for k, v in dataset.summary_metrics.items():
                p_metric = doc.add_paragraph(style="List Bullet")
                run_k = p_metric.add_run(f"{k}: ")
                run_k.bold = True
                p_metric.add_run(str(v))

        # Verified Data Table
        columns = dataset.columns or []
        if columns and dataset.rows:
            doc.add_heading("Verified Data Records", level=2)
            table = doc.add_table(rows=1, cols=len(columns))
            table.style = "Light Shading Accent 1" if "Light Shading Accent 1" in [s.name for s in doc.styles] else "Table Grid"

            # Header row
            hdr_cells = table.rows[0].cells
            for idx, col in enumerate(columns):
                hdr_cells[idx].text = col.label
                for p in hdr_cells[idx].paragraphs:
                    for r in p.runs:
                        r.bold = True
                        r.font.size = Pt(9)

            # Data rows
            for row in dataset.rows:
                row_cells = table.add_row().cells
                for idx, col in enumerate(columns):
                    raw_val = row.get(col.key, "")
                    val_str = str(raw_val) if raw_val is not None else ""
                    row_cells[idx].text = val_str
                    for p in row_cells[idx].paragraphs:
                        for r in p.runs:
                            r.font.size = Pt(8.5)


        # Save to buffer
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
