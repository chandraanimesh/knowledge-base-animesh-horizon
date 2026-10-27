"""Registry and factory for artifact renderers."""

from rag_app.reports.base import BaseArtifactRenderer
from rag_app.reports.csv_renderer import CSVRenderer
from rag_app.reports.docx_renderer import DOCXRenderer
from rag_app.reports.json_renderer import JSONRenderer
from rag_app.reports.pdf_renderer import PDFRenderer
from rag_app.schemas.artifacts import DownloadableArtifactFormat


class RendererRegistry:
    def __init__(self):
        self._renderers: dict[DownloadableArtifactFormat, BaseArtifactRenderer] = {
            DownloadableArtifactFormat.CSV: CSVRenderer(),
            DownloadableArtifactFormat.JSON: JSONRenderer(),
            DownloadableArtifactFormat.PDF: PDFRenderer(),
            DownloadableArtifactFormat.DOCX: DOCXRenderer(),
        }

    def get_renderer(self, fmt: DownloadableArtifactFormat | str) -> BaseArtifactRenderer:
        format_enum = DownloadableArtifactFormat(str(fmt).lower())
        if format_enum not in self._renderers:
            raise ValueError(f"Unsupported export format: {fmt}")
        return self._renderers[format_enum]


renderer_registry = RendererRegistry()
