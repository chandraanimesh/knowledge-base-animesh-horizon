"""Base interface and protocol for multi-format artifact renderers."""

from abc import ABC, abstractmethod
from typing import Any

from rag_app.schemas.artifacts import TabularDataset, DownloadableArtifactFormat


class BaseArtifactRenderer(ABC):
    """Abstract base class for all file format renderers."""

    format: DownloadableArtifactFormat
    mime_type: str
    file_extension: str

    @abstractmethod
    def render(
        self,
        dataset: TabularDataset,
        title: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        """Renders validated TabularDataset into binary format."""
        pass
