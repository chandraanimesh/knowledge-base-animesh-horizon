from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rag_app.ingestion.element import DocumentElement


class DocumentReaderProvider(ABC):
    """
    Abstract interface for document loading and parsing providers.
    Describes the capability to parse raw files (PDF, DOCX, CSV, TXT, etc.) into structured DocumentElements.
    """

    @abstractmethod
    def load_document(self, file_path: str | Path) -> list[DocumentElement]:
        """
        Parse a single file into a list of DocumentElement objects.
        """
        raise NotImplementedError

    def load_documents(self, file_paths: list[str | Path]) -> list[DocumentElement]:
        """
        Parse multiple files into a combined list of DocumentElement objects.
        """
        elements: list[DocumentElement] = []
        for path in file_paths:
            elements.extend(self.load_document(path))
        return elements
