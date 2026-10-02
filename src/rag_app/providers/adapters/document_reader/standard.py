from pathlib import Path
from rag_app.ingestion.element import DocumentElement
from rag_app.providers.base.document_reader import DocumentReaderProvider


class StandardDocumentReaderAdapter(DocumentReaderProvider):
    """
    Standard multi-format document reader adapter wrapping existing loaders.
    Supports PDF, DOCX, CSV, Excel, TXT, and Markdown files.
    """

    def load_document(self, file_path: str | Path) -> list[DocumentElement]:
        from rag_app.ingestion.loader import load_document as _load_doc
        return _load_doc(file_path)

    def load_documents(self, file_paths: list[str | Path]) -> list[DocumentElement]:
        from rag_app.ingestion.loader import load_documents as _load_docs
        return _load_docs(file_paths)
