from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from rag_app.ingestion.element import DocumentElement
from rag_app.ingestion.normaliser import (
    canonical_field_name,
    normalize_company_id,
)


TEXT_EXTENSIONS = {".txt", ".md"}
STRUCTURED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
PDF_EXTENSIONS = {".pdf"}


def load_document(
    file_path: str | Path,
) -> list[DocumentElement]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
        
    if suffix == ".docx":
        return _load_docx(path)

    if suffix in TEXT_EXTENSIONS:
        return _load_text(path)

    if suffix in STRUCTURED_EXTENSIONS:
        return _load_structured_file(path)

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )


def load_documents(
    file_paths: list[str | Path],
) -> list[DocumentElement]:

    elements: list[DocumentElement] = []

    for file_path in file_paths:
        elements.extend(
            load_document(file_path)
        )

    return elements


# ============================================================
# TXT / MARKDOWN
# ============================================================

def _load_text(
    path: Path,
) -> list[DocumentElement]:

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return []

    element_type = (
        "text"
        if path.suffix.lower() == ".txt"
        else "text"
    )

    return [
        DocumentElement(
            text=text,
            source_path=str(path),
            file_name=path.name,
            file_type=path.suffix.lower(),
            element_type=element_type,
            metadata={
                "source": str(path),
            },
        )
    ]


# ============================================================
# PDF
# ============================================================

def _load_pdf(
    path: Path,
) -> list[DocumentElement]:

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )
    import logging

    logger = logging.getLogger(__name__)

    def _attempt_extraction(do_ocr: bool) -> str:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        
        if do_ocr:
            pipeline_options.ocr_options = RapidOcrOptions(
                lang=["en"],
            )

        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        result = converter.convert(
            str(path)
        )

        return result.document

    # Attempt normal extraction first
    doc = _attempt_extraction(do_ocr=False)
    extraction_method = "pdf_text"

    # Evaluate extraction quality
    full_text = doc.export_to_markdown()
    text_only = full_text.replace("<!-- image -->", "").strip()

    if len(text_only) < 50:
        logger.info(
            "pdf_extraction_fallback_ocr",
            extra={
                "event": "pdf_extraction_fallback_ocr",
                "file_name": path.name,
                "reason": "insufficient_text",
            }
        )
        doc = _attempt_extraction(do_ocr=True)
        extraction_method = "rapidocr"

    elements = []
    
    for item, level in doc.iterate_items():
        el_type = _get_docling_element_type(item)
        
        try:
            # Use doc argument if supported to avoid deprecation warnings
            text = item.export_to_markdown(doc=doc)
        except TypeError:
            text = _extract_element_text(item)
        except Exception:
            text = _extract_element_text(item)

        if not text or not text.strip() or text.strip() == "<!-- image -->":
            continue

        page_number = _extract_page_number(item)
        
        elements.append(
            DocumentElement(
                text=text.strip(),
                source_path=str(path),
                file_name=path.name,
                file_type=".pdf",
                element_type=el_type,
                page_number=page_number,
                metadata={
                    "source": str(path),
                    "parser": "docling",
                    "extraction_method": extraction_method,
                    "ocr": extraction_method == "rapidocr",
                },
            )
        )
        
    return elements

def _get_docling_element_type(
    item: Any,
) -> str:

    name = type(item).__name__.lower()

    if "table" in name:
        return "table"

    if "picture" in name or "figure" in name:
        return "figure"

    if "text" in name:
        return "text"

    return "text"
def _extract_element_text(
    item: Any,
) -> str:

    if hasattr(item, "export_to_markdown"):
        try:
            return item.export_to_markdown()
        except Exception:
            pass

    if hasattr(item, "text"):
        value = item.text

        if isinstance(value, str):
            return value

    if hasattr(item, "caption"):
        caption = item.caption

        if caption is not None:
            return str(caption)

    return str(item)

def _extract_page_number(
    item: Any,
) -> int | None:

    try:
        provenance = getattr(
            item,
            "prov",
            None,
        )

        if not provenance:
            return None

        first_provenance = provenance[0]

        page_number = getattr(
            first_provenance,
            "page_no",
            None,
        )

        return page_number

    except (IndexError, AttributeError):
        return None

# ============================================================
# DOCX
# ============================================================

def _load_docx(
    path: Path,
) -> list[DocumentElement]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document

    elements = []
    
    for item, level in doc.iterate_items():
        el_type = _get_docling_element_type(item)
        
        try:
            text = item.export_to_markdown(doc=doc)
        except TypeError:
            text = _extract_element_text(item)
        except Exception:
            text = _extract_element_text(item)

        if not text or not text.strip() or text.strip() == "<!-- image -->":
            continue

        page_number = _extract_page_number(item)
        
        elements.append(
            DocumentElement(
                text=text.strip(),
                source_path=str(path),
                file_name=path.name,
                file_type=".docx",
                element_type=el_type,
                page_number=page_number,
                metadata={
                    "source": str(path),
                    "parser": "docling"
                },
            )
        )
        
    return elements



# ============================================================
# CSV / XLSX
# ============================================================

def _load_structured_file(
    path: Path,
) -> list[DocumentElement]:

    suffix = path.suffix.lower()

    if suffix == ".csv":
        dataframe = pd.read_csv(path)

    elif suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)

    else:
        raise ValueError(
            f"Unsupported structured file type: {suffix}"
        )

    return _dataframe_to_elements(
        dataframe,
        path,
    )


def _dataframe_to_elements(
    dataframe: pd.DataFrame,
    path: Path,
) -> list[DocumentElement]:

    elements: list[DocumentElement] = []

    for row_index, row in dataframe.iterrows():

        metadata: dict[str, Any] = {}

        for column, value in row.items():

            if pd.isna(value):
                continue

            field_name = canonical_field_name(
                str(column)
            )

            clean_value = str(value).strip()

            if not clean_value:
                continue

            if field_name == "company_id":
                clean_value = normalize_company_id(
                    clean_value
                )

            metadata[field_name] = clean_value

        if not metadata:
            continue

        text = _record_to_text(
            metadata
        )

        metadata.update(
            {
                "source": str(path),
                "row_number": row_index + 1,
            }
        )

        elements.append(
            DocumentElement(
                text=text,
                source_path=str(path),
                file_name=path.name,
                file_type=path.suffix.lower(),
                element_type="record",
                metadata=metadata,
            )
        )

    return elements


def _record_to_text(
    record: dict[str, Any],
) -> str:

    return "\n".join(
        f"{key}: {value}"
        for key, value in record.items()
    )


# ============================================================
# RAW CSV STRING
# ============================================================

def load_structured_text(
    text: str,
) -> list[dict[str, Any]]:

    reader = csv.DictReader(
        StringIO(text)
    )

    records: list[dict[str, Any]] = []

    for row in reader:

        metadata: dict[str, Any] = {}

        for column, value in row.items():

            if value is None:
                continue

            field_name = canonical_field_name(
                column
            )

            clean_value = value.strip()

            if not clean_value:
                continue

            if field_name == "company_id":
                clean_value = normalize_company_id(
                    clean_value
                )

            metadata[field_name] = clean_value

        if metadata:
            records.append(metadata)

    return records


def get_document_reader_provider():
    from rag_app.providers.registry import get_provider
    return get_provider("document_reader")