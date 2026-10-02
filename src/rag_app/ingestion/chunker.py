from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_app.ingestion.element import DocumentElement


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str

    chunk_index: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


def normalize_text(text: str) -> str:

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def chunk_elements(
    elements: list[DocumentElement],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TextChunk]:

    chunks: list[TextChunk] = []

    chunk_index = 0

    for element in elements:

        if not element.text.strip():
            continue

        element_chunks = _chunk_element(
            element,
            chunk_size,
            chunk_overlap,
        )

        for text, metadata in element_chunks:

            chunks.append(
                TextChunk(
                    text=text,
                    chunk_index=chunk_index,
                    metadata=metadata,
                )
            )

            chunk_index += 1

    return chunks

def _chunk_table(
    element: DocumentElement,
) -> list[tuple[str, dict[str, Any]]]:

    text = normalize_text(
        element.text
    )

    if not text:
        return []

    metadata = {
        **element.metadata,
        "element_type": "table",
    }

    if element.page_number is not None:
        metadata["page_number"] = (
            element.page_number
        )

    context_parts = []
    if "source" in element.metadata:
        from pathlib import Path
        import re
        filename = Path(element.metadata['source']).name
        clean_name = re.sub(r'^[a-fA-F0-9]{32}_', '', filename)
        clean_name = clean_name.replace('.pdf', '').replace('.md', '')
        clean_name = clean_name.replace('-', ' ').replace('_', ' ')
        context_parts.append(f"Document: {clean_name}")
    for k, v in element.metadata.items():
        if k.startswith("header_"):
            context_parts.append(f"Section: {v}")
            
    if context_parts:
        metadata["search_context"] = "\n".join(context_parts)
        
    lines = text.strip().split("\n")
    sentences = []
    ## Looping over sare row in the mardown
    for line in lines:
        line = line.strip()
        # skip empty lines or the md separator
        if not line or line.startswith("|-"):
            continue
        # row split kro |,
        parts = [p.strip() for p in line.split("|") if p.strip()]
        # # If it's a valid row with at least 2 columns, map Column 1 -> Column 2
        if len(parts) >= 2:
            key = parts[0]
            val = ", ".join(parts[1:])
            sentences.append(f"{key} is {val}.")
    flattened = " ".join(sentences)
    
    ctx = metadata.get("search_context", "")
    metadata["embedding_text"] = f"{ctx}\n\n{flattened}".strip()

    return [
        (
            text,
            metadata,
        )
    ]

def _chunk_element(
    element: DocumentElement,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, dict[str, Any]]]:

    if element.element_type == "record":
        return _chunk_record(element)

    if element.element_type == "table":
        return _chunk_table(element)

    if element.element_type == "figure":
        return _chunk_figure(element)

    if element.file_type in {".pdf", ".md"}:
        return _chunk_markdown(
            element,
            chunk_size,
            chunk_overlap,
        )

    return _chunk_plain_text(
        element,
        chunk_size,
        chunk_overlap,
    )
    
def _chunk_markdown(
    element: DocumentElement,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, dict[str, Any]]]:

    markdown = normalize_text(
        element.text
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ],
        strip_headers=True,
    )

    sections = header_splitter.split_text(
        markdown
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for section in sections:

        section_text = normalize_text(
            section.page_content
        )

        if not section_text:
            continue

        split_texts = recursive_splitter.split_text(
            section_text
        )

        for text in split_texts:

            metadata = dict(
                element.metadata
            )

            metadata.update(
                section.metadata
            )

            metadata["element_type"] = (
                "markdown"
            )

            chunks.append(
                (
                    text,
                    metadata,
                )
            )

    return chunks


def _chunk_plain_text(
    element: DocumentElement,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, dict[str, Any]]]:

    text = normalize_text(
        element.text
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return [
        (
            chunk,
            {
                **element.metadata,
                "element_type": "text",
            },
        )
        for chunk in splitter.split_text(text)
    ]


def _chunk_record(
    element: DocumentElement,
) -> list[tuple[str, dict[str, Any]]]:

    text = normalize_text(
        element.text
    )

    if not text:
        return []

    return [
        (
            text,
            {
                **element.metadata,
                "element_type": "record",
            },
        )
    ]

def _chunk_figure(
    element: DocumentElement,
) -> list[tuple[str, dict[str, Any]]]:

    text = normalize_text(
        element.text
    )

    if not text:
        return []

    # Strip base64 data URIs — they are huge binary blobs
    # that have no semantic value for retrieval and cause
    # DBAPIError when inserted into PostgreSQL TEXT columns.
    text = re.sub(
        r"!\[([^\]]*)\]\(data:[^)]+\)",
        r"[Figure: \1]" if r"\1" else "[Figure]",
        text,
    )
    text = text.strip()

    if not text:
        return []

    metadata = {
        **element.metadata,
        "element_type": "figure",
    }

    if element.page_number is not None:
        metadata["page_number"] = (
            element.page_number
        )

    return [
        (
            text,
            metadata,
        )
    ]