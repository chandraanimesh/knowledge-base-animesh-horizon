import io
import zipfile
import pytest
from pathlib import Path

from rag_app.ingestion.docx_media import (
    DocxMediaItem,
    extract_docx_media,
    MAX_EXTRACTED_MEDIA_FILES,
    MAX_UNCOMPRESSED_MEDIA_BYTES,
)


def _create_mock_docx(entries: dict[str, bytes]) -> bytes:
    """Helper creating an in-memory zip resembling a DOCX package."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Standard DOCX boilerplate
        zf.writestr("[Content_Types].xml", b"<Types></Types>")
        zf.writestr("word/document.xml", b"<w:document><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>")
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_docx_extracts_embedded_images():
    """Verify PNG and JPEG files inside word/media/ are safely extracted."""
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    fake_jpg = b"\xff\xd8\xff\xe0" + b"\x00" * 64

    docx_bytes = _create_mock_docx({
        "word/media/image1.png": fake_png,
        "word/media/diagram2.jpeg": fake_jpg,
        "word/other_file.txt": b"ignore me",
    })

    items = extract_docx_media(docx_bytes)
    assert len(items) == 2

    png_item = next(i for i in items if i.file_name == "image1.png")
    assert png_item.format == "png"
    assert png_item.byte_size == len(fake_png)
    assert png_item.sha256 is not None
    assert "[Embedded Image: image1.png" in png_item.text_representation

    jpg_item = next(i for i in items if i.file_name == "diagram2.jpeg")
    assert jpg_item.format == "jpeg"
    assert jpg_item.byte_size == len(fake_jpg)


def test_docx_no_media_passes_cleanly():
    """Verify DOCX with no images returns empty list without error."""
    docx_bytes = _create_mock_docx({})
    items = extract_docx_media(docx_bytes)
    assert items == []


def test_docx_rejects_zip_slip_traversal():
    """Verify attempts to escape word/media/ via directory traversal (..) are rejected."""
    malicious_bytes = b"\x89PNG\r\n\x1a\n" + b"malicious content"

    # Zip with traversal entry
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/media/../../malicious.png", malicious_bytes)
        zf.writestr("word/media/image_valid.png", b"\x89PNG\r\n\x1a\nvalid")
    
    items = extract_docx_media(buf.getvalue())
    # The malicious traversal entry must be dropped
    file_names = [i.file_name for i in items]
    assert "malicious.png" not in file_names
    assert len(items) == 1
    assert items[0].file_name == "image_valid.png"


def test_docx_guards_against_excessive_files_and_bytes():
    """Verify caps on file count and total uncompressed bytes."""
    entries = {}
    fake_png = b"\x89PNG\r\n\x1a\n" + b"X" * 100
    # Create 60 image entries (limit is 50)
    for i in range(60):
        entries[f"word/media/img_{i}.png"] = fake_png

    docx_bytes = _create_mock_docx(entries)
    items = extract_docx_media(docx_bytes, max_files=10)
    assert len(items) <= 10

    # Total bytes cap test
    items_capped = extract_docx_media(docx_bytes, max_total_bytes=500)
    assert len(items_capped) < 10

