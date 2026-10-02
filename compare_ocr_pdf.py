
from pathlib import Path
import re
import fitz  # PyMuPDF

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

import pytesseract
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = Path(
    r"C:\ragDemo\ragdemo1\sample_docs\Whitestork-office-timings-Attendance-policy-2026 (1).pdf"
)

OUTPUT_DIR = Path("ocr_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# COMMON CHECKS
# ============================================================

TARGET_TERMS = [
    "10:00 AM",
    "7:30 PM",
    "Office Timings",
    "Working Days",
    "Ahmedabad Branch",
]


def check_terms(text: str, label: str) -> None:
    print("\n" + "=" * 70)
    print(f"{label} — TARGET TERM CHECK")
    print("=" * 70)

    normalized = text.lower()

    for term in TARGET_TERMS:
        found = term.lower() in normalized
        print(f"{'✓' if found else '✗'} {term}")


def save_text(filename: str, text: str) -> None:
    path = OUTPUT_DIR / filename
    path.write_text(text, encoding="utf-8")
    print(f"\nSaved: {path}")


# ============================================================
# PIPELINE 1
# DOCLING + RAPIDOCR
# ============================================================

def extract_with_docling(pdf_path: Path) -> str:
    print("\n" + "#" * 70)
    print("PIPELINE 1: DOCLING + RAPIDOCR")
    print("#" * 70)

    pipeline_options = PdfPipelineOptions()

    # Enable OCR
    pipeline_options.do_ocr = True

    # Let Docling/RapidOCR process the PDF
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(str(pdf_path))

    markdown = result.document.export_to_markdown()

    print(f"\nExtracted characters: {len(markdown)}")

    save_text("docling_output.md", markdown)

    check_terms(markdown, "DOCLING + RAPIDOCR")

    return markdown


# ============================================================
# PIPELINE 2
# PYMUPDF + TESSERACT
# ============================================================

def extract_with_tesseract(pdf_path: Path) -> str:
    print("\n" + "#" * 70)
    print("PIPELINE 2: PYMUPDF + TESSERACT")
    print("#" * 70)

    document = fitz.open(pdf_path)

    page_outputs = []

    print(f"\nTotal pages: {len(document)}")

    for page_number, page in enumerate(document, start=1):

        print(f"Processing page {page_number}...")

        # Render PDF page to image
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        # OCR
        text = pytesseract.image_to_string(
            image,
            config="--psm 6",
        )

        page_outputs.append(
            f"\n--- PAGE {page_number} ---\n{text}"
        )

    full_text = "\n".join(page_outputs)

    print(f"\nExtracted characters: {len(full_text)}")

    save_text("tesseract_output.txt", full_text)

    check_terms(full_text, "PYMUPDF + TESSERACT")

    return full_text


# ============================================================
# FIND OFFICE-TIMING CONTEXT
# ============================================================

def find_context(text: str, label: str) -> None:
    print("\n" + "=" * 70)
    print(f"{label} — OFFICE TIMING CONTEXT")
    print("=" * 70)

    pattern = re.compile(
        r".{0,250}(?:office timings|10:00 AM|7:30 PM).{0,500}",
        re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(text)

    if not matches:
        print("No relevant context found.")
        return

    for i, match in enumerate(matches[:5], start=1):
        print(f"\n--- MATCH {i} ---\n")
        print(match)


# ============================================================
# COMPARISON
# ============================================================

def compare(docling_text: str, tesseract_text: str) -> None:
    print("\n" + "#" * 70)
    print("FINAL COMPARISON")
    print("#" * 70)

    print("\nMetric                         Docling        Tesseract")
    print("-" * 60)

    print(
        f"Characters                    "
        f"{len(docling_text):<14} "
        f"{len(tesseract_text)}"
    )

    for term in TARGET_TERMS:
        d_found = term.lower() in docling_text.lower()
        t_found = term.lower() in tesseract_text.lower()

        print(
            f"{term:<30} "
            f"{'✓' if d_found else '✗':<14} "
            f"{'✓' if t_found else '✗'}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    print("=" * 70)
    print("PDF OCR / EXTRACTION COMPARISON")
    print("=" * 70)

    print(f"\nPDF: {PDF_PATH.resolve()}")

    # -----------------------------
    # Pipeline 1
    # -----------------------------

    docling_text = extract_with_docling(PDF_PATH)

    find_context(
        docling_text,
        "DOCLING + RAPIDOCR",
    )

    # -----------------------------
    # Pipeline 2
    # -----------------------------

    tesseract_text = extract_with_tesseract(PDF_PATH)

    find_context(
        tesseract_text,
        "PYMUPDF + TESSERACT",
    )

    # -----------------------------
    # Comparison
    # -----------------------------

    compare(
        docling_text,
        tesseract_text,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "\nOpen these files for detailed inspection:"
    )

    print(
        f"  {OUTPUT_DIR / 'docling_output.md'}"
    )

    print(
        f"  {OUTPUT_DIR / 'tesseract_output.txt'}"
    )


if __name__ == "__main__":
    main()

