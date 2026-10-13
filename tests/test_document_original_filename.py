import uuid
import pytest
from rag_app.core.filenames import derive_original_filename
from rag_app.schemas.users import DocumentResponse
from rag_app.schemas.rag import DocumentChunkInfo
from rag_app.schemas.knowledge import CompanyNoteEvidenceItem


def test_derive_original_filename_standard_upload():
    # 32-hex UUID prefix with uploaded_docs path
    file_name = "e8c1715305764a119183b8820bd8f58d_ov2_implementation_plan.md"
    file_path = "uploaded_docs/e8c1715305764a119183b8820bd8f58d_ov2_implementation_plan.md"
    assert derive_original_filename(file_name, file_path) == "ov2_implementation_plan.md"


def test_derive_original_filename_windows_path():
    # Windows backslash storage path
    file_name = "0a54d58489d34ebca39bba76ae6aade3_Q3_Sales_Performance_Pipeline_Report.md"
    file_path = "uploaded_docs\\0a54d58489d34ebca39bba76ae6aade3_Q3_Sales_Performance_Pipeline_Report.md"
    assert derive_original_filename(file_name, file_path) == "Q3_Sales_Performance_Pipeline_Report.md"


def test_derive_original_filename_complex_name_with_parentheses():
    file_name = "6c8473c3ae39443d8036969fca055b6c_Taber_Connect_Feature_Requirement_Summary (2).docx (1).pdf"
    file_path = "uploaded_docs/6c8473c3ae39443d8036969fca055b6c_Taber_Connect_Feature_Requirement_Summary (2).docx (1).pdf"
    assert derive_original_filename(file_name, file_path) == "Taber_Connect_Feature_Requirement_Summary (2).docx (1).pdf"


def test_derive_original_filename_preserves_external_non_upload_paths():
    # File not stored in uploaded_docs must not be transformed
    file_name = "e8c1715305764a119183b8820bd8f58d_internal_system_file.md"
    file_path = "system_configs/e8c1715305764a119183b8820bd8f58d_internal_system_file.md"
    assert derive_original_filename(file_name, file_path) == file_name


def test_derive_original_filename_preserves_unprefixed_files():
    file_name = "company_handbook.pdf"
    file_path = "uploaded_docs/company_handbook.pdf"
    assert derive_original_filename(file_name, file_path) == "company_handbook.pdf"


def test_derive_original_filename_handles_none_empty():
    assert derive_original_filename(None) is None
    assert derive_original_filename("") == ""


def test_schemas_preserve_internal_ids_and_expose_original_name():
    doc_id = uuid.uuid4()
    doc_res = DocumentResponse(
        id=doc_id,
        file_name="e8c1715305764a119183b8820bd8f58d_ov2_implementation_plan.md",
        original_file_name=derive_original_filename(
            "e8c1715305764a119183b8820bd8f58d_ov2_implementation_plan.md",
            "uploaded_docs/e8c1715305764a119183b8820bd8f58d_ov2_implementation_plan.md",
        ),
        status="READY",
        created_at="2026-10-11T12:00:00Z",
    )
    assert doc_res.id == doc_id
    assert doc_res.file_name == "e8c1715305764a119183b8820bd8f58d_ov2_implementation_plan.md"
    assert doc_res.original_file_name == "ov2_implementation_plan.md"

    chunk_info = DocumentChunkInfo(
        id="chunk-123",
        document_id=str(doc_id),
        chunk_index=0,
        excerpt="Implementation plan details...",
        file_name=doc_res.file_name,
        original_file_name=doc_res.original_file_name,
    )
    assert chunk_info.id == "chunk-123"
    assert chunk_info.document_id == str(doc_id)
    assert chunk_info.original_file_name == "ov2_implementation_plan.md"

    ev_id = uuid.uuid4()
    ev_item = CompanyNoteEvidenceItem(
        id=ev_id,
        source_type="DOCUMENT",
        document_id=doc_id,
        file_name=doc_res.file_name,
        original_file_name=doc_res.original_file_name,
        excerpt="Implementation plan details...",
        created_at="2026-10-11T12:00:00Z",
    )
    assert ev_item.id == ev_id
    assert ev_item.document_id == doc_id
    assert ev_item.original_file_name == "ov2_implementation_plan.md"
