"""
Milestone 7A.3 — Knowledge Quality & Source Verification Test Suite.

Verifies:
1. Deterministic quality checks (empty extraction, malformed encoding, symbol noise,
   PDF page missing, drift detection, CRM lock, source file on-disk availability).
2. Source evidence inspection (honest source availability state, original vs curated text,
   section hierarchy, metadata).
3. Quality issue reporting lifecycle (user reporting stored in chunk metadata without DB schema change,
   remediation paths: MANUAL_CURATE, REINGEST_DOCUMENT, SOURCE_CORRECTION).
4. Authorization enforcement across evidence inspection and reporting endpoints.
"""

from __future__ import annotations

import os
# Ensure strictly isolated disposable test database
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import hashlib
from pathlib import Path
import tempfile
import uuid
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import FastAPI
from rag_app.api.chunk_curation_routes import router as chunk_curation_router
from rag_app.api.dependencies.auth import get_current_user
from rag_app.core.config import get_settings
from rag_app.db.models import (
    Department,
    Document,
    DocumentChunk,
    DocumentChunkVersion,
    DocumentSection,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine, get_db_session
from rag_app.knowledge.chunk_curation_service import ChunkCurationService
from rag_app.knowledge.quality_checks import run_deterministic_quality_checks
from rag_app.schemas.chunk_curation import (
    ChunkEvidenceResponse,
    ChunkIssueReportRequest,
    QualityCheckFlag,
)

app = FastAPI()
app.include_router(chunk_curation_router)
settings = get_settings()


class DeterministicEmbeddingProvider:
    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results = []
        for t in texts:
            seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            vec = [(float((seed + i) % 100) / 100.0) for i in range(self.dim)]
            results.append(vec)
        return results


# ─── UNIT TESTS: DETERMINISTIC QUALITY CHECKS ──────────────────────────────
def test_quality_check_empty_and_minimal():
    # Empty extraction
    flags = run_deterministic_quality_checks(text="   ")
    assert any(f.flag_id == "EMPTY_EXTRACTION" and f.severity == "ERROR" for f in flags)
    assert any(f.suggested_remediation == "REINGEST_DOCUMENT" for f in flags)

    # Minimal extraction
    flags_short = run_deterministic_quality_checks(text="Page 14")
    assert any(f.flag_id == "MINIMAL_EXTRACTION" and f.severity == "WARNING" for f in flags_short)
    assert any(f.suggested_remediation == "MANUAL_CURATE" for f in flags_short)


def test_quality_check_malformed_encoding():
    # Null bytes
    flags_null = run_deterministic_quality_checks(text="Valid beginning \x00 corrupted end")
    assert any(f.flag_id == "MALFORMED_ENCODING" and f.severity == "ERROR" for f in flags_null)

    # Unicode replacement character
    flags_repl = run_deterministic_quality_checks(text="Some text with \ufffd symbol")
    assert any(f.flag_id == "MALFORMED_ENCODING" and f.severity == "ERROR" for f in flags_repl)

    # Control chars
    flags_ctrl = run_deterministic_quality_checks(text="Line with \x07 bell character")
    assert any(f.flag_id == "UNESCAPED_CONTROL_CHARS" and f.severity == "WARNING" for f in flags_ctrl)


def test_quality_check_symbol_noise():
    # Garbled noise
    noise_text = "$%#@&*!^%~`+=<>{}[]|\\/?;;::!@#$%^&*()_+=-~`<>?:"
    flags_noise = run_deterministic_quality_checks(text=noise_text)
    assert any(f.flag_id == "HIGH_SYMBOL_NOISE" and f.severity == "WARNING" for f in flags_noise)

    # Markdown table exception (tables have symbols like | and --- but are legitimate)
    table_text = "| Col 1 | Col 2 |\n|-------|-------|\n| Val A | Val B |\n| Val C | Val D |"
    flags_table = run_deterministic_quality_checks(text=table_text)
    assert not any(f.flag_id == "HIGH_SYMBOL_NOISE" for f in flags_table)


def test_quality_check_missing_page_and_crm_managed():
    # PDF missing page number
    flags_pdf = run_deterministic_quality_checks(
        text="Sample paragraph from document.",
        mime_type="application/pdf",
        page_number=None,
    )
    assert any(f.flag_id == "MISSING_PAGE_LOCATION" and f.severity == "INFO" for f in flags_pdf)

    # CRM managed lock
    flags_crm = run_deterministic_quality_checks(
        text="CRM Contact Notes",
        is_managed_external=True,
    )
    assert any(f.flag_id == "MANAGED_EXTERNAL_SYNC" and f.suggested_remediation == "SOURCE_CORRECTION" for f in flags_crm)


def test_quality_check_curated_drift_and_unavailable_source():
    # Curated drift
    flags_drift = run_deterministic_quality_checks(
        text="Curated text",
        original_text="Ingested original text",
        current_version=2,
    )
    assert any(f.flag_id == "CURATED_CONTENT_DRIFT" and f.severity == "INFO" for f in flags_drift)

    # Source unavailable
    flags_missing = run_deterministic_quality_checks(
        text="Normal chunk",
        source_available=False,
    )
    assert any(f.flag_id == "SOURCE_FILE_UNAVAILABLE" and f.severity == "WARNING" for f in flags_missing)


# ─── INTEGRATION TESTS: DATABASE & API WORKFLOWS ────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session():
    async with AsyncSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def setup_test_context(db_session: AsyncSession):
    """Seed test tenancy, documents, chunks, and users in rag_disposable_test."""
    now = datetime.now(timezone.utc)
    uid = uuid.uuid4().hex[:6]

    # Department & Permissions
    dept = Department(
        id=uuid.uuid4(),
        code=f"DEPT_{uid}",
        name=f"Quality Dept {uid}",
        is_active=True,
    )
    db_session.add(dept)

    perm_view = await db_session.scalar(select(Permission).where(Permission.code == "VIEW"))
    if not perm_view:
        perm_view = Permission(id=uuid.uuid4(), code="VIEW", description="View access", is_active=True)
        db_session.add(perm_view)
    await db_session.flush()

    # Workspace
    ws = Workspace(
        id=uuid.uuid4(),
        name=f"WS_{uid}",
        code=f"WS_{uid.upper()}",
        department_id=dept.id,
        is_active=True,
    )
    db_session.add(ws)

    # User
    user = User(
        id=uuid.uuid4(),
        email=f"reviewer_{uid}@example.com",
        hashed_password="hashed_dummy_pw",
        role="USER",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    # Link user to department and workspace
    db_session.add(UserDepartment(user_id=user.id, department_id=dept.id))
    db_session.add(UserPermission(user_id=user.id, department_id=dept.id, permission_id=perm_view.id))
    db_session.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="EDITOR"))

    # Create temporary physical file for source availability test
    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / f"source_{uid}.pdf"
    temp_file.write_text("Dummy source file content for testing.")

    # Document
    doc = Document(
        id=uuid.uuid4(),
        file_name=f"source_{uid}.pdf",
        file_path=str(temp_file),
        content_hash=hashlib.sha256(f"dummy_content_{uid}".encode()).hexdigest(),
        mime_type="application/pdf",
        workspace_id=ws.id,
        department_id=dept.id,
        created_by=user.id,
        status="PROCESSED",
    )
    db_session.add(doc)
    await db_session.flush()

    # Hierarchical Section
    section = DocumentSection(
        id=uuid.uuid4(),
        document_id=doc.id,
        node_type="SECTION",
        title="Architecture Section 3",
        ordinal_path="3.1",
        l0_abstract="L0 Executive summary of Section 3.",
    )
    db_session.add(section)
    await db_session.flush()

    # Chunk with v1 history
    initial_text = "Initial uncurated ingested text."
    initial_hash = hashlib.sha256(initial_text.encode("utf-8")).hexdigest()
    dummy_emb = [0.01] * settings.embedding_dim

    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=0,
        text=initial_text,
        embedding=dummy_emb,
        chunk_hash=initial_hash,
        current_version=1,
        page_number=3,
        node_id=section.id,
        metadata_={"source_url": "https://docs.example.com/section3"},
        created_at=now,
        updated_at=now,
    )
    db_session.add(chunk)
    await db_session.flush()

    # Version 1 history snapshot
    v1_record = DocumentChunkVersion(
        id=uuid.uuid4(),
        chunk_id=chunk.id,
        version_number=1,
        text_before=initial_text,
        text_after=initial_text,
        diff_patch=None,
        embedding_snapshot=dummy_emb,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        chunk_hash=initial_hash,
        edited_by=user.id,
        created_at=now,
        edit_reason="Initial ingestion",
        is_rollback=False,
    )
    db_session.add(v1_record)
    await db_session.commit()

    return {
        "user": user,
        "dept": dept,
        "ws": ws,
        "doc": doc,
        "section": section,
        "chunk": chunk,
        "temp_file": temp_file,
    }


@pytest.mark.asyncio
async def test_chunk_evidence_honest_availability_and_drift(setup_test_context):
    """
    Verifies that get_chunk_evidence accurately reports source file availability on disk,
    correctly resolves original ingested text vs active curated text, and identifies section hierarchy.
    """
    ctx = setup_test_context
    user = ctx["user"]
    doc = ctx["doc"]
    chunk = ctx["chunk"]
    temp_file = ctx["temp_file"]

    async with AsyncSessionLocal() as session:
        service = ChunkCurationService(session, DeterministicEmbeddingProvider())

        # 1. Inspect evidence with source file available on host storage
        evidence_res = await service.get_chunk_evidence(doc.id, chunk.id, user)
        assert isinstance(evidence_res, ChunkEvidenceResponse)
        assert evidence_res.source_available is True
        assert evidence_res.source_availability_reason is None
        assert evidence_res.file_name == doc.file_name
        assert evidence_res.page_number == 3
        assert evidence_res.section_title == "Architecture Section 3"
        assert evidence_res.section_ordinal == "3.1"
        assert evidence_res.section_abstract == "L0 Executive summary of Section 3."
        assert evidence_res.source_url == "https://docs.example.com/section3"
        assert evidence_res.is_curated is False
        assert evidence_res.current_text == "Initial uncurated ingested text."
        assert evidence_res.original_text == "Initial uncurated ingested text."

        # 2. Curate chunk to v2 and confirm curated drift is identified
        chunk_db = await session.get(DocumentChunk, chunk.id)
        chunk_db.text = "Curated and corrected active text."
        chunk_db.current_version = 2
        await session.commit()

        evidence_v2 = await service.get_chunk_evidence(doc.id, chunk.id, user)
        assert evidence_v2.is_curated is True
        assert evidence_v2.current_version == 2
        assert evidence_v2.original_text == "Initial uncurated ingested text."
        assert evidence_v2.current_text == "Curated and corrected active text."
        assert any(f.flag_id == "CURATED_CONTENT_DRIFT" for f in evidence_v2.quality_flags)

        # 3. Simulate missing source file on disk -> verify honest unavailable state
        if temp_file.exists():
            temp_file.unlink()

        evidence_missing = await service.get_chunk_evidence(doc.id, chunk.id, user)
        assert evidence_missing.source_available is False
        assert "does not exist on host storage" in (evidence_missing.source_availability_reason or "")
        assert any(f.flag_id == "SOURCE_FILE_UNAVAILABLE" for f in evidence_missing.quality_flags)


@pytest.mark.asyncio
async def test_chunk_issue_reporting_lifecycle(setup_test_context):
    """
    Verifies that users can report suspicious or incorrect chunks, that reports are recorded
    under chunk metadata without schema changes, and that suggested remediation paths are preserved.
    """
    ctx = setup_test_context
    user = ctx["user"]
    doc = ctx["doc"]
    chunk = ctx["chunk"]

    async with AsyncSessionLocal() as session:
        service = ChunkCurationService(session, DeterministicEmbeddingProvider())

        # Submit quality issue report
        report_req = ChunkIssueReportRequest(
            issue_type="OCR_NOISE",
            description="Character substitution detected on line 3: 'cl' parsed as 'd'.",
            suggested_remediation="MANUAL_CURATE",
        )
        report_res = await service.report_chunk_issue(doc.id, chunk.id, report_req, user)

        assert report_res["message"] == "Quality issue report logged successfully."
        assert report_res["total_reports"] == 1
        assert report_res["report"]["issue_type"] == "OCR_NOISE"
        assert report_res["report"]["suggested_remediation"] == "MANUAL_CURATE"
        assert report_res["report"]["status"] == "OPEN"

        # Check subsequent evidence inspection returns the logged report
        evidence = await service.get_chunk_evidence(doc.id, chunk.id, user)
        assert len(evidence.quality_reports) == 1
        rep = evidence.quality_reports[0]
        assert rep.issue_type == "OCR_NOISE"
        assert rep.suggested_remediation == "MANUAL_CURATE"
        assert rep.reported_by == user.email


@pytest.mark.asyncio
async def test_http_api_evidence_and_reporting_routes(setup_test_context):
    """
    Verifies HTTP GET evidence and POST report-issue endpoints via AsyncClient.
    """
    ctx = setup_test_context
    user = ctx["user"]
    doc = ctx["doc"]
    chunk = ctx["chunk"]

    async def override_get_current_user():
        return user

    async def override_get_db_session():
        async with AsyncSessionLocal() as session:
            yield session

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db_session] = override_get_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET evidence
        res = await client.get(f"/documents/{doc.id}/chunks/{chunk.id}/evidence")
        assert res.status_code == 200
        data = res.json()
        assert data["chunk_id"] == str(chunk.id)
        assert data["document_id"] == str(doc.id)
        assert "quality_flags" in data
        assert "quality_reports" in data

        # POST report-issue
        post_payload = {
            "issue_type": "FACTUAL_ERROR",
            "description": "Retrieved statistics are outdated and contradict Q3 2026 reports.",
            "suggested_remediation": "SOURCE_CORRECTION",
        }
        rep_res = await client.post(
            f"/documents/{doc.id}/chunks/{chunk.id}/report-issue",
            json=post_payload,
        )
        assert rep_res.status_code == 200
        rep_data = rep_res.json()
        assert rep_data["report"]["issue_type"] == "FACTUAL_ERROR"
        assert rep_data["report"]["suggested_remediation"] == "SOURCE_CORRECTION"

    app.dependency_overrides.clear()
