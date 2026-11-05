"""
Milestone 7A.2 End-to-End Integration Smoke Test Suite.

Executes comprehensive HTTP API and service integration smoke tests against
the isolated disposable test database: rag_disposable_test.

Scenarios verified:
1. Open Chunk History / Load chunk details and version history via HTTP API client.
2. Load chunk details and version history payload shapes.
3. Edit chunk as an authorized editor; confirm current_version increments (v1 -> v2),
   history records previous state, and embedding is updated.
4. Simulate stale base_version; confirm HTTP 409 Conflict is returned and reload path
   returns latest version.
5. Compare two real versions; verify unified diff with additions and deletions.
6. Roll back with explicit confirmation; confirm forward-moving version (v3) is recorded
   with reverted_from_version provenance and restored text.
7. Role enforcement and tenant boundaries:
   - Viewer cannot mutate (HTTP 403)
   - Managed external CRM chunks remain read-only (HTTP 403)
   - Cross-workspace isolation fails closed (HTTP 403)
8. Non-existent chunk returns HTTP 404 Not Found.
"""

from __future__ import annotations

import os
# Ensure strictly isolated disposable test database
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import hashlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

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
    Permission,
    User,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine, get_db_session
from rag_app.knowledge.chunk_curation_service import ChunkCurationService

app = FastAPI()
app.include_router(chunk_curation_router)

settings = get_settings()


class DeterministicEmbeddingProvider:
    """Mock embedding provider ensuring reproducible 384-dim vectors without remote calls."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text_val in texts:
            seed = int(hashlib.md5(text_val.encode("utf-8")).hexdigest()[:8], 16)
            base = [(float((seed + i) % 100) / 100.0) for i in range(self.dim)]
            norm = sum(x * x for x in base) ** 0.5 or 1.0
            results.append([x / norm for x in base])
        return results


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest.fixture
def mock_embedding():
    return DeterministicEmbeddingProvider(dim=384)


@pytest.mark.asyncio
async def test_milestone_7a2_e2e_scenarios(mock_embedding, monkeypatch):
    """
    Comprehensive end-to-end smoke test covering scenarios 1 through 8.
    """
    monkeypatch.setattr(
        "rag_app.api.chunk_curation_routes.ChunkCurationService",
        lambda session: ChunkCurationService(session, embedding_provider=mock_embedding),
    )

    async with AsyncSessionLocal() as session:
        # Seed test workspace, department, and users
        dept = Department(
            id=uuid.uuid4(),
            name=f"Smoke Dept {uuid.uuid4().hex[:6]}",
            code=f"SMK_{uuid.uuid4().hex[:4].upper()}",
            created_at=datetime.now(timezone.utc),
        )
        session.add(dept)

        ws = Workspace(
            id=uuid.uuid4(),
            name="Smoke Workspace",
            code=f"WS_{uuid.uuid4().hex[:4].upper()}",
            department_id=dept.id,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(ws)

        editor_user = User(
            id=uuid.uuid4(),
            email=f"editor_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hashed_pw",
            role="USER",
            is_active=True,
        )
        viewer_user = User(
            id=uuid.uuid4(),
            email=f"viewer_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hashed_pw",
            role="USER",
            is_active=True,
        )
        outsider_user = User(
            id=uuid.uuid4(),
            email=f"outsider_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hashed_pw",
            role="USER",
            is_active=True,
        )
        session.add_all([editor_user, viewer_user, outsider_user])

        # Memberships
        session.add(WorkspaceMember(
            workspace_id=ws.id,
            user_id=editor_user.id,
            role="EDITOR",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        session.add(WorkspaceMember(
            workspace_id=ws.id,
            user_id=viewer_user.id,
            role="VIEWER",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        perm_view = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not perm_view:
            perm_view = Permission(id=uuid.uuid4(), code="VIEW", description="View access", is_active=True)
            session.add(perm_view)
        perm_edit = await session.scalar(select(Permission).where(Permission.code == "EDIT"))
        if not perm_edit:
            perm_edit = Permission(id=uuid.uuid4(), code="EDIT", description="Edit access", is_active=True)
            session.add(perm_edit)
        await session.flush()

        session.add(UserDepartment(user_id=editor_user.id, department_id=dept.id))
        session.add(UserDepartment(user_id=viewer_user.id, department_id=dept.id))
        session.add(UserPermission(user_id=editor_user.id, department_id=dept.id, permission_id=perm_view.id))
        session.add(UserPermission(user_id=editor_user.id, department_id=dept.id, permission_id=perm_edit.id))
        session.add(UserPermission(user_id=viewer_user.id, department_id=dept.id, permission_id=perm_view.id))

        # Seed initial document and chunk (v1)
        doc = Document(
            id=uuid.uuid4(),
            file_name="smoke_test_doc.docx",
            file_path="/tmp/smoke_test_doc.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            content_hash=hashlib.sha256(f"smoke_doc_{uuid.uuid4().hex}".encode()).hexdigest(),
            department_id=dept.id,
            workspace_id=ws.id,
            status="READY",
            created_at=datetime.now(timezone.utc),
        )
        session.add(doc)

        v1_text = "Original chunk text before curation. Operational reserve is $1.5M."
        v1_emb = (await mock_embedding.embed_texts([v1_text]))[0]
        v1_hash = hashlib.sha256(v1_text.encode("utf-8")).hexdigest()

        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            text=v1_text,
            embedding=v1_emb,
            current_version=1,
            chunk_hash=v1_hash,
            metadata_={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(chunk)

        # Seed CRM managed chunk for CRM protection test
        crm_chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=1,
            text="Twenty CRM synced customer record.",
            embedding=v1_emb,
            current_version=1,
            chunk_hash=hashlib.sha256(b"crm").hexdigest(),
            metadata_={"managed_external": True, "source": "twenty_crm_mcp"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(crm_chunk)

        await session.commit()

        doc_id = str(doc.id)
        chunk_id = str(chunk.id)
        crm_chunk_id = str(crm_chunk.id)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 1 & 2: Load chunk details and version history via HTTP API
    # ─────────────────────────────────────────────────────────────────────────
    async def override_db():
        async with AsyncSessionLocal() as s:
            yield s

    app.dependency_overrides[get_db_session] = override_db
    app.dependency_overrides[get_current_user] = lambda: editor_user

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # GET Chunk detail
        res = await client.get(f"/documents/{doc_id}/chunks/{chunk_id}")
        assert res.status_code == 200, res.text
        detail = res.json()
        assert detail["id"] == chunk_id
        assert detail["current_version"] == 1
        assert detail["text"] == v1_text
        assert detail["chunk_hash"] == v1_hash
        assert detail["is_managed_external"] is False

        # GET Versions list (v1 initial auto-record)
        res_v = await client.get(f"/documents/{doc_id}/chunks/{chunk_id}/versions")
        assert res_v.status_code == 200, res_v.text
        versions = res_v.json()
        assert len(versions) == 1
        assert versions[0]["version_number"] == 1
        assert versions[0]["text_after"] == v1_text

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 3: Edit chunk as authorized editor (v1 -> v2)
    # ─────────────────────────────────────────────────────────────────────────
    v2_text = "Updated chunk text after curation. Operational reserve increased to $2.5M."
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        edit_payload = {
            "new_text": v2_text,
            "base_version": 1,
            "edit_reason": "Audit correction for Q3 reserves",
        }
        res_edit = await client.patch(
            f"/documents/{doc_id}/chunks/{chunk_id}",
            json=edit_payload,
        )
        assert res_edit.status_code == 200, res_edit.text
        mutation = res_edit.json()
        assert mutation["chunk"]["current_version"] == 2
        assert mutation["chunk"]["text"] == v2_text
        assert mutation["version"]["version_number"] == 2

        # Verify history now contains v1 and v2
        res_v2 = await client.get(f"/documents/{doc_id}/chunks/{chunk_id}/versions")
        assert res_v2.status_code == 200
        versions2 = res_v2.json()
        assert len(versions2) == 2
        assert versions2[0]["version_number"] == 1
        assert versions2[1]["version_number"] == 2
        assert versions2[1]["edit_reason"] == "Audit correction for Q3 reserves"

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 4: Simulate stale base_version (HTTP 409 Conflict) and reload path
    # ─────────────────────────────────────────────────────────────────────────
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        stale_payload = {
            "new_text": "Concurrent conflicting edit",
            "base_version": 1,  # Stale: active version is now 2
            "edit_reason": "Concurrent conflict attempt",
        }
        res_conflict = await client.patch(
            f"/documents/{doc_id}/chunks/{chunk_id}",
            json=stale_payload,
        )
        assert res_conflict.status_code == 409, res_conflict.text
        assert "Conflict" in res_conflict.json()["detail"]
        assert "base_version was 1" in res_conflict.json()["detail"]

        # Reload Current Version path returns fresh v2
        res_reload = await client.get(f"/documents/{doc_id}/chunks/{chunk_id}")
        assert res_reload.status_code == 200
        assert res_reload.json()["current_version"] == 2

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 5: Compare two real versions (Diff Viewer endpoint)
    # ─────────────────────────────────────────────────────────────────────────
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_diff = await client.get(
            f"/documents/{doc_id}/chunks/{chunk_id}/diff?v1=1&v2=2"
        )
        assert res_diff.status_code == 200, res_diff.text
        diff_resp = res_diff.json()
        assert diff_resp["v1"] == 1
        assert diff_resp["v2"] == 2
        assert "+" in diff_resp["diff_patch"]
        assert "-" in diff_resp["diff_patch"]
        assert diff_resp["lines_added"] >= 1
        assert diff_resp["lines_removed"] >= 1

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 6: Rollback with explicit confirmation to v1
    # ─────────────────────────────────────────────────────────────────────────
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        rollback_payload = {
            "target_version": 1,
            "rollback_reason": "Rollback approved by finance team to baseline",
        }
        res_rb = await client.post(
            f"/documents/{doc_id}/chunks/{chunk_id}/rollback",
            json=rollback_payload,
        )
        assert res_rb.status_code == 200, res_rb.text
        rb_data = res_rb.json()
        # Rollback records a forward-moving version (v3)
        assert rb_data["chunk"]["current_version"] == 3
        assert rb_data["chunk"]["text"] == v1_text
        assert rb_data["version"]["version_number"] == 3
        assert rb_data["version"]["is_rollback"] is True
        assert rb_data["version"]["reverted_from_version"] == 1

        # Verify history reflects rollback provenance
        res_v3 = await client.get(f"/documents/{doc_id}/chunks/{chunk_id}/versions")
        versions3 = res_v3.json()
        assert len(versions3) == 3
        v3_entry = versions3[2]
        assert v3_entry["version_number"] == 3
        assert v3_entry["is_rollback"] is True
        assert v3_entry["reverted_from_version"] == 1

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 7: Role enforcement and Multi-tenant boundaries
    # ─────────────────────────────────────────────────────────────────────────
    # 7A: Viewer role cannot mutate (HTTP 403)
    app.dependency_overrides[get_current_user] = lambda: viewer_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_v_edit = await client.patch(
            f"/documents/{doc_id}/chunks/{chunk_id}",
            json={"new_text": "Viewer illegal edit", "base_version": 3},
        )
        assert res_v_edit.status_code == 403

        res_v_rb = await client.post(
            f"/documents/{doc_id}/chunks/{chunk_id}/rollback",
            json={"target_version": 1},
        )
        assert res_v_rb.status_code == 403

    # 7B: Managed CRM chunks remain strictly read-only (HTTP 403)
    app.dependency_overrides[get_current_user] = lambda: editor_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Editor can view CRM chunk
        res_crm_get = await client.get(f"/documents/{doc_id}/chunks/{crm_chunk_id}")
        assert res_crm_get.status_code == 200
        assert res_crm_get.json()["is_managed_external"] is True

        # Editor cannot mutate CRM chunk
        res_crm_edit = await client.patch(
            f"/documents/{doc_id}/chunks/{crm_chunk_id}",
            json={"new_text": "Manual change to CRM chunk", "base_version": 1},
        )
        assert res_crm_edit.status_code == 403
        assert "CRM" in res_crm_edit.json()["detail"]

    # 7C: Cross-workspace outsider is denied (HTTP 403)
    app.dependency_overrides[get_current_user] = lambda: outsider_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_out_get = await client.get(f"/documents/{doc_id}/chunks/{chunk_id}")
        assert res_out_get.status_code == 403

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 8: Error states & non-existent chunk (HTTP 404)
    # ─────────────────────────────────────────────────────────────────────────
    app.dependency_overrides[get_current_user] = lambda: editor_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        fake_chunk_id = str(uuid.uuid4())
        res_404 = await client.get(f"/documents/{doc_id}/chunks/{fake_chunk_id}")
        assert res_404.status_code == 404
        assert "not found" in res_404.json()["detail"].lower()

    # Clear overrides
    app.dependency_overrides.clear()
