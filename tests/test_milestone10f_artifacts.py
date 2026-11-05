"""
Milestone 10F Verification Test Suite: Symbolic Execution Memory & Artifact Offloading.

Verifies:
1. Inline threshold evaluation: outputs >= 4000 chars trigger offloading.
2. Offloading generates SHA-256 hash, creates GeneratedArtifact record, and returns typed symbolic reference:
   artifact://{run_id}/{task_key}/{artifact_id}
3. Authorized retrieval by owner or workspace member succeeds.
4. Path traversal protection blocks traversal outside designated artifact directory.
5. Cryptographic integrity check detects file tampering and rejects altered artifacts.
6. Unauthorized access (cross-user/cross-workspace) fails closed.
7. Execution Canvas Mermaid generator produces valid Mermaid syntax with node dependencies.
"""

from __future__ import annotations

import os

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.db.models import GeneratedArtifact, User, Department, Workspace, WorkspaceMember
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.orchestration.artifact_service import (
    ArtifactOffloadingService,
    SymbolicArtifactReference,
    generate_execution_canvas_mermaid,
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_artifact_context():
    async with AsyncSessionLocal() as session:
        dept = Department(
            id=uuid.uuid4(),
            name="Artifact Dept",
            code=f"ART_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(dept)
        await session.flush()

        user1 = User(
            id=uuid.uuid4(),
            email=f"art_owner_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        user2 = User(
            id=uuid.uuid4(),
            email=f"art_other_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        session.add(user1)
        session.add(user2)
        await session.flush()

        workspace = Workspace(
            id=uuid.uuid4(),
            name="Artifact Workspace",
            code=f"art_ws_{uuid.uuid4().hex[:6]}",
            department_id=dept.id,
            is_active=True,
            created_by=user1.id,
        )
        session.add(workspace)
        await session.flush()

        session.add(WorkspaceMember(workspace_id=workspace.id, user_id=user1.id, role="OWNER"))
        await session.commit()

        return {
            "owner": user1,
            "other_user": user2,
            "workspace": workspace,
        }


# ==============================================================================
# Unit & Integration Tests: Artifact Offloading Service
# ==============================================================================

def test_should_offload_threshold_enforcement():
    service = ArtifactOffloadingService(char_threshold=4000)
    # Small payload
    small_text = "Short summary under 4000 characters."
    assert service.should_offload(small_text) is False

    # Large payload (5000 characters)
    large_text = "A" * 5000
    assert service.should_offload(large_text) is True


@pytest.mark.asyncio
async def test_offload_output_and_symbolic_reference_generation(setup_artifact_context):
    ctx = setup_artifact_context
    owner = ctx["owner"]
    ws = ctx["workspace"]
    run_id = uuid.uuid4()
    task_key = "extract_findings"

    with tempfile.TemporaryDirectory() as tmp_dir:
        service = ArtifactOffloadingService(storage_dir=tmp_dir, char_threshold=500)

        # 1000 characters payload
        large_content = "Extracted Fact: SLA agreement at 99.95%.\n" * 25

        async with AsyncSessionLocal() as session:
            ref = await service.offload_output(
                session=session,
                user_id=owner.id,
                content=large_content,
                title="Deep Fact Extraction Results",
                workspace_id=ws.id,
                run_id=run_id,
                task_key=task_key,
                format_type="md",
            )
            await session.commit()

            # Verify typed symbolic reference structure
            assert ref.is_offloaded is True
            assert ref.symbolic_uri.startswith(f"artifact://{run_id}/{task_key}/")
            assert len(ref.content_hash) == 64
            assert ref.size_bytes == len(large_content.encode("utf-8"))
            assert "99.95%" in ref.summary_snippet

            # Verify DB persistence in GeneratedArtifact table
            db_artifact = await session.get(GeneratedArtifact, uuid.UUID(ref.artifact_id))
            assert db_artifact is not None
            assert db_artifact.title == "Deep Fact Extraction Results"
            assert db_artifact.checksum_sha256 == ref.content_hash

            # Authorized retrieval by owner
            retrieved_content = await service.retrieve_artifact(
                session=session,
                user=owner,
                artifact_id=ref.artifact_id,
                workspace_id=ws.id,
            )
            assert retrieved_content == large_content


@pytest.mark.asyncio
async def test_artifact_tampering_fails_integrity_check(setup_artifact_context):
    ctx = setup_artifact_context
    owner = ctx["owner"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        service = ArtifactOffloadingService(storage_dir=tmp_dir, char_threshold=100)
        content = "Original authentic findings that must not be altered."

        async with AsyncSessionLocal() as session:
            ref = await service.offload_output(
                session=session,
                user_id=owner.id,
                content=content,
                title="Integrity Test Artifact",
            )
            await session.commit()

            # Tamper with file directly on disk
            db_artifact = await session.get(GeneratedArtifact, uuid.UUID(ref.artifact_id))
            disk_path = Path(db_artifact.storage_path)
            disk_path.write_text("TAMPERED malicious text injected into artifact file!", encoding="utf-8")

            # Retrieval must detect SHA-256 mismatch and fail closed
            with pytest.raises(ValueError) as exc:
                await service.retrieve_artifact(
                    session=session,
                    user=owner,
                    artifact_id=ref.artifact_id,
                )
            assert "Integrity check failed: Artifact content hash mismatch" in str(exc.value)


@pytest.mark.asyncio
async def test_path_traversal_protection(setup_artifact_context):
    ctx = setup_artifact_context
    owner = ctx["owner"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        service = ArtifactOffloadingService(storage_dir=tmp_dir, char_threshold=100)

        async with AsyncSessionLocal() as session:
            # Create artifact record pointing outside storage directory
            outside_artifact = GeneratedArtifact(
                id=uuid.uuid4(),
                user_id=owner.id,
                title="Traversal Test",
                format="txt",
                safe_filename="outside.txt",
                mime_type="text/plain",
                byte_size=12,
                storage_path=str(Path(tmp_dir).parent / "etc_passwd.txt"),  # Path traversal attempt!
                checksum_sha256="fakehash",
                expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            )
            session.add(outside_artifact)
            await session.commit()

            # Retrieval must reject path traversal attempt
            with pytest.raises(PermissionError) as exc:
                await service.retrieve_artifact(
                    session=session,
                    user=owner,
                    artifact_id=outside_artifact.id,
                )
            assert "Security alert: Attempted directory traversal" in str(exc.value)


@pytest.mark.asyncio
async def test_unauthorized_cross_user_access_blocked(setup_artifact_context):
    ctx = setup_artifact_context
    owner = ctx["owner"]
    other_user = ctx["other_user"]
    ws = ctx["workspace"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        service = ArtifactOffloadingService(storage_dir=tmp_dir, char_threshold=100)

        async with AsyncSessionLocal() as session:
            ref = await service.offload_output(
                session=session,
                user_id=owner.id,
                content="Private confidential owner findings.",
                title="Confidential Artifact",
                workspace_id=ws.id,
            )
            await session.commit()

            # Other user from a different workspace attempting to retrieve
            other_ws_id = uuid.uuid4()
            with pytest.raises(PermissionError) as exc:
                await service.retrieve_artifact(
                    session=session,
                    user=other_user,
                    artifact_id=ref.artifact_id,
                    workspace_id=other_ws_id,  # Mismatched workspace
                )
            assert "Access denied" in str(exc.value)


def test_execution_canvas_mermaid_generation():
    tasks = [
        {"task_key": "rag", "task_type": "RETRIEVAL", "status": "COMPLETED", "description": "RAG internal docs", "dependencies": []},
        {"task_key": "web", "task_type": "WEB_SEARCH", "status": "COMPLETED", "description": "Web research", "dependencies": []},
        {"task_key": "extract", "task_type": "EXTRACTION", "status": "COMPLETED", "description": "Fact extraction", "dependencies": ["rag", "web"]},
        {"task_key": "synth", "task_type": "SYNTHESIS", "status": "PENDING", "description": "Final synthesis", "dependencies": ["extract"]},
    ]
    mermaid = generate_execution_canvas_mermaid(tasks, run_id="run-12345678")

    assert "flowchart TD" in mermaid
    assert "subgraph Run_run-1234" in mermaid
    assert "rag --> extract" in mermaid
    assert "web --> extract" in mermaid
    assert "extract --> synth" in mermaid
    assert "✓ [RETRIEVAL] rag" in mermaid
    assert "⏳ [SYNTHESIS] synth" in mermaid
