"""
Milestone 10H Verification Test Suite: Artifact Preview Endpoint & HITL Guardrails.

Verifies:
1. Authenticated artifact preview retrieval by owner succeeds with SHA-256 integrity verification.
2. Cross-user/unauthorized access fails closed with HTTP 403.
3. Expired artifacts return HTTP 410 Gone.
4. Tampered content fails cryptographic integrity check with HTTP 500.
5. Binary artifacts (PDF, DOCX) return structured binary preview metadata without decoding errors.
6. Bounded preview truncation respects max_chars limit and sets is_truncated=True.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from rag_app.api.artifact_routes import preview_artifact
from rag_app.db.models import GeneratedArtifact, User
from rag_app.reports.storage import ArtifactStorageService
from rag_app.schemas.artifacts import ArtifactPreviewResponse


def create_test_user(
    user_id: uuid.UUID | None = None,
    role: str = "USER",
    is_active: bool = True,
) -> User:
    u = User()
    u.id = user_id or uuid.uuid4()
    u.email = f"user_{u.id}@example.com"
    u.role = role
    u.is_active = is_active
    return u


def build_mock_session(artifact: GeneratedArtifact | None = None, allowed_dept_ids: list[uuid.UUID] | None = None) -> AsyncMock:
    session = AsyncMock()
    session.get.return_value = artifact
    dept_ids = allowed_dept_ids or [uuid.uuid4()]
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = dept_ids
    session.scalars = AsyncMock(return_value=scalars_mock)
    return session


def create_test_artifact(
    user_id: uuid.UUID,
    format_type: str = "md",
    content_bytes: bytes = b"# Test Report\n\nAll metrics verified.",
    expires_at: datetime | None = None,
    tamper_hash: bool = False,
) -> tuple[GeneratedArtifact, str]:
    now = datetime.now(timezone.utc)
    storage = ArtifactStorageService()
    art_id = uuid.uuid4()
    storage_path, actual_hash, byte_size = storage.save_artifact(
        artifact_id=art_id,
        format_ext=format_type,
        content_bytes=content_bytes,
    )

    recorded_hash = "tampered_hash_value" if tamper_hash else actual_hash

    art = GeneratedArtifact()
    art.id = art_id
    art.user_id = user_id
    art.title = "Financial Summary Q3"
    art.format = format_type
    art.safe_filename = storage_path
    art.mime_type = "text/markdown" if format_type == "md" else "application/octet-stream"
    art.byte_size = byte_size
    art.storage_path = storage_path
    art.checksum_sha256 = recorded_hash
    art.source_departments = []
    art.expires_at = expires_at or (now + timedelta(hours=48))
    art.created_at = now
    art.download_count = 0

    return art, storage_path


@pytest.mark.asyncio
async def test_preview_artifact_success():
    owner = create_test_user()
    content = b"# Quarterly Overview\n\nRevenue grew by 24%."
    artifact, _ = create_test_artifact(user_id=owner.id, format_type="md", content_bytes=content)

    mock_session = build_mock_session(artifact)

    resp: ArtifactPreviewResponse = await preview_artifact(
        artifact_id=artifact.id,
        user=owner,
        session=mock_session,
        max_chars=50000,
    )

    assert resp.artifact_id == artifact.id
    assert resp.title == "Financial Summary Q3"
    assert resp.format == "md"
    assert resp.is_binary is False
    assert resp.is_truncated is False
    assert "Revenue grew by 24%" in resp.content
    assert resp.checksum_sha256 == hashlib.sha256(content).hexdigest()


@pytest.mark.asyncio
async def test_preview_artifact_unauthorized_fails_closed():
    owner = create_test_user()
    attacker = create_test_user()
    artifact, _ = create_test_artifact(user_id=owner.id)

    mock_session = build_mock_session(artifact)

    with pytest.raises(HTTPException) as exc_info:
        await preview_artifact(
            artifact_id=artifact.id,
            user=attacker,
            session=mock_session,
        )

    assert exc_info.value.status_code == 403
    assert "Access denied: You do not own this artifact" in exc_info.value.detail


@pytest.mark.asyncio
async def test_preview_artifact_expired_returns_410():
    owner = create_test_user()
    now = datetime.now(timezone.utc)
    expired_at = now - timedelta(hours=2)
    artifact, _ = create_test_artifact(user_id=owner.id, expires_at=expired_at)

    mock_session = build_mock_session(artifact)

    with pytest.raises(HTTPException) as exc_info:
        await preview_artifact(
            artifact_id=artifact.id,
            user=owner,
            session=mock_session,
        )

    assert exc_info.value.status_code == 410
    assert "Artifact has expired" in exc_info.value.detail


@pytest.mark.asyncio
async def test_preview_artifact_tampered_content_fails_integrity():
    owner = create_test_user()
    artifact, _ = create_test_artifact(user_id=owner.id, tamper_hash=True)

    mock_session = build_mock_session(artifact)

    with pytest.raises(HTTPException) as exc_info:
        await preview_artifact(
            artifact_id=artifact.id,
            user=owner,
            session=mock_session,
        )

    assert exc_info.value.status_code == 500
    assert "Integrity check failed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_preview_artifact_binary_format_handling():
    owner = create_test_user()
    binary_bytes = b"%PDF-1.4 Mock binary PDF content that is not raw text"
    artifact, _ = create_test_artifact(user_id=owner.id, format_type="pdf", content_bytes=binary_bytes)
    artifact.mime_type = "application/pdf"

    mock_session = build_mock_session(artifact)

    resp: ArtifactPreviewResponse = await preview_artifact(
        artifact_id=artifact.id,
        user=owner,
        session=mock_session,
    )

    assert resp.is_binary is True
    assert resp.is_truncated is False
    assert "[Binary artifact:" in resp.content
    assert "Please download to view full document" in resp.content


@pytest.mark.asyncio
async def test_preview_artifact_truncation_bound():
    owner = create_test_user()
    long_content = ("Line of report data with key metrics.\n" * 100).encode("utf-8")
    artifact, _ = create_test_artifact(user_id=owner.id, format_type="txt", content_bytes=long_content)
    artifact.mime_type = "text/plain"

    mock_session = build_mock_session(artifact)

    resp: ArtifactPreviewResponse = await preview_artifact(
        artifact_id=artifact.id,
        user=owner,
        session=mock_session,
        max_chars=200,
    )

    assert resp.is_truncated is True
    assert "[TRUNCATED: Exceeded preview limit of 200 characters]" in resp.content
    assert len(resp.content) < len(long_content)
