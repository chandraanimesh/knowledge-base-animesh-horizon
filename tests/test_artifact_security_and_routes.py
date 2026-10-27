"""Security and lifecycle tests for artifact storage, permissions, and download/regeneration."""

from datetime import datetime, timezone, timedelta
import uuid
import pytest
from fastapi import HTTPException

from rag_app.api.artifact_routes import _verify_artifact_access
from rag_app.db.models import GeneratedArtifact, User
from rag_app.reports.storage import ArtifactStorageService
from rag_app.schemas.artifacts import (
    ColumnDataType,
    CoverageReport,
    ExportTruncation,
    RetrievalCoverage,
    TableColumn,
    TabularDataset,
)


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


def create_test_artifact(
    user_id: uuid.UUID,
    source_departments: list[str] | None = None,
    expires_at: datetime | None = None,
    tabular_data: dict | None = None,
) -> GeneratedArtifact:
    now = datetime.now(timezone.utc)
    a = GeneratedArtifact()
    a.id = uuid.uuid4()
    a.user_id = user_id
    a.title = "Test Financial Report"
    a.format = "pdf"
    a.safe_filename = "Test_Report.pdf"
    a.mime_type = "application/pdf"
    a.byte_size = 1024
    a.storage_path = f"{a.id}.pdf"
    a.checksum_sha256 = "dummy_sha256"
    a.source_departments = source_departments or []
    a.tabular_data = tabular_data
    a.expires_at = expires_at or (now + timedelta(hours=48))
    a.download_count = 0
    a.created_at = now
    return a


@pytest.mark.asyncio
async def test_cross_user_isolation_denied():
    user_owner = create_test_user()
    user_attacker = create_test_user()
    
    artifact = create_test_artifact(user_id=user_owner.id)

    # user_attacker tries to access user_owner's artifact
    with pytest.raises(HTTPException) as exc_info:
        await _verify_artifact_access(artifact, user_attacker, session=None)
    
    assert exc_info.value.status_code == 403
    assert "You do not own this artifact" in exc_info.value.detail


@pytest.mark.asyncio
async def test_inactive_user_denied():
    user = create_test_user(is_active=False)
    artifact = create_test_artifact(user_id=user.id)

    with pytest.raises(HTTPException) as exc_info:
        await _verify_artifact_access(artifact, user, session=None)

    assert exc_info.value.status_code == 403
    assert "inactive" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_admin_can_access():
    user_owner = create_test_user()
    admin_user = create_test_user(role="ADMIN")
    artifact = create_test_artifact(user_id=user_owner.id)

    # Admin should pass access check
    await _verify_artifact_access(artifact, admin_user, session=None)


def test_path_traversal_rejection(tmp_path):
    storage = ArtifactStorageService(base_dir=str(tmp_path))
    
    # Save a normal artifact
    art_id = uuid.uuid4()
    filename, checksum, size = storage.save_artifact(
        artifact_id=art_id,
        format_ext="json",
        content_bytes=b'{"test": true}',
    )
    assert filename == f"{art_id}.json"
    assert storage.read_artifact(filename) == b'{"test": true}'

    # Attempt path traversal
    with pytest.raises(ValueError) as exc_info:
        storage._resolve_safe_path("../../windows/system32/cmd.exe")
    assert "Path traversal detected" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        storage.read_artifact("../other_dir/secret.txt")
    assert "Path traversal detected" in str(exc_info.value)


def test_artifact_regeneration_flow(tmp_path):
    storage = ArtifactStorageService(base_dir=str(tmp_path))
    
    # Prepare dataset
    dataset = TabularDataset(
        title="Q1 Expenses",
        columns=[
            TableColumn(key="item", label="Item", data_type=ColumnDataType.STRING),
            TableColumn(key="cost", label="Cost", data_type=ColumnDataType.NUMBER),
        ],
        rows=[{"item": "Server", "cost": 500.0}],
    )

    from rag_app.reports.registry import renderer_registry
    renderer = renderer_registry.get_renderer("csv")
    csv_bytes = renderer.render(dataset, title=dataset.title)

    art_id = uuid.uuid4()
    storage_path, sha, size = storage.save_artifact(art_id, "csv", csv_bytes)

    assert storage.read_artifact(storage_path).startswith(b"\xef\xbb\xbfItem,Cost")


@pytest.mark.asyncio
async def test_revoked_department_denies_access():
    from unittest.mock import AsyncMock, patch

    user_owner = create_test_user()
    revoked_dept_id = uuid.uuid4()
    active_dept_id = uuid.uuid4()

    # Artifact was generated from data in revoked_dept_id
    artifact = create_test_artifact(
        user_id=user_owner.id,
        source_departments=[str(revoked_dept_id)],
    )

    mock_session = AsyncMock()
    with patch("rag_app.api.artifact_routes.get_allowed_departments", return_value=[active_dept_id]):
        with pytest.raises(HTTPException) as exc_info:
            await _verify_artifact_access(artifact, user_owner, session=mock_session)
        assert exc_info.value.status_code == 403
        assert "revoked" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_zero_department_permissions_denies_access():
    from unittest.mock import AsyncMock, patch

    user_owner = create_test_user()
    artifact = create_test_artifact(user_id=user_owner.id, source_departments=[])

    mock_session = AsyncMock()
    with patch("rag_app.api.artifact_routes.get_allowed_departments", return_value=[]):
        with pytest.raises(HTTPException) as exc_info:
            await _verify_artifact_access(artifact, user_owner, session=mock_session)
        assert exc_info.value.status_code == 403
        assert "no active department permissions" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_admin_cannot_bypass_deleted_workspace():
    from unittest.mock import AsyncMock, patch

    admin_user = create_test_user(role="ADMIN")
    deleted_ws_id = uuid.uuid4()
    artifact = create_test_artifact(user_id=admin_user.id)
    artifact.workspace_id = deleted_ws_id

    mock_session = AsyncMock()
    with patch("rag_app.workspace.service.WorkspaceService.get_workspace", side_effect=HTTPException(status_code=404, detail="Workspace not found")):
        with pytest.raises(HTTPException) as exc_info:
            await _verify_artifact_access(artifact, admin_user, session=mock_session)
        assert exc_info.value.status_code == 404
        assert "Workspace not found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_expired_artifact_download_raises_410():
    from unittest.mock import AsyncMock, patch
    from rag_app.api.artifact_routes import download_artifact

    user_owner = create_test_user()
    now = datetime.now(timezone.utc)
    expired_time = now - timedelta(hours=2)
    artifact = create_test_artifact(
        user_id=user_owner.id,
        expires_at=expired_time,
    )

    mock_session = AsyncMock()
    mock_session.get.return_value = artifact

    with patch("rag_app.api.artifact_routes.get_allowed_departments", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            await download_artifact(artifact.id, user_owner, session=mock_session)
        assert exc_info.value.status_code == 410
        assert "expired" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_artifact_metadata_returns_expiration_status():
    from unittest.mock import AsyncMock, patch
    from rag_app.api.artifact_routes import get_artifact_metadata

    user_owner = create_test_user()
    now = datetime.now(timezone.utc)
    expired_time = now - timedelta(hours=1)
    artifact = create_test_artifact(
        user_id=user_owner.id,
        expires_at=expired_time,
    )

    mock_session = AsyncMock()
    mock_session.get.return_value = artifact

    with patch("rag_app.api.artifact_routes.get_allowed_departments", return_value=None):
        meta = await get_artifact_metadata(artifact.id, user_owner, session=mock_session)
        assert meta.id == artifact.id
        assert meta.is_expired is True


@pytest.mark.asyncio
async def test_successful_downloads_all_formats(tmp_path):
    from unittest.mock import AsyncMock, patch
    from rag_app.api.artifact_routes import download_artifact
    from rag_app.reports.registry import renderer_registry
    from rag_app.schemas.artifacts import DownloadableArtifactFormat

    user_owner = create_test_user()
    storage = ArtifactStorageService(base_dir=str(tmp_path))

    columns = [
        TableColumn(key="name", label="Name", data_type=ColumnDataType.STRING),
        TableColumn(key="amount", label="Amount", data_type=ColumnDataType.NUMBER),
    ]
    rows = [{"name": "Sales Q3", "amount": 95000.0}]
    dataset = TabularDataset(title="Sales Summary", columns=columns, rows=rows)

    format_expectations = [
        ("pdf", "application/pdf", b"%PDF-"),
        ("csv", "text/csv; charset=utf-8", b"\xef\xbb\xbfName,Amount"),
        ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"PK\x03\x04"),
        ("json", "application/json", b"\"title\": \"Sales Summary\""),
    ]

    for fmt_str, expected_mime, magic_bytes in format_expectations:
        renderer = renderer_registry.get_renderer(fmt_str)
        content_bytes = renderer.render(dataset, title=dataset.title)

        art_id = uuid.uuid4()
        storage_path, sha, size = storage.save_artifact(
            artifact_id=art_id,
            format_ext=renderer.file_extension,
            content_bytes=content_bytes,
        )

        artifact = create_test_artifact(user_id=user_owner.id)
        artifact.id = art_id
        artifact.format = fmt_str
        artifact.mime_type = expected_mime
        artifact.storage_path = storage_path
        artifact.byte_size = size

        mock_session = AsyncMock()
        mock_session.get.return_value = artifact

        with patch("rag_app.api.artifact_routes.ArtifactStorageService", return_value=storage):
            with patch("rag_app.api.artifact_routes.get_allowed_departments", return_value=None):
                file_resp = await download_artifact(artifact.id, user_owner, session=mock_session)
                assert file_resp.media_type == expected_mime
                assert file_resp.filename == artifact.safe_filename
                # Verify physical file content
                file_bytes = open(file_resp.path, "rb").read()
                assert len(file_bytes) == size
                assert magic_bytes in file_bytes


def test_failed_artifact_metadata_error_handling():
    from rag_app.schemas.artifacts import DownloadableArtifactFormat, ArtifactMetadata

    # When generation fails, ArtifactMetadata captures error without fake download URL
    failed_meta = ArtifactMetadata(
        id=uuid.uuid4(),
        format=DownloadableArtifactFormat.PDF,
        filename="report.pdf",
        mime_type="application/octet-stream",
        size_bytes=0,
        download_url="",
        created_at=datetime.now(timezone.utc),
        error="Failed to generate PDF: Missing font glyph",
    )
    assert failed_meta.error is not None
    assert "Missing font" in failed_meta.error
    assert failed_meta.download_url == ""


@pytest.mark.asyncio
async def test_get_conversation_populates_artifacts_on_messages():
    from unittest.mock import AsyncMock, patch, MagicMock
    from rag_app.conversation.service import ConversationService
    from rag_app.db.models import Conversation, ChatMessage

    user = create_test_user()
    conv_id = uuid.uuid4()
    msg_id = uuid.uuid4()

    conv = Conversation()
    conv.id = conv_id
    conv.title = "Sales Multi-Format Thread"
    conv.workspace_id = None
    conv.created_at = datetime.now(timezone.utc)
    conv.updated_at = datetime.now(timezone.utc)

    msg = ChatMessage()
    msg.id = msg_id
    msg.conversation_id = conv_id
    msg.role = "assistant"
    msg.content = "Here are the sales records."
    msg.visualization = None
    msg.sources = []
    msg.execution_metrics = None
    msg.created_at = datetime.now(timezone.utc)

    conv.messages = [msg]

    art = create_test_artifact(user_id=user.id)
    art.conversation_id = conv_id
    art.message_id = msg_id

    mock_session = AsyncMock()
    mock_session.scalar.return_value = conv

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [art]
    mock_session.scalars.return_value = mock_scalars

    service = ConversationService(mock_session)
    with patch.object(service, "_require_global_chat_admin"):
        detail = await service.get_conversation_detail(conv_id, user)
        assert len(detail.messages) == 1
        asst_msg = detail.messages[0]
        assert asst_msg.artifacts is not None
        assert len(asst_msg.artifacts) == 1
        assert asst_msg.artifacts[0].id == art.id
        assert asst_msg.artifacts[0].download_url == f"/api/reports/{art.id}/download"
