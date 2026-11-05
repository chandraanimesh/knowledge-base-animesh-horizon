"""API router for multi-format RAG artifacts download and regeneration."""

from datetime import datetime, timezone, timedelta
import logging
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.core.config import settings
from rag_app.db.models import GeneratedArtifact, User
from rag_app.db.session import get_db_session
from rag_app.reports.registry import renderer_registry
from rag_app.reports.storage import ArtifactStorageService
from rag_app.schemas.artifacts import (
    ArtifactMetadata,
    ArtifactPreviewResponse,
    DownloadableArtifactFormat,
    TabularDataset,
)
from rag_app.security.authorization import get_allowed_departments

from fastapi.security import OAuth2PasswordBearer
from fastapi import Query
import jwt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])
oauth2_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

async def get_user_from_header_or_query(
    header_token: str | None = Depends(oauth2_optional),
    query_token: str | None = Query(None, alias="token"),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    raw_token = header_token or query_token
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    clean_token = raw_token.replace("Bearer ", "").replace('"', '').strip()
    try:
        payload = jwt.decode(
            clean_token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        user_uuid = uuid.UUID(user_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await session.get(User, user_uuid)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive or not found")
    return user

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_user_from_header_or_query)]


async def _verify_artifact_access(
    artifact: GeneratedArtifact,
    user: User,
    session: AsyncSession,
) -> None:
    """
    Enforces strict access policy:
    1. Tenant/Workspace boundary
    2. Dynamic department-level permissions (fails closed if permission was revoked)
    3. User/Admin ownership
    """
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    # 1. Superadmin has broad administrative access, but workspace-scoped checks still apply
    if user.role != "ADMIN" and artifact.user_id != user.id:
        # If user is not admin and not the creator, deny access
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not own this artifact.",
        )

    # 2. Workspace boundary check (Admin cannot bypass deleted or non-existent workspaces)
    if artifact.workspace_id and session is not None:
        from rag_app.workspace.service import WorkspaceService
        ws_service = WorkspaceService(session)
        try:
            await ws_service.get_workspace(user, artifact.workspace_id)
        except HTTPException as he:
            raise he
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Workspace boundary violation.",
            )

    # 3. Dynamic Department Permission Re-evaluation (Fail Closed)
    if user.role != "ADMIN" and session is not None:
        allowed_depts = await get_allowed_departments(session, user)
        if allowed_depts is not None:
            allowed_dept_strs = {str(d) for d in allowed_depts}
            if artifact.source_departments:
                for source_dept in artifact.source_departments:
                    # If any source department is no longer permitted, revoke download!
                    if str(source_dept) not in allowed_dept_strs and source_dept != "None":
                        logger.warning(
                            f"Artifact access denied: permission to department {source_dept} was revoked for user {user.id}"
                        )
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Access denied: permission to one or more source departments in this artifact has been revoked.",
                        )
            elif not allowed_dept_strs and not artifact.workspace_id:
                # User has zero department permissions
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: User has no active department permissions.",
                )


@router.get("/{artifact_id}/metadata", response_model=ArtifactMetadata)
async def get_artifact_metadata(
    artifact_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> ArtifactMetadata:
    """Fetch authoritative metadata for a generated artifact."""
    artifact = await session.get(GeneratedArtifact, artifact_id)
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found.",
        )

    await _verify_artifact_access(artifact, user, session)

    now = datetime.now(timezone.utc)
    is_expired = artifact.expires_at < now

    return ArtifactMetadata(
        id=artifact.id,
        format=DownloadableArtifactFormat(artifact.format),
        filename=artifact.safe_filename,
        mime_type=artifact.mime_type,
        size_bytes=artifact.byte_size,
        download_url=f"/api/reports/{artifact.id}/download",
        created_at=artifact.created_at,
        expires_at=artifact.expires_at,
        is_expired=is_expired,
    )


@router.get("/{artifact_id}/download")
async def download_artifact(
    artifact_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> FileResponse:
    """
    Downloads an authenticated, server-verified artifact.
    Re-checks dynamic department permissions and expiration TTL.
    """
    artifact = await session.get(GeneratedArtifact, artifact_id)
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found.",
        )

    await _verify_artifact_access(artifact, user, session)

    now = datetime.now(timezone.utc)
    if artifact.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Artifact has expired. Please use the regenerate endpoint.",
        )

    storage = ArtifactStorageService()
    try:
        abs_path = storage.get_absolute_path(artifact.storage_path)
    except (FileNotFoundError, ValueError) as err:
        logger.error(f"Failed to locate artifact binary on disk: {err}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file is missing or inaccessible.",
        )

    # Record download audit
    artifact.download_count += 1
    artifact.last_downloaded_at = now
    await session.commit()

    return FileResponse(
        path=str(abs_path),
        media_type=artifact.mime_type,
        filename=artifact.safe_filename,
    )


@router.get("/{artifact_id}/preview", response_model=ArtifactPreviewResponse)
async def preview_artifact(
    artifact_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    max_chars: int = Query(50000, ge=100, le=200000),
) -> ArtifactPreviewResponse:
    """
    Authenticated preview endpoint for execution and report artifacts.
    Enforces authorization, dynamic permissions, path traversal guard,
    and cryptographic SHA-256 integrity verification.
    """
    artifact = await session.get(GeneratedArtifact, artifact_id)
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found.",
        )

    await _verify_artifact_access(artifact, user, session)

    now = datetime.now(timezone.utc)
    if artifact.expires_at and artifact.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Artifact has expired. Please use the regenerate endpoint.",
        )

    storage = ArtifactStorageService()
    try:
        raw_bytes = storage.read_artifact(artifact.storage_path)
    except (FileNotFoundError, ValueError) as err:
        logger.error(f"Failed to read artifact binary on disk: {err}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact file is missing or inaccessible.",
        )

    import hashlib
    actual_hash = hashlib.sha256(raw_bytes).hexdigest()
    if artifact.checksum_sha256 and actual_hash != artifact.checksum_sha256:
        logger.error(f"Integrity check failed for artifact {artifact_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Integrity check failed: Artifact content hash mismatch.",
        )

    fmt_lower = (artifact.format or "").lower().lstrip(".")
    is_binary = fmt_lower in ("pdf", "docx", "xlsx", "zip", "bin", "png", "jpg", "jpeg")

    if is_binary:
        content = f"[Binary artifact: {artifact.safe_filename} ({artifact.byte_size} bytes, format: {fmt_lower.upper()}). Please download to view full document.]"
        is_truncated = False
    else:
        try:
            full_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            full_text = raw_bytes.decode("latin-1", errors="replace")

        if len(full_text) > max_chars:
            content = full_text[:max_chars] + f"\n\n[TRUNCATED: Exceeded preview limit of {max_chars} characters]"
            is_truncated = True
        else:
            content = full_text
            is_truncated = False

    return ArtifactPreviewResponse(
        artifact_id=artifact.id,
        title=artifact.title,
        format=artifact.format,
        mime_type=artifact.mime_type,
        content=content,
        byte_size=artifact.byte_size,
        checksum_sha256=actual_hash,
        is_truncated=is_truncated,
        is_binary=is_binary,
        expires_at=artifact.expires_at,
    )



@router.post("/{artifact_id}/regenerate", response_model=ArtifactMetadata)
async def regenerate_artifact(
    artifact_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> ArtifactMetadata:
    """
    Regenerates an expired or missing artifact from retained structured dataset
    WITHOUT reinvoking the LLM. Re-evaluates current user permissions.
    """
    artifact = await session.get(GeneratedArtifact, artifact_id)
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found.",
        )

    await _verify_artifact_access(artifact, user, session)

    if not artifact.tabular_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot regenerate: structured dataset is not preserved in artifact record.",
        )

    try:
        dataset = TabularDataset.model_validate(artifact.tabular_data)
        renderer = renderer_registry.get_renderer(artifact.format)
        content_bytes = renderer.render(
            dataset=dataset,
            title=artifact.title,
            summary=None,
        )
    except Exception as exc:
        logger.error(f"Failed to re-render artifact from structured dataset: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to re-render artifact binary.",
        )

    storage = ArtifactStorageService()
    storage_path, sha256_hash, byte_size = storage.save_artifact(
        artifact_id=artifact.id,
        format_ext=renderer.file_extension,
        content_bytes=content_bytes,
    )

    now = datetime.now(timezone.utc)
    new_expires_at = now + timedelta(hours=settings.artifact_retention_hours)

    artifact.storage_path = storage_path
    artifact.checksum_sha256 = sha256_hash
    artifact.byte_size = byte_size
    artifact.expires_at = new_expires_at
    await session.commit()

    return ArtifactMetadata(
        id=artifact.id,
        format=DownloadableArtifactFormat(artifact.format),
        filename=artifact.safe_filename,
        mime_type=artifact.mime_type,
        size_bytes=byte_size,
        download_url=f"/api/reports/{artifact.id}/download",
        created_at=artifact.created_at,
        expires_at=new_expires_at,
        is_expired=False,
    )
