"""
Milestone 7B: Living Articles & Staged Knowledge Updates API Router.
Endpoints for:
- Document version history
- Version detail and page diffs
- Shadow comparison inspection
- Safe version publishing
- Atomic rollback
- Incremental staged update processing
"""

from __future__ import annotations

import uuid
from typing import Annotated, Optional
from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import Document, DocumentVersion, ShadowIngestionReport, User
from rag_app.db.session import get_db_session
from rag_app.knowledge.living_article_service import LivingArticleService
from rag_app.schemas.living_articles import (
    DocumentVersionItem,
    DocumentVersionListResponse,
    PublishVersionResponse,
    RollbackVersionResponse,
    ShadowReportResponse,
)
from rag_app.security.authorization import authorize_document

router = APIRouter(prefix="/api/documents", tags=["living-articles"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


class RollbackRequest(BaseModel):
    target_version_id: uuid.UUID


@router.get("/{document_id}/versions", response_model=DocumentVersionListResponse)
async def list_document_versions(
    document_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
):
    """Lists all versions for a document with status and change summaries."""
    is_allowed = await authorize_document(session, user, "VIEW", document_id)
    if not is_allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this document")

    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    versions = list(
        await session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number.desc())
        )
    )

    items = [DocumentVersionItem.model_validate(v) for v in versions]
    return DocumentVersionListResponse(
        document_id=document_id,
        current_version_id=doc.current_version_id,
        versions=items,
    )


@router.get("/{document_id}/versions/{version_id}", response_model=DocumentVersionItem)
async def get_document_version(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
):
    """Retrieves a specific version record by ID."""
    is_allowed = await authorize_document(session, user, "VIEW", document_id)
    if not is_allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this document")

    ver = await session.get(DocumentVersion, version_id)
    if not ver or ver.document_id != document_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    return DocumentVersionItem.model_validate(ver)


@router.get("/{document_id}/versions/{version_id}/shadow-report", response_model=ShadowReportResponse)
async def get_shadow_report(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
):
    """Retrieves the shadow comparison report for a staged document version."""
    is_allowed = await authorize_document(session, user, "VIEW", document_id)
    if not is_allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this document")

    ver = await session.get(DocumentVersion, version_id)
    if not ver or ver.document_id != document_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    report = await session.scalar(
        select(ShadowIngestionReport)
        .where(ShadowIngestionReport.document_version_id == version_id)
        .order_by(ShadowIngestionReport.created_at.desc())
    )
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shadow report not found for this version")

    return ShadowReportResponse.model_validate(report)


@router.post("/{document_id}/versions/{version_id}/publish", response_model=PublishVersionResponse)
async def publish_staged_version(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
):
    """Atomically publishes a staged knowledge version."""
    is_allowed = await authorize_document(session, user, "EDIT", document_id)
    if not is_allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to publish versions for this document")

    service = LivingArticleService(session)
    try:
        published_version = await service.publish_version(version_id, user=user)
        return PublishVersionResponse(
            success=True,
            version_id=published_version.id,
            document_id=document_id,
            status=published_version.status,
            published_at=published_version.published_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{document_id}/versions/{version_id}/rollback", response_model=RollbackVersionResponse)
async def rollback_version_endpoint(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    payload: RollbackRequest,
    session: DbSession,
    user: User = Depends(get_current_user),
):
    """Rolls back knowledge state to a previous valid version."""
    is_allowed = await authorize_document(session, user, "EDIT", document_id)
    if not is_allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to rollback versions for this document")

    service = LivingArticleService(session)
    try:
        target_version = await service.rollback_version(
            current_version_id=version_id,
            target_version_id=payload.target_version_id,
            user=user,
        )
        return RollbackVersionResponse(
            success=True,
            document_id=document_id,
            rolled_back_version_id=version_id,
            active_version_id=target_version.id,
            status=target_version.status,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{document_id}/process-update", response_model=DocumentVersionItem)
async def process_document_update(
    document_id: uuid.UUID,
    session: DbSession,
    file: UploadFile = File(None),
    file_path: Optional[str] = Form(None),
    page_hint: Optional[int] = Form(None),
    force_full_reingest: bool = Form(False),
    user: User = Depends(get_current_user),
):
    """
    Submits an incremental or full update for an existing living document.
    Stages the version, detects changed pages, reuses unchanged chunks, and performs shadow verification.
    """
    is_allowed = await authorize_document(session, user, "EDIT", document_id)
    if not is_allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to edit this document")

    temp_path: Path | None = None
    target_path_str: str = ""

    if file:
        temp_dir = Path(tempfile.gettempdir()) / "living_articles_staging"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        target_path_str = str(temp_path)
    elif file_path:
        target_path_str = file_path
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide file upload or file_path")

    service = LivingArticleService(session)
    try:
        staged_ver = await service.execute_staged_update(
            document_id=document_id,
            new_file_path=target_path_str,
            page_hint=page_hint,
            force_full_reingest=force_full_reingest,
            user=user,
        )
        return DocumentVersionItem.model_validate(staged_ver)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

