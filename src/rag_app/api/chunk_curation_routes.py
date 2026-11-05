"""
API Endpoints for Milestone 7A.1: Chunk Curation & Rollback Lifecycle.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List
import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.db.models import User
from rag_app.db.session import get_db_session
from rag_app.knowledge.chunk_curation_service import ChunkCurationService
from rag_app.schemas.chunk_curation import (
    ChunkDetailResponse,
    ChunkDiffResponse,
    ChunkEditMutationResponse,
    ChunkEditRequest,
    ChunkEvidenceResponse,
    ChunkIssueReportRequest,
    ChunkRollbackRequest,
    ChunkVersionResponse,
)

router = APIRouter(prefix="/documents", tags=["chunk-curation"])

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.get(
    "/{document_id}/chunks/{chunk_id}",
    response_model=ChunkDetailResponse,
    status_code=status.HTTP_200_OK,
)
async def get_chunk_detail(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> ChunkDetailResponse:
    """Fetch chunk details including current version, hash, and managed external flag."""
    service = ChunkCurationService(session)
    return await service.get_chunk_detail(document_id, chunk_id, user)


@router.get(
    "/{document_id}/chunks/{chunk_id}/versions",
    response_model=List[ChunkVersionResponse],
    status_code=status.HTTP_200_OK,
)
async def list_chunk_versions(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> List[ChunkVersionResponse]:
    """List all immutable version snapshots for a chunk."""
    service = ChunkCurationService(session)
    return await service.list_chunk_versions(document_id, chunk_id, user)


@router.get(
    "/{document_id}/chunks/{chunk_id}/diff",
    response_model=ChunkDiffResponse,
    status_code=status.HTTP_200_OK,
)
async def get_chunk_diff(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    v1: int = Query(..., ge=1, description="Source version number"),
    v2: int = Query(..., ge=1, description="Target version number"),
) -> ChunkDiffResponse:
    """Compare two chunk versions and return unified diff with change metrics."""
    service = ChunkCurationService(session)
    return await service.get_chunk_diff(document_id, chunk_id, v1, v2, user)


@router.patch(
    "/{document_id}/chunks/{chunk_id}",
    response_model=ChunkEditMutationResponse,
    status_code=status.HTTP_200_OK,
)
async def edit_chunk(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    payload: ChunkEditRequest,
    user: CurrentUser,
    session: DbSession,
) -> ChunkEditMutationResponse:
    """
    Curate chunk text with atomic embedding recalculation and optimistic concurrency.
    Fails closed with 409 Conflict if base_version does not match active version.
    """
    service = ChunkCurationService(session)
    chunk, version = await service.edit_chunk(document_id, chunk_id, payload, user)
    return ChunkEditMutationResponse(
        chunk=ChunkDetailResponse.model_validate(chunk),
        version=ChunkVersionResponse.model_validate(version),
    )


@router.post(
    "/{document_id}/chunks/{chunk_id}/rollback",
    response_model=ChunkEditMutationResponse,
    status_code=status.HTTP_200_OK,
)
async def rollback_chunk(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    payload: ChunkRollbackRequest,
    user: CurrentUser,
    session: DbSession,
) -> ChunkEditMutationResponse:
    """
    Roll back chunk to an existing historical version.
    Creates a new forward-moving version entry while preserving all prior history.
    """
    service = ChunkCurationService(session)
    chunk, version = await service.rollback_chunk(document_id, chunk_id, payload, user)
    return ChunkEditMutationResponse(
        chunk=ChunkDetailResponse.model_validate(chunk),
        version=ChunkVersionResponse.model_validate(version),
    )


@router.get(
    "/{document_id}/chunks/{chunk_id}/evidence",
    response_model=ChunkEvidenceResponse,
    status_code=status.HTTP_200_OK,
)
async def get_chunk_evidence(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
) -> ChunkEvidenceResponse:
    """
    Inspect source evidence for a chunk (Milestone 7A.3).
    Returns honest availability state, original vs curated text, section location, and deterministic quality flags.
    """
    service = ChunkCurationService(session)
    return await service.get_chunk_evidence(document_id, chunk_id, user)


@router.post(
    "/{document_id}/chunks/{chunk_id}/report-issue",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def report_chunk_issue(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    payload: ChunkIssueReportRequest,
    user: CurrentUser,
    session: DbSession,
) -> Dict[str, Any]:
    """
    Report a quality issue on a chunk or source (Milestone 7A.3).
    Stores issue report under chunk metadata with suggested remediation path.
    """
    service = ChunkCurationService(session)
    return await service.report_chunk_issue(document_id, chunk_id, payload, user)
