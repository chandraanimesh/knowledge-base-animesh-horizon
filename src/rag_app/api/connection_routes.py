import logging
import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user
from rag_app.core.config import get_settings
from rag_app.db.models import ConnectionDiscoveryJob, User
from rag_app.db.session import get_db_session
from rag_app.knowledge.connection_service import ConnectionDiscoveryService
from rag_app.knowledge.connection_worker import enqueue_discovery_job
from rag_app.schemas.connections import (
    ConnectionCandidateResponse,
    ConnectionDiscoverRequest,
    ConnectionJobResponse,
    ConnectionReviewRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge/connections", tags=["knowledge-connections"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]
settings = get_settings()


def require_feature_enabled():
    if not settings.enable_connection_discovery:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Connection discovery feature is disabled",
        )


@router.post(
    "/discover",
    response_model=ConnectionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue an asynchronous knowledge connection discovery job for an authorized item",
)
async def discover_connections(
    request: ConnectionDiscoverRequest,
    session: DbSession,
    user: CurrentUser,
) -> ConnectionJobResponse:
    require_feature_enabled()
    service = ConnectionDiscoveryService(session)

    # 1. Validate caller has access to source item and obtain current version hash
    _, version_hash, _, _ = await service.resolve_source_and_version(
        source_id=request.source_item_id,
        source_type=request.source_item_type,
        user=user,
    )

    # 2. Enqueue discovery job
    job = await enqueue_discovery_job(
        session=session,
        source_item_id=request.source_item_id,
        source_item_type=request.source_item_type,
        requesting_user_id=user.id,
        target_source_hash=version_hash,
    )

    return ConnectionJobResponse(
        id=job.id,
        source_item_id=job.source_item_id,
        source_item_type=job.source_item_type,
        target_source_hash=job.target_source_hash,
        requesting_user_id=job.requesting_user_id,
        status=job.status,
        attempt_count=job.attempt_count,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=ConnectionJobResponse,
    summary="Retrieve operational status of a connection discovery job",
)
async def get_discovery_job_status(
    job_id: uuid.UUID,
    session: DbSession,
    user: CurrentUser,
) -> ConnectionJobResponse:
    require_feature_enabled()
    job = await session.get(ConnectionDiscoveryJob, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Discovery job {job_id} not found",
        )

    if user.role != "ADMIN" and job.requesting_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    return ConnectionJobResponse(
        id=job.id,
        source_item_id=job.source_item_id,
        source_item_type=job.source_item_type,
        target_source_hash=job.target_source_hash,
        requesting_user_id=job.requesting_user_id,
        status=job.status,
        attempt_count=job.attempt_count,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get(
    "",
    response_model=List[ConnectionCandidateResponse],
    summary="List authorized knowledge connection candidates",
)
async def list_connections(
    session: DbSession,
    user: CurrentUser,
    source_id: Optional[uuid.UUID] = Query(None, description="Filter by source item ID"),
    lifecycle_state: Optional[str] = Query(None, description="Filter by state (SUGGESTED, CONFIRMED, etc.)"),
    connection_type: Optional[str] = Query(None, description="Filter by connection type"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> List[ConnectionCandidateResponse]:
    require_feature_enabled()
    service = ConnectionDiscoveryService(session)
    return await service.list_candidates(
        user=user,
        source_id=source_id,
        lifecycle_state=lifecycle_state,
        connection_type=connection_type,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{candidate_id}",
    response_model=ConnectionCandidateResponse,
    summary="Retrieve details and evidence for an authorized connection candidate",
)
async def get_connection(
    candidate_id: uuid.UUID,
    session: DbSession,
    user: CurrentUser,
) -> ConnectionCandidateResponse:
    require_feature_enabled()
    service = ConnectionDiscoveryService(session)
    return await service.get_candidate(candidate_id=candidate_id, user=user)


@router.post(
    "/{candidate_id}/review",
    response_model=ConnectionCandidateResponse,
    summary="Submit curator review for a connection candidate (CONFIRMED, REJECTED, CONTRADICTED)",
)
async def review_connection(
    candidate_id: uuid.UUID,
    review_request: ConnectionReviewRequest,
    session: DbSession,
    user: CurrentUser,
) -> ConnectionCandidateResponse:
    require_feature_enabled()
    service = ConnectionDiscoveryService(session)
    return await service.review_candidate(
        candidate_id=candidate_id,
        user=user,
        request=review_request,
    )
