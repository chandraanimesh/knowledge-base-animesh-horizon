import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.api.dependencies.auth import get_current_user, require_admin
from rag_app.db.models import User
from rag_app.db.session import get_db_session
from rag_app.knowledge.candidate_service import KnowledgeCandidateService
from rag_app.knowledge.company_notes_service import CompanyNotesService
from rag_app.schemas.knowledge import (
    BatchPublicationRequest,
    BatchPublicationResponse,
    CandidateCreateRequest,
    CandidateRejectRequest,
    CompanyNoteArchiveRequest,
    CompanyNoteCreateRequest,
    CompanyNoteEvidenceAttachRequest,
    CompanyNoteResponse,
    CompanyNoteReviewRequest,
    CompanyNoteSubmitRequest,
    CompanyNoteSupersedeRequest,
    CompanyNoteUpdateRequest,
    KnowledgeCandidateResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


@router.post(
    "/candidates",
    response_model=KnowledgeCandidateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Extract and create an unpublished knowledge candidate from an authorized conversation",
)
async def create_candidate(
    request: CandidateCreateRequest,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> KnowledgeCandidateResponse:
    service = KnowledgeCandidateService(session)
    return await service.create_candidate(user=user, request=request)


@router.post(
    "/candidates/{candidate_id}/submit",
    response_model=KnowledgeCandidateResponse,
    summary="Submit an unpublished candidate for publication review (DRAFT -> PENDING)",
)
async def submit_candidate(
    candidate_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> KnowledgeCandidateResponse:
    service = KnowledgeCandidateService(session)
    return await service.submit_candidate(user=user, candidate_id=candidate_id)


@router.post(
    "/candidates/{candidate_id}/approve",
    response_model=KnowledgeCandidateResponse,
    summary="Review and approve publication of a candidate (PENDING -> APPROVED, visibility=SHARED). Requires global ADMIN.",
)
async def approve_candidate(
    candidate_id: uuid.UUID,
    session: DbSession,
    admin_user: User = Depends(require_admin),
) -> KnowledgeCandidateResponse:
    service = KnowledgeCandidateService(session)
    return await service.approve_candidate(user=admin_user, candidate_id=candidate_id)


@router.post(
    "/candidates/{candidate_id}/reject",
    response_model=KnowledgeCandidateResponse,
    summary="Reject publication of a candidate (PENDING -> REJECTED). Requires global ADMIN.",
)
async def reject_candidate(
    candidate_id: uuid.UUID,
    session: DbSession,
    request: Optional[CandidateRejectRequest] = None,
    admin_user: User = Depends(require_admin),
) -> KnowledgeCandidateResponse:
    service = KnowledgeCandidateService(session)
    reason = request.reason if request else None
    return await service.reject_candidate(user=admin_user, candidate_id=candidate_id, reason=reason)


@router.post(
    "/candidates/{candidate_id}/revise",
    response_model=KnowledgeCandidateResponse,
    summary="Return a rejected or pending candidate back to DRAFT for revision",
)
async def return_to_draft(
    candidate_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> KnowledgeCandidateResponse:
    service = KnowledgeCandidateService(session)
    return await service.return_to_draft(user=user, candidate_id=candidate_id)


@router.delete(
    "/candidates/{candidate_id}/evidence/{evidence_id}",
    response_model=KnowledgeCandidateResponse,
    summary="Withdraw supporting evidence. If evidence count drops to 0, candidate is de-published.",
)
async def withdraw_evidence(
    candidate_id: uuid.UUID,
    evidence_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> KnowledgeCandidateResponse:
    service = KnowledgeCandidateService(session)
    return await service.withdraw_evidence(user=user, candidate_id=candidate_id, evidence_id=evidence_id)


@router.get(
    "/candidates/{candidate_id}",
    response_model=KnowledgeCandidateResponse,
    summary="Get publication status, approval details, and provenance for a candidate",
)
async def get_candidate(
    candidate_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> KnowledgeCandidateResponse:
    service = KnowledgeCandidateService(session)
    return await service.get_candidate(user=user, candidate_id=candidate_id)


@router.get(
    "/candidates",
    response_model=list[KnowledgeCandidateResponse],
    summary="List knowledge candidates scoped by user authorization and optional filters",
)
async def list_candidates(
    session: DbSession,
    user: User = Depends(get_current_user),
    status: Optional[str] = Query(default=None),
    visibility: Optional[str] = Query(default=None),
    approval_status: Optional[str] = Query(default=None),
    workspace_id: Optional[uuid.UUID] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[KnowledgeCandidateResponse]:
    service = KnowledgeCandidateService(session)
    return await service.list_candidates(
        user=user,
        status=status,
        visibility=visibility,
        approval_status=approval_status,
        workspace_id=workspace_id,
        limit=limit,
    )


@router.get(
    "/shared",
    response_model=list[KnowledgeCandidateResponse],
    summary="Inspect explicitly approved shared organizational knowledge (enforces strict retrieval boundary)",
)
async def get_approved_shared_knowledge(
    session: DbSession,
    user: User = Depends(get_current_user),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[KnowledgeCandidateResponse]:
    from rag_app.security.authorization import get_allowed_departments
    allowed_departments = None
    if user.role != "ADMIN":
        allowed_departments = await get_allowed_departments(session, user)
        if not allowed_departments:
            return []
    service = KnowledgeCandidateService(session)
    return await service.get_approved_shared_knowledge(allowed_departments=allowed_departments, limit=limit)


# ---------------------------------------------------------------------------
# Company Notes with Evidence Review & Controlled Ingestion
# ---------------------------------------------------------------------------

@router.post(
    "/company-notes",
    response_model=CompanyNoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Company Note observation (defaults to author-private draft)",
)
async def create_company_note(
    request: CompanyNoteCreateRequest,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.create_note(user=user, request=request)


@router.get(
    "/company-notes",
    response_model=list[CompanyNoteResponse],
    summary="List company notes scoped by author, workspace, or shared permissions",
)
async def list_company_notes(
    session: DbSession,
    user: User = Depends(get_current_user),
    company_name: Optional[str] = Query(default=None),
    visibility: Optional[str] = Query(default=None),
    approval_status: Optional[str] = Query(default=None),
    workspace_id: Optional[uuid.UUID] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[CompanyNoteResponse]:
    service = CompanyNotesService(session)
    return await service.list_notes(
        user=user,
        company_name=company_name,
        visibility=visibility,
        approval_status=approval_status,
        workspace_id=workspace_id,
        limit=limit,
    )


@router.get(
    "/company-notes/{note_id}",
    response_model=CompanyNoteResponse,
    summary="Get a company note with full source document evidence provenance",
)
async def get_company_note(
    note_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.get_note(user=user, note_id=note_id)


@router.patch(
    "/company-notes/{note_id}",
    response_model=CompanyNoteResponse,
    summary="Update a company note observation or metadata",
)
async def update_company_note(
    note_id: uuid.UUID,
    request: CompanyNoteUpdateRequest,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.update_note(user=user, note_id=note_id, request=request)


@router.post(
    "/company-notes/{note_id}/evidence",
    response_model=CompanyNoteResponse,
    summary="Attach an authorized source document chunk reference to a note",
)
async def attach_note_evidence(
    note_id: uuid.UUID,
    request: CompanyNoteEvidenceAttachRequest,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.attach_evidence(user=user, note_id=note_id, request=request)


@router.delete(
    "/company-notes/{note_id}/evidence/{evidence_id}",
    response_model=CompanyNoteResponse,
    summary="Remove a source document evidence reference from a note",
)
async def remove_note_evidence(
    note_id: uuid.UUID,
    evidence_id: uuid.UUID,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.remove_evidence(user=user, note_id=note_id, evidence_id=evidence_id)


@router.post(
    "/company-notes/{note_id}/share",
    response_model=CompanyNoteResponse,
    summary="Share an author-private draft with a workspace (distinct from publishing)",
)
async def share_note_to_workspace(
    note_id: uuid.UUID,
    workspace_id: uuid.UUID = Query(...),
    session: DbSession = None,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.share_to_workspace(user=user, note_id=note_id, workspace_id=workspace_id)


@router.post(
    "/company-notes/{note_id}/submit",
    response_model=CompanyNoteResponse,
    summary="Submit a note for publication review (DRAFT -> PENDING). Requires evidence.",
)
async def submit_company_note(
    note_id: uuid.UUID,
    request: CompanyNoteSubmitRequest,
    session: DbSession,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.submit_note(user=user, note_id=note_id, request=request)


@router.post(
    "/company-notes/{note_id}/review",
    response_model=CompanyNoteResponse,
    summary="Admin review: APPROVE, REJECT, REVISE, WITHDRAW, or ARCHIVE. Preserves GraphRAG provenance.",
)
async def review_company_note(
    note_id: uuid.UUID,
    request: CompanyNoteReviewRequest,
    session: DbSession,
    admin_user: User = Depends(require_admin),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.review_note(admin_user=admin_user, note_id=note_id, request=request)


@router.post(
    "/company-notes/{note_id}/archive",
    response_model=CompanyNoteResponse,
    summary="Archive or withdraw a company note (author or admin). Excludes from shared retrieval and archives GraphRAG relationships.",
)
async def archive_company_note(
    note_id: uuid.UUID,
    request: Optional[CompanyNoteArchiveRequest] = None,
    session: DbSession = None,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    reason = request.reason if request else None
    return await service.archive_note(user=user, note_id=note_id, reason=reason)


@router.post(
    "/company-notes/{note_id}/supersede",
    response_model=CompanyNoteResponse,
    summary="Supersede an old company note with a new replacement note (author or admin). Sets status=SUPERSEDED and superseded_by_id.",
)
async def supersede_company_note(
    note_id: uuid.UUID,
    request: CompanyNoteSupersedeRequest,
    session: DbSession = None,
    user: User = Depends(get_current_user),
) -> CompanyNoteResponse:
    service = CompanyNotesService(session)
    return await service.supersede_note(user=user, old_note_id=note_id, new_note_id=request.replacement_note_id)


@router.post(
    "/company-notes/batch-publish",
    response_model=BatchPublicationResponse,
    summary="Controlled batch ingestion coordinator: validate, track outcomes, recover from partial failures, and idempotently publish",
)
async def publish_company_notes_batch(
    request: BatchPublicationRequest,
    session: DbSession,
    admin_user: User = Depends(require_admin),
) -> BatchPublicationResponse:
    service = CompanyNotesService(session)
    return await service.publish_batch(admin_user=admin_user, request=request)


@router.post(
    "/company-notes/reconcile-orphans",
    summary="Admin maintenance: Reconcile orphaned company notes that lost all evidence (transitions to ARCHIVED/REJECTED). Requires global ADMIN.",
)
async def reconcile_orphaned_company_notes(
    session: DbSession,
    admin_user: User = Depends(require_admin),
) -> dict:
    service = CompanyNotesService(session)
    reconciled_count = await service.reconcile_orphaned_notes()
    return {"status": "SUCCESS", "reconciled_count": reconciled_count}
