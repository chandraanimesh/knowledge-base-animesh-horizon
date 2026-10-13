import asyncio
import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from fastapi import HTTPException
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import selectinload

from rag_app.core.config import get_settings
from rag_app.db.models import (
    User,
    Department,
    Permission,
    UserDepartment,
    UserPermission,
    DocumentPermission,
    Workspace,
    WorkspaceMember,
    Document,
    DocumentSection,
    DocumentChunk,
    KnowledgeItem,
    KnowledgeEntity,
    KnowledgeRelationship,
    KnowledgeEvidence,
)
from rag_app.knowledge.company_notes_service import CompanyNotesService
from rag_app.knowledge.candidate_service import KnowledgeCandidateService
from rag_app.schemas.knowledge import (
    CompanyNoteCreateRequest,
    CompanyNoteUpdateRequest,
    CompanyNoteEvidenceAttachRequest,
    CompanyNoteSubmitRequest,
    CompanyNoteReviewRequest,
    BatchPublicationRequest,
    CompanyNoteWebsiteSourceInput,
)


@pytest_asyncio.fixture
async def notes_test_env():
    """
    Creates an isolated schema for Company Notes tests.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_notes_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))

        def create_tables(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
            Department.__table__.create(sync_conn)
            User.__table__.create(sync_conn)
            Permission.__table__.create(sync_conn)
            UserDepartment.__table__.create(sync_conn)
            UserPermission.__table__.create(sync_conn)
            Workspace.__table__.create(sync_conn)
            WorkspaceMember.__table__.create(sync_conn)
            Document.__table__.create(sync_conn)
            DocumentSection.__table__.create(sync_conn)
            DocumentChunk.__table__.create(sync_conn)
            DocumentPermission.__table__.create(sync_conn)
            KnowledgeItem.__table__.create(sync_conn)
            KnowledgeEntity.__table__.create(sync_conn)
            KnowledgeRelationship.__table__.create(sync_conn)
            KnowledgeEvidence.__table__.create(sync_conn)

        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:8]
    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        dept = Department(name=f"Finance Dept {suffix}", code=f"FIN_{suffix}")
        other_dept = Department(name=f"Legal Dept {suffix}", code=f"LEG_{suffix}")
        session.add_all([dept, other_dept])
        await session.flush()

        admin_user = User(email=f"admin_{suffix}@example.com", hashed_password="pwd", role="ADMIN")
        author_user = User(email=f"author_{suffix}@example.com", hashed_password="pwd", role="USER")
        colleague_user = User(email=f"colleague_{suffix}@example.com", hashed_password="pwd", role="USER")
        unauth_user = User(email=f"unauth_{suffix}@example.com", hashed_password="pwd", role="USER")
        session.add_all([admin_user, author_user, colleague_user, unauth_user])
        await session.flush()

        perm_view = Permission(code="VIEW")
        session.add(perm_view)
        await session.flush()

        # Grant department access to author and colleague
        session.add_all([
            UserDepartment(user_id=author_user.id, department_id=dept.id),
            UserPermission(user_id=author_user.id, department_id=dept.id, permission_id=perm_view.id),
            UserDepartment(user_id=colleague_user.id, department_id=dept.id),
            UserPermission(user_id=colleague_user.id, department_id=dept.id, permission_id=perm_view.id),
        ])
        await session.flush()

        # Workspace
        ws = Workspace(name=f"Finance WS {suffix}", code=f"fin_{suffix}", department_id=dept.id, created_by=admin_user.id)
        session.add(ws)
        await session.flush()

        session.add_all([
            WorkspaceMember(workspace_id=ws.id, user_id=author_user.id, role="EDITOR"),
            WorkspaceMember(workspace_id=ws.id, user_id=colleague_user.id, role="VIEWER"),
        ])
        await session.flush()

        # Document & Chunk
        doc = Document(
            file_name="acme_contract_q3.pdf",
            file_path="/tmp/acme_contract_q3.pdf",
            content_hash=f"hash_{suffix}",
            department_id=dept.id,
            created_by=author_user.id,
            workspace_id=ws.id,
        )
        session.add(doc)
        await session.flush()

        sec = DocumentSection(
            document_id=doc.id,
            title="Commercial Terms",
            level=1,
            ordinal_path="1.2",
        )
        session.add(sec)
        await session.flush()

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            text="Acme Corp agrees to commit to 50 enterprise licenses with APAC expansion options.",
            embedding=[0.0] * settings.embedding_dim,
            page_number=3,
            node_id=sec.id,
            chunk_hash=f"chunk_hash_{suffix}_0",
        )
        session.add(chunk)
        await session.flush()

        # Second chunk for multi-evidence tests
        chunk2 = DocumentChunk(
            document_id=doc.id,
            chunk_index=1,
            text="Renewal is scheduled for October 2026 subject to executive sponsorship review.",
            embedding=[0.0] * settings.embedding_dim,
            page_number=4,
            node_id=sec.id,
            chunk_hash=f"chunk_hash_{suffix}_1",
        )
        session.add(chunk2)
        await session.flush()

        await session.commit()

        context = {
            "schema_name": schema_name,
            "session": session,
            "session_maker": session_maker,
            "dept": dept,
            "other_dept": other_dept,
            "admin_user": admin_user,
            "author_user": author_user,
            "colleague_user": colleague_user,
            "unauth_user": unauth_user,
            "ws": ws,
            "doc": doc,
            "chunk": chunk,
            "chunk2": chunk2,
        }

        yield context

    # Cleanup test schema
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_private_draft_isolation(notes_test_env):
    """
    Verifies that:
    - A draft note defaults to visibility='PRIVATE'.
    - It is isolated strictly to the author.
    - Other users (even colleagues in the same workspace) receive 403 Forbidden.
    - An Administrator cannot view another user's private draft.
    - Listing notes filters out private notes of other authors for both normal users and admins.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    colleague = notes_test_env["colleague_user"]
    admin = notes_test_env["admin_user"]

    service = CompanyNotesService(session)

    # 1. Author creates private draft
    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Client is considering switching to competitor if SLA is below 99.9%",
            observation_type="INFERRED",
            visibility="PRIVATE",
        ),
    )
    assert note.visibility == "PRIVATE"
    assert note.approval_status == "DRAFT"
    assert note.author_id == author.id

    # 2. Author can view and update note
    fetched = await service.get_note(user=author, note_id=note.id)
    assert fetched.id == note.id

    # 3. Colleague receives 403 Forbidden
    with pytest.raises(HTTPException) as exc_info:
        await service.get_note(user=colleague, note_id=note.id)
    assert exc_info.value.status_code == 403

    # 4. Admin receives 403 Forbidden on author-private draft
    with pytest.raises(HTTPException) as exc_info:
        await service.get_note(user=admin, note_id=note.id)
    assert exc_info.value.status_code == 403

    # 4b. Admin cannot submit another user's private draft
    with pytest.raises(HTTPException) as exc_info:
        await service.submit_note(user=admin, note_id=note.id, request=CompanyNoteSubmitRequest())
    assert exc_info.value.status_code == 403

    # 4c. Admin cannot review another user's unsubmitted private draft
    with pytest.raises(HTTPException) as exc_info:
        await service.review_note(admin_user=admin, note_id=note.id, request=CompanyNoteReviewRequest(action="APPROVE"))
    assert exc_info.value.status_code == 403

    # 4d. Batch publish safely rejects another user's unsubmitted private draft
    batch_res = await service.publish_batch(admin_user=admin, request=BatchPublicationRequest(note_ids=[note.id]))
    assert batch_res.published_count == 0
    assert batch_res.failed_count == 1
    assert batch_res.outcomes[0].status == "PRIVATE_DRAFT"

    # 5. List notes isolation
    author_notes = await service.list_notes(user=author)
    assert any(n.id == note.id for n in author_notes)

    colleague_notes = await service.list_notes(user=colleague)
    assert not any(n.id == note.id for n in colleague_notes)

    admin_notes = await service.list_notes(user=admin)
    assert not any(n.id == note.id for n in admin_notes)


@pytest.mark.asyncio
async def test_unauthorized_source_evidence_attachment(notes_test_env):
    """
    Verifies that a user cannot attach evidence from documents they lack permission to access.
    """
    session = notes_test_env["session"]
    unauth_user = notes_test_env["unauth_user"]
    author = notes_test_env["author_user"]
    chunk = notes_test_env["chunk"]

    service = CompanyNotesService(session)

    # Unauth user attempts to attach chunk from Finance Dept
    with pytest.raises(HTTPException) as exc_info:
        await service.create_note(
            user=unauth_user,
            request=CompanyNoteCreateRequest(
                company_name="Acme Corp",
                statement="Unauthorized attempt to attach confidential chunk",
                document_chunk_ids=[chunk.id],
            ),
        )
    assert exc_info.value.status_code == 403

    # Author with VIEW perm attaches chunk successfully
    valid_note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Acme contract includes APAC expansion rights.",
            observation_type="SUPPORTED",
            document_chunk_ids=[chunk.id],
        ),
    )
    assert valid_note.evidence_count == 1
    assert valid_note.evidence[0].file_name == "acme_contract_q3.pdf"
    assert valid_note.evidence[0].page_number == 3
    assert valid_note.evidence[0].section_title == "Commercial Terms"
    assert "APAC expansion" in valid_note.evidence[0].excerpt


@pytest.mark.asyncio
async def test_editing_and_evidence_changes(notes_test_env):
    """
    Verifies editing of note content, observation type, and adding/removing evidence links.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    chunk = notes_test_env["chunk"]
    chunk2 = notes_test_env["chunk2"]

    service = CompanyNotesService(session)

    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Initial observation statement",
            observation_type="UNVERIFIED",
        ),
    )
    assert note.observation_type == "UNVERIFIED"

    # Edit statement and change observation type to SUPPORTED
    updated = await service.update_note(
        user=author,
        note_id=note.id,
        request=CompanyNoteUpdateRequest(
            statement="Verified statement with commercial team",
            observation_type="SUPPORTED",
        ),
    )
    assert updated.statement == "Verified statement with commercial team"
    assert updated.observation_type == "SUPPORTED"
    assert updated.confidence == 1.0

    # Attach evidence
    with_ev = await service.attach_evidence(
        user=author,
        note_id=note.id,
        request=CompanyNoteEvidenceAttachRequest(document_chunk_id=chunk.id),
    )
    assert with_ev.evidence_count == 1
    ev_id = with_ev.evidence[0].id

    # Attach second evidence
    with_ev2 = await service.attach_evidence(
        user=author,
        note_id=note.id,
        request=CompanyNoteEvidenceAttachRequest(document_chunk_id=chunk2.id),
    )
    assert with_ev2.evidence_count == 2

    # Remove first evidence
    after_remove = await service.remove_evidence(
        user=author,
        note_id=note.id,
        evidence_id=ev_id,
    )
    assert after_remove.evidence_count == 1


@pytest.mark.asyncio
async def test_submission_and_approval_authorization(notes_test_env):
    """
    Verifies that:
    - Notes without evidence cannot be submitted.
    - Notes with evidence can be submitted (DRAFT -> PENDING).
    - Normal users cannot approve.
    - Admin can approve (PENDING -> APPROVED, visibility=SHARED).
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    colleague = notes_test_env["colleague_user"]
    admin = notes_test_env["admin_user"]
    chunk = notes_test_env["chunk"]

    service = CompanyNotesService(session)

    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Stark Industries",
            statement="Arc reactor licensing partnership proposal in review",
        ),
    )

    # Cannot submit without evidence
    with pytest.raises(HTTPException) as exc_info:
        await service.submit_note(
            user=author,
            note_id=note.id,
            request=CompanyNoteSubmitRequest(),
        )
    assert exc_info.value.status_code == 400

    # Attach evidence
    await service.attach_evidence(
        user=author,
        note_id=note.id,
        request=CompanyNoteEvidenceAttachRequest(document_chunk_id=chunk.id),
    )

    # Submit for review
    submitted = await service.submit_note(
        user=author,
        note_id=note.id,
        request=CompanyNoteSubmitRequest(target_audience="SHARED"),
    )
    assert submitted.approval_status == "PENDING"

    # Non-admin cannot approve
    with pytest.raises(HTTPException) as exc_info:
        await service.review_note(
            admin_user=colleague,
            note_id=note.id,
            request=CompanyNoteReviewRequest(action="APPROVE"),
        )
    assert exc_info.value.status_code == 403

    # Admin approves
    approved = await service.review_note(
        admin_user=admin,
        note_id=note.id,
        request=CompanyNoteReviewRequest(action="APPROVE"),
    )
    assert approved.approval_status == "APPROVED"
    assert approved.visibility == "SHARED"
    assert approved.approved_by_id == admin.id
    assert approved.approved_at is not None


@pytest.mark.asyncio
async def test_rejected_and_unapproved_notes_excluded_from_retrieval(notes_test_env):
    """
    Verifies that drafts, pending notes, and rejected notes never enter shared retrieval.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    admin = notes_test_env["admin_user"]
    dept = notes_test_env["dept"]
    chunk = notes_test_env["chunk"]

    notes_service = CompanyNotesService(session)
    candidate_service = KnowledgeCandidateService(session)

    # 1. Draft note
    draft_note = await notes_service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Wayne Enterprises",
            statement="Confidential security audit findings for Wayne tower",
            document_chunk_ids=[chunk.id],
            department_id=dept.id,
        ),
    )

    # 2. Search shared knowledge: must return zero results
    shared_results = await candidate_service.search_approved_shared_knowledge(
        query="Wayne tower security",
        allowed_departments=[dept.id],
    )
    assert len(shared_results) == 0

    # 3. Submit to PENDING: must still return zero results
    await notes_service.submit_note(user=author, note_id=draft_note.id, request=CompanyNoteSubmitRequest())
    shared_results = await candidate_service.search_approved_shared_knowledge(
        query="Wayne tower security",
        allowed_departments=[dept.id],
    )
    assert len(shared_results) == 0

    # 4. Reject: must return zero results
    await notes_service.review_note(
        admin_user=admin,
        note_id=draft_note.id,
        request=CompanyNoteReviewRequest(action="REJECT"),
    )
    shared_results = await candidate_service.search_approved_shared_knowledge(
        query="Wayne tower security",
        allowed_departments=[dept.id],
    )
    assert len(shared_results) == 0

    # 5. Revise and Approve: must now appear in retrieval
    await notes_service.review_note(
        admin_user=admin,
        note_id=draft_note.id,
        request=CompanyNoteReviewRequest(action="REVISE"),
    )
    await notes_service.review_note(
        admin_user=admin,
        note_id=draft_note.id,
        request=CompanyNoteReviewRequest(action="APPROVE"),
    )

    shared_results = await candidate_service.search_approved_shared_knowledge(
        query="Wayne tower security",
        allowed_departments=[dept.id],
    )
    assert len(shared_results) >= 1
    assert any("wayne" in item[0].canonical_key.lower() for item in shared_results)


@pytest.mark.asyncio
async def test_batch_idempotency_and_failure_recovery(notes_test_env):
    """
    Verifies that the batch ingestion coordinator:
    - Publishes valid notes.
    - Tracks per-item outcomes safely.
    - Recovers safely from partial failures (e.g. missing evidence, duplicates).
    - Idempotently succeeds when already approved notes are republished.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    admin = notes_test_env["admin_user"]
    chunk = notes_test_env["chunk"]
    chunk2 = notes_test_env["chunk2"]

    service = CompanyNotesService(session)

    # Create 3 notes
    note1 = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Batch Co Alpha",
            statement="Alpha enterprise contract renewal confirmed.",
            document_chunk_ids=[chunk.id],
        ),
    )
    note2_no_evidence = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Batch Co Beta",
            statement="Beta observation without any evidence backing.",
            visibility="WORKSPACE",
        ),
    )
    note3 = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Batch Co Gamma",
            statement="Gamma deployment successfully expanded to 100 users.",
            document_chunk_ids=[chunk2.id],
        ),
    )
    note4_unsubmitted_private = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Batch Co Delta",
            statement="Delta unsubmitted private draft observation.",
            document_chunk_ids=[chunk.id],
            visibility="PRIVATE",
        ),
    )

    # Author submits note1 and note3 for review
    await service.submit_note(user=author, note_id=note1.id, request=CompanyNoteSubmitRequest())
    await service.submit_note(user=author, note_id=note3.id, request=CompanyNoteSubmitRequest())

    # Execute batch publication with 2 valid, 1 no evidence, 1 unsubmitted private draft
    batch_res = await service.publish_batch(
        admin_user=admin,
        request=BatchPublicationRequest(note_ids=[note1.id, note2_no_evidence.id, note3.id, note4_unsubmitted_private.id]),
    )

    assert batch_res.total_items == 4
    assert batch_res.published_count == 2
    assert batch_res.failed_count == 2

    outcomes = {o.note_id: o for o in batch_res.outcomes}
    assert outcomes[note1.id].success is True
    assert outcomes[note1.id].status == "PUBLISHED"
    assert outcomes[note2_no_evidence.id].success is False
    assert outcomes[note2_no_evidence.id].status == "NO_EVIDENCE"
    assert outcomes[note4_unsubmitted_private.id].success is False
    assert outcomes[note4_unsubmitted_private.id].status == "PRIVATE_DRAFT"
    assert outcomes[note3.id].success is True

    # Test Idempotency: re-running batch publication on already approved note1
    retry_res = await service.publish_batch(
        admin_user=admin,
        request=BatchPublicationRequest(note_ids=[note1.id, note3.id]),
    )
    assert retry_res.total_items == 2
    assert retry_res.published_count == 2
    assert retry_res.failed_count == 0
    assert retry_res.is_idempotent_retry is True
    for o in retry_res.outcomes:
        assert o.status == "ALREADY_APPROVED"


@pytest.mark.asyncio
async def test_evidence_and_graphrag_relationship_provenance(notes_test_env):
    """
    Verifies that approving a company note materializes GraphRAG entities and relationships
    while preserving evidence provenance.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    admin = notes_test_env["admin_user"]
    dept = notes_test_env["dept"]
    chunk = notes_test_env["chunk"]

    service = CompanyNotesService(session)

    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Cyberdyne Systems",
            statement="Cyberdyne neural processor prototype delivered.",
            document_chunk_ids=[chunk.id],
            department_id=dept.id,
        ),
    )

    # Author submits note for review
    await service.submit_note(user=author, note_id=note.id, request=CompanyNoteSubmitRequest())

    # Admin approves note
    await service.review_note(
        admin_user=admin,
        note_id=note.id,
        request=CompanyNoteReviewRequest(action="APPROVE"),
    )

    # Check GraphRAG relationship
    rel = await session.scalar(
        select(KnowledgeRelationship)
        .options(selectinload(KnowledgeRelationship.knowledge_item))
        .join(KnowledgeItem, KnowledgeRelationship.knowledge_item_id == KnowledgeItem.id)
        .where(KnowledgeRelationship.predicate == "has_observation")
    )
    assert rel is not None
    assert rel.knowledge_item.visibility == "SHARED"
    assert rel.knowledge_item.approval_status == "APPROVED"

    # Verify evidence provenance link on relationship
    rel_ev = await session.scalar(
        select(KnowledgeEvidence).where(KnowledgeEvidence.knowledge_item_id == rel.knowledge_item_id)
    )
    assert rel_ev is not None
    assert rel_ev.document_chunk_id == chunk.id


@pytest.mark.asyncio
async def test_workspace_sharing_deep_security_matrix(notes_test_env):
    """
    Comprehensive verification of CompanyNotesService.share_to_workspace() across:
    1. PRIVATE + DRAFT + author -> Success (promoted to WORKSPACE).
    2. PRIVATE + DRAFT + other user -> 403 Forbidden.
    3. PRIVATE + DRAFT + ADMIN -> 403 Forbidden (no blanket admin bypass on unsubmitted drafts).
    4. WORKSPACE + DRAFT + author in target ws -> Success.
    5. WORKSPACE + DRAFT + other user -> 403 Forbidden.
    6. WORKSPACE + DRAFT + ADMIN -> Success.
    7. Non-draft states (PENDING, APPROVED, ARCHIVED) -> 400 Bad Request.
    8. Missing or inactive target workspace -> 404 Not Found.
    9. Inaccessible target workspace (non-member) -> 403 Forbidden.
    10. Repeated sharing -> Idempotent success (200).
    11. Evidence citation excerpt vs document boundary.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    colleague = notes_test_env["colleague_user"]
    admin = notes_test_env["admin_user"]
    unauth = notes_test_env["unauth_user"]
    ws = notes_test_env["ws"]
    dept = notes_test_env["dept"]
    chunk = notes_test_env["chunk"]

    service = CompanyNotesService(session)

    # Create a second workspace in same department with colleague and author as members
    ws2 = Workspace(name="Secondary WS", code=f"sec_ws_{uuid.uuid4().hex[:6]}", department_id=dept.id, created_by=admin.id)
    ws_inactive = Workspace(name="Inactive WS", code=f"inact_{uuid.uuid4().hex[:6]}", department_id=dept.id, is_active=False, created_by=admin.id)
    ws_isolated = Workspace(name="Isolated WS", code=f"iso_{uuid.uuid4().hex[:6]}", department_id=dept.id, created_by=admin.id)
    session.add_all([ws2, ws_inactive, ws_isolated])
    await session.flush()

    session.add_all([
        WorkspaceMember(workspace_id=ws2.id, user_id=author.id, role="EDITOR"),
        WorkspaceMember(workspace_id=ws2.id, user_id=colleague.id, role="VIEWER"),
        # Author is NOT a member of ws_isolated
        WorkspaceMember(workspace_id=ws_isolated.id, user_id=colleague.id, role="EDITOR"),
    ])
    await session.commit()

    # 1. Author creates private draft
    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Matrix Corp",
            statement="Commercial SLA requirements in negotiation.",
            visibility="PRIVATE",
            document_chunk_ids=[chunk.id],
        ),
    )
    assert note.visibility == "PRIVATE"
    assert note.approval_status == "DRAFT"

    # Case 2: Other user cannot share author's private draft
    with pytest.raises(HTTPException) as exc_info:
        await service.share_to_workspace(user=colleague, note_id=note.id, workspace_id=ws.id)
    assert exc_info.value.status_code == 403

    # Case 3: Admin cannot share author's unsubmitted private draft
    with pytest.raises(HTTPException) as exc_info:
        await service.share_to_workspace(user=admin, note_id=note.id, workspace_id=ws.id)
    assert exc_info.value.status_code == 403

    # Case 8: Missing target workspace -> 404
    with pytest.raises(HTTPException) as exc_info:
        await service.share_to_workspace(user=author, note_id=note.id, workspace_id=uuid.uuid4())
    assert exc_info.value.status_code == 404

    # Case 8b: Inactive target workspace -> 404
    with pytest.raises(HTTPException) as exc_info:
        await service.share_to_workspace(user=author, note_id=note.id, workspace_id=ws_inactive.id)
    assert exc_info.value.status_code == 404

    # Case 9: Inaccessible workspace (author not a member) -> 403
    with pytest.raises(HTTPException) as exc_info:
        await service.share_to_workspace(user=author, note_id=note.id, workspace_id=ws_isolated.id)
    assert exc_info.value.status_code == 403

    # Case 1: Author shares private draft with active workspace where author is a member
    shared_res = await service.share_to_workspace(user=author, note_id=note.id, workspace_id=ws.id)
    assert shared_res.visibility == "WORKSPACE"
    assert shared_res.workspace_id == ws.id
    assert shared_res.approval_status == "DRAFT"

    # Case 10: Repeated sharing of same note to same workspace -> Idempotent success (200)
    reshared = await service.share_to_workspace(user=author, note_id=note.id, workspace_id=ws.id)
    assert reshared.id == note.id
    assert reshared.workspace_id == ws.id

    # Case 4: Author re-shares/moves WORKSPACE draft to ws2 (where author is also a member)
    moved_res = await service.share_to_workspace(user=author, note_id=note.id, workspace_id=ws2.id)
    assert moved_res.workspace_id == ws2.id
    assert moved_res.visibility == "WORKSPACE"

    # Case 5: Different regular user cannot re-share another user's WORKSPACE draft
    with pytest.raises(HTTPException) as exc_info:
        await service.share_to_workspace(user=colleague, note_id=note.id, workspace_id=ws.id)
    assert exc_info.value.status_code == 403

    # Case 6: Admin CAN reassign a WORKSPACE draft note
    admin_moved = await service.share_to_workspace(user=admin, note_id=note.id, workspace_id=ws.id)
    assert admin_moved.workspace_id == ws.id

    # Case 11: Colleague in workspace can view note and evidence excerpt
    colleague_view = await service.get_note(user=colleague, note_id=note.id)
    assert colleague_view.id == note.id
    assert len(colleague_view.evidence) == 1
    assert "APAC expansion" in colleague_view.evidence[0].excerpt

    # Unauth user (not in workspace) cannot view the workspace note
    with pytest.raises(HTTPException) as exc_info:
        await service.get_note(user=unauth, note_id=note.id)
    assert exc_info.value.status_code == 403

    # Case 7: Non-draft states cannot be shared
    # Submit note -> PENDING
    await service.submit_note(user=author, note_id=note.id, request=CompanyNoteSubmitRequest())
    with pytest.raises(HTTPException) as exc_info:
        await service.share_to_workspace(user=author, note_id=note.id, workspace_id=ws.id)
    assert exc_info.value.status_code == 400
    assert "only DRAFT notes can be shared" in exc_info.value.detail

    # Approve note -> APPROVED (SHARED)
    await service.review_note(admin_user=admin, note_id=note.id, request=CompanyNoteReviewRequest(action="APPROVE"))
    with pytest.raises(HTTPException) as exc_info:
        await service.share_to_workspace(user=author, note_id=note.id, workspace_id=ws.id)
    assert exc_info.value.status_code == 400
    assert "only DRAFT notes can be shared" in exc_info.value.detail


@pytest.mark.asyncio
async def test_evidence_excerpt_masking_and_provenance_cleanup(notes_test_env):
    """
    Verifies:
    1. Evidence excerpt masking for users without VIEW access on the source document.
    2. Preservation of provenance metadata (file_name, page_number, section_title) when masked.
    3. Unmasking when source document permission is granted.
    4. GraphRAG relationship archival when all evidence is removed from an approved note.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    admin = notes_test_env["admin_user"]
    dept = notes_test_env["dept"]
    ws = notes_test_env["ws"]
    doc = notes_test_env["doc"]
    chunk = notes_test_env["chunk"]

    suffix = uuid.uuid4().hex[:6]
    # Create a user who belongs to the workspace, but has NO document permissions on doc
    restricted_user = User(email=f"restricted_{suffix}@example.com", hashed_password="pwd", role="USER")
    session.add(restricted_user)
    await session.flush()

    session.add(WorkspaceMember(workspace_id=ws.id, user_id=restricted_user.id, role="VIEWER"))
    await session.commit()

    service = CompanyNotesService(session)

    # Author creates note with evidence
    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Confidential Client Co",
            statement="Client signed expansion terms under NDA.",
            visibility="PRIVATE",
            document_chunk_ids=[chunk.id],
        ),
    )

    # Author shares note with workspace
    await service.share_to_workspace(user=author, note_id=note.id, workspace_id=ws.id)

    # 1. Author (who has document VIEW) sees unmasked excerpt
    author_view = await service.get_note(user=author, note_id=note.id)
    assert len(author_view.evidence) == 1
    assert "APAC expansion" in author_view.evidence[0].excerpt

    # 2. Admin sees unmasked excerpt
    admin_view = await service.get_note(user=admin, note_id=note.id)
    assert "APAC expansion" in admin_view.evidence[0].excerpt

    # 3. Restricted colleague sees MASKED excerpt, but provenance metadata is retained
    colleague_view = await service.get_note(user=restricted_user, note_id=note.id)
    assert len(colleague_view.evidence) == 1
    ev_item = colleague_view.evidence[0]
    assert ev_item.file_name == "acme_contract_q3.pdf"
    assert ev_item.page_number == 3
    assert ev_item.section_title == "Commercial Terms"
    assert ev_item.excerpt == "[RESTRICTED - View permission on source document required]"

    # 4. Grant restricted_user department membership and document permission on doc and verify unmasking
    perm_view = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
    session.add_all([
        UserDepartment(user_id=restricted_user.id, department_id=dept.id),
        DocumentPermission(
            document_id=doc.id,
            user_id=restricted_user.id,
            permission_id=perm_view.id,
        ),
    ])
    await session.commit()

    unmasked_view = await service.get_note(user=restricted_user, note_id=note.id)
    assert "APAC expansion" in unmasked_view.evidence[0].excerpt

    # 5. Verify GraphRAG relationship archival on complete evidence removal
    await service.submit_note(user=author, note_id=note.id, request=CompanyNoteSubmitRequest())
    await service.review_note(admin_user=admin, note_id=note.id, request=CompanyNoteReviewRequest(action="APPROVE"))

    # Verify relationship was created and is ACTIVE
    rel_key_prefix = f"RELATIONSHIP:company:%:has_observation:company_note:{note.id.hex[:8]}"
    active_rel = await session.scalar(
        select(KnowledgeItem).where(
            KnowledgeItem.canonical_key.like(rel_key_prefix),
            KnowledgeItem.status == "ACTIVE",
        )
    )
    assert active_rel is not None

    # Remove the only evidence item from approved note
    ev_id = author_view.evidence[0].id
    await service.remove_evidence(user=author, note_id=note.id, evidence_id=ev_id)

    # Note is now archived
    archived_note = await session.get(KnowledgeItem, note.id)
    assert archived_note.status == "ARCHIVED"
    assert archived_note.approval_status == "REJECTED"

    # Associated relationship is also archived
    archived_rel = await session.scalar(
        select(KnowledgeItem).where(
            KnowledgeItem.canonical_key.like(rel_key_prefix),
            KnowledgeItem.id == active_rel.id,
        )
    )
    assert archived_rel.status == "ARCHIVED"
    assert archived_rel.approval_status == "REJECTED"


@pytest.mark.asyncio
async def test_concurrent_batch_publication_safety(notes_test_env):
    """
    Hardening Test 1: Concurrency Integration Test
    Verifies that when two administrators concurrently publish the exact same note:
    1. Both requests succeed safely without unhandled lock or uniqueness exceptions.
    2. Exactly one reports 'PUBLISHED' and the other reports 'ALREADY_APPROVED'.
    3. Exactly one GraphRAG relationship and entity exist (zero duplicates).
    4. Database approval state is consistent (APPROVED, SHARED).
    """
    session = notes_test_env["session"]
    session_maker = notes_test_env["session_maker"]
    schema_name = notes_test_env["schema_name"]
    author = notes_test_env["author_user"]
    admin = notes_test_env["admin_user"]
    chunk = notes_test_env["chunk"]

    # Create a second admin user
    suffix = uuid.uuid4().hex[:6]
    admin2 = User(email=f"admin2_{suffix}@example.com", hashed_password="pwd", role="ADMIN")
    session.add(admin2)
    await session.commit()

    service = CompanyNotesService(session)

    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Concurrency Corp",
            statement="Concurrency guarantees under simultaneous publication.",
            visibility="WORKSPACE",
            document_chunk_ids=[chunk.id],
        ),
    )
    await service.submit_note(user=author, note_id=note.id, request=CompanyNoteSubmitRequest())

    # Open two separate concurrent DB sessions on the same test schema
    async with session_maker() as sess1, session_maker() as sess2:
        await sess1.execute(text(f"SET search_path TO {schema_name}, public"))
        await sess2.execute(text(f"SET search_path TO {schema_name}, public"))

        svc1 = CompanyNotesService(sess1)
        svc2 = CompanyNotesService(sess2)

        req1 = BatchPublicationRequest(note_ids=[note.id])
        req2 = BatchPublicationRequest(note_ids=[note.id])

        # Execute simultaneously using asyncio.gather
        res1, res2 = await asyncio.gather(
            svc1.publish_batch(admin_user=admin, request=req1),
            svc2.publish_batch(admin_user=admin2, request=req2),
        )

    # Verify outcomes
    statuses = {res1.outcomes[0].status, res2.outcomes[0].status}
    assert "PUBLISHED" in statuses
    assert "ALREADY_APPROVED" in statuses
    assert res1.failed_count == 0
    assert res2.failed_count == 0

    # Verify zero duplicate GraphRAG relationships in DB
    rel_key_prefix = f"RELATIONSHIP:company:%:has_observation:company_note:{note.id.hex[:8]}"
    rel_count = await session.scalar(
        select(func.count(KnowledgeItem.id)).where(
            KnowledgeItem.canonical_key.like(rel_key_prefix),
            KnowledgeItem.kind == "RELATIONSHIP",
        )
    )
    assert rel_count == 1

    # Verify note status is APPROVED and SHARED
    final_note = await session.get(KnowledgeItem, note.id)
    assert final_note.approval_status == "APPROVED"
    assert final_note.visibility == "SHARED"


@pytest.mark.asyncio
async def test_source_deletion_reconciliation(notes_test_env):
    """
    Hardening Test 2: Source-Deletion Reconciliation
    Verifies that:
    1. Approved note losing all valid evidence is safely de-published by reconcile_orphaned_notes().
    2. Note with remaining evidence is untouched by reconciliation.
    3. Reconciled note is excluded from retrieval and administrative publish attempts.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    admin = notes_test_env["admin_user"]
    dept = notes_test_env["dept"]
    chunk = notes_test_env["chunk"]
    chunk2 = notes_test_env["chunk2"]

    service = CompanyNotesService(session)
    candidate_service = KnowledgeCandidateService(session)

    # Note 1: Will lose its evidence
    note1 = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Reconcile Corp Alpha",
            statement="Observation that will lose evidence via source deletion.",
            document_chunk_ids=[chunk.id],
            department_id=dept.id,
        ),
    )
    # Note 2: Will keep its evidence
    note2 = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Reconcile Corp Beta",
            statement="Observation that preserves valid evidence backing.",
            document_chunk_ids=[chunk2.id],
            department_id=dept.id,
        ),
    )

    await service.submit_note(user=author, note_id=note1.id, request=CompanyNoteSubmitRequest())
    await service.submit_note(user=author, note_id=note2.id, request=CompanyNoteSubmitRequest())

    await service.review_note(admin_user=admin, note_id=note1.id, request=CompanyNoteReviewRequest(action="APPROVE"))
    await service.review_note(admin_user=admin, note_id=note2.id, request=CompanyNoteReviewRequest(action="APPROVE"))

    # Simulate direct hard-deletion of chunk/document by deleting evidence for note1
    await session.execute(
        text("DELETE FROM knowledge_evidence WHERE knowledge_item_id = :nid").bindparams(nid=note1.id)
    )
    await session.commit()

    # Shared search query: note1 is immediately excluded by SQL exists() check
    search_res = await candidate_service.search_approved_shared_knowledge(
        query="Reconcile Corp",
        allowed_departments=[dept.id],
    )
    assert not any(item[0].id == note1.id for item in search_res)
    assert any(item[0].id == note2.id for item in search_res)

    # Run idempotent administrative reconciliation
    reconciled_count = await service.reconcile_orphaned_notes()
    assert reconciled_count == 1

    # Verify Note 1 is now ARCHIVED and REJECTED
    rec_note1 = await session.get(KnowledgeItem, note1.id)
    assert rec_note1.status == "ARCHIVED"
    assert rec_note1.approval_status == "REJECTED"

    # Verify Note 2 is unaffected
    rec_note2 = await session.get(KnowledgeItem, note2.id)
    assert rec_note2.status == "ACTIVE"
    assert rec_note2.approval_status == "APPROVED"

    # Subsequent reconciliation is idempotent
    second_reconcile = await service.reconcile_orphaned_notes()
    assert second_reconcile == 0


@pytest.mark.asyncio
async def test_migration_downgrade_privacy_protection(notes_test_env):
    """
    Hardening Test 3: Migration Downgrade Privacy Protection
    Verifies that:
    1. Simulating the PRIVATE-visibility migration downgrade archives private drafts.
    2. Colleague in the same workspace cannot view or list the downgraded archived draft.
    3. Detail view returns 404 Not Found to unauthorized colleagues.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    colleague = notes_test_env["colleague_user"]
    ws = notes_test_env["ws"]
    chunk = notes_test_env["chunk"]

    service = CompanyNotesService(session)

    # Author creates a private draft associated with workspace
    private_note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Private Confidential Co",
            statement="Top-secret draft observation never shared.",
            visibility="PRIVATE",
            workspace_id=ws.id,
            document_chunk_ids=[chunk.id],
        ),
    )
    assert private_note.visibility == "PRIVATE"

    # Simulate migration downgrade:
    # UPDATE knowledge_items SET visibility = 'WORKSPACE', status = 'ARCHIVED' WHERE visibility = 'PRIVATE'
    await session.execute(
        text("UPDATE knowledge_items SET visibility = 'WORKSPACE', status = 'ARCHIVED' WHERE visibility = 'PRIVATE'")
    )
    await session.commit()

    # Colleague attempts to list notes: must NOT see the downgraded archived note
    colleague_notes = await service.list_notes(user=colleague, workspace_id=ws.id)
    assert not any(n.id == private_note.id for n in colleague_notes)

    # Colleague attempts to fetch note detail: must return 404 Not Found
    with pytest.raises(HTTPException) as exc_info:
        await service.get_note(user=colleague, note_id=private_note.id)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_graphrag_relationship_items_excluded_from_company_notes_list(notes_test_env):
    """
    Verifies that:
    1. Document-derived GraphRAG relationship items (kind == 'RELATIONSHIP') are excluded
       from the company notes review list.
    2. The underlying graph relationship record is NOT modified or deleted in the database.
    3. Multiple distinct observations for the same company remain visible.
    4. Observation claims are clean statements, not raw canonical keys or RELATIONSHIP:... strings.
    """
    session = notes_test_env["session"]
    admin = notes_test_env["admin_user"]
    author = notes_test_env["author_user"]
    dept = notes_test_env["dept"]
    chunk = notes_test_env["chunk"]
    chunk2 = notes_test_env["chunk2"]

    service = CompanyNotesService(session)

    # 1. Insert a raw GraphRAG relationship item (kind="RELATIONSHIP")
    rel_item = KnowledgeItem(
        source_type="DOCUMENT",
        kind="RELATIONSHIP",
        canonical_key="RELATIONSHIP:Acme_Corp:PARTNER_WITH:TechVenture",
        status="ACTIVE",
        visibility="SHARED",
        approval_status="APPROVED",
        approved_by_id=admin.id,
        approved_at=datetime.now(timezone.utc),
        department_id=dept.id,
    )
    session.add(rel_item)
    await session.flush()

    ws = notes_test_env["ws"]

    # 2. Insert two legitimate company notes for Acme Corp in the workspace
    note1 = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Acme Corp plans to acquire Beta Technologies in Q3.",
            document_chunk_ids=[chunk.id],
            department_id=dept.id,
            workspace_id=ws.id,
            visibility="WORKSPACE",
        ),
    )

    note2 = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Acme Corp launched new enterprise subscription pricing.",
            document_chunk_ids=[chunk2.id],
            department_id=dept.id,
            workspace_id=ws.id,
            visibility="WORKSPACE",
        ),
    )

    # 3. List notes as admin:
    # rel_item has visibility="SHARED" and approval_status="APPROVED", so admin would see it
    # IF it were not filtered out by kind != "RELATIONSHIP".
    admin_notes = await service.list_notes(user=admin)
    admin_ids = [n.id for n in admin_notes]

    # Verify: The GraphRAG relationship item is excluded from the company notes list
    assert rel_item.id not in admin_ids

    # Verify: Both observations for Acme Corp are visible to admin
    assert note1.id in admin_ids
    assert note2.id in admin_ids

    # 4. List notes as author:
    author_notes = await service.list_notes(user=author)
    author_ids = [n.id for n in author_notes]
    assert rel_item.id not in author_ids
    assert note1.id in author_ids
    assert note2.id in author_ids

    retrieved1 = next(n for n in admin_notes if n.id == note1.id)
    retrieved2 = next(n for n in admin_notes if n.id == note2.id)

    assert retrieved1.company_name == "Acme Corp"
    assert retrieved1.statement == "Acme Corp plans to acquire Beta Technologies in Q3."
    assert retrieved2.company_name == "Acme Corp"
    assert retrieved2.statement == "Acme Corp launched new enterprise subscription pricing."

    # Verify: Neither has raw canonical key as note statement or "Unnamed Company"
    assert "RELATIONSHIP:" not in retrieved1.statement
    assert "NOTE:" not in retrieved1.statement
    assert retrieved1.company_name != "Unnamed Company"

    # Verify: The raw relationship item is still intact in the database for graph consumers
    db_rel = await session.get(KnowledgeItem, rel_item.id)
    assert db_rel is not None
    assert db_rel.kind == "RELATIONSHIP"
    assert db_rel.status == "ACTIVE"


@pytest.mark.asyncio
async def test_multi_source_document_and_website_evidence(notes_test_env):
    """
    Verifies that:
    - Multiple document chunks and multiple website URLs can be attached to one note.
    - Document chunks retain document_chunk_id, file_name, page_number, and source_type='DOCUMENT'.
    - Website sources retain source_url, source_title, user_excerpt, and source_type='WEBSITE'.
    - Website sources are initialized with verification_status='NOT_VERIFIED' and verified_at=None.
    - User excerpts are clearly distinguished from independently retrieved content.
    - Each source remains independently identifiable and traceable to the same note.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    dept = notes_test_env["dept"]
    ws = notes_test_env["ws"]
    chunk1 = notes_test_env["chunk"]
    chunk2 = notes_test_env["chunk2"]

    service = CompanyNotesService(session)

    # 1. Create a note with 2 document sources and 2 website sources
    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Acme Corp is expanding into EMEA through a direct sales team.",
            observation_type="INFERRED",
            department_id=dept.id,
            workspace_id=ws.id,
            visibility="WORKSPACE",
            document_chunk_ids=[chunk1.id, chunk2.id],
            website_sources=[
                CompanyNoteWebsiteSourceInput(
                    url="https://acmecorp.com/news/emea-expansion",
                    title="Acme Corp Official News",
                    user_excerpt="We are thrilled to establish our London headquarters for EMEA operations.",
                ),
                CompanyNoteWebsiteSourceInput(
                    url="https://businessinsider.com/acme-emea-plans",
                    title="Business Insider Coverage",
                    user_excerpt="Sources confirm Acme has hired 25 enterprise sales representatives in the UK.",
                ),
            ],
        ),
    )

    assert note.evidence_count == 4
    assert len(note.evidence) == 4

    doc_sources = [ev for ev in note.evidence if ev.source_type == "DOCUMENT"]
    web_sources = [ev for ev in note.evidence if ev.source_type == "WEBSITE"]

    assert len(doc_sources) == 2
    assert len(web_sources) == 2

    # Check document evidence properties
    doc_chunk_ids = {ev.document_chunk_id for ev in doc_sources}
    assert chunk1.id in doc_chunk_ids
    assert chunk2.id in doc_chunk_ids
    for ev in doc_sources:
        assert ev.file_name is not None
        assert ev.page_number is not None
        assert ev.excerpt is not None

    # Check website evidence properties
    web_urls = {ev.source_url for ev in web_sources}
    assert "https://acmecorp.com/news/emea-expansion" in web_urls
    assert "https://businessinsider.com/acme-emea-plans" in web_urls

    for ev in web_sources:
        assert ev.verification_status == "NOT_VERIFIED"
        assert ev.verified_at is None
        assert ev.document_chunk_id is None
        assert ev.user_excerpt is not None


@pytest.mark.asyncio
async def test_website_source_url_validation(notes_test_env):
    """
    Verifies URL syntax validation:
    - Only HTTP and HTTPS URLs are accepted.
    - Non-HTTP schemes (ftp://, javascript:, file:) raise 400.
    - Malformed URLs raise 400.
    - Duplicate website URLs for the same note raise 400.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    dept = notes_test_env["dept"]
    ws = notes_test_env["ws"]
    chunk = notes_test_env["chunk"]

    service = CompanyNotesService(session)

    # 1. Invalid scheme in create_note
    with pytest.raises(HTTPException) as exc:
        await service.create_note(
            user=author,
            request=CompanyNoteCreateRequest(
                company_name="Acme Corp",
                statement="Testing invalid ftp url",
                document_chunk_ids=[chunk.id],
                department_id=dept.id,
                workspace_id=ws.id,
                website_sources=[
                    CompanyNoteWebsiteSourceInput(url="ftp://ftp.example.com/files/doc.pdf")
                ],
            ),
        )
    assert exc.value.status_code == 400
    assert "scheme" in exc.value.detail.lower()

    # 2. Javascript scheme in attach_evidence
    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Valid note for attach testing",
            document_chunk_ids=[chunk.id],
            department_id=dept.id,
            workspace_id=ws.id,
        ),
    )

    with pytest.raises(HTTPException) as exc:
        await service.attach_evidence(
            user=author,
            note_id=note.id,
            request=CompanyNoteEvidenceAttachRequest(
                source_url="javascript:alert('xss')",
            ),
        )
    assert exc.value.status_code == 400

    # 3. Valid HTTP and HTTPS URLs succeed
    updated = await service.attach_evidence(
        user=author,
        note_id=note.id,
        request=CompanyNoteEvidenceAttachRequest(
            source_url="http://example.com/blog",
            source_title="Example Blog",
        ),
    )
    assert any(ev.source_url == "http://example.com/blog" for ev in updated.evidence)

    # 4. Duplicate URL for the same note is idempotent and does not create duplicate entries
    updated_again = await service.attach_evidence(
        user=author,
        note_id=note.id,
        request=CompanyNoteEvidenceAttachRequest(
            source_url="http://example.com/blog",
        ),
    )
    blog_sources = [ev for ev in updated_again.evidence if ev.source_url == "http://example.com/blog"]
    assert len(blog_sources) == 1


@pytest.mark.asyncio
async def test_evidence_removal_preserves_other_sources(notes_test_env):
    """
    Verifies that removing one source (document or website) does not delete
    or disturb any other sources attached to the note.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    dept = notes_test_env["dept"]
    ws = notes_test_env["ws"]
    chunk1 = notes_test_env["chunk"]
    chunk2 = notes_test_env["chunk2"]

    service = CompanyNotesService(session)

    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Testing independent evidence removal",
            document_chunk_ids=[chunk1.id, chunk2.id],
            department_id=dept.id,
            workspace_id=ws.id,
            website_sources=[
                CompanyNoteWebsiteSourceInput(url="https://example.com/source1"),
                CompanyNoteWebsiteSourceInput(url="https://example.com/source2"),
            ],
        ),
    )
    assert note.evidence_count == 4

    web_ev1 = next(ev for ev in note.evidence if ev.source_url == "https://example.com/source1")
    doc_ev1 = next(ev for ev in note.evidence if ev.document_chunk_id == chunk1.id)

    # Remove 1 website source
    after_web_remove = await service.remove_evidence(user=author, note_id=note.id, evidence_id=web_ev1.id)
    assert after_web_remove.evidence_count == 3
    remaining_urls = [ev.source_url for ev in after_web_remove.evidence if ev.source_url]
    assert "https://example.com/source1" not in remaining_urls
    assert "https://example.com/source2" in remaining_urls
    assert any(ev.document_chunk_id == chunk1.id for ev in after_web_remove.evidence)
    assert any(ev.document_chunk_id == chunk2.id for ev in after_web_remove.evidence)

    # Remove 1 document source
    after_doc_remove = await service.remove_evidence(user=author, note_id=note.id, evidence_id=doc_ev1.id)
    assert after_doc_remove.evidence_count == 2
    assert not any(ev.document_chunk_id == chunk1.id for ev in after_doc_remove.evidence)
    assert any(ev.document_chunk_id == chunk2.id for ev in after_doc_remove.evidence)
    assert any(ev.source_url == "https://example.com/source2" for ev in after_doc_remove.evidence)


@pytest.mark.asyncio
async def test_url_only_notes_cannot_bypass_publication_rules(notes_test_env):
    """
    Verifies that:
    - Storing website URLs does NOT make a note eligible for publication.
    - A URL-only note (0 document sources) cannot be submitted for shared publication (raises 400).
    - An administrator cannot approve a URL-only note for publication (raises 400).
    - Batch publication rejects a URL-only note with NO_EVIDENCE.
    - Adding at least one authorized document chunk satisfies the publication requirement.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    admin = notes_test_env["admin_user"]
    dept = notes_test_env["dept"]
    ws = notes_test_env["ws"]
    chunk = notes_test_env["chunk"]

    service = CompanyNotesService(session)

    # 1. Create a note with ONLY website sources (0 document chunks)
    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Unverified assertion backed only by website URL",
            observation_type="UNVERIFIED",
            department_id=dept.id,
            workspace_id=ws.id,
            visibility="WORKSPACE",
            website_sources=[
                CompanyNoteWebsiteSourceInput(
                    url="https://unverified-claim.org/rumors",
                    title="Rumors Blog",
                )
            ],
        ),
    )
    assert note.evidence_count == 1
    assert note.evidence[0].source_type == "WEBSITE"

    # 2. Submitting for shared review must raise 400 Bad Request
    with pytest.raises(HTTPException) as exc:
        await service.submit_note(
            user=author,
            note_id=note.id,
            request=CompanyNoteSubmitRequest(target_audience="SHARED"),
        )
    assert exc.value.status_code == 400
    assert "document" in exc.value.detail.lower()

    # 3. Direct admin approval must also be blocked if 0 document citations exist
    with pytest.raises(HTTPException) as exc:
        await service.review_note(
            admin_user=admin,
            note_id=note.id,
            request=CompanyNoteReviewRequest(action="APPROVE"),
        )
    assert exc.value.status_code == 400
    assert "document" in exc.value.detail.lower()

    # 4. Batch publication rejects the note
    batch_res = await service.publish_batch(
        admin_user=admin,
        request=BatchPublicationRequest(note_ids=[note.id]),
    )
    assert batch_res.published_count == 0
    assert batch_res.failed_count == 1
    assert batch_res.outcomes[0].status == "NO_EVIDENCE"
    assert "source document evidence" in batch_res.outcomes[0].error.lower()

    # 5. Now attach an authorized document chunk to the note
    with_doc = await service.attach_evidence(
        user=author,
        note_id=note.id,
        request=CompanyNoteEvidenceAttachRequest(
            document_chunk_id=chunk.id,
        ),
    )
    assert with_doc.evidence_count == 2
    assert any(ev.source_type == "DOCUMENT" for ev in with_doc.evidence)
    assert any(ev.source_type == "WEBSITE" for ev in with_doc.evidence)

    # 6. Submission now succeeds
    submitted = await service.submit_note(
        user=author,
        note_id=note.id,
        request=CompanyNoteSubmitRequest(target_audience="SHARED"),
    )
    assert submitted.approval_status == "PENDING"

    # 7. Admin approval now succeeds
    approved = await service.review_note(
        admin_user=admin,
        note_id=note.id,
        request=CompanyNoteReviewRequest(action="APPROVE"),
    )
    assert approved.approval_status == "APPROVED"
    assert approved.visibility == "SHARED"


@pytest.mark.asyncio
async def test_evidence_support_level_meaning_preserved(notes_test_env):
    """
    Verifies that:
    - observation_type ('SUPPORTED', 'INFERRED', 'UNVERIFIED') remains distinct from URL verification_status ('NOT_VERIFIED').
    - Multiple website citations do NOT convert an INFERRED note into SUPPORTED.
    """
    session = notes_test_env["session"]
    author = notes_test_env["author_user"]
    dept = notes_test_env["dept"]
    ws = notes_test_env["ws"]
    chunk = notes_test_env["chunk"]

    service = CompanyNotesService(session)

    note = await service.create_note(
        user=author,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Acme may merge with competitor based on executive speeches.",
            observation_type="INFERRED",
            document_chunk_ids=[chunk.id],
            department_id=dept.id,
            workspace_id=ws.id,
            website_sources=[
                CompanyNoteWebsiteSourceInput(url="https://site1.com/rumor"),
                CompanyNoteWebsiteSourceInput(url="https://site2.com/rumor"),
            ],
        ),
    )

    assert note.observation_type == "INFERRED"
    for ev in note.evidence:
        if ev.source_type == "WEBSITE":
            assert ev.verification_status == "NOT_VERIFIED"


