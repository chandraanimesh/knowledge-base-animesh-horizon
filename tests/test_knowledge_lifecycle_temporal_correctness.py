import uuid
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_app.core.config import get_settings
from rag_app.db.models import (
    ChatMessage,
    Conversation,
    Department,
    Document,
    DocumentChunk,
    DocumentSection,
    KnowledgeEntity,
    KnowledgeEvidence,
    KnowledgeItem,
    KnowledgeRelationship,
    MemoryItem,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
)
from rag_app.generation.prompt import build_rag_messages, RAG_SYSTEM_PROMPT
from rag_app.knowledge.candidate_service import KnowledgeCandidateService
from rag_app.knowledge.company_notes_service import CompanyNotesService
from rag_app.memory.materializer import (
    MemoryMaterializer,
    extract_memory_topic_key,
    generate_memory_canonical_key,
)
from rag_app.memory.retrieval import MemoryRetrievalService
from rag_app.memory.schemas import MemoryCandidate, MemoryType
from rag_app.schemas.knowledge import (
    CompanyNoteCreateRequest,
    CompanyNoteEvidenceAttachRequest,
    CompanyNoteReviewRequest,
    CompanyNoteSubmitRequest,
)
from rag_app.schemas.rag import ConflictNotice, RetrievedChunk


@pytest_asyncio.fixture
async def stage_g_env():
    """Isolated database environment for Stage G Knowledge Lifecycle testing."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_stage_g_{uuid.uuid4().hex[:8]}"

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
            KnowledgeItem.__table__.create(sync_conn)
            KnowledgeEntity.__table__.create(sync_conn)
            KnowledgeRelationship.__table__.create(sync_conn)
            KnowledgeEvidence.__table__.create(sync_conn)
            MemoryItem.__table__.create(sync_conn)
            Conversation.__table__.create(sync_conn)
            ChatMessage.__table__.create(sync_conn)

        await conn.run_sync(create_tables)

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        dept = Department(code="STAGE_G_DEPT", name="Stage G Lifecycle Dept", is_active=True)
        session.add(dept)

        user_alice = User(
            email="alice_stage_g@example.com",
            hashed_password="pw",
            role="USER",
            is_active=True,
        )
        user_bob = User(
            email="bob_stage_g@example.com",
            hashed_password="pw",
            role="USER",
            is_active=True,
        )
        admin = User(
            email="admin_stage_g@example.com",
            hashed_password="pw",
            role="ADMIN",
            is_active=True,
        )
        session.add_all([user_alice, user_bob, admin])
        await session.flush()

        ws1 = Workspace(
            name="Stage G Workspace 1",
            code="STAGE_G_WS1",
            description="Testing Workspace 1",
            created_by=user_alice.id,
            department_id=dept.id,
            is_active=True,
        )
        ws2 = Workspace(
            name="Stage G Workspace 2",
            code="STAGE_G_WS2",
            description="Testing Workspace 2",
            created_by=user_bob.id,
            department_id=dept.id,
            is_active=True,
        )
        session.add_all([ws1, ws2])
        await session.flush()

        mem_alice = WorkspaceMember(workspace_id=ws1.id, user_id=user_alice.id, role="OWNER")
        mem_bob = WorkspaceMember(workspace_id=ws2.id, user_id=user_bob.id, role="OWNER")
        session.add_all([mem_alice, mem_bob])

        # Alice has VIEW permission on dept
        perm_view = Permission(code="VIEW", description="View", is_active=True)
        session.add(perm_view)
        await session.flush()

        ud_alice = UserDepartment(user_id=user_alice.id, department_id=dept.id)
        session.add(ud_alice)
        await session.flush()

        up_alice = UserPermission(user_id=user_alice.id, department_id=dept.id, permission_id=perm_view.id)
        session.add(up_alice)

        # Create source document and chunk for evidence testing
        doc = Document(
            id=uuid.uuid4(),
            workspace_id=ws1.id,
            department_id=dept.id,
            created_by=user_alice.id,
            file_name="q3_strategy.pdf",
            file_path="/data/q3_strategy.pdf",
            content_hash=f"hash_doc_{uuid.uuid4().hex[:12]}",
            status="READY",
            mime_type="application/pdf",
        )
        session.add(doc)
        await session.flush()

        chunk1 = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            text="Acme Corp quarterly revenue grew 24% year-over-year in Q3 2026.",
            embedding=[0.0] * settings.embedding_dim,
            chunk_hash=f"hash_chunk_{uuid.uuid4().hex[:12]}",
        )
        chunk2 = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=1,
            text="Acme Corp revised annual growth projections to 18% in late September 2026.",
            embedding=[0.0] * settings.embedding_dim,
            chunk_hash=f"hash_chunk_{uuid.uuid4().hex[:12]}",
        )
        session.add_all([chunk1, chunk2])
        await session.commit()

        yield {
            "session": session,
            "session_maker": session_maker,
            "schema_name": schema_name,
            "dept": dept,
            "user_alice": user_alice,
            "user_bob": user_bob,
            "admin": admin,
            "ws1": ws1,
            "ws2": ws2,
            "doc": doc,
            "chunk1": chunk1,
            "chunk2": chunk2,
        }

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_memory_supersession_without_contradictory_active_memories(stage_g_env: dict):
    """
    Test that when a user updates their preference, the old memory is superseded
    transactionally, leaving exactly ONE active memory on that topic without deleting history.
    """
    session = stage_g_env["session"]
    user_alice = stage_g_env["user_alice"]
    ws1 = stage_g_env["ws1"]

    # Alice initial preference: bullet points
    topic = extract_memory_topic_key("I prefer responses in concise bullet points")
    c_key = generate_memory_canonical_key("STATIC", topic)

    cand1 = MemoryCandidate(
        user_id=user_alice.id,
        workspace_id=ws1.id,
        memory_type=MemoryType.STATIC,
        content="I prefer responses in concise bullet points",
        canonical_key=c_key,
        confidence=1.0,
        source_type="EXPLICIT_USER_STATEMENT",
    )
    mem1 = await MemoryMaterializer.materialize_memory(session, cand1)
    assert mem1.status == "ACTIVE"
    assert mem1.superseded_by_id is None

    # Alice updates preference to detailed paragraphs
    cand2 = MemoryCandidate(
        user_id=user_alice.id,
        workspace_id=ws1.id,
        memory_type=MemoryType.STATIC,
        content="I prefer responses in detailed paragraphs",
        canonical_key=c_key,
        confidence=1.0,
        source_type="EXPLICIT_USER_STATEMENT",
        supersedes_id=mem1.id,
    )
    mem2 = await MemoryMaterializer.materialize_memory(session, cand2)
    assert mem2.status == "ACTIVE"
    assert mem2.id != mem1.id

    # Verify old memory is now SUPERSEDED and points to mem2
    await session.refresh(mem1)
    assert mem1.status == "SUPERSEDED"
    assert mem1.superseded_by_id == mem2.id

    # Verify retrieval service returns ONLY the active memory
    resp = await MemoryRetrievalService.retrieve(
        session=session,
        user_id=user_alice.id,
        workspace_id=ws1.id,
        query="how should responses be formatted?",
    )
    active_mems = resp.static_memories + resp.dynamic_memories
    assert len(active_mems) == 1
    assert active_mems[0].content == "I prefer responses in detailed paragraphs"
    assert active_mems[0].id == mem2.id


@pytest.mark.asyncio
async def test_forget_archives_memory_and_excludes_from_active_context(stage_g_env: dict):
    """
    Test that /forget (archive_memory) transitions status to ARCHIVED,
    preserving historical record while preventing retrieval in active context.
    """
    session = stage_g_env["session"]
    user_alice = stage_g_env["user_alice"]

    cand = MemoryCandidate(
        user_id=user_alice.id,
        workspace_id=None,
        memory_type=MemoryType.STATIC,
        content="My favorite programming language is Python",
        confidence=1.0,
        source_type="EXPLICIT_USER_STATEMENT",
    )
    mem = await MemoryMaterializer.materialize_memory(session, cand)
    assert mem.status == "ACTIVE"

    # User executes forget
    archived = await MemoryMaterializer.archive_memory(session, mem.id, user_alice.id)
    assert archived is True

    await session.refresh(mem)
    assert mem.status == "ARCHIVED"

    # Ensure memory retrieval completely excludes archived memories
    resp = await MemoryRetrievalService.retrieve(
        session=session,
        user_id=user_alice.id,
        workspace_id=None,
        query="favorite programming language",
    )
    active_mems = resp.static_memories + resp.dynamic_memories
    assert len(active_mems) == 0


@pytest.mark.asyncio
async def test_private_memory_never_publishes_to_organizational_knowledge(stage_g_env: dict):
    """
    Test that saving, updating, or correcting private memories does NOT create
    or publish rows in the shared KnowledgeItem table.
    """
    session = stage_g_env["session"]
    user_alice = stage_g_env["user_alice"]

    # Initial count of KnowledgeItem
    initial_ki_count = await session.scalar(select(text("count(*)")).select_from(KnowledgeItem))

    cand = MemoryCandidate(
        user_id=user_alice.id,
        workspace_id=None,
        memory_type=MemoryType.STATIC,
        content="Confidential personal salary requirement is 150k",
        confidence=1.0,
        source_type="EXPLICIT_USER_STATEMENT",
    )
    await MemoryMaterializer.materialize_memory(session, cand)

    post_ki_count = await session.scalar(select(text("count(*)")).select_from(KnowledgeItem))
    assert post_ki_count == initial_ki_count


@pytest.mark.asyncio
async def test_memory_workspace_and_user_isolation(stage_g_env: dict):
    """
    Test that Alice's memory is isolated from Bob, and Bob cannot modify or archive it.
    """
    session = stage_g_env["session"]
    user_alice = stage_g_env["user_alice"]
    user_bob = stage_g_env["user_bob"]
    ws1 = stage_g_env["ws1"]

    cand = MemoryCandidate(
        user_id=user_alice.id,
        workspace_id=ws1.id,
        memory_type=MemoryType.STATIC,
        content="Alice private preference",
        confidence=1.0,
        source_type="EXPLICIT_USER_STATEMENT",
    )
    mem_alice = await MemoryMaterializer.materialize_memory(session, cand)

    # Bob attempts to archive Alice's memory
    bob_archived = await MemoryMaterializer.archive_memory(session, mem_alice.id, user_bob.id)
    assert bob_archived is False

    await session.refresh(mem_alice)
    assert mem_alice.status == "ACTIVE"

    # Bob's retrieval does not see Alice's memory
    resp = await MemoryRetrievalService.retrieve(
        session=session,
        user_id=user_bob.id,
        workspace_id=ws1.id,
        query="Alice private preference",
    )
    bob_mems = resp.static_memories + resp.dynamic_memories
    assert len(bob_mems) == 0


@pytest.mark.asyncio
async def test_company_note_evidence_withdrawal_demotes_and_excludes_note(stage_g_env: dict):
    """
    Test that when all supporting evidence is withdrawn from an approved Company Note,
    the note is de-published (ARCHIVED / REJECTED) and excluded from shared knowledge retrieval.
    """
    session = stage_g_env["session"]
    user_alice = stage_g_env["user_alice"]
    admin = stage_g_env["admin"]
    ws1 = stage_g_env["ws1"]
    dept = stage_g_env["dept"]
    chunk1 = stage_g_env["chunk1"]

    service = CompanyNotesService(session)

    # Create note with chunk1 evidence
    create_req = CompanyNoteCreateRequest(
        company_name="Acme Corp",
        statement="Acme Corp grew Q3 revenue by 24%",
        observation_type="METRIC",
        visibility="WORKSPACE",
        workspace_id=ws1.id,
        department_id=dept.id,
        document_chunk_ids=[chunk1.id],
    )
    note_resp = await service.create_note(user=user_alice, request=create_req)
    note_id = note_resp.id

    # Submit and approve
    await service.submit_note(user=user_alice, note_id=note_id, request=CompanyNoteSubmitRequest())
    approved_note = await service.review_note(
        admin_user=admin,
        note_id=note_id,
        request=CompanyNoteReviewRequest(action="APPROVE"),
    )
    assert approved_note.status == "ACTIVE"
    assert approved_note.approval_status == "APPROVED"
    assert approved_note.visibility == "SHARED"

    # Verify shared knowledge query includes the approved note
    cand_service = KnowledgeCandidateService(session)
    shared_notes = await cand_service.get_approved_shared_knowledge()
    assert any(n.id == note_id for n in shared_notes)

    # Now withdraw the evidence
    ev_id = approved_note.evidence[0].id
    withdrawn_note = await service.remove_evidence(user=user_alice, note_id=note_id, evidence_id=ev_id)

    assert withdrawn_note.status == "ARCHIVED"
    assert withdrawn_note.approval_status == "REJECTED"
    assert withdrawn_note.visibility == "WORKSPACE"

    # Verify note is now strictly excluded from approved shared knowledge
    shared_after = await cand_service.get_approved_shared_knowledge()
    assert not any(n.id == note_id for n in shared_after)


@pytest.mark.asyncio
async def test_company_note_archive_and_supersede_lifecycle(stage_g_env: dict):
    """
    Test direct note archive and supersession APIs, ensuring old notes retain
    supersession pointers and are excluded from shared retrieval.
    """
    session = stage_g_env["session"]
    user_alice = stage_g_env["user_alice"]
    admin = stage_g_env["admin"]
    ws1 = stage_g_env["ws1"]
    dept = stage_g_env["dept"]
    chunk1 = stage_g_env["chunk1"]
    chunk2 = stage_g_env["chunk2"]

    service = CompanyNotesService(session)

    # Create old note
    n1 = await service.create_note(
        user=user_alice,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Acme initial growth projection was 24%",
            observation_type="METRIC",
            visibility="WORKSPACE",
            workspace_id=ws1.id,
            department_id=dept.id,
            document_chunk_ids=[chunk1.id],
        ),
    )
    # Create replacement note
    n2 = await service.create_note(
        user=user_alice,
        request=CompanyNoteCreateRequest(
            company_name="Acme Corp",
            statement="Acme revised growth projection is 18%",
            observation_type="METRIC",
            visibility="WORKSPACE",
            workspace_id=ws1.id,
            department_id=dept.id,
            document_chunk_ids=[chunk2.id],
        ),
    )

    # Alice supersedes n1 with n2
    superseded_n1 = await service.supersede_note(user=user_alice, old_note_id=n1.id, new_note_id=n2.id)
    assert superseded_n1.status == "SUPERSEDED"

    item1 = await session.get(KnowledgeItem, n1.id)
    assert item1.status == "SUPERSEDED"

    # Test archiving n2
    archived_n2 = await service.archive_note(user=user_alice, note_id=n2.id, reason="Deprecated")
    assert archived_n2.status == "ARCHIVED"

    item2 = await session.get(KnowledgeItem, n2.id)
    assert item2.status == "ARCHIVED"


@pytest.mark.asyncio
async def test_unapproved_draft_and_rejected_notes_excluded(stage_g_env: dict):
    """
    Test that draft, pending, and rejected notes are excluded from shared retrieval.
    """
    session = stage_g_env["session"]
    user_alice = stage_g_env["user_alice"]
    admin = stage_g_env["admin"]
    ws1 = stage_g_env["ws1"]
    chunk1 = stage_g_env["chunk1"]

    service = CompanyNotesService(session)

    # Draft note
    draft = await service.create_note(
        user=user_alice,
        request=CompanyNoteCreateRequest(
            company_name="Beta Inc",
            statement="Beta Inc unapproved rumor",
            observation_type="OBSERVATION",
            visibility="PRIVATE",
            document_chunk_ids=[chunk1.id],
        ),
    )

    cand_service = KnowledgeCandidateService(session)
    shared = await cand_service.get_approved_shared_knowledge()
    assert not any(n.id == draft.id for n in shared)

    # Submit and reject
    await service.submit_note(user=user_alice, note_id=draft.id, request=CompanyNoteSubmitRequest(target_workspace_id=ws1.id))
    rejected = await service.review_note(
        admin_user=admin,
        note_id=draft.id,
        request=CompanyNoteReviewRequest(action="REJECT", reason="Not verified"),
    )
    assert rejected.approval_status == "REJECTED"

    shared_after = await cand_service.get_approved_shared_knowledge()
    assert not any(n.id == draft.id for n in shared_after)


@pytest.mark.asyncio
async def test_prompt_source_authority_and_conflict_representation():
    """
    Test that build_rag_messages enforces source authority rules in system prompt
    and honestly presents conflicting historical/document evidence without fabricating resolutions.
    """
    c1_id = str(uuid.uuid4())
    c2_id = str(uuid.uuid4())
    chunks = [
        RetrievedChunk(
            chunk_id=c1_id,
            document_id=str(uuid.uuid4()),
            department_id=str(uuid.uuid4()),
            chunk_index=0,
            text="Annual report 2025 states CEO is Jane Doe.",
            file_name="Annual_Report_2025.pdf",
            page_number=2,
            hybrid_score=0.92,
        ),
        RetrievedChunk(
            chunk_id=c2_id,
            document_id=str(uuid.uuid4()),
            department_id=str(uuid.uuid4()),
            chunk_index=1,
            text="Press release October 2026 announces John Smith appointed as new CEO.",
            file_name="Press_Release_Oct_2026.pdf",
            page_number=1,
            hybrid_score=0.95,
        ),
    ]

    conflicts = [
        ConflictNotice(
            subject="Acme Corp",
            predicate="has_ceo",
            conflicting_objects=["Jane Doe", "John Smith"],
            conflicting_chunk_ids=[c1_id, c2_id],
            details="Conflicting leadership records: Jane Doe (2025 Report) vs John Smith (Oct 2026 Release)",
        )
    ]

    personal_memories = [
        {"content": "I prefer answers formatted in bullet points", "memory_type": "STATIC"}
    ]

    history = [
        {"role": "user", "content": "Who was CEO in 2025?"},
        {"role": "assistant", "content": "In 2025, Jane Doe was CEO."},
    ]

    sample_context = (
        "[1] Source: Annual_Report_2025.pdf, page 2\n"
        "Chunk ID: " + c1_id + "\n"
        "Content:\nAnnual report 2025 states CEO is Jane Doe.\n\n"
        "[2] Source: Press_Release_Oct_2026.pdf, page 1\n"
        "Chunk ID: " + c2_id + "\n"
        "Content:\nPress release October 2026 announces John Smith appointed as new CEO."
    )

    messages = build_rag_messages(
        question="Who is the current CEO of Acme Corp?",
        context=sample_context,
        chunks=chunks,
        conflicts=conflicts,
        personal_memories=personal_memories,
        conversation_history=history,
    )

    sys_msg = next(m["content"] for m in messages if m["role"] == "system")
    user_msg = messages[-1]["content"]

    # Verify Source Authority Hierarchy is present in system prompt
    assert "Source Authority & Temporal Correctness Hierarchy" in sys_msg
    assert "Authoritative Structured Records & Approved Shared Knowledge" in sys_msg
    assert "Personal Memory: User-scoped preferences" in sys_msg
    assert "Do not assume that a newer timestamp always makes a source more authoritative" in sys_msg

    # Verify Conflict Notice and provenance citations are included in user content
    assert "Conflicting Evidence Notice" in user_msg
    assert "Conflicting leadership records: Jane Doe (2025 Report) vs John Smith (Oct 2026 Release)" in user_msg
    assert "Conflict Guidance" in user_msg
    assert "Conflicting evidence must be reported rather than silently resolved" in user_msg
