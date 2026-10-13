import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import (
    User,
    Department,
    Permission,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
    Conversation,
    ChatMessage,
    Document,
    DocumentChunk,
    KnowledgeItem,
    KnowledgeEntity,
    KnowledgeEvidence,
)
from rag_app.knowledge.candidate_service import KnowledgeCandidateService
from rag_app.schemas.knowledge import CandidateCreateRequest
from rag_app.schemas.rag import RetrievedChunk, ChunkProvenance, RetrievalResponse
from rag_app.research.graph import retrieve_rag, ResearchState


@pytest_asyncio.fixture
async def stage_d_test_env():
    """
    Creates an isolated PostgreSQL test schema for Stage D verification.
    Clean teardown ensures no test contamination.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_stage_d_{uuid.uuid4().hex[:8]}"

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
            DocumentChunk.__table__.create(sync_conn)
            Conversation.__table__.create(sync_conn)
            ChatMessage.__table__.create(sync_conn)
            KnowledgeItem.__table__.create(sync_conn)
            KnowledgeEntity.__table__.create(sync_conn)
            KnowledgeEvidence.__table__.create(sync_conn)

        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:8]
    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # Create two departments: Finance and HR
        dept_fin = Department(name=f"Finance {suffix}", code=f"FIN_{suffix}")
        dept_hr = Department(name=f"HR {suffix}", code=f"HR_{suffix}")
        session.add_all([dept_fin, dept_hr])
        await session.flush()

        # Create users: Admin, Finance User, HR User
        admin_user = User(email=f"admin_{suffix}@example.com", hashed_password="pwd", role="ADMIN")
        fin_user = User(email=f"fin_user_{suffix}@example.com", hashed_password="pwd", role="USER")
        hr_user = User(email=f"hr_user_{suffix}@example.com", hashed_password="pwd", role="USER")
        session.add_all([admin_user, fin_user, hr_user])
        await session.flush()

        # VIEW Permission
        perm_view = Permission(code="VIEW")
        session.add(perm_view)
        await session.flush()

        # Finance user has access to Finance department only
        session.add(UserDepartment(user_id=fin_user.id, department_id=dept_fin.id))
        session.add(UserPermission(user_id=fin_user.id, department_id=dept_fin.id, permission_id=perm_view.id))

        # HR user has access to HR department only
        session.add(UserDepartment(user_id=hr_user.id, department_id=dept_hr.id))
        session.add(UserPermission(user_id=hr_user.id, department_id=dept_hr.id, permission_id=perm_view.id))
        await session.flush()

        # Workspaces
        ws_fin = Workspace(name=f"Fin WS {suffix}", code=f"ws_fin_{suffix}", department_id=dept_fin.id, created_by=admin_user.id)
        ws_hr = Workspace(name=f"HR WS {suffix}", code=f"ws_hr_{suffix}", department_id=dept_hr.id, created_by=admin_user.id)
        session.add_all([ws_fin, ws_hr])
        await session.flush()

        session.add(WorkspaceMember(workspace_id=ws_fin.id, user_id=fin_user.id, role="EDITOR"))
        session.add(WorkspaceMember(workspace_id=ws_hr.id, user_id=hr_user.id, role="EDITOR"))
        await session.flush()

        # Conversations
        conv_fin = Conversation(user_id=fin_user.id, workspace_id=ws_fin.id, title="Fin Discussions")
        msg_fin = ChatMessage(conversation_id=conv_fin.id, role="assistant", content="Approved discount is 15% for Annual enterprise deals.")
        conv_fin.messages.append(msg_fin)
        session.add(conv_fin)

        # Personal conversation without workspace for HR user
        conv_personal = Conversation(user_id=hr_user.id, workspace_id=None, title="Personal Diary")
        msg_personal = ChatMessage(conversation_id=conv_personal.id, role="user", content="My personal secret notes.")
        conv_personal.messages.append(msg_personal)
        session.add(conv_personal)

        # Document & chunk for Finance
        doc_fin = Document(file_name="finance_handbook.pdf", file_path="/tmp/fin.pdf", content_hash=f"h_{suffix}", department_id=dept_fin.id, created_by=admin_user.id, workspace_id=ws_fin.id)
        session.add(doc_fin)
        await session.flush()

        chunk_fin = DocumentChunk(
            document_id=doc_fin.id,
            chunk_index=0,
            text="General finance billing guidelines and payment terms.",
            embedding=[0.0] * settings.embedding_dim,
            chunk_hash=f"ch_{suffix}",
        )
        session.add(chunk_fin)
        await session.commit()

        yield {
            "schema_name": schema_name,
            "session": session,
            "dept_fin_id": dept_fin.id,
            "dept_hr_id": dept_hr.id,
            "admin_user": admin_user,
            "fin_user": fin_user,
            "hr_user": hr_user,
            "ws_fin_id": ws_fin.id,
            "conv_fin_id": conv_fin.id,
            "msg_fin_id": msg_fin.id,
            "conv_personal_id": conv_personal.id,
            "msg_personal_id": msg_personal.id,
            "doc_fin_id": doc_fin.id,
            "chunk_fin_id": chunk_fin.id,
        }

    # Teardown
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


# Helper to publish an approved shared knowledge item
async def _create_approved_shared_item(
    session: AsyncSession,
    service: KnowledgeCandidateService,
    user: User,
    admin: User,
    conv_id: uuid.UUID,
    msg_id: uuid.UUID,
    statement: str,
    topic: str,
) -> KnowledgeItem:
    cand = await service.create_candidate(
        user=user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement=statement,
            topic=topic,
            confidence=0.95,
        ),
    )
    await service.submit_candidate(user=user, candidate_id=cand.id)
    approved = await service.approve_candidate(user=admin, candidate_id=cand.id)
    item = await session.get(KnowledgeItem, approved.id)
    return item


@pytest.mark.asyncio
async def test_global_chat_retrieves_approved_shared_item(stage_d_test_env: dict):
    """
    Test 1: Global Chat retrieves a relevant approved shared item.
    Verified: visibility='SHARED', approval_status='APPROVED', status='ACTIVE', valid evidence.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    fin_user = stage_d_test_env["fin_user"]

    item = await _create_approved_shared_item(
        session=session,
        service=service,
        user=fin_user,
        admin=admin,
        conv_id=stage_d_test_env["conv_fin_id"],
        msg_id=stage_d_test_env["msg_fin_id"],
        statement="Annual contract discount is capped strictly at 15%.",
        topic="discount_policy",
    )

    chunks = await service.retrieve_shared_knowledge_as_chunks(
        query="What is the annual contract discount policy?",
        allowed_departments=[stage_d_test_env["dept_fin_id"]],
        limit=5,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == f"shared_kn_{item.id}"
    assert "Annual contract discount is capped strictly at 15%." in chunks[0].text
    assert "[Approved Shared Organizational Knowledge: discount_policy]" in chunks[0].text
    assert chunks[0].department_id == str(stage_d_test_env["dept_fin_id"])


@pytest.mark.asyncio
async def test_lifecycle_exclusions_draft_pending_rejected_archived(stage_d_test_env: dict):
    """
    Test 2: DRAFT, PENDING, REJECTED, and ARCHIVED items are excluded from retrieval.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    fin_user = stage_d_test_env["fin_user"]

    # 1. Draft
    draft_cand = await service.create_candidate(
        user=fin_user,
        request=CandidateCreateRequest(
            conversation_id=stage_d_test_env["conv_fin_id"],
            chat_message_id=stage_d_test_env["msg_fin_id"],
            statement="Draft policy: Unlimited expense accounts for managers.",
            topic="expenses",
        ),
    )

    # 2. Pending
    pending_cand = await service.create_candidate(
        user=fin_user,
        request=CandidateCreateRequest(
            conversation_id=stage_d_test_env["conv_fin_id"],
            chat_message_id=stage_d_test_env["msg_fin_id"],
            statement="Pending policy: Extra vacation days in winter.",
            topic="vacation",
        ),
    )
    await service.submit_candidate(user=fin_user, candidate_id=pending_cand.id)

    # 3. Rejected
    rej_cand = await service.create_candidate(
        user=fin_user,
        request=CandidateCreateRequest(
            conversation_id=stage_d_test_env["conv_fin_id"],
            chat_message_id=stage_d_test_env["msg_fin_id"],
            statement="Rejected policy: Free company vehicles for all.",
            topic="vehicles",
        ),
    )
    await service.reject_candidate(user=admin, candidate_id=rej_cand.id)

    # 4. Archived
    arch_item = await _create_approved_shared_item(
        session=session,
        service=service,
        user=fin_user,
        admin=admin,
        conv_id=stage_d_test_env["conv_fin_id"],
        msg_id=stage_d_test_env["msg_fin_id"],
        statement="Archived policy: Old 2020 office parking rules.",
        topic="parking",
    )
    arch_item.status = "ARCHIVED"
    await session.commit()

    # Query for each
    for q in ["expenses", "vacation", "vehicles", "parking"]:
        res = await service.retrieve_shared_knowledge_as_chunks(
            query=q,
            allowed_departments=[stage_d_test_env["dept_fin_id"]],
            limit=5,
        )
        assert len(res) == 0, f"Item with query '{q}' should be excluded due to lifecycle"


@pytest.mark.asyncio
async def test_inactive_and_unsupported_items_excluded(stage_d_test_env: dict):
    """
    Test 3: Inactive items and items whose supporting evidence was removed are strictly excluded.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    fin_user = stage_d_test_env["fin_user"]

    item = await _create_approved_shared_item(
        session=session,
        service=service,
        user=fin_user,
        admin=admin,
        conv_id=stage_d_test_env["conv_fin_id"],
        msg_id=stage_d_test_env["msg_fin_id"],
        statement="Supported policy: Per-diem food allowance is $75.",
        topic="per_diem",
    )

    # Verify retrievable initially
    res_before = await service.retrieve_shared_knowledge_as_chunks(
        query="per-diem food allowance",
        allowed_departments=[stage_d_test_env["dept_fin_id"]],
    )
    assert len(res_before) == 1

    # Remove all evidence
    ev_stmt = select(KnowledgeEvidence).where(KnowledgeEvidence.knowledge_item_id == item.id)
    ev_rows = (await session.scalars(ev_stmt)).all()
    for ev in ev_rows:
        await session.delete(ev)
    await session.commit()

    # Verify excluded now that evidence is absent
    res_after = await service.retrieve_shared_knowledge_as_chunks(
        query="per-diem food allowance",
        allowed_departments=[stage_d_test_env["dept_fin_id"]],
    )
    assert len(res_after) == 0, "Item without supporting evidence must be excluded"


@pytest.mark.asyncio
async def test_regular_user_cannot_retrieve_unauthorized_departments(stage_d_test_env: dict):
    """
    Test 4: Regular users cannot retrieve items from unauthorized departments.
    HR user (only authorized for HR dept) cannot retrieve Finance approved knowledge.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    fin_user = stage_d_test_env["fin_user"]

    # Publish Finance approved knowledge
    await _create_approved_shared_item(
        session=session,
        service=service,
        user=fin_user,
        admin=admin,
        conv_id=stage_d_test_env["conv_fin_id"],
        msg_id=stage_d_test_env["msg_fin_id"],
        statement="Finance confidential billing threshold is $100,000.",
        topic="billing_threshold",
    )

    # HR user queries with only HR dept authorized
    hr_res = await service.retrieve_shared_knowledge_as_chunks(
        query="billing threshold",
        allowed_departments=[stage_d_test_env["dept_hr_id"]],
    )
    assert len(hr_res) == 0, "HR user cannot retrieve Finance approved knowledge"

    # User with empty allowed departments
    empty_res = await service.retrieve_shared_knowledge_as_chunks(
        query="billing threshold",
        allowed_departments=[],
    )
    assert len(empty_res) == 0, "Empty allowed_departments must fail closed"


@pytest.mark.asyncio
async def test_null_department_cannot_bypass_authorization(stage_d_test_env: dict):
    """
    Test 5: A NULL department cannot bypass department authorization.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    fin_user = stage_d_test_env["fin_user"]

    item = await _create_approved_shared_item(
        session=session,
        service=service,
        user=fin_user,
        admin=admin,
        conv_id=stage_d_test_env["conv_fin_id"],
        msg_id=stage_d_test_env["msg_fin_id"],
        statement="Global unassigned corporate guideline.",
        topic="guideline",
    )

    # Force department_id to None
    item.department_id = None
    await session.commit()

    # Query with regular allowed departments
    res = await service.retrieve_shared_knowledge_as_chunks(
        query="corporate guideline",
        allowed_departments=[stage_d_test_env["dept_fin_id"]],
    )
    assert len(res) == 0, "NULL department cannot match allowed_departments or bypass RBAC"


@pytest.mark.asyncio
async def test_admin_global_chat_retrieves_shared_records_without_raw_chat_messages(stage_d_test_env: dict):
    """
    Test 6: ADMIN Global Chat retrieves approved shared records without retrieving raw cross-workspace messages.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    fin_user = stage_d_test_env["fin_user"]

    # Create approved shared record
    await _create_approved_shared_item(
        session=session,
        service=service,
        user=fin_user,
        admin=admin,
        conv_id=stage_d_test_env["conv_fin_id"],
        msg_id=stage_d_test_env["msg_fin_id"],
        statement="Official approved travel policy: Flight class is Economy for domestic trips.",
        topic="travel_flight",
    )

    # Mock retrieval_service returning document chunks
    mock_retrieval_service = AsyncMock()
    mock_retrieval_service.retrieve.return_value = RetrievalResponse(
        query="domestic flight class",
        results=[],
        strategy_trace={},
        chunk_provenance={},
        chunk_hierarchy={},
    )

    state: ResearchState = {
        "query": "domestic flight class",
        "top_k": 5,
        "rerank": False,
        "allowed_departments": [stage_d_test_env["dept_fin_id"], stage_d_test_env["dept_hr_id"]],
        "workspace_id": None,  # Global Chat
    }
    config = {
        "configurable": {
            "retrieval_service": mock_retrieval_service,
            "db_session": session,
        }
    }

    node_out = await retrieve_rag(state, config)
    retrieved = node_out["retrieved_chunks"]

    assert len(retrieved) == 1
    assert retrieved[0].chunk_id.startswith("shared_kn_")
    assert "Flight class is Economy for domestic trips." in retrieved[0].text
    # Ensure no raw ChatMessage fields or unapproved raw messages
    assert not any(getattr(c, "role", None) for c in retrieved)
    assert node_out["strategy_trace"]["shared_knowledge"] == 1


@pytest.mark.asyncio
async def test_workspace_chat_retrieval_unmodified(stage_d_test_env: dict):
    """
    Test 7: Workspace-chat retrieval remains intact and does not invoke shared-knowledge retrieval.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    fin_user = stage_d_test_env["fin_user"]

    # Publish approved shared knowledge
    await _create_approved_shared_item(
        session=session,
        service=service,
        user=fin_user,
        admin=admin,
        conv_id=stage_d_test_env["conv_fin_id"],
        msg_id=stage_d_test_env["msg_fin_id"],
        statement="Some shared organizational fact.",
        topic="fact",
    )

    mock_chunk = RetrievedChunk(
        chunk_id="doc_chunk_1",
        document_id="doc_1",
        department_id=str(stage_d_test_env["dept_fin_id"]),
        file_name="ws_doc.txt",
        chunk_index=0,
        text="Workspace document content.",
    )
    mock_retrieval_service = AsyncMock()
    mock_retrieval_service.retrieve.return_value = RetrievalResponse(
        query="shared fact",
        results=[mock_chunk],
        strategy_trace={"vector": 1},
        chunk_provenance={},
        chunk_hierarchy={},
    )

    # Workspace Chat (workspace_id is not None)
    state: ResearchState = {
        "query": "shared fact",
        "top_k": 5,
        "rerank": False,
        "allowed_departments": [stage_d_test_env["dept_fin_id"]],
        "workspace_id": stage_d_test_env["ws_fin_id"],  # Workspace chat
    }
    config = {
        "configurable": {
            "retrieval_service": mock_retrieval_service,
            "db_session": session,
        }
    }

    node_out = await retrieve_rag(state, config)
    retrieved = node_out["retrieved_chunks"]

    # In workspace chat, retrieve_rag returns ONLY document results from retrieval_service
    assert len(retrieved) == 1
    assert retrieved[0].chunk_id == "doc_chunk_1"
    assert "shared_knowledge" not in node_out["strategy_trace"]


@pytest.mark.asyncio
async def test_no_original_private_message_text_leaks(stage_d_test_env: dict):
    """
    Test 8: No original private message text enters Global Chat context or sources through the shared-knowledge path.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    fin_user = stage_d_test_env["fin_user"]

    raw_message_secret = "Confidential chat conversation rambling and personal thoughts that should not leak."
    msg = await session.get(ChatMessage, stage_d_test_env["msg_fin_id"])
    msg.content = raw_message_secret
    await session.commit()

    # Formulate a clean candidate statement
    item = await _create_approved_shared_item(
        session=session,
        service=service,
        user=fin_user,
        admin=admin,
        conv_id=stage_d_test_env["conv_fin_id"],
        msg_id=stage_d_test_env["msg_fin_id"],
        statement="Approved business rule: Vendor invoices require dual sign-off over $50k.",
        topic="procurement",
    )

    chunks = await service.retrieve_shared_knowledge_as_chunks(
        query="vendor invoices sign-off",
        allowed_departments=[stage_d_test_env["dept_fin_id"]],
    )

    assert len(chunks) == 1
    chunk_text = chunks[0].text
    assert "Vendor invoices require dual sign-off over $50k." in chunk_text
    assert raw_message_secret not in chunk_text, "Raw private chat message text must not be exposed"


@pytest.mark.asyncio
async def test_empty_shared_retrieval_does_not_break_existing_rag(stage_d_test_env: dict):
    """
    Test 9: Empty shared retrieval does not break existing RAG behavior.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    mock_chunk = RetrievedChunk(
        chunk_id="doc_chunk_rag",
        document_id="doc_1",
        department_id=str(stage_d_test_env["dept_fin_id"]),
        file_name="fin_manual.pdf",
        chunk_index=0,
        text="Normal document content.",
    )
    mock_retrieval_service = AsyncMock()
    mock_retrieval_service.retrieve.return_value = RetrievalResponse(
        query="nonexistent term query",
        results=[mock_chunk],
        strategy_trace={"hybrid": 1},
        chunk_provenance={},
        chunk_hierarchy={},
    )

    state: ResearchState = {
        "query": "nonexistent term query",
        "top_k": 5,
        "rerank": False,
        "allowed_departments": [stage_d_test_env["dept_fin_id"]],
        "workspace_id": None,
    }
    config = {
        "configurable": {
            "retrieval_service": mock_retrieval_service,
            "db_session": session,
        }
    }

    node_out = await retrieve_rag(state, config)
    retrieved = node_out["retrieved_chunks"]

    assert len(retrieved) == 1
    assert retrieved[0].chunk_id == "doc_chunk_rag"
    assert "shared_knowledge" not in node_out["strategy_trace"]


@pytest.mark.asyncio
async def test_shared_retrieval_exception_fails_safely_to_rag(stage_d_test_env: dict):
    """
    Test 10: Shared retrieval exceptions do not cause unauthorized fallback or fabricated answers.
    Normal RAG proceeds with document chunks.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    mock_chunk = RetrievedChunk(
        chunk_id="doc_chunk_safe",
        document_id="doc_1",
        department_id=str(stage_d_test_env["dept_fin_id"]),
        file_name="fin_manual.pdf",
        chunk_index=0,
        text="Safe document evidence.",
    )
    mock_retrieval_service = AsyncMock()
    mock_retrieval_service.retrieve.return_value = RetrievalResponse(
        query="any query",
        results=[mock_chunk],
        strategy_trace={"hybrid": 1},
        chunk_provenance={},
        chunk_hierarchy={},
    )

    state: ResearchState = {
        "query": "any query",
        "top_k": 5,
        "rerank": False,
        "allowed_departments": [stage_d_test_env["dept_fin_id"]],
        "workspace_id": None,
    }

    # Patch retrieve_shared_knowledge_as_chunks to raise an exception
    with patch("rag_app.knowledge.candidate_service.KnowledgeCandidateService.retrieve_shared_knowledge_as_chunks", side_effect=RuntimeError("DB query timeout")):
        config = {
            "configurable": {
                "retrieval_service": mock_retrieval_service,
                "db_session": session,
            }
        }
        node_out = await retrieve_rag(state, config)
        retrieved = node_out["retrieved_chunks"]

        assert len(retrieved) == 1
        assert retrieved[0].chunk_id == "doc_chunk_safe"


@pytest.mark.asyncio
async def test_evidence_fusion_and_context_compiler(stage_d_test_env: dict):
    """
    Test 11: Evidence fusion with both RAG and shared knowledge maintains correct attribution and grounding.
    Both chunks pass ContextCompiler and retain distinct provenance without ranking corruption.
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    from rag_app.context.compiler import ContextCompiler

    dept_id = stage_d_test_env["dept_fin_id"]
    doc_chunk = RetrievedChunk(
        chunk_id="doc_chunk_1",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="finance_sop.pdf",
        chunk_index=0,
        text="Standard Operating Procedure for enterprise accounts.",
        hybrid_score=0.92,
    )
    shared_chunk = RetrievedChunk(
        chunk_id="shared_kn_123",
        document_id="shared_kn_doc_123",
        department_id=str(dept_id),
        file_name="Approved Shared Knowledge: discount_policy",
        chunk_index=0,
        text="[Approved Shared Organizational Knowledge: discount_policy]\nMax discount is 15%.\nStatus: Approved",
        hybrid_score=0.95,
    )

    compiler = ContextCompiler()
    compiled = compiler.compile(
        chunks=[doc_chunk, shared_chunk],
        allowed_departments=[dept_id],
        chunk_provenance={
            "doc_chunk_1": ChunkProvenance(chunk_id="doc_chunk_1", strategies=["HYBRID"]),
            "shared_kn_123": ChunkProvenance(chunk_id="shared_kn_123", strategies=["SHARED_KNOWLEDGE"]),
        },
    )

    assert len(compiled.chunks) == 2
    assert compiled.chunks[0].chunk_id == "doc_chunk_1"
    assert compiled.chunks[1].chunk_id == "shared_kn_123"
    assert "Max discount is 15%." in compiled.rendered_context
    assert "Standard Operating Procedure" in compiled.rendered_context


@pytest.mark.asyncio
async def test_admin_cannot_extract_from_personal_conversation_thread(stage_d_test_env: dict):
    """
    Test 12 (Phase 1B Audit): An ADMIN cannot extract knowledge candidates from another user's personal direct conversation (workspace_id IS None).
    """
    session: AsyncSession = stage_d_test_env["session"]
    schema = stage_d_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = KnowledgeCandidateService(session)
    admin = stage_d_test_env["admin_user"]
    hr_user = stage_d_test_env["hr_user"]

    # HR user has a personal conversation with workspace_id=None
    personal_conv_id = stage_d_test_env["conv_personal_id"]
    personal_msg_id = stage_d_test_env["msg_personal_id"]

    # Admin attempts to extract from hr_user's personal conversation
    with pytest.raises(HTTPException) as exc:
        await service.create_candidate(
            user=admin,
            request=CandidateCreateRequest(
                conversation_id=personal_conv_id,
                chat_message_id=personal_msg_id,
                statement="Admin extracted personal secret",
                topic="privacy",
            ),
        )
    assert exc.value.status_code == 403
    assert "Personal conversation threads without a workspace are strictly private" in exc.value.detail

    # But the owner (hr_user) CAN access their own personal conversation
    cand = await service.create_candidate(
        user=hr_user,
        request=CandidateCreateRequest(
            conversation_id=personal_conv_id,
            chat_message_id=personal_msg_id,
            statement="Self-extracted fact from my own notes",
            topic="notes",
        ),
    )
    assert cand.id is not None
