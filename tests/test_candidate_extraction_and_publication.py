import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from fastapi import HTTPException
from sqlalchemy import text, select, func
from sqlalchemy.exc import IntegrityError
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


@pytest_asyncio.fixture
async def stage_c_test_env():
    """
    Creates an isolated schema for Stage C tests so that:
    1. All tables and constraints exist in an isolated test schema.
    2. Primary DB schema is never modified directly.
    3. Clean teardown drops the test schema.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_stage_c_{uuid.uuid4().hex[:8]}"

    # Setup isolated schema and tables
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

        dept = Department(
            name=f"Finance Dept {suffix}",
            code=f"FIN_{suffix}",
        )
        session.add(dept)
        await session.flush()

        admin_user = User(
            email=f"admin_{suffix}@example.com",
            hashed_password="hashed_pwd",
            role="ADMIN",
        )
        author_user = User(
            email=f"author_{suffix}@example.com",
            hashed_password="hashed_pwd",
            role="USER",
        )
        unauth_user = User(
            email=f"unauth_{suffix}@example.com",
            hashed_password="hashed_pwd",
            role="USER",
        )
        session.add_all([admin_user, author_user, unauth_user])
        await session.flush()

        # Grant department access to author_user
        user_dept = UserDepartment(user_id=author_user.id, department_id=dept.id)
        perm_view = Permission(code="VIEW")
        session.add_all([user_dept, perm_view])
        await session.flush()

        user_perm = UserPermission(
            user_id=author_user.id,
            department_id=dept.id,
            permission_id=perm_view.id,
        )
        session.add(user_perm)
        await session.flush()

        ws = Workspace(
            name=f"Finance Workspace {suffix}",
            code=f"fin_ws_{suffix}",
            department_id=dept.id,
            created_by=admin_user.id,
        )
        session.add(ws)
        await session.flush()

        # Add author as member of workspace
        member = WorkspaceMember(
            workspace_id=ws.id,
            user_id=author_user.id,
            role="EDITOR",
        )
        session.add(member)
        await session.flush()

        # Create Conversation for author_user
        conv = Conversation(
            user_id=author_user.id,
            workspace_id=ws.id,
            title="Q3 Budget Review",
        )
        session.add(conv)
        await session.flush()

        # Create ChatMessages
        msg1 = ChatMessage(
            conversation_id=conv.id,
            role="user",
            content="Can we offer a 15% discount for Annual Tier customers in Q3?",
        )
        msg2 = ChatMessage(
            conversation_id=conv.id,
            role="assistant",
            content="Yes, the CFO approved up to 15% discount for Annual Enterprise contracts exceeding $50k.",
        )
        session.add_all([msg1, msg2])
        await session.flush()

        # Create second conversation for unauth_user
        other_conv = Conversation(
            user_id=unauth_user.id,
            workspace_id=ws.id,
            title="Private HR Query",
        )
        session.add(other_conv)
        await session.flush()
        other_msg = ChatMessage(
            conversation_id=other_conv.id,
            role="user",
            content="Private personal inquiry.",
        )
        session.add(other_msg)
        await session.flush()

        # Create legacy document & chunk
        doc = Document(
            file_name="legacy_policy.txt",
            file_path="/tmp/legacy_policy.txt",
            content_hash=f"hash_{suffix}",
            department_id=dept.id,
            created_by=admin_user.id,
            workspace_id=ws.id,
        )
        session.add(doc)
        await session.flush()

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            text="Official corporate travel policy document chunk.",
            embedding=[0.0] * settings.embedding_dim,
            chunk_hash=f"chunk_hash_{suffix}",
        )
        session.add(chunk)
        await session.flush()

        await session.commit()

        entities = {
            "schema_name": schema_name,
            "session": session,
            "dept_id": dept.id,
            "admin_user_id": admin_user.id,
            "author_user_id": author_user.id,
            "unauth_user_id": unauth_user.id,
            "workspace_id": ws.id,
            "conversation_id": conv.id,
            "msg1_id": msg1.id,
            "msg2_id": msg2.id,
            "other_conv_id": other_conv.id,
            "other_msg_id": other_msg.id,
            "doc_id": doc.id,
            "chunk_id": chunk.id,
        }
        yield entities

    # Teardown
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_migration_publication_invariants_upgrade_and_downgrade():
    """
    Validates Alembic migration add_shared_knowledge_publication_invariants:
    1. Upgrade adds check constraint ck_knowledgeitem_shared_requires_approved.
    2. Downgrade cleanly drops the constraint without affecting tables or data.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    test_schema = f"test_mig_inv_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {test_schema}"))
        def setup_base_tables(sync_conn):
            sync_conn.execute(text(f"""
                CREATE TABLE {test_schema}.departments (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR NOT NULL,
                    code VARCHAR NOT NULL UNIQUE
                )
            """))
            sync_conn.execute(text(f"""
                CREATE TABLE {test_schema}.knowledge_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    canonical_key VARCHAR NOT NULL,
                    kind VARCHAR NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'ACTIVE',
                    visibility VARCHAR(20) NOT NULL DEFAULT 'WORKSPACE',
                    approval_status VARCHAR(20) NOT NULL DEFAULT 'DRAFT',
                    department_id UUID REFERENCES {test_schema}.departments(id),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
        await conn.run_sync(setup_base_tables)

    try:
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        import migrations.versions.add_shared_knowledge_publication_invariants as mig
        import alembic.op as alembic_op

        async with engine.connect() as conn:
            def run_upgrade(sync_conn):
                sync_conn.execute(text(f"SET search_path TO {test_schema}, public"))
                ctx = MigrationContext.configure(sync_conn, opts={"version_table_schema": test_schema})
                op = Operations(ctx)
                alembic_op._proxy = op
                mig.upgrade()

            await conn.run_sync(run_upgrade)

            # Verify constraint exists
            def check_constraint(sync_conn):
                from sqlalchemy import inspect
                inspector = inspect(sync_conn)
                checks = [c['name'] for c in inspector.get_check_constraints('knowledge_items', schema=test_schema)]
                assert 'ck_knowledgeitem_shared_requires_approved' in checks

            await conn.run_sync(check_constraint)

            # Run downgrade
            def run_downgrade(sync_conn):
                sync_conn.execute(text(f"SET search_path TO {test_schema}, public"))
                ctx = MigrationContext.configure(sync_conn, opts={"version_table_schema": test_schema})
                op = Operations(ctx)
                alembic_op._proxy = op
                mig.downgrade()

            await conn.run_sync(run_downgrade)

            # Verify constraint was dropped cleanly
            def check_dropped(sync_conn):
                from sqlalchemy import inspect
                inspector = inspect(sync_conn)
                checks = [c['name'] for c in inspector.get_check_constraints('knowledge_items', schema=test_schema)]
                assert 'ck_knowledgeitem_shared_requires_approved' not in checks

            await conn.run_sync(check_dropped)

    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"))
        await engine.dispose()


@pytest.mark.asyncio
async def test_ordinary_user_cannot_approve_publication(stage_c_test_env: dict):
    """
    Requirement 1: An ordinary USER cannot approve organization-wide publication (403 Forbidden).
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    author_user = await session.get(User, stage_c_test_env["author_user_id"])
    conv_id = stage_c_test_env["conversation_id"]
    msg_id = stage_c_test_env["msg2_id"]

    await session.execute(text(f"SET search_path TO {schema}, public"))
    service = KnowledgeCandidateService(session)

    # Create candidate
    candidate = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="CFO approved 15% discount for Annual Enterprise contracts over $50k.",
            topic="pricing",
        ),
    )
    assert candidate.approval_status == "DRAFT"
    assert candidate.visibility == "WORKSPACE"

    # Submit candidate
    submitted = await service.submit_candidate(user=author_user, candidate_id=candidate.id)
    assert submitted.approval_status == "PENDING"

    # Ordinary user tries to approve -> must fail with 403 Forbidden
    with pytest.raises(HTTPException) as exc:
        await service.approve_candidate(user=author_user, candidate_id=candidate.id)
    assert exc.value.status_code == 403
    assert "Admin privileges required" in exc.value.detail


@pytest.mark.asyncio
async def test_authorized_admin_can_approve_valid_candidate(stage_c_test_env: dict):
    """
    Requirement 2: An authorized ADMIN can approve a valid candidate.
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    author_user = await session.get(User, stage_c_test_env["author_user_id"])
    admin_user = await session.get(User, stage_c_test_env["admin_user_id"])
    conv_id = stage_c_test_env["conversation_id"]
    msg_id = stage_c_test_env["msg2_id"]

    await session.execute(text(f"SET search_path TO {schema}, public"))
    service = KnowledgeCandidateService(session)

    candidate = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="CFO approved 15% discount for Annual Enterprise contracts over $50k.",
            topic="pricing",
        ),
    )
    await service.submit_candidate(user=author_user, candidate_id=candidate.id)

    # Admin approves
    approved = await service.approve_candidate(user=admin_user, candidate_id=candidate.id)
    assert approved.approval_status == "APPROVED"
    assert approved.visibility == "SHARED"
    assert approved.approved_by_id == admin_user.id
    assert approved.approved_at is not None


@pytest.mark.asyncio
async def test_unauthorized_source_conversation_access_cannot_create_candidates(stage_c_test_env: dict):
    """
    Requirement 3: Unauthorized source-conversation access cannot create candidates.
    A user cannot create a candidate from someone else's conversation.
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    unauth_user = await session.get(User, stage_c_test_env["unauth_user_id"])
    author_conv_id = stage_c_test_env["conversation_id"]
    author_msg_id = stage_c_test_env["msg2_id"]

    await session.execute(text(f"SET search_path TO {schema}, public"))
    service = KnowledgeCandidateService(session)

    # unauth_user tries to extract from author_user's conversation
    with pytest.raises(HTTPException) as exc:
        await service.create_candidate(
            user=unauth_user,
            request=CandidateCreateRequest(
                conversation_id=author_conv_id,
                chat_message_id=author_msg_id,
                statement="Unauthorized snooped fact",
            ),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_draft_pending_rejected_facts_excluded_from_shared_retrieval(stage_c_test_env: dict):
    """
    Requirement 4: DRAFT, PENDING, and REJECTED facts are strictly excluded from shared retrieval.
    Only approved shared facts are returned.
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    admin_user = await session.get(User, stage_c_test_env["admin_user_id"])
    author_user = await session.get(User, stage_c_test_env["author_user_id"])
    conv_id = stage_c_test_env["conversation_id"]
    msg_id = stage_c_test_env["msg2_id"]

    await session.execute(text(f"SET search_path TO {schema}, public"))
    service = KnowledgeCandidateService(session)

    # Item 1: DRAFT
    item_draft = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="Draft Fact 1: Q3 bonuses are under consideration.",
            topic="hr",
        ),
    )

    # Item 2: PENDING
    item_pending = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="Pending Fact 2: New office location in Berlin.",
            topic="expansion",
        ),
    )
    await service.submit_candidate(user=author_user, candidate_id=item_pending.id)

    # Item 3: REJECTED
    item_rejected = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="Rejected Fact 3: Unlimited vacation policy rejected.",
            topic="hr",
        ),
    )
    await service.reject_candidate(user=admin_user, candidate_id=item_rejected.id)

    # Item 4: APPROVED
    item_approved = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="Approved Fact 4: Official mileage reimbursement rate is $0.65/mile.",
            topic="finance",
        ),
    )
    await service.approve_candidate(user=admin_user, candidate_id=item_approved.id)

    # Query shared knowledge retrieval
    shared_items = await service.get_approved_shared_knowledge()
    shared_ids = {it.id for it in shared_items}

    assert item_approved.id in shared_ids, "Approved item must be retrievable"
    assert item_draft.id not in shared_ids, "DRAFT item must be excluded from shared knowledge"
    assert item_pending.id not in shared_ids, "PENDING item must be excluded from shared knowledge"
    assert item_rejected.id not in shared_ids, "REJECTED item must be excluded from shared knowledge"


@pytest.mark.asyncio
async def test_invalid_approval_transition_fails_closed(stage_c_test_env: dict):
    """
    Requirement 5: An invalid approval transition fails without partial updates.
    Database check constraint prevents setting visibility='SHARED' without approval_status='APPROVED'.
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    author_id = stage_c_test_env["author_user_id"]
    ws_id = stage_c_test_env["workspace_id"]

    await session.execute(text(f"SET search_path TO {schema}, public"))

    # Attempt to bypass workflow by inserting visibility='SHARED' while approval_status='DRAFT'
    invalid_item = KnowledgeItem(
        canonical_key="FACT:test:unapproved_shared_hack",
        kind="FACT",
        status="ACTIVE",
        visibility="SHARED",  # Violation!
        approval_status="DRAFT",  # Violation!
        source_type="CONVERSATION",
        workspace_id=ws_id,
        author_id=author_id,
    )
    session.add(invalid_item)
    with pytest.raises(IntegrityError) as exc:
        await session.commit()
    assert "ck_knowledgeitem_shared_requires_approved" in str(exc.value)
    await session.rollback()


@pytest.mark.asyncio
async def test_conversation_and_message_consistency(stage_c_test_env: dict):
    """
    Requirement 6: conversation_id and chat_message_id must be consistent.
    Supplying a chat_message_id belonging to a different conversation must be rejected.
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    author_user = await session.get(User, stage_c_test_env["author_user_id"])
    author_conv_id = stage_c_test_env["conversation_id"]
    other_msg_id = stage_c_test_env["other_msg_id"]  # Belongs to other_conv_id!

    await session.execute(text(f"SET search_path TO {schema}, public"))
    service = KnowledgeCandidateService(session)

    with pytest.raises(HTTPException) as exc:
        await service.create_candidate(
            user=author_user,
            request=CandidateCreateRequest(
                conversation_id=author_conv_id,
                chat_message_id=other_msg_id,
                statement="Cross-conversation mismatch test statement",
            ),
        )
    assert exc.value.status_code == 400
    assert "does not belong to the supplied conversation_id" in exc.value.detail


@pytest.mark.asyncio
async def test_legacy_document_and_research_knowledge_retains_behavior(stage_c_test_env: dict):
    """
    Requirement 7: Legacy document and research knowledge retain their correct behavior.
    Pre-existing items with document evidence remain WORKSPACE scoped, DRAFT/ACTIVE, intact.
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    dept_id = stage_c_test_env["dept_id"]
    chunk_id = stage_c_test_env["chunk_id"]

    await session.execute(text(f"SET search_path TO {schema}, public"))

    # Insert legacy document-derived KnowledgeItem
    legacy_item = KnowledgeItem(
        canonical_key="ENTITY:policy:travel_reimbursement",
        kind="ENTITY",
        status="ACTIVE",
        visibility="WORKSPACE",
        approval_status="DRAFT",
        source_type="DOCUMENT",
        department_id=dept_id,
        confidence=0.95,
    )
    session.add(legacy_item)
    await session.flush()

    legacy_ev = KnowledgeEvidence(
        knowledge_item_id=legacy_item.id,
        document_chunk_id=chunk_id,
        confidence=0.95,
    )
    session.add(legacy_ev)
    await session.commit()

    refetched = await session.get(KnowledgeItem, legacy_item.id)
    assert refetched is not None
    assert refetched.visibility == "WORKSPACE"
    assert refetched.approval_status == "DRAFT"
    assert refetched.source_type == "DOCUMENT"
    assert refetched.department_id == dept_id

    # Must NOT appear in shared knowledge query
    service = KnowledgeCandidateService(session)
    shared_items = await service.get_approved_shared_knowledge()
    shared_ids = {it.id for it in shared_items}
    assert legacy_item.id not in shared_ids


@pytest.mark.asyncio
async def test_duplicate_submission_does_not_create_duplicate_shared_facts(stage_c_test_env: dict):
    """
    Requirement 8: Duplicate submission does not create unintended duplicate shared facts.
    Submitting the same statement twice reuses the existing candidate.
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    author_user = await session.get(User, stage_c_test_env["author_user_id"])
    conv_id = stage_c_test_env["conversation_id"]
    msg_id = stage_c_test_env["msg2_id"]

    await session.execute(text(f"SET search_path TO {schema}, public"))
    service = KnowledgeCandidateService(session)

    cand1 = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="Enterprise support SLA is 1 hour for P1 tickets.",
            topic="support_sla",
        ),
    )

    cand2 = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="Enterprise support SLA is 1 hour for P1 tickets.",
            topic="support_sla",
        ),
    )

    assert cand1.id == cand2.id, "Duplicate candidate submission must reuse existing KnowledgeItem"


@pytest.mark.asyncio
async def test_evidence_deletion_or_withdrawal_prevents_unsupported_knowledge_served(stage_c_test_env: dict):
    """
    Requirement 9: Evidence deletion or withdrawal prevents unsupported knowledge from being served.
    When evidence is withdrawn so evidence count reaches 0, the item is de-published.
    """
    session: AsyncSession = stage_c_test_env["session"]
    schema = stage_c_test_env["schema_name"]
    admin_user = await session.get(User, stage_c_test_env["admin_user_id"])
    author_user = await session.get(User, stage_c_test_env["author_user_id"])
    conv_id = stage_c_test_env["conversation_id"]
    msg_id = stage_c_test_env["msg2_id"]

    await session.execute(text(f"SET search_path TO {schema}, public"))
    service = KnowledgeCandidateService(session)

    cand = await service.create_candidate(
        user=author_user,
        request=CandidateCreateRequest(
            conversation_id=conv_id,
            chat_message_id=msg_id,
            statement="Temporary promo code expires tomorrow.",
            topic="promo",
        ),
    )
    # Approve candidate
    await service.approve_candidate(user=admin_user, candidate_id=cand.id)

    # Verify it is in shared knowledge
    shared = await service.get_approved_shared_knowledge()
    assert cand.id in {it.id for it in shared}

    # Withdraw the supporting evidence
    assert len(cand.evidence) > 0
    ev_id = cand.evidence[0].id
    withdrawn = await service.withdraw_evidence(user=author_user, candidate_id=cand.id, evidence_id=ev_id)

    assert withdrawn.evidence_count == 0
    assert withdrawn.visibility == "WORKSPACE"
    assert withdrawn.approval_status == "REJECTED"
    assert withdrawn.status == "ARCHIVED"

    # Verify it is no longer returned in shared knowledge
    shared_after = await service.get_approved_shared_knowledge()
    assert cand.id not in {it.id for it in shared_after}
