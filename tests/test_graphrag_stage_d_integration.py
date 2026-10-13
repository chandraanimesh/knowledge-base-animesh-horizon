import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from sqlalchemy import text
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
    Document,
    DocumentChunk,
    KnowledgeItem,
    KnowledgeEntity,
    KnowledgeRelationship,
    KnowledgeEvidence,
)
from rag_app.retrieval.service import RetrievalService
from rag_app.context.compiler import ContextCompiler
from rag_app.schemas.rag import (
    QueryPlan,
    RelationshipQuery,
    CompanyFilters,
    RetrievedChunk,
    ChunkProvenance,
    GraphEdgeProvenance,
)
from rag_app.knowledge.candidate_service import KnowledgeCandidateService


@pytest_asyncio.fixture
async def graphrag_test_env():
    """
    Isolated schema test environment populated with realistic entities,
    relationships, document chunks, and knowledge items.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_graphrag_{uuid.uuid4().hex[:8]}"

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
            KnowledgeItem.__table__.create(sync_conn)
            KnowledgeEntity.__table__.create(sync_conn)
            KnowledgeRelationship.__table__.create(sync_conn)
            KnowledgeEvidence.__table__.create(sync_conn)

        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:8]
    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # 1. Departments: Engineering and Sales
        dept_eng = Department(name=f"Engineering {suffix}", code=f"ENG_{suffix}")
        dept_sales = Department(name=f"Sales {suffix}", code=f"SALES_{suffix}")
        session.add_all([dept_eng, dept_sales])
        await session.flush()

        # 2. Users
        admin_user = User(email=f"admin_{suffix}@example.com", hashed_password="pwd", role="ADMIN")
        sales_user = User(email=f"sales_{suffix}@example.com", hashed_password="pwd", role="USER")
        session.add_all([admin_user, sales_user])
        await session.flush()

        # Permissions: Sales user only has VIEW for Sales department
        perm_view = Permission(code="VIEW")
        session.add(perm_view)
        await session.flush()

        session.add(UserDepartment(user_id=sales_user.id, department_id=dept_sales.id))
        session.add(UserPermission(user_id=sales_user.id, department_id=dept_sales.id, permission_id=perm_view.id))

        # 3. Workspaces
        ws_eng = Workspace(name=f"Eng WS {suffix}", code=f"ws_eng_{suffix}", department_id=dept_eng.id, created_by=admin_user.id)
        ws_sales = Workspace(name=f"Sales WS {suffix}", code=f"ws_sales_{suffix}", department_id=dept_sales.id, created_by=admin_user.id)
        session.add_all([ws_eng, ws_sales])
        await session.flush()

        # 4. Documents & DocumentChunks
        # Doc 1: Eng team doc
        doc_eng = Document(file_name="eng_architecture.md", file_path="/tmp/eng.md", content_hash=f"h_eng_{suffix}", department_id=dept_eng.id, created_by=admin_user.id, workspace_id=ws_eng.id)
        # Doc 2: Sales team doc
        doc_sales = Document(file_name="sales_org.md", file_path="/tmp/sales.md", content_hash=f"h_sales_{suffix}", department_id=dept_sales.id, created_by=admin_user.id, workspace_id=ws_sales.id)
        session.add_all([doc_eng, doc_sales])
        await session.flush()

        chunk_eng_1 = DocumentChunk(document_id=doc_eng.id, chunk_index=0, text="Project Titan is developed by the Core Systems team.", embedding=[0.0] * settings.embedding_dim, chunk_hash=f"ch_e1_{suffix}")
        chunk_eng_2 = DocumentChunk(document_id=doc_eng.id, chunk_index=1, text="The Core Systems team reports to Director Sarah Connor.", embedding=[0.0] * settings.embedding_dim, chunk_hash=f"ch_e2_{suffix}")
        chunk_sales_1 = DocumentChunk(document_id=doc_sales.id, chunk_index=0, text="Alice Walker works at Acme Corp as Lead Account Executive.", embedding=[0.0] * settings.embedding_dim, chunk_hash=f"ch_s1_{suffix}")
        session.add_all([chunk_eng_1, chunk_eng_2, chunk_sales_1])
        await session.flush()

        # 5. KnowledgeItems & Entities & Relationships
        now = datetime.now(timezone.utc)

        # Entity 1: Alice Walker
        ki_alice = KnowledgeItem(canonical_key=f"ENTITY:person:alice_{suffix}", kind="ENTITY", status="ACTIVE", approval_status="APPROVED", approved_by_id=admin_user.id, approved_at=now, visibility="SHARED", department_id=dept_sales.id, workspace_id=ws_sales.id, author_id=admin_user.id, confidence=0.98)
        session.add(ki_alice)
        await session.flush()
        ent_alice = KnowledgeEntity(knowledge_item_id=ki_alice.id, entity_type="person", canonical_name="Alice Walker", attributes={"title": "Lead AE"})
        ev_alice = KnowledgeEvidence(knowledge_item_id=ki_alice.id, document_chunk_id=chunk_sales_1.id, confidence=0.98)
        session.add_all([ent_alice, ev_alice])

        # Entity 2: Acme Corp
        ki_acme = KnowledgeItem(canonical_key=f"ENTITY:company:acme_{suffix}", kind="ENTITY", status="ACTIVE", approval_status="APPROVED", approved_by_id=admin_user.id, approved_at=now, visibility="SHARED", department_id=dept_sales.id, workspace_id=ws_sales.id, author_id=admin_user.id, confidence=0.98)
        session.add(ki_acme)
        await session.flush()
        ent_acme = KnowledgeEntity(knowledge_item_id=ki_acme.id, entity_type="company", canonical_name="Acme Corp", attributes={})
        session.add(ent_acme)

        # Relationship 1: Alice Walker -> works_at -> Acme Corp
        ki_sales_1 = KnowledgeItem(canonical_key=f"REL:SALES:{suffix}_1", kind="RELATIONSHIP", status="ACTIVE", approval_status="APPROVED", approved_by_id=admin_user.id, approved_at=now, visibility="SHARED", department_id=dept_sales.id, workspace_id=ws_sales.id, author_id=admin_user.id, confidence=0.98)
        session.add(ki_sales_1)
        await session.flush()
        rel_sales_1 = KnowledgeRelationship(knowledge_item_id=ki_sales_1.id, subject_entity_id=ent_alice.id, predicate="works_at", object_entity_id=ent_acme.id)
        ev_sales_1 = KnowledgeEvidence(knowledge_item_id=ki_sales_1.id, document_chunk_id=chunk_sales_1.id, confidence=0.98)
        session.add_all([rel_sales_1, ev_sales_1])

        # Entity 3: Project Titan
        ki_titan = KnowledgeItem(canonical_key=f"ENTITY:project:titan_{suffix}", kind="ENTITY", status="ACTIVE", approval_status="APPROVED", approved_by_id=admin_user.id, approved_at=now, visibility="SHARED", department_id=dept_eng.id, workspace_id=ws_eng.id, author_id=admin_user.id, confidence=0.95)
        session.add(ki_titan)
        await session.flush()
        ent_titan = KnowledgeEntity(knowledge_item_id=ki_titan.id, entity_type="project", canonical_name="Project Titan", attributes={})
        ev_titan = KnowledgeEvidence(knowledge_item_id=ki_titan.id, document_chunk_id=chunk_eng_1.id, confidence=0.95)
        session.add_all([ent_titan, ev_titan])

        # Entity 4: Core Systems
        ki_core = KnowledgeItem(canonical_key=f"ENTITY:team:core_{suffix}", kind="ENTITY", status="ACTIVE", approval_status="APPROVED", approved_by_id=admin_user.id, approved_at=now, visibility="SHARED", department_id=dept_eng.id, workspace_id=ws_eng.id, author_id=admin_user.id, confidence=0.95)
        session.add(ki_core)
        await session.flush()
        ent_core_sys = KnowledgeEntity(knowledge_item_id=ki_core.id, entity_type="team", canonical_name="Core Systems", attributes={})
        session.add(ent_core_sys)

        # Entity 5: Sarah Connor
        ki_sarah = KnowledgeItem(canonical_key=f"ENTITY:person:sarah_{suffix}", kind="ENTITY", status="ACTIVE", approval_status="APPROVED", approved_by_id=admin_user.id, approved_at=now, visibility="SHARED", department_id=dept_eng.id, workspace_id=ws_eng.id, author_id=admin_user.id, confidence=0.92)
        session.add(ki_sarah)
        await session.flush()
        ent_sarah = KnowledgeEntity(knowledge_item_id=ki_sarah.id, entity_type="person", canonical_name="Sarah Connor", attributes={"role": "Director"})
        session.add(ent_sarah)

        # Relationship 2: Project Titan -> developed_by -> Core Systems
        ki_eng_1 = KnowledgeItem(canonical_key=f"REL:ENG:{suffix}_1", kind="RELATIONSHIP", status="ACTIVE", approval_status="APPROVED", approved_by_id=admin_user.id, approved_at=now, visibility="SHARED", department_id=dept_eng.id, workspace_id=ws_eng.id, author_id=admin_user.id, confidence=0.95)
        session.add(ki_eng_1)
        await session.flush()
        rel_eng_1 = KnowledgeRelationship(knowledge_item_id=ki_eng_1.id, subject_entity_id=ent_titan.id, predicate="developed_by", object_entity_id=ent_core_sys.id)
        ev_eng_1 = KnowledgeEvidence(knowledge_item_id=ki_eng_1.id, document_chunk_id=chunk_eng_1.id, confidence=0.95)
        session.add_all([rel_eng_1, ev_eng_1])

        # Relationship 3: Core Systems -> reports_to -> Sarah Connor
        ki_eng_2 = KnowledgeItem(canonical_key=f"REL:ENG:{suffix}_2", kind="RELATIONSHIP", status="ACTIVE", approval_status="APPROVED", approved_by_id=admin_user.id, approved_at=now, visibility="SHARED", department_id=dept_eng.id, workspace_id=ws_eng.id, author_id=admin_user.id, confidence=0.92)
        session.add(ki_eng_2)
        await session.flush()
        rel_eng_2 = KnowledgeRelationship(knowledge_item_id=ki_eng_2.id, subject_entity_id=ent_core_sys.id, predicate="reports_to", object_entity_id=ent_sarah.id)
        ev_eng_2 = KnowledgeEvidence(knowledge_item_id=ki_eng_2.id, document_chunk_id=chunk_eng_2.id, confidence=0.92)
        session.add_all([rel_eng_2, ev_eng_2])

        await session.commit()

        yield {
            "schema_name": schema_name,
            "session": session,
            "dept_eng_id": dept_eng.id,
            "dept_sales_id": dept_sales.id,
            "admin_user": admin_user,
            "sales_user": sales_user,
            "ws_eng_id": ws_eng.id,
            "ws_sales_id": ws_sales.id,
            "chunk_sales_1_id": chunk_sales_1.id,
            "chunk_eng_1_id": chunk_eng_1.id,
            "chunk_eng_2_id": chunk_eng_2.id,
            "ki_sales_1_id": ki_sales_1.id,
            "ki_eng_1_id": ki_eng_1.id,
            "ki_eng_2_id": ki_eng_2.id,
        }

    # Clean teardown
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_direct_entity_lookup(graphrag_test_env: dict):
    """
    Test 1: Direct entity lookup via _entity_search.
    Retrieves the document chunk associated with the entity and attaches KNOWLEDGE provenance.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    prov = {}
    rows = await service._entity_search(
        search_term="Alice Walker",
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
        provenance_collector=prov,
    )

    assert len(rows) == 1
    chunk, doc, score = rows[0]
    assert chunk.id == graphrag_test_env["chunk_sales_1_id"]
    assert "Alice Walker works at Acme Corp" in chunk.text
    assert str(chunk.id) in prov
    assert "KNOWLEDGE" in prov[str(chunk.id)].strategies
    assert prov[str(chunk.id)].entity_name == "Alice Walker"


@pytest.mark.asyncio
async def test_one_hop_relationship_retrieval(graphrag_test_env: dict):
    """
    Test 2: One-hop relationship traversal via _relationship_search.
    Query: 'who works at Acme Corp' (predicate=works_at, object=Acme Corp, direction=in).
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    rel_q = RelationshipQuery(
        predicate="works_at",
        object_entity="Acme Corp",
        direction="in",
        hop_limit=1,
    )
    prov = {}
    rows = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
        provenance_collector=prov,
    )

    assert len(rows) == 1
    chunk, doc, score = rows[0]
    assert chunk.id == graphrag_test_env["chunk_sales_1_id"]
    assert "Alice Walker works at Acme Corp" in chunk.text
    assert str(chunk.id) in prov
    assert "GRAPH" in prov[str(chunk.id)].strategies
    assert len(prov[str(chunk.id)].graph_path) >= 1
    edge = prov[str(chunk.id)].graph_path[0]
    assert edge.predicate == "works_at"
    assert edge.object == "Acme Corp"


@pytest.mark.asyncio
async def test_two_hop_relationship_retrieval(graphrag_test_env: dict):
    """
    Test 3: Two-hop relationship traversal.
    Chain: Project Titan -> developed_by -> Core Systems -> reports_to -> Sarah Connor.
    Query: subject='Project Titan', hop_limit=2.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    rel_q = RelationshipQuery(
        subject_entity="Project Titan",
        direction="out",
        hop_limit=2,
    )
    prov = {}
    rows = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_eng_id"]],
        provenance_collector=prov,
    )

    assert len(rows) >= 1
    chunk_ids = [r[0].id for r in rows]
    # In 2-hop traversal, the terminal evidence chunk is chunk_eng_2
    assert graphrag_test_env["chunk_eng_2_id"] in chunk_ids

    cid_str = str(graphrag_test_env["chunk_eng_2_id"])
    assert cid_str in prov
    p = prov[cid_str]
    assert "GRAPH" in p.strategies
    assert len(p.graph_path) == 2
    assert p.graph_path[0].hop == 1
    assert p.graph_path[0].subject == "Project Titan"
    assert p.graph_path[0].object == "Core Systems"
    assert p.graph_path[1].hop == 2
    assert p.graph_path[1].subject == "Core Systems"
    assert p.graph_path[1].object == "Sarah Connor"


@pytest.mark.asyncio
async def test_nonexistent_relationship_returns_empty(graphrag_test_env: dict):
    """
    Test 4: When no relevant relationship exists, returns empty list without error.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    rel_q = RelationshipQuery(
        predicate="acquired_by",
        object_entity="NonExistentCorp",
        direction="in",
        hop_limit=1,
    )
    prov = {}
    rows = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
        provenance_collector=prov,
    )
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_deterministic_results_for_repeated_queries(graphrag_test_env: dict):
    """
    Test 5: Repeated equivalent queries return strictly identical results and ordering.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    rel_q = RelationshipQuery(predicate="works_at", object_entity="Acme Corp", direction="in", hop_limit=1)

    prov1, prov2 = {}, {}
    rows1 = await service._relationship_search(rel_q, [graphrag_test_env["dept_sales_id"]], prov1)
    rows2 = await service._relationship_search(rel_q, [graphrag_test_env["dept_sales_id"]], prov2)

    assert len(rows1) == len(rows2)
    assert [r[0].id for r in rows1] == [r[0].id for r in rows2]
    assert prov1 == prov2


@pytest.mark.asyncio
async def test_department_authorization_containment(graphrag_test_env: dict):
    """
    Test 6: Regular users can only retrieve relationships within their authorized departments.
    Sales user cannot retrieve Engineering relationship chunks.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    rel_q = RelationshipQuery(subject_entity="Project Titan", direction="out", hop_limit=1)

    # Sales user is authorized ONLY for Sales department
    rows = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
    )
    assert len(rows) == 0, "Sales user must not access Engineering relationship"

    # User with empty allowed_departments must fail closed
    rows_empty = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[],
    )
    assert len(rows_empty) == 0, "Empty allowed departments must return 0 rows"


@pytest.mark.asyncio
async def test_unauthorized_connected_entities_do_not_leak_in_2hop(graphrag_test_env: dict):
    """
    Test 7: In 2-hop traversal, unauthorized connected entities and documents cannot leak.
    Even if Hop 1 is authorized, if Hop 2 is in an unauthorized department, it is blocked.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    # Project Titan (Engineering) -> Core Systems -> Sarah Connor (Engineering)
    rel_q = RelationshipQuery(subject_entity="Project Titan", direction="out", hop_limit=2)

    # If only Sales department is authorized, 2-hop traversal must yield 0 results
    rows = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
    )
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_null_department_cannot_bypass_graph_authorization(graphrag_test_env: dict):
    """
    Test 8: A NULL department on KnowledgeItem or Document cannot bypass department authorization.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    # Force department_id on Sales KnowledgeItem to NULL
    ki = await session.get(KnowledgeItem, graphrag_test_env["ki_sales_1_id"])
    ki.department_id = None
    await session.commit()

    service = RetrievalService(session)
    rel_q = RelationshipQuery(predicate="works_at", object_entity="Acme Corp", direction="in", hop_limit=1)

    rows = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
    )
    assert len(rows) == 0, "Item with department_id=NULL must not be retrievable by department-scoped user"

    # Restore department_id
    ki.department_id = graphrag_test_env["dept_sales_id"]
    await session.commit()


@pytest.mark.asyncio
async def test_approval_lifecycle_shared_and_rejected_handling(graphrag_test_env: dict):
    """
    Test 9: Shared organizational knowledge requires approval.
    - Database check constraint ck_knowledgeitem_shared_requires_approved strictly prevents
      unapproved records (DRAFT/PENDING/REJECTED) from having visibility='SHARED'.
    - If approval_status='REJECTED' (even WORKSPACE), it is excluded from graph search.
    - If status='ARCHIVED', it is excluded from graph search.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    ki = await session.get(KnowledgeItem, graphrag_test_env["ki_sales_1_id"])
    ki.department_id = graphrag_test_env["dept_sales_id"]
    admin_id = graphrag_test_env["admin_user"].id

    # 1. DB constraint enforces that SHARED requires APPROVED
    for unapproved_status in ["DRAFT", "PENDING", "REJECTED"]:
        ki.visibility = "SHARED"
        ki.approval_status = unapproved_status
        with pytest.raises(Exception) as excinfo:
            await session.commit()
        await session.rollback()
        assert "ck_knowledgeitem_shared_requires_approved" in str(excinfo.value).lower()

    # Refresh ki after rollbacks
    ki = await session.get(KnowledgeItem, graphrag_test_env["ki_sales_1_id"])

    # 2. REJECTED item is excluded even if visibility=WORKSPACE
    ki.visibility = "WORKSPACE"
    ki.approval_status = "REJECTED"
    await session.commit()
    rel_q = RelationshipQuery(predicate="works_at", object_entity="Acme Corp", direction="in", hop_limit=1)
    rows_rejected = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
    )
    assert len(rows_rejected) == 0, "REJECTED item must be excluded even for WORKSPACE visibility"

    # 3. ARCHIVED item is excluded
    ki.approval_status = "APPROVED"
    ki.status = "ARCHIVED"
    await session.commit()
    rows_archived = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
    )
    assert len(rows_archived) == 0, "ARCHIVED item must be excluded"

    # Reset to active approved
    ki.status = "ACTIVE"
    ki.visibility = "SHARED"
    ki.approval_status = "APPROVED"
    ki.approved_by_id = admin_id
    ki.approved_at = datetime.now(timezone.utc)
    await session.commit()


@pytest.mark.asyncio
async def test_legacy_workspace_knowledge_and_workspace_scoping(graphrag_test_env: dict):
    """
    Test 10: Legacy workspace-scoped knowledge retrieval and workspace scoping.
    - Pre-existing workspace-scoped document knowledge with migration default (approval_status='DRAFT',
      visibility='WORKSPACE', source_type='DOCUMENT') is retrievable by authorized workspace users.
    - Scoped with concrete workspace_id restricts document resolution strictly to that workspace.
    - Scoped with workspace_id=None (Global Chat) preserves cross-workspace retrieval for authorized departments.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)
    ki = await session.get(KnowledgeItem, graphrag_test_env["ki_sales_1_id"])

    # Set to legacy migration defaults
    ki.visibility = "WORKSPACE"
    ki.approval_status = "DRAFT"
    ki.source_type = "DOCUMENT"
    ki.status = "ACTIVE"
    await session.commit()

    rel_q = RelationshipQuery(predicate="works_at", object_entity="Acme Corp", direction="in", hop_limit=1)

    # 1. Scoped to correct workspace: retrievable under legacy policy without org-wide sharing approval
    rows_ws = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
        workspace_id=graphrag_test_env["ws_sales_id"],
    )
    assert len(rows_ws) == 1, "Authorized workspace chat must retrieve legacy DRAFT workspace document knowledge"
    assert rows_ws[0][0].id == graphrag_test_env["chunk_sales_1_id"]

    # 2. Scoped to different workspace: excluded
    rows_diff_ws = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
        workspace_id=graphrag_test_env["ws_eng_id"],
    )
    assert len(rows_diff_ws) == 0, "Workspace chat must exclude chunks from a different workspace"

    # 3. Global Chat (workspace_id=None): retrievable across authorized workspaces
    rows_global = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
        workspace_id=None,
    )
    assert len(rows_global) == 1, "Global Chat (workspace_id=None) must preserve authorized department document retrieval"

    # Reset ki back to approved
    admin_id = graphrag_test_env["admin_user"].id
    ki.approval_status = "APPROVED"
    ki.approved_by_id = admin_id
    ki.approved_at = datetime.now(timezone.utc)
    await session.commit()


@pytest.mark.asyncio
async def test_relationship_divergence_conflict_handling(graphrag_test_env: dict):
    """
    Test 11: Handling conflicting relationship evidence across active graph provenance.
    ContextCompiler detects relationship divergence and generates ConflictNotice.
    """
    dept_id = graphrag_test_env["dept_sales_id"]

    chunk_a = RetrievedChunk(
        chunk_id="chunk_a",
        document_id="doc_a",
        department_id=str(dept_id),
        file_name="vendor_a.md",
        chunk_index=0,
        text="Acme Corp is headquartered in Boston.",
    )
    chunk_b = RetrievedChunk(
        chunk_id="chunk_b",
        document_id="doc_b",
        department_id=str(dept_id),
        file_name="vendor_b.md",
        chunk_index=0,
        text="Acme Corp is headquartered in San Francisco.",
    )

    prov_a = ChunkProvenance(
        chunk_id="chunk_a",
        strategies=["GRAPH"],
        graph_path=[GraphEdgeProvenance(subject="Acme Corp", predicate="headquartered_in", object="Boston", hop=1)],
    )
    prov_b = ChunkProvenance(
        chunk_id="chunk_b",
        strategies=["GRAPH"],
        graph_path=[GraphEdgeProvenance(subject="Acme Corp", predicate="headquartered_in", object="San Francisco", hop=1)],
    )

    compiler = ContextCompiler()
    compiled = compiler.compile(
        chunks=[chunk_a, chunk_b],
        allowed_departments=[dept_id],
        chunk_provenance={"chunk_a": prov_a, "chunk_b": prov_b},
    )

    assert len(compiled.conflicts) == 1
    notice = compiled.conflicts[0]
    assert notice.conflict_type == "RELATIONSHIP_DIVERGENCE"
    assert "boston" in [o.lower() for o in notice.conflicting_objects]
    assert "san francisco" in [o.lower() for o in notice.conflicting_objects]

    from rag_app.generation.prompt import build_rag_messages
    messages = build_rag_messages(
        context=compiled.rendered_context,
        question="Where is Acme Corp headquartered?",
        conflicts=compiled.conflicts,
        chunks=compiled.chunks,
    )
    user_prompt = messages[-1]["content"]
    assert "Conflicting Evidence Notice" in user_prompt


@pytest.mark.asyncio
async def test_fusion_graphrag_with_stage_d_shared_knowledge(graphrag_test_env: dict):
    """
    Test 12: GraphRAG document chunks and Stage D approved shared-knowledge chunks
    fuse seamlessly into the RAG context without corrupting provenance or citations.
    """
    dept_id = graphrag_test_env["dept_sales_id"]

    # Graph chunk
    graph_chunk = RetrievedChunk(
        chunk_id="chunk_graph_1",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="sales_org.md",
        chunk_index=0,
        text="Alice Walker works at Acme Corp.",
        hybrid_score=0.95,
    )
    # Stage D Shared Knowledge chunk
    shared_chunk = RetrievedChunk(
        chunk_id="shared_kn_corp_1",
        document_id="shared_kn_doc_1",
        department_id=str(dept_id),
        file_name="Approved Shared Knowledge: discount_policy",
        chunk_index=0,
        text="[Approved Shared Organizational Knowledge: discount_policy]\nMax enterprise discount is 15%.\nStatus: Approved",
        hybrid_score=0.90,
    )

    compiler = ContextCompiler()
    compiled = compiler.compile(
        chunks=[graph_chunk, shared_chunk],
        allowed_departments=[dept_id],
        chunk_provenance={
            "chunk_graph_1": ChunkProvenance(chunk_id="chunk_graph_1", strategies=["GRAPH"]),
            "shared_kn_corp_1": ChunkProvenance(chunk_id="shared_kn_corp_1", strategies=["SHARED_KNOWLEDGE"]),
        },
    )

    assert len(compiled.chunks) == 2
    assert compiled.chunks[0].chunk_id == "chunk_graph_1"
    assert compiled.chunks[1].chunk_id == "shared_kn_corp_1"
    assert "Alice Walker works at Acme Corp" in compiled.rendered_context
    assert "Max enterprise discount is 15%" in compiled.rendered_context


@pytest.mark.asyncio
async def test_global_shared_knowledge_excludes_unapproved_and_archived(graphrag_test_env: dict):
    """
    Test 13: Stage D global shared-knowledge retrieval via KnowledgeCandidateService
    strictly excludes DRAFT, PENDING, REJECTED, and ARCHIVED records.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    kn_service = KnowledgeCandidateService(session)
    ki = await session.get(KnowledgeItem, graphrag_test_env["ki_sales_1_id"])
    admin_id = graphrag_test_env["admin_user"].id

    # 1. Unapproved statuses with WORKSPACE visibility must not be returned by shared knowledge search
    for unapproved in ["DRAFT", "PENDING", "REJECTED"]:
        ki.visibility = "WORKSPACE"
        ki.approval_status = unapproved
        await session.commit()

        shared_res = await kn_service.retrieve_shared_knowledge_as_chunks(
            query="Acme Corp",
            allowed_departments=[graphrag_test_env["dept_sales_id"]],
        )
        item_ids = [c.chunk_id for c in shared_res]
        assert f"shared_kn_{ki.id}" not in item_ids, f"Shared knowledge retrieval must exclude status={unapproved}"

    # 2. Inactive / ARCHIVED status
    ki.visibility = "SHARED"
    ki.approval_status = "APPROVED"
    ki.status = "ARCHIVED"
    await session.commit()

    shared_res_archived = await kn_service.retrieve_shared_knowledge_as_chunks(
        query="Acme Corp",
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
    )
    assert f"shared_kn_{ki.id}" not in [c.chunk_id for c in shared_res_archived], "Shared knowledge retrieval must exclude ARCHIVED status"

    # Reset
    ki.status = "ACTIVE"
    ki.visibility = "SHARED"
    ki.approval_status = "APPROVED"
    ki.approved_by_id = admin_id
    ki.approved_at = datetime.now(timezone.utc)
    await session.commit()


@pytest.mark.asyncio
async def test_approved_shared_entity_cannot_expose_unauthorized_department(graphrag_test_env: dict):
    """
    Test 14: Graph traversal cannot use an approved shared entity to expose an unauthorized
    connected item or its document evidence.
    Scenario: Entity 1 is in Sales (authorized). Hop 2 connects to Engineering (unauthorized).
    Sales-only user must receive 0 rows for the unauthorized branch.
    """
    session: AsyncSession = graphrag_test_env["session"]
    schema = graphrag_test_env["schema_name"]
    await session.execute(text(f"SET search_path TO {schema}, public"))

    service = RetrievalService(session)

    # 2-hop query from Project Titan (Engineering) to Sarah Connor (Engineering)
    rel_q = RelationshipQuery(
        subject_entity="Project Titan",
        direction="out",
        hop_limit=2,
    )

    # User with only Sales department authorization:
    rows = await service._relationship_search(
        rel_query=rel_q,
        allowed_departments=[graphrag_test_env["dept_sales_id"]],
        workspace_id=None,
    )
    assert len(rows) == 0, "Graph traversal must never expose unauthorized department documents even in Global Chat"

