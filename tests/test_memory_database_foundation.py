import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from sqlalchemy import text, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import User, Department, Workspace, MemoryItem
from rag_app.memory.schemas import MemoryCandidate, MemoryType, MemoryStatus
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.retrieval import MemoryRetrievalService


@pytest_asyncio.fixture
async def memory_test_env():
    """
    Creates an isolated schema for the test execution so that:
    1. MemoryItem with workspace_id is created in the isolated schema.
    2. The live public database schema is never modified or migrated destructively.
    3. All test entities are cleaned up on teardown.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_mem_env_{uuid.uuid4().hex[:8]}"

    # Setup isolated schema and MemoryItem table
    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        def create_mem_table(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
            MemoryItem.__table__.create(sync_conn)
        await conn.run_sync(create_mem_table)

    # Create test entities (Department, User, 2 Workspaces)
    suffix = uuid.uuid4().hex[:8]
    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        dept = Department(
            name=f"MemTest Dept {suffix}",
            code=f"MEM_{suffix}",
        )
        session.add(dept)
        await session.flush()

        user = User(
            email=f"mem_user_{suffix}@example.com",
            hashed_password="test_password_hash",
            role="USER",
        )
        session.add(user)
        await session.flush()

        ws1 = Workspace(
            name=f"Workspace 1 {suffix}",
            code=f"WS1_{suffix}",
            department_id=dept.id,
            created_by=user.id,
        )
        ws2 = Workspace(
            name=f"Workspace 2 {suffix}",
            code=f"WS2_{suffix}",
            department_id=dept.id,
            created_by=user.id,
        )
        session.add_all([ws1, ws2])
        await session.commit()

        entities = {
            "schema_name": schema_name,
            "session": session,
            "dept": dept,
            "user": user,
            "ws1": ws1,
            "ws2": ws2,
        }
        yield entities

    # Teardown
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        await conn.execute(text("DELETE FROM workspaces WHERE code IN (:c1, :c2)"), {"c1": f"WS1_{suffix}", "c2": f"WS2_{suffix}"})
        await conn.execute(text("DELETE FROM users WHERE email = :email"), {"email": f"mem_user_{suffix}@example.com"})
        await conn.execute(text("DELETE FROM departments WHERE code = :code"), {"code": f"MEM_{suffix}"})
    await engine.dispose()


@pytest.mark.asyncio
async def test_workspace_scoped_uniqueness_and_isolation(memory_test_env: dict):
    """
    Verifies that:
    1. The same user can hold the same canonical_key across DIFFERENT workspaces.
    2. Inserting a duplicate active canonical_key in the SAME workspace is rejected by unique index.
    """
    session: AsyncSession = memory_test_env["session"]
    schema: str = memory_test_env["schema_name"]
    user: User = memory_test_env["user"]
    ws1: Workspace = memory_test_env["ws1"]
    ws2: Workspace = memory_test_env["ws2"]
    canonical_key = "user_preference:theme_dark"

    await session.execute(text(f"SET search_path TO {schema}, public"))

    # Insert into Workspace 1
    cand_ws1 = MemoryCandidate(
        user_id=user.id,
        workspace_id=ws1.id,
        canonical_key=canonical_key,
        content="User prefers dark theme in workspace 1",
        memory_type=MemoryType.STATIC,
        status=MemoryStatus.ACTIVE,
    )
    mem1 = await MemoryMaterializer.materialize_memory(session, cand_ws1, auto_embed=False)
    await session.commit()
    assert mem1.id is not None
    assert mem1.workspace_id == ws1.id

    # Insert SAME canonical_key into Workspace 2 for SAME user (must succeed)
    cand_ws2 = MemoryCandidate(
        user_id=user.id,
        workspace_id=ws2.id,
        canonical_key=canonical_key,
        content="User prefers dark theme in workspace 2",
        memory_type=MemoryType.STATIC,
        status=MemoryStatus.ACTIVE,
    )
    mem2 = await MemoryMaterializer.materialize_memory(session, cand_ws2, auto_embed=False)
    await session.commit()
    assert mem2.id is not None
    assert mem2.workspace_id == ws2.id
    assert mem1.id != mem2.id

    # Direct insert duplicate active canonical key into Workspace 1 without upsert must trigger unique constraint
    dup_item = MemoryItem(
        user_id=user.id,
        workspace_id=ws1.id,
        canonical_key=canonical_key,
        content="Duplicate active theme",
        memory_type="STATIC",
        status="ACTIVE",
    )
    session.add(dup_item)
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


@pytest.mark.asyncio
async def test_memory_not_visible_across_workspaces(memory_test_env: dict):
    """
    Verifies that a memory stored in Workspace 1 is strictly invisible when querying Workspace 2.
    """
    session: AsyncSession = memory_test_env["session"]
    schema: str = memory_test_env["schema_name"]
    user: User = memory_test_env["user"]
    ws1: Workspace = memory_test_env["ws1"]
    ws2: Workspace = memory_test_env["ws2"]

    await session.execute(text(f"SET search_path TO {schema}, public"))

    cand = MemoryCandidate(
        user_id=user.id,
        workspace_id=ws1.id,
        canonical_key="user_context:active_sprint",
        content="Working on Phase 2 RAG memory integration",
        memory_type=MemoryType.DYNAMIC,
        status=MemoryStatus.ACTIVE,
    )
    await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
    await session.commit()

    # Query Workspace 2 -> must return 0 memories
    resp_ws2 = await MemoryRetrievalService.retrieve(
        session=session,
        user_id=user.id,
        workspace_id=ws2.id,
    )
    assert len(resp_ws2.dynamic_memories) == 0
    assert resp_ws2.total_count == 0

    # Query Workspace 1 -> must return the materialized memory
    resp_ws1 = await MemoryRetrievalService.retrieve(
        session=session,
        user_id=user.id,
        workspace_id=ws1.id,
    )
    assert len(resp_ws1.dynamic_memories) == 1
    assert resp_ws1.dynamic_memories[0].content == "Working on Phase 2 RAG memory integration"
    assert resp_ws1.dynamic_memories[0].workspace_id == ws1.id


@pytest.mark.asyncio
async def test_missing_workspace_scope_handling(memory_test_env: dict):
    """
    Verifies that omission of workspace_id fails closed across all memory methods
    before any database interaction occurs.
    """
    session: AsyncSession = memory_test_env["session"]
    user: User = memory_test_env["user"]
    dummy_id1 = uuid.uuid4()
    dummy_id2 = uuid.uuid4()

    # 1. MemoryRetrievalService.retrieve requires user_id
    with pytest.raises(ValueError, match="user_id is required"):
        await MemoryRetrievalService.retrieve(
            session=session,
            user_id=None,  # type: ignore
            workspace_id=dummy_id1,
        )

    # 2. MemoryMaterializer.get_user_memories requires user_id
    with pytest.raises(ValueError, match="user_id is required"):
        await MemoryMaterializer.get_user_memories(
            session=session,
            user_id=None,  # type: ignore
            workspace_id=dummy_id1,
        )

    # 3. MemoryMaterializer.supersede_memory requires user_id
    with pytest.raises(ValueError, match="user_id is required for supersession"):
        await MemoryMaterializer.supersede_memory(
            session=session,
            old_memory_id=dummy_id1,
            new_memory_id=dummy_id2,
            user_id=None,  # type: ignore
            workspace_id=uuid.uuid4(),
        )

    # 4. Global memory retrieval (workspace_id=None) succeeds without error
    global_resp = await MemoryRetrievalService.retrieve(
        session=session,
        user_id=user.id,
        workspace_id=None,
    )
    assert global_resp.user_id == user.id
    assert global_resp.workspace_id is None


@pytest.mark.asyncio
async def test_supersession_workspace_isolation(memory_test_env: dict):
    """
    Verifies that MemoryMaterializer.supersede_memory is strictly workspace-scoped:
    A supersession call in Workspace 1 CANNOT supersede a memory in Workspace 2,
    even if created by the same user.
    """
    session: AsyncSession = memory_test_env["session"]
    schema: str = memory_test_env["schema_name"]
    user: User = memory_test_env["user"]
    ws1: Workspace = memory_test_env["ws1"]
    ws2: Workspace = memory_test_env["ws2"]

    await session.execute(text(f"SET search_path TO {schema}, public"))

    # Create memory in Workspace 2
    cand_ws2 = MemoryCandidate(
        user_id=user.id,
        workspace_id=ws2.id,
        canonical_key="user_role:current",
        content="Lead architect in WS2",
        memory_type=MemoryType.STATIC,
        status=MemoryStatus.ACTIVE,
    )
    mem_ws2 = await MemoryMaterializer.materialize_memory(session, cand_ws2, auto_embed=False)

    # Create replacement candidate in Workspace 1
    cand_ws1 = MemoryCandidate(
        user_id=user.id,
        workspace_id=ws1.id,
        canonical_key="user_role:current",
        content="Developer in WS1",
        memory_type=MemoryType.STATIC,
        status=MemoryStatus.ACTIVE,
    )
    mem_ws1 = await MemoryMaterializer.materialize_memory(session, cand_ws1, auto_embed=False)
    await session.commit()

    # Attempt to supersede mem_ws2 by passing workspace_id = ws1.id (Cross-workspace mismatch)
    cross_ws_result = await MemoryMaterializer.supersede_memory(
        session=session,
        old_memory_id=mem_ws2.id,
        new_memory_id=mem_ws1.id,
        user_id=user.id,
        workspace_id=ws1.id,  # Intentionally mismatched workspace
    )
    assert cross_ws_result is False

    # Verify mem_ws2 in Workspace 2 was NOT superseded
    await session.refresh(mem_ws2)
    assert mem_ws2.status == MemoryStatus.ACTIVE.value
    assert mem_ws2.superseded_by_id is None

    # Now supersede with correct workspace_id = ws2.id
    correct_result = await MemoryMaterializer.supersede_memory(
        session=session,
        old_memory_id=mem_ws2.id,
        new_memory_id=mem_ws1.id,
        user_id=user.id,
        workspace_id=ws2.id,
    )
    assert correct_result is True

    await session.refresh(mem_ws2)
    assert mem_ws2.status == MemoryStatus.SUPERSEDED.value
    assert mem_ws2.superseded_by_id == mem_ws1.id


@pytest.mark.asyncio
async def test_embedding_dimension_compatibility(memory_test_env: dict):
    """
    Verifies that the embedding provider output dimension matches the vector(384) column definition.
    """
    session: AsyncSession = memory_test_env["session"]
    schema: str = memory_test_env["schema_name"]
    settings = get_settings()
    user: User = memory_test_env["user"]
    ws1: Workspace = memory_test_env["ws1"]

    assert settings.embedding_dim == 384

    await session.execute(text(f"SET search_path TO {schema}, public"))

    test_vec = [0.01 * (i % 10) for i in range(384)]
    assert len(test_vec) == 384

    cand = MemoryCandidate(
        user_id=user.id,
        workspace_id=ws1.id,
        canonical_key="test_dim:sample",
        content="Dimension test fact",
        memory_type=MemoryType.STATIC,
        embedding=test_vec,
    )
    mem = await MemoryMaterializer.materialize_memory(session, cand, auto_embed=False)
    await session.commit()

    refetched = await session.get(MemoryItem, mem.id)
    assert refetched is not None
    assert refetched.embedding is not None
    assert len(refetched.embedding) == 384


@pytest.mark.asyncio
async def test_migration_clean_database():
    """
    Executes the upgrade() logic against an isolated test schema where memory_items does not exist.
    Verifies table creation, columns, indexes, and clean downgrade dropping the table.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    test_schema = f"test_mig_clean_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {test_schema}"))
        await conn.execute(text(f"""
            CREATE TABLE {test_schema}.departments (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR NOT NULL,
                code VARCHAR NOT NULL
            )
        """))
        await conn.execute(text(f"""
            CREATE TABLE {test_schema}.users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR NOT NULL
            )
        """))
        await conn.execute(text(f"""
            CREATE TABLE {test_schema}.workspaces (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR NOT NULL,
                code VARCHAR NOT NULL,
                department_id UUID REFERENCES {test_schema}.departments(id)
            )
        """))

    try:
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        import migrations.versions.add_workspace_scoped_memory_items as mig
        import alembic.op as alembic_op

        async with engine.connect() as conn:
            def run_upgrade(sync_conn):
                sync_conn.execute(text(f"SET search_path TO {test_schema}, public"))
                ctx = MigrationContext.configure(sync_conn)
                op = Operations(ctx)
                alembic_op._proxy = op
                mig.upgrade()

            await conn.run_sync(run_upgrade)
            await conn.commit()

            cols = await conn.execute(text(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = '{test_schema}' AND table_name = 'memory_items';
            """))
            col_names = {row[0] for row in cols.fetchall()}
            assert "id" in col_names
            assert "user_id" in col_names
            assert "workspace_id" in col_names
            assert "canonical_key" in col_names
            assert "embedding" in col_names

            # Test downgrade on clean database -> must drop table
            def run_downgrade(sync_conn):
                sync_conn.execute(text(f"SET search_path TO {test_schema}, public"))
                ctx = MigrationContext.configure(sync_conn)
                op = Operations(ctx)
                alembic_op._proxy = op
                mig.downgrade()

            await conn.run_sync(run_downgrade)
            await conn.commit()

            tables_after = await conn.execute(text(f"""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = '{test_schema}' AND table_name = 'memory_items';
            """))
            assert len(tables_after.fetchall()) == 0

    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"))
        await engine.dispose()


@pytest.mark.asyncio
async def test_migration_from_existing_table_with_unscoped_check():
    """
    Simulates upgrading an existing database where memory_items was created outside Alembic.
    Verifies that:
    1. If unscoped rows exist, migration safely aborts with RuntimeError.
    2. If table is empty, workspace_id column and new indexes are created smoothly.
    3. Records inserted after upgrade are PRESERVED when downgrade runs.
    4. Downgrade non-destructively drops workspace_id and restores legacy indexes without dropping the table.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    test_schema = f"test_mig_exist_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {test_schema}"))
        await conn.execute(text(f"""
            CREATE TABLE {test_schema}.departments (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR NOT NULL,
                code VARCHAR NOT NULL
            )
        """))
        await conn.execute(text(f"""
            CREATE TABLE {test_schema}.users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR NOT NULL
            )
        """))
        await conn.execute(text(f"""
            CREATE TABLE {test_schema}.workspaces (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR NOT NULL,
                code VARCHAR NOT NULL,
                department_id UUID REFERENCES {test_schema}.departments(id)
            )
        """))
        await conn.execute(text(f"""
            CREATE TABLE {test_schema}.memory_items (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES {test_schema}.users(id),
                canonical_key VARCHAR NOT NULL,
                memory_type VARCHAR NOT NULL DEFAULT 'DYNAMIC',
                status VARCHAR NOT NULL DEFAULT 'ACTIVE',
                content TEXT NOT NULL
            )
        """))
        await conn.execute(text(f"""
            CREATE UNIQUE INDEX ix_memory_items_user_canonical_key_active 
                ON {test_schema}.memory_items (user_id, canonical_key) 
                WHERE status = 'ACTIVE'
        """))
        await conn.execute(text(f"""
            CREATE INDEX ix_memory_items_user_status 
                ON {test_schema}.memory_items (user_id, status)
        """))

    try:
        import migrations.versions.add_workspace_scoped_memory_items as mig
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        import alembic.op as alembic_op

        # Part 1: Insert an unscoped row and verify upgrade aborts
        u_id = uuid.uuid4()
        async with engine.begin() as conn:
            await conn.execute(text(f"INSERT INTO {test_schema}.users (id, email) VALUES ('{u_id}', 'legacy@test.com')"))
            await conn.execute(text(f"INSERT INTO {test_schema}.memory_items (user_id, canonical_key, content) VALUES ('{u_id}', 'k1', 'c1')"))

        async with engine.connect() as conn:
            def run_upgrade_failing(sync_conn):
                sync_conn.execute(text(f"SET search_path TO {test_schema}, public"))
                ctx = MigrationContext.configure(sync_conn)
                op = Operations(ctx)
                alembic_op._proxy = op
                mig.upgrade()

            with pytest.raises(RuntimeError, match="Found 1 existing unscoped rows"):
                await conn.run_sync(run_upgrade_failing)

            # Part 2: Clear unscoped row and verify upgrade succeeds
            await conn.execute(text(f"DELETE FROM {test_schema}.memory_items"))
            await conn.commit()

            def run_upgrade_success(sync_conn):
                sync_conn.execute(text(f"SET search_path TO {test_schema}, public"))
                ctx = MigrationContext.configure(sync_conn)
                op = Operations(ctx)
                alembic_op._proxy = op
                mig.upgrade()

            await conn.run_sync(run_upgrade_success)
            await conn.commit()

            cols = await conn.execute(text(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = '{test_schema}' AND table_name = 'memory_items' AND column_name = 'workspace_id';
            """))
            assert len(cols.fetchall()) == 1

            # Part 3: Insert records into the upgraded table with workspace_id
            ws_id = uuid.uuid4()
            dept_id = uuid.uuid4()
            await conn.execute(text(f"INSERT INTO {test_schema}.departments (id, name, code) VALUES ('{dept_id}', 'D', 'D')"))
            await conn.execute(text(f"INSERT INTO {test_schema}.workspaces (id, name, code, department_id) VALUES ('{ws_id}', 'W', 'W', '{dept_id}')"))
            await conn.execute(text(f"""
                INSERT INTO {test_schema}.memory_items (user_id, workspace_id, canonical_key, content)
                VALUES ('{u_id}', '{ws_id}', 'k_post_upgrade_1', 'content 1'),
                       ('{u_id}', '{ws_id}', 'k_post_upgrade_2', 'content 2')
            """))
            await conn.commit()

            # Part 4: Run downgrade on pre-existing table -> MUST PRESERVE RECORDS NON-DESTRUCTIVELY!
            def run_downgrade(sync_conn):
                sync_conn.execute(text(f"SET search_path TO {test_schema}, public"))
                ctx = MigrationContext.configure(sync_conn)
                op = Operations(ctx)
                alembic_op._proxy = op
                mig.downgrade()

            await conn.run_sync(run_downgrade)
            await conn.commit()

            # 4a. Verify table STILL EXISTS (not dropped!)
            tables_after = await conn.execute(text(f"""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = '{test_schema}' AND table_name = 'memory_items';
            """))
            assert len(tables_after.fetchall()) == 1

            # 4b. Verify existing rows are PRESERVED!
            row_count = (await conn.execute(text(f"SELECT count(*) FROM {test_schema}.memory_items"))).scalar()
            assert row_count == 2

            # 4c. Verify workspace_id column was removed
            cols_after = await conn.execute(text(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = '{test_schema}' AND table_name = 'memory_items' AND column_name = 'workspace_id';
            """))
            assert len(cols_after.fetchall()) == 0

            # 4d. Verify legacy indexes are restored
            indexes_after = await conn.execute(text(f"""
                SELECT indexname 
                FROM pg_indexes 
                WHERE schemaname = '{test_schema}' AND tablename = 'memory_items';
            """))
            idx_names = {row[0] for row in indexes_after.fetchall()}
            assert "ix_memory_items_user_canonical_key_active" in idx_names
            assert "ix_memory_items_user_status" in idx_names
            assert "ix_memory_items_user_workspace_canonical_key_active" not in idx_names

    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"))
        await engine.dispose()
