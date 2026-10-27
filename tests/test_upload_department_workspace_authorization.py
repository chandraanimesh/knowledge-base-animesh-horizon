"""
Regression and security test suite for Phase 2B.1:
Department/Workspace Upload Authorization Fix.

Covers:
1. Authorized USER uploading into a matching department/workspace -> HTTP 200 / IngestResponse.
2. USER attempting a mismatched department/workspace upload -> HTTP 403 Forbidden.
3. ADMIN uploading into a mismatched department/workspace -> Allowed per existing policy.
4. Missing or invalid department/workspace values -> HTTP 404.
5. Existing upload regression behavior: department upload without workspace_id -> Success.
"""

import io
import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rag_app.core.config import get_settings
from rag_app.db.models import (
    User,
    Department,
    Workspace,
    WorkspaceMember,
    Permission,
    UserPermission,
    UserDepartment,
)
from rag_app.schemas.rag import IngestResponse
from rag_app.api.routes import upload_document


@pytest_asyncio.fixture
async def upload_test_env():
    """Sets up an isolated test schema with live PostgreSQL tables."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    schema_name = f"test_up_auth_{uuid.uuid4().hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        def create_tables(sync_conn):
            sync_conn.execute(text(f"SET search_path TO {schema_name}, public"))
            User.__table__.create(sync_conn)
            Department.__table__.create(sync_conn)
            Workspace.__table__.create(sync_conn)
            WorkspaceMember.__table__.create(sync_conn)
            Permission.__table__.create(sync_conn)
            UserDepartment.__table__.create(sync_conn)
            UserPermission.__table__.create(sync_conn)
        await conn.run_sync(create_tables)

    suffix = uuid.uuid4().hex[:6]
    admin_id = uuid.uuid4()
    user_id = uuid.uuid4()
    sales_dept_id = uuid.uuid4()
    eng_dept_id = uuid.uuid4()
    sales_ws_id = uuid.uuid4()
    eng_ws_id = uuid.uuid4()

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))

        # Create permissions
        upload_perm = Permission(id=uuid.uuid4(), code="UPLOAD", description="Upload files", is_active=True)
        view_perm = Permission(id=uuid.uuid4(), code="VIEW", description="View files", is_active=True)
        session.add_all([upload_perm, view_perm])

        # Create 2 Departments: SALES and ENGINEERING
        sales_dept = Department(id=sales_dept_id, code=f"SALES_{suffix}", name="Sales Department", is_active=True)
        eng_dept = Department(id=eng_dept_id, code=f"ENG_{suffix}", name="Engineering Department", is_active=True)
        session.add_all([sales_dept, eng_dept])

        # Create Admin
        admin = User(
            id=admin_id,
            email=f"admin_{suffix}@acme.com",
            hashed_password="pw",
            role="ADMIN",
            is_active=True,
        )

        # Create User with UPLOAD and VIEW on SALES, and VIEW on ENG
        user = User(
            id=user_id,
            email=f"user_{suffix}@acme.com",
            hashed_password="pw",
            role="USER",
            is_active=True,
        )
        session.add_all([admin, user])

        # Assign user department memberships
        session.add(UserDepartment(user_id=user_id, department_id=sales_dept_id))
        session.add(UserDepartment(user_id=user_id, department_id=eng_dept_id))
        await session.flush()

        # Assign user permissions
        session.add(UserPermission(user_id=user_id, department_id=sales_dept_id, permission_id=upload_perm.id))
        session.add(UserPermission(user_id=user_id, department_id=sales_dept_id, permission_id=view_perm.id))
        session.add(UserPermission(user_id=user_id, department_id=eng_dept_id, permission_id=view_perm.id))

        # Create Workspaces: Sales Workspace (under SALES) and Engineering Workspace (under ENG)
        sales_ws = Workspace(
            id=sales_ws_id,
            code=f"sales_ws_{suffix}",
            name="Sales Team Workspace",
            department_id=sales_dept_id,
            created_by=admin_id,
            is_active=True,
        )
        eng_ws = Workspace(
            id=eng_ws_id,
            code=f"eng_ws_{suffix}",
            name="Engineering Team Workspace",
            department_id=eng_dept_id,
            created_by=admin_id,
            is_active=True,
        )
        session.add_all([sales_ws, eng_ws])

        # Enroll user as member in both workspaces
        session.add(WorkspaceMember(workspace_id=sales_ws_id, user_id=user_id, role="EDITOR"))
        session.add(WorkspaceMember(workspace_id=eng_ws_id, user_id=user_id, role="EDITOR"))

        await session.commit()

    yield {
        "engine": engine,
        "session_maker": session_maker,
        "schema_name": schema_name,
        "admin_id": admin_id,
        "user_id": user_id,
        "sales_dept_code": f"SALES_{suffix}",
        "eng_dept_code": f"ENG_{suffix}",
        "sales_dept_id": sales_dept_id,
        "eng_dept_id": eng_dept_id,
        "sales_ws_id": sales_ws_id,
        "eng_ws_id": eng_ws_id,
    }

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
    await engine.dispose()


def make_mock_file(filename: str = "test_document.txt", content: bytes = b"Sample document text"):
    return UploadFile(filename=filename, file=io.BytesIO(content))


def make_mock_ingestion_service(doc_id: str | None = None):
    svc = AsyncMock()
    svc.ingest_file = AsyncMock(
        return_value=IngestResponse(
            document_id=doc_id or str(uuid.uuid4()),
            chunks_count=2,
            skipped=False,
            file_name="test_document.txt",
        )
    )
    return svc


# ── Test 1: Matching Department / Workspace Upload (USER) ────────────────────

@pytest.mark.asyncio
async def test_user_upload_matching_department_and_workspace_succeeds(upload_test_env):
    """Authorized USER uploading to a workspace whose department matches department_code succeeds."""
    session_maker = upload_test_env["session_maker"]
    schema_name = upload_test_env["schema_name"]
    user_id = upload_test_env["user_id"]
    sales_dept_code = upload_test_env["sales_dept_code"]
    sales_ws_id = upload_test_env["sales_ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user = await session.get(User, user_id)
        assert user is not None

        service = make_mock_ingestion_service()
        file = make_mock_file()
        bg_tasks = BackgroundTasks()

        resp = await upload_document(
            department_code=sales_dept_code,
            service=service,
            session=session,
            background_tasks=bg_tasks,
            file=file,
            workspace_id=str(sales_ws_id),
            user=user,
        )

        assert resp.document_id is not None
        assert resp.file_name == "test_document.txt"
        assert service.ingest_file.called
        kwargs = service.ingest_file.call_args.kwargs
        assert kwargs["department_id"] == upload_test_env["sales_dept_id"]
        assert kwargs["workspace_id"] == sales_ws_id


# ── Test 2: Mismatched Department / Workspace Upload (USER) ──────────────────

@pytest.mark.asyncio
async def test_user_upload_mismatched_department_and_workspace_rejected_403(upload_test_env):
    """
    USER uploading to sales department but targeting engineering workspace
    must be explicitly rejected with HTTP 403 Forbidden.
    """
    session_maker = upload_test_env["session_maker"]
    schema_name = upload_test_env["schema_name"]
    user_id = upload_test_env["user_id"]
    sales_dept_code = upload_test_env["sales_dept_code"]
    eng_ws_id = upload_test_env["eng_ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user = await session.get(User, user_id)
        assert user is not None

        service = make_mock_ingestion_service()
        file = make_mock_file()
        bg_tasks = BackgroundTasks()

        with pytest.raises(HTTPException) as exc_info:
            await upload_document(
                department_code=sales_dept_code,
                service=service,
                session=session,
                background_tasks=bg_tasks,
                file=file,
                workspace_id=str(eng_ws_id),
                user=user,
            )

        assert exc_info.value.status_code == 403
        assert "belongs to a different department" in exc_info.value.detail
        assert not service.ingest_file.called, "Ingestion must not be invoked on mismatched upload"


# ── Test 3: ADMIN Behavior for Mismatched Upload ─────────────────────────────

@pytest.mark.asyncio
async def test_admin_upload_mismatched_department_allowed_per_policy(upload_test_env):
    """ADMIN role preserves the exemption to assign documents across workspace/department boundaries."""
    session_maker = upload_test_env["session_maker"]
    schema_name = upload_test_env["schema_name"]
    admin_id = upload_test_env["admin_id"]
    sales_dept_code = upload_test_env["sales_dept_code"]
    eng_ws_id = upload_test_env["eng_ws_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        admin = await session.get(User, admin_id)
        assert admin is not None

        service = make_mock_ingestion_service()
        file = make_mock_file()
        bg_tasks = BackgroundTasks()

        resp = await upload_document(
            department_code=sales_dept_code,
            service=service,
            session=session,
            background_tasks=bg_tasks,
            file=file,
            workspace_id=str(eng_ws_id),
            user=admin,
        )

        assert resp.document_id is not None
        assert service.ingest_file.called
        kwargs = service.ingest_file.call_args.kwargs
        assert kwargs["department_id"] == upload_test_env["sales_dept_id"]
        assert kwargs["workspace_id"] == eng_ws_id


# ── Test 4: Missing or Invalid Department / Workspace Values ─────────────────

@pytest.mark.asyncio
async def test_upload_invalid_department_returns_404(upload_test_env):
    """Upload targeting a non-existent department code must return HTTP 404."""
    session_maker = upload_test_env["session_maker"]
    schema_name = upload_test_env["schema_name"]
    user_id = upload_test_env["user_id"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user = await session.get(User, user_id)

        service = make_mock_ingestion_service()
        file = make_mock_file()
        bg_tasks = BackgroundTasks()

        with pytest.raises(HTTPException) as exc_info:
            await upload_document(
                department_code="NON_EXISTENT_DEPT",
                service=service,
                session=session,
                background_tasks=bg_tasks,
                file=file,
                user=user,
            )

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_upload_inactive_department_returns_400(upload_test_env):
    """Upload targeting an inactive department code must return HTTP 400."""
    session_maker = upload_test_env["session_maker"]
    schema_name = upload_test_env["schema_name"]
    user_id = upload_test_env["user_id"]
    suffix = uuid.uuid4().hex[:6]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        inactive_dept = Department(
            id=uuid.uuid4(),
            code=f"INACTIVE_{suffix}",
            name="Inactive Dept",
            is_active=False,
        )
        session.add(inactive_dept)
        await session.commit()

        user = await session.get(User, user_id)
        service = make_mock_ingestion_service()
        file = make_mock_file()
        bg_tasks = BackgroundTasks()

        with pytest.raises(HTTPException) as exc_info:
            await upload_document(
                department_code=inactive_dept.code,
                service=service,
                session=session,
                background_tasks=bg_tasks,
                file=file,
                user=user,
            )

        assert exc_info.value.status_code == 400
        assert "inactive" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_upload_unauthorized_workspace_returns_403(upload_test_env):
    """Upload targeting a workspace in a department the user has no access to must return HTTP 403."""
    session_maker = upload_test_env["session_maker"]
    schema_name = upload_test_env["schema_name"]
    admin_id = upload_test_env["admin_id"]
    user_id = upload_test_env["user_id"]
    sales_dept_code = upload_test_env["sales_dept_code"]
    suffix = uuid.uuid4().hex[:6]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        # Create third department LEGAL where user has no VIEW or UPLOAD permission
        legal_dept = Department(
            id=uuid.uuid4(),
            code=f"LEGAL_{suffix}",
            name="Legal Dept",
            is_active=True,
        )
        session.add(legal_dept)

        legal_ws = Workspace(
            id=uuid.uuid4(),
            code=f"legal_ws_{suffix}",
            name="Legal Workspace",
            department_id=legal_dept.id,
            created_by=admin_id,
            is_active=True,
        )
        session.add(legal_ws)
        await session.commit()

        user = await session.get(User, user_id)
        service = make_mock_ingestion_service()
        file = make_mock_file()
        bg_tasks = BackgroundTasks()

        with pytest.raises(HTTPException) as exc_info:
            await upload_document(
                department_code=sales_dept_code,
                service=service,
                session=session,
                background_tasks=bg_tasks,
                file=file,
                workspace_id=str(legal_ws.id),
                user=user,
            )

        assert exc_info.value.status_code == 403
        assert not service.ingest_file.called


# ── Test 5: Existing Regression Behavior (No Workspace ID) ───────────────────

@pytest.mark.asyncio
async def test_user_upload_without_workspace_id_succeeds(upload_test_env):
    """Department-level upload without workspace_id succeeds and sets workspace_id=None."""
    session_maker = upload_test_env["session_maker"]
    schema_name = upload_test_env["schema_name"]
    user_id = upload_test_env["user_id"]
    sales_dept_code = upload_test_env["sales_dept_code"]

    async with session_maker() as session:
        await session.execute(text(f"SET search_path TO {schema_name}, public"))
        user = await session.get(User, user_id)

        service = make_mock_ingestion_service()
        file = make_mock_file()
        bg_tasks = BackgroundTasks()

        resp = await upload_document(
            department_code=sales_dept_code,
            service=service,
            session=session,
            background_tasks=bg_tasks,
            file=file,
            workspace_id=None,
            user=user,
        )

        assert resp.document_id is not None
        assert service.ingest_file.called
        kwargs = service.ingest_file.call_args.kwargs
        assert kwargs["department_id"] == upload_test_env["sales_dept_id"]
        assert kwargs["workspace_id"] is None
