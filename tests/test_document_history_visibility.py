import uuid
import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_app.api.routes import list_documents
from rag_app.core.config import get_settings
from rag_app.db.models import Department, Document, Permission, User, UserDepartment, UserPermission


@pytest_asyncio.fixture
async def db_session():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_document_history_admin_visibility_and_user_isolation(db_session: AsyncSession):
    suffix = uuid.uuid4().hex[:8]

    # Look up VIEW permission
    view_perm = await db_session.scalar(select(Permission).where(Permission.code == "VIEW"))
    assert view_perm is not None, "VIEW permission must exist in DB"

    # Create 2 test departments
    dept_sales = Department(name=f"Sales {suffix}", code=f"SLS_{suffix}", is_active=True)
    dept_hr = Department(name=f"HR {suffix}", code=f"HR_{suffix}", is_active=True)
    db_session.add_all([dept_sales, dept_hr])
    await db_session.commit()

    # Create Admin and User
    admin = User(
        email=f"admin_{suffix}@test.local",
        hashed_password="pw",
        role="ADMIN",
        is_active=True,
    )
    user_sales = User(
        email=f"sales_rep_{suffix}@test.local",
        hashed_password="pw",
        role="USER",
        is_active=True,
    )
    db_session.add_all([admin, user_sales])
    await db_session.commit()

    # Grant user VIEW permission to SLS only
    ud_sales = UserDepartment(user_id=user_sales.id, department_id=dept_sales.id)
    up_sales = UserPermission(user_id=user_sales.id, department_id=dept_sales.id, permission_id=view_perm.id)
    db_session.add_all([ud_sales, up_sales])
    await db_session.commit()

    # Create 2 documents: one in Sales, one in HR
    doc_sales = Document(
        file_name=f"{uuid.uuid4().hex}_q3_pipeline.xlsx",
        file_path=f"uploaded_docs/{uuid.uuid4().hex}_q3_pipeline.xlsx",
        content_hash=f"hash_sls_{suffix}",
        department_id=dept_sales.id,
        created_by=user_sales.id,
        status="READY",
    )
    doc_hr = Document(
        file_name=f"{uuid.uuid4().hex}_benefits_guide.pdf",
        file_path=f"uploaded_docs/{uuid.uuid4().hex}_benefits_guide.pdf",
        content_hash=f"hash_hr_{suffix}",
        department_id=dept_hr.id,
        created_by=admin.id,
        status="PROCESSING",
    )
    db_session.add_all([doc_sales, doc_hr])
    await db_session.commit()

    try:
        # 1. Admin lists documents
        admin_docs = await list_documents(session=db_session, user=admin)
        admin_doc_ids = {d.id for d in admin_docs}
        assert doc_sales.id in admin_doc_ids
        assert doc_hr.id in admin_doc_ids

        # Verify admin metadata on Sales doc
        sls_entry = next(d for d in admin_docs if d.id == doc_sales.id)
        assert sls_entry.original_file_name == "q3_pipeline.xlsx"
        assert sls_entry.department_code == dept_sales.code
        assert sls_entry.department_name == dept_sales.name
        assert sls_entry.uploaded_by_email == user_sales.email
        assert sls_entry.uploaded_by_name == f"Sales_rep_{suffix}"
        assert sls_entry.status == "READY"
        assert sls_entry.created_at is not None

        # Verify admin metadata on HR doc
        hr_entry = next(d for d in admin_docs if d.id == doc_hr.id)
        assert hr_entry.original_file_name == "benefits_guide.pdf"
        assert hr_entry.department_code == dept_hr.code
        assert hr_entry.uploaded_by_email == admin.email
        assert hr_entry.status == "PROCESSING"

        # 2. Regular user (with Sales VIEW access only) lists documents
        user_docs = await list_documents(session=db_session, user=user_sales)
        user_doc_ids = {d.id for d in user_docs}
        assert doc_sales.id in user_doc_ids, "User must see documents from their authorized department"
        assert doc_hr.id not in user_doc_ids, "User must NOT see documents from unauthorized department"

    finally:
        # Clean up test records
        await db_session.execute(delete(Document).where(Document.id.in_([doc_sales.id, doc_hr.id])))
        await db_session.execute(
            delete(UserPermission).where(
                UserPermission.user_id == user_sales.id,
                UserPermission.department_id == dept_sales.id,
            )
        )
        await db_session.execute(
            delete(UserDepartment).where(
                UserDepartment.user_id == user_sales.id,
                UserDepartment.department_id == dept_sales.id,
            )
        )
        await db_session.execute(delete(User).where(User.id.in_([admin.id, user_sales.id])))
        await db_session.execute(delete(Department).where(Department.id.in_([dept_sales.id, dept_hr.id])))
        await db_session.commit()
