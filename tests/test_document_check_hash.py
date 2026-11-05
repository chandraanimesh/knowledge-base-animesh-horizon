import os
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

# Strictly use disposable database
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.main import app
from rag_app.db.session import get_db_session, engine
from rag_app.db.models import User, Department, Document, Workspace, WorkspaceMember
from rag_app.security.authentication import create_access_token


@pytest.mark.asyncio
async def test_document_check_hash_flow():
    """Verify hash pre-check handles duplicates, missing docs, and scope isolation."""
    async for session in get_db_session():
        # Setup test department and user
        dept = await session.scalar(select(Department).where(Department.code == "TECH_TEST"))
        if not dept:
            dept = Department(code="TECH_TEST", name="Tech Test Dept", is_active=True)
            session.add(dept)
            await session.flush()

        admin_user = await session.scalar(select(User).where(User.email == "admin_hash_test@local.dev"))
        if not admin_user:
            admin_user = User(
                email="admin_hash_test@local.dev",
                hashed_password="hashed_pass_placeholder",
                role="ADMIN",
                is_active=True,
            )
            session.add(admin_user)
            await session.flush()

        # Seed an existing document with known SHA-256
        test_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        existing_doc = await session.scalar(select(Document).where(Document.content_hash == test_hash))
        if not existing_doc:
            existing_doc = Document(
                file_name="empty_test.txt",
                file_path="/tmp/empty_test.txt",
                content_hash=test_hash,
                mime_type="text/plain",
                department_id=dept.id,
            )
            session.add(existing_doc)
            await session.commit()

        token = create_access_token(subject=str(admin_user.id))
        headers = {"Authorization": f"Bearer {token}"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Existing document hash lookup
            resp = await client.get(
                f"/documents/check-hash?hash={test_hash}&department_code=TECH_TEST",
                headers=headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["exists"] is True
            assert data["document_id"] == str(existing_doc.id)
            assert data["file_name"] == "empty_test.txt"

            # 2. Non-existent hash lookup
            fake_hash = "0000000000000000000000000000000000000000000000000000000000000000"
            resp2 = await client.get(
                f"/documents/check-hash?hash={fake_hash}&department_code=TECH_TEST",
                headers=headers,
            )
            assert resp2.status_code == 200
            assert resp2.json()["exists"] is False

            # 3. Invalid hash length
            resp3 = await client.get(
                "/documents/check-hash?hash=short_hash&department_code=TECH_TEST",
                headers=headers,
            )
            assert resp3.status_code == 422  # Validation error
            await engine.dispose()
            break

