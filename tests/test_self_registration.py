import os
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.db.session import get_db_session
from rag_app.db.models import User, Department, Permission
from rag_app.main import app
from rag_app.security.authentication import create_access_token


@pytest.mark.asyncio
async def test_self_registration_and_admin_approval_lifecycle():
    """Verify registration, login block, admin approval, login success, and rejection."""
    async for session in get_db_session():
        # 1. Setup admin and test department
        admin = await session.scalar(select(User).where(User.email == "admin_reg@local.dev"))
        if not admin:
            admin = User(email="admin_reg@local.dev", hashed_password="pw", role="ADMIN", is_active=True)
            session.add(admin)
            await session.flush()

        dept = await session.scalar(select(Department).where(Department.code == "REG_TEST"))
        if not dept:
            dept = Department(code="REG_TEST", name="Registration Test Dept", is_active=True)
            session.add(dept)
            await session.flush()

        perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not perm:
            perm = Permission(code="VIEW", name="View", description="View access", is_active=True)
            session.add(perm)
            await session.flush()

        await session.commit()

        admin_token = create_access_token(subject=str(admin.id))
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            user_email = f"candidate_{uuid.uuid4().hex[:6]}@local.dev"
            user_pw = "Password123!"

            # 2. Candidate self-registers
            reg_resp = await client.post(
                "/auth/register",
                json={
                    "email": user_email,
                    "password": user_pw,
                    "requested_department": "REG_TEST",
                    "reason": "Needs access for security auditing",
                },
            )
            assert reg_resp.status_code == 201
            reg_data = reg_resp.json()
            assert reg_data["status"] == "PENDING_APPROVAL"
            candidate_id = reg_data["user_id"]

            # 3. Candidate attempts login before approval -> 403 Forbidden
            login_resp_pending = await client.post(
                "/auth/login",
                data={"username": user_email, "password": user_pw},
            )
            assert login_resp_pending.status_code == 403
            assert "pending administrator approval" in login_resp_pending.json()["detail"].lower()

            # 4. Admin checks pending registrations
            pending_list_resp = await client.get("/admin/registrations/pending", headers=admin_headers)
            assert pending_list_resp.status_code == 200
            pending_users = pending_list_resp.json()
            found = [u for u in pending_users if u["id"] == candidate_id]
            assert len(found) == 1
            assert found[0]["email"] == user_email

            # 5. Admin approves candidate
            approve_resp = await client.post(
                f"/admin/registrations/{candidate_id}/approve",
                headers=admin_headers,
                json={"department_codes": ["REG_TEST"], "role": "USER"},
            )
            assert approve_resp.status_code == 200
            assert approve_resp.json()["status"] == "APPROVED"

            # 6. Candidate logs in successfully after approval
            login_resp_approved = await client.post(
                "/auth/login",
                data={"username": user_email, "password": user_pw},
            )
            assert login_resp_approved.status_code == 200
            assert "access_token" in login_resp_approved.json()

            # 7. Candidate 2 registers and gets rejected
            user2_email = f"reject_{uuid.uuid4().hex[:6]}@local.dev"
            reg2_resp = await client.post(
                "/auth/register",
                json={"email": user2_email, "password": user_pw},
            )
            assert reg2_resp.status_code == 201
            user2_id = reg2_resp.json()["user_id"]

            reject_resp = await client.post(
                f"/admin/registrations/{user2_id}/reject",
                headers=admin_headers,
                json={"reason": "Role unverified"},
            )
            assert reject_resp.status_code == 200
            assert reject_resp.json()["status"] == "REJECTED"

            # 8. Rejected user attempts login -> 403 Forbidden
            login_resp_rejected = await client.post(
                "/auth/login",
                data={"username": user2_email, "password": user_pw},
            )
            assert login_resp_rejected.status_code == 403
            assert "declined" in login_resp_rejected.json()["detail"].lower()
        from rag_app.db.session import engine
        await engine.dispose()
        break

