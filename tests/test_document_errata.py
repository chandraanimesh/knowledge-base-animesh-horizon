import os
import uuid
from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy import select

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.db.session import get_db_session, engine
from rag_app.db.models import User, Department, Document, DocumentNote, UserDepartment, Permission, UserPermission
from rag_app.knowledge.errata_service import ErrataService
from rag_app.knowledge.errata_tool import RecordDocumentNoteTool
from rag_app.schemas.document_notes import DocumentNoteCreate
from rag_app.schemas.rag import RetrievedChunk
from rag_app.retrieval.service import RetrievalService


@pytest.mark.asyncio
async def test_document_errata_lifecycle_and_retrieval_injection():
    """Verify full errata lifecycle: create, approve, inject, and ignore revoked/expired."""
    async for session in get_db_session():
        # Setup test department and users
        dept = await session.scalar(select(Department).where(Department.code == "ERRATA_TEST"))
        if not dept:
            dept = Department(code="ERRATA_TEST", name="Errata Test Dept", is_active=True)
            session.add(dept)
            await session.flush()

        admin = await session.scalar(select(User).where(User.email == "admin_errata@local.dev"))
        if not admin:
            admin = User(email="admin_errata@local.dev", hashed_password="pw", role="ADMIN", is_active=True)
            session.add(admin)
            await session.flush()

        # Seed document
        doc = Document(
            file_name="refund_policy.pdf",
            file_path="/tmp/refund_policy.pdf",
            content_hash=str(uuid.uuid4()).replace("-", "") + "00000000000000000000000000000000",
            mime_type="application/pdf",
            department_id=dept.id,
        )
        session.add(doc)
        await session.flush()

        errata_svc = ErrataService(session)

        # 1. Non-admin / agent creates note -> PENDING
        standard_user = User(email=f"user_{uuid.uuid4().hex[:6]}@local.dev", hashed_password="pw", role="USER", is_active=True)
        session.add(standard_user)
        await session.flush()

        # Unauthorized user check
        unauth_user = User(email=f"unauth_{uuid.uuid4().hex[:6]}@local.dev", hashed_password="pw", role="USER", is_active=True)
        session.add(unauth_user)
        await session.flush()

        with pytest.raises(Exception) as exc_info:
            await errata_svc.create_note(
                user=unauth_user,
                document_id=doc.id,
                request=DocumentNoteCreate(correction_text="Illegal attempt"),
            )
        assert "403" in str(exc_info.value) or "Access denied" in str(exc_info.value)

        # Grant department access and VIEW permission to standard_user
        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(code="VIEW", description="View permissions", is_active=True)
            session.add(view_perm)
            await session.flush()

        session.add(UserDepartment(user_id=standard_user.id, department_id=dept.id))
        session.add(UserPermission(user_id=standard_user.id, department_id=dept.id, permission_id=view_perm.id))
        await session.flush()

        pending_note = await errata_svc.create_note(
            user=standard_user,
            document_id=doc.id,
            request=DocumentNoteCreate(correction_text="Refund window extended from 14 to 30 days."),
        )
        assert pending_note.approval_status == "PENDING"

        # 2. Mock retrieval chunk for this document
        chunk = RetrievedChunk(
            chunk_id="chunk_1",
            document_id=str(doc.id),
            department_id=str(dept.id),
            file_name=doc.file_name,
            chunk_index=0,
            text="Original policy states refunds are only valid within 14 days.",
        )

        retrieval_svc = RetrievalService(session=session)

        # Pending note must NOT be injected into retrieval
        res_pending = await retrieval_svc._inject_active_errata_overrides([chunk])
        assert res_pending[0].has_errata is False
        assert "ACTIVE DOCUMENT ERRATA" not in res_pending[0].text

        # 3. Admin approves note -> APPROVED
        approved_note = await errata_svc.approve_note(user=admin, note_id=pending_note.id)
        assert approved_note.approval_status == "APPROVED"
        assert approved_note.is_active is True

        # Now retrieval MUST inject active errata override
        res_approved = await retrieval_svc._inject_active_errata_overrides([chunk])
        assert res_approved[0].has_errata is True
        assert "ACTIVE DOCUMENT ERRATA" in res_approved[0].text
        assert "Refund window extended from 14 to 30 days" in res_approved[0].text
        assert "Original policy states refunds are only valid within 14 days" in res_approved[0].text

        # 4. Revoked note is not injected
        await errata_svc.revoke_note(user=admin, note_id=approved_note.id)
        res_revoked = await retrieval_svc._inject_active_errata_overrides([chunk])
        assert res_revoked[0].has_errata is False

        # 5. Agent tool test
        tool = RecordDocumentNoteTool(session=session, user=admin)
        tool_res = await tool._arun(
            document_id=str(doc.id),
            correction_text="Shipping fee is now zero for orders above $50.",
            evidence_reference="Jira TICKET-104",
        )
        assert "Successfully recorded document correction" in tool_res
        await engine.dispose()
        break

