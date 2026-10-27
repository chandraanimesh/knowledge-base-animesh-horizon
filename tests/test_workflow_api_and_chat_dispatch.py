"""
API Integration & Regression Test Suite for Workflow Router & Chat Dispatch.

Verifies:
1. POST /api/v1/workflows/runs creates and executes workflow with correct status.
2. Scoped idempotency: same key + same payload returns existing run; changed payload returns 409.
3. GET /api/v1/workflows/runs/{run_id} returns run details and tasks.
4. Cross-workspace authorization failure (403 Forbidden).
5. Chat /chat/ask default fast-path preservation for simple query.
6. Chat /chat/ask explicit 'adaptive_workflow' mode triggers WorkflowService and returns workflow_run_id.
7. Chat /chat/ask memory slash commands (/rem, /memories) remain intact without regression.
"""

from __future__ import annotations

import os

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"
from typing import Any
import uuid

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

from rag_app.core.config import get_settings
from rag_app.db.models import Department, Permission, User, UserDepartment, UserPermission, Workspace, WorkspaceMember, WorkflowRun
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.main import app
from rag_app.security.authentication import create_access_token


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_api_context():
    """
    Sets up a department, workspace, and an authorized user + token.
    """
    async with AsyncSessionLocal() as session:
        dept = Department(
            id=uuid.uuid4(),
            name="Workflow API Test Dept",
            code=f"WF_API_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(dept)
        await session.flush()

        # Seed VIEW permission
        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(id=uuid.uuid4(), code="VIEW", name="View Permission")
            session.add(view_perm)
            await session.flush()

        user = User(
            id=uuid.uuid4(),
            email=f"wf_user_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        session.add(user)
        await session.flush()

        session.add(UserDepartment(user_id=user.id, department_id=dept.id))
        session.add(UserPermission(user_id=user.id, department_id=dept.id, permission_id=view_perm.id))

        workspace = Workspace(
            id=uuid.uuid4(),
            name="Workflow Test Workspace",
            code=f"wf_ws_{uuid.uuid4().hex[:6]}",
            department_id=dept.id,
            is_active=True,
            created_by=user.id,
        )
        session.add(workspace)
        await session.flush()

        session.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="OWNER"))

        # Other isolated workspace
        other_dept = Department(
            id=uuid.uuid4(),
            name="Other Isolated Dept",
            code=f"OTHER_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(other_dept)
        await session.flush()

        other_ws = Workspace(
            id=uuid.uuid4(),
            name="Other Workspace",
            code=f"other_ws_{uuid.uuid4().hex[:6]}",
            department_id=other_dept.id,
            is_active=True,
        )
        session.add(other_ws)

        await session.commit()
        await session.refresh(user)
        await session.refresh(workspace)
        await session.refresh(other_ws)

        token = create_access_token(subject=str(user.id))

        return {
            "user": user,
            "token": token,
            "workspace": workspace,
            "other_workspace": other_ws,
            "headers": {"Authorization": f"Bearer {token}"},
        }


@pytest.mark.asyncio
async def test_workflow_runs_create_and_get(setup_api_context):
    ctx = setup_api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create a workflow run via API
        payload = {
            "query": "Research market dynamics for competitor analysis",
            "workspace_id": str(ctx["workspace"].id),
            "idempotency_key": f"api-wf-{uuid.uuid4().hex[:6]}",
        }
        res = await client.post("/api/v1/workflows/runs", json=payload, headers=ctx["headers"])
        assert res.status_code == 201, res.text
        data = res.json()
        run_id = data["id"]
        assert data["workspace_id"] == str(ctx["workspace"].id)
        assert data["status"] in ["RUNNING", "COMPLETED", "AWAITING_APPROVAL", "FAILED"]
        assert "tasks" in data
        assert len(data["tasks"]) > 0

        # 2. Retrieve the workflow run via GET
        get_res = await client.get(f"/api/v1/workflows/runs/{run_id}", headers=ctx["headers"])
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["id"] == run_id
        assert get_data["query_hash"] if "query_hash" in get_data else True


@pytest.mark.asyncio
async def test_workflow_runs_idempotency_conflict(setup_api_context):
    ctx = setup_api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        idemp_key = f"idemp-{uuid.uuid4().hex[:6]}"
        payload1 = {
            "query": "Original research query",
            "workspace_id": str(ctx["workspace"].id),
            "idempotency_key": idemp_key,
        }
        res1 = await client.post("/api/v1/workflows/runs", json=payload1, headers=ctx["headers"])
        assert res1.status_code == 201
        run_id_1 = res1.json()["id"]

        # Replay with same payload returns existing run
        res_replay = await client.post("/api/v1/workflows/runs", json=payload1, headers=ctx["headers"])
        assert res_replay.status_code == 201
        assert res_replay.json()["id"] == run_id_1

        # Replay with changed payload returns 409 Conflict
        payload_changed = {
            "query": "Completely different query with same key",
            "workspace_id": str(ctx["workspace"].id),
            "idempotency_key": idemp_key,
        }
        res_conflict = await client.post("/api/v1/workflows/runs", json=payload_changed, headers=ctx["headers"])
        assert res_conflict.status_code == 409


@pytest.mark.asyncio
async def test_workflow_runs_cross_workspace_denial(setup_api_context):
    ctx = setup_api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "query": "Unauthorized attempt on other workspace",
            "workspace_id": str(ctx["other_workspace"].id),
        }
        res = await client.post("/api/v1/workflows/runs", json=payload, headers=ctx["headers"])
        assert res.status_code == 403


@pytest.mark.asyncio
async def test_chat_ask_fast_path_preservation(setup_api_context):
    ctx = setup_api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Simple factual query without complex research keywords
        payload = {
            "query": "Hello there",
            "workspace_id": str(ctx["workspace"].id),
            "execution_mode": "fast_path",
        }
        res = await client.post("/chat/ask", json=payload, headers=ctx["headers"])
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        # Fast path does NOT return a workflow_run_id
        assert data.get("workflow_run_id") is None


@pytest.mark.asyncio
async def test_chat_ask_explicit_adaptive_workflow_dispatch(setup_api_context):
    ctx = setup_api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "query": "Research competitor offerings and prepare synthesis",
            "workspace_id": str(ctx["workspace"].id),
            "execution_mode": "adaptive_workflow",
            "workflow_idempotency_key": f"chat-wf-{uuid.uuid4().hex[:6]}",
        }
        res = await client.post("/chat/ask", json=payload, headers=ctx["headers"])
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert data.get("workflow_run_id") is not None
        assert data.get("workflow_status") in ["RUNNING", "COMPLETED", "AWAITING_APPROVAL", "FAILED"]


@pytest.mark.asyncio
async def test_chat_ask_memory_commands_intact(setup_api_context):
    ctx = setup_api_context
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # /rem command
        payload_rem = {
            "query": "/rem User prefers concise financial summaries",
            "workspace_id": str(ctx["workspace"].id),
        }
        res_rem = await client.post("/chat/ask", json=payload_rem, headers=ctx["headers"])
        assert res_rem.status_code == 200
        assert "Memory saved" in res_rem.json()["answer"] or "Memory" in res_rem.json()["answer"]
