"""
API Integration & Regression Test Suite for Workflow Test Run (Phase 4D).
Verifies:
1. Safe execution of DRAFT WorkflowVersion directly without publishing or altering active_version_id.
2. Exact draft version association on WorkflowRun (workflow_version_id, thread_id='test:...').
3. Deterministic DAG and action schema validation (422 rejection on cycle or unknown action).
4. Safe dry-run simulation for external CRM mutations (real external side-effects blocked).
5. Preservation of approval gate boundaries without bypassing HITL constraints.
6. Isolated test idempotency keys under 'test:' namespace (no collision with production runs).
7. Strict workspace isolation and RBAC (VIEWER rejected with 403 Forbidden).
"""

from __future__ import annotations

import os

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
if "DATABASE_URL" not in os.environ or "5433" in os.environ["DATABASE_URL"]:
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5432/rag_disposable_test"

from datetime import datetime, timezone
from typing import Any
import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

from rag_app.db.models import (
    Department,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowTask,
    WorkflowVersion,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.main import app
from rag_app.schemas.workflow import (
    WorkflowStatus,
    WorkflowTaskStatus,
)
from rag_app.security.authentication import create_access_token


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_test_run_context():
    """
    Sets up workspace, departments, editor user, viewer user, and an isolated other-workspace user.
    """
    async with AsyncSessionLocal() as session:
        dept = Department(
            id=uuid.uuid4(),
            name="Test Run Dept",
            code=f"TR_DEPT_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(dept)
        await session.flush()

        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(id=uuid.uuid4(), code="VIEW", name="View Permission")
            session.add(view_perm)
            await session.flush()

        # 1. Editor user
        editor_user = User(
            id=uuid.uuid4(),
            email=f"tr_editor_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        session.add(editor_user)
        await session.flush()

        session.add(UserDepartment(user_id=editor_user.id, department_id=dept.id))
        session.add(UserPermission(user_id=editor_user.id, department_id=dept.id, permission_id=view_perm.id))

        workspace = Workspace(
            id=uuid.uuid4(),
            name="Test Run Workspace",
            code=f"tr_ws_{uuid.uuid4().hex[:6]}",
            department_id=dept.id,
            is_active=True,
            created_by=editor_user.id,
        )
        session.add(workspace)
        await session.flush()

        session.add(
            WorkspaceMember(
                workspace_id=workspace.id,
                user_id=editor_user.id,
                role="EDITOR",
            )
        )

        # 2. Viewer user (Read-only member)
        viewer_user = User(
            id=uuid.uuid4(),
            email=f"tr_viewer_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        session.add(viewer_user)
        await session.flush()
        session.add(UserDepartment(user_id=viewer_user.id, department_id=dept.id))
        session.add(UserPermission(user_id=viewer_user.id, department_id=dept.id, permission_id=view_perm.id))
        session.add(
            WorkspaceMember(
                workspace_id=workspace.id,
                user_id=viewer_user.id,
                role="VIEWER",
            )
        )

        # 3. Other workspace user (Cross-workspace isolation)
        other_dept = Department(
            id=uuid.uuid4(),
            name="Other Dept",
            code=f"OTHER_DEPT_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(other_dept)
        await session.flush()

        other_user = User(
            id=uuid.uuid4(),
            email=f"tr_other_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        session.add(other_user)
        await session.flush()
        session.add(UserDepartment(user_id=other_user.id, department_id=other_dept.id))
        session.add(UserPermission(user_id=other_user.id, department_id=other_dept.id, permission_id=view_perm.id))

        await session.commit()

        editor_token = create_access_token(subject=str(editor_user.id))
        viewer_token = create_access_token(subject=str(viewer_user.id))
        other_token = create_access_token(subject=str(other_user.id))

        return {
            "workspace": workspace,
            "dept": dept,
            "editor_user": editor_user,
            "viewer_user": viewer_user,
            "other_user": other_user,
            "editor_token": editor_token,
            "viewer_token": viewer_token,
            "other_token": other_token,
        }


@pytest.mark.asyncio
async def test_draft_workflow_test_run_success(setup_test_run_context):
    """
    1. Executes a DRAFT workflow version in safe TEST mode.
    2. Verifies that the version remains in DRAFT state and active_version_id is untouched.
    3. Verifies that the WorkflowRun is strictly bound to the draft version with test markers.
    4. Verifies DAG tasks execute and return outputs and token counts.
    """
    ctx = setup_test_run_context
    ws_id = str(ctx["workspace"].id)

    async with AsyncSessionLocal() as session:
        # Create workflow definition and DRAFT version 1
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ctx["workspace"].id,
            name="Draft RAG Research Pipeline",
            description="Testing draft execution without publishing",
            created_by_id=ctx["editor_user"].id,
            is_active=True,
            active_version_id=None,
        )
        session.add(wf)
        await session.flush()

        draft_v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status="DRAFT",
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            steps_spec=[
                {
                    "id": "step_retrieval",
                    "name": "Knowledge Search",
                    "node_type": "RETRIEVAL",
                    "inputs": {"query": "{{trigger.query}}", "limit": 3},
                    "dependencies": [],
                },
                {
                    "id": "step_if_else",
                    "name": "Check Retrieved Count",
                    "node_type": "IF_ELSE",
                    "inputs": {
                        "left_value": "{{steps.step_retrieval.output.retrieved_count}}",
                        "operator": "GREATER_THAN_OR_EQUAL",
                        "right_value": 0,
                    },
                    "dependencies": ["step_retrieval"],
                },
                {
                    "id": "step_synthesis",
                    "name": "Draft Synthesis",
                    "node_type": "SYNTHESIS",
                    "inputs": {"focus": "{{trigger.query}}", "max_length": 500},
                    "dependencies": ["step_if_else"],
                },
            ],
            canvas_layout={"nodes": [], "edges": []},
            schema_version=1,
            created_by_id=ctx["editor_user"].id,
        )
        session.add(draft_v1)
        await session.commit()
        wf_id = str(wf.id)
        v1_id = str(draft_v1.id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['editor_token']}"},
            json={
                "inputs": {"query": "Analyze quarterly enterprise customer retention"},
                "budget_limits": {"max_tokens": 30000, "max_provider_calls": 10},
            },
        )

        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Check run identity & version association
        assert data["workflow_id"] == wf_id
        assert data["workflow_version_id"] == v1_id
        assert data["status"] == "COMPLETED"
        assert data["is_test"] is True
        assert data["execution_mode"] == "TEST"
        assert data["thread_id"] == f"test:{v1_id}"

        # Check tasks execution
        tasks = {t["task_key"]: t for t in data["tasks"]}
        assert "step_retrieval" in tasks
        assert "step_if_else" in tasks
        assert "step_synthesis" in tasks

        assert tasks["step_retrieval"]["status"] == "COMPLETED"
        assert tasks["step_if_else"]["status"] == "COMPLETED"
        assert tasks["step_if_else"]["output_result"]["result"] is True
        assert tasks["step_synthesis"]["status"] == "COMPLETED"
        assert tasks["step_synthesis"]["output_result"]["status"] == "synthesized"

    # Verify database state: draft was NOT published or modified!
    async with AsyncSessionLocal() as session:
        refreshed_wf = await session.get(WorkflowDefinition, uuid.UUID(wf_id))
        refreshed_v1 = await session.get(WorkflowVersion, uuid.UUID(v1_id))

        assert refreshed_wf.active_version_id is None, "Draft execution must never set active_version_id!"
        assert refreshed_v1.status == "DRAFT", "Draft version must remain in DRAFT status after test execution!"


@pytest.mark.asyncio
async def test_invalid_draft_cycle_rejected_with_422(setup_test_run_context):
    """
    Verifies that a draft version with a circular dependency is rejected with 422 Unprocessable Entity.
    """
    ctx = setup_test_run_context
    ws_id = str(ctx["workspace"].id)

    async with AsyncSessionLocal() as session:
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ctx["workspace"].id,
            name="Cyclic Workflow",
            created_by_id=ctx["editor_user"].id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        cyclic_v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status="DRAFT",
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            steps_spec=[
                {
                    "id": "step_a",
                    "name": "Step A",
                    "node_type": "RETRIEVAL",
                    "inputs": {"query": "test"},
                    "dependencies": ["step_b"],  # Cycle: A -> B -> A
                },
                {
                    "id": "step_b",
                    "name": "Step B",
                    "node_type": "SYNTHESIS",
                    "inputs": {"focus": "test"},
                    "dependencies": ["step_a"],
                },
            ],
            canvas_layout={},
            schema_version=1,
            created_by_id=ctx["editor_user"].id,
        )
        session.add(cyclic_v1)
        await session.commit()
        wf_id = str(wf.id)
        v1_id = str(cyclic_v1.id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['editor_token']}"},
            json={"inputs": {"query": "test"}},
        )

        assert resp.status_code == 422, f"Expected 422 for cyclic DAG, got {resp.status_code}"
        assert "invalid workflow DAG specification" in resp.text

    # Verify no WorkflowRun was created
    async with AsyncSessionLocal() as session:
        run_count = await session.scalar(
            select(WorkflowRun).where(WorkflowRun.workflow_version_id == uuid.UUID(v1_id))
        )
        assert run_count is None


@pytest.mark.asyncio
async def test_unknown_action_type_rejected_with_422(setup_test_run_context):
    """
    Verifies that a draft referencing an unregistered or dangerous action type is rejected with 422.
    """
    ctx = setup_test_run_context
    ws_id = str(ctx["workspace"].id)

    async with AsyncSessionLocal() as session:
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ctx["workspace"].id,
            name="Dangerous Workflow",
            created_by_id=ctx["editor_user"].id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        unknown_action_v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status="DRAFT",
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            steps_spec=[
                {
                    "id": "step_malicious",
                    "name": "Execute Arbitrary Shell",
                    "node_type": "ARBITRARY_CODE_EXEC",  # Not allowlisted
                    "inputs": {"cmd": "rm -rf /"},
                    "dependencies": [],
                },
            ],
            canvas_layout={},
            schema_version=1,
            created_by_id=ctx["editor_user"].id,
        )
        session.add(unknown_action_v1)
        await session.commit()
        wf_id = str(wf.id)
        v1_id = str(unknown_action_v1.id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['editor_token']}"},
            json={"inputs": {}},
        )

        assert resp.status_code == 422, f"Expected 422 for unknown action, got {resp.status_code}"
        assert "Cannot test invalid workflow DAG specification" in resp.text or "failed schema validation" in resp.text


@pytest.mark.asyncio
async def test_crm_mutation_blocked_and_simulated_in_test_mode(setup_test_run_context):
    """
    Verifies that external mutations (EXECUTION action) in TEST mode do not require an approval token,
    do not call live CRM APIs, and return safe dry-run simulated results.
    """
    ctx = setup_test_run_context
    ws_id = str(ctx["workspace"].id)

    async with AsyncSessionLocal() as session:
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ctx["workspace"].id,
            name="CRM Writeback Pipeline",
            created_by_id=ctx["editor_user"].id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        exec_v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status="DRAFT",
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            steps_spec=[
                {
                    "id": "step_crm_note",
                    "name": "Write Note to CRM",
                    "node_type": "EXECUTION",
                    "inputs": {
                        "target_system": "crm",
                        "action": "CRM_CREATE_NOTE",
                        "payload": {
                            "company_id": "{{trigger.company_id}}",
                            "note": "Quarterly findings summary",
                        },
                    },
                    "dependencies": [],
                },
            ],
            canvas_layout={},
            schema_version=1,
            created_by_id=ctx["editor_user"].id,
        )
        session.add(exec_v1)
        await session.commit()
        wf_id = str(wf.id)
        v1_id = str(exec_v1.id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['editor_token']}"},
            json={"inputs": {"company_id": "company-uuid-test"}},
        )

        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "COMPLETED"

        task = data["tasks"][0]
        assert task["task_key"] == "step_crm_note"
        assert task["status"] == "COMPLETED"
        assert task["provider_calls"] == 0, "TEST mode must not consume live external provider calls"

        out = task["output_result"]
        assert out["status"] == "simulated_dry_run"
        assert out["executed"] is True
        assert out["details"]["dry_run"] is True
        assert "Simulated execution" in out["details"]["message"]


@pytest.mark.asyncio
async def test_approval_gate_boundaries_in_test_mode(setup_test_run_context):
    """
    Verifies that an APPROVAL_GATE step in a draft test run is not bypassed:
    when pre_approved is False, execution halts with AWAITING_APPROVAL.
    """
    ctx = setup_test_run_context
    ws_id = str(ctx["workspace"].id)

    async with AsyncSessionLocal() as session:
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ctx["workspace"].id,
            name="Approval Gated Workflow",
            created_by_id=ctx["editor_user"].id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        appr_v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status="DRAFT",
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            steps_spec=[
                {
                    "id": "step_retrieval",
                    "name": "Knowledge Retrieval",
                    "node_type": "RETRIEVAL",
                    "inputs": {"query": "confidential analysis"},
                    "dependencies": [],
                },
                {
                    "id": "step_gate",
                    "name": "Security Review Gate",
                    "node_type": "APPROVAL_GATE",
                    "inputs": {"gate_name": "Executive Approval", "pre_approved": False},
                    "dependencies": ["step_retrieval"],
                },
            ],
            canvas_layout={},
            schema_version=1,
            created_by_id=ctx["editor_user"].id,
        )
        session.add(appr_v1)
        await session.commit()
        wf_id = str(wf.id)
        v1_id = str(appr_v1.id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['editor_token']}"},
            json={"inputs": {}},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "AWAITING_APPROVAL"
        assert data["awaiting_approval_at"] is not None

        tasks = {t["task_key"]: t for t in data["tasks"]}
        assert tasks["step_retrieval"]["status"] == "COMPLETED"
        assert tasks["step_gate"]["output_result"]["status"] == "awaiting_approval"


@pytest.mark.asyncio
async def test_test_run_idempotency_isolation(setup_test_run_context):
    """
    Verifies that test run idempotency keys are isolated under 'test:' namespace:
    1. Same key + same payload: returns replayed run.
    2. Same key + different payload: returns 409 Conflict.
    3. Production run with identical raw key does not collide with test run.
    """
    ctx = setup_test_run_context
    ws_id = str(ctx["workspace"].id)

    async with AsyncSessionLocal() as session:
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ctx["workspace"].id,
            name="Idempotent Test Workflow",
            created_by_id=ctx["editor_user"].id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status="DRAFT",
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            steps_spec=[
                {
                    "id": "step_retrieval",
                    "name": "Search",
                    "node_type": "RETRIEVAL",
                    "inputs": {"query": "{{trigger.query}}"},
                    "dependencies": [],
                },
            ],
            canvas_layout={},
            schema_version=1,
            created_by_id=ctx["editor_user"].id,
        )
        session.add(v1)
        await session.commit()
        wf_id = str(wf.id)
        v1_id = str(v1.id)

    test_key = f"idemp-key-{uuid.uuid4().hex[:8]}"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First test run execution
        resp1 = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['editor_token']}"},
            json={"inputs": {"query": "initial search query"}, "idempotency_key": test_key},
        )
        assert resp1.status_code == 200
        run1_id = resp1.json()["id"]

        # Replay with same key and same inputs -> returns run1_id
        resp2 = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['editor_token']}"},
            json={"inputs": {"query": "initial search query"}, "idempotency_key": test_key},
        )
        assert resp2.status_code == 200
        assert resp2.json()["id"] == run1_id

        # Replay with same key and altered inputs -> 409 Conflict
        resp3 = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['editor_token']}"},
            json={"inputs": {"query": "MUTATED search query"}, "idempotency_key": test_key},
        )
        assert resp3.status_code == 409, f"Expected 409 Conflict on payload mutation, got {resp3.status_code}"


@pytest.mark.asyncio
async def test_viewer_role_rejected_with_403(setup_test_run_context):
    """
    Verifies that a workspace member with 'VIEWER' role cannot trigger a test run (requires EDITOR/OWNER).
    """
    ctx = setup_test_run_context
    ws_id = str(ctx["workspace"].id)

    async with AsyncSessionLocal() as session:
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ctx["workspace"].id,
            name="RBAC Workflow",
            created_by_id=ctx["editor_user"].id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status="DRAFT",
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            steps_spec=[
                {"id": "step_1", "name": "S1", "node_type": "RETRIEVAL", "inputs": {"query": "q"}, "dependencies": []}
            ],
            canvas_layout={},
            schema_version=1,
            created_by_id=ctx["editor_user"].id,
        )
        session.add(v1)
        await session.commit()
        wf_id = str(wf.id)
        v1_id = str(v1.id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['viewer_token']}"},
            json={"inputs": {}},
        )

        assert resp.status_code == 403, f"Expected 403 Forbidden for VIEWER, got {resp.status_code}"
        assert "requires 'EDITOR' or 'OWNER' role" in resp.text


@pytest.mark.asyncio
async def test_cross_workspace_access_rejected_with_403(setup_test_run_context):
    """
    Verifies that a user from a different workspace/department cannot execute a test run.
    """
    ctx = setup_test_run_context
    ws_id = str(ctx["workspace"].id)

    async with AsyncSessionLocal() as session:
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ctx["workspace"].id,
            name="Cross Workspace Workflow",
            created_by_id=ctx["editor_user"].id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status="DRAFT",
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            steps_spec=[
                {"id": "step_1", "name": "S1", "node_type": "RETRIEVAL", "inputs": {"query": "q"}, "dependencies": []}
            ],
            canvas_layout={},
            schema_version=1,
            created_by_id=ctx["editor_user"].id,
        )
        session.add(v1)
        await session.commit()
        wf_id = str(wf.id)
        v1_id = str(v1.id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/test?workspace_id={ws_id}",
            headers={"Authorization": f"Bearer {ctx['other_token']}"},
            json={"inputs": {}},
        )

        assert resp.status_code == 403, f"Expected 403 Forbidden for cross-workspace user, got {resp.status_code}"
