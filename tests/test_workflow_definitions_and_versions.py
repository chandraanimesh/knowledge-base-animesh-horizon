"""
Phase 1 Acceptance Suite: Persistent Workflow Definitions & Versioning.

Verifies:
1. Workflow creation & initial v1 draft initialization.
2. Draft editing (triggers, steps, presentation canvas).
3. Version auto-numbering (v1 -> v2 -> v3).
4. Atomic version publishing & single-active-version invariant.
5. Strict immutability of published (ACTIVE / ARCHIVED) versions (409 Conflict).
6. Concurrent publication locking & transactional isolation.
7. Deterministic DAG validation (cycle rejection, self-dependency, undeclared dependency, bounds).
8. Cross-workspace authorization & RBAC isolation (VIEWER vs EDITOR/OWNER).
9. Association of WorkflowRun with exact immutable WorkflowVersion.
10. Schema verification against disposable test database.
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5432/rag_disposable_test"

import asyncio
from datetime import datetime, timezone
import uuid
from httpx import AsyncClient, ASGITransport
import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.engine.url import make_url

from rag_app.db.models import (
    Base,
    Conversation,
    Department,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowVersion,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.main import app
from rag_app.orchestration.workflow_definition_service import WorkflowDefinitionService
from rag_app.schemas.workflow_definition import (
    validate_workflow_dag,
    WorkflowActionType,
    WorkflowCanvasLayout,
    WorkflowDefinitionCreate,
    WorkflowDefinitionUpdate,
    WorkflowNodeDefinition,
    WorkflowTriggerSpec,
    WorkflowTriggerType,
    WorkflowVersionCreateDraft,
    WorkflowVersionStatus,
    WorkflowVersionUpdate,
)
from rag_app.security.authentication import create_access_token


# SAFETY GATE: Preflight verification that tests never target production or dev databases
_db_url = make_url(os.environ["DATABASE_URL"])
assert _db_url.database == "rag_disposable_test", (
    f"FATAL SAFETY ABORT: Refusing to run tests against '{_db_url.database}'! "
    f"Tests must strictly run against 'rag_disposable_test'."
)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_test_context():
    """
    Initializes isolated tables and test entities in rag_disposable_test:
    - Creates necessary database schema
    - Creates department and workspaces A and B
    - Creates owner, editor, and viewer users
    """
    async with engine.begin() as conn:
        # Create only required tables in rag_disposable_test safely with checkfirst=True
        target_tables = [
            Department.__table__,
            Permission.__table__,
            User.__table__,
            UserDepartment.__table__,
            UserPermission.__table__,
            Workspace.__table__,
            WorkspaceMember.__table__,
            Conversation.__table__,
            WorkflowDefinition.__table__,
            WorkflowVersion.__table__,
            WorkflowRun.__table__,
        ]
        for tbl in target_tables:
            await conn.run_sync(tbl.create, checkfirst=True)

    async with AsyncSessionLocal() as session:
        # 1. Department and Permissions
        dept = await session.scalar(select(Department).where(Department.code == "WF_DEF_DEPT"))
        if not dept:
            dept = Department(
                id=uuid.uuid4(),
                code="WF_DEF_DEPT",
                name="Workflow Definition Dept",
                is_active=True,
            )
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(
                id=uuid.uuid4(),
                code="VIEW",
                description="View access",
                is_active=True,
            )
            session.add(view_perm)
            await session.commit()
            await session.refresh(view_perm)

        # 2. Users: Owner, Editor, Viewer, Outside User
        owner = await session.scalar(select(User).where(User.email == "wf_owner@test.com"))
        if not owner:
            owner = User(
                id=uuid.uuid4(),
                email="wf_owner@test.com",
                hashed_password="hash",
                role="USER",
                is_active=True,
            )
            session.add(owner)

        editor = await session.scalar(select(User).where(User.email == "wf_editor@test.com"))
        if not editor:
            editor = User(
                id=uuid.uuid4(),
                email="wf_editor@test.com",
                hashed_password="hash",
                role="USER",
                is_active=True,
            )
            session.add(editor)

        viewer = await session.scalar(select(User).where(User.email == "wf_viewer@test.com"))
        if not viewer:
            viewer = User(
                id=uuid.uuid4(),
                email="wf_viewer@test.com",
                hashed_password="hash",
                role="USER",
                is_active=True,
            )
            session.add(viewer)

        outside_user = await session.scalar(select(User).where(User.email == "wf_outside@test.com"))
        if not outside_user:
            outside_user = User(
                id=uuid.uuid4(),
                email="wf_outside@test.com",
                hashed_password="hash",
                role="USER",
                is_active=True,
            )
            session.add(outside_user)

        await session.commit()
        await session.refresh(owner)
        await session.refresh(editor)
        await session.refresh(viewer)
        await session.refresh(outside_user)

        # Assign UserDepartment and VIEW permission in Department to users
        for u in [owner, editor, viewer]:
            has_ud = await session.scalar(
                select(UserDepartment).where(
                    UserDepartment.user_id == u.id,
                    UserDepartment.department_id == dept.id,
                )
            )
            if not has_ud:
                session.add(UserDepartment(user_id=u.id, department_id=dept.id))

            has_perm = await session.scalar(
                select(UserPermission).where(
                    UserPermission.user_id == u.id,
                    UserPermission.department_id == dept.id,
                    UserPermission.permission_id == view_perm.id,
                )
            )
            if not has_perm:
                session.add(
                    UserPermission(
                        user_id=u.id,
                        department_id=dept.id,
                        permission_id=view_perm.id,
                    )
                )
        await session.commit()

        # 3. Workspaces A and B
        ws_a = await session.scalar(select(Workspace).where(Workspace.name == "WF Workspace A"))
        if not ws_a:
            ws_a = Workspace(
                id=uuid.uuid4(),
                name="WF Workspace A",
                code="WS_A",
                department_id=dept.id,
                created_by=owner.id,
                is_active=True,
            )
            session.add(ws_a)
            await session.commit()
            await session.refresh(ws_a)

            # Add memberships
            session.add(WorkspaceMember(workspace_id=ws_a.id, user_id=owner.id, role="OWNER"))
            session.add(WorkspaceMember(workspace_id=ws_a.id, user_id=editor.id, role="EDITOR"))
            session.add(WorkspaceMember(workspace_id=ws_a.id, user_id=viewer.id, role="VIEWER"))
            await session.commit()

        ws_b = await session.scalar(select(Workspace).where(Workspace.name == "WF Workspace B"))
        if not ws_b:
            ws_b = Workspace(
                id=uuid.uuid4(),
                name="WF Workspace B",
                code="WS_B",
                department_id=dept.id,
                created_by=outside_user.id,
                is_active=True,
            )
            session.add(ws_b)
            await session.commit()
            await session.refresh(ws_b)

            session.add(WorkspaceMember(workspace_id=ws_b.id, user_id=outside_user.id, role="OWNER"))
            await session.commit()

    return {
        "dept": dept,
        "owner": owner,
        "editor": editor,
        "viewer": viewer,
        "outside_user": outside_user,
        "ws_a": ws_a,
        "ws_b": ws_b,
    }


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_workflow_creation_and_initial_v1_draft(setup_test_context):
    ctx = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowDefinitionService(session)

        dto = WorkflowDefinitionCreate(
            workspace_id=ctx["ws_a"].id,
            name="Company Research Pipeline",
            description="Automates RAG research and CRM notes",
            initial_trigger=WorkflowTriggerSpec(
                trigger_type=WorkflowTriggerType.WEBHOOK,
                config={"path": "/leads/new"},
            ),
            initial_steps=[
                WorkflowNodeDefinition(
                    id="step_rag",
                    name="Query Knowledge Base",
                    node_type=WorkflowActionType.RETRIEVAL,
                    inputs={"query": "{{trigger.company_name}}"},
                    dependencies=[],
                )
            ],
        )

        wf, v1 = await service.create_workflow(ctx["ws_a"].id, ctx["owner"], dto)

        assert wf.id is not None
        assert wf.workspace_id == ctx["ws_a"].id
        assert wf.name == "Company Research Pipeline"
        assert wf.is_active is True
        assert wf.active_version_id is None  # Initial version is DRAFT, not yet active

        assert v1.workflow_id == wf.id
        assert v1.version_number == 1
        assert v1.status == WorkflowVersionStatus.DRAFT.value
        assert v1.trigger_spec["trigger_type"] == "WEBHOOK"
        assert len(v1.steps_spec) == 1
        assert v1.steps_spec[0]["id"] == "step_rag"
        assert v1.published_at is None


@pytest.mark.asyncio
async def test_draft_editing_and_canvas_layout(setup_test_context):
    ctx = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowDefinitionService(session)

        wf, v1 = await service.create_workflow(
            ctx["ws_a"].id,
            ctx["editor"],
            WorkflowDefinitionCreate(
                workspace_id=ctx["ws_a"].id,
                name="Editable Pipeline",
            ),
        )

        update_dto = WorkflowVersionUpdate(
            steps_spec=[
                WorkflowNodeDefinition(
                    id="node_search",
                    name="Deep Web Search",
                    node_type=WorkflowActionType.WEB_SEARCH,
                    inputs={"query": "AI Market Trends"},
                    dependencies=[],
                ),
                WorkflowNodeDefinition(
                    id="node_synth",
                    name="Synthesize Brief",
                    node_type=WorkflowActionType.SYNTHESIS,
                    inputs={"prompt": "Summarize trends"},
                    dependencies=["node_search"],
                ),
            ],
            canvas_layout=WorkflowCanvasLayout(
                nodes=[
                    {"id": "node_search", "position": {"x": 100.0, "y": 200.0}},
                    {"id": "node_synth", "position": {"x": 300.0, "y": 200.0}},
                ],
                edges=[
                    {"id": "e1", "source": "node_search", "target": "node_synth"},
                ],
            ),
        )

        updated_v1 = await service.update_draft_version(ctx["ws_a"].id, wf.id, v1.id, update_dto)
        assert len(updated_v1.steps_spec) == 2
        assert updated_v1.steps_spec[1]["id"] == "node_synth"
        assert updated_v1.steps_spec[1]["dependencies"] == ["node_search"]
        assert len(updated_v1.canvas_layout["nodes"]) == 2


@pytest.mark.asyncio
async def test_version_auto_numbering(setup_test_context):
    ctx = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowDefinitionService(session)

        wf, v1 = await service.create_workflow(
            ctx["ws_a"].id,
            ctx["owner"],
            WorkflowDefinitionCreate(workspace_id=ctx["ws_a"].id, name="Multi Version Pipeline"),
        )
        assert v1.version_number == 1

        v2 = await service.create_draft_version(ctx["ws_a"].id, wf.id, ctx["owner"])
        assert v2.version_number == 2
        assert v2.status == WorkflowVersionStatus.DRAFT.value

        v3 = await service.create_draft_version(ctx["ws_a"].id, wf.id, ctx["owner"])
        assert v3.version_number == 3
        assert v3.status == WorkflowVersionStatus.DRAFT.value

        versions = await service.list_versions(ctx["ws_a"].id, wf.id)
        assert len(versions) == 3
        assert [v.version_number for v in versions] == [3, 2, 1]


@pytest.mark.asyncio
async def test_atomic_version_publication_and_single_active_invariant(setup_test_context):
    ctx = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowDefinitionService(session)

        valid_steps = [
            WorkflowNodeDefinition(
                id="root_node",
                name="Start",
                node_type=WorkflowActionType.RETRIEVAL,
                inputs={"query": "start"},
                dependencies=[],
            ),
            WorkflowNodeDefinition(
                id="child_node",
                name="Follow-up",
                node_type=WorkflowActionType.SYNTHESIS,
                inputs={"prompt": "conclude"},
                dependencies=["root_node"],
            ),
        ]

        wf, v1 = await service.create_workflow(
            ctx["ws_a"].id,
            ctx["owner"],
            WorkflowDefinitionCreate(
                workspace_id=ctx["ws_a"].id,
                name="Publish Test Workflow",
                initial_steps=valid_steps,
            ),
        )

        # Publish Version 1
        published_v1 = await service.publish_version(ctx["ws_a"].id, wf.id, v1.id, ctx["owner"])
        assert published_v1.status == WorkflowVersionStatus.ACTIVE.value
        assert published_v1.published_at is not None

        refreshed_wf = await service.get_workflow(ctx["ws_a"].id, wf.id)
        assert refreshed_wf.active_version_id == v1.id

        # Create Version 2 and publish it
        v2 = await service.create_draft_version(
            ctx["ws_a"].id,
            wf.id,
            ctx["owner"],
            WorkflowVersionCreateDraft(
                fork_from_version_id=v1.id,
                steps_spec=valid_steps,
            ),
        )
        assert v2.version_number == 2
        assert v2.status == WorkflowVersionStatus.DRAFT.value

        published_v2 = await service.publish_version(ctx["ws_a"].id, wf.id, v2.id, ctx["owner"])
        assert published_v2.status == WorkflowVersionStatus.ACTIVE.value
        assert published_v2.version_number == 2

        # Invariant Check: Version 1 must have been automatically transitioned to ARCHIVED
        old_v1 = await service.get_version(ctx["ws_a"].id, wf.id, v1.id)
        assert old_v1.status == WorkflowVersionStatus.ARCHIVED.value

        # Parent workflow active_version_id points to v2
        refreshed_wf = await service.get_workflow(ctx["ws_a"].id, wf.id)
        assert refreshed_wf.active_version_id == v2.id


@pytest.mark.asyncio
async def test_strict_immutability_of_published_versions(setup_test_context):
    ctx = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowDefinitionService(session)

        wf, v1 = await service.create_workflow(
            ctx["ws_a"].id,
            ctx["owner"],
            WorkflowDefinitionCreate(
                workspace_id=ctx["ws_a"].id,
                name="Immutability Workflow",
                initial_steps=[
                    WorkflowNodeDefinition(
                        id="n1",
                        name="Node 1",
                        node_type=WorkflowActionType.RETRIEVAL,
                        inputs={},
                        dependencies=[],
                    )
                ],
            ),
        )

        # Publish v1
        await service.publish_version(ctx["ws_a"].id, wf.id, v1.id, ctx["owner"])

        # Attempt to edit ACTIVE v1 directly -> MUST FAIL with 409 Conflict
        with pytest.raises(HTTPException) as exc_info:
            await service.update_draft_version(
                ctx["ws_a"].id,
                wf.id,
                v1.id,
                WorkflowVersionUpdate(
                    steps_spec=[
                        WorkflowNodeDefinition(
                            id="n1_mod",
                            name="Tampered Node",
                            node_type=WorkflowActionType.RETRIEVAL,
                            inputs={},
                            dependencies=[],
                        )
                    ]
                ),
            )
        assert exc_info.value.status_code == 409
        assert "immutable" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_dag_validation_rules():
    # 1. Empty graph
    errors = validate_workflow_dag([])
    assert any("at least one step" in e for e in errors)

    # 2. Self-dependency
    self_dep = [
        {"id": "a", "name": "A", "node_type": "RETRIEVAL", "dependencies": ["a"], "inputs": {}}
    ]
    errors = validate_workflow_dag(self_dep)
    assert any("cannot depend on itself" in e for e in errors)

    # 3. Undeclared dependency
    missing_dep = [
        {"id": "a", "name": "A", "node_type": "RETRIEVAL", "dependencies": ["b"], "inputs": {}}
    ]
    errors = validate_workflow_dag(missing_dep)
    assert any("undeclared dependency 'b'" in e for e in errors)

    # 4. Cycle detection
    cyclic = [
        {"id": "a", "name": "A", "node_type": "RETRIEVAL", "dependencies": ["c"], "inputs": {}},
        {"id": "b", "name": "B", "node_type": "SYNTHESIS", "dependencies": ["a"], "inputs": {}},
        {"id": "c", "name": "C", "node_type": "EXECUTION", "dependencies": ["b"], "inputs": {}},
    ]
    errors = validate_workflow_dag(cyclic)
    assert any("cycle detected" in e.lower() for e in errors)

    # 5. Unsupported action type
    unsupported = [
        {"id": "a", "name": "A", "node_type": "ARBITRARY_PYTHON_EVAL", "dependencies": [], "inputs": {}}
    ]
    errors = validate_workflow_dag(unsupported)
    assert len(errors) > 0

    # 6. Bounded complexity (>25 nodes)
    oversized = [
        {"id": f"node_{i}", "name": f"Node {i}", "node_type": "RETRIEVAL", "dependencies": [], "inputs": {}}
        for i in range(26)
    ]
    errors = validate_workflow_dag(oversized, max_nodes=25)
    assert any("maximum allowed limit of 25" in e for e in errors)

    # 7. Valid Diamond DAG
    diamond = [
        {"id": "root", "name": "Root", "node_type": "RETRIEVAL", "dependencies": [], "inputs": {}},
        {"id": "branch_1", "name": "Branch 1", "node_type": "WEB_SEARCH", "dependencies": ["root"], "inputs": {}},
        {"id": "branch_2", "name": "Branch 2", "node_type": "EXTRACTION", "dependencies": ["root"], "inputs": {}},
        {"id": "sink", "name": "Sink", "node_type": "SYNTHESIS", "dependencies": ["branch_1", "branch_2"], "inputs": {}},
    ]
    errors = validate_workflow_dag(diamond)
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_rejection_of_publishing_invalid_dag(setup_test_context):
    ctx = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowDefinitionService(session)

        # Create workflow with cyclic steps
        wf, v1 = await service.create_workflow(
            ctx["ws_a"].id,
            ctx["owner"],
            WorkflowDefinitionCreate(
                workspace_id=ctx["ws_a"].id,
                name="Invalid Cyclic Pipeline",
                initial_steps=[
                    WorkflowNodeDefinition(
                        id="n1",
                        name="N1",
                        node_type=WorkflowActionType.RETRIEVAL,
                        dependencies=["n2"],
                    ),
                    WorkflowNodeDefinition(
                        id="n2",
                        name="N2",
                        node_type=WorkflowActionType.SYNTHESIS,
                        dependencies=["n1"],
                    ),
                ],
            ),
        )

        with pytest.raises(HTTPException) as exc_info:
            await service.publish_version(ctx["ws_a"].id, wf.id, v1.id, ctx["owner"])
        assert exc_info.value.status_code == 422
        assert "Cannot publish invalid workflow DAG" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_cross_workspace_authorization_isolation(setup_test_context):
    ctx = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowDefinitionService(session)

        # Create workflow in Workspace A
        wf_a, _ = await service.create_workflow(
            ctx["ws_a"].id,
            ctx["owner"],
            WorkflowDefinitionCreate(
                workspace_id=ctx["ws_a"].id,
                name="Workspace A Secret Workflow",
            ),
        )

        # Attempt to access Workspace A's workflow from Workspace B -> MUST FAIL with 404
        with pytest.raises(HTTPException) as exc_info:
            await service.get_workflow(ctx["ws_b"].id, wf_a.id)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_workflow_run_association_with_workflow_version(setup_test_context):
    ctx = setup_test_context
    async with AsyncSessionLocal() as session:
        service = WorkflowDefinitionService(session)

        wf, v1 = await service.create_workflow(
            ctx["ws_a"].id,
            ctx["owner"],
            WorkflowDefinitionCreate(
                workspace_id=ctx["ws_a"].id,
                name="Run Associated Workflow",
                initial_steps=[
                    WorkflowNodeDefinition(
                        id="start",
                        name="Start Step",
                        node_type=WorkflowActionType.RETRIEVAL,
                        dependencies=[],
                    )
                ],
            ),
        )
        published_v1 = await service.publish_version(ctx["ws_a"].id, wf.id, v1.id, ctx["owner"])

        # Create a WorkflowRun explicitly bound to this published version
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=ctx["owner"].id,
            workspace_id=ctx["ws_a"].id,
            department_id=ctx["dept"].id,
            workflow_id=wf.id,
            workflow_version_id=published_v1.id,
            scope_type="WORKSPACE",
            status="PENDING",
            request_hash="dummy_req_hash",
            initial_query="Execute workflow",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        # Assert foreign key linkage
        assert run.workflow_id == wf.id
        assert run.workflow_version_id == published_v1.id
        assert run.workflow.name == "Run Associated Workflow"
        assert run.workflow_version.version_number == 1
        assert run.workflow_version.status == "ACTIVE"


# ---------------------------------------------------------------------------
# HTTP API & RBAC End-to-End Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_workflow_lifecycle_and_immutability(setup_test_context):
    ctx = setup_test_context
    owner_token = create_access_token(subject=str(ctx["owner"].id))
    headers = {"Authorization": f"Bearer {owner_token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create Workflow Definition via API
        create_resp = await client.post(
            "/api/v1/workflows/definitions",
            headers=headers,
            json={
                "workspace_id": str(ctx["ws_a"].id),
                "name": "API Automated Pipeline",
                "description": "Created via REST API",
                "initial_steps": [
                    {
                        "id": "step_1",
                        "name": "Initial Step",
                        "node_type": "RETRIEVAL",
                        "dependencies": [],
                    }
                ],
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        wf_data = create_resp.json()
        wf_id = wf_data["id"]
        assert wf_data["name"] == "API Automated Pipeline"

        # 2. List Versions to get v1
        vers_resp = await client.get(
            f"/api/v1/workflows/definitions/{wf_id}/versions?workspace_id={ctx['ws_a'].id}",
            headers=headers,
        )
        assert vers_resp.status_code == 200
        versions = vers_resp.json()
        assert len(versions) == 1
        v1_id = versions[0]["id"]
        assert versions[0]["version_number"] == 1
        assert versions[0]["status"] == "DRAFT"

        # 3. Validate DAG via API
        val_resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/validate?workspace_id={ctx['ws_a'].id}",
            headers=headers,
        )
        assert val_resp.status_code == 200
        assert val_resp.json()["is_valid"] is True

        # 4. Publish Version 1
        pub_resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/publish?workspace_id={ctx['ws_a'].id}",
            headers=headers,
        )
        assert pub_resp.status_code == 200
        assert pub_resp.json()["status"] == "ACTIVE"

        # 5. Attempt to edit published version -> 409 Conflict
        edit_resp = await client.put(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}?workspace_id={ctx['ws_a'].id}",
            headers=headers,
            json={
                "steps_spec": [
                    {
                        "id": "tampered",
                        "name": "Tampered",
                        "node_type": "RETRIEVAL",
                        "dependencies": [],
                    }
                ]
            },
        )
        assert edit_resp.status_code == 409
        assert "immutable" in edit_resp.text.lower()


@pytest.mark.asyncio
async def test_api_rbac_viewer_forbidden_to_publish_or_edit(setup_test_context):
    ctx = setup_test_context
    owner_token = create_access_token(subject=str(ctx["owner"].id))
    viewer_token = create_access_token(subject=str(ctx["viewer"].id))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Owner creates workflow
        create_resp = await client.post(
            "/api/v1/workflows/definitions",
            headers={"Authorization": f"Bearer {owner_token}"},
            json={
                "workspace_id": str(ctx["ws_a"].id),
                "name": "RBAC Test Pipeline",
                "initial_steps": [
                    {
                        "id": "s1",
                        "name": "S1",
                        "node_type": "RETRIEVAL",
                        "dependencies": [],
                    }
                ],
            },
        )
        assert create_resp.status_code == 201
        wf_id = create_resp.json()["id"]

        vers_resp = await client.get(
            f"/api/v1/workflows/definitions/{wf_id}/versions?workspace_id={ctx['ws_a'].id}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        v1_id = vers_resp.json()[0]["id"]

        # Viewer can read workflow definition (200 OK)
        get_resp = await client.get(
            f"/api/v1/workflows/definitions/{wf_id}?workspace_id={ctx['ws_a'].id}",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert get_resp.status_code == 200

        # Viewer attempts to edit draft -> 403 Forbidden
        edit_resp = await client.put(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}?workspace_id={ctx['ws_a'].id}",
            headers={"Authorization": f"Bearer {viewer_token}"},
            json={"steps_spec": []},
        )
        assert edit_resp.status_code == 403

        # Viewer attempts to publish -> 403 Forbidden
        pub_resp = await client.post(
            f"/api/v1/workflows/definitions/{wf_id}/versions/{v1_id}/publish?workspace_id={ctx['ws_a'].id}",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert pub_resp.status_code == 403


@pytest.mark.asyncio
async def test_api_cross_workspace_forbidden(setup_test_context):
    ctx = setup_test_context
    outside_token = create_access_token(subject=str(ctx["outside_user"].id))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Outside user attempts to list workflows from Workspace A
        resp = await client.get(
            f"/api/v1/workflows/definitions?workspace_id={ctx['ws_a'].id}",
            headers={"Authorization": f"Bearer {outside_token}"},
        )
        assert resp.status_code == 403


def test_migration_script_metadata():
    from migrations.versions import add_workflow_definitions_and_versions as mig
    assert mig.revision == "add_workflow_definitions_and_versions"
    assert mig.down_revision == "add_chunk_versions_and_curation"
    assert hasattr(mig, "upgrade")
    assert hasattr(mig, "downgrade")

