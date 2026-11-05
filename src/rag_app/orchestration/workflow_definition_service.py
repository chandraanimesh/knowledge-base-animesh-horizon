"""
Workflow Definition & Versioning Service.
Coordinates lifecycle management, unique version numbering, atomic draft publishing with
PostgreSQL row-level locking, immutable published version enforcement, and DAG validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Optional
import uuid

from fastapi import HTTPException, status
from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import User, WorkflowDefinition, WorkflowRun, WorkflowTask, WorkflowVersion, Workspace
from rag_app.orchestration.action_registry import get_action_registry
from rag_app.orchestration.executor import DAGExecutionEngine
from rag_app.orchestration.workflow_plan_compiler import compile_workflow_version_to_plan
from rag_app.schemas.workflow import (
    BudgetLimits,
    compute_canonical_request_hash,
    ScopeType,
    WorkflowStatus,
    WorkflowTaskStatus,
)
from rag_app.schemas.workflow_definition import (
    validate_workflow_dag,
    WorkflowDAGValidationResponse,
    WorkflowDefinitionCreate,
    WorkflowDefinitionResponse,
    WorkflowDefinitionUpdate,
    WorkflowTestRunRequest,
    WorkflowVersionCreateDraft,
    WorkflowVersionResponse,
    WorkflowVersionStatus,
    WorkflowVersionUpdate,
)

logger = logging.getLogger(__name__)


class WorkflowDefinitionService:
    """
    Service encapsulating business invariants for Workflow Definitions and Workflow Versions.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_workflow(
        self,
        workspace_id: uuid.UUID,
        user: User,
        payload: WorkflowDefinitionCreate,
    ) -> tuple[WorkflowDefinition, WorkflowVersion]:
        """
        Creates a new WorkflowDefinition and its initial Version 1 in DRAFT state.
        """
        initial_trigger = payload.initial_trigger.model_dump() if payload.initial_trigger else {"trigger_type": "MANUAL", "config": {}}
        initial_steps = [s.model_dump() for s in payload.initial_steps] if payload.initial_steps else []
        initial_canvas = payload.initial_canvas_layout.model_dump() if payload.initial_canvas_layout else {"nodes": [], "edges": []}

        workflow = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            name=payload.name.strip(),
            description=payload.description.strip() if payload.description else None,
            created_by_id=user.id,
            is_active=True,
            active_version_id=None,
        )
        self.session.add(workflow)
        await self.session.flush()

        v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=workflow.id,
            version_number=1,
            status=WorkflowVersionStatus.DRAFT.value,
            trigger_spec=initial_trigger,
            steps_spec=initial_steps,
            canvas_layout=initial_canvas,
            schema_version=1,
            created_by_id=user.id,
        )
        self.session.add(v1)
        await self.session.commit()
        await self.session.refresh(workflow)
        await self.session.refresh(v1)

        return workflow, v1

    async def get_workflow(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
    ) -> WorkflowDefinition:
        """
        Fetches an active workflow definition within the specified workspace.
        """
        stmt = select(WorkflowDefinition).where(
            WorkflowDefinition.id == workflow_id,
            WorkflowDefinition.workspace_id == workspace_id,
            WorkflowDefinition.deleted_at.is_(None),
        )
        workflow = await self.session.scalar(stmt)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow definition '{workflow_id}' not found in workspace.",
            )
        return workflow

    async def list_workflows(
        self,
        workspace_id: uuid.UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> list[WorkflowDefinition]:
        """
        Lists active workflow definitions for a workspace.
        """
        stmt = (
            select(WorkflowDefinition)
            .where(
                WorkflowDefinition.workspace_id == workspace_id,
                WorkflowDefinition.deleted_at.is_(None),
            )
            .order_by(WorkflowDefinition.created_at.desc())
            .offset(skip)
            .limit(min(limit, 100))
        )
        return list((await self.session.scalars(stmt)).all())

    async def update_workflow(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
        payload: WorkflowDefinitionUpdate,
    ) -> WorkflowDefinition:
        """
        Updates metadata (name, description, is_active) of a workflow definition.
        """
        workflow = await self.get_workflow(workspace_id, workflow_id)

        if payload.name is not None:
            workflow.name = payload.name.strip()
        if payload.description is not None:
            workflow.description = payload.description.strip() if payload.description else None
        if payload.is_active is not None:
            workflow.is_active = payload.is_active

        await self.session.commit()
        await self.session.refresh(workflow)
        return workflow

    async def delete_workflow(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
    ) -> None:
        """
        Soft-deletes a workflow definition.
        """
        workflow = await self.get_workflow(workspace_id, workflow_id)
        workflow.deleted_at = datetime.now(timezone.utc)
        workflow.is_active = False
        await self.session.commit()

    async def create_draft_version(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
        user: User,
        payload: Optional[WorkflowVersionCreateDraft] = None,
    ) -> WorkflowVersion:
        """
        Creates a new draft version with auto-incremented version number.
        Can optionally fork trigger and steps from an existing version.
        """
        # Ensure workflow exists
        await self.get_workflow(workspace_id, workflow_id)

        # Compute next version number
        max_ver = await self.session.scalar(
            select(func.coalesce(func.max(WorkflowVersion.version_number), 0)).where(
                WorkflowVersion.workflow_id == workflow_id
            )
        )
        next_ver = (max_ver or 0) + 1

        trigger_spec = {"trigger_type": "MANUAL", "config": {}}
        steps_spec = []
        canvas_layout = {"nodes": [], "edges": []}

        # Check if forking from a designated version
        fork_id = payload.fork_from_version_id if payload else None
        if fork_id:
            source = await self.session.scalar(
                select(WorkflowVersion).where(
                    WorkflowVersion.id == fork_id,
                    WorkflowVersion.workflow_id == workflow_id,
                )
            )
            if source:
                trigger_spec = source.trigger_spec.copy() if isinstance(source.trigger_spec, dict) else trigger_spec
                steps_spec = [s.copy() for s in source.steps_spec] if isinstance(source.steps_spec, list) else steps_spec
                canvas_layout = source.canvas_layout.copy() if isinstance(source.canvas_layout, dict) else canvas_layout

        # Overwrite with explicitly provided fields in payload
        if payload:
            if payload.trigger_spec is not None:
                trigger_spec = payload.trigger_spec.model_dump()
            if payload.steps_spec is not None:
                steps_spec = [s.model_dump() for s in payload.steps_spec]
            if payload.canvas_layout is not None:
                canvas_layout = payload.canvas_layout.model_dump()

        new_version = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=workflow_id,
            version_number=next_ver,
            status=WorkflowVersionStatus.DRAFT.value,
            trigger_spec=trigger_spec,
            steps_spec=steps_spec,
            canvas_layout=canvas_layout,
            schema_version=1,
            created_by_id=user.id,
        )
        self.session.add(new_version)
        await self.session.commit()
        await self.session.refresh(new_version)
        return new_version

    async def get_version(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> WorkflowVersion:
        """
        Retrieves a specific workflow version, verifying parent workspace ownership.
        """
        await self.get_workflow(workspace_id, workflow_id)
        version = await self.session.scalar(
            select(WorkflowVersion).where(
                WorkflowVersion.id == version_id,
                WorkflowVersion.workflow_id == workflow_id,
            )
        )
        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow version '{version_id}' not found for workflow '{workflow_id}'.",
            )
        return version

    async def list_versions(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
    ) -> list[WorkflowVersion]:
        """
        Lists version history for a workflow definition, ordered by version_number descending.
        """
        await self.get_workflow(workspace_id, workflow_id)
        stmt = (
            select(WorkflowVersion)
            .where(WorkflowVersion.workflow_id == workflow_id)
            .order_by(WorkflowVersion.version_number.desc())
        )
        return list((await self.session.scalars(stmt)).all())

    async def update_draft_version(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
        version_id: uuid.UUID,
        payload: WorkflowVersionUpdate,
    ) -> WorkflowVersion:
        """
        Updates a draft version. Enforces strict immutability for published (ACTIVE/ARCHIVED) versions.
        """
        version = await self.get_version(workspace_id, workflow_id, version_id)

        # Immutability invariant
        if version.status != WorkflowVersionStatus.DRAFT.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version {version.version_number} is '{version.status}' and immutable. Edits must be made on a new DRAFT version.",
            )

        if payload.trigger_spec is not None:
            version.trigger_spec = payload.trigger_spec.model_dump()
        if payload.steps_spec is not None:
            version.steps_spec = [s.model_dump() for s in payload.steps_spec]
        if payload.canvas_layout is not None:
            version.canvas_layout = payload.canvas_layout.model_dump()

        await self.session.commit()
        await self.session.refresh(version)
        return version

    async def validate_version_dag(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> WorkflowDAGValidationResponse:
        """
        Deterministically inspects and validates the DAG structure of a workflow version.
        """
        version = await self.get_version(workspace_id, workflow_id, version_id)
        errors = validate_workflow_dag(version.steps_spec)
        return WorkflowDAGValidationResponse(
            is_valid=len(errors) == 0,
            errors=errors,
            node_count=len(version.steps_spec) if isinstance(version.steps_spec, list) else 0,
        )

    async def publish_version(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
        version_id: uuid.UUID,
        user: User,
    ) -> WorkflowVersion:
        """
        Atomically publishes a version:
        1. Acquires row lock (SELECT FOR UPDATE) on WorkflowDefinition to prevent concurrent publishing race conditions.
        2. Validates DAG: raises 422 Unprocessable Entity if invalid.
        3. Sets any currently ACTIVE version for this workflow to ARCHIVED.
        4. Sets this version to ACTIVE with published_at timestamp.
        5. Sets WorkflowDefinition.active_version_id to this version.
        """
        # 1. Acquire row lock on parent workflow definition
        lock_stmt = (
            select(WorkflowDefinition)
            .where(
                WorkflowDefinition.id == workflow_id,
                WorkflowDefinition.workspace_id == workspace_id,
                WorkflowDefinition.deleted_at.is_(None),
            )
            .with_for_update()
        )
        workflow = await self.session.scalar(lock_stmt)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow definition '{workflow_id}' not found in workspace.",
            )

        # 2. Retrieve the target version
        version = await self.session.scalar(
            select(WorkflowVersion).where(
                WorkflowVersion.id == version_id,
                WorkflowVersion.workflow_id == workflow_id,
            ).with_for_update()
        )
        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow version '{version_id}' not found.",
            )

        if version.status == WorkflowVersionStatus.ACTIVE.value:
            # Idempotent response
            return version

        # 3. Deterministic DAG Validation
        errors = validate_workflow_dag(version.steps_spec)
        if errors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Cannot publish invalid workflow DAG specification.",
                    "errors": errors,
                },
            )

        # 4. Archive any currently ACTIVE versions for this workflow
        await self.session.execute(
            update(WorkflowVersion)
            .where(
                WorkflowVersion.workflow_id == workflow_id,
                WorkflowVersion.status == WorkflowVersionStatus.ACTIVE.value,
            )
            .values(status=WorkflowVersionStatus.ARCHIVED.value)
        )

        # 5. Activate target version and update parent pointer
        now = datetime.now(timezone.utc)
        version.status = WorkflowVersionStatus.ACTIVE.value
        version.published_at = now

        workflow.active_version_id = version.id
        workflow.updated_at = now

        await self.session.commit()
        await self.session.refresh(version)
        return version

    async def execute_test_run(
        self,
        workspace_id: uuid.UUID,
        workflow_id: uuid.UUID,
        version_id: uuid.UUID,
        user: User,
        payload: WorkflowTestRunRequest,
    ) -> tuple[WorkflowRun, list[WorkflowTask]]:
        """
        Executes a workflow version (DRAFT or ACTIVE) in server-controlled safe TEST mode.
        Guarantees:
        1. Never mutates parent workflow active_version_id or version.status (draft remains draft).
        2. Validates DAG structure and action type existence prior to execution.
        3. Isolates test idempotency keys under 'test:' namespace to avoid production collisions.
        4. Binds run strictly to the tested version_id and marks thread_id as 'test:{version_id}'.
        5. Disables real external mutations in TEST mode via deterministic dry-run adapters.
        6. Preserves token and provider call budget accounting and timeout bounds.
        """
        # 1. Fetch workflow and target version
        workflow = await self.get_workflow(workspace_id, workflow_id)
        version = await self.get_version(workspace_id, workflow_id, version_id)

        # 2. Validate DAG specification
        dag_errors = validate_workflow_dag(version.steps_spec)
        if dag_errors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Cannot test invalid workflow DAG specification.",
                    "errors": dag_errors,
                },
            )

        # Validate that all node_types are registered action strategies
        registry = get_action_registry()
        for step in version.steps_spec:
            node_type = step.get("node_type")
            if not registry.has(node_type):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Cannot test workflow: Step '{step.get('id')}' references unknown action type '{node_type}'.",
                )

        # 3. Handle isolated test idempotency
        test_idempotency_key: Optional[str] = None
        if payload.idempotency_key:
            test_idempotency_key = f"test:{payload.idempotency_key.strip()}"

        limits = payload.budget_limits
        if limits is None or not isinstance(limits, BudgetLimits):
            if isinstance(limits, dict):
                limits = BudgetLimits(**limits)
            else:
                limits = BudgetLimits()

        req_hash = compute_canonical_request_hash(
            query=payload.inputs,
            workspace_id=workspace_id,
            budget_limits=limits.model_dump(),
        )

        if test_idempotency_key:
            stmt = select(WorkflowRun).where(
                and_(
                    WorkflowRun.user_id == user.id,
                    WorkflowRun.workspace_id == workspace_id,
                    WorkflowRun.idempotency_key == test_idempotency_key,
                )
            )
            existing_run = await self.session.scalar(stmt)
            if existing_run:
                if existing_run.request_hash != req_hash:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Test idempotency key reused with different request payload.",
                    )
                # Replay existing test run
                tasks_res = await self.session.scalars(
                    select(WorkflowTask)
                    .where(WorkflowTask.workflow_run_id == existing_run.id)
                    .order_by(WorkflowTask.created_at.asc())
                )
                return existing_run, list(tasks_res.all())

        # 4. Create isolated WorkflowRun for TEST execution with immutable runtime snapshot (Phase 5E)
        run_query = payload.inputs.get("query") or f"[TEST] Draft execution for {workflow.name} (v{version.version_number})"
        from rag_app.orchestration.runtime_snapshot import capture_runtime_snapshot
        snapshot = await capture_runtime_snapshot(
            session=self.session,
            workspace_id=workspace_id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            steps_spec=version.steps_spec,
            trigger_payload={"is_test": True, "inputs": payload.inputs},
        )

        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=workspace_id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            scope_type=ScopeType.WORKSPACE.value,
            thread_id=f"test:{version.id}",
            status=WorkflowStatus.RUNNING.value,
            idempotency_key=test_idempotency_key,
            request_hash=req_hash,
            initial_query=run_query,
            max_tokens_budget=limits.max_tokens,
            max_provider_calls_budget=limits.max_provider_calls,
            wall_clock_timeout_seconds=limits.wall_clock_timeout_seconds,
            started_at=datetime.now(timezone.utc),
            provider_snapshot=snapshot,
        )
        self.session.add(run)
        await self.session.commit()
        await self.session.refresh(run)

        # 5. Compile plan with allow_draft=True
        plan = compile_workflow_version_to_plan(
            workflow=workflow,
            version=version,
            run=run,
            trigger_payload=payload.inputs,
            allow_draft=True,
        )

        # 6. Execute DAG via DAGExecutionEngine with explicit server-controlled TEST context
        engine = DAGExecutionEngine(session=self.session)
        task_results = await engine.execute_plan(
            plan=plan,
            run=run,
            context={
                "trigger": payload.inputs,
                "execution_mode": "TEST",
                "is_test": True,
            },
        )

        # 7. Reconcile final run status
        now = datetime.now(timezone.utc)
        has_awaiting_approval = any(
            r.output_result and isinstance(r.output_result, dict) and r.output_result.get("status") == "awaiting_approval"
            for r in task_results.values()
        )
        all_succeeded = all(
            r.status in (WorkflowTaskStatus.COMPLETED, WorkflowTaskStatus.SKIPPED)
            for r in task_results.values()
        )

        if has_awaiting_approval:
            run.status = WorkflowStatus.AWAITING_APPROVAL.value
            run.awaiting_approval_at = now
        elif all_succeeded:
            run.status = WorkflowStatus.COMPLETED.value
            run.completed_at = now
        else:
            run.status = WorkflowStatus.FAILED.value
            run.failed_at = now
            for r in task_results.values():
                if r.status == WorkflowTaskStatus.FAILED and r.error_message:
                    run.error_message = r.error_message
                    break

        await self.session.commit()
        await self.session.refresh(run)

        # 8. Load persisted tasks
        tasks_stmt = (
            select(WorkflowTask)
            .where(WorkflowTask.workflow_run_id == run.id)
            .order_by(WorkflowTask.created_at.asc())
        )
        tasks = list((await self.session.scalars(tasks_stmt)).all())

        return run, tasks


