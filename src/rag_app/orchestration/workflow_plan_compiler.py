"""
Workflow Plan Compiler.
Compiles a published WorkflowDefinition and immutable WorkflowVersion into an AdaptiveTaskPlan DAG,
populating typed ExecutionTasks, calculating resource estimates from the Action Strategy Registry,
and linking the execution run to the exact immutable version.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import uuid

from rag_app.db.models import WorkflowDefinition, WorkflowRun, WorkflowVersion
from rag_app.orchestration.action_registry import get_action_registry
from rag_app.orchestration.dag_validator import DAGValidationError, validate_task_dag
from rag_app.schemas.workflow import AdaptiveTaskPlan, ExecutionTask, TaskType
from rag_app.schemas.workflow_definition import (
    validate_workflow_dag,
    WorkflowActionType,
    WorkflowVersionStatus,
)

logger = logging.getLogger(__name__)


class WorkflowCompilationError(ValueError):
    """Raised when compiling a WorkflowVersion into an AdaptiveTaskPlan fails."""
    pass


def compile_workflow_version_to_plan(
    workflow: WorkflowDefinition,
    version: WorkflowVersion,
    run: WorkflowRun,
    trigger_payload: Optional[dict[str, Any]] = None,
    allow_draft: bool = False,
) -> AdaptiveTaskPlan:
    """
    Deterministically transforms a WorkflowVersion's step specification into an AdaptiveTaskPlan.
    
    Invariants enforced:
    1. WorkflowVersion must be ACTIVE (or DRAFT if allow_draft=True for simulation/testing).
    2. Version DAG structure must pass deterministic topological validation.
    3. Run record is bound to the exact immutable workflow_id and workflow_version_id.
    4. Task resource estimates and safety flags are populated from registered action strategies.
    5. Resulting AdaptiveTaskPlan is verified with validate_task_dag.
    """
    if not allow_draft and version.status != WorkflowVersionStatus.ACTIVE.value:
        raise WorkflowCompilationError(
            f"Cannot compile workflow '{workflow.name}': Version {version.version_number} "
            f"has status '{version.status}', but only '{WorkflowVersionStatus.ACTIVE.value}' versions can be executed."
        )

    # 1. Deterministic DAG validation
    dag_errors = validate_workflow_dag(version.steps_spec)
    if dag_errors:
        raise WorkflowCompilationError(
            f"WorkflowVersion {version.version_number} has invalid DAG specification: {'; '.join(dag_errors)}"
        )

    # 2. Bind immutable version reference to run
    run.workflow_id = workflow.id
    run.workflow_version_id = version.id

    # 3. Retrieve Action Registry for resource estimation
    registry = get_action_registry()

    # 4. Construct ExecutionTask items
    tasks: list[ExecutionTask] = []
    total_est_tokens = 0
    total_est_calls = 0

    for step in version.steps_spec:
        step_id = step["id"]
        node_type_raw = step["node_type"]
        node_name = step.get("name", f"Execute {node_type_raw}")
        deps = step.get("dependencies", [])
        inputs = dict(step.get("inputs", {}))

        # Map to TaskType enum
        try:
            task_type = TaskType(node_type_raw)
        except ValueError:
            # Fall back to matching string or unknown
            task_type = TaskType.RETRIEVAL

        strategy = registry.get(node_type_raw)
        if strategy:
            est_tokens = strategy.estimated_tokens
            est_calls = strategy.estimated_provider_calls
            timeout = strategy.timeout_seconds
            is_read_only = (node_type_raw != WorkflowActionType.EXECUTION.value)
        else:
            est_tokens = 500
            est_calls = 1
            timeout = 60
            is_read_only = True

        total_est_tokens += est_tokens
        total_est_calls += est_calls

        task = ExecutionTask(
            task_key=step_id,
            task_type=task_type,
            description=node_name,
            dependencies=deps,
            input_payload=inputs,
            estimated_tokens=est_tokens,
            estimated_provider_calls=est_calls,
            timeout_seconds=timeout,
            is_read_only=is_read_only,
        )
        tasks.append(task)

    # 5. Build AdaptiveTaskPlan
    goal_str = f"Execute Workflow '{workflow.name}' (v{version.version_number})"
    if trigger_payload and "query" in trigger_payload:
        goal_str = f"{goal_str}: {trigger_payload['query']}"

    plan = AdaptiveTaskPlan(
        plan_id=uuid.uuid4().hex[:16],
        workflow_run_id=run.id,
        goal=goal_str,
        tasks=tasks,
        iteration=1,
        total_estimated_tokens=total_est_tokens,
        total_estimated_provider_calls=total_est_calls,
        metadata={
            "workflow_id": str(workflow.id),
            "workflow_version_id": str(version.id),
            "version_number": version.version_number,
            "trigger_payload": trigger_payload or {},
        },
    )

    # 6. Verify compiled DAG
    dag_val = validate_task_dag(plan, max_concurrency_limit=5, raise_on_error=True)
    if not dag_val.is_valid:
        raise WorkflowCompilationError(
            f"Compiled task DAG failed validation: {'; '.join(dag_val.errors)}"
        )

    return plan

