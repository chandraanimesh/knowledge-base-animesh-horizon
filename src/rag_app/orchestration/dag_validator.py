"""
Deterministic DAG Validator for AdaptiveTaskPlan.
Enforces graph integrity, cycle detection, dependency completeness,
task allowlists, fan-out/concurrency limits, and budget sanity.
"""

from collections import defaultdict, deque
from typing import Any, Optional

from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    DAGValidationResult,
    ExecutionTask,
    TaskType,
)


class DAGValidationError(ValueError):
    """Raised when an AdaptiveTaskPlan fails deterministic DAG validation."""

    def __init__(self, message: str, errors: Optional[list[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]


def validate_task_dag(
    plan: AdaptiveTaskPlan,
    max_concurrency_limit: int = 5,
    max_tasks_limit: int = 15,
    max_replan_iterations: int = 3,
    capability_registry: Optional[Any] = None,
    raise_on_error: bool = False,
) -> DAGValidationResult:
    """
    Deterministically validates an AdaptiveTaskPlan prior to execution.

    Invariants enforced:
    1. Plan must contain between 1 and max_tasks_limit tasks.
    2. Replan iteration must not exceed max_replan_iterations (hard limit 3).
    3. Task keys must be non-empty and unique within the plan.
    4. Task types must belong to the server-side TaskType allowlist.
    5. No self-dependencies (a task cannot depend on itself).
    6. No unknown dependencies (every dependency must reference a defined task_key).
    7. No dependency cycles (must form a valid Directed Acyclic Graph).
    8. Maximum layer fan-out / concurrent runnable tasks must not exceed max_concurrency_limit (hard limit 5).
    9. Resource estimates and timeouts must be positive and within bounds.
    """
    errors: list[str] = []

    # 1. Task count bounds
    if not plan.tasks:
        errors.append("Plan must contain at least one task.")
    elif len(plan.tasks) > max_tasks_limit:
        errors.append(
            f"Plan task count ({len(plan.tasks)}) exceeds hard limit of {max_tasks_limit}."
        )

    # 2. Replan iteration bounds
    if plan.iteration < 1 or plan.iteration > max_replan_iterations:
        errors.append(
            f"Plan iteration ({plan.iteration}) is invalid; must be between 1 and {max_replan_iterations}."
        )

    task_keys: set[str] = set()
    allowed_types = {t.value for t in TaskType}

    # 3. Task-level inspection
    for task in plan.tasks:
        # Unique keys
        if not task.task_key or not task.task_key.strip():
            errors.append("Encountered task with empty task_key.")
        elif task.task_key in task_keys:
            errors.append(f"Duplicate task_key '{task.task_key}' found in plan.")
        else:
            task_keys.add(task.task_key)

        # Allowlisted task types
        task_type_str = task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type)
        if task_type_str not in allowed_types:
            errors.append(
                f"Task '{task.task_key}' has unsupported task_type '{task_type_str}'. Allowed: {sorted(allowed_types)}."
            )

        # Self-dependency
        if task.task_key in task.dependencies:
            errors.append(f"Task '{task.task_key}' cannot depend on itself (self-dependency detected).")

        # Resource bounds sanity
        if task.estimated_tokens < 0:
            errors.append(f"Task '{task.task_key}' has negative estimated_tokens.")
        if task.estimated_provider_calls < 0:
            errors.append(f"Task '{task.task_key}' has negative estimated_provider_calls.")
        if task.timeout_seconds < 1:
            errors.append(f"Task '{task.task_key}' has invalid timeout_seconds ({task.timeout_seconds}).")

        # Unknown capability verification (fail closed)
        if capability_registry is not None and task.capability_id:
            cap = capability_registry.get(task.capability_id)
            if not cap:
                errors.append(
                    f"Task '{task.task_key}' references unknown or unregistered capability '{task.capability_id}'."
                )

    # 4. Dependency existence check
    for task in plan.tasks:
        for dep in task.dependencies:
            if dep not in task_keys:
                errors.append(
                    f"Task '{task.task_key}' depends on unknown task '{dep}'."
                )

    # If structural errors already exist, return early before cycle analysis
    if errors:
        result = DAGValidationResult(
            is_valid=False,
            errors=errors,
            topological_order=[],
            max_concurrency=0,
        )
        if raise_on_error:
            raise DAGValidationError(f"DAG validation failed with {len(errors)} error(s): {errors[0]}", errors)
        return result

    # 5. Graph construction for Cycle Detection & Layer Concurrency (Kahn's Algorithm)
    in_degree: dict[str, int] = {t.task_key: 0 for t in plan.tasks}
    adjacency: dict[str, list[str]] = defaultdict(list)
    task_map: dict[str, ExecutionTask] = {t.task_key: t for t in plan.tasks}

    for task in plan.tasks:
        for dep in task.dependencies:
            adjacency[dep].append(task.task_key)
            in_degree[task.task_key] += 1

    # Queue of nodes with zero in-degree
    queue: deque[tuple[str, int]] = deque((k, 0) for k, deg in in_degree.items() if deg == 0)
    topological_order: list[str] = []
    layer_map: dict[int, list[str]] = defaultdict(list)
    node_depth: dict[str, int] = {}

    while queue:
        curr, depth = queue.popleft()
        topological_order.append(curr)
        layer_map[depth].append(curr)
        node_depth[curr] = depth

        for neighbor in adjacency[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                next_depth = max(node_depth.get(dep, 0) + 1 for dep in task_map[neighbor].dependencies)
                queue.append((neighbor, next_depth))

    # Cycle check
    if len(topological_order) != len(plan.tasks):
        unvisited = [k for k, deg in in_degree.items() if deg > 0]
        errors.append(
            f"Dependency cycle detected in plan involving tasks: {sorted(unvisited)}."
        )

    # 6. Concurrency / Fan-out check across execution layers
    max_layer_width = max((len(nodes) for nodes in layer_map.values()), default=0)
    if max_layer_width > max_concurrency_limit:
        errors.append(
            f"Maximum task fan-out ({max_layer_width}) exceeds concurrency hard limit of {max_concurrency_limit}."
        )

    is_valid = len(errors) == 0
    res = DAGValidationResult(
        is_valid=is_valid,
        errors=errors,
        topological_order=topological_order if is_valid else [],
        max_concurrency=max_layer_width,
    )
    if not is_valid and raise_on_error:
        raise DAGValidationError(f"DAG validation failed with {len(errors)} error(s): {errors[0]}", errors)
    return res
