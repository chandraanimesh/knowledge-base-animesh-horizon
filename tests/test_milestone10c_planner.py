"""
Milestone 10C Verification Test Suite: Adaptive Planner & Typed Task DAG.

Verifies:
1. Typed initial plan formulation with explicit objectives, task dependencies, capability IDs, and resource estimates.
2. All generated tasks resolve to valid descriptors in CapabilityRegistry.
3. Static & dynamic DAG validation: cycle detection, unknown dependencies, self-dependencies.
4. Hard limits enforcement: max 15 tasks, max 5 parallel fan-out.
5. Unknown capability detection fails closed.
6. Bounded replanning upon missing evidence or contradictions.
7. Replan iteration bounding: strictly limited to max 3 iterations.
8. Remaining budget enforcement during replanning.
"""

from __future__ import annotations

import pytest

from rag_app.orchestration.capability_registry import (
    CapabilityRegistry,
    get_capability_registry,
)
from rag_app.orchestration.dag_validator import (
    DAGValidationError,
    validate_task_dag,
)
from rag_app.orchestration.planner import AdaptiveTaskPlanner
from rag_app.schemas.research import SufficiencyEvaluation, SufficiencyStatus
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    ExecutionTask,
    TaskType,
)


@pytest.fixture
def registry():
    return get_capability_registry()


@pytest.fixture
def planner(registry):
    return AdaptiveTaskPlanner(capability_registry=registry)


def test_initial_plan_formulation_and_capability_binding(planner, registry):
    goal = "Research modern CRM integration approaches and compare with our internal system"
    plan = planner.plan_initial_request(goal)

    assert plan.goal == goal
    assert plan.iteration == 1
    assert len(plan.tasks) >= 4
    assert plan.total_estimated_tokens > 0
    assert plan.total_estimated_provider_calls > 0

    # Verify all tasks have resolved capability IDs registered in CapabilityRegistry
    for task in plan.tasks:
        assert task.capability_id is not None, f"Task '{task.task_key}' missing capability_id"
        cap = registry.get(task.capability_id)
        assert cap is not None, f"Task '{task.task_key}' capability '{task.capability_id}' not found in registry"
        assert task.side_effect_level is not None
        assert isinstance(task.required_permissions, list)


def test_dag_validation_detects_cycles(registry):
    # A -> B -> C -> A
    tasks = [
        ExecutionTask(
            task_key="task_a",
            task_type=TaskType.RETRIEVAL,
            capability_id="rag.retrieve_internal",
            description="Task A",
            dependencies=["task_c"],
        ),
        ExecutionTask(
            task_key="task_b",
            task_type=TaskType.EXTRACTION,
            capability_id="research.extract_facts",
            description="Task B",
            dependencies=["task_a"],
        ),
        ExecutionTask(
            task_key="task_c",
            task_type=TaskType.SYNTHESIS,
            capability_id="synthesis.generate_findings",
            description="Task C",
            dependencies=["task_b"],
        ),
    ]
    plan = AdaptiveTaskPlan(goal="Cycle test", tasks=tasks)
    with pytest.raises(DAGValidationError) as exc:
        validate_task_dag(plan, capability_registry=registry, raise_on_error=True)
    assert "cycle detected" in str(exc.value).lower()


def test_dag_validation_enforces_max_task_count(registry):
    tasks = [
        ExecutionTask(
            task_key=f"task_{i}",
            task_type=TaskType.RETRIEVAL,
            capability_id="rag.retrieve_internal",
            description=f"Task {i}",
            dependencies=[f"task_{i-1}"] if i > 0 else [],
        )
        for i in range(16)  # Exceeds max limit of 15
    ]
    plan = AdaptiveTaskPlan(goal="Too many tasks", tasks=tasks)
    with pytest.raises(DAGValidationError) as exc:
        validate_task_dag(plan, max_tasks_limit=15, raise_on_error=True)
    assert "exceeds hard limit of 15" in str(exc.value)


def test_dag_validation_enforces_max_concurrency_fan_out(registry):
    # 6 tasks in layer 0 (all independent parallel tasks) -> exceeds limit of 5
    tasks = [
        ExecutionTask(
            task_key=f"parallel_{i}",
            task_type=TaskType.RETRIEVAL,
            capability_id="rag.retrieve_internal",
            description=f"Parallel task {i}",
            dependencies=[],
        )
        for i in range(6)
    ]
    plan = AdaptiveTaskPlan(goal="Excess fan-out", tasks=tasks)
    with pytest.raises(DAGValidationError) as exc:
        validate_task_dag(plan, max_concurrency_limit=5, raise_on_error=True)
    assert "exceeds concurrency hard limit of 5" in str(exc.value)


def test_dag_validation_rejects_unknown_capability(registry):
    tasks = [
        ExecutionTask(
            task_key="valid_task",
            task_type=TaskType.RETRIEVAL,
            capability_id="unregistered.fake_tool_xyz",
            description="Task with unregistered capability",
            dependencies=[],
        )
    ]
    plan = AdaptiveTaskPlan(goal="Unknown cap test", tasks=tasks)
    with pytest.raises(DAGValidationError) as exc:
        validate_task_dag(plan, capability_registry=registry, raise_on_error=True)
    assert "references unknown or unregistered capability" in str(exc.value)


def test_dag_validation_rejects_self_dependency(registry):
    tasks = [
        ExecutionTask(
            task_key="self_dep_task",
            task_type=TaskType.RETRIEVAL,
            capability_id="rag.retrieve_internal",
            description="Self dependent",
            dependencies=["self_dep_task"],
        )
    ]
    plan = AdaptiveTaskPlan(goal="Self dep test", tasks=tasks)
    with pytest.raises(DAGValidationError) as exc:
        validate_task_dag(plan, capability_registry=registry, raise_on_error=True)
    assert "cannot depend on itself" in str(exc.value)


def test_bounded_replan_formulation(planner):
    initial_plan = planner.plan_initial_request("Initial research goal")
    assert initial_plan.iteration == 1

    partial_sufficiency = SufficiencyEvaluation(
        status=SufficiencyStatus.PARTIAL,
        reason="Missing compliance certifications and performance SLAs",
        missing_requirements=["compliance certifications", "performance SLAs"],
        contradictions=[],
    )

    replan = planner.replan_for_missing_evidence(
        current_plan=initial_plan,
        sufficiency=partial_sufficiency,
        remaining_token_budget=10000,
        remaining_provider_calls=10,
    )

    assert replan is not None
    assert replan.iteration == 2
    assert len(replan.tasks) >= 3
    # Verify capability_id bound on replan tasks
    for t in replan.tasks:
        assert t.capability_id is not None


def test_replan_hard_stop_at_iteration_three(planner):
    # Create plan already at iteration 3
    plan_iter_3 = AdaptiveTaskPlan(
        goal="Deep research",
        tasks=[
            ExecutionTask(
                task_key="t1",
                task_type=TaskType.RETRIEVAL,
                capability_id="rag.retrieve_internal",
                description="T1",
                dependencies=[],
            )
        ],
        iteration=3,
    )

    insufficient = SufficiencyEvaluation(
        status=SufficiencyStatus.INSUFFICIENT,
        reason="Still missing critical data",
        missing_requirements=["data missing"],
    )

    replan = planner.replan_for_missing_evidence(
        current_plan=plan_iter_3,
        sufficiency=insufficient,
        remaining_token_budget=10000,
        remaining_provider_calls=10,
    )
    # Must enforce hard stop
    assert replan is None


def test_replan_stops_safely_when_budget_exhausted(planner):
    initial_plan = planner.plan_initial_request("Initial research")

    partial_sufficiency = SufficiencyEvaluation(
        status=SufficiencyStatus.PARTIAL,
        reason="Need more data",
        missing_requirements=["extra data 1"],
    )

    # Budget has only 100 tokens remaining (less than the required tokens for replan tasks)
    replan = planner.replan_for_missing_evidence(
        current_plan=initial_plan,
        sufficiency=partial_sufficiency,
        remaining_token_budget=100,
        remaining_provider_calls=10,
    )
    assert replan is None

