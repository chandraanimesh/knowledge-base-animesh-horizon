"""
Milestone 2 Acceptance Test Suite:
Adaptive Task Planner, Deterministic Typed DAG Validation, Bounded Execution Engine,
Evidence Reducer, Sufficiency Gate, and Scoped Idempotency.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
from rag_app.orchestration.dag_validator import (
    DAGValidationError,
    validate_task_dag,
)
from rag_app.orchestration.evidence import EvidenceAggregator
from rag_app.orchestration.executor import (
    BudgetExceededError,
    DAGExecutionEngine,
    FencingConflictError,
)
from rag_app.orchestration.planner import AdaptiveTaskPlanner
from rag_app.orchestration.service import WorkflowService
from rag_app.schemas.research import SufficiencyStatus
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    BudgetLimits,
    ExecutionTask,
    ScopeType,
    TaskEvidenceRef,
    TaskExecutionResult,
    TaskType,
    WorkflowRunCreate,
    WorkflowStatus,
    WorkflowTaskStatus,
)


# ============================================================================
# Gate 1: Deterministic DAG Validation Tests
# ============================================================================

def test_valid_dag_validation_succeeds():
    """A valid DAG plan passes validation with computed topological order and concurrency."""
    plan = AdaptiveTaskPlan(
        goal="Analyze Q3 Market Performance",
        tasks=[
            ExecutionTask(
                task_key="t1_retrieve",
                task_type=TaskType.RETRIEVAL,
                description="Retrieve internal docs",
                dependencies=[],
            ),
            ExecutionTask(
                task_key="t2_web",
                task_type=TaskType.WEB_SEARCH,
                description="Web search competitor metrics",
                dependencies=[],
            ),
            ExecutionTask(
                task_key="t3_extract",
                task_type=TaskType.EXTRACTION,
                description="Extract tabular metrics",
                dependencies=["t1_retrieve", "t2_web"],
            ),
            ExecutionTask(
                task_key="t4_synth",
                task_type=TaskType.SYNTHESIS,
                description="Synthesize final report",
                dependencies=["t3_extract"],
            ),
        ],
    )

    res = validate_task_dag(plan, max_concurrency_limit=5)
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert res.topological_order == ["t1_retrieve", "t2_web", "t3_extract", "t4_synth"]
    assert res.max_concurrency == 2  # t1 and t2 run concurrently in layer 0


def test_validation_rejects_duplicate_task_keys():
    """Duplicate task keys in plan are rejected."""
    plan = AdaptiveTaskPlan(
        goal="Test Duplicate Keys",
        tasks=[
            ExecutionTask(
                task_key="duplicate_key",
                task_type=TaskType.RETRIEVAL,
                description="First task",
                dependencies=[],
            ),
            ExecutionTask(
                task_key="duplicate_key",
                task_type=TaskType.WEB_SEARCH,
                description="Second task with same key",
                dependencies=[],
            ),
        ],
    )

    res = validate_task_dag(plan)
    assert res.is_valid is False
    assert any("Duplicate task_key" in err for err in res.errors)

    with pytest.raises(DAGValidationError):
        validate_task_dag(plan, raise_on_error=True)


def test_validation_rejects_missing_dependencies():
    """Tasks depending on undefined task keys are rejected."""
    plan = AdaptiveTaskPlan(
        goal="Test Missing Dependency",
        tasks=[
            ExecutionTask(
                task_key="task_a",
                task_type=TaskType.RETRIEVAL,
                description="Task A",
                dependencies=["non_existent_task_x"],
            ),
        ],
    )

    res = validate_task_dag(plan)
    assert res.is_valid is False
    assert any("depends on unknown task 'non_existent_task_x'" in err for err in res.errors)


def test_validation_rejects_self_dependency():
    """A task depending on itself is rejected."""
    plan = AdaptiveTaskPlan(
        goal="Test Self Dependency",
        tasks=[
            ExecutionTask(
                task_key="task_self",
                task_type=TaskType.RETRIEVAL,
                description="Self loop",
                dependencies=["task_self"],
            ),
        ],
    )

    res = validate_task_dag(plan)
    assert res.is_valid is False
    assert any("self-dependency detected" in err for err in res.errors)


def test_validation_rejects_dependency_cycle():
    """Circular dependencies (e.g., A -> B -> C -> A) are rejected."""
    plan = AdaptiveTaskPlan(
        goal="Test Cycle",
        tasks=[
            ExecutionTask(
                task_key="node_a",
                task_type=TaskType.RETRIEVAL,
                description="A",
                dependencies=["node_c"],
            ),
            ExecutionTask(
                task_key="node_b",
                task_type=TaskType.EXTRACTION,
                description="B",
                dependencies=["node_a"],
            ),
            ExecutionTask(
                task_key="node_c",
                task_type=TaskType.SYNTHESIS,
                description="C",
                dependencies=["node_b"],
            ),
        ],
    )

    res = validate_task_dag(plan)
    assert res.is_valid is False
    assert any("Dependency cycle detected" in err for err in res.errors)


def test_validation_rejects_excessive_fan_out():
    """Layer concurrency exceeding hard limit of 5 is rejected."""
    # 6 independent tasks in layer 0
    tasks = [
        ExecutionTask(
            task_key=f"task_{i}",
            task_type=TaskType.RETRIEVAL,
            description=f"Task {i}",
            dependencies=[],
        )
        for i in range(6)
    ]
    plan = AdaptiveTaskPlan(goal="Test Fan-out", tasks=tasks)

    res = validate_task_dag(plan, max_concurrency_limit=5)
    assert res.is_valid is False
    assert any("exceeds concurrency hard limit of 5" in err for err in res.errors)


def test_validation_rejects_invalid_replan_iteration():
    """Plan iteration exceeding max 3 is rejected."""
    task = ExecutionTask(
        task_key="t1",
        task_type=TaskType.RETRIEVAL,
        description="Task 1",
    )
    with pytest.raises(Exception):
        # Pydantic validates le=3
        AdaptiveTaskPlan(goal="Test Iteration", tasks=[task], iteration=4)


# ============================================================================
# Gate 2: Bounded DAG Execution & Concurrency Tests
# ============================================================================

@pytest.mark.asyncio
async def test_executor_respects_task_dependencies():
    """Downstream tasks wait for prerequisite tasks to complete before running."""
    execution_order: list[str] = []

    async def make_handler(name: str, delay: float):
        async def handler(task: ExecutionTask, session, run):
            await asyncio.sleep(delay)
            execution_order.append(name)
            return TaskExecutionResult(
                task_key=task.task_key,
                status=WorkflowTaskStatus.COMPLETED,
            )
        return handler

    engine = DAGExecutionEngine(max_concurrency=5)
    engine.register_handler(TaskType.RETRIEVAL.value, await make_handler("retrieval", 0.05))
    engine.register_handler(TaskType.EXTRACTION.value, await make_handler("extraction", 0.01))

    plan = AdaptiveTaskPlan(
        goal="Order test",
        tasks=[
            ExecutionTask(
                task_key="step_1",
                task_type=TaskType.RETRIEVAL,
                description="Prerequisite step",
                dependencies=[],
            ),
            ExecutionTask(
                task_key="step_2",
                task_type=TaskType.EXTRACTION,
                description="Dependent step",
                dependencies=["step_1"],
            ),
        ],
    )

    mock_run = MagicMock()
    mock_run.id = uuid.uuid4()

    results = await engine.execute_plan(plan, mock_run)
    assert results["step_1"].status == WorkflowTaskStatus.COMPLETED
    assert results["step_2"].status == WorkflowTaskStatus.COMPLETED
    assert execution_order == ["retrieval", "extraction"]


@pytest.mark.asyncio
async def test_executor_enforces_max_concurrency_limit():
    """Active concurrent tasks never exceed max_concurrency limit (hard cap 5)."""
    max_active_observed = 0
    current_active = 0
    active_lock = asyncio.Lock()

    async def concurrency_tracking_handler(task: ExecutionTask, session, run):
        nonlocal max_active_observed, current_active
        async with active_lock:
            current_active += 1
            if current_active > max_active_observed:
                max_active_observed = current_active

        await asyncio.sleep(0.05)

        async with active_lock:
            current_active -= 1

        return TaskExecutionResult(
            task_key=task.task_key,
            status=WorkflowTaskStatus.COMPLETED,
        )

    engine = DAGExecutionEngine(max_concurrency=3)  # Test with limit 3
    engine.register_handler(TaskType.RETRIEVAL.value, concurrency_tracking_handler)

    tasks = [
        ExecutionTask(
            task_key=f"concurrent_task_{i}",
            task_type=TaskType.RETRIEVAL,
            description=f"Task {i}",
            dependencies=[],
        )
        for i in range(5)
    ]
    plan = AdaptiveTaskPlan(goal="Concurrency test", tasks=tasks)
    mock_run = MagicMock()
    mock_run.id = uuid.uuid4()

    await engine.execute_plan(plan, mock_run)
    assert max_active_observed <= 3
    assert max_active_observed >= 2


@pytest.mark.asyncio
async def test_task_failure_propagation_skips_dependents():
    """When an upstream task fails, downstream dependent tasks are marked SKIPPED."""
    async def failing_handler(task: ExecutionTask, session, run):
        raise RuntimeError("External retrieval failed permanently")

    engine = DAGExecutionEngine()
    engine.register_handler(TaskType.RETRIEVAL.value, failing_handler)

    plan = AdaptiveTaskPlan(
        goal="Failure propagation test",
        tasks=[
            ExecutionTask(
                task_key="upstream_failing",
                task_type=TaskType.RETRIEVAL,
                description="Fails",
                dependencies=[],
            ),
            ExecutionTask(
                task_key="downstream_dependent",
                task_type=TaskType.SYNTHESIS,
                description="Should be skipped",
                dependencies=["upstream_failing"],
            ),
        ],
    )
    mock_run = MagicMock()
    mock_run.id = uuid.uuid4()

    results = await engine.execute_plan(plan, mock_run)
    assert results["upstream_failing"].status == WorkflowTaskStatus.FAILED
    assert engine.task_states["downstream_dependent"] == WorkflowTaskStatus.SKIPPED


@pytest.mark.asyncio
async def test_stale_fencing_token_rejected_on_completion():
    """A worker presenting an outdated fencing token is rejected with FencingConflictError."""
    engine = DAGExecutionEngine()
    task = ExecutionTask(
        task_key="fence_test",
        task_type=TaskType.RETRIEVAL,
        description="Fencing test",
    )
    engine.task_fencing_tokens["fence_test"] = 5  # Current token is 5

    with pytest.raises(FencingConflictError):
        # Worker attempts to complete with old token 4
        await engine._complete_task(
            task=task,
            result=TaskExecutionResult(task_key="fence_test", status=WorkflowTaskStatus.COMPLETED),
            fencing_token=4,
        )


# ============================================================================
# Gate 3: Evidence Aggregation, Provenance & Sufficiency Tests
# ============================================================================

def test_evidence_aggregator_preserves_provenance_and_deduplicates():
    """Aggregator deduplicates identical snippets while preserving unique sources and citations."""
    aggregator = EvidenceAggregator()

    ref1 = TaskEvidenceRef(
        task_key="task_1",
        source_type="internal_rag",
        source_id="doc_101",
        snippet="Enterprise revenue grew 24% YoY in fiscal 2025.",
    )
    ref1_duplicate = TaskEvidenceRef(
        task_key="task_2",
        source_type="internal_rag",
        source_id="doc_101",
        snippet="Enterprise revenue grew 24% YoY in fiscal 2025.",  # Exact duplicate
    )
    ref2 = TaskEvidenceRef(
        task_key="task_3",
        source_type="web_search",
        source_id="https://news.example.com/growth",
        snippet="Market analysts report enterprise growth at 18% YoY.",
        contradictions=["Growth rate mismatch: 24% internal vs 18% external"],
    )

    assert aggregator.add_evidence(ref1) is True
    assert aggregator.add_evidence(ref1_duplicate) is False  # Deduplicated!
    assert aggregator.add_evidence(ref2) is True

    all_evidence = aggregator.get_all_evidence()
    assert len(all_evidence) == 2
    assert len(aggregator.get_contradictions()) == 1
    assert "Growth rate mismatch" in aggregator.get_contradictions()[0]


def test_evidence_sufficiency_evaluation_status_classification():
    """Sufficiency gate correctly identifies SUFFICIENT, PARTIAL, and INSUFFICIENT evidence."""
    aggregator = EvidenceAggregator()

    # 1. No evidence -> INSUFFICIENT
    eval_empty = aggregator.evaluate_sufficiency(
        goal="Report on Q3 Cloud Revenue and Margin",
        required_dimensions=["Q3 Cloud Revenue", "Operating Margin"],
    )
    assert eval_empty.status == SufficiencyStatus.INSUFFICIENT
    assert eval_empty.answerable_from_context is False
    assert len(eval_empty.missing_requirements) == 2

    # 2. Add partial evidence -> PARTIAL
    aggregator.add_evidence(
        TaskEvidenceRef(
            task_key="t1",
            source_type="internal_rag",
            source_id="fin_q3",
            snippet="Q3 Cloud Revenue reached $450M.",
        )
    )
    eval_partial = aggregator.evaluate_sufficiency(
        goal="Report on Q3 Cloud Revenue and Margin",
        required_dimensions=["Q3 Cloud Revenue", "Operating Margin"],
    )
    assert eval_partial.status == SufficiencyStatus.PARTIAL
    assert "Operating Margin" in eval_partial.missing_requirements

    # 3. Add remaining dimension -> SUFFICIENT
    aggregator.add_evidence(
        TaskEvidenceRef(
            task_key="t2",
            source_type="internal_rag",
            source_id="margin_doc",
            snippet="Operating Margin remained steady at 28.5%.",
        )
    )
    eval_sufficient = aggregator.evaluate_sufficiency(
        goal="Report on Q3 Cloud Revenue and Margin",
        required_dimensions=["Q3 Cloud Revenue", "Operating Margin"],
    )
    assert eval_sufficient.status == SufficiencyStatus.SUFFICIENT
    assert eval_sufficient.answerable_from_context is True
    assert len(eval_sufficient.missing_requirements) == 0


# ============================================================================
# Gate 4: Adaptive Planner & Bounded Replanning Tests
# ============================================================================

def test_planner_generates_valid_initial_plan():
    """Planner creates a valid DAG plan satisfying fan-out, type, and dependency constraints."""
    planner = AdaptiveTaskPlanner()
    plan = planner.plan_initial_request("Evaluate competitive CRM landscape")

    assert plan.iteration == 1
    assert len(plan.tasks) == 5
    assert plan.total_estimated_tokens > 0
    assert plan.total_estimated_provider_calls > 0

    val = validate_task_dag(plan)
    assert val.is_valid is True


def test_planner_bounded_replanning_stops_at_limit_and_budget():
    """Replanning never exceeds 3 iterations and stops when budget is exhausted."""
    planner = AdaptiveTaskPlanner()
    initial_plan = planner.plan_initial_request("Investigate AI infrastructure costs")

    # Mock sufficiency evaluation with missing requirements
    partial_eval = EvidenceAggregator().evaluate_sufficiency(
        goal="Investigate AI infrastructure costs",
        required_dimensions=["Server hardware", "Electricity cost", "GPU leases"],
    )

    # 1. Iteration 1 -> Iteration 2
    replan_1 = planner.replan_for_missing_evidence(
        current_plan=initial_plan,
        sufficiency=partial_eval,
        remaining_token_budget=10000,
        remaining_provider_calls=5,
    )
    assert replan_1 is not None
    assert replan_1.iteration == 2

    # 2. Iteration 2 -> Iteration 3
    replan_2 = planner.replan_for_missing_evidence(
        current_plan=replan_1,
        sufficiency=partial_eval,
        remaining_token_budget=10000,
        remaining_provider_calls=5,
    )
    assert replan_2 is not None
    assert replan_2.iteration == 3

    # 3. Iteration 3 -> Stopped! (Hard cap of 3 iterations)
    replan_3 = planner.replan_for_missing_evidence(
        current_plan=replan_2,
        sufficiency=partial_eval,
        remaining_token_budget=10000,
        remaining_provider_calls=5,
    )
    assert replan_3 is None  # Safe stop triggered!

    # 4. Budget exhaustion stops replanning
    replan_budget_stop = planner.replan_for_missing_evidence(
        current_plan=initial_plan,
        sufficiency=partial_eval,
        remaining_token_budget=100,  # Insufficient tokens
        remaining_provider_calls=0,  # Zero calls
    )
    assert replan_budget_stop is None  # Safe stop triggered due to budget!


# ============================================================================
# Gate 5: Scoped Idempotency & Workflow Lifecycle Tests
# ============================================================================

@pytest.mark.asyncio
async def test_scoped_idempotency_same_key_same_payload_returns_existing():
    """Same authorized scope + same key + same payload re-authorizes and returns existing run."""
    mock_session = AsyncMock()
    user = MagicMock()
    user.id = uuid.uuid4()
    ws_id = uuid.uuid4()
    key = "idem-test-key-101"

    existing_run = MagicMock()
    existing_run.id = uuid.uuid4()
    from rag_app.schemas.workflow import compute_canonical_request_hash
    req_hash = compute_canonical_request_hash("Analyze market", workspace_id=ws_id, budget_limits=BudgetLimits().model_dump())
    existing_run.request_hash = req_hash

    mock_session.scalar.return_value = existing_run

    service = WorkflowService(session=mock_session)
    req = WorkflowRunCreate(
        query="Analyze market",
        workspace_id=ws_id,
        idempotency_key=key,
    )

    run, created = await service.get_or_create_workflow_run(user=user, request=req)
    assert created is False
    assert run.id == existing_run.id


@pytest.mark.asyncio
async def test_scoped_idempotency_changed_payload_returns_409_conflict():
    """Same scope + same key + different request hash raises HTTP 409 Conflict."""
    mock_session = AsyncMock()
    user = MagicMock()
    user.id = uuid.uuid4()
    ws_id = uuid.uuid4()
    key = "idem-test-key-102"

    existing_run = MagicMock()
    existing_run.id = uuid.uuid4()
    existing_run.request_hash = "different_original_hash_value"
    mock_session.scalar.return_value = existing_run

    service = WorkflowService(session=mock_session)
    req = WorkflowRunCreate(
        query="Modified market analysis payload",  # Changed!
        workspace_id=ws_id,
        idempotency_key=key,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.get_or_create_workflow_run(user=user, request=req)

    assert exc_info.value.status_code == 409
    assert "Idempotency key reused with different request payload" in exc_info.value.detail


@pytest.mark.asyncio
async def test_global_scope_vs_workspace_scope_isolation():
    """Global scope (workspace_id IS NULL) is strictly isolated from workspace scope."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    user = MagicMock()
    user.id = uuid.uuid4()
    key = "shared-idem-key"

    # In global scope query, workspace_id must be None
    mock_session.scalar.return_value = None  # No existing global run

    service = WorkflowService(session=mock_session)
    req = WorkflowRunCreate(
        query="Global query",
        workspace_id=None,  # GLOBAL
        idempotency_key=key,
    )

    new_run, created = await service.get_or_create_workflow_run(user=user, request=req)
    assert created is True
    assert new_run.scope_type == ScopeType.GLOBAL.value
    assert new_run.workspace_id is None
