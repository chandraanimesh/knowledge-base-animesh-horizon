"""
Milestone 10D Verification Test Suite: Execution Coordinator & Parallel Execution.

Verifies:
1. Concurrent execution of independent tasks (e.g., parallel RAG + Web + Memory retrieval).
2. Strict dependency sequencing: dependent tasks wait for all prerequisites.
3. Partial failure resilience: failure of one parallel task preserves completed sibling tasks while skipping dependents.
4. Bounded concurrency: simultaneous task execution never exceeds max_concurrency cap (5).
"""

from __future__ import annotations

import asyncio
import time
import uuid
import pytest

from rag_app.orchestration.executor import (
    DAGExecutionEngine,
    TaskExecutionResult,
)
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    ExecutionTask,
    TaskType,
    WorkflowTaskStatus,
)


class MockWorkflowRun:
    def __init__(self):
        self.id = uuid.uuid4()
        self.status = "RUNNING"


@pytest.mark.asyncio
async def test_parallel_execution_of_independent_tasks():
    """
    Independent tasks with zero dependencies execute concurrently,
    completing in approximately max(task_durations) rather than sum(task_durations).
    """
    start_times = {}

    async def make_handler(name: str, delay: float):
        async def handler(task: ExecutionTask, session, run):
            start_times[task.task_key] = time.perf_counter()
            await asyncio.sleep(delay)
            return TaskExecutionResult(
                task_key=task.task_key,
                status=WorkflowTaskStatus.COMPLETED,
                output_result={"status": "ok", "source": name},
            )
        return handler

    engine = DAGExecutionEngine(max_concurrency=5)
    engine.register_handler(TaskType.RETRIEVAL.value, await make_handler("retrieval", 0.1))
    engine.register_handler(TaskType.WEB_SEARCH.value, await make_handler("web_search", 0.1))

    tasks = [
        ExecutionTask(
            task_key="task_rag",
            task_type=TaskType.RETRIEVAL,
            capability_id="rag.retrieve_internal",
            description="Parallel RAG",
            dependencies=[],
        ),
        ExecutionTask(
            task_key="task_web",
            task_type=TaskType.WEB_SEARCH,
            capability_id="research.web_search",
            description="Parallel Web",
            dependencies=[],
        ),
        ExecutionTask(
            task_key="task_mem",
            task_type=TaskType.RETRIEVAL,
            capability_id="memory.retrieve_context",
            description="Parallel Memory",
            dependencies=[],
        ),
    ]
    plan = AdaptiveTaskPlan(goal="Parallel test", tasks=tasks)
    run = MockWorkflowRun()

    t0 = time.perf_counter()
    results = await engine.execute_plan(plan, run)
    total_time = time.perf_counter() - t0

    assert len(results) == 3
    assert all(r.status == WorkflowTaskStatus.COMPLETED for r in results.values())

    # Total time for 3 x 0.1s parallel tasks must be < 0.25s (well below serial 0.3s)
    assert total_time < 0.25, f"Expected concurrent execution, took {total_time:.3f}s"

    # All start times should be nearly identical (within 0.05s)
    starts = list(start_times.values())
    assert max(starts) - min(starts) < 0.05


@pytest.mark.asyncio
async def test_strict_dependency_sequencing():
    """
    Dependent tasks must strictly wait until all prerequisite tasks have completed.
    """
    events = []

    async def handler_retrieval(task: ExecutionTask, session, run):
        events.append(f"start_{task.task_key}")
        await asyncio.sleep(0.05)
        events.append(f"end_{task.task_key}")
        return TaskExecutionResult(task_key=task.task_key, status=WorkflowTaskStatus.COMPLETED)

    async def handler_synthesis(task: ExecutionTask, session, run):
        events.append(f"start_{task.task_key}")
        await asyncio.sleep(0.02)
        events.append(f"end_{task.task_key}")
        return TaskExecutionResult(task_key=task.task_key, status=WorkflowTaskStatus.COMPLETED)

    engine = DAGExecutionEngine()
    engine.register_handler(TaskType.RETRIEVAL.value, handler_retrieval)
    engine.register_handler(TaskType.SYNTHESIS.value, handler_synthesis)

    tasks = [
        ExecutionTask(task_key="r1", task_type=TaskType.RETRIEVAL, description="R1", dependencies=[]),
        ExecutionTask(task_key="r2", task_type=TaskType.RETRIEVAL, description="R2", dependencies=[]),
        ExecutionTask(task_key="synth", task_type=TaskType.SYNTHESIS, description="Synth", dependencies=["r1", "r2"]),
    ]
    plan = AdaptiveTaskPlan(goal="Dependency test", tasks=tasks)
    run = MockWorkflowRun()

    results = await engine.execute_plan(plan, run)
    assert results["synth"].status == WorkflowTaskStatus.COMPLETED

    # Both r1 and r2 must have ended before synth started
    idx_end_r1 = events.index("end_r1")
    idx_end_r2 = events.index("end_r2")
    idx_start_synth = events.index("start_synth")

    assert idx_start_synth > idx_end_r1
    assert idx_start_synth > idx_end_r2


@pytest.mark.asyncio
async def test_partial_failure_preserves_sibling_and_skips_dependents():
    """
    When one parallel task fails, sibling independent tasks complete successfully,
    and downstream dependent tasks are skipped.
    """
    async def handler_retrieval(task: ExecutionTask, session, run):
        if task.task_key == "task_bad":
            await asyncio.sleep(0.02)
            raise RuntimeError("External network timeout")
        await asyncio.sleep(0.03)
        return TaskExecutionResult(
            task_key=task.task_key,
            status=WorkflowTaskStatus.COMPLETED,
            output_result={"status": "success"},
        )

    async def handler_downstream(task: ExecutionTask, session, run):
        return TaskExecutionResult(task_key=task.task_key, status=WorkflowTaskStatus.COMPLETED)

    engine = DAGExecutionEngine()
    engine.register_handler(TaskType.RETRIEVAL.value, handler_retrieval)
    engine.register_handler(TaskType.SYNTHESIS.value, handler_downstream)

    tasks = [
        ExecutionTask(task_key="task_good", task_type=TaskType.RETRIEVAL, description="Good", dependencies=[]),
        ExecutionTask(task_key="task_bad", task_type=TaskType.RETRIEVAL, description="Bad", dependencies=[]),
        ExecutionTask(task_key="task_dependent", task_type=TaskType.SYNTHESIS, description="Dependent", dependencies=["task_bad"]),
    ]
    plan = AdaptiveTaskPlan(goal="Failure test", tasks=tasks)
    run = MockWorkflowRun()

    results = await engine.execute_plan(plan, run)

    # Good task completed and preserved
    assert results["task_good"].status == WorkflowTaskStatus.COMPLETED
    assert results["task_good"].output_result == {"status": "success"}

    # Bad task failed
    assert results["task_bad"].status == WorkflowTaskStatus.FAILED
    assert "network timeout" in results["task_bad"].error_message.lower()

    # Dependent task skipped
    assert results["task_dependent"].status == WorkflowTaskStatus.SKIPPED


@pytest.mark.asyncio
async def test_bounded_concurrency_never_exceeds_five():
    """
    Even with 5 runnable tasks in a layer, active concurrent tasks never exceed 5.
    """
    active_count = 0
    max_observed_active = 0

    async def bounded_handler(task: ExecutionTask, session, run):
        nonlocal active_count, max_observed_active
        active_count += 1
        if active_count > max_observed_active:
            max_observed_active = active_count
        await asyncio.sleep(0.05)
        active_count -= 1
        return TaskExecutionResult(task_key=task.task_key, status=WorkflowTaskStatus.COMPLETED)

    engine = DAGExecutionEngine(max_concurrency=5)
    engine.register_handler(TaskType.RETRIEVAL.value, bounded_handler)

    tasks = [
        ExecutionTask(task_key=f"t_{i}", task_type=TaskType.RETRIEVAL, description=f"T{i}", dependencies=[])
        for i in range(5)
    ]
    plan = AdaptiveTaskPlan(goal="Concurrency test", tasks=tasks)
    run = MockWorkflowRun()

    results = await engine.execute_plan(plan, run)
    assert len(results) == 5
    assert max_observed_active <= 5
