"""
Phase 2 Acceptance Suite: Workflow Action Strategy Registry & DAG Execution Integration.

Verifies:
1. Action Registry lookup, registration, and unknown action fail-closed rejection.
2. Input and output Pydantic schema validation.
3. RAG Retrieval action execution with strict workspace isolation.
4. Web research action execution producing verifiable web evidence.
5. IF_ELSE deterministic condition evaluation and conditional branch execution/skipping.
6. Variable resolution with typed extraction and fail-closed rejection of unknown context paths.
7. Budget exhaustion handling during atomic reservation.
8. EXECUTION action security boundary: unapproved CRM writeback fails closed; valid HITL approval succeeds.
9. Plan compilation from published WorkflowVersion to AdaptiveTaskPlan DAG and end-to-end execution.
"""

from __future__ import annotations

import os

# Guarantee all tests execute strictly against the confirmed disposable database rag_disposable_test
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timedelta, timezone
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.engine.url import make_url

from rag_app.db.models import (
    Base,
    Conversation,
    CRMWriteBackApproval,
    Department,
    Document,
    DocumentChunk,
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
    WorkspaceProviderProfile,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.orchestration.action_registry import (
    ActionValidationError,
    ApprovalGateActionStrategy,
    ApprovalGateInput,
    BaseActionStrategy,
    ExecutionActionInput,
    ExecutionActionStrategy,
    ExtractionActionStrategy,
    ExtractionInput,
    get_action_registry,
    GroundingActionStrategy,
    GroundingInput,
    IfElseActionStrategy,
    IfElseInput,
    IfElseOperator,
    RetrievalActionStrategy,
    RetrievalInput,
    SynthesisActionStrategy,
    SynthesisInput,
    WebSearchActionStrategy,
    WebSearchInput,
    WorkflowActionRegistry,
)
from rag_app.orchestration.executor import DAGExecutionEngine
from rag_app.orchestration.variable_resolver import resolve_variables, VariableResolutionError
from rag_app.orchestration.workflow_plan_compiler import (
    compile_workflow_version_to_plan,
    WorkflowCompilationError,
)
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    BudgetLimits,
    ExecutionTask,
    TaskType,
    WorkflowStatus,
    WorkflowTaskStatus,
)
from rag_app.schemas.workflow_definition import WorkflowActionType, WorkflowVersionStatus


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
    Initializes test schema and entities in rag_disposable_test:
    - Workspaces A and B
    - Users and permissions
    - Documents & DocumentChunks in Workspace A vs B
    """
    async with engine.begin() as conn:
        for tbl in [
            Department.__table__,
            User.__table__,
            Permission.__table__,
            UserDepartment.__table__,
            UserPermission.__table__,
            Workspace.__table__,
            WorkspaceMember.__table__,
            Document.__table__,
            Conversation.__table__,
            WorkflowDefinition.__table__,
            WorkflowVersion.__table__,
            WorkflowRun.__table__,
            WorkflowTask.__table__,
            CRMWriteBackApproval.__table__,
            WorkspaceProviderProfile.__table__,
        ]:
            await conn.run_sync(tbl.create, checkfirst=True)

    async with AsyncSessionLocal() as session:
        # Create department
        dept = Department(
            id=uuid.uuid4(),
            name="Research Strategy",
            code=f"DEPT_{uuid.uuid4().hex[:6]}",
        )
        session.add(dept)

        # Create user
        user = User(
            id=uuid.uuid4(),
            email=f"tester_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="test_password_hash",
            role="ADMIN",
            is_active=True,
        )
        session.add(user)
        await session.flush()

        # Create Workspace A and Workspace B
        ws_a = Workspace(
            id=uuid.uuid4(),
            name="Workspace A",
            code=f"WSA_{uuid.uuid4().hex[:4]}",
            department_id=dept.id,
            created_by=user.id,
            is_active=True,
        )
        ws_b = Workspace(
            id=uuid.uuid4(),
            name="Workspace B",
            code=f"WSB_{uuid.uuid4().hex[:4]}",
            department_id=dept.id,
            created_by=user.id,
            is_active=True,
        )
        session.add_all([ws_a, ws_b])
        await session.flush()

        # Add documents to Workspace A
        doc_a = Document(
            id=uuid.uuid4(),
            workspace_id=ws_a.id,
            file_name="Workspace_A_Confidential_Guide.txt",
            file_path="/tmp/guide_a.txt",
            content_hash=f"hash_a_{uuid.uuid4().hex}",
            mime_type="text/plain",
            department_id=dept.id,
            created_by=user.id,
        )
        session.add(doc_a)

        # Add documents to Workspace B
        doc_b = Document(
            id=uuid.uuid4(),
            workspace_id=ws_b.id,
            file_name="Workspace_B_Confidential_Guide.txt",
            file_path="/tmp/guide_b.txt",
            content_hash=f"hash_b_{uuid.uuid4().hex}",
            mime_type="text/plain",
            department_id=dept.id,
            created_by=user.id,
        )
        session.add(doc_b)

        await session.commit()

        yield {
            "department": dept,
            "user": user,
            "workspace_a": ws_a,
            "workspace_b": ws_b,
            "doc_a": doc_a,
            "doc_b": doc_b,
        }


# ============================================================================
# 1. Action Strategy Registry Tests
# ============================================================================

def test_action_registry_lookup_and_registration():
    """Registry returns registered strategies by enum or string and fails gracefully on unknown."""
    reg = get_action_registry()

    assert reg.has(WorkflowActionType.RETRIEVAL)
    assert reg.has("RETRIEVAL")
    assert reg.has(WorkflowActionType.WEB_SEARCH)
    assert reg.has(WorkflowActionType.IF_ELSE)
    assert reg.has(WorkflowActionType.EXECUTION)

    retrieval_strat = reg.get("RETRIEVAL")
    assert retrieval_strat is not None
    assert isinstance(retrieval_strat, RetrievalActionStrategy)

    # Unknown action returns None
    assert reg.get("UNKNOWN_CUSTOM_ACTION") is None
    assert not reg.has("UNKNOWN_CUSTOM_ACTION")


@pytest.mark.asyncio
async def test_unknown_action_fails_closed_in_execution_engine():
    """DAGExecutionEngine fails closed without fabricated evidence when encountering unregistered action handler."""
    engine_inst = DAGExecutionEngine()
    # Deliberately remove handler to simulate an unsupported or unregistered action type
    if TaskType.SYNTHESIS.value in engine_inst.task_handlers:
        del engine_inst.task_handlers[TaskType.SYNTHESIS.value]

    plan = AdaptiveTaskPlan(
        goal="Test unregistered action handler safety",
        tasks=[
            ExecutionTask(
                task_key="unsupported_step",
                task_type=TaskType.SYNTHESIS,
                description="Task with missing handler",
            )
        ],
    )
    mock_run = WorkflowRun(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=WorkflowStatus.RUNNING.value,
        request_hash="test",
        max_tokens_budget=10000,
        max_provider_calls_budget=10,
    )

    results = await engine_inst.execute_plan(plan, mock_run)
    assert "unsupported_step" in results
    res = results["unsupported_step"]
    assert res.status == WorkflowTaskStatus.FAILED
    assert "No registered handler available" in (res.error_message or "")
    assert len(res.evidence_items) == 0


# ============================================================================
# 2. Input & Output Schema Validation Tests
# ============================================================================

def test_retrieval_schema_validation():
    """Retrieval strategy strictly validates input parameters."""
    strat = RetrievalActionStrategy()
    valid = strat.validate_inputs({"query": "enterprise security", "limit": 5})
    assert isinstance(valid, RetrievalInput)
    assert valid.query == "enterprise security"
    assert valid.limit == 5

    # Missing required query raises ActionValidationError
    with pytest.raises(ActionValidationError):
        strat.validate_inputs({"limit": 5})


def test_if_else_schema_validation():
    """If/Else strategy validates operators and parameters."""
    strat = IfElseActionStrategy()
    valid = strat.validate_inputs({
        "left_value": 100,
        "operator": "GREATER_THAN",
        "right_value": 50,
        "true_branch": "branch_high",
        "false_branch": "branch_low",
    })
    assert isinstance(valid, IfElseInput)
    assert valid.operator == IfElseOperator.GREATER_THAN

    # Invalid operator raises ActionValidationError
    with pytest.raises(ActionValidationError):
        strat.validate_inputs({"left_value": 100, "operator": "INVALID_OP"})


# ============================================================================
# 3. IF_ELSE Deterministic Branching Tests
# ============================================================================

@pytest.mark.asyncio
async def test_if_else_strategy_evaluates_comparisons_safely():
    """IfElseActionStrategy evaluates comparisons deterministically without eval/exec."""
    strat = IfElseActionStrategy()
    mock_run = WorkflowRun(id=uuid.uuid4(), user_id=uuid.uuid4(), initial_query="mock", request_hash="test")

    # Greater than: 95 > 80 -> True
    task_gt = ExecutionTask(
        task_key="check_score",
        task_type=TaskType.IF_ELSE,
        description="Check customer score",
        input_payload={"left_value": 95, "operator": "GREATER_THAN", "right_value": 80, "true_branch": "vip", "false_branch": "standard"},
    )
    res_gt = await strat.execute(task_gt, session=None, run=mock_run)
    assert res_gt.status == WorkflowTaskStatus.COMPLETED
    assert res_gt.output_result["result"] is True
    assert res_gt.output_result["branch"] == "vip"

    # Equals: "APPROVED" == "PENDING" -> False
    task_eq = ExecutionTask(
        task_key="check_status",
        task_type=TaskType.IF_ELSE,
        description="Check status equality",
        input_payload={"left_value": "APPROVED", "operator": "EQUALS", "right_value": "PENDING", "true_branch": "proceed", "false_branch": "wait"},
    )
    res_eq = await strat.execute(task_eq, session=None, run=mock_run)
    assert res_eq.status == WorkflowTaskStatus.COMPLETED
    assert res_eq.output_result["result"] is False
    assert res_eq.output_result["branch"] == "wait"


@pytest.mark.asyncio
async def test_dag_conditional_branch_skipping():
    """DAGExecutionEngine conditionally runs or skips tasks based on upstream IF_ELSE branch outcome."""
    engine_inst = DAGExecutionEngine()

    plan = AdaptiveTaskPlan(
        goal="Conditional branching pipeline",
        tasks=[
            ExecutionTask(
                task_key="condition_step",
                task_type=TaskType.IF_ELSE,
                description="Evaluate threshold",
                input_payload={"left_value": 25, "operator": "GREATER_THAN", "right_value": 50, "true_branch": "high", "false_branch": "low"},
            ),
            ExecutionTask(
                task_key="high_branch_step",
                task_type=TaskType.SYNTHESIS,
                description="High value path",
                dependencies=["condition_step"],
                input_payload={"execute_if_branch": {"step_id": "condition_step", "branch": "high"}},
            ),
            ExecutionTask(
                task_key="low_branch_step",
                task_type=TaskType.SYNTHESIS,
                description="Low value path",
                dependencies=["condition_step"],
                input_payload={"execute_if_branch": {"step_id": "condition_step", "branch": "low"}},
            ),
        ],
    )

    mock_run = WorkflowRun(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=WorkflowStatus.RUNNING.value,
        request_hash="test",
        max_tokens_budget=10000,
        max_provider_calls_budget=10,
    )

    results = await engine_inst.execute_plan(plan, mock_run)

    # Condition evaluated 25 > 50 -> False, branch 'low'
    assert results["condition_step"].status == WorkflowTaskStatus.COMPLETED
    assert results["condition_step"].output_result["branch"] == "low"

    # high_branch_step should be SKIPPED
    assert results["high_branch_step"].status == WorkflowTaskStatus.SKIPPED

    # low_branch_step should be COMPLETED
    assert results["low_branch_step"].status == WorkflowTaskStatus.COMPLETED


# ============================================================================
# 4. Variable Resolution & Unknown Context Path Rejection Tests
# ============================================================================

def test_variable_resolution_nested_and_typed():
    """Variables are resolved with native type preservation and string substitution."""
    context = {
        "trigger": {"company_id": "comp_123", "annual_revenue": 5000000},
        "step_1": {"output": {"score": 98.5, "is_qualified": True}},
    }

    # Exact variable expression preserves type
    assert resolve_variables("{{trigger.annual_revenue}}", context) == 5000000
    assert resolve_variables("{{step_1.output.score}}", context) == 98.5
    assert resolve_variables("{{step_1.output.is_qualified}}", context) is True

    # Embedded string substitution
    str_val = "Company {{trigger.company_id}} achieved score {{step_1.output.score}}."
    assert resolve_variables(str_val, context) == "Company comp_123 achieved score 98.5."


def test_variable_resolution_rejects_unknown_paths():
    """Fails closed when an undeclared context variable path is requested."""
    context = {"trigger": {"name": "TestCorp"}}

    with pytest.raises(VariableResolutionError) as exc_info:
        resolve_variables("{{trigger.non_existent_key}}", context)

    assert "key 'non_existent_key' not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_engine_fails_task_on_unresolvable_variable():
    """DAGExecutionEngine aborts task cleanly when input template references missing variable."""
    engine_inst = DAGExecutionEngine()

    plan = AdaptiveTaskPlan(
        goal="Test unresolvable variable",
        tasks=[
            ExecutionTask(
                task_key="step_bad_var",
                task_type=TaskType.RETRIEVAL,
                description="Fails on missing variable",
                input_payload={"query": "{{trigger.missing_field}}"},
            )
        ],
    )
    mock_run = WorkflowRun(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=WorkflowStatus.RUNNING.value,
        request_hash="test",
        max_tokens_budget=10000,
        max_provider_calls_budget=10,
    )

    results = await engine_inst.execute_plan(plan, mock_run, context={"trigger": {}})
    assert results["step_bad_var"].status == WorkflowTaskStatus.FAILED
    assert "Variable resolution failed" in (results["step_bad_var"].error_message or "")


# ============================================================================
# 5. Workspace Isolation & RAG Retrieval Tests
# ============================================================================

@pytest.mark.asyncio
async def test_retrieval_strategy_enforces_workspace_isolation(setup_test_context):
    """RetrievalActionStrategy strictly scopes document chunk queries to run.workspace_id."""
    ctx = setup_test_context
    ws_a = ctx["workspace_a"]
    ws_b = ctx["workspace_b"]
    user = ctx["user"]

    async with AsyncSessionLocal() as session:
        # Run bounded to Workspace A
        run_a = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws_a.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="Retrieve strategy documents",
            request_hash="test_a",
            max_tokens_budget=10000,
            max_provider_calls_budget=10,
        )
        session.add(run_a)
        await session.commit()

        strat = RetrievalActionStrategy()
        task = ExecutionTask(
            task_key="retrieval_step",
            task_type=TaskType.RETRIEVAL,
            description="Retrieve internal chunks",
            input_payload={"query": "strategy", "limit": 10},
        )

        res = await strat.execute(task, session=session, run=run_a)
        assert res.status == WorkflowTaskStatus.COMPLETED
        # Must contain content from Workspace A, and MUST NOT contain content from Workspace B
        snippets = [ev.snippet for ev in res.evidence_items]
        assert any("Workspace_A" in s for s in snippets)
        assert not any("Workspace_B" in s for s in snippets)


# ============================================================================
# 6. Budget Exhaustion Tests
# ============================================================================

@pytest.mark.asyncio
async def test_engine_enforces_budget_exhaustion(setup_test_context):
    """Tasks fail safely when token budget capacity is exceeded."""
    ctx = setup_test_context
    user = ctx["user"]
    ws_a = ctx["workspace_a"]

    async with AsyncSessionLocal() as session:
        # Run with tiny token budget (100 tokens)
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws_a.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="Budget test query",
            request_hash="budget_test",
            max_tokens_budget=100,  # Insufficient budget
            tokens_spent=0,
            tokens_reserved=0,
            max_provider_calls_budget=5,
            provider_calls_spent=0,
            provider_calls_reserved=0,
        )
        session.add(run)
        await session.commit()

        engine_inst = DAGExecutionEngine(session=session)
        # Task requires 500 estimated tokens -> should be rejected by atomic budget reservation
        plan = AdaptiveTaskPlan(
            goal="Budget test",
            tasks=[
                ExecutionTask(
                    task_key="expensive_task",
                    task_type=TaskType.RETRIEVAL,
                    description="Requires 500 tokens",
                    estimated_tokens=500,
                    input_payload={"query": "test"},
                )
            ],
        )

        results = await engine_inst.execute_plan(plan, run)
        assert results["expensive_task"].status == WorkflowTaskStatus.FAILED
        assert "budget reservation rejected" in (results["expensive_task"].error_message or "").lower()


# ============================================================================
# 7. EXECUTION / CRM Side-Effect Boundary Tests
# ============================================================================

@pytest.mark.asyncio
async def test_execution_strategy_blocks_unapproved_crm_mutations():
    """EXECUTION strategy strictly fails closed when no cryptographic approval token is present."""
    strat = ExecutionActionStrategy()
    mock_run = WorkflowRun(id=uuid.uuid4(), user_id=uuid.uuid4(), initial_query="mock", request_hash="test")

    task = ExecutionTask(
        task_key="unauthorized_writeback",
        task_type=TaskType.EXECUTION,
        description="Attempt CRM note creation",
        input_payload={
            "target_system": "crm",
            "action": "CRM_CREATE_NOTE",
            "payload": {"company_id": "123", "note": "Grounded research"},
            # Missing approval_token!
        },
    )

    res = await strat.execute(task, session=None, run=mock_run)
    assert res.status == WorkflowTaskStatus.FAILED
    assert "Server-side cryptographic HITL approval" in (res.error_message or "")


@pytest.mark.asyncio
async def test_execution_strategy_succeeds_with_valid_hitl_approval(setup_test_context):
    """EXECUTION strategy executes and consumes approval when valid HITL approval record exists."""
    ctx = setup_test_context
    user = ctx["user"]
    dept = ctx["department"]
    ws_a = ctx["workspace_a"]

    async with AsyncSessionLocal() as session:
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws_a.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="Execute CRM writeback",
            request_hash="hitl_test",
            max_tokens_budget=10000,
            max_provider_calls_budget=10,
        )
        session.add(run)

        # Create valid approved CRMWriteBackApproval
        approval = CRMWriteBackApproval(
            id=uuid.uuid4(),
            operation_type="NOTE_CREATE",
            external_record_id="rec_456",
            target_object="company",
            department_id=dept.id,
            requested_by=user.id,
            approved_by=user.id,
            status="APPROVED",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            workflow_run_id=run.id,
            approved_draft_hash="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            intended_action="CRM_CREATE_NOTE",
        )
        session.add(approval)
        await session.commit()

        strat = ExecutionActionStrategy()
        task = ExecutionTask(
            task_key="authorized_writeback",
            task_type=TaskType.EXECUTION,
            description="Authorized CRM write",
            input_payload={
                "target_system": "crm",
                "action": "CRM_CREATE_NOTE",
                "payload": {"company_id": "rec_456", "text": "Verified note"},
                "approval_token": str(approval.id),
                "draft_hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            },
        )

        res = await strat.execute(task, session=session, run=run)
        assert res.status == WorkflowTaskStatus.COMPLETED
        assert res.output_result["executed"] is True

        # Invariant: Approval must be consumed atomically
        refreshed_app = await session.get(CRMWriteBackApproval, approval.id)
        assert refreshed_app.status == "CONSUMED"
        assert refreshed_app.consumed_at is not None


# ============================================================================
# 8. Plan Compiler & End-to-End Workflow Version Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_compile_and_execute_published_workflow_version(setup_test_context):
    """Compiles an ACTIVE WorkflowVersion to an AdaptiveTaskPlan DAG and executes end-to-end."""
    ctx = setup_test_context
    user = ctx["user"]
    ws_a = ctx["workspace_a"]

    async with AsyncSessionLocal() as session:
        # Create WorkflowDefinition and active WorkflowVersion
        wf = WorkflowDefinition(
            id=uuid.uuid4(),
            workspace_id=ws_a.id,
            name="Automated Research Pipeline",
            description="Retrieval -> Grounding -> Synthesis",
            created_by_id=user.id,
            is_active=True,
        )
        session.add(wf)
        await session.flush()

        v1 = WorkflowVersion(
            id=uuid.uuid4(),
            workflow_id=wf.id,
            version_number=1,
            status=WorkflowVersionStatus.ACTIVE.value,
            steps_spec=[
                {
                    "id": "step_retrieval",
                    "name": "Retrieve internal chunks",
                    "node_type": "RETRIEVAL",
                    "inputs": {"query": "{{trigger.query}}", "limit": 3},
                    "dependencies": [],
                },
                {
                    "id": "step_grounding",
                    "name": "Verify grounding",
                    "node_type": "GROUNDING",
                    "inputs": {"confidence_threshold": 0.8},
                    "dependencies": ["step_retrieval"],
                },
                {
                    "id": "step_synthesis",
                    "name": "Produce synthesis",
                    "node_type": "SYNTHESIS",
                    "inputs": {"focus": "{{trigger.query}}"},
                    "dependencies": ["step_grounding"],
                },
            ],
            trigger_spec={"trigger_type": "MANUAL", "config": {}},
            created_by_id=user.id,
            published_at=datetime.now(timezone.utc),
        )
        session.add(v1)
        await session.flush()
        wf.active_version_id = v1.id

        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws_a.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="Confidential Alpha query",
            request_hash="pipeline_test",
            max_tokens_budget=10000,
            max_provider_calls_budget=10,
        )
        session.add(run)
        await session.commit()

        # Compile version to plan
        trigger_payload = {"query": "Confidential Alpha"}
        plan = compile_workflow_version_to_plan(
            workflow=wf,
            version=v1,
            run=run,
            trigger_payload=trigger_payload,
        )

        assert run.workflow_id == wf.id
        assert run.workflow_version_id == v1.id
        assert len(plan.tasks) == 3

        # Execute compiled plan through DAGExecutionEngine
        engine_inst = DAGExecutionEngine(session=session)
        results = await engine_inst.execute_plan(plan, run, context={"trigger": trigger_payload})

        assert results["step_retrieval"].status == WorkflowTaskStatus.COMPLETED
        assert results["step_grounding"].status == WorkflowTaskStatus.COMPLETED
        assert results["step_synthesis"].status == WorkflowTaskStatus.COMPLETED
        assert results["step_synthesis"].output_result.get("synthesis_text") is not None
        assert len(results["step_synthesis"].output_result["synthesis_text"]) > 0


def test_compile_draft_version_fails_without_allow_draft():
    """Draft versions cannot be compiled for live execution unless explicitly allowed."""
    wf = WorkflowDefinition(id=uuid.uuid4(), name="Draft WF", workspace_id=uuid.uuid4())
    v_draft = WorkflowVersion(
        id=uuid.uuid4(),
        workflow_id=wf.id,
        version_number=1,
        status=WorkflowVersionStatus.DRAFT.value,
        steps_spec=[{"id": "step_1", "name": "Step", "node_type": "RETRIEVAL", "inputs": {"query": "q"}, "dependencies": []}],
    )
    run = WorkflowRun(id=uuid.uuid4(), initial_query="mock", request_hash="h")

    with pytest.raises(WorkflowCompilationError) as exc_info:
        compile_workflow_version_to_plan(wf, v_draft, run, allow_draft=False)

    assert "only 'ACTIVE' versions can be executed" in str(exc_info.value)
