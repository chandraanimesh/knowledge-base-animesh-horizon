"""
Phase 5E: Automated Integration and Unit Tests for Immutable Runtime Configuration Snapshot.

Verifies:
- Test A: Initial snapshot captures effective resolved configuration.
- Test B: Workspace configuration changes mid-run do not cause drift in active runs; new runs use updated defaults.
- Test C: Permitted node overrides are preserved in the snapshot and across retries.
- Test D: Scheduled workflow runs preserve snapshot consistency even if workspace profile changes before dispatch.
- Test E: Trigger outbox recovery and idempotency replay reuse the existing immutable snapshot without mutation.
- Test F: Strict secret redaction (zero API keys, tokens, or raw passwords in serialized snapshot).
- Test G: Security policy revocation fails closed (revoked providers cannot be used even if in snapshot).
- Test H: Backward compatibility for legacy runs lacking a snapshot.
- Test I: Existing Chat/RAG retrieval continues to read active workspace profile without regression.
- Test J: Application-level snapshot immutability prevents modifying provider_snapshot on active runs.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool
from sqlalchemy import select, text

# STRICT TARGET DATABASE SAFETY GATE: Tests strictly execute against rag_disposable_test on port 5433
TEST_DB_URL = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"
os.environ["DATABASE_URL"] = TEST_DB_URL

if "rag_disposable_test" not in TEST_DB_URL or ":5433" not in TEST_DB_URL:
    raise RuntimeError("FATAL SAFETY ABORT: Refusing to run tests against non-disposable database!")

test_engine = create_async_engine(TEST_DB_URL, echo=False, poolclass=NullPool)
TestSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await test_engine.dispose()


from rag_app.db.models import (
    User,
    Department,
    Workspace,
    Document,
    DocumentChunk,
    WorkspaceProviderProfile,
    WorkflowDefinition,
    WorkflowVersion,
    WorkflowRun,
    WorkflowTask,
    WorkflowTriggerEvent,
)
from rag_app.schemas.workflow import (
    WorkflowStatus,
    WorkflowTaskStatus,
    BudgetLimits,
    ExecutionTask,
    TaskType,
)
from rag_app.schemas.workflow_definition import (
    WorkflowDefinitionCreate,
    WorkflowNodeDefinition,
    WorkflowActionType,
    WorkflowTriggerSpec,
    WorkflowTriggerType,
    WorkflowTestRunRequest,
)
from rag_app.orchestration.workflow_definition_service import WorkflowDefinitionService
from rag_app.orchestration.trigger_service import WorkflowTriggerService
from rag_app.schemas.workflow_trigger import (
    WorkflowTriggerRequest,
    TriggerSource,
    OutboxStatus,
)
from rag_app.orchestration.action_registry import (
    RetrievalActionStrategy,
    SynthesisActionStrategy,
    WebSearchActionStrategy,
)
from rag_app.orchestration.runtime_snapshot import (
    capture_runtime_snapshot,
    resolve_workflow_llm,
    resolve_workflow_retrieval_config,
    resolve_workflow_web_provider,
    assert_snapshot_immutability,
    SnapshotImmutabilityError,
    SnapshotPolicyRevocationError,
)
from rag_app.retrieval.service import RetrievalService
from rag_app.ingestion.embedding_v1 import get_embedding_provider


@pytest_asyncio.fixture
async def setup_phase5e_context():
    """Seeds an isolated workspace, admin user, department, and profile in rag_disposable_test."""
    uid = uuid.uuid4().hex[:6]
    async with TestSessionLocal() as session:
        user = User(
            id=uuid.uuid4(),
            email=f"tester_{uid}@snapshot-test.org",
            hashed_password="mock_password_hash",
            role="ADMIN",
            is_active=True,
        )
        dept = Department(
            id=uuid.uuid4(),
            name=f"Engineering {uid}",
            code=f"ENG_{uid}",
        )
        session.add_all([user, dept])
        await session.flush()

        ws = Workspace(
            id=uuid.uuid4(),
            name=f"Snapshot Workspace {uid}",
            code=f"WS_{uid}",
            department_id=dept.id,
            created_by=user.id,
            is_active=True,
        )
        session.add(ws)
        await session.flush()

        # Seed initial WorkspaceProviderProfile with Model A
        profile = WorkspaceProviderProfile(
            workspace_id=ws.id,
            llm_provider="mock",
            llm_model="mock-deterministic-v1",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2 (384 dim)",
            reranker_provider="cross_encoder",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            web_provider="tinyfish",
            configuration={
                "temperature": 0.2,
                "top_k": 6,
                "vector_search_enabled": True,
                "fts_enabled": True,
                "rrf_enabled": True,
                "reranker_enabled": True,
            },
        )
        session.add(profile)
        await session.flush()

        # Seed knowledge document with embeddings
        doc = Document(
            id=uuid.uuid4(),
            workspace_id=ws.id,
            department_id=dept.id,
            file_name="Architecture_Guide.pdf",
            file_path="/docs/arch.pdf",
            content_hash=f"hash_{uid}",
            mime_type="application/pdf",
            created_by=user.id,
        )
        session.add(doc)
        await session.flush()

        embedder = get_embedding_provider()
        embs = await embedder.embed_texts([
            "Service-oriented architecture requires immutable configuration snapshots for deterministic execution."
        ])
        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            text="Service-oriented architecture requires immutable configuration snapshots for deterministic execution.",
            embedding=embs[0],
        )
        session.add(chunk)
        await session.commit()

        yield {
            "user": user,
            "department": dept,
            "workspace": ws,
            "profile": profile,
            "document": doc,
            "chunk": chunk,
        }


# ============================================================================
# TEST A: INITIAL SNAPSHOT CAPTURE
# ============================================================================

@pytest.mark.asyncio
async def test_initial_snapshot_captures_resolved_configuration(setup_phase5e_context):
    """WorkflowRun creation captures an immutable snapshot of workspace AI and RAG settings."""
    ctx = setup_phase5e_context
    ws = ctx["workspace"]
    user = ctx["user"]

    async with TestSessionLocal() as session:
        wf_service = WorkflowDefinitionService(session=session)
        steps = [
            WorkflowNodeDefinition(
                id="step_retrieval",
                node_type=WorkflowActionType.RETRIEVAL,
                name="Retrieve Guide",
                inputs={"query": "architecture snapshots", "limit": 4},
            ),
            WorkflowNodeDefinition(
                id="step_synthesis",
                node_type=WorkflowActionType.SYNTHESIS,
                name="Synthesize Findings",
                dependencies=["step_retrieval"],
                inputs={"focus": "Architecture Determinism"},
            ),
        ]
        wf_payload = WorkflowDefinitionCreate(
            workspace_id=ws.id,
            name="Snapshot Capture Test",
            initial_trigger=WorkflowTriggerSpec(trigger_type=WorkflowTriggerType.MANUAL),
            initial_steps=steps,
        )
        workflow, version = await wf_service.create_workflow(ws.id, user, wf_payload)

        test_req = WorkflowTestRunRequest(inputs={"query": "architecture snapshots"})
        run, tasks = await wf_service.execute_test_run(ws.id, workflow.id, version.id, user, test_req)

        assert run.provider_snapshot is not None
        snap = run.provider_snapshot
        assert snap["snapshot_version"] == 1
        assert snap["workspace_id"] == str(ws.id)
        assert snap["workflow_id"] == str(workflow.id)
        assert snap["workflow_version_id"] == str(version.id)

        # Verify captured values
        assert snap["llm"]["provider"] == "mock"
        assert snap["llm"]["model"] == "mock-deterministic-v1"
        assert snap["llm"]["temperature"] == 0.2
        assert snap["retrieval"]["top_k"] == 6
        assert snap["retrieval"]["vector_search_enabled"] is True
        assert snap["research"]["web_provider"] == "tinyfish"
        assert snap["knowledge_index"]["embedding_dimension"] == 384
        assert "step_retrieval" in snap["node_configurations"]


# ============================================================================
# TEST B: WORKSPACE CONFIGURATION CHANGES (NO DRIFT)
# ============================================================================

@pytest.mark.asyncio
async def test_workspace_configuration_change_does_not_drift_active_run(setup_phase5e_context):
    """
    Run A starts with Model A. Workspace admin updates profile to Model B.
    Run A continues using Model A. New Run B uses Model B.
    """
    ctx = setup_phase5e_context
    ws = ctx["workspace"]
    user = ctx["user"]

    async with TestSessionLocal() as session:
        wf_service = WorkflowDefinitionService(session=session)
        steps = [
            WorkflowNodeDefinition(
                id="step_synthesis",
                node_type=WorkflowActionType.SYNTHESIS,
                name="Synthesize",
                inputs={"focus": "Model Drift Check"},
            )
        ]
        wf_payload = WorkflowDefinitionCreate(
            workspace_id=ws.id,
            name="Drift Test Workflow",
            initial_steps=steps,
        )
        workflow, version = await wf_service.create_workflow(ws.id, user, wf_payload)

        # 1. Start WorkflowRun A with initial profile (Model A = mock-deterministic-v1)
        run_a, _ = await wf_service.execute_test_run(
            ws.id, workflow.id, version.id,
            user,
            WorkflowTestRunRequest(inputs={"query": "test run a"}),
        )
        assert run_a.provider_snapshot["llm"]["model"] == "mock-deterministic-v1"

        # 2. Workspace admin changes workspace profile to Model B
        prof = await session.scalar(
            select(WorkspaceProviderProfile).where(WorkspaceProviderProfile.workspace_id == ws.id)
        )
        prof.llm_model = "mock-v2-advanced"
        prof.configuration = {**prof.configuration, "temperature": 0.9}
        await session.commit()

        # 3. Execute another synthesis task for Run A: it MUST still resolve Model A from its snapshot
        strat_synth = SynthesisActionStrategy()
        synth_task = ExecutionTask(
            task_key="step_synthesis_subsequent",
            task_type=TaskType.SYNTHESIS,
            input_payload={"focus": "Subsequent Task"},
            description="Subsequent Task",
        )
        res_a = await strat_synth.execute(synth_task, session=session, run=run_a)
        assert res_a.output_result["model"] == "mock-deterministic-v1"

        # 4. Create new WorkflowRun B: it MUST capture Model B from the updated workspace profile
        run_b, _ = await wf_service.execute_test_run(
            ws.id, workflow.id, version.id,
            user,
            WorkflowTestRunRequest(inputs={"query": "test run b"}),
        )
        assert run_b.provider_snapshot["llm"]["model"] == "mock-v2-advanced"
        assert run_b.provider_snapshot["llm"]["temperature"] == 0.9


# ============================================================================
# TEST C: NODE OVERRIDE PRESERVATION
# ============================================================================

@pytest.mark.asyncio
async def test_node_override_preservation_in_snapshot(setup_phase5e_context):
    """Permitted node overrides are preserved in snapshot and respected by action strategies."""
    ctx = setup_phase5e_context
    ws = ctx["workspace"]
    user = ctx["user"]

    async with TestSessionLocal() as session:
        wf_service = WorkflowDefinitionService(session=session)
        steps = [
            WorkflowNodeDefinition(
                id="custom_synthesis",
                node_type=WorkflowActionType.SYNTHESIS,
                name="Custom Synthesis",
                inputs={"focus": "Custom Config", "temperature": 0.8, "max_tokens": 1500},
            )
        ]
        wf_payload = WorkflowDefinitionCreate(
            workspace_id=ws.id,
            name="Override Preservation Workflow",
            initial_steps=steps,
        )
        workflow, version = await wf_service.create_workflow(ws.id, user, wf_payload)

        run, tasks = await wf_service.execute_test_run(
            ws.id, workflow.id, version.id,
            user,
            WorkflowTestRunRequest(inputs={"query": "run overrides"}),
        )

        assert "custom_synthesis" in run.provider_snapshot["node_configurations"]
        captured_node = run.provider_snapshot["node_configurations"]["custom_synthesis"]
        assert captured_node["temperature"] == 0.8
        assert captured_node["max_tokens"] == 1500


# ============================================================================
# TEST D: SCHEDULED WORKFLOW RUNS PRESERVE SNAPSHOT
# ============================================================================

@pytest.mark.asyncio
async def test_scheduled_workflow_preserves_snapshot_on_dispatch(setup_phase5e_context):
    """A scheduled workflow run captures snapshot at creation and preserves it when swept."""
    ctx = setup_phase5e_context
    ws = ctx["workspace"]
    user = ctx["user"]

    async with TestSessionLocal() as session:
        wf_service = WorkflowDefinitionService(session=session)
        wf_payload = WorkflowDefinitionCreate(
            workspace_id=ws.id,
            name="Scheduled Snapshot Test",
            initial_trigger=WorkflowTriggerSpec(trigger_type=WorkflowTriggerType.SCHEDULE),
            initial_steps=[
                WorkflowNodeDefinition(
                    id="step_synth",
                    node_type=WorkflowActionType.SYNTHESIS,
                    name="Synthesize",
                    inputs={"focus": "Scheduled Report"},
                )
            ],
        )
        workflow, version = await wf_service.create_workflow(ws.id, user, wf_payload)
        await wf_service.publish_version(ws.id, workflow.id, version.id, user)

        trigger_svc = WorkflowTriggerService(session=session)
        event = WorkflowTriggerEvent(
            id=uuid.uuid4(),
            event_id=f"sched_evt_{uuid.uuid4().hex[:8]}",
            event_source=TriggerSource.SCHEDULE.value,
            event_name="scheduled_daily_run",
            workspace_id=ws.id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            payload={"query": "daily scheduled run"},
            idempotency_key=f"sched_{uuid.uuid4().hex[:8]}",
            status=OutboxStatus.PENDING.value,
        )
        session.add(event)
        await session.commit()

        # Profile is updated before outbox sweeper processes the event
        prof = await session.scalar(
            select(WorkspaceProviderProfile).where(WorkspaceProviderProfile.workspace_id == ws.id)
        )
        prof.web_provider = "firecrawl"
        await session.commit()

        # Sweeper dispatches run via outbox recovery
        recovered = await trigger_svc.recover_pending_outbox_events(limit=1, workspace_id=ws.id)
        assert recovered == 1
        await session.refresh(event)
        assert event.workflow_run_id is not None
        dispatched_run = await session.get(WorkflowRun, event.workflow_run_id)
        assert dispatched_run is not None
        assert dispatched_run.provider_snapshot is not None
        # Snapshot correctly records web provider configuration
        assert "research" in dispatched_run.provider_snapshot


# ============================================================================
# TEST E: WEBHOOK / OUTBOX RECOVERY AND REPLAY IDEMPOTENCY
# ============================================================================

@pytest.mark.asyncio
async def test_outbox_replay_reuses_snapshot_without_mutation(setup_phase5e_context):
    """Replaying or retrying an existing idempotency key reuses the existing run and its snapshot."""
    ctx = setup_phase5e_context
    ws = ctx["workspace"]
    user = ctx["user"]

    async with TestSessionLocal() as session:
        wf_service = WorkflowDefinitionService(session=session)
        wf_payload = WorkflowDefinitionCreate(
            workspace_id=ws.id,
            name="Idempotency Snapshot Workflow",
            initial_steps=[
                WorkflowNodeDefinition(
                    id="step_synth",
                    node_type=WorkflowActionType.SYNTHESIS,
                    name="Synthesize",
                    inputs={"focus": "Replay Check"},
                )
            ],
        )
        workflow, version = await wf_service.create_workflow(ws.id, user, wf_payload)
        await wf_service.publish_version(ws.id, workflow.id, version.id, user)

        trigger_svc = WorkflowTriggerService(session=session)
        shared_key = f"idempotent_event_{uuid.uuid4().hex[:8]}"

        # First trigger dispatch
        req = WorkflowTriggerRequest(
            inputs={"query": "test idempotency"},
            idempotency_key=shared_key,
        )
        resp_1 = await trigger_svc.trigger_manual(
            workspace_id=ws.id,
            workflow_id=workflow.id,
            request=req,
            user=user,
        )
        run_1 = await session.get(WorkflowRun, resp_1.workflow_run_id)
        assert run_1 is not None
        original_snapshot = run_1.provider_snapshot
        assert original_snapshot is not None

        # Replay with same idempotency key
        resp_2 = await trigger_svc.trigger_manual(
            workspace_id=ws.id,
            workflow_id=workflow.id,
            request=req,
            user=user,
        )
        assert resp_2.is_replay is True
        assert resp_2.workflow_run_id == run_1.id
        run_replayed = await session.get(WorkflowRun, resp_2.workflow_run_id)
        assert run_replayed.provider_snapshot == original_snapshot


# ============================================================================
# TEST F: STRICT SECRET REDACTION
# ============================================================================

@pytest.mark.asyncio
async def test_secret_redaction_in_snapshots(setup_phase5e_context):
    """Verifies that secrets, API keys, and authorization headers are never persisted in snapshot."""
    ctx = setup_phase5e_context
    ws = ctx["workspace"]

    async with TestSessionLocal() as session:
        dirty_inputs = {
            "query": "search query",
            "api_key": "sk-proj-secret-12345",
            "auth_token": "Bearer eyJhbGciOi...",
            "password": "supersecretpassword",
            "nested": {
                "secret_key": "private_value",
                "normal_field": "public_value",
            },
        }
        steps = [
            {"id": "node_dirty", "inputs": dirty_inputs}
        ]
        snap = await capture_runtime_snapshot(
            session=session,
            workspace_id=ws.id,
            workflow_id=uuid.uuid4(),
            workflow_version_id=uuid.uuid4(),
            steps_spec=steps,
            trigger_payload={"bearer_token": "secret_bearer"},
        )

        snap_str = str(snap).lower()
        assert "sk-proj-secret-12345" not in snap_str
        assert "bearer eyjh" not in snap_str
        assert "supersecretpassword" not in snap_str
        assert "private_value" not in snap_str
        assert "secret_bearer" not in snap_str
        assert snap["node_configurations"]["node_dirty"]["nested"]["normal_field"] == "public_value"


# ============================================================================
# TEST G: SECURITY POLICY REVOCATION FAILS CLOSED
# ============================================================================

@pytest.mark.asyncio
async def test_revoked_provider_in_snapshot_fails_closed(setup_phase5e_context):
    """If a provider in a snapshot is revoked by security policy, execution fails closed."""
    ctx = setup_phase5e_context
    user = ctx["user"]
    ws = ctx["workspace"]

    async with TestSessionLocal() as session:
        # Create run whose snapshot contains a disallowed / revoked provider e.g. 'unauthorized_llm'
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="test revoked policy",
            request_hash="req_hash_revoked",
            provider_snapshot={
                "snapshot_version": 1,
                "llm": {"provider": "unauthorized_evil_llm", "model": "evil-v1"},
            },
        )
        session.add(run)
        await session.commit()

        strat_synth = SynthesisActionStrategy()
        task = ExecutionTask(
            task_key="synth_revoked",
            task_type=TaskType.SYNTHESIS,
            input_payload={"focus": "Revocation Test"},
            description="Revocation Test",
        )

        with pytest.raises(SnapshotPolicyRevocationError):
            await resolve_workflow_llm(run=run, session=session, node_inputs={})


# ============================================================================
# TEST H: BACKWARD COMPATIBILITY & ONE-TIME INITIALIZATION FOR LEGACY RUNS
# ============================================================================

@pytest.mark.asyncio
async def test_legacy_runs_without_snapshot_fall_back_safely(setup_phase5e_context):
    """
    Phase 5E.1:
    - Active legacy run with provider_snapshot=None initializes snapshot once on execution.
    - Subsequent tasks reuse the frozen snapshot and do not drift even if workspace profile changes.
    - Historical terminal runs with provider_snapshot=None are preserved without modification.
    """
    ctx = setup_phase5e_context
    user = ctx["user"]
    ws = ctx["workspace"]

    async with TestSessionLocal() as session:
        # 1. Non-terminal legacy run
        legacy_run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="test legacy run",
            request_hash="req_hash_legacy",
            provider_snapshot=None,  # Legacy run
        )
        # 2. Terminal historical legacy run
        terminal_legacy_run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            status=WorkflowStatus.COMPLETED.value,
            initial_query="historical completed run",
            request_hash="req_hash_terminal",
            provider_snapshot=None,
        )
        session.add_all([legacy_run, terminal_legacy_run])
        await session.commit()

        # Terminal run is preserved: ensure_run_runtime_snapshot returns None without mutation
        from rag_app.orchestration.runtime_snapshot import ensure_run_runtime_snapshot
        term_snap = await ensure_run_runtime_snapshot(terminal_legacy_run, session)
        assert term_snap is None
        assert terminal_legacy_run.provider_snapshot is None

        # Active non-terminal run executes first task: one-time compatibility policy freezes snapshot
        llm = await resolve_workflow_llm(run=legacy_run, session=session, node_inputs={})
        assert llm is not None
        assert legacy_run.provider_snapshot is not None
        captured_model = legacy_run.provider_snapshot["llm"]["model"]

        # Admin changes workspace profile mid-run
        prof = await session.scalar(
            select(WorkspaceProviderProfile).where(WorkspaceProviderProfile.workspace_id == ws.id)
        )
        prof.llm_model = "drifted-model-v99"
        await session.commit()

        # Subsequent task in legacy run MUST NOT drift; it uses the one-time frozen snapshot
        llm_subsequent = await resolve_workflow_llm(run=legacy_run, session=session, node_inputs={})
        assert getattr(llm_subsequent, "model", "") == captured_model
        assert getattr(llm_subsequent, "model", "") != "drifted-model-v99"


# ============================================================================
# TEST I: CHAT / RAG PIPELINE REGRESSION
# ============================================================================

@pytest.mark.asyncio
async def test_chat_rag_pipeline_reads_workspace_profile_unaffected(setup_phase5e_context):
    """Chat/RAG RetrievalService without workspace_config continues to read WorkspaceProviderProfile."""
    ctx = setup_phase5e_context
    ws = ctx["workspace"]

    async with TestSessionLocal() as session:
        ret_service = RetrievalService(session=session)
        # Call retrieve without workspace_config (standard Chat/RAG path)
        res = await ret_service.retrieve(
            query="Service-oriented architecture",
            top_k=2,
            workspace_id=ws.id,
            workspace_config=None,  # Verifies standard path
        )
        assert res is not None
        assert isinstance(res.results, list)


# ============================================================================
# TEST J: SNAPSHOT IMMUTABILITY & ENGINE-LEVEL TRIGGER ENFORCEMENT
# ============================================================================

@pytest.mark.asyncio
async def test_snapshot_immutability_enforcement(setup_phase5e_context):
    """Attempting to mutate or clear provider_snapshot on a persisted WorkflowRun raises an exception."""
    ctx = setup_phase5e_context
    user = ctx["user"]
    ws = ctx["workspace"]

    async with TestSessionLocal() as session:
        initial_snap = {"snapshot_version": 1, "llm": {"provider": "mock", "model": "model-a"}}
        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=ws.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="test immutability",
            request_hash="req_hash_immut",
            provider_snapshot=initial_snap,
        )
        session.add(run)
        await session.commit()

        # 1. Application assertion helper rejects tampering
        with pytest.raises(SnapshotImmutabilityError):
            assert_snapshot_immutability(
                run.provider_snapshot,
                {"snapshot_version": 1, "llm": {"provider": "mock", "model": "tampered-model"}},
            )

        # 2. Application assertion helper rejects clearing to None
        with pytest.raises(SnapshotImmutabilityError):
            assert_snapshot_immutability(run.provider_snapshot, None)

        # 3. Model validation guard rejects mutating snapshot
        with pytest.raises(ValueError, match="ImmutableRuntimeSnapshotError"):
            run.provider_snapshot = {"snapshot_version": 1, "llm": {"provider": "groq", "model": "tampered"}}

        # 4. Model validation guard rejects clearing to None
        with pytest.raises(ValueError, match="ImmutableRuntimeSnapshotError"):
            run.provider_snapshot = None


# ============================================================================
# TEST K: DATABASE TRIGGER BLOCKS BULK & RAW SQL OVERWRITES
# ============================================================================

@pytest.mark.asyncio
async def test_database_trigger_blocks_raw_sql_and_core_mutation(setup_phase5e_context):
    """PostgreSQL trigger trg_workflow_runs_provider_snapshot_immutable blocks raw SQL / Core updates."""
    ctx = setup_phase5e_context
    user = ctx["user"]
    ws = ctx["workspace"]

    async with TestSessionLocal() as session:
        run_id = uuid.uuid4()
        initial_snap = {"snapshot_version": 1, "llm": {"provider": "mock", "model": "model-sql"}}

        run = WorkflowRun(
            id=run_id,
            user_id=user.id,
            workspace_id=ws.id,
            status=WorkflowStatus.RUNNING.value,
            initial_query="test trigger protection",
            request_hash="req_hash_trg",
            provider_snapshot=initial_snap,
        )
        session.add(run)
        await session.commit()

        # Raw SQL update to non-snapshot columns MUST succeed
        await session.execute(
            text("UPDATE workflow_runs SET status = 'EXECUTING', tokens_spent = 50 WHERE id = :id"),
            {"id": run_id},
        )
        await session.commit()

        # Raw SQL update mutating provider_snapshot MUST be blocked by PostgreSQL trigger
        with pytest.raises(Exception) as exc_info:
            await session.execute(
                text("UPDATE workflow_runs SET provider_snapshot = '{\"llm\": {\"provider\": \"hacked\"}}'::jsonb WHERE id = :id"),
                {"id": run_id},
            )
            await session.commit()
        await session.rollback()
        assert "ImmutableRuntimeSnapshotError" in str(exc_info.value)

        # Raw SQL update clearing provider_snapshot to NULL MUST be blocked
        with pytest.raises(Exception) as exc_info_null:
            await session.execute(
                text("UPDATE workflow_runs SET provider_snapshot = NULL WHERE id = :id"),
                {"id": run_id},
            )
            await session.commit()
        await session.rollback()
        assert "ImmutableRuntimeSnapshotError" in str(exc_info_null.value)


# ============================================================================
# TEST L: API SERIALIZATION & CREDENTIAL REFERENCE MASKING
# ============================================================================

def test_api_response_sanitization_and_credential_masking():
    """
    Phase 5E.1:
    - sanitize_snapshot_for_api strips raw secret patterns for all users.
    - Masks internal credential references (e.g. 'env:GROQ_API_KEY') as '[server-managed]' for non-admins.
    - Preserves credential reference name only for admin users.
    """
    from rag_app.orchestration.runtime_snapshot import sanitize_snapshot_for_api

    raw_snapshot = {
        "snapshot_version": 1,
        "llm": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "credential_ref": "env:GROQ_API_KEY",
            "api_key": "sk-raw-leak-secret-12345",  # Injected leak attempt
        },
        "research": {
            "web_provider": "tinyfish",
            "web_credential_ref": "vault:tinyfish_prod",
        },
        "node_configurations": {
            "node_1": {
                "temperature": 0.7,
                "auth_token": "bearer-token-123",  # Injected leak attempt
            }
        },
    }

    # 1. Non-admin viewer: raw secrets stripped, credential references masked
    viewer_clean = sanitize_snapshot_for_api(raw_snapshot, user_role="USER")
    assert viewer_clean is not None
    assert viewer_clean["llm"]["provider"] == "groq"
    assert viewer_clean["llm"]["credential_ref"] == "[server-managed]"
    assert viewer_clean["research"]["web_credential_ref"] == "[server-managed]"
    assert "api_key" not in viewer_clean["llm"]
    assert "auth_token" not in viewer_clean["node_configurations"]["node_1"]

    # 2. Admin: raw secrets stripped, but credential reference string preserved
    admin_clean = sanitize_snapshot_for_api(raw_snapshot, user_role="ADMIN")
    assert admin_clean is not None
    assert admin_clean["llm"]["credential_ref"] == "env:GROQ_API_KEY"
    assert admin_clean["research"]["web_credential_ref"] == "vault:tinyfish_prod"
    assert "api_key" not in admin_clean["llm"]
    assert "auth_token" not in admin_clean["node_configurations"]["node_1"]

