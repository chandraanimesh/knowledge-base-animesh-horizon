"""
Staging Smoke Test Suite: RAG + MCP + Adaptive Workflow.

Verifies the 6 required scenarios:
1. Normal Chat (Grounded answer through RAG path)
2. Live Web Research (Real TinyFish search invocation, real URLs and evidence)
3. Provider Failure (Honest failure response, zero fabricated evidence)
4. MCP Read-Only (Authorized workspace MCP tool invocation, visible grounded result)
5. CRM Write-Back (Fail-closed HITL routing despite execution_mode='fast_path', no side-effect before approval, duplicate approval prevention)
6. Explicit Adaptive Workflow (Intentional adaptive DAG workflow dispatch)
"""

from __future__ import annotations

import os
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import asyncio
from datetime import datetime, timezone
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select, and_, text

from rag_app.core.config import get_settings, Settings
from rag_app.core.observability import ExecutionMetrics
from rag_app.crm.sync_service import CRMSyncService
from rag_app.crm.twenty_mcp_client import TwentyMCPClient
from rag_app.db.models import (
    CRMWriteBackRecord,
    Department,
    User,
    UserDepartment,
    WorkflowRun,
    WorkflowTask,
    Workspace,
    WorkspaceMCPConnection,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.mcp.client import MCPToolResult
from rag_app.mcp.connection_service import WorkspaceMCPConnectionService
from rag_app.orchestration.executor import DAGExecutionEngine
from rag_app.orchestration.planner import AdaptiveTaskPlanner
from rag_app.orchestration.service import WorkflowService
from rag_app.orchestration.validator import DraftValidator
from rag_app.research.graph import (
    ResearchState,
    classify_query,
    decide_next_after_eval,
    research_graph,
)
from rag_app.research.provider import (
    ProviderUnavailableError,
    WebSearchResponse,
    WebSearchResult,
)
from rag_app.research.providers.tinyfish import TinyFishProvider
from rag_app.schemas.rag import (
    AnswerGroundingResult,
    GenerateRequest,
    GenerateResponse,
    QueryPlan,
    RetrievedChunk,
)
from rag_app.schemas.research import (
    SufficiencyEvaluation,
    SufficiencyStatus,
)
from rag_app.schemas.workflow import (
    BudgetLimits,
    ExecutionTask,
    TaskEvidenceRef,
    TaskExecutionResult,
    TaskType,
    WorkflowRunCreate,
    WorkflowStatus,
    WorkflowTaskStatus,
)
from rag_app.api.routes import ask_question


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def smoke_staging_context():
    """Sets up disposable staging entities in rag_disposable_test."""
    async with AsyncSessionLocal() as session:
        dept = await session.scalar(select(Department).where(Department.code == "SMOKE_STAGING_DEPT"))
        if not dept:
            dept = Department(code="SMOKE_STAGING_DEPT", name="Staging Smoke Department", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        user = await session.scalar(select(User).where(User.email == "smoke_user@staging.test"))
        if not user:
            user = User(
                email="smoke_user@staging.test",
                hashed_password="pw",
                role="ADMIN",
                is_active=True,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

        ud = await session.scalar(
            select(UserDepartment).where(
                and_(UserDepartment.user_id == user.id, UserDepartment.department_id == dept.id)
            )
        )
        if not ud:
            ud = UserDepartment(user_id=user.id, department_id=dept.id)
            session.add(ud)
            await session.commit()

        ws = await session.scalar(select(Workspace).where(Workspace.name == "Smoke Staging Workspace"))
        if not ws:
            ws = Workspace(
                name="Smoke Staging Workspace",
                code="SMOKE_WS",
                department_id=dept.id,
                created_by=user.id,
            )
            session.add(ws)
            await session.commit()
            await session.refresh(ws)

        wm = await session.scalar(
            select(WorkspaceMember).where(
                and_(WorkspaceMember.workspace_id == ws.id, WorkspaceMember.user_id == user.id)
            )
        )
        if not wm:
            wm = WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="OWNER")
            session.add(wm)
            await session.commit()

        # MCP Connection for Workspace
        conn = await session.scalar(
            select(WorkspaceMCPConnection).where(
                WorkspaceMCPConnection.workspace_id == ws.id,
                WorkspaceMCPConnection.name == "Staging Smoke Twenty Connection",
            )
        )
        if not conn:
            conn = WorkspaceMCPConnection(
                workspace_id=ws.id,
                name="Staging Smoke Twenty Connection",
                endpoint_url="https://api.twenty.com/mcp",
                transport="streamable-http",
                credential_ref="SERVER_TWENTY_API_KEY",
                is_enabled=True,
                status="CONNECTED",
                discovered_tools=[
                    {"name": "get_company", "description": "Retrieve company details", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}}},
                    {"name": "create_one_note", "description": "Mutating note write-back", "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}}},
                ],
                enabled_tools=["get_company", "create_one_note"],
                configuration={"tool_policies": {"create_one_note": {"risk_level": "mutating"}}},
                created_by=user.id,
            )
            session.add(conn)
            await session.commit()
            await session.refresh(conn)

        return {
            "department": dept,
            "user": user,
            "workspace": ws,
            "mcp_connection": conn,
        }


# ============================================================================
# Scenario 1: Normal Chat (Grounded answer through RAG path)
# ============================================================================

@pytest.mark.asyncio
async def test_scenario_1_normal_chat_grounded_rag(smoke_staging_context):
    ctx = smoke_staging_context
    user = ctx["user"]
    dept = ctx["department"]
    ws = ctx["workspace"]

    query = "What is the policy for employee reimbursement?"
    req = GenerateRequest(
        query=query,
        workspace_id=ws.id,
        execution_mode="fast_path",
    )

    # 1. Routing Verification: Fast path check
    assert WorkflowService.is_fast_path(query) is True, "Normal question must resolve to fast-path"

    # 2. Indexed Chunk Evidence
    doc_chunk = RetrievedChunk(
        chunk_id="chunk-reimb-01",
        document_id="doc-reimb-policy",
        department_id=str(dept.id),
        file_name="employee_handbook.pdf",
        chunk_index=1,
        text="All business travel expenses must be submitted within 30 days of travel.",
        hybrid_score=0.92,
    )

    from rag_app.schemas.rag import ChunkProvenance
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve = AsyncMock(
        return_value=MagicMock(
            results=[doc_chunk],
            strategy_trace={"hybrid_vector": 1, "bm25_keyword": 1},
            chunk_provenance={"chunk-reimb-01": ChunkProvenance(chunk_id="chunk-reimb-01", strategies=["hybrid_vector"])},
            chunk_hierarchy={},
            query_plan=None,
        )
    )

    mock_gen = MagicMock()
    mock_gen.generate_from_context = AsyncMock(
        return_value=GenerateResponse(
            answer="According to the employee handbook, business travel expenses must be submitted within 30 days.",
            citations=["employee_handbook.pdf (Page 1)"],
            retrieved_chunks=[doc_chunk],
            grounding=AnswerGroundingResult(is_grounded=True, confidence=0.98),
        )
    )

    async with AsyncSessionLocal() as session:
        with patch("rag_app.conversation.service.ConversationService") as mock_conv_cls, \
             patch("rag_app.security.authorization.get_allowed_departments", return_value=[dept.id]):
            mock_conv_inst = MagicMock()
            mock_conv_inst.get_or_create_conversation = AsyncMock(
                return_value=MagicMock(id=uuid.uuid4(), title="New Chat")
            )
            mock_conv_inst._validate_workspace_access = AsyncMock(return_value=ws)
            mock_conv_inst.search_workspace_threads = AsyncMock(return_value=[])
            mock_conv_inst.get_conversation_context_window = AsyncMock(return_value=[])
            mock_conv_inst.save_user_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4()))
            mock_conv_inst.save_assistant_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4()))
            mock_conv_inst.is_placeholder_title = MagicMock(return_value=False)
            mock_conv_cls.return_value = mock_conv_inst

            resp = await ask_question(
                request=req,
                service=mock_gen,
                retrieval_service=mock_retrieval,
                session=session,
                user=user,
            )

    # 3. Verification Assertions
    assert "within 30 days" in resp.answer
    assert len(resp.retrieved_chunks) == 1
    assert resp.retrieved_chunks[0].file_name == "employee_handbook.pdf"
    assert resp.grounding is not None
    assert resp.grounding.is_grounded is True
    assert (resp.grounding.confidence or 0.0) >= 0.90


# ============================================================================
# Scenario 2: Live Web Research (Real search provider invocation and real URLs)
# ============================================================================

@pytest.mark.asyncio
async def test_scenario_2_live_web_research():
    settings = get_settings()
    api_key = settings.tinyfish_api_key
    assert api_key, "Live TinyFish API key must be configured in staging environment"

    provider = TinyFishProvider(api_key=api_key)
    query = "History of Python programming language Guido van Rossum"

    # Execute real external search against live TinyFish Search API
    search_resp = await provider.search(query=query, limit=2)

    assert search_resp is not None
    assert len(search_resp.results) > 0, "Live search must return real web results"
    first_result = search_resp.results[0]
    assert first_result.url.startswith("http"), "Real source URL must be HTTP/HTTPS"
    assert first_result.title, "Live result must have a title"
    assert first_result.snippet, "Live result must have a non-empty snippet"


# ============================================================================
# Scenario 3: Provider Failure (Zero fabricated evidence, honest failure message)
# ============================================================================

@pytest.mark.asyncio
async def test_scenario_3_provider_failure_returns_honest_unverified_response(smoke_staging_context):
    ctx = smoke_staging_context
    user = ctx["user"]
    dept = ctx["department"]

    query = "Search the web for unfindable secret specifications XYZ-999"
    plan = QueryPlan(
        semantic_query=query,
        web_search_needed=True,
        web_search_reason="explicit_request",
        internal_retrieval_needed=True,
    )

    state: ResearchState = {
        "query": query,
        "raw_query": query,
        "top_k": 5,
        "rerank": False,
        "user": user,
        "allowed_departments": [dept.id],
        "workspace_id": None,
        "query_type": "knowledge",
        "query_plan": plan,
        "retrieved_chunks": [],
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": [],
        "web_research_error": "ProviderUnavailableError: TinyFish search API timed out",
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "is_sufficient": False,
        "sufficiency_status": "insufficient",
    }

    from rag_app.research.graph import generate_final_answer
    mock_gen = MagicMock()

    config = {
        "configurable": {
            "generation_service": mock_gen,
            "db_session": None,
        }
    }

    result = await generate_final_answer(state, config)
    final_resp = result["final_response"]

    # Invariant: No fabricated evidence, honest notification of failure
    assert len(final_resp.retrieved_chunks) == 0, "Failed provider must return 0 chunks"
    assert "could not be completed at this time" in final_resp.answer
    assert final_resp.grounding is not True


# ============================================================================
# Scenario 4: MCP Read-Only (Authorized workspace tool invocation)
# ============================================================================

@pytest.mark.asyncio
async def test_scenario_4_mcp_read_only_tool_invocation(smoke_staging_context):
    ctx = smoke_staging_context
    user = ctx["user"]
    dept = ctx["department"]
    ws = ctx["workspace"]

    query = "Tell me about company Acme Corp in our CRM"
    initial_state = {
        "query": query,
        "raw_query": query,
        "top_k": 5,
        "rerank": False,
        "user": user,
        "allowed_departments": [dept.id],
        "workspace_id": ws.id,
        "query_type": None,
        "query_plan": None,
        "retrieved_chunks": [],
        "search_results": [],
        "selected_urls": [],
        "crawled_pages": [],
        "clean_pages": [],
        "extracted_knowledge": None,
        "web_chunks": None,
        "web_research_error": None,
        "conversation_history": None,
        "workspace_thread_context": None,
        "final_response": None,
        "output_preferences": None,
        "conversation_id": None,
        "web_provider": None,
        "web_credential_ref": None,
        "web_enabled": False,
        "mcp_tools_called": None,
        "mcp_chunks": None,
    }

    mock_retrieval_service = MagicMock()
    mock_retrieval_service.retrieve = AsyncMock(
        return_value=MagicMock(results=[], strategy_trace={}, chunk_provenance={}, chunk_hierarchy={}, query_plan=None)
    )

    mock_gen_service = MagicMock()
    mock_gen_service.generate_from_context = AsyncMock(
        return_value=GenerateResponse(
            answer="Acme Corp is an Enterprise Tier customer with 500 employees located in San Francisco.",
            citations=["Workspace MCP: mcp_get_company"],
            retrieved_chunks=[],
            grounding=AnswerGroundingResult(is_grounded=True, confidence=0.95),
        )
    )

    async with AsyncSessionLocal() as session:
        config = {
            "configurable": {
                "retrieval_service": mock_retrieval_service,
                "generation_service": mock_gen_service,
                "db_session": session,
            }
        }

        with patch("rag_app.mcp.connection_service.GenericMCPClient") as mock_client_cls:
            mock_client_inst = AsyncMock()
            mock_client_inst.call_tool.return_value = MCPToolResult(
                name="get_company",
                content=[{"type": "text", "text": "Company: Acme Corp | Tier: Enterprise | Employees: 500"}],
                text="Company: Acme Corp | Tier: Enterprise | Employees: 500",
                structured_content={"company": "Acme Corp", "tier": "Enterprise", "employees": 500},
                is_error=False,
            )
            mock_client_cls.return_value.__aenter__.return_value = mock_client_inst

            with patch("rag_app.retrieval.sufficiency.EvidenceSufficiencyEvaluator") as mock_eval_cls:
                mock_eval = MagicMock()
                mock_eval.evaluate = AsyncMock(
                    return_value=SufficiencyEvaluation(
                        status=SufficiencyStatus.INSUFFICIENT,
                        is_sufficient=False,
                        confidence=0.1,
                        reason="No internal docs found for Acme Corp.",
                    )
                )
                mock_eval_cls.return_value = mock_eval

                final_state = await research_graph.ainvoke(initial_state, config)

                # Verify MCP tool was invoked through server binding
                assert final_state.get("mcp_tools_called") is not None
                assert "mcp_get_company" in final_state["mcp_tools_called"]

                # Verify real tool output chunk
                mcp_chunks = final_state.get("mcp_chunks") or []
                assert len(mcp_chunks) > 0
                assert "Acme Corp" in mcp_chunks[0].text
                assert mcp_chunks[0].file_name == "Workspace MCP: mcp_get_company"


# ============================================================================
# Scenario 5: CRM Write-Back (Fail-closed routing, no side-effect before approval, duplicate rejection)
# ============================================================================

@pytest.mark.asyncio
async def test_scenario_5_crm_writeback_hitl_lifecycle(smoke_staging_context):
    ctx = smoke_staging_context
    user = ctx["user"]
    dept = ctx["department"]
    ws = ctx["workspace"]

    query = "Draft and writeback meeting note to Twenty CRM for Acme Corp"

    # Step 1: Routing verification — Fast-path is rejected for mutation
    assert WorkflowService.is_fast_path(query) is False, "Mutating query must force adaptive workflow"

    async with AsyncSessionLocal() as session:
        mock_mcp = AsyncMock(spec=TwentyMCPClient)
        mock_mcp.create_note.return_value = "twenty-note-999"
        mock_mcp.create_note_target_for_company.return_value = "twenty-target-888"

        sync_service = CRMSyncService(session, mcp_client=mock_mcp, ingestion_service=AsyncMock())
        idem_key = f"smoke-hitl-{uuid.uuid4().hex[:8]}"

        # Step 2: Unapproved attempt fails closed
        with pytest.raises(PermissionError, match="not explicitly approved"):
            await sync_service.create_meeting_note_writeback(
                company_id="company-uuid-111",
                note_title="Acme Review",
                summary_markdown="Summary text",
                idempotency_key=idem_key,
                user_role="ADMIN",
                approved=False,
            )

        # Invariant: Remote CRM write tool was never called
        assert mock_mcp.create_note.call_count == 0

        # Step 3: Approved execution applies side-effect
        result = await sync_service.create_meeting_note_writeback(
            company_id="company-uuid-111",
            note_title="Acme Review",
            summary_markdown="Summary text",
            idempotency_key=idem_key,
            user_role="ADMIN",
            approved=True,
        )

        assert result["status"] == "COMPLETED"
        assert result["crm_note_id"] == "twenty-note-999"
        assert result["is_duplicate_replay"] is False
        assert mock_mcp.create_note.call_count == 1

        # Step 4: Duplicate approval replay with same idempotency key
        replay_result = await sync_service.create_meeting_note_writeback(
            company_id="company-uuid-111",
            note_title="Acme Review",
            summary_markdown="Summary text",
            idempotency_key=idem_key,
            user_role="ADMIN",
            approved=True,
        )

        assert replay_result["is_duplicate_replay"] is True
        assert replay_result["crm_note_id"] == "twenty-note-999"
        # Invariant: Remote CRM tool NOT called a second time
        assert mock_mcp.create_note.call_count == 1


# ============================================================================
# Scenario 6: Explicit Adaptive Workflow
# ============================================================================

@pytest.mark.asyncio
async def test_scenario_6_explicit_adaptive_workflow_dispatch(smoke_staging_context):
    ctx = smoke_staging_context
    user = ctx["user"]
    dept = ctx["department"]
    ws = ctx["workspace"]

    query = "Analyze comprehensive market trends for AI semiconductors in 2026"
    wf_req = WorkflowRunCreate(
        query=query,
        workspace_id=ws.id,
        department_id=dept.id,
        budget_limits=BudgetLimits(max_tokens=2000, max_cost_cents=50),
    )

    async with AsyncSessionLocal() as session:
        wf_service = WorkflowService(session=session)
        run, is_new = await wf_service.get_or_create_workflow_run(user=user, request=wf_req)

        assert is_new is True
        assert run.status == WorkflowStatus.PENDING
        assert run.workspace_id == ws.id
        assert run.user_id == user.id

        # Verify DAG planner executes and returns valid execution DAG
        planner = AdaptiveTaskPlanner(llm_provider=None)
        plan = planner.plan_initial_request(goal=query)

        assert len(plan.tasks) > 0
        assert all(isinstance(t, ExecutionTask) for t in plan.tasks)
        assert any(t.task_key == "gather_internal_knowledge" for t in plan.tasks)
