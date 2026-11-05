"""
Milestone 10A Verification Test Suite: Adaptive Request Routing.

Verifies:
1. Deterministic FAST_RAG classification for policy and internal knowledge queries (zero planner overhead).
2. Deterministic CONTEXTUAL classification for personal preferences, persona, and past discussion queries (zero task DAG overhead).
3. Deterministic ADAPTIVE_PLAN classification for multi-step research, comparisons, gap analysis, and CRM write operations.
4. LLM / Conservative fallback defaulting safely to FAST_RAG for ambiguous requests.
5. Runtime budget resolution on classifications (max tasks, parallel limits, retries, token bounds).
6. Security boundary: Mutating CRM requests fail closed to ADAPTIVE_PLAN even if client specifies fast_path.
7. API integration: POST /api/v1/chat/ask populates routing_mode appropriately.
"""

from __future__ import annotations

import os

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

import uuid
from unittest.mock import patch, MagicMock
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

from rag_app.core.config import get_settings
from rag_app.db.models import Department, Permission, User, UserDepartment, UserPermission, Workspace, WorkspaceMember
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.main import app
from rag_app.orchestration.router import (
    AdaptiveRequestRouter,
    AdaptiveRouteMode,
    AdaptiveRequestClassification,
)
from rag_app.security.authentication import create_access_token


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_routing_context():
    """Sets up an authorized user, department, and workspace for API tests."""
    async with AsyncSessionLocal() as session:
        dept = Department(
            id=uuid.uuid4(),
            name="Routing Test Dept",
            code=f"RT_{uuid.uuid4().hex[:6].upper()}",
            is_active=True,
        )
        session.add(dept)
        await session.flush()

        view_perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not view_perm:
            view_perm = Permission(id=uuid.uuid4(), code="VIEW", name="View Permission")
            session.add(view_perm)
            await session.flush()

        user = User(
            id=uuid.uuid4(),
            email=f"router_user_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="fakehashpassword",
            role="USER",
            is_active=True,
        )
        session.add(user)
        await session.flush()

        session.add(UserDepartment(user_id=user.id, department_id=dept.id))
        session.add(UserPermission(user_id=user.id, department_id=dept.id, permission_id=view_perm.id))

        workspace = Workspace(
            id=uuid.uuid4(),
            name="Routing Test Workspace",
            code=f"rt_ws_{uuid.uuid4().hex[:6]}",
            department_id=dept.id,
            is_active=True,
            created_by=user.id,
        )
        session.add(workspace)
        await session.flush()

        session.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="OWNER"))
        await session.commit()

        token = create_access_token(subject=str(user.id))
        return {
            "user": user,
            "token": token,
            "department": dept,
            "workspace": workspace,
        }


# ==============================================================================
# Unit Tests: AdaptiveRequestRouter Classifications
# ==============================================================================

@pytest.mark.asyncio
async def test_fast_rag_routing_for_company_policy():
    router = AdaptiveRequestRouter()
    queries = [
        "What is our company expense approval policy?",
        "Where is the employee handbook located?",
        "What is our leave policy for medical emergencies?",
        "What are our company office benefits and insurance details?",
    ]
    for q in queries:
        cls = await router.classify(q)
        assert cls.route == AdaptiveRouteMode.FAST_RAG, f"Expected FAST_RAG for query: {q}"
        assert cls.requires_planning is False
        assert any(c in cls.required_capabilities for c in ("RAG", "knowledge_retrieval", "documents"))
        assert cls.runtime_budgets.get("max_tasks") == 15
        assert cls.runtime_budgets.get("max_parallel_tasks") == 5


@pytest.mark.asyncio
async def test_contextual_routing_for_preferences_and_history():
    router = AdaptiveRequestRouter()
    queries = [
        "Explain our CRM architecture using the approach we discussed earlier.",
        "Generate a summary in my preferred style.",
        "What did we agree regarding the caching strategy as we discussed?",
        "Remember how I prefer technical notes formatted?",
    ]
    for q in queries:
        cls = await router.classify(q)
        assert cls.route == AdaptiveRouteMode.CONTEXTUAL, f"Expected CONTEXTUAL for query: {q}"
        assert cls.requires_planning is False
        assert "MEMORY" in cls.required_capabilities or "RAG" in cls.required_capabilities


@pytest.mark.asyncio
async def test_adaptive_plan_routing_for_complex_tasks_and_crm():
    router = AdaptiveRequestRouter()
    queries = [
        "Research modern CRM integration approaches, compare them with our current architecture, identify gaps and prepare a technical recommendation.",
        "Investigate industry microservice patterns and compare with our setup and summarize findings.",
        "Update twenty CRM company record for Acme Corp with our latest notes.",
        "Create a note into CRM for the prospect discussion.",
    ]
    for q in queries:
        cls = await router.classify(q)
        assert cls.route == AdaptiveRouteMode.ADAPTIVE_PLAN, f"Expected ADAPTIVE_PLAN for query: {q}"
        assert cls.requires_planning is True
        assert cls.runtime_budgets.get("max_tasks") == 15
        assert cls.runtime_budgets.get("max_plan_iterations") == 3


@pytest.mark.asyncio
async def test_ambiguous_query_safe_fallback():
    router = AdaptiveRequestRouter()
    # When LLM raises or is unavailable, fallback must safely choose FAST_RAG
    with patch("rag_app.orchestration.router.get_llm", side_effect=RuntimeError("Provider offline")):
        cls = await router.classify("Some totally obscure random input")
        assert cls.route == AdaptiveRouteMode.FAST_RAG
        assert cls.requires_planning is False
        assert "max_tasks" in cls.runtime_budgets


# ==============================================================================
# Integration Tests: Chat API Execution Mode & Security Boundaries
# ==============================================================================

@pytest.mark.asyncio
async def test_chat_ask_returns_fast_rag_routing_mode(setup_routing_context):
    ctx = setup_routing_context
    headers = {"Authorization": f"Bearer {ctx['token']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat/ask",
            json={
                "query": "Hello there",
                "workspace_id": str(ctx["workspace"].id),
                "execution_mode": "fast_path",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("routing_mode") == "FAST_RAG"
        assert data.get("workflow_run_id") is None


@pytest.mark.asyncio
async def test_chat_ask_explicit_adaptive_workflow_mode(setup_routing_context):
    ctx = setup_routing_context
    headers = {"Authorization": f"Bearer {ctx['token']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat/ask",
            json={
                "query": "Research best CRM practices for our team",
                "workspace_id": str(ctx["workspace"].id),
                "execution_mode": "adaptive_workflow",
                "workflow_idempotency_key": f"chat-wf-{uuid.uuid4().hex[:6]}",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("routing_mode") == "ADAPTIVE_PLAN"
        assert data.get("workflow_run_id") is not None


@pytest.mark.asyncio
async def test_chat_ask_mutating_crm_action_fails_closed_to_adaptive_plan(setup_routing_context):
    """
    Even if client specifies 'fast_path', mutating CRM operations must fail closed
    to ADAPTIVE_PLAN to guarantee approval gates.
    """
    ctx = setup_routing_context
    headers = {"Authorization": f"Bearer {ctx['token']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat/ask",
            json={
                "query": "Update twenty crm note for client Acme Corp",
                "workspace_id": str(ctx["workspace"].id),
                "execution_mode": "fast_path",  # Client tries to bypass workflow!
                "workflow_idempotency_key": f"crm-wf-{uuid.uuid4().hex[:6]}",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Security invariant: Must NOT be fast path!
        assert data.get("routing_mode") == "ADAPTIVE_PLAN"
        assert data.get("workflow_run_id") is not None
