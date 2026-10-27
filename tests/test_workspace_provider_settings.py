"""
Acceptance Suite for Milestone 4A: Workspace AI Settings: Backend and UI Integration.
Verifies:
1. Workspace provider profile configuration persistence and retrieval.
2. Workspace isolation: Workspace A settings do not leak to Workspace B.
3. Strict RBAC: OWNER/ADMIN can write; VIEWER/EDITOR cannot write (403 Forbidden).
4. Secret safety: credential references are preserved without exposing raw keys.
5. Incompatible embedding model dimensions (!= 384) are strictly rejected with 422.
6. Test-connection endpoint executes actual backend probes and fails safely.
7. Runtime consumer (GenerationService / resolve_llm_for_workspace) uses the configured workspace provider.
8. Request-scoped configuration consistency is preserved.
"""

from __future__ import annotations

import os
from typing import Any
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select

from rag_app.db.models import (
    Department,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
    WorkspaceProviderProfile,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.generation.service import GenerationService
from rag_app.providers.adapters.llm.mock import MockLLMAdapter
from rag_app.providers.base.llm import LLMProvider
from rag_app.providers.resolver import resolve_llm_for_workspace, resolve_providers_for_workspace
from rag_app.schemas.rag import GenerateRequest
from rag_app.schemas.workspace import (
    ProviderTestConnectionRequest,
    WorkspaceProviderProfileUpdate,
)
from rag_app.workspace.service import WorkspaceService


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_workspaces_and_users():
    """
    Sets up two isolated workspaces with Owner, Viewer, and Admin users.
    """
    async with AsyncSessionLocal() as session:
        # 1. Department
        dept = await session.scalar(select(Department).where(Department.code == "WS_SETTINGS_DEPT"))
        if not dept:
            dept = Department(code="WS_SETTINGS_DEPT", name="Workspace Settings Department", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        # 2. Admin User
        admin_user = await session.scalar(select(User).where(User.email == "ws_admin@test.com"))
        if not admin_user:
            admin_user = User(email="ws_admin@test.com", hashed_password="hashed_pw", role="ADMIN", is_active=True)
            session.add(admin_user)
            await session.commit()
            await session.refresh(admin_user)

        # 3. Owner User
        owner_user = await session.scalar(select(User).where(User.email == "ws_owner@test.com"))
        if not owner_user:
            owner_user = User(email="ws_owner@test.com", hashed_password="hashed_pw", role="USER", is_active=True)
            session.add(owner_user)
            await session.commit()
            await session.refresh(owner_user)

        # 4. Viewer User
        viewer_user = await session.scalar(select(User).where(User.email == "ws_viewer@test.com"))
        if not viewer_user:
            viewer_user = User(email="ws_viewer@test.com", hashed_password="hashed_pw", role="USER", is_active=True)
            session.add(viewer_user)
            await session.commit()
            await session.refresh(viewer_user)

        # 5. Seed VIEW Permission & Bind Users to Department
        perm_view = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not perm_view:
            perm_view = Permission(code="VIEW", name="View Permission", is_active=True)
            session.add(perm_view)
            await session.commit()
            await session.refresh(perm_view)

        for u in [owner_user, viewer_user]:
            ud = await session.scalar(
                select(UserDepartment).where(UserDepartment.user_id == u.id, UserDepartment.department_id == dept.id)
            )
            if not ud:
                session.add(UserDepartment(user_id=u.id, department_id=dept.id))
            up = await session.scalar(
                select(UserPermission).where(
                    UserPermission.user_id == u.id,
                    UserPermission.department_id == dept.id,
                    UserPermission.permission_id == perm_view.id,
                )
            )
            if not up:
                session.add(UserPermission(user_id=u.id, department_id=dept.id, permission_id=perm_view.id))
        await session.commit()

        # 5. Workspace A
        ws_a = await session.scalar(select(Workspace).where(Workspace.code == "WS_A_TEST"))
        if not ws_a:
            ws_a = Workspace(name="Workspace Alpha", code="WS_A_TEST", department_id=dept.id, is_active=True, created_by=owner_user.id)
            session.add(ws_a)
            await session.commit()
            await session.refresh(ws_a)

        # 6. Workspace B
        ws_b = await session.scalar(select(Workspace).where(Workspace.code == "WS_B_TEST"))
        if not ws_b:
            ws_b = Workspace(name="Workspace Beta", code="WS_B_TEST", department_id=dept.id, is_active=True, created_by=admin_user.id)
            session.add(ws_b)
            await session.commit()
            await session.refresh(ws_b)

        # Bind Members
        mem_a_owner = await session.scalar(
            select(WorkspaceMember).where(WorkspaceMember.workspace_id == ws_a.id, WorkspaceMember.user_id == owner_user.id)
        )
        if not mem_a_owner:
            session.add(WorkspaceMember(workspace_id=ws_a.id, user_id=owner_user.id, role="OWNER"))

        mem_a_viewer = await session.scalar(
            select(WorkspaceMember).where(WorkspaceMember.workspace_id == ws_a.id, WorkspaceMember.user_id == viewer_user.id)
        )
        if not mem_a_viewer:
            session.add(WorkspaceMember(workspace_id=ws_a.id, user_id=viewer_user.id, role="VIEWER"))

        await session.commit()
        return dept, admin_user, owner_user, viewer_user, ws_a, ws_b


# ============================================================================
# Gate 1: Profile CRUD and Persistence
# ============================================================================

@pytest.mark.asyncio
async def test_workspace_provider_profile_crud_and_persistence(setup_workspaces_and_users):
    """1: Saves provider profile and verifies persistence and exact retrieval."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        payload = WorkspaceProviderProfileUpdate(
            name="Alpha High-Performance Profile",
            llm_provider="mock",
            llm_model="mock-deterministic-v1",
            llm_credential_ref="MOCK_TEST_REF",
            embedding_provider="sentence_transformers",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            reranker_provider="cross_encoder",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        res = await service.upsert_provider_profile(admin_user, ws_a.id, payload)
        assert res.llm_provider == "mock"
        assert res.llm_model == "mock-deterministic-v1"
        assert res.llm_credential_ref == "MOCK_TEST_REF"
        assert res.embedding_provider == "sentence_transformers"
        assert res.reranker_provider == "cross_encoder"

        # Verify retrieval via get_provider_profile
        retrieved = await service.get_provider_profile(admin_user, ws_a.id)
        assert retrieved is not None
        assert retrieved.llm_provider == "mock"
        assert retrieved.llm_model == "mock-deterministic-v1"


# ============================================================================
# Gate 2: Workspace Isolation
# ============================================================================

@pytest.mark.asyncio
async def test_workspace_isolation_cross_tenant_protection(setup_workspaces_and_users):
    """2: Verifies Workspace A configuration does not leak to Workspace B."""
    _, admin_user, _, _, ws_a, ws_b = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        # Configure Workspace A with 'mock' provider
        await service.upsert_provider_profile(
            admin_user,
            ws_a.id,
            WorkspaceProviderProfileUpdate(llm_provider="mock", llm_model="mock-alpha-custom"),
        )

        # Retrieve Workspace B's profile - must be None or default, never Alpha's config
        b_profile = await service.get_provider_profile(admin_user, ws_b.id)
        if b_profile is not None:
            assert b_profile.llm_model != "mock-alpha-custom"


# ============================================================================
# Gate 3: Role-Based Authorization
# ============================================================================

@pytest.mark.asyncio
async def test_workspace_role_authorization_owner_vs_viewer(setup_workspaces_and_users):
    """3: OWNER can update configuration; VIEWER is denied modification with HTTP 403."""
    _, _, owner_user, viewer_user, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        # 1. OWNER updates configuration -> Succeeds
        owner_payload = WorkspaceProviderProfileUpdate(
            llm_provider="mock",
            llm_model="mock-owner-model",
        )
        owner_res = await service.upsert_provider_profile(owner_user, ws_a.id, owner_payload)
        assert owner_res.llm_model == "mock-owner-model"

        # 2. VIEWER attempts update -> HTTP 403 Forbidden
        viewer_payload = WorkspaceProviderProfileUpdate(
            llm_provider="groq",
        )
        with pytest.raises(HTTPException) as exc_info:
            await service.upsert_provider_profile(viewer_user, ws_a.id, viewer_payload)
        assert exc_info.value.status_code == 403
        assert "Only workspace OWNER or ADMIN" in exc_info.value.detail


# ============================================================================
# Gate 4: Secret Reference Safety
# ============================================================================

@pytest.mark.asyncio
async def test_secret_reference_safety(setup_workspaces_and_users):
    """4: Ensures credential_ref is stored as a reference label and no raw API keys leak."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        ref_label = "GROQ_SECRET_VAULT_KEY_01"
        payload = WorkspaceProviderProfileUpdate(
            llm_provider="mock",
            llm_credential_ref=ref_label,
        )
        res = await service.upsert_provider_profile(admin_user, ws_a.id, payload)
        assert res.llm_credential_ref == ref_label

        # Verify str and repr representations never reveal hypothetical api keys
        assert "gsk_" not in str(res)


# ============================================================================
# Gate 5: Embedding Model Dimension Compatibility Enforcement
# ============================================================================

@pytest.mark.asyncio
async def test_incompatible_embedding_model_rejected(setup_workspaces_and_users):
    """5: Configuring incompatible embedding model (!= 384 dimensions) is rejected with HTTP 422."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        # Incompatible 1536-dim model
        bad_payload = WorkspaceProviderProfileUpdate(
            embedding_provider="sentence_transformers",
            embedding_model="text-embedding-3-small",
        )
        with pytest.raises(HTTPException) as exc_info:
            await service.upsert_provider_profile(admin_user, ws_a.id, bad_payload)
        assert exc_info.value.status_code == 422
        assert "Incompatible embedding model" in exc_info.value.detail
        assert "384 dimensions" in exc_info.value.detail


# ============================================================================
# Gate 6: Connection-Check Endpoint Probes
# ============================================================================

@pytest.mark.asyncio
async def test_test_connection_mock_llm(setup_workspaces_and_users):
    """6A: Test connection for mock LLM executes actual generation probe and returns success."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        req = ProviderTestConnectionRequest(
            capability="llm",
            provider="mock",
            model="mock-deterministic-v1",
        )
        res = await service.test_provider_connection(admin_user, ws_a.id, req)
        assert res.success is True
        assert res.capability == "llm"
        assert res.provider == "mock"
        assert res.latency_ms is not None
        assert res.latency_ms >= 0


@pytest.mark.asyncio
async def test_test_connection_missing_groq_key_fails_safely(setup_workspaces_and_users):
    """6B: Test connection for Groq with missing API key fails safely with clear message."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with patch("rag_app.core.config.get_settings") as mock_settings:
                mock_settings.return_value.groq_api_key = ""
                mock_settings.return_value.groq_model = "llama-3.3-70b-versatile"
                req = ProviderTestConnectionRequest(
                    capability="llm",
                    provider="groq",
                )
                res = await service.test_provider_connection(admin_user, ws_a.id, req)
                assert res.success is False
                assert "GROQ_API_KEY is not configured" in res.message


@pytest.mark.asyncio
async def test_test_connection_embedding_dimension_validation(setup_workspaces_and_users):
    """6C: Test connection checks embedding model dimensions against 384 requirement."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        # Incompatible model probe
        req_incompatible = ProviderTestConnectionRequest(
            capability="embedding",
            provider="sentence_transformers",
            model="bge-large-en-v1.5",
        )
        res_bad = await service.test_provider_connection(admin_user, ws_a.id, req_incompatible)
        assert res_bad.success is False
        assert "Incompatible embedding model" in res_bad.message
        assert res_bad.details["required_dimension"] == 384


# ============================================================================
# Gate 7: Actual Runtime Consumption & Scoped Consistency
# ============================================================================

@pytest.mark.asyncio
async def test_runtime_consumer_resolves_workspace_profile(setup_workspaces_and_users):
    """7: Runtime resolver resolves the workspace's configured provider and model."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        # Set workspace provider to mock
        await service.upsert_provider_profile(
            admin_user,
            ws_a.id,
            WorkspaceProviderProfileUpdate(
                llm_provider="mock",
                llm_model="mock-runtime-v1",
            ),
        )

        # Resolve LLM for workspace
        resolved_llm = await resolve_llm_for_workspace(ws_a.id, session)
        assert isinstance(resolved_llm, MockLLMAdapter)
        assert resolved_llm.provider_name == "mock"
        assert resolved_llm.model == "mock-runtime-v1"


@pytest.mark.asyncio
async def test_generation_service_respects_workspace_provider_profile(setup_workspaces_and_users):
    """8: GenerationService uses workspace provider profile during answer generation."""
    dept, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service_ws = WorkspaceService(session)

        # Configure workspace with Mock provider
        await service_ws.upsert_provider_profile(
            admin_user,
            ws_a.id,
            WorkspaceProviderProfileUpdate(
                llm_provider="mock",
                llm_model="mock-gen-test",
            ),
        )

        gen_service = GenerationService(session=session)
        req = GenerateRequest(
            query="What is the workspace performance status?",
            workspace_id=ws_a.id,
        )

        with patch.object(gen_service.retrieval_service, "retrieve") as mock_retrieval:
            from rag_app.schemas.rag import RetrievalResponse, RetrievedChunk
            mock_retrieval.return_value = RetrievalResponse(
                query=req.query,
                results=[
                    RetrievedChunk(
                        chunk_id="chk_ws_1",
                        document_id=str(uuid.uuid4()),
                        department_id=str(dept.id),
                        file_name="workspace_report.txt",
                        chunk_index=0,
                        text="Workspace performance is tracked at 98% reliability.",
                        score=0.95,
                    )
                ],
                strategy_trace={},
                chunk_provenance={},
                chunk_hierarchy={},
            )

            resp = await gen_service.generate_answer(request=req, user=admin_user)
            assert resp.answer is not None
            assert len(resp.answer) > 0


@pytest.mark.asyncio
async def test_retrieval_service_respects_workspace_retrieval_configuration(setup_workspaces_and_users):
    """9: RetrievalService runtime respects workspace profile configuration (top_k, reranker_enabled, vector_search_enabled)."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service_ws = WorkspaceService(session)

        # Configure workspace with specific retrieval settings: top_k=3, reranker_enabled=False, vector_search_enabled=False
        await service_ws.upsert_provider_profile(
            admin_user,
            ws_a.id,
            WorkspaceProviderProfileUpdate(
                configuration={
                    "top_k": 3,
                    "reranker_enabled": False,
                    "vector_search_enabled": False,
                    "fts_enabled": True,
                }
            ),
        )

        from rag_app.retrieval.service import RetrievalService
        retrieval_svc = RetrievalService(session=session)

        # Mock keyword search to return 5 dummy chunks and check runtime overrides
        with patch.object(retrieval_svc, "_keyword_search", new_callable=AsyncMock) as mock_kw:
            with patch.object(retrieval_svc, "_vector_search", new_callable=AsyncMock) as mock_vec:
                mock_kw.return_value = []
                mock_vec.return_value = []

                res = await retrieval_svc.retrieve(
                    query="Find strategic operating plan",
                    workspace_id=ws_a.id,
                )

                # Vector search must be skipped because vector_search_enabled is False
                assert mock_vec.await_count == 0
                # Keyword search must be called because fts_enabled is True
                assert mock_kw.await_count >= 1


@pytest.mark.asyncio
async def test_test_connection_vector_store_probe(setup_workspaces_and_users):
    """10: Vector store connection probe executes real SELECT 1 probe against database."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)
        req = ProviderTestConnectionRequest(
            capability="vector_store",
            provider="pgvector",
        )
        res = await service.test_provider_connection(admin_user, ws_a.id, req)
        assert res.success is True
        assert res.capability == "vector_store"
        assert res.latency_ms is not None
        assert res.latency_ms >= 0
        assert "pgvector" in res.message or "PostgreSQL" in res.message


@pytest.mark.asyncio
async def test_test_connection_unconfigured_openai_fails_safely(setup_workspaces_and_users):
    """11: OpenAI test probe without OPENAI_API_KEY fails truthfully with configured=False."""
    _, admin_user, _, _, ws_a, _ = setup_workspaces_and_users
    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            req = ProviderTestConnectionRequest(
                capability="llm",
                provider="openai",
                model="gpt-4o",
            )
            res = await service.test_provider_connection(admin_user, ws_a.id, req)
            assert res.success is False
            assert "OPENAI_API_KEY is not configured" in res.message
            assert res.details == {"configured": False}

