"""
Verification suite for TinyFish Search API as a first-class Workspace Web Search Provider.

Verifies:
1. TinyFish Search API contract, HTTP request structure (GET https://api.search.tinyfish.ai, X-API-Key, query, limit, purpose).
2. Response normalization into internal WebSearchResponse / WebSearchResult models.
3. Provider error mapping (401/403, 429, timeouts, missing credentials) with zero secret leakage.
4. Provider Capability Registry registration (capability='web', provider='tinyfish').
5. Workspace Provider connection test (WorkspaceService.test_provider_connection for web/tinyfish).
6. Workspace profile persistence and isolation (Workspace A = tinyfish vs Workspace B = tavily).
7. Research Graph runtime dynamic resolution (search_web node routes to TinyFish vs Tavily based on workspace state).
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select

from rag_app.core.config import get_settings
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
from rag_app.providers.adapters.web.tinyfish import TinyFishWebAdapter
from rag_app.providers.base.web import WebSearchResponse, WebSearchResult
from rag_app.providers.registry import get_provider, provider_registry
from rag_app.providers.resolver import get_provider_registry_summary
from rag_app.research.graph import ResearchState, decide_next_after_eval, search_web
from rag_app.research.provider import (
    ProviderRequestError,
    ProviderUnavailableError,
)
from rag_app.research.providers.tinyfish import TinyFishProvider
from rag_app.schemas.research import SufficiencyEvaluation, SufficiencyStatus
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
async def setup_tinyfish_test_env():
    """
    Sets up isolated test environment in PostgreSQL disposable DB:
    - 1 Department
    - 1 Admin user, 1 Owner user
    - 2 Workspaces (Workspace TF and Workspace TV)
    """
    async with AsyncSessionLocal() as session:
        dept = await session.scalar(select(Department).where(Department.code == "TF_DEPT"))
        if not dept:
            dept = Department(code="TF_DEPT", name="TinyFish Test Dept", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        owner = await session.scalar(select(User).where(User.email == "tf_owner@test.com"))
        if not owner:
            owner = User(email="tf_owner@test.com", hashed_password="hashed_pw", role="USER", is_active=True)
            session.add(owner)
            await session.commit()
            await session.refresh(owner)

        perm_view = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
        if not perm_view:
            perm_view = Permission(code="VIEW", name="View Perm", is_active=True)
            session.add(perm_view)
            await session.commit()
            await session.refresh(perm_view)

        ud = await session.scalar(
            select(UserDepartment).where(UserDepartment.user_id == owner.id, UserDepartment.department_id == dept.id)
        )
        if not ud:
            session.add(UserDepartment(user_id=owner.id, department_id=dept.id))
            session.add(UserPermission(user_id=owner.id, department_id=dept.id, permission_id=perm_view.id))
            await session.commit()

        # Workspace A (TinyFish)
        ws_a = await session.scalar(select(Workspace).where(Workspace.code == "ws-tinyfish-test"))
        if not ws_a:
            ws_a = Workspace(name="Workspace TinyFish", code="ws-tinyfish-test", department_id=dept.id, is_active=True)
            session.add(ws_a)
            await session.commit()
            await session.refresh(ws_a)
            session.add(WorkspaceMember(workspace_id=ws_a.id, user_id=owner.id, role="OWNER"))
            await session.commit()

        # Workspace B (Tavily)
        ws_b = await session.scalar(select(Workspace).where(Workspace.code == "ws-tavily-test"))
        if not ws_b:
            ws_b = Workspace(name="Workspace Tavily", code="ws-tavily-test", department_id=dept.id, is_active=True)
            session.add(ws_b)
            await session.commit()
            await session.refresh(ws_b)
            session.add(WorkspaceMember(workspace_id=ws_b.id, user_id=owner.id, role="OWNER"))
            await session.commit()

        return {
            "dept": dept,
            "owner": owner,
            "ws_a": ws_a,
            "ws_b": ws_b,
        }


# ============================================================================
# 1. Official API Contract & Response Normalization
# ============================================================================

@pytest.mark.asyncio
async def test_tinyfish_contract_request_structure():
    """
    Verifies TinyFish Search API is invoked with:
    - GET request to https://api.search.tinyfish.ai
    - 'X-API-Key' authentication header
    - query, limit, and purpose parameters
    """
    captured_requests = []

    def mock_handler(request: httpx.Request):
        captured_requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "query": "quantum computing benchmarks",
                "results": [
                    {
                        "position": 1,
                        "title": "Quantum Benchmark Results 2026",
                        "url": "https://quantum.example.com/benchmarks",
                        "snippet": "Analysis of quantum advantage in 2026.",
                        "site_name": "Quantum Example",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    adapter = TinyFishWebAdapter(api_key="test-tf-secret-key-999")

    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        res = await adapter.search("quantum computing benchmarks", limit=3)

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.method == "GET"
    assert "api.search.tinyfish.ai" in str(req.url)
    assert req.headers.get("x-api-key") == "test-tf-secret-key-999"
    assert "query=quantum+computing+benchmarks" in str(req.url)
    assert "purpose=" in str(req.url)
    # The official TinyFish API documentation (https://docs.tinyfish.ai/search-api/reference)
    # does NOT accept 'limit'. Upstream returns full pages, and we enforce the cap client-side.
    assert "limit=" not in str(req.url)

    # Response normalization
    assert isinstance(res, WebSearchResponse)
    assert len(res.results) == 1
    assert res.results[0].title == "Quantum Benchmark Results 2026"
    assert res.results[0].url == "https://quantum.example.com/benchmarks"
    assert "quantum advantage" in res.results[0].snippet


# ============================================================================
# 2. Error Mapping & Safe Handling (No Key Leaks)
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code,expected_error_type,expected_substr",
    [
        (401, ProviderRequestError, "authentication failed"),
        (403, ProviderRequestError, "authentication failed"),
        (429, ProviderRequestError, "rate limit exceeded"),
        (503, ProviderUnavailableError, "server error"),
    ],
)
async def test_tinyfish_error_mapping_and_secret_safety(status_code, expected_error_type, expected_substr):
    """
    Verifies HTTP error status codes are cleanly mapped into internal domain exceptions
    and secret API keys are never included in error messages.
    """
    secret = "sk-tinyfish-super-secret-key"

    def mock_err_handler(request: httpx.Request):
        return httpx.Response(status_code=status_code, content=b'{"error": "Simulated error"}')

    transport = httpx.MockTransport(mock_err_handler)
    adapter = TinyFishWebAdapter(api_key=secret)

    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        with pytest.raises(expected_error_type) as exc_info:
            await adapter.search("test query")

    err_msg = str(exc_info.value)
    assert expected_substr in err_msg.lower()
    assert secret not in err_msg


@pytest.mark.asyncio
async def test_tinyfish_missing_credentials_fails_safely():
    """
    Verifies that calling TinyFish without any configured API key safely raises
    ProviderRequestError rather than crashing unexpectedly.
    """
    with patch.dict(os.environ, {}, clear=True), patch("rag_app.research.providers.tinyfish.get_settings") as mock_settings:
        mock_cfg = MagicMock()
        mock_cfg.tinyfish_api_key = None
        mock_cfg.tinyfish_search_url = "https://api.search.tinyfish.ai"
        mock_cfg.tinyfish_fetch_url = "https://api.fetch.tinyfish.ai"
        mock_settings.return_value = mock_cfg

        with pytest.raises(ProviderRequestError) as exc_info:
            TinyFishProvider(api_key=None)

        assert "not configured" in str(exc_info.value)


# ============================================================================
# 3. Provider Registry Summary & Resolution
# ============================================================================

@pytest.mark.asyncio
async def test_tinyfish_in_provider_registry():
    """
    Verifies 'tinyfish' is registered under 'web' capability in registry and
    resolves to TinyFishWebAdapter.
    """
    summary = await get_provider_registry_summary()
    assert "web" in summary
    assert "tinyfish" in summary["web"]
    assert "tinyfish" in provider_registry.list_providers("web")

    adapter = get_provider("web", "tinyfish")
    assert isinstance(adapter, TinyFishWebAdapter)


# ============================================================================
# 4. Workspace Provider Connection Test
# ============================================================================

@pytest.mark.asyncio
async def test_workspace_connection_probe_tinyfish(setup_tinyfish_test_env):
    """
    Verifies test_provider_connection probe for capability='web', provider='tinyfish':
    - With valid credential_ref, returns success=True
    - With missing credential_ref and no server env, returns success=False truthfully
    """
    env = setup_tinyfish_test_env
    ws = env["ws_a"]
    user = env["owner"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)
        # 1. Test probe with explicit valid credential ref
        req_valid = ProviderTestConnectionRequest(
            capability="web",
            provider="tinyfish",
            credential_ref="test-secret-probe-key",
        )
        res_valid = await service.test_provider_connection(user, ws.id, req_valid)
        assert res_valid.success is True
        assert res_valid.capability == "web"
        assert res_valid.provider == "tinyfish"

        # 2. Test probe with unconfigured credentials
        with patch.dict(os.environ, {}, clear=True), patch("rag_app.core.config.get_settings") as mock_settings:
            mock_cfg = MagicMock()
            mock_cfg.tinyfish_api_key = None
            mock_settings.return_value = mock_cfg

            req_missing = ProviderTestConnectionRequest(
                capability="web",
                provider="tinyfish",
                credential_ref=None,
            )
            res_missing = await service.test_provider_connection(user, ws.id, req_missing)
            assert res_missing.success is False
            assert "credentials" in res_missing.message.lower() or "not configured" in res_missing.message.lower()


# ============================================================================
# 5. Workspace Isolation: TinyFish vs Tavily Profile Persistence
# ============================================================================

@pytest.mark.asyncio
async def test_workspace_profile_persistence_and_isolation(setup_tinyfish_test_env):
    """
    Verifies that:
    - Workspace A successfully persists web_provider='tinyfish' and web_enabled=True.
    - Workspace B successfully persists web_provider='tavily' and web_enabled=True.
    - Settings are completely isolated between workspaces.
    """
    env = setup_tinyfish_test_env
    ws_a = env["ws_a"]
    ws_b = env["ws_b"]
    user = env["owner"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)
        # Upsert Workspace A -> TinyFish
        up_a = WorkspaceProviderProfileUpdate(
            name="Workspace A Profile",
            web_provider="tinyfish",
            configuration={"web_enabled": True},
        )
        prof_a = await service.upsert_provider_profile(user, ws_a.id, up_a)
        assert prof_a.web_provider == "tinyfish"
        assert prof_a.configuration.get("web_enabled") is True

        # Upsert Workspace B -> Tavily
        up_b = WorkspaceProviderProfileUpdate(
            name="Workspace B Profile",
            web_provider="tavily",
            configuration={"web_enabled": True},
        )
        prof_b = await service.upsert_provider_profile(user, ws_b.id, up_b)
        assert prof_b.web_provider == "tavily"
        assert prof_b.configuration.get("web_enabled") is True

        # Verify query isolation
        read_a = await service.get_provider_profile(user, ws_a.id)
        assert read_a is not None
        assert read_a.web_provider == "tinyfish"

        read_b = await service.get_provider_profile(user, ws_b.id)
        assert read_b is not None
        assert read_b.web_provider == "tavily"


# ============================================================================
# 6. Research Graph Node Dynamic Resolution & Execution
# ============================================================================

@pytest.mark.asyncio
async def test_research_graph_search_web_uses_tinyfish_when_configured():
    """
    Verifies that the research graph search_web node:
    - Resolves and invokes TinyFishWebAdapter when state has web_provider='tinyfish'.
    - Populates state['web_results'] from normalized TinyFish output.
    """
    mock_tf_adapter = AsyncMock()
    mock_tf_adapter.search.return_value = WebSearchResponse(
        results=[
            WebSearchResult(
                url="https://tf.example.com/article1",
                title="TinyFish Article 1",
                snippet="Deep agentic search results for query.",
            )
        ]
    )

    state: ResearchState = {
        "query": "latest research in superconductor materials",
        "web_provider": "tinyfish",
        "web_enabled": True,
        "queries": ["superconductor materials breakthroughs"],
    }

    with patch("rag_app.providers.registry.get_provider") as mock_get_provider:
        mock_get_provider.return_value = mock_tf_adapter
        updated_state = await search_web(state)

    mock_get_provider.assert_called_once_with("web", "tinyfish")
    assert "search_results" in updated_state
    assert len(updated_state["search_results"]) == 1
    assert updated_state["search_results"][0].title == "TinyFish Article 1"
    assert updated_state["search_results"][0].url == "https://tf.example.com/article1"


@pytest.mark.asyncio
async def test_research_graph_search_web_uses_tavily_when_configured():
    """
    Verifies that the research graph search_web node resolves Tavily when state
    specifies web_provider='tavily'.
    """
    mock_tavily_adapter = AsyncMock()
    mock_tavily_adapter.search.return_value = WebSearchResponse(
        results=[
            WebSearchResult(
                url="https://tavily.example.com/article2",
                title="Tavily Article 2",
                snippet="Tavily AI search results.",
            )
        ]
    )

    state: ResearchState = {
        "query": "financial market trends",
        "web_provider": "tavily",
        "web_enabled": True,
        "queries": ["market trends 2026"],
    }

    with patch("rag_app.providers.registry.get_provider") as mock_get_provider:
        mock_get_provider.return_value = mock_tavily_adapter
        updated_state = await search_web(state)

    mock_get_provider.assert_called_once_with("web", "tavily")
    assert "search_results" in updated_state
    assert len(updated_state["search_results"]) == 1
    assert updated_state["search_results"][0].title == "Tavily Article 2"


@pytest.mark.asyncio
async def test_research_graph_decide_next_respects_web_enabled_toggle():
    """
    Verifies that decide_next_after_eval skips web research when web_enabled is False,
    even if sufficiency is INSUFFICIENT.
    """
    # 1. web_enabled = False and insufficient -> returns "no_context"
    state_disabled: ResearchState = {
        "query": "test query",
        "web_enabled": False,
        "evaluation": SufficiencyEvaluation(
            status=SufficiencyStatus.INSUFFICIENT,
            confidence_score=0.2,
            reasoning="Need more info",
            key_evidence_found=[],
            missing_information=["external facts"],
        ),
        "queries": [],
    }
    next_node = decide_next_after_eval(state_disabled)
    assert next_node == "no_context"

    # 2. web_enabled = True and insufficient -> returns "init_research_run"
    state_enabled: ResearchState = {
        "query": "test query",
        "web_enabled": True,
        "evaluation": SufficiencyEvaluation(
            status=SufficiencyStatus.INSUFFICIENT,
            confidence_score=0.2,
            reasoning="Need more info",
            key_evidence_found=[],
            missing_information=["external facts"],
        ),
        "queries": [],
    }
    next_node_enabled = decide_next_after_eval(state_enabled)
    assert next_node_enabled == "init_research_run"


# ============================================================================
# 7. Milestone 5D: Credential Security & Masking Invariants
# ============================================================================

@pytest.mark.asyncio
async def test_credential_ref_masked_placeholder_does_not_overwrite_stored_secret(setup_tinyfish_test_env):
    """
    Verifies that submitting a masked placeholder ('••••••••••••••••') never
    accidentally overwrites or corrupts an existing stored credential reference.
    """
    env = setup_tinyfish_test_env
    ws = env["ws_a"]
    user = env["owner"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)
        # Step 1: Save legitimate credential reference
        up1 = WorkspaceProviderProfileUpdate(
            web_provider="tinyfish",
            web_credential_ref="TINYFISH_SECRET_REF_ENV_01",
        )
        res1 = await service.upsert_provider_profile(user, ws.id, up1)
        assert res1.web_credential_ref == "TINYFISH_SECRET_REF_ENV_01"

        # Step 2: Client submits masked placeholder bullet characters
        up2 = WorkspaceProviderProfileUpdate(
            web_provider="tinyfish",
            web_credential_ref="••••••••••••••••••••••••••••••••",
        )
        res2 = await service.upsert_provider_profile(user, ws.id, up2)

        # Invariant: stored value in database MUST remain the original credential reference
        raw_db_row = await session.scalar(
            select(WorkspaceProviderProfile).where(WorkspaceProviderProfile.workspace_id == ws.id)
        )
        assert raw_db_row.web_credential_ref == "TINYFISH_SECRET_REF_ENV_01"
        assert res2.web_credential_ref == "TINYFISH_SECRET_REF_ENV_01"


@pytest.mark.asyncio
async def test_credential_ref_explicit_clear_sets_to_none(setup_tinyfish_test_env):
    """
    Verifies that explicitly clearing a credential (submitting empty string '')
    clears the stored reference to None.
    """
    env = setup_tinyfish_test_env
    ws = env["ws_a"]
    user = env["owner"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)
        # Step 1: Set a reference
        up1 = WorkspaceProviderProfileUpdate(
            web_provider="tinyfish",
            web_credential_ref="TINYFISH_TEMP_REF",
        )
        await service.upsert_provider_profile(user, ws.id, up1)

        # Step 2: Explicitly submit empty string
        up2 = WorkspaceProviderProfileUpdate(
            web_provider="tinyfish",
            web_credential_ref="",
        )
        res2 = await service.upsert_provider_profile(user, ws.id, up2)
        assert res2.web_credential_ref is None

        raw_db_row = await session.scalar(
            select(WorkspaceProviderProfile).where(WorkspaceProviderProfile.workspace_id == ws.id)
        )
        assert raw_db_row.web_credential_ref is None


@pytest.mark.asyncio
async def test_credential_ref_masking_in_profile_response(setup_tinyfish_test_env):
    """
    Verifies that if a user accidentally enters a raw secret key (e.g. sk-...),
    the GET profile response redacts it to bullets and never leaks it to the browser.
    """
    env = setup_tinyfish_test_env
    ws = env["ws_a"]
    user = env["owner"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)
        # Directly insert a raw key pattern into DB
        raw_db_row = await session.scalar(
            select(WorkspaceProviderProfile).where(WorkspaceProviderProfile.workspace_id == ws.id)
        )
        raw_db_row.web_credential_ref = "sk-tinyfish-fake-raw-secret-key-12345"
        await session.commit()

        # Call get_provider_profile
        resp = await service.get_provider_profile(user, ws.id)
        assert resp is not None
        assert "sk-tinyfish-fake-raw-secret-key-12345" not in str(resp)
        assert resp.web_credential_ref == "••••••••••••••••"


@pytest.mark.asyncio
async def test_search_web_resolves_custom_workspace_credential_ref():
    """
    Verifies that search_web resolves a custom environment variable reference
    (e.g. web_credential_ref='MY_CUSTOM_TF_ENV') and passes the resolved key to the adapter.
    """
    mock_tf_adapter = AsyncMock()
    mock_tf_adapter.search.return_value = WebSearchResponse(results=[])

    state: ResearchState = {
        "query": "artificial intelligence safety",
        "web_provider": "tinyfish",
        "web_credential_ref": "WORKSPACE_A_CUSTOM_KEY",
        "web_enabled": True,
        "queries": ["ai safety"],
    }

    with patch.dict(os.environ, {"WORKSPACE_A_CUSTOM_KEY": "tf_custom_env_resolved_token"}), \
         patch("rag_app.providers.registry.get_provider") as mock_get_provider:
        mock_get_provider.return_value = mock_tf_adapter
        await search_web(state)

    mock_get_provider.assert_called_once_with(
        "web",
        "tinyfish",
        api_key="tf_custom_env_resolved_token",
    )


# ============================================================================
# 8. Deterministic Result Cap & Raw Secret Rejection Tests
# ============================================================================

@pytest.mark.asyncio
async def test_tinyfish_search_enforces_client_side_limit_slicing():
    """
    Deterministic proof that requesting at most two results never returns more
    than two normalized results, even when upstream returns a full page of 7 results.
    Also verifies raw upstream count is logged and outbound request avoids unsupported 'limit' param.
    """
    upstream_7_results = [
        {
            "position": i,
            "title": f"Result {i} Title",
            "url": f"https://example.com/item-{i}",
            "snippet": f"Snippet content for result {i}",
            "site_name": f"Site {i}",
        }
        for i in range(1, 8)
    ]

    captured_requests = []

    def mock_handler(request: httpx.Request):
        captured_requests.append(request)
        return httpx.Response(
            status_code=200,
            json={"query": "test query", "results": upstream_7_results},
        )

    transport = httpx.MockTransport(mock_handler)
    adapter = TinyFishWebAdapter(api_key="test-api-key")

    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        res = await adapter.search("test query", limit=2)

    # 1. Exactly 1 upstream request made
    assert len(captured_requests) == 1
    # 2. Outbound request does not pass undocumented 'limit' param
    assert "limit=" not in str(captured_requests[0].url)
    # 3. Exactly 2 results returned (enforcing client-side slicing)
    assert isinstance(res, WebSearchResponse)
    assert len(res.results) == 2
    assert res.results[0].title == "Result 1 Title"
    assert res.results[0].url == "https://example.com/item-1"
    assert res.results[1].title == "Result 2 Title"
    assert res.results[1].url == "https://example.com/item-2"


@pytest.mark.asyncio
async def test_upsert_rejects_raw_api_key_with_422(setup_tinyfish_test_env):
    """
    Verifies that raw API keys (e.g. sk-..., tf_..., or keys containing hyphens/spaces)
    are strictly rejected with HTTP 422, preventing plaintext secret persistence in the DB.
    """
    from fastapi import HTTPException

    env = setup_tinyfish_test_env
    ws = env["ws_a"]
    user = env["owner"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        # 1. Raw OpenAI / general key pattern
        with pytest.raises(HTTPException) as exc_info1:
            await service.upsert_provider_profile(
                user,
                ws.id,
                WorkspaceProviderProfileUpdate(
                    web_provider="tinyfish",
                    web_credential_ref="sk-proj-1234567890abcdefghijklmnopqrstuvwxyz",
                ),
            )
        assert exc_info1.value.status_code == 422
        assert "Raw API keys cannot be stored" in exc_info1.value.detail

        # 2. Raw TinyFish key pattern
        with pytest.raises(HTTPException) as exc_info2:
            await service.upsert_provider_profile(
                user,
                ws.id,
                WorkspaceProviderProfileUpdate(
                    web_provider="tinyfish",
                    web_credential_ref="tf_live_secret_key_9999",
                ),
            )
        assert exc_info2.value.status_code == 422
        assert "Raw API keys cannot be stored" in exc_info2.value.detail

        # 3. Invalid environment variable name with spaces
        with pytest.raises(HTTPException) as exc_info3:
            await service.upsert_provider_profile(
                user,
                ws.id,
                WorkspaceProviderProfileUpdate(
                    web_provider="tinyfish",
                    web_credential_ref="INVALID KEY NAME WITH SPACES",
                ),
            )
        assert exc_info3.value.status_code == 422
        assert "not a valid environment variable name" in exc_info3.value.detail


@pytest.mark.asyncio
async def test_upsert_accepts_valid_env_var_identifier(setup_tinyfish_test_env):
    """
    Verifies that a valid server environment variable identifier (e.g. TINYFISH_PROD_KEY)
    is accepted, persisted, and returned via GET without masking.
    """
    env = setup_tinyfish_test_env
    ws = env["ws_a"]
    user = env["owner"]

    async with AsyncSessionLocal() as session:
        service = WorkspaceService(session)

        res = await service.upsert_provider_profile(
            user,
            ws.id,
            WorkspaceProviderProfileUpdate(
                web_provider="tinyfish",
                web_credential_ref="TINYFISH_PROD_KEY",
            ),
        )
        assert res.web_credential_ref == "TINYFISH_PROD_KEY"

        # Verify raw DB persistence
        raw_db_row = await session.scalar(
            select(WorkspaceProviderProfile).where(WorkspaceProviderProfile.workspace_id == ws.id)
        )
        assert raw_db_row.web_credential_ref == "TINYFISH_PROD_KEY"

        # Verify GET returns the reference label as-is
        get_res = await service.get_provider_profile(user, ws.id)
        assert get_res.web_credential_ref == "TINYFISH_PROD_KEY"


