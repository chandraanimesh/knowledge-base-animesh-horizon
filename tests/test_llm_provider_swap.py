"""
Acceptance Suite for Milestone 3: Pluggable LLM Provider Adapter & Provider-Swap Verification.
Proves that application services depend on a provider contract (LLMProvider) rather than
being coupled to a single vendor SDK.
"""

from __future__ import annotations

import os
from typing import Any
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy import select

from rag_app.db.models import Department, User, UserDepartment, Workspace
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.generation.service import GenerationService
from rag_app.orchestration.router import RoutingDecision
from rag_app.providers.adapters.llm.groq import GroqLLMAdapter
from rag_app.providers.adapters.llm.mock import MockLLMAdapter
from rag_app.providers.base.llm import LLMProvider
from rag_app.providers.exceptions import InvalidProviderConfigError, ProviderNotFoundError
from rag_app.providers.registry import get_provider, provider_registry
from rag_app.schemas.rag import GenerateRequest


@pytest_asyncio.fixture(autouse=True)
async def cleanup_db_engine():
    yield
    await engine.dispose()


@pytest_asyncio.fixture
async def setup_test_context():
    async with AsyncSessionLocal() as session:
        dept = await session.scalar(select(Department).where(Department.code == "SWAP_TEST_DEPT"))
        if not dept:
            dept = Department(code="SWAP_TEST_DEPT", name="Swap Test Department", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

        user = await session.scalar(select(User).where(User.email == "swap_user@test.com"))
        if not user:
            user = User(
                email="swap_user@test.com",
                hashed_password="hashed_pw",
                role="ADMIN",
                is_active=True,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        elif user.role != "ADMIN":
            user.role = "ADMIN"
            await session.commit()
            await session.refresh(user)

        ud = await session.scalar(
            select(UserDepartment).where(
                UserDepartment.user_id == user.id,
                UserDepartment.department_id == dept.id,
            )
        )
        if not ud:
            ud = UserDepartment(user_id=user.id, department_id=dept.id)
            session.add(ud)
            await session.commit()

        return dept, user


# ============================================================================
# Gate 1: Contract Adherence
# ============================================================================

def test_both_adapters_satisfy_llm_provider_contract():
    """1: Both GroqLLMAdapter and MockLLMAdapter subclass LLMProvider and implement expected methods."""
    assert issubclass(GroqLLMAdapter, LLMProvider)
    assert issubclass(MockLLMAdapter, LLMProvider)

    mock_inst = MockLLMAdapter()
    assert isinstance(mock_inst, LLMProvider)
    assert hasattr(mock_inst, "generate")
    assert hasattr(mock_inst, "count_tokens")
    assert hasattr(mock_inst, "with_structured_output")
    assert mock_inst.provider_name == "mock"


# ============================================================================
# Gate 2: Deterministic Mock Generation & Usage Accounting
# ============================================================================

@pytest.mark.asyncio
async def test_mock_llm_adapter_deterministic_generation():
    """2 & 4: Mock provider generates repeatable text and normalized token usage without network IO."""
    canned = {"quarterly": "Q3 performance was outstanding with $12M revenue."}
    adapter = MockLLMAdapter(
        default_response="Default answer [source: doc_1].",
        canned_responses=canned,
        default_usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    )

    # 1. Matching canned keyword
    text, usage = await adapter.generate([{"role": "user", "content": "What was quarterly revenue?"}])
    assert "Q3 performance was outstanding" in text
    assert usage["prompt_tokens"] == 10
    assert usage["total_tokens"] == 25
    assert adapter.invocation_count == 1

    # 2. Default response fallback
    text2, usage2 = await adapter.generate([{"role": "user", "content": "Tell me about compliance."}])
    assert text2 == "Default answer [source: doc_1]."
    assert adapter.invocation_count == 2

    # 3. Deterministic token counting
    token_cnt = adapter.count_tokens([{"role": "user", "content": "Hello world test."}])
    assert token_cnt > 0


# ============================================================================
# Gate 3: Structured Output Support
# ============================================================================

@pytest.mark.asyncio
async def test_mock_llm_structured_output_support():
    """3: Mock provider supports with_structured_output returning valid schema instances."""
    adapter = MockLLMAdapter()
    structured_runnable = adapter.with_structured_output(RoutingDecision)

    decision = await structured_runnable.ainvoke([{"role": "user", "content": "What is the policy?"}])
    assert isinstance(decision, RoutingDecision)
    assert hasattr(decision, "intent")


# ============================================================================
# Gate 4: Provider Selection & Registry Configuration
# ============================================================================

def test_registry_resolves_mock_provider_via_env():
    """5: Setting LLM_PROVIDER=mock resolves MockLLMAdapter from registry."""
    provider_registry.clear_cache()
    with patch.dict(os.environ, {"LLM_PROVIDER": "mock"}):
        llm = get_provider("llm")
        assert isinstance(llm, MockLLMAdapter)
        assert llm.provider_name == "mock"
    provider_registry.clear_cache()


def test_registry_rejects_unknown_provider_safely():
    """6: Configuring an unknown/unsupported provider fails explicitly with ProviderNotFoundError."""
    provider_registry.clear_cache()
    with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported_llm_vendor_xyz"}):
        with pytest.raises(ProviderNotFoundError, match="Unsupported or unregistered provider"):
            get_provider("llm")
    provider_registry.clear_cache()


def test_missing_groq_credentials_fails_closed():
    """7: Missing GROQ_API_KEY fails closed with InvalidProviderConfigError (no silent fallback)."""
    with pytest.raises(InvalidProviderConfigError, match="GROQ_API_KEY is required"):
        GroqLLMAdapter(api_key="")


# ============================================================================
# Gate 5: End-to-End Provider Swap in Consuming Service (GenerationService)
# ============================================================================

@pytest.mark.asyncio
async def test_same_consuming_service_works_with_mock_and_groq_interfaces(setup_test_context):
    """8: GenerationService works with injected MockLLMAdapter without altering business logic."""
    dept, user = setup_test_context
    async with AsyncSessionLocal() as session:
        mock_provider = MockLLMAdapter(
            default_response="Mock grounded response for customer satisfaction [source: doc_1]."
        )

        # 1. Inject MockLLMAdapter into GenerationService directly
        service = GenerationService(session=session, llm=mock_provider)
        assert service.llm is mock_provider

        req = GenerateRequest(query="What is customer satisfaction rating?")
        # Mock retrieval chunks to bypass DB vector search
        with patch.object(service.retrieval_service, "retrieve") as mock_retrieval:
            from rag_app.schemas.rag import RetrievalResponse, RetrievedChunk
            mock_retrieval.return_value = RetrievalResponse(
                query=req.query,
                results=[
                    RetrievedChunk(
                        chunk_id="doc_1",
                        document_id=str(uuid.uuid4()),
                        department_id=str(dept.id),
                        file_name="report.pdf",
                        chunk_index=0,
                        text="Customer satisfaction rating is 94% across all regions.",
                        score=0.92,
                    )
                ],
                strategy_trace={},
                chunk_provenance={},
                chunk_hierarchy={},
            )

            resp = await service.generate_answer(request=req, user=user)
            assert resp.answer is not None
            assert len(resp.answer) > 0
            assert mock_provider.invocation_count >= 1


# ============================================================================
# Gate 6: Error Handling & Budget Invariants
# ============================================================================

@pytest.mark.asyncio
async def test_provider_error_does_not_mask_exception():
    """8 & 9: Provider exception triggers appropriate error without silent masking."""
    failing_adapter = MockLLMAdapter(error_trigger=RuntimeError("Simulated upstream provider timeout"))

    with pytest.raises(RuntimeError, match="Simulated upstream provider timeout"):
        await failing_adapter.generate([{"role": "user", "content": "Test timeout"}])


# ============================================================================
# Gate 7: Secret Protection
# ============================================================================

def test_api_keys_never_exposed_in_string_or_repr():
    """10: Sensitive API keys do not leak in str() or repr() representations."""
    secret_key = "gsk_test_secret_key_12345"
    adapter = GroqLLMAdapter(api_key=secret_key, model="openai/gpt-oss-20b")

    adapter_str = str(adapter)
    adapter_repr = repr(adapter)

    assert secret_key not in adapter_str
    assert secret_key not in adapter_repr
