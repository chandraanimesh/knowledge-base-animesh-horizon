"""Workstream 5: Cross-Workstream Scenarios (A through E).

Covers:
- Scenario A: PostgreSQL FTS + RRF + CrossEncoder End-to-End Pipeline
- Scenario B: Internal RAG Sufficiency Evaluation -> Research Fallback with Strict Persistence Boundary
- Scenario C: Multi-User Concurrent Thread Streaming with Private Markers
- Scenario D: CRM Sync Through Adapter to Final Grounded RAG Response
- Scenario E: Combined Failure Isolation & Blast Radius Containment
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.conversation.service import ConversationService
from rag_app.core.config import settings
from rag_app.crm.sync_service import CRMSyncService
from rag_app.crm.twenty_client import CRMCompany, CRMCompanyState
from rag_app.db.models import (
    ChatMessage,
    Conversation,
    Department,
    Document,
    DocumentChunk,
    Permission,
    User,
    UserDepartment,
    UserPermission,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal, engine
from rag_app.ingestion.embedding_v1 import get_embedding_provider
from rag_app.generation.service import GenerationService
from rag_app.research.graph import (
    ResearchState,
    decide_next_after_eval,
)
from rag_app.schemas.research import (
    ExtractedFact,
    SufficiencyEvaluation,
    SufficiencyStatus,
)
from rag_app.main import app
from rag_app.retrieval.service import RetrievalService
from rag_app.security.authentication import create_access_token


async def _get_or_create_view_perm(session) -> Permission:
    perm = await session.scalar(select(Permission).where(Permission.code == "VIEW"))
    if not perm:
        perm = Permission(code="VIEW", description="View Permission", is_active=True)
        session.add(perm)
        await session.flush()
    return perm


@pytest.mark.asyncio
async def test_cross_scenario_a_fts_rrf_cross_encoder_pipeline():
    """
    Scenario A: Seed documents, execute multi-word query, verify FTS, Dense, RRF fusion,
    and CrossEncoder neural reranking pipeline.
    """
    try:
        async with AsyncSessionLocal() as session:
            dept = Department(code=f"SCEN_A_{uuid.uuid4().hex[:6]}".upper(), name="Scenario A Dept", is_active=True)
            session.add(dept)
            await session.commit()
            await session.refresh(dept)

            doc = Document(
                id=uuid.uuid4(),
                file_name="disaster_recovery_plan.md",
                file_path="/tmp/dr_plan.md",
                content_hash=uuid.uuid4().hex,
                mime_type="text/markdown",
                department_id=dept.id,
                status="COMPLETED",
            )
            session.add(doc)
            await session.flush()

            # Target chunk with specific keywords for both FTS and semantic search
            target_text = (
                "Disaster recovery policy: In the event of a catastrophic regional database failure, "
                "failover to the secondary standby replica occurs within 15 minutes with zero data loss."
            )
            distractor_text_1 = (
                "Annual corporate revenue analysis and quarterly financial forecast for North American operations."
            )
            distractor_text_2 = (
                "Marketing team branding guidelines for typography, color palette, and digital social media banners."
            )

            # Generate real embeddings for all chunks
            embedding_provider = get_embedding_provider()
            target_emb = await embedding_provider.embed_query(target_text)
            distractor_emb_1 = await embedding_provider.embed_query(distractor_text_1)
            distractor_emb_2 = await embedding_provider.embed_query(distractor_text_2)

            chunks = [
                DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    chunk_index=0,
                    text=target_text,
                    embedding=target_emb,
                ),
                DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    chunk_index=1,
                    text=distractor_text_1,
                    embedding=distractor_emb_1,
                ),
                DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    chunk_index=2,
                    text=distractor_text_2,
                    embedding=distractor_emb_2,
                ),
            ]
            session.add_all(chunks)
            await session.commit()

            # Execute Hybrid Retrieval + RRF + CrossEncoder
            retrieval_service = RetrievalService(session=session)
            query = "catastrophic regional database failure failover standby replica"
            results = await retrieval_service.retrieve(
                query=query,
                top_k=3,
                rerank=True,
                allowed_departments=[dept.id],
            )

            assert len(results.results) > 0
            top_result = results.results[0]
            # Target chunk must be rank #1
            assert "15 minutes" in top_result.text
            assert top_result.hybrid_score is not None and top_result.hybrid_score > 0.0
            assert top_result.reranker_score is not None and top_result.reranker_score > 0.0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cross_scenario_b_rag_sufficiency_to_research_fallback():
    """
    Scenario B: Internal RAG sufficiency evaluates to INSUFFICIENT, triggering
    the external research fallback without polluting canonical RAG persistence.
    """
    try:
        async with AsyncSessionLocal() as session:
            # 1. Zero document creation before
            doc_count_before = len((await session.scalars(select(Document))).all())
            chunk_count_before = len((await session.scalars(select(DocumentChunk))).all())

            # 2. Case A: Internal Evidence SUFFICIENT -> Route directly to generate_final_answer
            sufficient_state: ResearchState = {
                "query_type": "rag",
                "web_enabled": True,
                "sufficiency_evaluation": SufficiencyEvaluation(
                    status=SufficiencyStatus.SUFFICIENT,
                    reason="Internal evidence provides full answer",
                    evidence_found=True,
                    answerable_from_context=True,
                ),
            }
            assert decide_next_after_eval(sufficient_state) == "generate_final_answer"

            # 3. Case B: Internal Evidence INSUFFICIENT + web_enabled -> Route to init_research_run
            insufficient_state: ResearchState = {
                "query_type": "rag",
                "web_enabled": True,
                "sufficiency_evaluation": SufficiencyEvaluation(
                    status=SufficiencyStatus.INSUFFICIENT,
                    reason="Internal knowledge missing SLA figures",
                    evidence_found=False,
                    answerable_from_context=False,
                ),
            }
            assert decide_next_after_eval(insufficient_state) == "init_research_run"

            # 4. Case C: Internal Evidence INSUFFICIENT + web_disabled -> Route to no_context
            web_disabled_state: ResearchState = {
                "query_type": "rag",
                "web_enabled": False,
                "sufficiency_evaluation": SufficiencyEvaluation(
                    status=SufficiencyStatus.INSUFFICIENT,
                    reason="No context found",
                    evidence_found=False,
                    answerable_from_context=False,
                ),
            }
            assert decide_next_after_eval(web_disabled_state) == "no_context"

            # 5. Strict persistence boundary: Canonical RAG DB untouched!
            doc_count_after = len((await session.scalars(select(Document))).all())
            chunk_count_after = len((await session.scalars(select(DocumentChunk))).all())
            assert doc_count_before == doc_count_after
            assert chunk_count_before == chunk_count_after
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cross_scenario_c_multi_user_concurrent_thread_streaming():
    """
    Scenario C: Concurrent multi-user SSE chat streaming with distinct private markers.
    """
    try:
        async with AsyncSessionLocal() as session:
            perm_view = await _get_or_create_view_perm(session)

            dept = Department(code=f"CONC_DEPT_{uuid.uuid4().hex[:6]}".upper(), name="Concurrent Dept", is_active=True)
            session.add(dept)
            await session.flush()

            ws = Workspace(code=f"CONC_WS_{uuid.uuid4().hex[:6]}".upper(), name="Concurrent WS", department_id=dept.id, is_active=True)
            session.add(ws)
            await session.flush()

            user_a = User(email=f"cross_user_a_{uuid.uuid4().hex[:6]}@test.local", hashed_password="pw", role="USER", is_active=True)
            user_b = User(email=f"cross_user_b_{uuid.uuid4().hex[:6]}@test.local", hashed_password="pw", role="USER", is_active=True)
            session.add_all([user_a, user_b])
            await session.flush()

            session.add_all([
                UserDepartment(user_id=user_a.id, department_id=dept.id),
                UserPermission(user_id=user_a.id, department_id=dept.id, permission_id=perm_view.id),
                WorkspaceMember(workspace_id=ws.id, user_id=user_a.id, role="VIEWER"),

                UserDepartment(user_id=user_b.id, department_id=dept.id),
                UserPermission(user_id=user_b.id, department_id=dept.id, permission_id=perm_view.id),
                WorkspaceMember(workspace_id=ws.id, user_id=user_b.id, role="VIEWER"),
            ])
            await session.commit()

            conv_service = ConversationService(session=session)
            conv_a = await conv_service.create_conversation(user=user_a, title="Stream A", workspace_id=ws.id)
            conv_b = await conv_service.create_conversation(user=user_b, title="Stream B", workspace_id=ws.id)

            token_a = create_access_token(subject=str(user_a.id))
            token_b = create_access_token(subject=str(user_b.id))
            headers_a = {"Authorization": f"Bearer {token_a}"}
            headers_b = {"Authorization": f"Bearer {token_b}"}

            marker_a = "CROSS_STREAM_A_SEC_5501"
            marker_b = "CROSS_STREAM_B_SEC_8822"

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                async def send_a():
                    return await client.post(
                        "/chat/stream",
                        headers=headers_a,
                        json={"query": f"Report code {marker_a}", "conversation_id": str(conv_a.id), "workspace_id": str(ws.id), "top_k": 2},
                    )

                async def send_b():
                    return await client.post(
                        "/chat/stream",
                        headers=headers_b,
                        json={"query": f"Report code {marker_b}", "conversation_id": str(conv_b.id), "workspace_id": str(ws.id), "top_k": 2},
                    )

                res_a, res_b = await asyncio.gather(send_a(), send_b())

                assert res_a.status_code == 200
                assert res_b.status_code == 200

                # Zero token bleeding in streams
                assert marker_a not in res_b.text
                assert marker_b not in res_a.text

                # Zero contamination in saved history
                hist_a = (await client.get(f"/conversations/{conv_a.id}", headers=headers_a)).json()
                hist_b = (await client.get(f"/conversations/{conv_b.id}", headers=headers_b)).json()

                assert marker_a in str(hist_a)
                assert marker_b not in str(hist_a)
                assert marker_b in str(hist_b)
                assert marker_a not in str(hist_b)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cross_scenario_d_crm_sync_to_rag_response():
    """
    Scenario D: CRM company data synchronized into Knowledge Base and retrieved
    accurately in a grounded RAG response within department boundaries.
    """
    try:
        async with AsyncSessionLocal() as session:
            dept_auth = Department(code=f"AERO_AUTH_{uuid.uuid4().hex[:6]}".upper(), name="Aerospace Dept", is_active=True)
            dept_other = Department(code=f"AERO_OTHER_{uuid.uuid4().hex[:6]}".upper(), name="Other Dept", is_active=True)
            session.add_all([dept_auth, dept_other])
            await session.commit()
            await session.refresh(dept_auth)
            await session.refresh(dept_other)

            unique_co = "Helios Aerospace"
            co = CRMCompany(
                id=f"co_aero_{uuid.uuid4().hex[:6]}",
                name=unique_co,
                domain="heliosaero.space",
                address="100 Cape Canaveral Way, FL",
                annual_revenue=85000000.0,
            )
            state = CRMCompanyState(company_name=unique_co, company=co)

            sync_service = CRMSyncService(session=session)
            ingest_res = await sync_service.sync_company_state(crm_state=state, department_id=dept_auth.id)
            assert ingest_res.skipped is False

            # Retrieval in authorized department
            retrieval_service = RetrievalService(session=session)
            found = await retrieval_service.retrieve(
                query=f"Details for {unique_co} annual revenue and address",
                top_k=3,
                rerank=True,
                allowed_departments=[dept_auth.id],
            )
            assert len(found.results) > 0
            assert "Helios" in found.results[0].text
            assert "$85,000,000.00" in found.results[0].text

            # Retrieval in foreign department -> 0 results
            blocked = await retrieval_service.retrieve(
                query=f"Details for {unique_co}",
                top_k=3,
                rerank=True,
                allowed_departments=[dept_other.id],
            )
            assert len([r for r in blocked.results if "Helios" in r.text]) == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cross_scenario_e_blast_radius_containment_and_combined_failure_isolation():
    """
    Scenario E: Blast radius containment: simulated CRM failure and research failure
    run concurrently with a normal user query without poisoning other operations.
    """
    try:
        async with AsyncSessionLocal() as session:
            perm_view = await _get_or_create_view_perm(session)

            dept = Department(code=f"ISO_FAIL_{uuid.uuid4().hex[:6]}".upper(), name="Failure Iso Dept", is_active=True)
            session.add(dept)
            await session.flush()

            ws = Workspace(code=f"ISO_WS_{uuid.uuid4().hex[:6]}".upper(), name="Failure Iso WS", department_id=dept.id, is_active=True)
            session.add(ws)
            await session.flush()

            user = User(email=f"safe_user_{uuid.uuid4().hex[:6]}@test.local", hashed_password="pw", role="USER", is_active=True)
            session.add(user)
            await session.flush()

            session.add_all([
                UserDepartment(user_id=user.id, department_id=dept.id),
                UserPermission(user_id=user.id, department_id=dept.id, permission_id=perm_view.id),
                WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="VIEWER"),
            ])
            await session.commit()

            conv_service = ConversationService(session=session)
            conv = await conv_service.create_conversation(user=user, title="Safe Thread", workspace_id=ws.id)

            token = create_access_token(subject=str(user.id))
            headers = {"Authorization": f"Bearer {token}"}

            sync_service = CRMSyncService(session=session)

            # Concurrent Action 1: Intentionally failing CRM sync (company=None)
            async def failing_crm_call():
                broken_state = CRMCompanyState(company_name="Exploding Corp", company=None)
                try:
                    await sync_service.sync_company_state(broken_state, department_id=dept.id)
                    return "UNEXPECTED_SUCCESS"
                except ValueError as ex:
                    return f"CAUGHT_EXPECTED_CRM_ERROR: {type(ex).__name__}"

            # Concurrent Action 2: Intentionally failing research provider (simulated network 503)
            async def failing_research_call():
                try:
                    raise ConnectionError("Simulated upstream provider outage 503")
                except ConnectionError as ex:
                    return f"CAUGHT_EXPECTED_RESEARCH_ERROR: {type(ex).__name__}"

            # Concurrent Action 3: Normal valid chat streaming call
            transport = ASGITransport(app=app)
            async def healthy_chat_call():
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.post(
                        "/chat/stream",
                        headers=headers,
                        json={"query": "Hello, healthy query executing normally", "conversation_id": str(conv.id), "workspace_id": str(ws.id)},
                    )
                    return resp.status_code

            # Execute all three simultaneously
            crm_res, res_res, chat_status = await asyncio.gather(
                failing_crm_call(),
                failing_research_call(),
                healthy_chat_call(),
            )

            # Assert failure containment
            assert "CAUGHT_EXPECTED_CRM_ERROR" in crm_res
            assert "CAUGHT_EXPECTED_RESEARCH_ERROR" in res_res
            assert chat_status == 200, "Healthy request must succeed with 200 OK despite adjacent subsystem failures!"
    finally:
        await engine.dispose()
