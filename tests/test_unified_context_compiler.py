import uuid
import pytest
from datetime import datetime, timezone
from rag_app.context.compiler import ContextCompiler, estimate_tokens
from rag_app.context.contracts import SourceAuthority, ContextEvidence, ContextBudget
from rag_app.schemas.rag import RetrievedChunk, ChunkProvenance, ChunkHierarchyMetadata, ConflictNotice
from rag_app.generation.citations import validate_and_render_citations


def test_single_authoritative_internal_source():
    """Context correctness: One authoritative internal source answers a query."""
    compiler = ContextCompiler()
    dept_id = uuid.uuid4()
    chunk = RetrievedChunk(
        chunk_id="doc_chunk_1",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="financial_policy.pdf",
        chunk_index=0,
        text="Acme Corp annual recurring revenue reached $15M in 2025.",
        hybrid_score=0.95,
    )

    compiled = compiler.compile(
        chunks=[chunk],
        query="What was Acme Corp revenue in 2025?",
        allowed_departments=[dept_id],
    )

    assert len(compiled.chunks) == 1
    assert compiled.chunks[0].chunk_id == "doc_chunk_1"
    assert "Acme Corp annual recurring revenue reached $15M" in compiled.rendered_context
    assert compiled.dropped_count == 0
    assert compiled.budget_status == "NORMAL"
    assert len(compiled.selected_evidence) == 1
    assert compiled.selected_evidence[0].source_authority == SourceAuthority.DOCUMENT_EVIDENCE.value


def test_query_requiring_two_sources():
    """Context correctness: A query requiring two distinct sources."""
    compiler = ContextCompiler()
    dept_id = uuid.uuid4()
    chunk1 = RetrievedChunk(
        chunk_id="chunk_a",
        document_id="doc_a",
        department_id=str(dept_id),
        file_name="pricing_guide.pdf",
        chunk_index=0,
        text="Base subscription fee is $500 per month.",
        hybrid_score=0.90,
    )
    chunk2 = RetrievedChunk(
        chunk_id="chunk_b",
        document_id="doc_b",
        department_id=str(dept_id),
        file_name="discount_matrix.pdf",
        chunk_index=0,
        text="Volume discounts apply for teams larger than 20 seats.",
        hybrid_score=0.88,
    )

    compiled = compiler.compile(
        chunks=[chunk1, chunk2],
        query="What is the pricing and discount policy?",
        allowed_departments=[dept_id],
    )

    assert len(compiled.chunks) == 2
    assert "Base subscription fee is $500" in compiled.rendered_context
    assert "Volume discounts apply" in compiled.rendered_context
    assert compiled.dropped_count == 0


def test_conflicting_source_claims_preserved():
    """Context correctness: Conflicting source claims (relationship divergence) are protected from pruning."""
    from rag_app.schemas.rag import GraphEdgeProvenance
    compiler = ContextCompiler(max_context_tokens=100)  # Very tight budget
    dept_id = uuid.uuid4()

    chunk1 = RetrievedChunk(
        chunk_id="chunk_rev_old",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="report_2023.pdf",
        chunk_index=0,
        text="Company revenue is reported as 10 million dollars.",
        hybrid_score=0.95,
    )
    chunk2 = RetrievedChunk(
        chunk_id="chunk_rev_new",
        document_id="doc_2",
        department_id=str(dept_id),
        file_name="report_2024.pdf",
        chunk_index=0,
        text="Company revenue is reported as 25 million dollars.",
        hybrid_score=0.92,
    )

    provenance = {
        "chunk_rev_old": ChunkProvenance(
            chunk_id="chunk_rev_old",
            graph_path=[GraphEdgeProvenance(subject="Company", predicate="revenue", object="$10M")],
        ),
        "chunk_rev_new": ChunkProvenance(
            chunk_id="chunk_rev_new",
            graph_path=[GraphEdgeProvenance(subject="Company", predicate="revenue", object="$25M")],
        ),
    }

    compiled = compiler.compile(
        chunks=[chunk1, chunk2],
        query="What is the company revenue?",
        allowed_departments=[dept_id],
        chunk_provenance=provenance,
    )

    # Both conflicting chunks MUST be preserved deterministically despite tight budget
    assert len(compiled.conflicts) == 1
    assert compiled.conflicts[0].subject.lower() == "company"
    assert compiled.conflicts[0].predicate.lower() == "revenue"
    assert len(compiled.chunks) == 2
    assert {c.chunk_id for c in compiled.chunks} == {"chunk_rev_old", "chunk_rev_new"}


def test_unanswerable_query_empty_candidates():
    """Context correctness: Empty candidates yield safe, empty compiled context."""
    compiler = ContextCompiler()
    dept_id = uuid.uuid4()

    compiled = compiler.compile(
        chunks=[],
        query="What is the secret passkey?",
        allowed_departments=[dept_id],
    )

    assert len(compiled.chunks) == 0
    assert compiled.token_count == 0
    assert compiled.rendered_context == ""
    assert compiled.budget_status == "NORMAL"


def test_followup_query_preserves_recent_history():
    """Context correctness: Recent conversation turns preserved for reference resolution."""
    compiler = ContextCompiler(max_context_tokens=1500)
    dept_id = uuid.uuid4()

    history = [
        {"role": "user", "content": "Tell me about Project Titan."},
        {"role": "assistant", "content": "Project Titan is an autonomous robotics platform."},
        {"role": "user", "content": "Who is the engineering lead?"},
        {"role": "assistant", "content": "Dr. Sarah Connor leads the team."},
        {"role": "user", "content": "When is its launch date?"},
    ]

    chunk = RetrievedChunk(
        chunk_id="titan_launch_chunk",
        document_id="doc_titan",
        department_id=str(dept_id),
        file_name="titan_roadmap.pdf",
        chunk_index=0,
        text="Project Titan launch is targeted for Q4 2026.",
        hybrid_score=0.91,
    )

    compiled = compiler.compile(
        chunks=[chunk],
        query="When is its launch date?",
        conversation_history=history,
        allowed_departments=[dept_id],
    )

    assert len(compiled.compiled_conversation_history) >= 2
    # Most recent turn is preserved
    assert compiled.compiled_conversation_history[-1]["content"] == "When is its launch date?"
    assert len(compiled.chunks) == 1


def test_relevant_personal_memory_combined_with_internal_evidence():
    """Context correctness: Relevant personal memory combined with internal document evidence."""
    compiler = ContextCompiler()
    user_id = uuid.uuid4()
    dept_id = uuid.uuid4()

    mem = {
        "id": "mem_pref_1",
        "user_id": str(user_id),
        "content": "User prefers financial summaries in USD with bullet points.",
        "memory_type": "PREFERENCE",
    }
    chunk = RetrievedChunk(
        chunk_id="chunk_eu_sales",
        document_id="doc_sales",
        department_id=str(dept_id),
        file_name="sales_report.pdf",
        chunk_index=0,
        text="European sales reached 12M EUR.",
        hybrid_score=0.93,
    )

    compiled = compiler.compile(
        chunks=[chunk],
        query="Give me a sales summary.",
        personal_memories=[mem],
        user=type("UserMock", (), {"id": user_id, "role": "USER"})(),
        allowed_departments=[dept_id],
    )

    assert len(compiled.chunks) == 1
    assert len(compiled.compiled_memories) == 1
    assert "EUR" in compiled.rendered_context
    selected_kinds = {e.source_kind for e in compiled.selected_evidence}
    assert "rag_chunk" in selected_kinds
    assert "memory_fact" in selected_kinds or "memory_episode" in selected_kinds


def test_fresh_web_evidence_combined_with_internal_evidence():
    """Context correctness: Fresh web evidence combined with internal document evidence."""
    compiler = ContextCompiler()
    dept_id = uuid.uuid4()

    internal_chunk = RetrievedChunk(
        chunk_id="internal_spec_1",
        document_id="doc_internal",
        department_id=str(dept_id),
        file_name="product_spec.pdf",
        chunk_index=0,
        text="Internal system architecture v2.0 utilizes PostgreSQL 16.",
        hybrid_score=0.92,
    )
    web_chunk = RetrievedChunk(
        chunk_id="web_tinyfish_123",
        document_id="https://postgresql.org/news/release17",
        department_id="None",
        file_name="PostgreSQL 17 Release Notes",
        chunk_index=0,
        text="PostgreSQL 17 released with improved query planner.",
        hybrid_score=0.88,
    )

    compiled = compiler.compile(
        chunks=[internal_chunk, web_chunk],
        query="What PostgreSQL versions are available and used?",
        allowed_departments=[dept_id],
        is_web_context=True,
    )

    assert len(compiled.chunks) == 2
    web_ev = next(e for e in compiled.selected_evidence if e.source_id == "web_tinyfish_123")
    assert web_ev.is_external is True
    assert web_ev.source_authority == SourceAuthority.WEB_RESEARCH.value


def test_global_budget_accounting_and_output_reservation():
    """Budgeting: Output tokens are reserved and total context stays within model limit."""
    model_limit = 4000
    reserved_output = 500
    compiler = ContextCompiler(
        model_context_limit=model_limit,
        reserved_output_tokens=reserved_output,
        max_context_tokens=1500,
    )

    dept_id = uuid.uuid4()
    chunks = []
    for i in range(10):
        chunks.append(
            RetrievedChunk(
                chunk_id=f"large_chunk_{i}",
                document_id=f"doc_{i}",
                department_id=str(dept_id),
                file_name=f"file_{i}.pdf",
                chunk_index=0,
                text=f"Unique distinct content for document {i} with facts: " + ("detailed evidence fact sentence for testing budget allocation " * 20),  # ~160 tokens each
                hybrid_score=0.95 - (i * 0.05),
            )
        )

    compiled = compiler.compile(
        chunks=chunks,
        query="Summarize all documents.",
        allowed_departments=[dept_id],
    )

    # Total planned tokens MUST NOT exceed model_limit - reserved_output
    assert compiled.total_planned_tokens <= (model_limit - reserved_output)
    assert compiled.dropped_count > 0
    assert "large_chunk_9" in compiled.dropped_reasons
    assert compiled.dropped_reasons["large_chunk_9"] == "TOKEN_BUDGET_EXCEEDED"


def test_deterministic_history_trimming_under_budget():
    """Budgeting: History is trimmed deterministically from oldest to newest."""
    compiler = ContextCompiler(max_context_tokens=1000)
    dept_id = uuid.uuid4()

    # Create 10 turns of history
    long_history = []
    for i in range(10):
        long_history.append({"role": "user", "content": f"User statement {i}: " + ("x" * 200)})
        long_history.append({"role": "assistant", "content": f"Assistant reply {i}: " + ("y" * 200)})

    chunk = RetrievedChunk(
        chunk_id="chunk_primary",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="doc.pdf",
        chunk_index=0,
        text="Essential core document text.",
        hybrid_score=0.95,
    )

    compiled = compiler.compile(
        chunks=[chunk],
        query="Follow-up question",
        conversation_history=long_history,
        allowed_departments=[dept_id],
    )

    # Older history must be trimmed, recent history preserved
    assert len(compiled.compiled_conversation_history) < len(long_history)
    # The most recent assistant reply must be present in the retained history
    assert any("Assistant reply 9" in msg["content"] for msg in compiled.compiled_conversation_history)


def test_dropped_chunks_cannot_be_cited():
    """Budgeting: Chunks dropped by compiler cannot be cited in answer."""
    compiler = ContextCompiler(max_context_tokens=80)
    dept_id = uuid.uuid4()

    chunk1 = RetrievedChunk(
        chunk_id="chunk_keep",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="keep.pdf",
        chunk_index=0,
        text="High priority evidence text. " + ("A" * 200),
        hybrid_score=0.98,
    )
    chunk2 = RetrievedChunk(
        chunk_id="chunk_dropped",
        document_id="doc_2",
        department_id=str(dept_id),
        file_name="dropped.pdf",
        chunk_index=0,
        text="Low priority evidence text that gets dropped. " + ("B" * 200),
        hybrid_score=0.40,
    )

    compiled = compiler.compile(
        chunks=[chunk1, chunk2],
        query="Test query",
        allowed_departments=[dept_id],
    )

    assert "chunk_keep" in [c.chunk_id for c in compiled.chunks]
    assert "chunk_dropped" not in [c.chunk_id for c in compiled.chunks]

    # Citation validation against compiled chunks MUST reject chunk_dropped
    raw_answer = "This is verified by [chunk: chunk_keep] but also [chunk: chunk_dropped]."
    sanitized_answer, citations = validate_and_render_citations(raw_answer, compiled.chunks)

    assert "[chunk: chunk_keep]" in sanitized_answer
    assert "[chunk: chunk_dropped]" not in sanitized_answer
    assert citations == ["keep.pdf"]


def test_cross_user_personal_memory_excluded():
    """Security: Personal memory from a different user is strictly excluded."""
    compiler = ContextCompiler()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    dept_id = uuid.uuid4()

    mem_user_b = {
        "id": "mem_leak_risk",
        "user_id": str(user_b),
        "content": "Secret salary note: $200k",
        "memory_type": "FACT",
    }
    chunk = RetrievedChunk(
        chunk_id="public_doc",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="public.pdf",
        chunk_index=0,
        text="Company holiday is next Monday.",
        hybrid_score=0.9,
    )

    compiled = compiler.compile(
        chunks=[chunk],
        query="What is my salary?",
        personal_memories=[mem_user_b],
        user=type("UserMock", (), {"id": user_a, "role": "USER"})(),
        allowed_departments=[dept_id],
    )

    assert len(compiled.compiled_memories) == 0
    assert "mem_leak_risk" in compiled.dropped_reasons
    assert compiled.dropped_reasons["mem_leak_risk"] == "UNAUTHORIZED_USER_MEMORY"
    assert not any(e.source_id == "mem_leak_risk" for e in compiled.selected_evidence)


def test_cross_workspace_thread_excluded():
    """Security: Workspace thread from another workspace is excluded."""
    compiler = ContextCompiler()
    ws_current = uuid.uuid4()
    ws_other = uuid.uuid4()
    dept_id = uuid.uuid4()

    thread_other = {
        "message_id": "msg_other_1",
        "thread_id": "thread_other_1",
        "workspace_id": str(ws_other),
        "content": "Confidential discussion from other department workspace.",
    }
    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="doc.pdf",
        chunk_index=0,
        text="General notes.",
        hybrid_score=0.9,
    )

    compiled = compiler.compile(
        chunks=[chunk],
        query="Show discussions",
        workspace_thread_context=[thread_other],
        workspace_id=ws_current,
        allowed_departments=[dept_id],
    )

    assert len(compiled.compiled_workspace_threads) == 0
    assert "msg_other_1" in compiled.dropped_reasons
    assert compiled.dropped_reasons["msg_other_1"] == "CROSS_WORKSPACE_THREAD_EXCLUDED"


def test_global_chat_thread_isolation_policy():
    """Security: In Global Chat (workspace_id=None), raw employee threads are isolated/excluded."""
    compiler = ContextCompiler()
    dept_id = uuid.uuid4()

    thread_item = {
        "message_id": "msg_raw_1",
        "thread_id": "thread_raw_1",
        "workspace_id": str(uuid.uuid4()),
        "content": "Raw internal employee discussion.",
    }
    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        department_id=str(dept_id),
        file_name="doc.pdf",
        chunk_index=0,
        text="General policy.",
        hybrid_score=0.9,
    )

    compiled = compiler.compile(
        chunks=[chunk],
        query="Policy overview",
        workspace_thread_context=[thread_item],
        workspace_id=None,  # Global Chat
        allowed_departments=[dept_id],
    )

    assert len(compiled.compiled_workspace_threads) == 0
    assert "msg_raw_1" in compiled.dropped_reasons
    assert compiled.dropped_reasons["msg_raw_1"] == "GLOBAL_CHAT_THREAD_ISOLATION"


def test_unapproved_shared_knowledge_excluded():
    """Security: Unapproved shared knowledge is excluded from context compilation."""
    compiler = ContextCompiler()
    dept_id = uuid.uuid4()

    unapproved_shared_chunk = RetrievedChunk(
        chunk_id="shared_kn_unapproved_1",
        document_id="shared_kn_doc_1",
        department_id=str(dept_id),
        file_name="Shared Knowledge: unapproved_draft",
        chunk_index=0,
        text="Proposed pricing change to $100. Status: Unapproved",
        hybrid_score=0.95,
    )
    approved_chunk = RetrievedChunk(
        chunk_id="shared_kn_approved_2",
        document_id="shared_kn_doc_2",
        department_id=str(dept_id),
        file_name="Shared Knowledge: approved_policy",
        chunk_index=0,
        text="[Approved Shared Organizational Knowledge: policy]\nStandard price is $50. Status: Approved",
        hybrid_score=0.94,
    )

    compiled = compiler.compile(
        chunks=[unapproved_shared_chunk, approved_chunk],
        query="What is the price?",
        allowed_departments=[dept_id],
    )

    assert len(compiled.chunks) == 1
    assert compiled.chunks[0].chunk_id == "shared_kn_approved_2"
    assert "shared_kn_unapproved_1" in compiled.dropped_reasons
    assert compiled.dropped_reasons["shared_kn_unapproved_1"] == "UNAPPROVED_SHARED_KNOWLEDGE"


def test_unauthorized_department_chunk_raises_security_violation():
    """Security: Unauthorized internal department chunk raises RuntimeError."""
    compiler = ContextCompiler()
    dept_authorized = uuid.uuid4()
    dept_unauthorized = uuid.uuid4()

    bad_chunk = RetrievedChunk(
        chunk_id="hr_chunk_secret",
        document_id="doc_hr",
        department_id=str(dept_unauthorized),
        file_name="hr_salaries.pdf",
        chunk_index=0,
        text="Executive compensation details.",
        hybrid_score=0.99,
    )

    with pytest.raises(RuntimeError, match="Security violation: Unauthorized chunk detected"):
        compiler.compile(
            chunks=[bad_chunk],
            query="Tell me about salaries.",
            allowed_departments=[dept_authorized],
            is_web_context=False,
        )
