"""
Milestone 10G Verification Test Suite: Evidence Validation, Grounding & Learning Proposals.

Verifies:
1. Evidence grounding evaluation:
   - SUPPORTED: All claims in the answer are grounded in retrieved evidence.
   - PARTIALLY_SUPPORTED: Some claims lack citations/evidence or contain minor contradictions.
   - UNSUPPORTED: Answer claims contradict retrieved evidence or have zero evidence support.
2. Decoupling invariant:
   - Task execution success (`is_execution_successful=True`) is strictly decoupled from evidence quality.
   - A DAG run can execute all tasks cleanly with 100% success while the evidence evaluation is UNSUPPORTED.
3. Learning proposal service:
   - Proposal-only invariant: Learning proposals are created in PENDING_REVIEW status with requires_admin_review=True.
   - Never automatically promoted or written to canonical knowledge without administrator approval.
"""

from __future__ import annotations

import os
import uuid
import pytest

# Guarantee tests execute strictly against confirmed disposable database rag_disposable_test
os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.orchestration.evidence import (
    AnswerSupportStatus,
    EvidenceSupportEvaluation,
    EvidenceSupportValidator,
    LearningProposal,
    LearningProposalService,
    EvidenceAggregator,
)
from rag_app.schemas.workflow import TaskEvidenceRef


class MockEvidence:
    def __init__(self, snippet: str, source_id: str = "doc_1"):
        self.snippet = snippet
        self.source_id = source_id


def test_evaluation_fully_supported():
    """Verify SUPPORTED status when all statements match retrieved evidence."""
    evidence = [
        MockEvidence(snippet="The company expense policy requires receipts for all expenses over $25.", source_id="policy_doc_1"),
        MockEvidence(snippet="Meals are reimbursable up to a daily maximum of $75.", source_id="policy_doc_2"),
    ]
    answer = (
        "The company expense policy requires receipts for all expenses over $25. "
        "Additionally, meals are reimbursable up to a daily maximum of $75."
    )

    result = EvidenceSupportValidator.evaluate_support(
        answer=answer,
        evidence_items=evidence,
        is_execution_successful=True,
    )

    assert result.status == AnswerSupportStatus.SUPPORTED
    assert result.confidence == 1.0
    assert len(result.grounded_claims) >= 1
    assert len(result.unsupported_claims) == 0
    assert result.is_execution_successful is True
    assert "policy_doc_1" in result.evidence_references


def test_evaluation_partially_supported():
    """Verify PARTIALLY_SUPPORTED status when some statements lack evidence backing."""
    evidence = [
        MockEvidence(snippet="The company expense policy requires receipts for all expenses over $25.", source_id="policy_doc_1"),
    ]
    answer = (
        "The company expense policy requires receipts for all expenses over $25. "
        "Employees can also fly first-class on international flights exceeding four hours without manager approval."
    )

    result = EvidenceSupportValidator.evaluate_support(
        answer=answer,
        evidence_items=evidence,
        is_execution_successful=True,
    )

    assert result.status == AnswerSupportStatus.PARTIALLY_SUPPORTED
    assert 0.0 < result.confidence < 1.0
    assert len(result.grounded_claims) >= 1
    assert len(result.unsupported_claims) >= 1
    assert result.is_execution_successful is True


def test_evaluation_unsupported_no_evidence():
    """Verify UNSUPPORTED status when no evidence items are provided."""
    answer = "The company allows four weeks of paid paternity leave."

    result = EvidenceSupportValidator.evaluate_support(
        answer=answer,
        evidence_items=[],
        is_execution_successful=True,
    )

    assert result.status == AnswerSupportStatus.UNSUPPORTED
    assert result.confidence == 0.0
    assert "No supporting evidence items retrieved" in result.summary_reason
    assert result.is_execution_successful is True


def test_evaluation_unsupported_empty_answer():
    """Verify empty or whitespace answers are flagged as UNSUPPORTED."""
    result = EvidenceSupportValidator.evaluate_support(
        answer="   ",
        evidence_items=[MockEvidence(snippet="Some text")],
        is_execution_successful=True,
    )

    assert result.status == AnswerSupportStatus.UNSUPPORTED
    assert result.confidence == 0.0


def test_evaluation_with_unresolved_contradictions():
    """Verify contradictions degrade status and are recorded in evaluation."""
    evidence = [
        MockEvidence(snippet="Standard vacation allowance is 15 days.", source_id="handbook_2024"),
    ]
    answer = "Standard vacation allowance is 15 days."
    contradictions = ["Handbook 2025 states vacation allowance is 20 days."]

    result = EvidenceSupportValidator.evaluate_support(
        answer=answer,
        evidence_items=evidence,
        unresolved_contradictions=contradictions,
        is_execution_successful=True,
    )

    assert result.status == AnswerSupportStatus.PARTIALLY_SUPPORTED
    assert len(result.contradictions) == 1
    assert "contradiction" in result.summary_reason.lower()


def test_decoupling_execution_success_from_evidence_grounding():
    """
    CRITICAL INVARIANT TEST:
    Verifies that DAG execution success (tasks completed without throwing exceptions)
    is strictly independent from evidence grounding quality.
    A run where all 5 tasks succeeded but the evidence yielded no support must show:
      - is_execution_successful = True
      - status = UNSUPPORTED
    """
    # Simulate a run where DAG tasks executed perfectly, but the external query returned zero matching documents
    result = EvidenceSupportValidator.evaluate_support(
        answer="Our enterprise architecture utilizes quantum teleportation for database replication.",
        evidence_items=[],
        is_execution_successful=True,
    )

    # Execution succeeded at runtime
    assert result.is_execution_successful is True
    # But evidence grounding failed
    assert result.status == AnswerSupportStatus.UNSUPPORTED
    assert result.confidence == 0.0

    # Conversely, simulate an execution failure
    failed_exec_result = EvidenceSupportValidator.evaluate_support(
        answer="Some partial answer",
        evidence_items=[MockEvidence(snippet="Some partial answer")],
        is_execution_successful=False,
    )
    assert failed_exec_result.is_execution_successful is False
    assert failed_exec_result.status == AnswerSupportStatus.SUPPORTED


def test_learning_proposal_service_proposal_only_invariant():
    """
    Verifies that LearningProposalService produces proposals in PENDING_REVIEW status,
    with requires_admin_review=True, ensuring no unreviewed proposals become live knowledge.
    """
    run_id = str(uuid.uuid4())
    proposal = LearningProposalService.create_proposal(
        source_run_id=run_id,
        proposal_type="ROUTING_CORRECTION",
        title="Improve routing for refund policy queries",
        description="Queries containing 'refund' were routed to FAST_RAG instead of CONTEXTUAL.",
        proposed_payload={"pattern": r"\brefund\b", "target_mode": "CONTEXTUAL"},
    )

    assert isinstance(proposal, LearningProposal)
    assert proposal.status == "PENDING_REVIEW"
    assert proposal.requires_admin_review is True
    assert proposal.source_run_id == run_id
    assert proposal.proposal_type == "ROUTING_CORRECTION"
    assert proposal.proposed_payload["target_mode"] == "CONTEXTUAL"


def test_learning_proposal_types_supported():
    """Verifies creation of different proposal types (WORKFLOW_TEMPLATE, KNOWLEDGE_GAP)."""
    run_id = str(uuid.uuid4())

    workflow_proposal = LearningProposalService.create_proposal(
        source_run_id=run_id,
        proposal_type="WORKFLOW_TEMPLATE",
        title="Promote high-frequency DAG to workflow template",
        description="Multi-step research DAG has succeeded 10 times with high confidence.",
        proposed_payload={"template_name": "MultiSourceResearchTemplate"},
    )
    assert workflow_proposal.proposal_type == "WORKFLOW_TEMPLATE"
    assert workflow_proposal.status == "PENDING_REVIEW"

    gap_proposal = LearningProposalService.create_proposal(
        source_run_id=run_id,
        proposal_type="KNOWLEDGE_GAP",
        title="Missing documentation on Kubernetes disaster recovery",
        description="Query could not be grounded in internal docs.",
        proposed_payload={"missing_topic": "Kubernetes DR"},
    )
    assert gap_proposal.proposal_type == "KNOWLEDGE_GAP"
    assert gap_proposal.requires_admin_review is True


def test_evidence_aggregator_deduplication_and_contradiction():
    """Verifies EvidenceAggregator deduplicates duplicate snippets while tracking contradictions."""
    aggregator = EvidenceAggregator()

    ref1 = TaskEvidenceRef(
        evidence_id="ev_1",
        task_key="task_1",
        source_type="RAG",
        source_id="doc_101",
        snippet="Quarterly revenue grew by 15% in Q3.",
        contradictions=["Source B reports 12% revenue growth."],
    )
    # Duplicate snippet with same source
    ref2 = TaskEvidenceRef(
        evidence_id="ev_2",
        task_key="task_2",
        source_type="RAG",
        source_id="doc_101",
        snippet="Quarterly revenue grew by 15% in Q3.",
        contradictions=[],
    )
    # Distinct snippet
    ref3 = TaskEvidenceRef(
        evidence_id="ev_3",
        task_key="task_3",
        source_type="WEB",
        source_id="url_202",
        snippet="Market cap expanded following the Q3 earnings announcement.",
        contradictions=[],
    )

    added1 = aggregator.add_evidence(ref1)
    added2 = aggregator.add_evidence(ref2)
    added3 = aggregator.add_evidence(ref3)

    assert added1 is True
    assert added2 is False  # Duplicate ignored
    assert added3 is True

    all_evidence = aggregator.get_all_evidence()
    assert len(all_evidence) == 2

    contradictions = aggregator.get_contradictions()
    assert len(contradictions) == 1
    assert "Source B reports 12% revenue growth." in contradictions
