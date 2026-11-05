"""
Evidence Reducer and Aggregator for Adaptive Workflow Engine.
Collects, deduplicates, and evaluates evidence while preserving source attribution,
provenance metadata, and conflicting claims.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field

from rag_app.schemas.research import SufficiencyEvaluation, SufficiencyStatus
from rag_app.schemas.workflow import TaskEvidenceRef

logger = logging.getLogger(__name__)


class AnswerSupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class EvidenceSupportEvaluation(BaseModel):
    """
    Evaluates whether answer claims are grounded in retrieved evidence.
    Crucially separates task execution success from evidence quality.
    """
    model_config = ConfigDict(from_attributes=True)

    status: AnswerSupportStatus
    grounded_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    summary_reason: str = ""
    is_execution_successful: bool = True  # Independent of evidence quality


class LearningProposal(BaseModel):
    """
    Structured learning proposal produced from an execution run.
    Proposal-only invariant: Never automatically promoted without administrator review.
    """
    model_config = ConfigDict(from_attributes=True)

    proposal_id: str
    source_run_id: str
    proposal_type: str  # ROUTING_CORRECTION, WORKFLOW_TEMPLATE, KNOWLEDGE_GAP
    title: str
    description: str
    proposed_payload: dict[str, Any] = Field(default_factory=dict)
    status: str = "PENDING_REVIEW"
    requires_admin_review: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EvidenceSupportValidator:
    """
    Validates answer grounding and factual support against retrieved evidence.
    Distinguishes execution status (successful execution of DAG tasks)
    from evidence quality (whether the retrieved text actually supports the generated claims).
    """

    @classmethod
    def evaluate_support(
        cls,
        answer: str,
        evidence_items: list[Any],
        unresolved_contradictions: Optional[list[str]] = None,
        is_execution_successful: bool = True,
    ) -> EvidenceSupportEvaluation:
        ans = (answer or "").strip()
        contradictions = unresolved_contradictions or []

        if not ans:
            return EvidenceSupportEvaluation(
                status=AnswerSupportStatus.UNSUPPORTED,
                summary_reason="Empty or null response cannot be grounded.",
                confidence=0.0,
                is_execution_successful=is_execution_successful,
            )

        if not evidence_items:
            return EvidenceSupportEvaluation(
                status=AnswerSupportStatus.UNSUPPORTED,
                unsupported_claims=[ans[:200]],
                summary_reason="No supporting evidence items retrieved.",
                confidence=0.0,
                is_execution_successful=is_execution_successful,
            )

        evidence_texts = []
        evidence_refs = []
        for e in evidence_items:
            if hasattr(e, "snippet"):
                evidence_texts.append(e.snippet.lower())
                evidence_refs.append(str(getattr(e, "source_id", "source")))
            elif hasattr(e, "text"):
                evidence_texts.append(e.text.lower())
                evidence_refs.append(str(getattr(e, "chunk_id", "chunk")))
            elif isinstance(e, dict):
                evidence_texts.append(str(e.get("snippet") or e.get("text") or "").lower())
                evidence_refs.append(str(e.get("source_id") or e.get("chunk_id") or "doc"))

        corpus = " ".join(evidence_texts)

        sentences = [s.strip() for s in ans.replace("\n", ". ").split(". ") if len(s.strip()) > 15]
        if not sentences:
            sentences = [ans]

        grounded = []
        unsupported = []

        for stmt in sentences:
            words = [w.lower().strip(".,;:\"'()[]") for w in stmt.split() if len(w) > 3]
            if not words:
                grounded.append(stmt)
                continue
            matches = sum(1 for w in words if w in corpus)
            match_ratio = matches / len(words)
            if match_ratio >= 0.4:
                grounded.append(stmt)
            else:
                unsupported.append(stmt)

        if contradictions:
            status = AnswerSupportStatus.PARTIALLY_SUPPORTED if grounded else AnswerSupportStatus.UNSUPPORTED
            reason = f"Answer has {len(grounded)} grounded claim(s) but contains {len(contradictions)} unresolved contradiction(s)."
            conf = 0.5
        elif not unsupported:
            status = AnswerSupportStatus.SUPPORTED
            reason = "All extracted claims are grounded in authorized retrieved evidence."
            conf = 1.0
        elif grounded and unsupported:
            status = AnswerSupportStatus.PARTIALLY_SUPPORTED
            reason = f"Partially supported: {len(grounded)} claims grounded, {len(unsupported)} claims lack clear evidence backing."
            conf = len(grounded) / len(sentences)
        else:
            status = AnswerSupportStatus.UNSUPPORTED
            reason = "No factual claims could be verified against retrieved evidence."
            conf = 0.0

        return EvidenceSupportEvaluation(
            status=status,
            grounded_claims=grounded,
            unsupported_claims=unsupported,
            contradictions=contradictions,
            evidence_references=evidence_refs,
            confidence=round(conf, 2),
            summary_reason=reason,
            is_execution_successful=is_execution_successful,
        )


class LearningProposalService:
    """
    Generates structured learning proposals (routing improvements, workflow templates, knowledge gaps)
    from adaptive runs.
    Guarantees proposal-only behavior: never automatically promotes unreviewed data to live company knowledge.
    """

    @classmethod
    def create_proposal(
        cls,
        source_run_id: str,
        proposal_type: str,
        title: str,
        description: str,
        proposed_payload: Optional[dict[str, Any]] = None,
    ) -> LearningProposal:
        return LearningProposal(
            proposal_id=str(uuid.uuid4()),
            source_run_id=source_run_id,
            proposal_type=proposal_type,
            title=title,
            description=description,
            proposed_payload=proposed_payload or {},
            status="PENDING_REVIEW",
            requires_admin_review=True,
        )


class EvidenceAggregator:
    """
    Evidence Aggregator and Reducer.
    Maintains provenance-bound evidence references across multiple execution tasks,
    surfaces factual contradictions, and evaluates evidence sufficiency.
    """

    def __init__(self):
        self._evidence_by_id: dict[str, TaskEvidenceRef] = {}
        self._dedup_keys: set[str] = set()
        self._contradictions: list[str] = []

    def add_evidence(self, item: TaskEvidenceRef) -> bool:
        """
        Adds an evidence reference. Deduplicates identical source excerpts
        while preserving distinct sources and differing viewpoints.
        """
        # Deduplication key based on source_type, source_id, and snippet normalized text
        norm_snippet = " ".join(item.snippet.split()).lower()
        dedup_key = f"{item.source_type}::{item.source_id}::{hash(norm_snippet)}"

        if dedup_key in self._dedup_keys:
            logger.debug(f"Duplicate evidence reference ignored: {item.source_id}")
            return False

        self._dedup_keys.add(dedup_key)
        self._evidence_by_id[item.evidence_id] = item

        # Collect contradictions if noted by extraction tasks
        for conflict in item.contradictions:
            if conflict not in self._contradictions:
                self._contradictions.append(conflict)

        return True

    def add_all(self, items: list[TaskEvidenceRef]) -> int:
        added = 0
        for item in items:
            if self.add_evidence(item):
                added += 1
        return added

    def get_all_evidence(self) -> list[TaskEvidenceRef]:
        return list(self._evidence_by_id.values())

    def get_contradictions(self) -> list[str]:
        return list(self._contradictions)

    def evaluate_sufficiency(
        self,
        goal: str,
        required_dimensions: Optional[list[str]] = None,
    ) -> SufficiencyEvaluation:
        """
        Evaluates whether the aggregated evidence is sufficient to answer the goal
        without hallucination or unsupported claims.
        """
        evidence_items = self.get_all_evidence()
        if not evidence_items:
            return SufficiencyEvaluation(
                status=SufficiencyStatus.INSUFFICIENT,
                evidence_found=False,
                required_information=required_dimensions or [goal],
                supported_information=[],
                missing_requirements=required_dimensions or [goal],
                contradictions=self._contradictions,
                answerable_from_context=False,
                reason="No evidence items were retrieved or extracted for the requested goal.",
                confidence=0.0,
                coverage=0.0,
            )

        # Check required dimensions against evidence
        dims = required_dimensions or [goal]
        supported: list[str] = []
        missing: list[str] = []

        all_text = " ".join(e.snippet.lower() for e in evidence_items)

        for dim in dims:
            dim_tokens = [w.lower() for w in dim.split() if len(w) > 3]
            if not dim_tokens or any(tok in all_text for tok in dim_tokens):
                supported.append(dim)
            else:
                missing.append(dim)

        coverage = len(supported) / max(len(dims), 1)

        # Determine status
        if missing and not supported:
            status = SufficiencyStatus.INSUFFICIENT
            answerable = False
            reason = f"Essential information missing: {', '.join(missing[:3])}"
        elif missing and supported:
            status = SufficiencyStatus.PARTIAL
            answerable = True
            reason = f"Partially supported ({len(supported)}/{len(dims)} items); missing: {', '.join(missing[:3])}"
        else:
            status = SufficiencyStatus.SUFFICIENT
            answerable = True
            reason = "All required information items are verified and supported by retrieved evidence."

        return SufficiencyEvaluation(
            status=status,
            evidence_found=True,
            required_information=dims,
            supported_information=supported,
            missing_requirements=missing,
            contradictions=self._contradictions,
            answerable_from_context=answerable,
            reason=reason,
            confidence=round(coverage, 2),
            coverage=round(coverage, 2),
        )
