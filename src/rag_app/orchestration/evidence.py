"""
Evidence Reducer and Aggregator for Adaptive Workflow Engine.
Collects, deduplicates, and evaluates evidence while preserving source attribution,
provenance metadata, and conflicting claims.
"""

import logging
from typing import Any, Optional

from rag_app.schemas.research import SufficiencyEvaluation, SufficiencyStatus
from rag_app.schemas.workflow import TaskEvidenceRef

logger = logging.getLogger(__name__)


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
