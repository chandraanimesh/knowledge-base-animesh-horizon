"""
Deterministic Validation Module for Grounded Drafts and Proposed Sensitive Actions.
Validates factual claims, citations, provenance, contradiction resolution,
output structure, and action/target authorization boundaries.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from pydantic import BaseModel, Field

from rag_app.generation.grounding import _extract_numbers, _is_number_in_evidence
from rag_app.schemas.workflow import TaskEvidenceRef

logger = logging.getLogger(__name__)

# Matches [chunk: <id>], [source: <id>], or [ev: <id>]
CITATION_PATTERN = re.compile(
    r"\[(?:chunk|source|ev):\s*([^\]]+)\]",
    re.IGNORECASE,
)

ALLOWLISTED_ACTIONS = {"CRM_CREATE_NOTE"}


class DraftValidationResult(BaseModel):
    """Result of deterministic draft validation."""
    is_valid: bool = Field(description="True if all grounding, citation, conflict, and action checks passed.")
    grounding_passed: bool = Field(default=True)
    citations_valid: bool = Field(default=True)
    conflicts_resolved: bool = Field(default=True)
    action_target_valid: bool = Field(default=True)
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    error_reasons: list[str] = Field(default_factory=list)


class DraftValidator:
    """
    Deterministic validator for grounded synthesis drafts and proposed side-effects.
    Prevents progression to approval or external execution if any safety or grounding constraint fails.
    """

    def __init__(self, allowlisted_actions: Optional[set[str]] = None):
        self.allowlisted_actions = allowlisted_actions or ALLOWLISTED_ACTIONS

    def validate_draft(
        self,
        draft_markdown: str,
        evidence_items: list[TaskEvidenceRef],
        intended_action: Optional[str] = None,
        target_record_id: Optional[str] = None,
        known_contradictions: Optional[list[str]] = None,
    ) -> DraftValidationResult:
        """
        Executes strict deterministic verification on draft content and proposed action.
        """
        clean_draft = (draft_markdown or "").strip()
        errors: list[str] = []
        unsupported: list[str] = []
        invalid_cits: list[str] = []
        unresolved: list[str] = []

        # 1. Non-empty draft check
        if not clean_draft:
            errors.append("Draft content cannot be empty.")
            return DraftValidationResult(
                is_valid=False,
                grounding_passed=False,
                error_reasons=errors,
            )

        # 2. Check evidence presence
        if not evidence_items:
            errors.append("Cannot validate draft: no authorized evidence items were provided.")
            return DraftValidationResult(
                is_valid=False,
                grounding_passed=False,
                error_reasons=errors,
            )

        # Build full evidence text and valid source identifiers
        all_evidence_text = " ".join(e.snippet for e in evidence_items)
        valid_source_ids: set[str] = set()
        for e in evidence_items:
            valid_source_ids.add(str(e.source_id).strip().lower())
            valid_source_ids.add(str(e.evidence_id).strip().lower())
            if e.task_key:
                valid_source_ids.add(str(e.task_key).strip().lower())

        # 3. Citation & Provenance Validation
        cited_ids = CITATION_PATTERN.findall(clean_draft)
        for cid in cited_ids:
            cid_norm = cid.strip().lower()
            if cid_norm not in valid_source_ids:
                invalid_cits.append(cid)
                errors.append(f"Citation reference '[source: {cid}]' does not match any authorized evidence source.")

        # 4. Deterministic Claim / Numeric Grounding
        extracted_numbers = _extract_numbers(clean_draft)
        for num in extracted_numbers:
            if not _is_number_in_evidence(num, all_evidence_text):
                unsupported.append(num)
                errors.append(f"Factual/numeric claim '{num}' is not supported by provided evidence.")

        # 5. Contradiction & Conflict Resolution
        contradictions = known_contradictions or []
        draft_lower = clean_draft.lower()
        for conflict in contradictions:
            conflict_lower = conflict.lower()
            # If evidence has an unresolved contradiction, the draft must explicitly acknowledge conflict/discrepancy
            acknowledgment_keywords = ["conflict", "discrepancy", "contradiction", "varies", "inconsistent", "differs"]
            if not any(kw in draft_lower for kw in acknowledgment_keywords):
                unresolved.append(conflict)
                errors.append(
                    f"Unresolved factual conflict '{conflict}' present in evidence but not acknowledged in draft."
                )

        # 6. Action and Target Validation
        action_valid = True
        if intended_action is not None:
            if intended_action not in self.allowlisted_actions:
                action_valid = False
                errors.append(f"Intended action '{intended_action}' is not in allowlisted actions: {self.allowlisted_actions}.")
            if not target_record_id or not str(target_record_id).strip():
                action_valid = False
                errors.append("Target record identifier cannot be empty for sensitive side-effect action.")

        is_valid = len(errors) == 0

        return DraftValidationResult(
            is_valid=is_valid,
            grounding_passed=len(unsupported) == 0,
            citations_valid=len(invalid_cits) == 0,
            conflicts_resolved=len(unresolved) == 0,
            action_target_valid=action_valid,
            unsupported_claims=unsupported,
            invalid_citations=invalid_cits,
            unresolved_conflicts=unresolved,
            error_reasons=errors,
        )
