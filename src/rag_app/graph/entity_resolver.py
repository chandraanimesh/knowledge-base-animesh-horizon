"""
Canonical Entity Resolution for Stage I Relationship-Aware Retrieval.

Resolves entity and company references (e.g. "ABC", "ABC Company", aliases)
to the authoritative canonical KnowledgeEntity without introducing a second company registry.
Prevents unsafe entity merging when ambiguous name similarities exist.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import KnowledgeEntity, KnowledgeItem, TaskItem

logger = logging.getLogger(__name__)

# Common business entity suffixes to strip during soft normalisation comparison
COMPANY_SUFFIXES_REGEX = re.compile(
    r"\b(company|co\.?|corp\.?|corporation|inc\.?|incorporated|ltd\.?|limited|llc|gmbh|sa|bv|pvt\.?)\b",
    re.IGNORECASE,
)


def normalize_business_name(name: str) -> str:
    """
    Normalizes a business name by removing punctuation, extra spaces,
    and trailing legal suffixes (Inc, LLC, Corp, Co, Ltd, Company).
    """
    if not name:
        return ""
    # Strip punctuation except alphanumeric and space
    cleaned = re.sub(r"[^\w\s]", " ", name)
    # Remove business entity suffixes
    cleaned = COMPANY_SUFFIXES_REGEX.sub(" ", cleaned)
    # Collapse multiple whitespaces and lower
    return " ".join(cleaned.lower().split())


@dataclass
class EntityResolutionResult:
    entity: Optional[KnowledgeEntity] = None
    canonical_name: Optional[str] = None
    canonical_key: Optional[str] = None
    entity_id: Optional[uuid.UUID] = None
    is_ambiguous: bool = False
    candidates: list[str] = field(default_factory=list)
    clarification_prompt: Optional[str] = None
    matched_by: str = "NONE"  # EXACT, ALIAS, NORMALIZED, TASK_PROVENANCE, NONE


class CanonicalEntityResolver:
    """
    Authoritative Canonical Entity Resolver backed by PostgreSQL KnowledgeEntity
    and KnowledgeItem models.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def resolve_company(
        self,
        name_or_query: str,
        allowed_departments: Optional[list[uuid.UUID]] = None,
    ) -> EntityResolutionResult:
        """
        Resolves a raw company query term (e.g. 'ABC', 'ABC Company')
        to the canonical KnowledgeEntity.

        Ambiguity Guard:
        If multiple active entities match and cannot be distinguished via exact
        name or verified alias, marks is_ambiguous=True with candidate options
        to prompt for clarification instead of guessing or merging entities.
        """
        raw_name = name_or_query.strip()
        if not raw_name:
            return EntityResolutionResult(is_ambiguous=False, matched_by="NONE")

        norm_query = normalize_business_name(raw_name)

        # Base query for active, non-superseded, non-archived company entities
        stmt = (
            select(KnowledgeEntity, KnowledgeItem)
            .join(KnowledgeItem, KnowledgeItem.id == KnowledgeEntity.knowledge_item_id)
            .where(
                func.lower(KnowledgeEntity.entity_type) == "company",
                KnowledgeItem.status == "ACTIVE",
            )
        )

        if allowed_departments is not None:
            # Enforce department access boundary if restricted
            stmt = stmt.where(
                or_(
                    KnowledgeItem.department_id.is_(None),
                    KnowledgeItem.department_id.in_(allowed_departments),
                )
            )

        rows = (await self.session.execute(stmt)).all()
        if not rows:
            # Check if company exists in Stage H TaskItem provenance
            task_stmt = select(TaskItem.company_name).where(
                TaskItem.company_name.is_not(None),
                func.lower(TaskItem.company_name) == raw_name.lower(),
            ).limit(1)
            found_task_company = await self.session.scalar(task_stmt)
            if found_task_company:
                return EntityResolutionResult(
                    entity=None,
                    canonical_name=found_task_company,
                    is_ambiguous=False,
                    matched_by="TASK_PROVENANCE",
                )
            return EntityResolutionResult(is_ambiguous=False, matched_by="NONE")

        # -------------------------------------------------------------
        # Phase 2 Priority 1: Exact canonical_name match (case-insensitive)
        # -------------------------------------------------------------
        exact_matches: list[tuple[KnowledgeEntity, KnowledgeItem]] = []
        for ent, ki in rows:
            if ent.canonical_name.strip().lower() == raw_name.lower():
                exact_matches.append((ent, ki))

        if len(exact_matches) == 1:
            ent, ki = exact_matches[0]
            return EntityResolutionResult(
                entity=ent,
                canonical_name=ent.canonical_name,
                canonical_key=ki.canonical_key,
                entity_id=ent.id,
                is_ambiguous=False,
                matched_by="EXACT",
            )
        elif len(exact_matches) > 1:
            # Multiple exact match records (e.g. across workspaces/departments)
            # Pick approved shared over draft/workspace if available
            approved = [m for m in exact_matches if m[1].approval_status == "APPROVED"]
            chosen = approved[0] if approved else exact_matches[0]
            ent, ki = chosen
            return EntityResolutionResult(
                entity=ent,
                canonical_name=ent.canonical_name,
                canonical_key=ki.canonical_key,
                entity_id=ent.id,
                is_ambiguous=False,
                matched_by="EXACT",
            )

        # -------------------------------------------------------------
        # Phase 2 Priority 2: Verified Alias Match in attributes["aliases"]
        # -------------------------------------------------------------
        alias_matches: list[tuple[KnowledgeEntity, KnowledgeItem]] = []
        for ent, ki in rows:
            attrs = ent.attributes or {}
            aliases = attrs.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            for a in aliases:
                if str(a).strip().lower() == raw_name.lower() or normalize_business_name(str(a)) == norm_query:
                    alias_matches.append((ent, ki))
                    break

        if len(alias_matches) == 1:
            ent, ki = alias_matches[0]
            return EntityResolutionResult(
                entity=ent,
                canonical_name=ent.canonical_name,
                canonical_key=ki.canonical_key,
                entity_id=ent.id,
                is_ambiguous=False,
                matched_by="ALIAS",
            )
        elif len(alias_matches) > 1:
            # Distinct entities claim the same alias -> Ambiguity!
            candidate_names = sorted(list({m[0].canonical_name for m in alias_matches}))
            if len(candidate_names) > 1:
                return EntityResolutionResult(
                    is_ambiguous=True,
                    candidates=candidate_names,
                    clarification_prompt=(
                        f"The name '{raw_name}' matches multiple companies via aliases: "
                        f"{', '.join(candidate_names)}. Please clarify which company you meant."
                    ),
                    matched_by="ALIAS",
                )

        # -------------------------------------------------------------
        # Phase 2 Priority 3: Normalized Base Name Comparison
        # -------------------------------------------------------------
        normalized_matches: dict[str, list[tuple[KnowledgeEntity, KnowledgeItem]]] = {}
        for ent, ki in rows:
            ent_norm = normalize_business_name(ent.canonical_name)
            if ent_norm and (ent_norm == norm_query or norm_query in ent_norm or ent_norm in norm_query):
                cname = ent.canonical_name
                if cname not in normalized_matches:
                    normalized_matches[cname] = []
                normalized_matches[cname].append((ent, ki))

        if len(normalized_matches) == 1:
            cname = list(normalized_matches.keys())[0]
            ent, ki = normalized_matches[cname][0]
            return EntityResolutionResult(
                entity=ent,
                canonical_name=ent.canonical_name,
                canonical_key=ki.canonical_key,
                entity_id=ent.id,
                is_ambiguous=False,
                matched_by="NORMALIZED",
            )
        elif len(normalized_matches) > 1:
            # Ambiguity guard: Multiple distinct companies match
            # e.g., "ABC Technologies" and "ABC Retail" both match "ABC"
            candidate_names = sorted(list(normalized_matches.keys()))
            return EntityResolutionResult(
                is_ambiguous=True,
                candidates=candidate_names,
                clarification_prompt=(
                    f"Multiple companies match '{raw_name}': {', '.join(candidate_names)}. "
                    f"Please clarify which company you are inquiring about."
                ),
                matched_by="NORMALIZED",
            )

        # -------------------------------------------------------------
        # Phase 2 Priority 4: TaskItem Provenance Fallback
        # -------------------------------------------------------------
        task_stmt = select(TaskItem.company_name).where(
            TaskItem.company_name.is_not(None),
            or_(
                func.lower(TaskItem.company_name) == raw_name.lower(),
                func.lower(TaskItem.company_name).like(f"%{raw_name.lower()}%"),
            ),
        ).distinct()
        task_companies = (await self.session.scalars(task_stmt)).all()
        if len(task_companies) == 1:
            return EntityResolutionResult(
                entity=None,
                canonical_name=task_companies[0],
                is_ambiguous=False,
                matched_by="TASK_PROVENANCE",
            )
        elif len(task_companies) > 1:
            candidates = sorted(task_companies)
            return EntityResolutionResult(
                is_ambiguous=True,
                candidates=candidates,
                clarification_prompt=(
                    f"Multiple companies with follow-up tasks match '{raw_name}': "
                    f"{', '.join(candidates)}. Please clarify which company you meant."
                ),
                matched_by="TASK_PROVENANCE",
            )

        return EntityResolutionResult(is_ambiguous=False, matched_by="NONE")
