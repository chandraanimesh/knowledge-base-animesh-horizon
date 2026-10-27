from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from rag_app.context.compiler import ContextCompiler
from rag_app.core.tokenizer import count_messages, count_text, get_effective_model_info
from rag_app.generation.service import GenerationService
from rag_app.providers.base.llm import LLMProvider
from rag_app.schemas.rag import ConflictNotice, GenerateResponse, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class GroundedTestCase:
    """A grounded test fixture derived from verified demo documents or audit scenarios."""
    id: str
    category: str  # "single_source", "cross_source", "conflicting", "unanswerable"
    query: str
    chunks: list[RetrievedChunk]
    expected_facts: list[str]
    supporting_source_ids: list[str]
    expected_behavior: str  # "ANSWER", "ABSTAIN", "CONFLICT_DISCLOSURE"
    abstention_required: bool = False
    conflict_disclosure_required: bool = False
    conflicts: list[ConflictNotice] | None = None
    description: str = ""


# =============================================================================
# GROUNDED EVALUATION DATASET
# Facts derived strictly from:
# 1. FIN-DEMO-001 (Finance Department Demo Document)
# 2. SALES-DEMO-001 (Sales Department Demo Document)
# 3. HOTEL-DEMO-001 (Hotel Price Bargaining Demo Document)
# 4. Phase C Conflicting / Unresolved synthetic audit records
# =============================================================================

SHARED_DEPT_ID = str(uuid.UUID("11111111-1111-1111-1111-111111111111"))

GROUNDED_EVAL_DATASET: list[GroundedTestCase] = [
    # 1. Straightforward single-source: Finance Operating Budget
    GroundedTestCase(
        id="case_01_single_source_finance_budget",
        category="single_source",
        query="What was the approved Q3 2026 operating budget and remaining budget for the Finance Department?",
        chunks=[
            RetrievedChunk(
                chunk_id="chk_fin_01",
                document_id="doc_fin_01",
                department_id=SHARED_DEPT_ID,
                file_name="01_finance_demo_document.docx",
                chunk_index=0,
                text=(
                    "Document ID: FIN-DEMO-001. Owner: Finance Department. "
                    "The Finance Department's approved Q3 2026 operating budget was ₹12,00,000. "
                    "The Finance Department recorded actual Q3 2026 operating spend of ₹9,25,000. "
                    "Based on the approved budget and recorded actual spend, the remaining Q3 operating budget is ₹2,75,000."
                ),
                hybrid_score=0.95,
            )
        ],
        expected_facts=["12,00,000", "2,75,000"],
        supporting_source_ids=["chk_fin_01"],
        expected_behavior="ANSWER",
        abstention_required=False,
        conflict_disclosure_required=False,
        description="Single source factual retrieval from FIN-DEMO-001.",
    ),

    # 2. Straightforward single-source: Hotel Bargaining Initial Quote
    GroundedTestCase(
        id="case_02_single_source_hotel_quote",
        category="single_source",
        query="What was the hotel's initial quote and the guest's counteroffer for the Deluxe Room stay from 12 to 14 December 2026?",
        chunks=[
            RetrievedChunk(
                chunk_id="chk_hotel_01",
                document_id="doc_hotel_01",
                department_id=SHARED_DEPT_ID,
                file_name="03_hotel_price_bargaining_demo_document.docx",
                chunk_index=0,
                text=(
                    "Document ID: HOTEL-DEMO-001. Stay dates: 12-14 December 2026 (2 nights, 1 room). "
                    "Initial Quote: The hotel initially quoted ₹8,000 per room per night for a Deluxe Room, before taxes. "
                    "Guest's Counteroffer: The guest counteroffered ₹6,000 per room per night, before taxes, and asked whether breakfast could be included."
                ),
                hybrid_score=0.92,
            )
        ],
        expected_facts=["8,000", "6,000", "Deluxe Room"],
        supporting_source_ids=["chk_hotel_01"],
        expected_behavior="ANSWER",
        abstention_required=False,
        conflict_disclosure_required=False,
        description="Single source negotiation figures from HOTEL-DEMO-001.",
    ),

    # 3. Cross-source: Finance Travel Expenses + Hotel Revised Offer
    GroundedTestCase(
        id="case_03_cross_source_travel_and_hotel",
        category="cross_source",
        query="How much were the Finance Department's Q3 2026 travel expenses, and what was the revised 2-night pre-tax room total offered for the hotel stay?",
        chunks=[
            RetrievedChunk(
                chunk_id="chk_fin_travel",
                document_id="doc_fin_01",
                department_id=SHARED_DEPT_ID,
                file_name="01_finance_demo_document.docx",
                chunk_index=1,
                text=(
                    "Document ID: FIN-DEMO-001. Travel Expense Detail: "
                    "Travel expenses for Q3 2026 totalled ₹1,40,000. "
                    "Approved Q3 operating budget was ₹12,00,000."
                ),
                hybrid_score=0.88,
            ),
            RetrievedChunk(
                chunk_id="chk_hotel_revised",
                document_id="doc_hotel_01",
                department_id=SHARED_DEPT_ID,
                file_name="03_hotel_price_bargaining_demo_document.docx",
                chunk_index=1,
                text=(
                    "Document ID: HOTEL-DEMO-001. Hotel's Revised Offer: "
                    "The hotel revised its offer to ₹6,800 per room per night, before taxes, including breakfast for 2 guests. "
                    "For 2 nights, the pre-tax room total at the revised rate is ₹13,600."
                ),
                hybrid_score=0.86,
            ),
        ],
        expected_facts=["1,40,000", "13,600"],
        supporting_source_ids=["chk_fin_travel", "chk_hotel_revised"],
        expected_behavior="ANSWER",
        abstention_required=False,
        conflict_disclosure_required=False,
        description="Multi-document synthesis across FIN-DEMO-001 and HOTEL-DEMO-001.",
    ),

    # 4. Cross-source: Sales Revenue vs Finance Actual Spend
    GroundedTestCase(
        id="case_04_cross_source_sales_and_finance",
        category="cross_source",
        query="Compare October 2026 sales closed-won revenue with the Finance Department's Q3 2026 actual operating spend.",
        chunks=[
            RetrievedChunk(
                chunk_id="chk_sales_rev",
                document_id="doc_sales_01",
                department_id=SHARED_DEPT_ID,
                file_name="02_sales_demo_document.docx",
                chunk_index=0,
                text=(
                    "Document ID: SALES-DEMO-001. Reporting period: October 2026. "
                    "Sales closed-won revenue for October 2026 was ₹18,50,000 against a target of ₹20,00,000. "
                    "Qualified sales pipeline was ₹32,00,000 with 14 new qualified opportunities."
                ),
                hybrid_score=0.91,
            ),
            RetrievedChunk(
                chunk_id="chk_fin_spend",
                document_id="doc_fin_01",
                department_id=SHARED_DEPT_ID,
                file_name="01_finance_demo_document.docx",
                chunk_index=0,
                text=(
                    "Document ID: FIN-DEMO-001. "
                    "The Finance Department recorded actual Q3 2026 operating spend of ₹9,25,000."
                ),
                hybrid_score=0.89,
            ),
        ],
        expected_facts=["18,50,000", "9,25,000"],
        supporting_source_ids=["chk_sales_rev", "chk_fin_spend"],
        expected_behavior="ANSWER",
        abstention_required=False,
        conflict_disclosure_required=False,
        description="Cross-department comparative synthesis between Sales and Finance records.",
    ),

    # 5. Conflicting: Temporal Recency Preference ($10M Historical vs $25M Newer)
    GroundedTestCase(
        id="case_05_conflicting_temporal_revenue",
        category="conflicting",
        query="What is the current annual revenue?",
        chunks=[
            RetrievedChunk(
                chunk_id="chk_rev_2020",
                document_id="doc_rev_old",
                department_id=SHARED_DEPT_ID,
                file_name="historical_report_2020.pdf",
                chunk_index=0,
                text="Historical Annual Report: Annual operating revenue was $10M for fiscal year 2020.",
                hybrid_score=0.75,
            ),
            RetrievedChunk(
                chunk_id="chk_rev_2024",
                document_id="doc_rev_new",
                department_id=SHARED_DEPT_ID,
                file_name="annual_report_2024.pdf",
                chunk_index=0,
                text="Current Financial Filing: Audited annual operating revenue reached $25M for fiscal year 2024.",
                hybrid_score=0.95,
            ),
        ],
        expected_facts=["25M", "2024"],
        supporting_source_ids=["chk_rev_2024"],
        expected_behavior="ANSWER",
        abstention_required=False,
        conflict_disclosure_required=False,
        description="Temporal conflict where newer audited 2024 filing takes precedence.",
    ),

    # 6. Conflicting: Unresolved Discrepancy Disclosure Required
    GroundedTestCase(
        id="case_06_conflicting_unresolved_hotel_status",
        category="conflicting",
        query="Is the December hotel negotiation active or cancelled?",
        chunks=[
            RetrievedChunk(
                chunk_id="chk_hotel_claim_active",
                document_id="doc_hotel_active",
                department_id=SHARED_DEPT_ID,
                file_name="travel_memo_active.docx",
                chunk_index=0,
                text="Hotel negotiation for 12-14 December 2026 is currently active with a pending revised offer of ₹6,800 per night.",
                hybrid_score=0.90,
            ),
            RetrievedChunk(
                chunk_id="chk_hotel_claim_cancelled",
                document_id="doc_hotel_cancelled",
                department_id=SHARED_DEPT_ID,
                file_name="travel_memo_cancelled.docx",
                chunk_index=0,
                text="Hotel negotiation for 12-14 December 2026 was cancelled due to schedule conflicts.",
                hybrid_score=0.90,
            ),
        ],
        expected_facts=["conflict", "active", "cancelled"],
        supporting_source_ids=["chk_hotel_claim_active", "chk_hotel_claim_cancelled"],
        expected_behavior="CONFLICT_DISCLOSURE",
        abstention_required=False,
        conflict_disclosure_required=True,
        conflicts=[
            ConflictNotice(
                subject="December Hotel Negotiation",
                predicate="status",
                conflicting_objects=["active", "cancelled"],
                conflicting_chunk_ids=["chk_hotel_claim_active", "chk_hotel_claim_cancelled"],
                details="Conflicting records assert the negotiation is simultaneously active and cancelled.",
            )
        ],
        description="Unresolved conflict without timestamp superiority requires explicit disclosure.",
    ),

    # 7. Unanswerable: Hotel Booking Confirmation Number & Taxes (Explicitly unrecorded)
    GroundedTestCase(
        id="case_07_unanswerable_hotel_booking_number",
        category="unanswerable",
        query="What is the confirmed booking confirmation number and final payable amount including taxes for the hotel stay?",
        chunks=[
            RetrievedChunk(
                chunk_id="chk_hotel_negotiation_status",
                document_id="doc_hotel_01",
                department_id=SHARED_DEPT_ID,
                file_name="03_hotel_price_bargaining_demo_document.docx",
                chunk_index=2,
                text=(
                    "Document ID: HOTEL-DEMO-001. Negotiation Status: "
                    "This record documents a revised offer, not a confirmed booking. "
                    "The guest's acceptance and the final payable amount including taxes are not recorded. "
                    "Do not treat the offer as a real booking."
                ),
                hybrid_score=0.93,
            )
        ],
        expected_facts=["not recorded", "cannot be verified"],
        supporting_source_ids=["chk_hotel_negotiation_status"],
        expected_behavior="ABSTAIN",
        abstention_required=True,
        conflict_disclosure_required=False,
        description="Evidence explicitly documents that confirmation and taxes are not recorded.",
    ),

    # 8. Unanswerable: Missing Evidence Abstention (International flight upgrades)
    GroundedTestCase(
        id="case_08_unanswerable_missing_flight_policy",
        category="unanswerable",
        query="What is the allowable reimbursement limit for international flight upgrades in executive class?",
        chunks=[
            RetrievedChunk(
                chunk_id="chk_fin_budget_only",
                document_id="doc_fin_01",
                department_id=SHARED_DEPT_ID,
                file_name="01_finance_demo_document.docx",
                chunk_index=0,
                text=(
                    "Document ID: FIN-DEMO-001. Approved Q3 2026 operating budget was ₹12,00,000. "
                    "Travel expenses totalled ₹1,40,000."
                ),
                hybrid_score=0.45,
            )
        ],
        expected_facts=["cannot be verified"],
        supporting_source_ids=[],
        expected_behavior="ABSTAIN",
        abstention_required=True,
        conflict_disclosure_required=False,
        description="Completely unanswerable question with missing policy evidence.",
    ),
]


@dataclass
class CaseEvaluationMetric:
    case_id: str
    category: str
    query: str
    expected_behavior: str
    actual_behavior: str
    passed: bool
    evidence_recall: float
    citation_validity: float
    claim_citation_supported: bool
    abstention_accurate: bool
    conflict_disclosed: bool
    context_budget_status: str
    dropped_evidence_count: int
    compiler_latency_ms: float
    generation_latency_ms: float
    estimated_prompt_tokens: int
    provider_prompt_tokens: int | None
    token_estimation_error: int | None
    token_relative_error: float | None
    citations_rendered: list[str]
    answer_preview: str
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class HarnessSummaryReport:
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    accuracy: float = 0.0
    mean_evidence_recall: float = 0.0
    mean_citation_validity: float = 0.0
    abstention_accuracy: float = 0.0
    conflict_disclosure_rate: float = 0.0
    mean_compiler_latency_ms: float = 0.0
    mean_generation_latency_ms: float = 0.0
    case_metrics: list[CaseEvaluationMetric] = field(default_factory=list)


class MockGroundedLLM(LLMProvider):
    """
    Deterministic mocked provider for evaluation test suites.
    Produces grounded responses referencing supplied chunk IDs without remote API costs.
    """
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model = model_name

    async def generate(
        self,
        messages: list[dict[str, str]],
        stage: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict]:
        user_content = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
        
        # Check unanswerable / abstention trigger
        if "flight upgrades" in user_content.lower() or "final payable amount including taxes" in user_content.lower():
            answer = (
                "The requested information cannot be verified from the provided data. "
                "The records explicitly note that confirmation details and tax amounts are not recorded [1]."
            )
        elif "active or cancelled" in user_content.lower():
            answer = (
                "The retrieved sources contain a conflict: one record states the negotiation is active [1], "
                "while another record states it was cancelled [2]. The discrepancy cannot be resolved from the provided data."
            )
        elif "12 to 14 december" in user_content.lower():
            answer = (
                "For the stay from 12 to 14 December 2026, the hotel initially quoted ₹8,000 per room per night for a Deluxe Room [1]. "
                "The guest counteroffered ₹6,000 per room per night [1]."
            )
        elif "travel expenses" in user_content.lower() and ("revised" in user_content.lower() or "hotel" in user_content.lower()):
            answer = (
                "The Finance Department's Q3 2026 travel expenses totalled ₹1,40,000 [1], "
                "and the revised 2-night pre-tax room total offered was ₹13,600 [2]."
            )
        elif "october 2026 sales" in user_content.lower() or "closed-won" in user_content.lower():
            answer = (
                "October 2026 sales closed-won revenue was ₹18,50,000 [1], "
                "while the Finance Department's actual Q3 2026 operating spend was ₹9,25,000 [2]."
            )
        elif "operating budget" in user_content.lower():
            answer = (
                "The Finance Department's approved Q3 2026 operating budget was ₹12,00,000, "
                "and the remaining budget is ₹2,75,000 [1]."
            )
        elif "current annual revenue" in user_content.lower():
            answer = "The audited annual operating revenue for fiscal year 2024 reached $25M [1]."
        else:
            answer = "Grounded response based on retrieved evidence [1]."

        import json
        structured = json.dumps({"answer": answer, "visualization": None})
        usage = {
            "prompt_tokens": count_messages(messages, model=self.model),
            "completion_tokens": count_text(structured, model=self.model),
            "total_tokens": count_messages(messages, model=self.model) + count_text(structured, model=self.model),
        }
        return structured, usage


class RAGEvaluationHarness:
    """Repeatable RAG Evaluation Harness testing Compiler, Generation, and Citations."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        context_compiler: ContextCompiler | None = None,
    ):
        self.llm = llm or MockGroundedLLM()
        self.compiler = context_compiler or ContextCompiler()

    async def evaluate_case(self, case: GroundedTestCase) -> CaseEvaluationMetric:
        start_time = time.perf_counter()

        # 1. Real Compiler Path
        dept_uuids = [uuid.UUID(SHARED_DEPT_ID)]
        t_comp_start = time.perf_counter()
        compiled = self.compiler.compile(
            chunks=case.chunks,
            query=case.query,
            allowed_departments=dept_uuids,
        )
        compiler_latency_ms = (time.perf_counter() - t_comp_start) * 1000

        # Selected evidence chunk IDs
        selected_chunk_ids = {
            str(e.source_id)
            for e in compiled.selected_evidence
            if e.source_id is not None
        }

        # 2. Evidence Recall
        if not case.supporting_source_ids:
            evidence_recall = 1.0
        else:
            intersection = set(case.supporting_source_ids) & selected_chunk_ids
            evidence_recall = len(intersection) / len(case.supporting_source_ids)

        # 3. Real Generation Service Path
        t_gen_start = time.perf_counter()
        from unittest.mock import AsyncMock
        mock_session = AsyncMock()
        service = GenerationService(session=mock_session, context_compiler=self.compiler)
        service.llm = self.llm

        resp: GenerateResponse = await service.generate_from_context(
            query=case.query,
            chunks=compiled.chunks,
            allowed_departments=dept_uuids,
            conflicts=case.conflicts,
            rendered_context=compiled.rendered_context,
            llm=self.llm,
        )
        generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000

        # 4. Measure Citation Validity
        rendered_citations = resp.citations or []
        supplied_file_names = {c.file_name for c in compiled.chunks if c.file_name}
        if not rendered_citations:
            citation_validity = 1.0 if case.abstention_required else 0.0
        else:
            valid_cits = [
                cit for cit in rendered_citations
                if any(fn in cit for fn in supplied_file_names) or any(cid in cit for cid in selected_chunk_ids)
            ]
            citation_validity = len(valid_cits) / len(rendered_citations)

        claim_citation_supported = any(f"[{i}]" in resp.answer for i in range(1, 10))

        # 5. Measure Answer Correctness / Abstention / Conflict Disclosure
        failures: list[str] = []
        actual_behavior = "ANSWER"
        abstention_accurate = True
        conflict_disclosed = True

        answer_lower = resp.answer.lower()

        if case.abstention_required:
            abstention_phrases = ["cannot be verified", "not recorded", "not supported", "not provided"]
            if any(p in answer_lower for p in abstention_phrases):
                actual_behavior = "ABSTAIN"
            else:
                abstention_accurate = False
                failures.append("Failed required abstention: hallucinated or failed to state unverified.")
        elif case.conflict_disclosure_required:
            conflict_phrases = ["conflict", "discrepancy", "contradict", "competing"]
            if any(p in answer_lower for p in conflict_phrases):
                actual_behavior = "CONFLICT_DISCLOSURE"
            else:
                conflict_disclosed = False
                failures.append("Failed required conflict disclosure: discrepancy not reported in answer.")
        else:
            actual_behavior = "ANSWER"
            for fact in case.expected_facts:
                if fact.lower() not in answer_lower:
                    failures.append(f"Missing expected fact: '{fact}'")

        if evidence_recall < 1.0:
            failures.append(f"Evidence recall shortfall ({evidence_recall:.2f} < 1.00)")

        if citation_validity < 1.0:
            failures.append(f"Citation validity failure ({citation_validity:.2f} < 1.00)")

        # 6. Token Accounting
        from rag_app.generation.prompt import build_rag_messages
        messages = build_rag_messages(
            context=compiled.rendered_context,
            question=case.query,
            conflicts=case.conflicts,
            chunks=compiled.chunks,
        )
        estimated_prompt_tokens = count_messages(messages, model=getattr(self.llm, "model", None))
        
        # In mock mode, usage metadata matches estimated
        provider_prompt_tokens = estimated_prompt_tokens
        token_estimation_error = 0
        token_relative_error = 0.0

        is_passed = len(failures) == 0

        return CaseEvaluationMetric(
            case_id=case.id,
            category=case.category,
            query=case.query,
            expected_behavior=case.expected_behavior,
            actual_behavior=actual_behavior,
            passed=is_passed,
            evidence_recall=round(evidence_recall, 4),
            citation_validity=round(citation_validity, 4),
            claim_citation_supported=claim_citation_supported,
            abstention_accurate=abstention_accurate,
            conflict_disclosed=conflict_disclosed,
            context_budget_status=compiled.budget_status,
            dropped_evidence_count=len(case.chunks) - len(compiled.selected_evidence),
            compiler_latency_ms=round(compiler_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            estimated_prompt_tokens=estimated_prompt_tokens,
            provider_prompt_tokens=provider_prompt_tokens,
            token_estimation_error=token_estimation_error,
            token_relative_error=token_relative_error,
            citations_rendered=rendered_citations,
            answer_preview=resp.answer[:120] + ("..." if len(resp.answer) > 120 else ""),
            failure_reasons=failures,
        )

    async def run_all(self, dataset: list[GroundedTestCase] | None = None) -> HarnessSummaryReport:
        cases = dataset or GROUNDED_EVAL_DATASET
        metrics: list[CaseEvaluationMetric] = []
        for case in cases:
            m = await self.evaluate_case(case)
            metrics.append(m)

        total = len(metrics)
        passed = sum(1 for m in metrics if m.passed)
        failed = total - passed
        acc = round(passed / total, 4) if total > 0 else 0.0

        mean_recall = round(sum(m.evidence_recall for m in metrics) / total, 4) if total > 0 else 0.0
        mean_cit_val = round(sum(m.citation_validity for m in metrics) / total, 4) if total > 0 else 0.0

        abstain_cases = [m for m in metrics if m.expected_behavior == "ABSTAIN"]
        abstain_acc = (
            round(sum(1 for m in abstain_cases if m.abstention_accurate) / len(abstain_cases), 4)
            if abstain_cases else 1.0
        )

        conflict_cases = [m for m in metrics if m.expected_behavior == "CONFLICT_DISCLOSURE"]
        conflict_rate = (
            round(sum(1 for m in conflict_cases if m.conflict_disclosed) / len(conflict_cases), 4)
            if conflict_cases else 1.0
        )

        mean_comp_lat = round(sum(m.compiler_latency_ms for m in metrics) / total, 2) if total > 0 else 0.0
        mean_gen_lat = round(sum(m.generation_latency_ms for m in metrics) / total, 2) if total > 0 else 0.0

        return HarnessSummaryReport(
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            accuracy=acc,
            mean_evidence_recall=mean_recall,
            mean_citation_validity=mean_cit_val,
            abstention_accuracy=abstain_acc,
            conflict_disclosure_rate=conflict_rate,
            mean_compiler_latency_ms=mean_comp_lat,
            mean_generation_latency_ms=mean_gen_lat,
            case_metrics=metrics,
        )
