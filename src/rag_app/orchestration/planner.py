"""
Adaptive Task Planner for Complex Workflows.
Generates typed task graphs, validates DAG constraints, evaluates evidence sufficiency,
and performs bounded replanning (maximum 3 iterations) without granting expanded scope.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import uuid

from rag_app.orchestration.capability_registry import get_capability_registry, CapabilityRegistry
from rag_app.orchestration.dag_validator import DAGValidationError, validate_task_dag
from rag_app.orchestration.evidence import EvidenceAggregator
from rag_app.schemas.research import SufficiencyEvaluation, SufficiencyStatus
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    ExecutionTask,
    TaskType,
)

logger = logging.getLogger(__name__)


class AdaptiveTaskPlanner:
    """
    Adaptive Task Planner.
    Constructs and replans DAGs for complex multi-step research and analysis.
    """

    MAX_REPLAN_ITERATIONS = 3
    MAX_FAN_OUT = 5
    MAX_PLAN_TASKS = 15

    def __init__(self, llm_provider: Optional[Any] = None, capability_registry: Optional[Any] = None):
        self.llm_provider = llm_provider
        self.capability_registry = capability_registry or get_capability_registry()

    def plan_initial_request(
        self,
        goal: str,
        workflow_run_id: Optional[uuid.UUID] = None,
        estimated_complexity: str = "moderate",
    ) -> AdaptiveTaskPlan:
        """
        Generates an initial typed execution plan for a complex query or goal.
        Decomposes request into independent retrieval tasks (RAG, Web, Memory),
        followed by sequential fact extraction, grounding verification, and synthesis.
        """
        clean_goal = goal.strip()
        g_lower = clean_goal.lower()

        # Detect intent dimensions
        has_mcp_intent = any(kw in g_lower for kw in ("crm", "twenty", "company", "contact", "mcp"))
        has_memory_intent = any(kw in g_lower for kw in ("discussed", "earlier", "previous", "decision", "preference", "style", "my stack"))
        has_research_intent = any(kw in g_lower for kw in ("research", "industry", "pattern", "compare", "external", "best practice")) or not has_mcp_intent

        tasks: list[ExecutionTask] = []
        extract_dependencies: list[str] = []

        # 1. Internal RAG Task (Independent parallel read)
        tasks.append(
            ExecutionTask(
                task_key="gather_internal_knowledge",
                task_type=TaskType.RETRIEVAL,
                capability_id="rag.retrieve_internal",
                description=f"Retrieve authoritative internal documents and guidelines for: {clean_goal}",
                dependencies=[],
                input_payload={"query": clean_goal, "scope": "internal"},
                estimated_tokens=800,
                estimated_provider_calls=1,
                is_read_only=True,
                side_effect_level="READ_ONLY",
                required_permissions=["DOCUMENTS_READ"],
            )
        )
        extract_dependencies.append("gather_internal_knowledge")

        # 2. Personal Memory Recall Task if relevant (Independent parallel read)
        if has_memory_intent:
            tasks.append(
                ExecutionTask(
                    task_key="gather_memory_context",
                    task_type=TaskType.RETRIEVAL,
                    capability_id="memory.retrieve_context",
                    description=f"Retrieve past decisions, personal preferences, and living profile for: {clean_goal}",
                    dependencies=[],
                    input_payload={"query": clean_goal, "scope": "memory"},
                    estimated_tokens=600,
                    estimated_provider_calls=1,
                    is_read_only=True,
                    side_effect_level="READ_ONLY",
                    required_permissions=["MEMORY_READ"],
                )
            )
            extract_dependencies.append("gather_memory_context")

        # 3. External Web Research Task (Independent parallel read)
        if has_research_intent:
            tasks.append(
                ExecutionTask(
                    task_key="gather_web_evidence",
                    task_type=TaskType.WEB_SEARCH,
                    capability_id="research.web_search",
                    description=f"Gather external research and industry context for: {clean_goal}",
                    dependencies=[],
                    input_payload={"query": clean_goal, "scope": "external"},
                    estimated_tokens=1000,
                    estimated_provider_calls=2,
                    is_read_only=True,
                    side_effect_level="EXTERNAL_READ",
                    required_permissions=["RESEARCH_READ"],
                )
            )
            extract_dependencies.append("gather_web_evidence")

        # 4. Workspace MCP / CRM Knowledge if relevant (Independent parallel read)
        if has_mcp_intent:
            tasks.append(
                ExecutionTask(
                    task_key="gather_mcp_knowledge",
                    task_type=TaskType.RETRIEVAL,
                    capability_id="crm.query_records",
                    description=f"Query authorized read-only workspace MCP tools for: {clean_goal}",
                    dependencies=[],
                    input_payload={"query": clean_goal, "scope": "workspace_mcp"},
                    estimated_tokens=800,
                    estimated_provider_calls=1,
                    is_read_only=True,
                    side_effect_level="EXTERNAL_READ",
                    required_permissions=["CRM_READ"],
                )
            )
            extract_dependencies.append("gather_mcp_knowledge")

        # 5. Fact Extraction & Normalization (Sequential dependent fan-in)
        tasks.append(
            ExecutionTask(
                task_key="extract_key_facts",
                task_type=TaskType.EXTRACTION,
                capability_id="research.extract_facts",
                description="Extract and normalize structured facts from internal and external evidence",
                dependencies=extract_dependencies,
                input_payload={"target_dimensions": ["metrics", "dates", "conclusions"]},
                estimated_tokens=600,
                estimated_provider_calls=1,
                is_read_only=True,
                side_effect_level="READ_ONLY",
                required_permissions=[],
            )
        )

        # 6. Evidence Grounding & Contradiction Check (Sequential dependent)
        tasks.append(
            ExecutionTask(
                task_key="verify_grounding",
                task_type=TaskType.GROUNDING,
                capability_id="validation.verify_grounding",
                description="Cross-verify extracted facts against citations and identify contradictions",
                dependencies=["extract_key_facts"],
                input_payload={},
                estimated_tokens=500,
                estimated_provider_calls=1,
                is_read_only=True,
                side_effect_level="READ_ONLY",
                required_permissions=[],
            )
        )

        # 7. Final Synthesis & Report (Sequential terminal task)
        tasks.append(
            ExecutionTask(
                task_key="synthesize_findings",
                task_type=TaskType.SYNTHESIS,
                capability_id="synthesis.generate_findings",
                description="Synthesize grounded research report with citations",
                dependencies=["verify_grounding"],
                input_payload={},
                estimated_tokens=1500,
                estimated_provider_calls=1,
                is_read_only=True,
                side_effect_level="READ_ONLY",
                required_permissions=[],
            )
        )

        total_tokens = sum(t.estimated_tokens for t in tasks)
        total_calls = sum(t.estimated_provider_calls for t in tasks)

        plan = AdaptiveTaskPlan(
            workflow_run_id=workflow_run_id,
            goal=clean_goal,
            tasks=tasks,
            iteration=1,
            total_estimated_tokens=total_tokens,
            total_estimated_provider_calls=total_calls,
        )

        # Strictly validate plan before returning
        validate_task_dag(
            plan,
            max_concurrency_limit=self.MAX_FAN_OUT,
            max_tasks_limit=self.MAX_PLAN_TASKS,
            max_replan_iterations=self.MAX_REPLAN_ITERATIONS,
            capability_registry=self.capability_registry,
            raise_on_error=True,
        )
        return plan

    def replan_for_missing_evidence(
        self,
        current_plan: AdaptiveTaskPlan,
        sufficiency: SufficiencyEvaluation,
        remaining_token_budget: int,
        remaining_provider_calls: int,
    ) -> Optional[AdaptiveTaskPlan]:
        """
        Creates a bounded replan when evidence sufficiency is PARTIAL or INSUFFICIENT.
        Enforces maximum 3 iterations, remaining budget bounds, and preserved scope.
        """
        # Hard check: Replan iteration limit
        if current_plan.iteration >= self.MAX_REPLAN_ITERATIONS:
            logger.info(
                f"Replan limit reached ({current_plan.iteration}/{self.MAX_REPLAN_ITERATIONS}). Safe stop triggered."
            )
            return None

        # If evidence is already sufficient, no replan needed
        if sufficiency.status == SufficiencyStatus.SUFFICIENT:
            return None

        missing = sufficiency.missing_requirements
        contradictions = sufficiency.contradictions

        if not missing and not contradictions:
            return None

        next_iter = current_plan.iteration + 1
        new_tasks: list[ExecutionTask] = []

        # Target specific missing dimensions
        for i, req in enumerate(missing[:3]):  # Bound fan-out
            task_key = f"targeted_investigation_iter{next_iter}_{i+1}"
            new_tasks.append(
                ExecutionTask(
                    task_key=task_key,
                    task_type=TaskType.RETRIEVAL,
                    capability_id="rag.retrieve_internal",
                    description=f"Targeted search for missing requirement: {req}",
                    dependencies=[],
                    input_payload={"focus_query": req},
                    estimated_tokens=600,
                    estimated_provider_calls=1,
                    is_read_only=True,
                    side_effect_level="READ_ONLY",
                    required_permissions=["DOCUMENTS_READ"],
                )
            )

        # Target contradiction resolution if present
        if contradictions:
            conflict_key = f"resolve_contradiction_iter{next_iter}"
            new_tasks.append(
                ExecutionTask(
                    task_key=conflict_key,
                    task_type=TaskType.GROUNDING,
                    capability_id="validation.verify_grounding",
                    description=f"Resolve conflicting evidence: {contradictions[0][:200]}",
                    dependencies=[t.task_key for t in new_tasks],
                    input_payload={"contradictions": contradictions},
                    estimated_tokens=500,
                    estimated_provider_calls=1,
                    is_read_only=True,
                    side_effect_level="READ_ONLY",
                    required_permissions=[],
                )
            )

        # Final synthesis following replan
        dep_keys = [t.task_key for t in new_tasks]
        new_tasks.append(
            ExecutionTask(
                task_key=f"synthesize_findings_iter{next_iter}",
                task_type=TaskType.SYNTHESIS,
                capability_id="synthesis.generate_findings",
                description="Synthesize updated findings incorporating targeted evidence",
                dependencies=dep_keys,
                input_payload={},
                estimated_tokens=1200,
                estimated_provider_calls=1,
                is_read_only=True,
                side_effect_level="READ_ONLY",
                required_permissions=[],
            )
        )

        total_est_tokens = sum(t.estimated_tokens for t in new_tasks)
        total_est_calls = sum(t.estimated_provider_calls for t in new_tasks)

        # Check remaining budget bounds
        if total_est_tokens > remaining_token_budget or total_est_calls > remaining_provider_calls:
            logger.warning(
                f"Replan requires {total_est_tokens} tokens / {total_est_calls} calls, "
                f"exceeding remaining budget ({remaining_token_budget} tokens / {remaining_provider_calls} calls). Safe stop."
            )
            return None

        replan = AdaptiveTaskPlan(
            workflow_run_id=current_plan.workflow_run_id,
            goal=current_plan.goal,
            tasks=new_tasks,
            iteration=next_iter,
            total_estimated_tokens=total_est_tokens,
            total_estimated_provider_calls=total_est_calls,
            metadata={"replan_reason": sufficiency.reason},
        )

        validate_task_dag(
            replan,
            max_concurrency_limit=self.MAX_FAN_OUT,
            max_tasks_limit=self.MAX_PLAN_TASKS,
            max_replan_iterations=self.MAX_REPLAN_ITERATIONS,
            capability_registry=self.capability_registry,
            raise_on_error=True,
        )
        return replan
