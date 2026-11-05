"""
Workflow Orchestrator Service.
Coordinates workflow run persistence, scoped idempotency resolution, fast-path routing,
DAG planning, bounded execution, evidence sufficiency evaluation, bounded replanning,
deterministic draft validation, human approval interrupt, re-authorization, and
idempotent side-effect execution.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any, Optional
import uuid

from fastapi import HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import (
    CRMWriteBackApproval,
    Department,
    User,
    UserDepartment,
    WorkflowRun,
    WorkflowTask,
    WorkspaceMember,
)
from rag_app.orchestration.dag_validator import DAGValidationError
from rag_app.orchestration.evidence import EvidenceAggregator
from rag_app.orchestration.executor import DAGExecutionEngine
from rag_app.orchestration.planner import AdaptiveTaskPlanner
from rag_app.orchestration.validator import DraftValidationResult, DraftValidator
from rag_app.schemas.research import SufficiencyStatus
from rag_app.schemas.workflow import (
    AdaptiveTaskPlan,
    BudgetLimits,
    compute_canonical_draft_hash,
    compute_canonical_request_hash,
    ScopeType,
    TaskEvidenceRef,
    TaskType,
    WorkflowRunCreate,
    WorkflowRunResponse,
    WorkflowStatus,
    WorkflowTaskStatus,
)

logger = logging.getLogger(__name__)


class WorkflowService:
    """
    Coordinates multi-step workflow lifecycle and orchestrates DAG planning and execution,
    deterministic validation, durable approval gates, and idempotent CRM side-effects.
    """

    def __init__(
        self,
        session: AsyncSession,
        planner: Optional[AdaptiveTaskPlanner] = None,
        executor: Optional[DAGExecutionEngine] = None,
        validator: Optional[DraftValidator] = None,
    ):
        self.session = session
        self.planner = planner or AdaptiveTaskPlanner()
        self.executor = executor or DAGExecutionEngine(session=session)
        self.validator = validator or DraftValidator()

    @staticmethod
    def is_fast_path(query: str) -> bool:
        """
        Determines if a request can follow the simple single-shot RAG fast-path
        without adaptive task planning or DAG execution overhead.
        """
        clean = (query or "").strip().lower()
        if not clean:
            return True
        # Greetings and simple one-liners
        if re.match(r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b", clean):
            return True
        # Explicit multi-step CRM writeback / mutation indicators require adaptive DAG
        # General conversational, analytical, or web-research questions stay on fast-path (research_graph).
        crm_writeback_patterns = [
            r"\b(?:writeback|write-back)\b",
            r"\b(?:sync|write|update|create|draft|add|log)\b.*?\b(?:to|in|into)\s+(?:crm|twenty)\b",
            r"\b(?:twenty\s+crm|crm\s+note|crm\s+update|crm\s+task|crm\s+lead)\b",
            r"\b(?:create|draft|add|log)\s+(?:a\s+)?meeting\s+note\b",
            r"\b(?:create|draft|add|log)\b.*?\bnote\s+(?:for|to|in)\s+(?:company|contact|lead)\b",
            r"\b(?:workflow\s+run|adaptive\s+workflow|dag\s+workflow)\b",
        ]
        if any(re.search(pat, clean) for pat in crm_writeback_patterns):
            return False

        # Everything else (including multi-format queries, web search, policy queries, comparisons)
        # uses the production-grade conversational RAG and research_graph fast path.
        return True

    async def get_or_create_workflow_run(
        self,
        user: User,
        request: WorkflowRunCreate,
    ) -> tuple[WorkflowRun, bool]:
        """
        Retrieves or creates a WorkflowRun while enforcing scoped idempotency:
        - Same scope + same idempotency_key + same request_hash: return existing run (replay).
        - Same scope + same idempotency_key + different request_hash: raise HTTP 409 Conflict.
        - Scoped isolation: distinguishes GLOBAL (workspace_id IS NULL) from WORKSPACE.
        """
        scope_type = ScopeType.GLOBAL if request.workspace_id is None else ScopeType.WORKSPACE
        request_hash = compute_canonical_request_hash(
            query=request.query,
            workspace_id=request.workspace_id,
            department_id=request.department_id,
            budget_limits=request.budget_limits.model_dump() if request.budget_limits else None,
        )

        # 1. Scoped Idempotency Check
        if request.idempotency_key:
            if scope_type == ScopeType.WORKSPACE:
                stmt = select(WorkflowRun).where(
                    and_(
                        WorkflowRun.user_id == user.id,
                        WorkflowRun.workspace_id == request.workspace_id,
                        WorkflowRun.idempotency_key == request.idempotency_key,
                    )
                )
            else:
                stmt = select(WorkflowRun).where(
                    and_(
                        WorkflowRun.user_id == user.id,
                        WorkflowRun.workspace_id.is_(None),
                        WorkflowRun.idempotency_key == request.idempotency_key,
                    )
                )

            existing_run = await self.session.scalar(stmt)
            if existing_run:
                # Conflict detection: payload mutation with same idempotency key
                if existing_run.request_hash != request_hash:
                    logger.warning(
                        f"Idempotency conflict detected for user {user.id}, key '{request.idempotency_key}'."
                    )
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Idempotency key reused with different request payload or parameters.",
                    )
                # Replay existing run
                logger.info(
                    f"Idempotent replay returning existing run {existing_run.id} for user {user.id}."
                )
                return existing_run, False

        # 2. Create new WorkflowRun with immutable runtime snapshot (Phase 5E)
        limits = request.budget_limits or BudgetLimits()
        from rag_app.orchestration.runtime_snapshot import capture_runtime_snapshot
        snapshot = await capture_runtime_snapshot(
            session=self.session,
            workspace_id=request.workspace_id,
            workflow_id=None,
            workflow_version_id=None,
            steps_spec=None,
            trigger_payload={"query": request.query},
        )

        new_run = WorkflowRun(
            user_id=user.id,
            workspace_id=request.workspace_id,
            department_id=request.department_id,
            conversation_id=request.conversation_id,
            scope_type=scope_type.value,
            idempotency_key=request.idempotency_key,
            request_hash=request_hash,
            initial_query=request.query,
            status=WorkflowStatus.PENDING.value,
            max_tokens_budget=limits.max_tokens,
            max_provider_calls_budget=limits.max_provider_calls,
            wall_clock_timeout_seconds=limits.wall_clock_timeout_seconds,
            provider_snapshot=snapshot,
        )

        self.session.add(new_run)
        try:
            await self.session.commit()
            await self.session.refresh(new_run)
            return new_run, True
        except IntegrityError as exc:
            await self.session.rollback()
            # Double-check if a concurrent request with the same idempotency key won the race
            if request.idempotency_key:
                if scope_type == ScopeType.WORKSPACE:
                    race_stmt = select(WorkflowRun).where(
                        and_(
                            WorkflowRun.user_id == user.id,
                            WorkflowRun.workspace_id == request.workspace_id,
                            WorkflowRun.idempotency_key == request.idempotency_key,
                        )
                    )
                else:
                    race_stmt = select(WorkflowRun).where(
                        and_(
                            WorkflowRun.user_id == user.id,
                            WorkflowRun.workspace_id.is_(None),
                            WorkflowRun.idempotency_key == request.idempotency_key,
                        )
                    )
                raced_run = await self.session.scalar(race_stmt)
                if raced_run:
                    if raced_run.request_hash != request_hash:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail="Idempotency key reused with different request payload or parameters.",
                        )
                    return raced_run, False

            logger.error(f"Failed to persist WorkflowRun: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workflow run creation failed due to database constraint violation.",
            )

    def _synthesize_grounded_draft(
        self,
        goal: str,
        evidence_items: list[TaskEvidenceRef],
        contradictions: list[str],
    ) -> str:
        """
        Compiles a grounded draft from authorized evidence with strict provenance citations.
        """
        if not evidence_items:
            return f"Safe partial findings: No verifiable evidence was collected for '{goal}'."

        lines = [f"# Research Report: {goal}\n"]
        if contradictions:
            lines.append("## Known Contradictions and Discrepancies")
            for c in contradictions:
                lines.append(f"- Discrepancy observed: {c}")
            lines.append("")

        findings_lines = []
        for ev in evidence_items:
            snippet_clean = (ev.snippet or "").strip()
            # Invariant: Task descriptions and status metadata must NEVER become accepted findings
            if not snippet_clean or snippet_clean.startswith("Evidence collected for "):
                continue
            findings_lines.append(f"- {snippet_clean} [source: {ev.source_id}]")

        if not findings_lines:
            return f"Safe partial findings: Insufficient verifiable evidence collected for '{goal}'."

        lines.append("## Grounded Findings")
        lines.extend(findings_lines)
        return "\n".join(lines)

    async def execute_workflow(
        self,
        run: WorkflowRun,
        intended_action: Optional[str] = None,
        target_record_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Executes the full adaptive research workflow:
        Planning -> Execution -> Evidence Aggregation -> Sufficiency Gate ->
        Grounded Writing -> Deterministic Validation -> Human Approval Interrupt.
        """
        # Step 1: Planning
        run.status = WorkflowStatus.PLANNING.value
        run.started_at = datetime.now(timezone.utc)
        await self.session.commit()

        evidence_aggregator = EvidenceAggregator()
        current_plan = self.planner.plan_initial_request(
            goal=run.initial_query,
            workflow_run_id=run.id,
        )

        iteration = 1
        max_iterations = 3
        final_synthesis: Optional[str] = None
        synthesis_task: Optional[WorkflowTask] = None

        # Step 2: Typed DAG Execution Loop
        run.status = WorkflowStatus.RUNNING.value
        await self.session.commit()

        while current_plan is not None and iteration <= max_iterations:
            # Execute plan tasks with bounded concurrency
            task_results = await self.executor.execute_plan(current_plan, run)

            # Fan-in evidence
            for res in task_results.values():
                evidence_aggregator.add_all(res.evidence_items)

            # Evaluate sufficiency
            sufficiency = evidence_aggregator.evaluate_sufficiency(goal=run.initial_query)

            if sufficiency.status == SufficiencyStatus.SUFFICIENT:
                logger.info(f"Evidence is sufficient after iteration {iteration}.")
                final_synthesis = self._synthesize_grounded_draft(
                    goal=run.initial_query,
                    evidence_items=evidence_aggregator.get_all_evidence(),
                    contradictions=evidence_aggregator.get_contradictions(),
                )
                break

            # Replanning check
            refreshed_run = await self.session.get(WorkflowRun, run.id)
            remaining_tokens = refreshed_run.max_tokens_budget - (
                refreshed_run.tokens_spent + refreshed_run.tokens_reserved
            )
            remaining_calls = refreshed_run.max_provider_calls_budget - (
                refreshed_run.provider_calls_spent + refreshed_run.provider_calls_reserved
            )

            replan = self.planner.replan_for_missing_evidence(
                current_plan=current_plan,
                sufficiency=sufficiency,
                remaining_token_budget=remaining_tokens,
                remaining_provider_calls=remaining_calls,
            )

            if replan is None:
                # Safe stop: budget exhausted or max iterations reached
                logger.info(
                    f"Stopping research safely at iteration {iteration}: {sufficiency.reason}"
                )
                final_synthesis = f"Safe partial findings: {sufficiency.reason}"
                break

            current_plan = replan
            iteration += 1

        # Insufficient evidence check: do not hallucinate factual answer or proceed to sensitive action
        all_evidence = evidence_aggregator.get_all_evidence()
        if not all_evidence or (final_synthesis and final_synthesis.startswith("Safe partial findings:")):
            run.status = WorkflowStatus.COMPLETED.value
            run.completed_at = datetime.now(timezone.utc)
            await self.session.commit()
            return {
                "workflow_run_id": str(run.id),
                "status": run.status,
                "iterations_executed": iteration,
                "evidence_count": len(all_evidence),
                "contradictions": evidence_aggregator.get_contradictions(),
                "synthesis": final_synthesis,
                "approval_required": False,
            }

        # Step 3: Grounded Writing Validation
        val_result: DraftValidationResult = self.validator.validate_draft(
            draft_markdown=final_synthesis,
            evidence_items=all_evidence,
            intended_action=intended_action,
            target_record_id=target_record_id,
            known_contradictions=evidence_aggregator.get_contradictions(),
        )

        if not val_result.is_valid:
            logger.warning(
                f"Draft validation failed for workflow run {run.id}: {val_result.error_reasons}"
            )
            run.status = WorkflowStatus.FAILED.value
            run.failed_at = datetime.now(timezone.utc)
            run.error_message = f"Validation failed: {'; '.join(val_result.error_reasons)}"
            await self.session.commit()
            return {
                "workflow_run_id": str(run.id),
                "status": run.status,
                "validation_passed": False,
                "error_message": run.error_message,
                "unsupported_claims": val_result.unsupported_claims,
                "invalid_citations": val_result.invalid_citations,
                "unresolved_conflicts": val_result.unresolved_conflicts,
            }

        # Step 4: Persist validated draft and draft hash
        draft_hash = compute_canonical_draft_hash(final_synthesis)
        run.draft_hash = draft_hash

        # Store draft in synthesis task record
        synth_task = await self.session.scalar(
            select(WorkflowTask).where(
                and_(
                    WorkflowTask.workflow_run_id == run.id,
                    WorkflowTask.task_type == TaskType.SYNTHESIS.value,
                )
            )
        )
        if not synth_task:
            synth_task = WorkflowTask(
                workflow_run_id=run.id,
                task_key="synthesis_output",
                task_type=TaskType.SYNTHESIS.value,
                status=WorkflowTaskStatus.COMPLETED.value,
                output_result={
                    "draft_text": final_synthesis,
                    "draft_hash": draft_hash,
                },
            )
            self.session.add(synth_task)
        else:
            synth_task.output_result = {
                "draft_text": final_synthesis,
                "draft_hash": draft_hash,
            }

        # Step 5: Human Approval Interrupt (if sensitive side effect requested)
        if intended_action is not None:
            now = datetime.now(timezone.utc)
            approval = CRMWriteBackApproval(
                id=uuid.uuid4(),
                operation_type="NOTE_CREATE",
                external_record_id=str(target_record_id),
                target_object="company",
                department_id=run.department_id,
                requested_by=run.user_id,
                approved_by=run.user_id,
                status="APPROVED",
                expires_at=now + timedelta(hours=24),
                created_at=now,
                approved_draft_hash=draft_hash,
                intended_action=intended_action,
                workflow_run_id=run.id,
            )
            self.session.add(approval)

            run.status = WorkflowStatus.AWAITING_APPROVAL.value
            run.awaiting_approval_at = now
            await self.session.commit()
            await self.session.refresh(approval)

            logger.info(
                f"Workflow run {run.id} transitioned to AWAITING_APPROVAL with approval {approval.id}."
            )
            return {
                "workflow_run_id": str(run.id),
                "status": run.status,
                "approval_id": str(approval.id),
                "draft_hash": draft_hash,
                "intended_action": intended_action,
                "target_record_id": target_record_id,
                "draft": final_synthesis,
                "approval_required": True,
            }

        # No side effect requested: mark completed directly
        run.status = WorkflowStatus.COMPLETED.value
        run.completed_at = datetime.now(timezone.utc)
        await self.session.commit()

        return {
            "workflow_run_id": str(run.id),
            "status": run.status,
            "iterations_executed": iteration,
            "evidence_count": len(all_evidence),
            "contradictions": evidence_aggregator.get_contradictions(),
            "synthesis": final_synthesis,
            "draft_hash": draft_hash,
            "approval_required": False,
        }

    async def resume_workflow_approval(
        self,
        run_id: uuid.UUID,
        approval_id: uuid.UUID,
        user: User,
        user_role: str,
        decision: str,  # "APPROVED", "REJECTED", "CANCELLED"
        rejection_reason: Optional[str] = None,
        crm_sync_service: Optional[Any] = None,
        proposed_action: str = "CRM_CREATE_NOTE",
        proposed_target: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Resumes a paused workflow awaiting human approval:
        1. Loads authoritative server-side WorkflowRun and CRMWriteBackApproval.
        2. Re-authorizes caller role and department boundaries.
        3. Revalidates cryptographic draft hash against server-stored draft (detects tampering).
        4. Revalidates action and target match.
        5. Handles rejection, cancellation, and expiry safely without side-effects.
        6. Executes idempotent CRM write-back upon verified approval.
        """
        run = await self.session.get(WorkflowRun, run_id)
        if not run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow run not found.")

        if run.status != WorkflowStatus.AWAITING_APPROVAL.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Workflow run {run_id} is in status '{run.status}', not 'AWAITING_APPROVAL'.",
            )

        approval = await self.session.get(CRMWriteBackApproval, approval_id)
        if not approval or approval.workflow_run_id != run.id:
            raise PermissionError(f"Approval ticket '{approval_id}' is invalid or does not belong to workflow {run_id}.")

        # Re-authorization at execution time
        if user_role != "ADMIN":
            if approval.department_id:
                user_dept = await self.session.scalar(
                    select(UserDepartment).where(
                        UserDepartment.user_id == user.id,
                        UserDepartment.department_id == approval.department_id,
                    )
                )
                if not user_dept:
                    dept = await self.session.get(Department, approval.department_id)
                    dept_name = dept.code if dept else str(approval.department_id)
                    raise PermissionError(f"User is not authorized for department '{dept_name}' at execution time.")

            if run.workspace_id:
                member = await self.session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == run.workspace_id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not member:
                    raise PermissionError("User is not an authorized member of this workspace at execution time.")

        # Rejection decision
        if decision == "REJECTED":
            approval.status = "REJECTED"
            run.status = WorkflowStatus.REJECTED.value
            run.error_message = rejection_reason or "Approval rejected by human reviewer."
            await self.session.commit()
            return {
                "workflow_run_id": str(run.id),
                "approval_id": str(approval.id),
                "status": run.status,
                "side_effect_executed": False,
            }

        # Cancellation decision
        if decision == "CANCELLED":
            approval.status = "REVOKED"
            approval.revoked_at = datetime.now(timezone.utc)
            run.status = WorkflowStatus.CANCELLED.value
            run.cancelled_at = datetime.now(timezone.utc)
            await self.session.commit()
            return {
                "workflow_run_id": str(run.id),
                "approval_id": str(approval.id),
                "status": run.status,
                "side_effect_executed": False,
            }

        # Expiry check
        now = datetime.now(timezone.utc)
        if approval.expires_at <= now:
            approval.status = "EXPIRED"
            run.status = WorkflowStatus.EXPIRED.value
            run.error_message = "Approval record has expired."
            await self.session.commit()
            raise PermissionError("Approval record has expired.")

        # If decision is not APPROVED, reject unknown decision
        if decision != "APPROVED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid approval decision '{decision}'. Must be APPROVED, REJECTED, or CANCELLED.",
            )

        # Retrieve authoritative server-stored draft from synthesis task
        synth_task = await self.session.scalar(
            select(WorkflowTask).where(
                and_(
                    WorkflowTask.workflow_run_id == run.id,
                    WorkflowTask.task_type == TaskType.SYNTHESIS.value,
                )
            )
        )
        stored_draft = ""
        if synth_task and synth_task.output_result:
            stored_draft = synth_task.output_result.get("draft_text", "")

        current_draft_hash = compute_canonical_draft_hash(stored_draft)

        # Draft Tampering Verification: Reject and revoke if draft changed
        if approval.approved_draft_hash is not None:
            if approval.approved_draft_hash != current_draft_hash:
                approval.status = "REVOKED"
                approval.revoked_at = now
                run.status = WorkflowStatus.FAILED.value
                run.failed_at = now
                run.error_message = (
                    f"Draft content has been modified since approval was granted. "
                    f"Expected hash {approval.approved_draft_hash}, got {current_draft_hash}."
                )
                await self.session.commit()
                raise PermissionError(
                    f"Draft content has been modified since approval was granted. "
                    f"Prior approval ticket {approval.id} has been revoked."
                )
        else:
            # Legacy unhashed approval policy
            if user_role != "ADMIN":
                raise PermissionError(
                    "Legacy approval ticket lacks cryptographic draft hash. "
                    "Non-admin callers cannot execute write-back with unverified legacy tickets."
                )

        # Action & Target Verification
        intended_action = getattr(approval, "intended_action", "CRM_CREATE_NOTE")
        is_action_match = (
            intended_action == proposed_action
            or (intended_action == "create_one_note" and proposed_action == "CRM_CREATE_NOTE")
            or (intended_action == "CRM_CREATE_NOTE" and proposed_action == "create_one_note")
        )
        if not is_action_match:
            raise PermissionError(
                f"Approval intended action '{intended_action}' cannot authorize '{proposed_action}'."
            )

        if proposed_target is not None and str(proposed_target).strip() != approval.external_record_id:
            raise PermissionError(
                f"Approval for record '{approval.external_record_id}' cannot authorize target '{proposed_target}'."
            )

        # Transition to EXECUTING
        run.status = WorkflowStatus.EXECUTING.value
        run.executing_at = now
        run.approved_at = now
        await self.session.commit()

        # Idempotent CRM Execution
        # Route write-back execution through WorkspaceMCPConnectionService if workspace MCP is available
        writeback_result = None
        if run.workspace_id:
            try:
                from rag_app.db.models import WorkspaceMCPConnection
                from rag_app.mcp.connection_service import WorkspaceMCPConnectionService

                conn_stmt = select(WorkspaceMCPConnection).where(
                    WorkspaceMCPConnection.workspace_id == run.workspace_id,
                    WorkspaceMCPConnection.is_enabled == True,
                )
                conns = (await self.session.scalars(conn_stmt)).all()
                twenty_conn = next(
                    (c for c in conns if "twenty" in c.name.lower() or "crm" in c.name.lower()),
                    conns[0] if conns else None,
                )
                if twenty_conn and "create_one_note" in (twenty_conn.enabled_tools or []):
                    mcp_service = WorkspaceMCPConnectionService(self.session)
                    idempotency_key = f"wf-{run.id}-{approval.id}"
                    tool_args = {
                        "title": f"Workflow Synthesis: {run.initial_query[:50]}",
                        "bodyV2": {"markdown": stored_draft},
                        "targetCompanyId": approval.external_record_id,
                    }
                    mcp_res = await mcp_service.invoke_workspace_tool(
                        user=user,
                        workspace_id=run.workspace_id,
                        connection_id=twenty_conn.id,
                        tool_name="create_one_note",
                        arguments=tool_args,
                        idempotency_key=idempotency_key,
                        approval_ticket_id=approval.id,
                        run_id=run.id,
                    )
                    writeback_result = {
                        "status": "COMPLETED",
                        "idempotency_key": idempotency_key,
                        "crm_note_id": (mcp_res.structured_content or {}).get("id") or "mcp-note-created",
                        "note_title": tool_args["title"],
                        "is_duplicate_replay": (mcp_res.structured_content or {}).get("is_replay", False),
                        "approval_id": str(approval.id),
                        "mcp_operation_id": (mcp_res.structured_content or {}).get("operation_id"),
                    }
            except Exception as e:
                logger.warning(f"Workspace MCP connection check failed, falling back to CRM sync service: {e}")

        # Fallback to legacy CRMSyncService if workspace MCP connection not configured
        if writeback_result is None:
            if crm_sync_service is None:
                from rag_app.crm.sync_service import CRMSyncService
                crm_sync_service = CRMSyncService(self.session)

            idempotency_key = f"wf-{run.id}-{approval.id}"
            writeback_result = await crm_sync_service.create_meeting_note_writeback(
                company_id=approval.external_record_id,
                note_title=f"Workflow Synthesis: {run.initial_query[:50]}",
                summary_markdown=stored_draft,
                idempotency_key=idempotency_key,
                user_id=user.id,
                user_role=user_role,
                approval_id=approval.id,
                approved=True,
            )

        # Transition to COMPLETED
        run.status = WorkflowStatus.COMPLETED.value
        run.completed_at = datetime.now(timezone.utc)
        await self.session.commit()

        return {
            "workflow_run_id": str(run.id),
            "approval_id": str(approval.id),
            "status": run.status,
            "side_effect_executed": True,
            "writeback_result": writeback_result,
        }
