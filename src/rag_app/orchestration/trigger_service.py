"""
Workflow Trigger Router, Transactional Outbox, and Scheduling Engine.
Provides:
1. Authenticated Manual/API Trigger with scoped idempotency and immutable version resolution.
2. Signed Event/Webhook Ingest with HMAC-SHA256 verification, tenant routing, and replay protection.
3. Transactional Outbox pattern: durable event acceptance and version pinning BEFORE acknowledgement.
4. Scheduled Workflow Engine with cron/interval evaluation, timezone support, and window idempotency.
5. Outbox Sweep & Crash Recovery for abandoned or interrupted trigger events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import logging
from typing import Any, Optional
import uuid
from zoneinfo import ZoneInfo

from fastapi import HTTPException, status
from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.core.config import settings
from rag_app.db.models import (
    Department,
    User,
    WorkflowDefinition,
    WorkflowRun,
    WorkflowScheduleState,
    WorkflowTask,
    WorkflowTriggerEvent,
    WorkflowVersion,
    Workspace,
    WorkspaceMember,
)
from rag_app.orchestration.executor import DAGExecutionEngine
from rag_app.orchestration.workflow_plan_compiler import compile_workflow_version_to_plan
from rag_app.schemas.workflow import (
    BudgetLimits,
    compute_canonical_request_hash,
    ScopeType,
    WorkflowStatus,
    WorkflowTaskStatus,
)
from rag_app.schemas.workflow_definition import WorkflowTriggerType, WorkflowVersionStatus
from rag_app.schemas.workflow_trigger import (
    MisfirePolicy,
    OutboxStatus,
    TriggerSource,
    WorkflowScheduleConfig,
    WorkflowTriggerRequest,
    WorkflowTriggerResponse,
    WorkflowWebhookResponse,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Deterministic Cron / Schedule Window Evaluator (Zero External Dependencies)
# ============================================================================

def _matches_cron_field(val: int, expr: str) -> bool:
    """Matches an integer against a single cron field expression."""
    expr = expr.strip()
    if expr == "*":
        return True
    if "/" in expr:
        parts = expr.split("/")
        if len(parts) == 2 and parts[1].isdigit():
            step = int(parts[1])
            start = 0 if parts[0] == "*" else int(parts[0])
            return step > 0 and (val - start) % step == 0 and val >= start
    if "," in expr:
        return any(_matches_cron_field(val, sub) for sub in expr.split(","))
    if "-" in expr:
        sub_parts = expr.split("-")
        if len(sub_parts) == 2 and sub_parts[0].isdigit() and sub_parts[1].isdigit():
            return int(sub_parts[0]) <= val <= int(sub_parts[1])
    if expr.isdigit():
        return val == int(expr)
    return False


def is_cron_due(cron_expr: str, dt: datetime) -> bool:
    """
    Evaluates standard 5-part cron: 'minute hour day_of_month month day_of_week'.
    Sunday can be 0 or 7.
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    min_expr, hr_expr, dom_expr, mon_expr, dow_expr = fields

    # Python weekday: Mon=0 .. Sun=6. Cron: Sun=0 or 7, Mon=1 .. Sat=6.
    cron_dow = (dt.weekday() + 1) % 7

    return (
        _matches_cron_field(dt.minute, min_expr)
        and _matches_cron_field(dt.hour, hr_expr)
        and _matches_cron_field(dt.day, dom_expr)
        and _matches_cron_field(dt.month, mon_expr)
        and (_matches_cron_field(cron_dow, dow_expr) or (dow_expr in ("0", "7") and cron_dow in (0, 7)))
    )


@dataclass
class ScheduleOccurrence:
    utc_time: datetime
    local_time: datetime
    window_key: str
    is_gap_catchup: bool = False


def compute_schedule_window_key(workflow_id: uuid.UUID, dt: datetime) -> str:
    """
    Computes a deterministic occurrence key for schedule deduplication.
    Distinguishes repeated local times (DST fall-back) via fold and UTC offset.
    For standard UTC datetimes without fold, preserves canonical 'schedule:{wf_id}:{YYYY-MM-DD-HH-MM}'.
    """
    base = dt.strftime("%Y-%m-%d-%H-%M")
    fold = getattr(dt, "fold", 0)
    tz_offset = dt.strftime("%z") if dt.tzinfo else ""
    if fold or (tz_offset and tz_offset not in ("+0000", "-0000", "")):
        return f"schedule:{workflow_id}:{base}:{tz_offset}:f{fold}"
    return f"schedule:{workflow_id}:{base}"


def evaluate_due_schedule_occurrences(
    workflow_id: uuid.UUID,
    cron_expr: Optional[str],
    interval_sec: Optional[int],
    local_tz: ZoneInfo | timezone,
    start_cursor_utc: Optional[datetime],
    target_utc: datetime,
    max_scan_minutes: int = 1440,
    dst_repeated_policy: str = "RUN_ONCE",
) -> tuple[list[ScheduleOccurrence], datetime, bool]:
    """
    Evaluates every due schedule occurrence between start_cursor_utc and target_utc.
    Returns:
    - occurrences: list of ScheduleOccurrence
    - advanced_cursor_utc: new UTC cursor position
    - has_more: True if catch-up scan was bounded by max_scan_minutes and more remain
    """
    target_minute_utc = target_utc.astimezone(timezone.utc).replace(second=0, microsecond=0)

    # 1. Initial activation (no previous schedule cursor recorded)
    if start_cursor_utc is None:
        local_time = target_minute_utc.astimezone(local_tz)
        is_due = False
        if cron_expr:
            is_due = is_cron_due(cron_expr, local_time)
        elif interval_sec:
            is_due = True

        occurrences: list[ScheduleOccurrence] = []
        if is_due:
            key = compute_schedule_window_key(workflow_id, local_time)
            occurrences.append(ScheduleOccurrence(target_minute_utc, local_time, key))
        return occurrences, target_minute_utc, False

    # 2. Subsequent evaluation: advance cursor from start_cursor_utc forward
    cursor_utc = start_cursor_utc.astimezone(timezone.utc).replace(second=0, microsecond=0)
    if cursor_utc >= target_minute_utc:
        return [], cursor_utc, False

    elapsed_minutes = int((target_minute_utc - cursor_utc).total_seconds() // 60)
    if elapsed_minutes <= 0:
        return [], cursor_utc, False

    if elapsed_minutes > max_scan_minutes:
        scan_end_utc = cursor_utc + timedelta(minutes=max_scan_minutes)
        has_more = True
        scan_minutes = max_scan_minutes
    else:
        scan_end_utc = target_minute_utc
        has_more = False
        scan_minutes = elapsed_minutes

    occurrences = []

    # 3. Interval-based evaluation
    if interval_sec:
        elapsed_sec = (scan_end_utc - cursor_utc).total_seconds()
        num_intervals = int(elapsed_sec // interval_sec)
        for i in range(1, num_intervals + 1):
            occ_utc = cursor_utc + timedelta(seconds=i * interval_sec)
            occ_local = occ_utc.astimezone(local_tz)
            key = compute_schedule_window_key(workflow_id, occ_local)
            occurrences.append(ScheduleOccurrence(occ_utc, occ_local, key))
        return occurrences, scan_end_utc, has_more

    # 4. Cron-based evaluation
    if cron_expr:
        seen_daily_times: set[tuple[int, int, int, int, int]] = set()
        prev_local: Optional[datetime] = cursor_utc.astimezone(local_tz)

        for step in range(1, scan_minutes + 1):
            m_utc = cursor_utc + timedelta(minutes=step)
            m_local = m_utc.astimezone(local_tz)

            # Check DST Spring-Forward Gap (nonexistent local time jump)
            if prev_local and m_local.date() == prev_local.date():
                local_diff_sec = (
                    (m_local.hour * 3600 + m_local.minute * 60) -
                    (prev_local.hour * 3600 + prev_local.minute * 60)
                )
                if local_diff_sec > 60:
                    for skipped_sec in range(60, local_diff_sec, 60):
                        gap_dt = prev_local + timedelta(seconds=skipped_sec)
                        if is_cron_due(cron_expr, gap_dt):
                            gap_key = compute_schedule_window_key(workflow_id, m_local) + ":gap_catchup"
                            occurrences.append(
                                ScheduleOccurrence(m_utc, m_local, gap_key, is_gap_catchup=True)
                            )

            # Standard Cron evaluation
            if is_cron_due(cron_expr, m_local):
                is_repeat = False
                time_tuple = (m_local.year, m_local.month, m_local.day, m_local.hour, m_local.minute)
                # Check DST Fall-Back Repeated Local Time Policy
                if m_local.fold == 1 and dst_repeated_policy == "RUN_ONCE":
                    if time_tuple in seen_daily_times:
                        is_repeat = True

                if not is_repeat:
                    seen_daily_times.add(time_tuple)
                    key = compute_schedule_window_key(workflow_id, m_local)
                    occurrences.append(ScheduleOccurrence(m_utc, m_local, key))

            prev_local = m_local

        return occurrences, scan_end_utc, has_more

    return [], scan_end_utc, has_more


# ============================================================================
# Workflow Trigger Service
# ============================================================================

class WorkflowTriggerService:
    """
    Handles reliable workflow triggering across Manual API calls, Webhook events,
    and Scheduled windows using a Transactional Outbox.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------------
    # 1. Manual / API Trigger
    # ------------------------------------------------------------------------

    async def trigger_manual(
        self,
        workflow_id: uuid.UUID,
        workspace_id: uuid.UUID,
        user: User,
        request: WorkflowTriggerRequest,
    ) -> WorkflowTriggerResponse:
        """
        Manually triggers a published workflow for execution:
        - Resolves active immutable WorkflowVersion.
        - Checks idempotency keys.
        - Persists transactional outbox event.
        - Executes DAG via DAGExecutionEngine.
        """
        # 1. Fetch workflow definition and verify workspace ownership
        wf_stmt = select(WorkflowDefinition).where(
            WorkflowDefinition.id == workflow_id,
            WorkflowDefinition.workspace_id == workspace_id,
            WorkflowDefinition.deleted_at.is_(None),
        )
        workflow = await self.session.scalar(wf_stmt)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found in workspace '{workspace_id}'.",
            )
        if not workflow.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Workflow '{workflow.name}' is inactive and cannot be triggered.",
            )
        if not workflow.active_version_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workflow '{workflow.name}' has no active published version. Publish a version before triggering.",
            )

        # 2. Fetch active published version (immutable snapshot)
        version = await self.session.get(WorkflowVersion, workflow.active_version_id)
        if not version or version.status != WorkflowVersionStatus.ACTIVE.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Active version '{workflow.active_version_id}' is missing or not in ACTIVE state.",
            )

        # 3. Idempotency Check
        if request.idempotency_key:
            existing_event = await self.session.scalar(
                select(WorkflowTriggerEvent).where(
                    WorkflowTriggerEvent.workspace_id == workspace_id,
                    WorkflowTriggerEvent.workflow_id == workflow_id,
                    WorkflowTriggerEvent.idempotency_key == request.idempotency_key,
                )
            )
            if existing_event:
                logger.info(
                    "Replaying existing workflow trigger for idempotency_key '%s'",
                    request.idempotency_key,
                )
                return WorkflowTriggerResponse(
                    event_id=existing_event.event_id,
                    workflow_id=existing_event.workflow_id,
                    workflow_version_id=existing_event.workflow_version_id,
                    workflow_run_id=existing_event.workflow_run_id,
                    status=existing_event.status,
                    is_replay=True,
                    created_at=existing_event.created_at,
                    dispatched_at=existing_event.dispatched_at,
                )

        # 4. Transactional Outbox Event Creation (PINNING immutable version)
        event_id = f"manual-{uuid.uuid4().hex[:16]}"
        outbox_event = WorkflowTriggerEvent(
            id=uuid.uuid4(),
            event_id=event_id,
            event_source=TriggerSource.API.value,
            event_name="manual.trigger",
            workspace_id=workspace_id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,  # Pinned exact immutable version
            payload=request.inputs,
            idempotency_key=request.idempotency_key,
            status=OutboxStatus.PENDING.value,
        )
        self.session.add(outbox_event)
        await self.session.flush()

        # 5. Create WorkflowRun record with immutable runtime snapshot (Phase 5E)
        run_query = request.inputs.get("query") or f"Manual execution for {workflow.name}"
        req_hash = compute_canonical_request_hash(
            query=request.inputs,
            workspace_id=workspace_id,
            budget_limits=request.budget_limits.model_dump() if request.budget_limits else None,
        )

        from rag_app.orchestration.runtime_snapshot import capture_runtime_snapshot
        snapshot = await capture_runtime_snapshot(
            session=self.session,
            workspace_id=workspace_id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            steps_spec=version.steps_spec,
            trigger_payload=request.inputs,
        )

        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user.id,
            workspace_id=workspace_id,
            workflow_id=workflow.id,
            workflow_version_id=version.id,
            scope_type=ScopeType.WORKSPACE.value,
            status=WorkflowStatus.RUNNING.value,
            initial_query=run_query,
            request_hash=req_hash,
            idempotency_key=request.idempotency_key,
            max_tokens_budget=request.budget_limits.max_tokens if request.budget_limits else 25000,
            max_provider_calls_budget=request.budget_limits.max_provider_calls if request.budget_limits else 15,
            provider_snapshot=snapshot,
        )
        self.session.add(run)
        await self.session.flush()

        # 6. Compile Plan from pinned version
        plan = compile_workflow_version_to_plan(
            workflow=workflow,
            version=version,
            run=run,
            trigger_payload=request.inputs,
        )

        # 7. Execute DAG Plan with DAGExecutionEngine
        engine_inst = DAGExecutionEngine(session=self.session)
        task_results = await engine_inst.execute_plan(
            plan=plan,
            run=run,
            context={"trigger": request.inputs},
        )

        # 8. Reconcile final run status & outbox status
        all_succeeded = all(
            r.status in (WorkflowTaskStatus.COMPLETED, WorkflowTaskStatus.SKIPPED)
            for r in task_results.values()
        )
        run.status = WorkflowStatus.COMPLETED.value if all_succeeded else WorkflowStatus.FAILED.value
        run.completed_at = datetime.now(timezone.utc)

        outbox_event.status = OutboxStatus.DISPATCHED.value
        outbox_event.workflow_run_id = run.id
        outbox_event.dispatched_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(outbox_event)
        await self.session.refresh(run)

        return WorkflowTriggerResponse(
            event_id=outbox_event.event_id,
            workflow_id=outbox_event.workflow_id,
            workflow_version_id=outbox_event.workflow_version_id,
            workflow_run_id=run.id,
            status=outbox_event.status,
            is_replay=False,
            created_at=outbox_event.created_at,
            dispatched_at=outbox_event.dispatched_at,
        )

    # ------------------------------------------------------------------------
    # 2. Event / Webhook Trigger
    # ------------------------------------------------------------------------

    async def handle_inbound_webhook(
        self,
        raw_bytes: bytes,
        signature_header: Optional[str],
        timestamp_header: Optional[str],
        nonce_header: Optional[str],
        system_user_id: Optional[uuid.UUID] = None,
    ) -> WorkflowWebhookResponse:
        """
        Receives signed webhook, validates HMAC, maps identity, pins active version,
        persists transactional outbox event, and dispatches execution.
        """
        # 1. Payload size check
        max_bytes = getattr(settings, "twenty_webhook_max_bytes", 1024 * 1024)
        if len(raw_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Webhook payload exceeds maximum size limit of {max_bytes} bytes.",
            )

        # 2. HMAC-SHA256 signature verification (fail closed)
        webhook_secret = getattr(settings, "twenty_webhook_secret", None)
        if not webhook_secret or not str(webhook_secret).strip():
            logger.error("Webhook rejected: twenty_webhook_secret is not configured on server.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Twenty CRM webhook secret is not configured on server; rejecting fail-closed.",
            )

        if not signature_header or not signature_header.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing webhook signature header (X-Twenty-Webhook-Signature).",
            )

        expected_sig = hmac.new(webhook_secret.encode("utf-8"), raw_bytes, hashlib.sha256).hexdigest()
        sig_clean = signature_header.replace("sha256=", "").strip()
        is_valid = hmac.compare_digest(expected_sig, sig_clean)

        # Timestamped format fallback
        if not is_valid and timestamp_header:
            try:
                raw_str = raw_bytes.decode("utf-8")
                ts_sig = hmac.new(
                    webhook_secret.encode("utf-8"),
                    f"{timestamp_header}:{raw_str}".encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                is_valid = hmac.compare_digest(ts_sig, sig_clean)
            except Exception:
                pass

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook HMAC-SHA256 signature.",
            )

        # 3. Timestamp drift / skew check (max 300s old, 60s future)
        if timestamp_header:
            try:
                ts_val = timestamp_header.strip()
                if ts_val.isdigit():
                    ts_sec = float(ts_val) / 1000.0 if len(ts_val) >= 12 else float(ts_val)
                elif "T" in ts_val:
                    ts_sec = datetime.fromisoformat(ts_val.replace("Z", "+00:00")).timestamp()
                else:
                    ts_sec = float(ts_val)

                now_sec = datetime.now(timezone.utc).timestamp()
                drift = now_sec - ts_sec
                if drift > 300.0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Webhook timestamp expired: event is {drift:.1f}s old (max skew 300s).",
                    )
                if drift < -60.0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Webhook timestamp is in the future: {-drift:.1f}s ahead of server clock.",
                    )
            except HTTPException:
                raise
            except Exception as te:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Malformed webhook timestamp: {te}",
                )

        # 4. Parse JSON payload
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except Exception as je:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed JSON payload: {je}",
            )

        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            parsed = parsed[0]

        # 5. Server-Verified Identity Mapping (Anti-spoofing)
        webhook_id = parsed.get("webhookId")
        mappings = settings.get_twenty_webhook_mappings() if hasattr(settings, "get_twenty_webhook_mappings") else {}
        mapping_entry = mappings.get(str(webhook_id)) if webhook_id else None

        if not mapping_entry:
            # Check if environment is test/dev and explicit routing is present
            env = getattr(settings, "environment", "development").lower()
            if env not in ("test", "testing", "dev", "development") or not parsed.get("workspace_id"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unauthorized or unregistered webhook identity '{webhook_id}'. Routing identity cannot be verified.",
                )
            target_workspace_id = uuid.UUID(parsed["workspace_id"])
            target_workflow_id = uuid.UUID(parsed["workflow_id"]) if parsed.get("workflow_id") else None
        else:
            if isinstance(mapping_entry, dict):
                target_workspace_id = uuid.UUID(mapping_entry["workspace_id"])
                target_workflow_id = uuid.UUID(mapping_entry["workflow_id"]) if mapping_entry.get("workflow_id") else None
            else:
                target_workspace_id = uuid.UUID(mapping_entry)
                target_workflow_id = None

        event_name = parsed.get("event") or parsed.get("eventName") or "unknown.event"
        resolved_event_id = str(nonce_header or parsed.get("eventId") or parsed.get("id") or uuid.uuid4())

        # 6. Deduplication check against outbox
        existing_event = await self.session.scalar(
            select(WorkflowTriggerEvent).where(WorkflowTriggerEvent.event_id == resolved_event_id)
        )
        if existing_event:
            return WorkflowWebhookResponse(
                status="duplicate",
                event_id=resolved_event_id,
                workflow_run_id=str(existing_event.workflow_run_id) if existing_event.workflow_run_id else None,
                message="Duplicate event ignored (replay protection).",
            )

        # 7. Resolve target workflow & PIN active version
        if not target_workflow_id:
            # Look up active workflow subscribing to WEBHOOK trigger in this workspace
            wf_candidates = (await self.session.scalars(
                select(WorkflowDefinition).where(
                    WorkflowDefinition.workspace_id == target_workspace_id,
                    WorkflowDefinition.is_active == True,
                    WorkflowDefinition.active_version_id.isnot(None),
                    WorkflowDefinition.deleted_at.is_(None),
                )
            )).all()

            for wf in wf_candidates:
                v = await self.session.get(WorkflowVersion, wf.active_version_id)
                if v and v.status == WorkflowVersionStatus.ACTIVE.value:
                    trig = v.trigger_spec or {}
                    t_type = trig.get("trigger_type")
                    if t_type in (WorkflowTriggerType.WEBHOOK.value, WorkflowTriggerType.EVENT.value):
                        target_workflow_id = wf.id
                        break

        if not target_workflow_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active workflow found to handle webhook event '{event_name}' in workspace '{target_workspace_id}'.",
            )

        wf = await self.session.get(WorkflowDefinition, target_workflow_id)
        if not wf or not wf.active_version_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Target workflow '{target_workflow_id}' has no published active version.",
            )

        version = await self.session.get(WorkflowVersion, wf.active_version_id)
        if not version or version.status != WorkflowVersionStatus.ACTIVE.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Active version '{wf.active_version_id}' is not in ACTIVE state.",
            )

        # 8. Transactional Outbox Persistence (Must commit BEFORE external acknowledgement)
        outbox_event = WorkflowTriggerEvent(
            id=uuid.uuid4(),
            event_id=resolved_event_id,
            event_source=TriggerSource.TWENTY_CRM.value,
            event_name=event_name,
            workspace_id=target_workspace_id,
            workflow_id=wf.id,
            workflow_version_id=version.id,  # PINNED VERSION
            payload=parsed,
            status=OutboxStatus.PENDING.value,
        )
        self.session.add(outbox_event)
        await self.session.commit()
        await self.session.refresh(outbox_event)

        # 9. Dispatch execution synchronously or hand off to outbox
        run = await self._dispatch_pinned_event(outbox_event, system_user_id=system_user_id)

        return WorkflowWebhookResponse(
            status="accepted",
            event_id=resolved_event_id,
            workflow_run_id=str(run.id) if run else None,
            message=f"Event '{event_name}' accepted and dispatched with pinned version {version.version_number}.",
        )

    # ------------------------------------------------------------------------
    # 3. Scheduled Triggers
    # ------------------------------------------------------------------------

    async def process_scheduled_workflows(
        self,
        as_of_time: Optional[datetime] = None,
        system_user_id: Optional[uuid.UUID] = None,
        workspace_id: Optional[uuid.UUID] = None,
        workflow_id: Optional[uuid.UUID] = None,
    ) -> int:
        """
        Evaluates active workflows configured with SCHEDULE triggers.
        Computes schedule windows, applies misfire policies, prevents duplicate runs,
        and executes eligible workflows.
        Returns count of workflows triggered.
        """
        now_utc = as_of_time or datetime.now(timezone.utc)
        triggered_count = 0

        # Query active workflows with published active versions
        stmt = select(WorkflowDefinition).where(
            WorkflowDefinition.is_active == True,
            WorkflowDefinition.active_version_id.isnot(None),
            WorkflowDefinition.deleted_at.is_(None),
        )
        if workflow_id:
            stmt = stmt.where(WorkflowDefinition.id == workflow_id)
        if workspace_id:
            stmt = stmt.where(WorkflowDefinition.workspace_id == workspace_id)

        workflows = (await self.session.scalars(stmt)).all()

        for wf in workflows:
            version = await self.session.get(WorkflowVersion, wf.active_version_id)
            if not version or version.status != WorkflowVersionStatus.ACTIVE.value:
                continue

            trig_spec = version.trigger_spec or {}
            if trig_spec.get("trigger_type") != WorkflowTriggerType.SCHEDULE.value:
                continue

            config = trig_spec.get("config", {})
            cron_expr = config.get("cron")
            interval_sec = config.get("interval_seconds")
            tz_str = config.get("timezone", "UTC")
            misfire_pol = config.get("misfire_policy", MisfirePolicy.RUN_ONCE.value)

            try:
                local_tz = ZoneInfo(tz_str)
            except Exception:
                local_tz = timezone.utc

            # Row-level locking to prevent concurrent scheduler instances from racing
            state = await self.session.scalar(
                select(WorkflowScheduleState)
                .where(WorkflowScheduleState.workflow_id == wf.id)
                .with_for_update(skip_locked=True)
            )
            if not state:
                # Check if state exists but was locked by another worker
                existing_check = await self.session.scalar(
                    select(WorkflowScheduleState).where(WorkflowScheduleState.workflow_id == wf.id)
                )
                if existing_check:
                    logger.debug("Workflow schedule state for %s is locked by another worker; skipping.", wf.id)
                    continue

                try:
                    state = WorkflowScheduleState(
                        id=uuid.uuid4(),
                        workflow_id=wf.id,
                        status="ACTIVE",
                    )
                    self.session.add(state)
                    await self.session.commit()
                    state = await self.session.scalar(
                        select(WorkflowScheduleState)
                        .where(WorkflowScheduleState.workflow_id == wf.id)
                        .with_for_update(skip_locked=True)
                    )
                except Exception:
                    await self.session.rollback()
                    state = await self.session.scalar(
                        select(WorkflowScheduleState)
                        .where(WorkflowScheduleState.workflow_id == wf.id)
                        .with_for_update(skip_locked=True)
                    )

            if not state:
                continue

            # Evaluate due schedule occurrences across the bounded window between last_scheduled_at and now_utc
            # The 1440-minute catch-up limit is a true per-sweep resource bound (strictly at most 1440m per sweep).
            occurrences, next_cursor, has_more = evaluate_due_schedule_occurrences(
                workflow_id=wf.id,
                cron_expr=cron_expr,
                interval_sec=interval_sec,
                local_tz=local_tz,
                start_cursor_utc=state.last_scheduled_at,
                target_utc=now_utc,
                max_scan_minutes=1440,
                dst_repeated_policy=config.get("dst_repeated_policy", "RUN_ONCE"),
            )

            if not occurrences:
                state.last_scheduled_at = next_cursor
                if not has_more:
                    state.status = "ACTIVE"
                elif state.status not in ("RECOVERED", "RECOVERING"):
                    state.status = "RECOVERING"
                await self.session.commit()
                continue

            # Occurrences found in window!
            if misfire_pol == MisfirePolicy.SKIP.value:
                state.misfire_count += len(occurrences)
                state.last_scheduled_at = next_cursor
                state.last_window_key = occurrences[-1].window_key
                state.status = "ACTIVE" if not has_more else "RECOVERED"
                await self.session.commit()
                logger.info(
                    "Misfire SKIP applied for workflow %s: skipped %d occurrences, cursor advanced to %s (has_more=%s)",
                    wf.id,
                    len(occurrences),
                    next_cursor,
                    has_more,
                )
                continue

            # RUN_ONCE policy:
            # If a catch-up run was ALREADY dispatched in an earlier slice of this recovery episode,
            # all occurrences in subsequent catch-up slices are treated as misfires.
            if state.status == "RECOVERED":
                state.misfire_count += len(occurrences)
                state.last_scheduled_at = next_cursor
                state.last_window_key = occurrences[-1].window_key
                state.status = "ACTIVE" if not has_more else "RECOVERED"
                await self.session.commit()
                logger.info(
                    "RUN_ONCE catch-up coalescing for workflow %s: run already dispatched in recovery episode, recorded %d misfires, cursor advanced to %s (has_more=%s)",
                    wf.id,
                    len(occurrences),
                    next_cursor,
                    has_more,
                )
                continue

            # No catch-up run has been dispatched yet for this episode: dispatch exactly ONE run!
            if len(occurrences) > 1:
                state.misfire_count += (len(occurrences) - 1)

            latest_occ = occurrences[-1]
            event_id = f"sched-{uuid.uuid4().hex[:12]}"
            outbox_event = WorkflowTriggerEvent(
                id=uuid.uuid4(),
                event_id=event_id,
                event_source=TriggerSource.SCHEDULE.value,
                event_name="schedule.tick",
                workspace_id=wf.workspace_id,
                workflow_id=wf.id,
                workflow_version_id=version.id,  # PINNED VERSION
                payload={
                    "window_key": latest_occ.window_key,
                    "scheduled_at": latest_occ.utc_time.isoformat(),
                    "coalesced_runs": len(occurrences),
                    "is_catchup": len(occurrences) > 1 or has_more,
                    "is_gap_catchup": latest_occ.is_gap_catchup,
                },
                idempotency_key=latest_occ.window_key,
                status=OutboxStatus.PENDING.value,
            )
            self.session.add(outbox_event)

            state.last_scheduled_at = next_cursor
            state.last_window_key = latest_occ.window_key
            state.status = "ACTIVE" if not has_more else "RECOVERED"
            await self.session.commit()
            await self.session.refresh(outbox_event)

            # Dispatch execution
            await self._dispatch_pinned_event(outbox_event, system_user_id=system_user_id)
            triggered_count += 1

        return triggered_count

    # ------------------------------------------------------------------------
    # 4. Outbox Recovery & Event Dispatch
    # ------------------------------------------------------------------------

    async def recover_pending_outbox_events(
        self,
        limit: int = 10,
        system_user_id: Optional[uuid.UUID] = None,
        workspace_id: Optional[uuid.UUID] = None,
        workflow_id: Optional[uuid.UUID] = None,
    ) -> int:
        """
        Sweeps abandoned or interrupted PENDING outbox events and executes them
        using their pinned immutable version.
        """
        stmt = (
            select(WorkflowTriggerEvent)
            .where(WorkflowTriggerEvent.status == OutboxStatus.PENDING.value)
        )
        if workflow_id:
            stmt = stmt.where(WorkflowTriggerEvent.workflow_id == workflow_id)
        if workspace_id:
            stmt = stmt.where(WorkflowTriggerEvent.workspace_id == workspace_id)
        stmt = stmt.order_by(WorkflowTriggerEvent.created_at.asc()).limit(limit)
        events = (await self.session.scalars(stmt)).all()
        recovered_count = 0

        for ev in events:
            try:
                ev.status = OutboxStatus.PROCESSING.value
                await self.session.commit()
                run = await self._dispatch_pinned_event(ev, system_user_id=system_user_id)
                if run:
                    recovered_count += 1
            except Exception as exc:
                logger.error("Outbox recovery failed for event '%s': %s", ev.event_id, exc)
                ev.status = OutboxStatus.FAILED.value
                ev.error_message = str(exc)
                ev.retry_count += 1
                await self.session.commit()

        return recovered_count

    async def _dispatch_pinned_event(
        self,
        event: WorkflowTriggerEvent,
        system_user_id: Optional[uuid.UUID] = None,
    ) -> Optional[WorkflowRun]:
        """
        Internal worker that loads the pinned WorkflowVersion, compiles the plan,
        and runs it through the existing DAGExecutionEngine.
        """
        # Resolve user
        user_id = system_user_id
        if not user_id:
            wf = await self.session.get(WorkflowDefinition, event.workflow_id)
            user_id = wf.created_by_id if wf else None

        if not user_id:
            first_user = await self.session.scalar(select(User).limit(1))
            user_id = first_user.id if first_user else uuid.uuid4()

        wf = await self.session.get(WorkflowDefinition, event.workflow_id)
        version = await self.session.get(WorkflowVersion, event.workflow_version_id)
        if not wf or not version:
            event.status = OutboxStatus.FAILED.value
            event.error_message = f"Pinned workflow or version not found (wf: {event.workflow_id}, ver: {event.workflow_version_id})"
            await self.session.commit()
            return None

        # Check if an existing run was already created for this idempotency key
        if event.idempotency_key:
            existing_run = await self.session.scalar(
                select(WorkflowRun).where(
                    WorkflowRun.user_id == user_id,
                    WorkflowRun.workspace_id == event.workspace_id,
                    WorkflowRun.idempotency_key == event.idempotency_key,
                )
            )
            if existing_run:
                logger.info(
                    "Replaying/reusing existing WorkflowRun %s for idempotency_key '%s'",
                    existing_run.id,
                    event.idempotency_key,
                )
                event.workflow_run_id = existing_run.id
                event.status = OutboxStatus.DISPATCHED.value
                await self.session.commit()
                return existing_run

        # Create WorkflowRun with immutable runtime snapshot (Phase 5E)
        query_text = event.payload.get("query") if isinstance(event.payload, dict) else None
        query_text = query_text or f"Triggered execution for {wf.name} ({event.event_name})"

        from rag_app.orchestration.runtime_snapshot import capture_runtime_snapshot
        snapshot = await capture_runtime_snapshot(
            session=self.session,
            workspace_id=event.workspace_id,
            workflow_id=wf.id,
            workflow_version_id=version.id,
            steps_spec=version.steps_spec,
            trigger_payload=event.payload if isinstance(event.payload, dict) else {},
        )

        run = WorkflowRun(
            id=uuid.uuid4(),
            user_id=user_id,
            workspace_id=event.workspace_id,
            workflow_id=wf.id,
            workflow_version_id=version.id,
            scope_type=ScopeType.WORKSPACE.value,
            status=WorkflowStatus.RUNNING.value,
            initial_query=query_text,
            request_hash=compute_canonical_request_hash(
                query=event.payload,
                workspace_id=event.workspace_id,
            ),
            idempotency_key=event.idempotency_key,
            max_tokens_budget=25000,
            max_provider_calls_budget=15,
            provider_snapshot=snapshot,
        )
        self.session.add(run)
        await self.session.flush()

        # Compile DAG plan using PINNED version
        plan = compile_workflow_version_to_plan(
            workflow=wf,
            version=version,
            run=run,
            trigger_payload=event.payload,
            allow_draft=True,  # allow execution of historically pinned version even if archived
        )

        # Execute DAG plan
        engine_inst = DAGExecutionEngine(session=self.session)
        task_results = await engine_inst.execute_plan(
            plan=plan,
            run=run,
            context={"trigger": event.payload},
        )

        all_ok = all(
            r.status in (WorkflowTaskStatus.COMPLETED, WorkflowTaskStatus.SKIPPED)
            for r in task_results.values()
        )
        run.status = WorkflowStatus.COMPLETED.value if all_ok else WorkflowStatus.FAILED.value
        run.completed_at = datetime.now(timezone.utc)

        event.status = OutboxStatus.DISPATCHED.value
        event.workflow_run_id = run.id
        event.dispatched_at = datetime.now(timezone.utc)

        await self.session.commit()
        return run

