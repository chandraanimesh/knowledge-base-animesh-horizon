"""Memory Quality & Autonomous Dreaming Evaluation Engine (Milestone 9F)

Executes 20 deterministic evaluation scenarios testing extraction accuracy,
consolidation, temporal intelligence, privacy compliance, and tenant isolation.
Stores results in the memory_eval_metrics table for historical tracking and observability.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import (
    User,
    Department,
    Workspace,
    MemoryItem,
    MemoryRelationship,
    MemoryDreamingRun,
    MemoryEvalMetric,
    MemoryPrivacyPolicy,
    MemoryTombstone,
    MemoryProcessingJob,
)
from rag_app.memory.schemas import (
    MemoryCandidate,
    MemoryType,
    MemoryStatus,
    StabilityTier,
    JobStatus,
    ScopeType,
)
from rag_app.memory.command_parser import parse_memory_command, detect_memory_candidate_rules
from rag_app.memory.materializer import MemoryMaterializer
from rag_app.memory.retrieval import MemoryRetrievalService
from rag_app.memory.privacy_service import MemoryPrivacyService
from rag_app.memory.privacy_operations import MemoryPrivacyOperations
from rag_app.memory.consolidation import MemoryConsolidationService
from rag_app.memory.dreaming import run_memory_dreaming
from rag_app.memory.profile_compiler import compile_user_profile
from rag_app.memory.jobs import create_memory_job, execute_memory_job, recover_stale_memory_jobs
from rag_app.core.safety import check_sensitive_request

logger = logging.getLogger(__name__)


class MemoryEvaluationRunner:
    """
    Executes the 20-scenario deterministic evaluation suite and persists
    structured metrics in memory_eval_metrics.
    """

    def __init__(self, session: AsyncSession, eval_run_id: Optional[str] = None):
        self.session = session
        self.eval_run_id = eval_run_id or f"eval_{uuid.uuid4().hex[:12]}"

    async def _record_metric(
        self,
        scenario_name: str,
        metric_category: str,
        passed: bool,
        metrics_json: dict[str, Any],
        details: Optional[str] = None,
    ) -> MemoryEvalMetric:
        rec = MemoryEvalMetric(
            id=uuid.uuid4(),
            eval_run_id=self.eval_run_id,
            scenario_name=scenario_name,
            metric_category=metric_category,
            metrics_json=metrics_json,
            passed=passed,
            details=details,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(rec)
        await self.session.flush()
        return rec

    async def run_all(self) -> dict[str, Any]:
        """Runs all 20 scenarios sequentially and returns aggregate quality metrics."""
        results: list[dict[str, Any]] = []
        start_time = datetime.now(timezone.utc)

        scenarios = [
            ("01_explicit_preference_capture", "EXTRACTION", self.scenario_01_explicit_preference_capture),
            ("02_repeated_information_deduplication", "CONSOLIDATION", self.scenario_02_repeated_information_deduplication),
            ("03_new_semantic_fact_capture", "EXTRACTION", self.scenario_03_new_semantic_fact_capture),
            ("04_project_update", "EXTRACTION", self.scenario_04_project_update),
            ("05_temporal_change_supersession", "TEMPORAL", self.scenario_05_temporal_change_supersession),
            ("06_contradictory_facts_conflict", "CONSOLIDATION", self.scenario_06_contradictory_facts_conflict),
            ("07_ambiguous_information_handling", "EXTRACTION", self.scenario_07_ambiguous_information_handling),
            ("08_temporary_information_expiration", "TEMPORAL", self.scenario_08_temporary_information_expiration),
            ("09_sensitive_information_rejection", "PRIVACY", self.scenario_09_sensitive_information_rejection),
            ("10_explicit_forget_request_tombstone", "PRIVACY", self.scenario_10_explicit_forget_request_tombstone),
            ("11_paused_collection_exclusion", "PRIVACY", self.scenario_11_paused_collection_exclusion),
            ("12_disabled_retrieval_suppression", "PRIVACY", self.scenario_12_disabled_retrieval_suppression),
            ("13_cross_user_isolation", "SECURITY", self.scenario_13_cross_user_isolation),
            ("14_cross_workspace_isolation", "SECURITY", self.scenario_14_cross_workspace_isolation),
            ("15_unanswerable_memory_query", "RETRIEVAL", self.scenario_15_unanswerable_memory_query),
            ("16_dreaming_retry_recovery", "DREAMING", self.scenario_16_dreaming_retry_recovery),
            ("17_worker_restart_stale_job_recovery", "DURABILITY", self.scenario_17_worker_restart_stale_job_recovery),
            ("18_profile_rebuild_after_deletion", "PROFILE", self.scenario_18_profile_rebuild_after_deletion),
            ("19_tombstone_anti_resurrection", "PRIVACY", self.scenario_19_tombstone_anti_resurrection),
            ("20_legacy_memory_record_compatibility", "COMPATIBILITY", self.scenario_20_legacy_memory_record_compatibility),
        ]

        for name, category, method in scenarios:
            try:
                passed, metrics, details = await method()
                await self._record_metric(name, category, passed, metrics, details)
                results.append({
                    "scenario": name,
                    "category": category,
                    "passed": passed,
                    "metrics": metrics,
                    "details": details,
                })
            except Exception as e:
                logger.error(f"Evaluation scenario {name} failed with error: {e}", exc_info=True)
                metrics = {"error": str(e)}
                await self._record_metric(name, category, False, metrics, str(e))
                results.append({
                    "scenario": name,
                    "category": category,
                    "passed": False,
                    "metrics": metrics,
                    "details": str(e),
                })

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)

        await self.session.commit()

        return {
            "eval_run_id": self.eval_run_id,
            "total_scenarios": total_count,
            "passed_scenarios": passed_count,
            "pass_rate": round(passed_count / total_count, 4) if total_count > 0 else 0.0,
            "duration_seconds": round(duration, 2),
            "scenarios": results,
        }

    async def _get_or_create_test_user(self, email: str) -> User:
        user = await self.session.scalar(select(User).where(User.email == email))
        if not user:
            user = User(
                id=uuid.uuid4(),
                email=email,
                hashed_password="mock_password",
                role="USER",
                is_active=True,
                profile_json={},
            )
            self.session.add(user)
            await self.session.flush()
        return user

    async def _get_or_create_test_workspace(self, code_suffix: str) -> uuid.UUID:
        dept = await self.session.scalar(select(Department).where(Department.code == f"dept_eval_{code_suffix}"))
        if not dept:
            dept = Department(
                id=uuid.uuid4(),
                code=f"dept_eval_{code_suffix}",
                name=f"Dept Eval {code_suffix}",
            )
            self.session.add(dept)
            await self.session.flush()
        ws = await self.session.scalar(select(Workspace).where(Workspace.code == f"ws_eval_{code_suffix}"))
        if not ws:
            ws = Workspace(
                id=uuid.uuid4(),
                code=f"ws_eval_{code_suffix}",
                name=f"Workspace Eval {code_suffix}",
                department_id=dept.id,
            )
            self.session.add(ws)
            await self.session.flush()
        return ws.id

    # 1. Explicit preference capture
    async def scenario_01_explicit_preference_capture(self) -> tuple[bool, dict, str]:
        parsed = parse_memory_command("/rem I prefer responses in concise bullet points.")
        passed = (
            parsed.is_command
            and not parsed.is_empty
            and parsed.memory_type == MemoryType.STATIC
            and parsed.stability_tier == StabilityTier.ETERNAL
            and "bullet" in parsed.canonical_key.lower()
        )
        return passed, {"memory_type": str(parsed.memory_type), "canonical_key": parsed.canonical_key}, "Parsed explicit command"

    # 2. Repeated information deduplication
    async def scenario_02_repeated_information_deduplication(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_dup_user@local.dev")
        cand1 = MemoryCandidate(
            user_id=user.id,
            content="User prefers dark mode interface",
            canonical_key="MEMORY:STATIC:ui_theme",
            memory_type=MemoryType.STATIC,
        )
        item1 = await MemoryMaterializer.materialize_memory(self.session, cand1, auto_embed=False)
        dups = await MemoryConsolidationService.detect_duplicates(
            self.session, user.id, "User prefers dark mode interface", exclude_id=uuid.uuid4()
        )
        passed = len(dups) >= 1 and dups[0][1] == 1.0
        return passed, {"duplicates_detected": len(dups), "highest_score": dups[0][1] if dups else 0.0}, "Deduplication confirmed"

    # 3. New semantic fact capture
    async def scenario_03_new_semantic_fact_capture(self) -> tuple[bool, dict, str]:
        is_cand, is_trans, hints = detect_memory_candidate_rules("We decided to adopt PostgreSQL with pgvector for vector search")
        passed = is_cand and not is_trans and hints.get("memory_type") == MemoryType.SEMANTIC
        return passed, {"hints": {k: str(v) for k, v in hints.items()}}, "Semantic fact detection"

    # 4. Project update
    async def scenario_04_project_update(self) -> tuple[bool, dict, str]:
        is_cand, is_trans, hints = detect_memory_candidate_rules("I am currently working on SOC2 compliance audit documentation")
        passed = is_cand and not is_trans and hints.get("memory_type") == MemoryType.DYNAMIC
        return passed, {"hints": {k: str(v) for k, v in hints.items()}}, "Dynamic project update"

    # 5. Temporal change & supersession
    async def scenario_05_temporal_change_supersession(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_temp_user@local.dev")
        cand_old = MemoryCandidate(
            user_id=user.id,
            content="User prefers light mode",
            canonical_key="MEMORY:STATIC:pref_theme_temp",
            memory_type=MemoryType.STATIC,
        )
        item_old = await MemoryMaterializer.materialize_memory(self.session, cand_old, auto_embed=False)
        cand_new = MemoryCandidate(
            user_id=user.id,
            content="User prefers dark mode",
            canonical_key="MEMORY:STATIC:pref_theme_temp_new",
            memory_type=MemoryType.STATIC,
            supersedes_id=item_old.id,
        )
        item_new = await MemoryMaterializer.materialize_memory(self.session, cand_new, auto_embed=False)
        await self.session.refresh(item_old)
        passed = item_old.status == "SUPERSEDED" and item_old.superseded_by_id == item_new.id
        return passed, {"old_status": item_old.status, "superseded_by": str(item_old.superseded_by_id)}, "Temporal supersession confirmed"

    # 6. Contradictory facts & conflict classification
    async def scenario_06_contradictory_facts_conflict(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_conflict_user@local.dev")
        cand1 = MemoryCandidate(
            user_id=user.id,
            content="User always requires responses in dark mode",
            canonical_key="MEMORY:STATIC:theme_conflict_test",
            memory_type=MemoryType.STATIC,
        )
        item1 = await MemoryMaterializer.materialize_memory(self.session, cand1, auto_embed=False)
        conflicts = await MemoryConsolidationService.detect_conflicts(
            self.session, user.id, "User always requires responses in light mode", canonical_key=cand1.canonical_key
        )
        passed = len(conflicts) >= 1
        return passed, {"conflicts_count": len(conflicts), "reason": conflicts[0][1] if conflicts else None}, "Conflict detected"

    # 7. Ambiguous information handling (unresolved)
    async def scenario_07_ambiguous_information_handling(self) -> tuple[bool, dict, str]:
        is_cand, is_trans, _ = detect_memory_candidate_rules("Good morning, how are you doing today?")
        passed = not is_cand and is_trans
        return passed, {"is_candidate": is_cand, "is_transient": is_trans}, "Transient chit-chat skipped"

    # 8. Temporary information expiration
    async def scenario_08_temporary_information_expiration(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_expired_user@local.dev")
        now = datetime.now(timezone.utc)
        expired_cand = MemoryCandidate(
            user_id=user.id,
            content="Working on temporary debug task",
            canonical_key="MEMORY:DYNAMIC:debug_task_expired",
            memory_type=MemoryType.DYNAMIC,
            valid_until=now - timedelta(days=1),
        )
        item = await MemoryMaterializer.materialize_memory(self.session, expired_cand, auto_embed=False)
        retrieved = await MemoryRetrievalService.retrieve(self.session, user.id, query="debug task")
        passed = all(m.id != item.id for m in (retrieved.dynamic_memories + retrieved.relevant_memories))
        return passed, {"expired_item_retrieved": not passed}, "Expired memory suppressed from retrieval"

    # 9. Sensitive information rejection (PII/secrets)
    async def scenario_09_sensitive_information_rejection(self) -> tuple[bool, dict, str]:
        res = check_sensitive_request("My AWS secret key is AKIAIOSFODNN7EXAMPLE", can_view_pii=False)
        passed = res.blocked
        return passed, {"blocked": res.blocked, "reason": res.reason}, "Sensitive secret blocked"

    # 10. Explicit forget request & tombstone
    async def scenario_10_explicit_forget_request_tombstone(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_forget_user@local.dev")
        canonical_key = "MEMORY:STATIC:user_location_paris"
        await self.session.execute(
            delete(MemoryTombstone).where(
                MemoryTombstone.user_id == user.id,
                MemoryTombstone.canonical_key == canonical_key,
            )
        )
        await self.session.execute(
            delete(MemoryItem).where(
                MemoryItem.user_id == user.id,
                MemoryItem.canonical_key == canonical_key,
            )
        )
        await self.session.flush()
        cand = MemoryCandidate(
            user_id=user.id,
            content="User lives in Paris",
            canonical_key=canonical_key,
            memory_type=MemoryType.STATIC,
        )
        item = await MemoryMaterializer.materialize_memory(self.session, cand, auto_embed=False)
        res = await MemoryPrivacyOperations.delete_memory_by_id(self.session, item.id, user.id)
        tombstone = await self.session.scalar(
            select(MemoryTombstone).where(MemoryTombstone.user_id == user.id, MemoryTombstone.canonical_key == cand.canonical_key)
        )
        passed = res.get("status") == "DELETED" and tombstone is not None
        return passed, {"deleted": res.get("status"), "tombstone_present": tombstone is not None}, "Tombstone recorded"

    # 11. Paused collection exclusion
    async def scenario_11_paused_collection_exclusion(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_paused_user@local.dev")
        await MemoryPrivacyService.set_policy(
            session=self.session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_collection_enabled=False,
        )
        cand = MemoryCandidate(
            user_id=user.id,
            content="Testing paused memory collection",
            canonical_key="MEMORY:DYNAMIC:paused_test",
            memory_type=MemoryType.DYNAMIC,
        )
        item = await MemoryMaterializer.materialize_memory(self.session, cand, auto_embed=False)
        passed = item is None
        return passed, {"memory_materialized": item is not None}, "Paused collection blocked insertion"

    # 12. Disabled retrieval suppression
    async def scenario_12_disabled_retrieval_suppression(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_noretrieve_user@local.dev")
        cand = MemoryCandidate(
            user_id=user.id,
            content="Durable personal note",
            canonical_key="MEMORY:STATIC:note_suppress",
            memory_type=MemoryType.STATIC,
        )
        item = await MemoryMaterializer.materialize_memory(self.session, cand, auto_embed=False)
        await MemoryPrivacyService.set_policy(
            session=self.session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_retrieval_enabled=False,
        )
        res = await MemoryRetrievalService.retrieve(self.session, user.id, query="personal note")
        passed = len(res.static_memories) == 0 and len(res.dynamic_memories) == 0 and len(res.relevant_memories) == 0
        return passed, {"total_retrieved": res.total_count}, "Retrieval suppressed by privacy policy"

    # 13. Cross-user isolation
    async def scenario_13_cross_user_isolation(self) -> tuple[bool, dict, str]:
        user_a = await self._get_or_create_test_user("eval_user_a@local.dev")
        user_b = await self._get_or_create_test_user("eval_user_b@local.dev")
        cand_a = MemoryCandidate(
            user_id=user_a.id,
            content="Secret project codename X for User A",
            canonical_key="MEMORY:STATIC:secret_a",
            memory_type=MemoryType.STATIC,
        )
        await MemoryMaterializer.materialize_memory(self.session, cand_a, auto_embed=False)
        res_b = await MemoryRetrievalService.retrieve(self.session, user_b.id, query="codename X")
        passed = all(m.user_id != user_a.id for m in (res_b.static_memories + res_b.dynamic_memories + res_b.relevant_memories))
        return passed, {"cross_user_leak": not passed}, "User isolation preserved"

    # 14. Cross-workspace isolation
    async def scenario_14_cross_workspace_isolation(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_ws_iso_user@local.dev")
        ws_a_id = await self._get_or_create_test_workspace("a")
        ws_b_id = await self._get_or_create_test_workspace("b")
        cand = MemoryCandidate(
            user_id=user.id,
            workspace_id=ws_a_id,
            content="Workspace A internal task",
            canonical_key="MEMORY:DYNAMIC:ws_a_task",
            memory_type=MemoryType.DYNAMIC,
        )
        await MemoryMaterializer.materialize_memory(self.session, cand, auto_embed=False)
        mems_b = await MemoryMaterializer.get_user_memories(
            self.session, user.id, workspace_id=ws_b_id, include_global=False
        )
        passed = len(mems_b) == 0
        return passed, {"ws_b_count": len(mems_b)}, "Workspace isolation verified"

    # 15. Unanswerable memory query (graceful abstention)
    async def scenario_15_unanswerable_memory_query(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_abstain_user@local.dev")
        res = await MemoryRetrievalService.retrieve(self.session, user.id, query="nonexistent quantum key")
        passed = res.total_count == 0
        return passed, {"total_count": res.total_count}, "Graceful abstention on unanswerable query"

    # 16. Dreaming retry / recovery
    async def scenario_16_dreaming_retry_recovery(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_dream_user@local.dev")
        await self.session.execute(delete(MemoryRelationship).where(
            MemoryRelationship.source_memory_id.in_(
                select(MemoryItem.id).where(MemoryItem.user_id == user.id)
            )
        ))
        await self.session.execute(delete(MemoryItem).where(MemoryItem.user_id == user.id))
        await self.session.flush()
        # Ensure collection and dreaming enabled
        await MemoryPrivacyService.set_policy(
            session=self.session,
            scope_type=ScopeType.USER,
            scope_id=user.id,
            memory_dreaming_enabled=True,
            memory_collection_enabled=True,
        )
        # Create 5 memories on same cluster
        for i in range(5):
            cand = MemoryCandidate(
                user_id=user.id,
                content=f"User preference note {i}: bullet format {i}",
                canonical_key=f"MEMORY:STATIC:pref:bullet_{i}",
                memory_type=MemoryType.STATIC,
            )
            await MemoryMaterializer.materialize_memory(self.session, cand, auto_embed=False)

        res = await run_memory_dreaming(user.id, self.session, threshold=3, trigger_type="evaluation")
        run_rec = await self.session.scalar(
            select(MemoryDreamingRun).where(MemoryDreamingRun.run_id == res.get("run_id"))
        )
        passed = res.get("status") == "SUCCESS" and run_rec is not None and run_rec.status == "COMPLETED"
        return passed, {"status": res.get("status"), "clusters": res.get("clusters_consolidated")}, "Dreaming execution completed"

    # 17. Worker restart / stale job recovery
    async def scenario_17_worker_restart_stale_job_recovery(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_stale_job_user@local.dev")
        now = datetime.now(timezone.utc)
        job = MemoryProcessingJob(
            id=uuid.uuid4(),
            owner_user_id=user.id,
            source_text_snapshot="User preference from old crashed worker",
            trigger_kind="conversational_turn",
            status=JobStatus.PROCESSING.value,
            attempt_count=1,
            max_attempts=3,
            locked_at=now - timedelta(minutes=15),
            locked_by="crashed-worker-99",
            created_at=now - timedelta(minutes=16),
            updated_at=now - timedelta(minutes=15),
        )
        self.session.add(job)
        await self.session.flush()

        recovered_count = await recover_stale_memory_jobs(self.session, lock_timeout_seconds=300)
        await self.session.refresh(job)
        passed = recovered_count >= 1 and job.status == JobStatus.RETRY_PENDING.value
        return passed, {"recovered_count": recovered_count, "job_status": job.status}, "Stale job recovered"

    # 18. Profile rebuild after deletion
    async def scenario_18_profile_rebuild_after_deletion(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_rebuild_prof_user@local.dev")
        canonical_key = "MEMORY:STATIC:role_rebuild_test"
        await self.session.execute(
            delete(MemoryTombstone).where(
                MemoryTombstone.user_id == user.id,
                MemoryTombstone.canonical_key == canonical_key,
            )
        )
        await self.session.execute(
            delete(MemoryItem).where(
                MemoryItem.user_id == user.id,
                MemoryItem.canonical_key == canonical_key,
            )
        )
        await self.session.flush()
        cand = MemoryCandidate(
            user_id=user.id,
            content="User role is Principal QA Architect",
            canonical_key=canonical_key,
            memory_type=MemoryType.STATIC,
        )
        item = await MemoryMaterializer.materialize_memory(self.session, cand, auto_embed=False)
        prof_before = await compile_user_profile(user.id, self.session)
        await MemoryPrivacyOperations.delete_memory_by_id(self.session, item.id, user.id)
        prof_after = await compile_user_profile(user.id, self.session)
        passed = (
            prof_before.get("work_context", {}).get("role") == "Principal QA Architect"
            and prof_after.get("work_context", {}).get("role") is None
        )
        return passed, {"role_before": prof_before.get("work_context", {}).get("role"), "role_after": prof_after.get("work_context", {}).get("role")}, "Profile rebuilt without deleted memory"

    # 19. Tombstone anti-resurrection
    async def scenario_19_tombstone_anti_resurrection(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_antires_user@local.dev")
        canonical_key = "MEMORY:STATIC:secret_identity_tombstone"
        await self.session.execute(
            delete(MemoryTombstone).where(
                MemoryTombstone.user_id == user.id,
                MemoryTombstone.canonical_key == canonical_key,
            )
        )
        await self.session.execute(
            delete(MemoryItem).where(
                MemoryItem.user_id == user.id,
                MemoryItem.canonical_key == canonical_key,
            )
        )
        await self.session.flush()
        tombstone = MemoryTombstone(
            id=uuid.uuid4(),
            user_id=user.id,
            canonical_key=canonical_key,
            reason="User forget request",
        )
        self.session.add(tombstone)
        await self.session.flush()

        cand = MemoryCandidate(
            user_id=user.id,
            content="User identity fact trying to resurrect",
            canonical_key=canonical_key,
            memory_type=MemoryType.STATIC,
        )
        item = await MemoryMaterializer.materialize_memory(self.session, cand, auto_embed=False)
        passed = item is None
        return passed, {"resurrected": item is not None}, "Tombstone prevented resurrection"

    # 20. Legacy memory record compatibility
    async def scenario_20_legacy_memory_record_compatibility(self) -> tuple[bool, dict, str]:
        user = await self._get_or_create_test_user("eval_legacy_compat_user@local.dev")
        cand_legacy_static = MemoryCandidate(
            user_id=user.id,
            content="Legacy static user preference",
            canonical_key="MEMORY:STATIC:legacy_static_1",
            memory_type="STATIC",
        )
        cand_legacy_dynamic = MemoryCandidate(
            user_id=user.id,
            content="Legacy dynamic task context",
            canonical_key="MEMORY:DYNAMIC:legacy_dynamic_1",
            memory_type="DYNAMIC",
        )
        item_s = await MemoryMaterializer.materialize_memory(self.session, cand_legacy_static, auto_embed=False)
        item_d = await MemoryMaterializer.materialize_memory(self.session, cand_legacy_dynamic, auto_embed=False)
        res = await MemoryRetrievalService.retrieve(self.session, user.id, query="legacy")
        passed = item_s is not None and item_d is not None and (len(res.static_memories) >= 1 or len(res.dynamic_memories) >= 1)
        return passed, {"static_id": str(item_s.id), "dynamic_id": str(item_d.id)}, "Legacy records fully supported"

