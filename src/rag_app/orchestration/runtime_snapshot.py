"""
Phase 5E: Immutable Runtime Configuration Snapshot and Execution Consistency.

Captures effective provider and execution configuration at WorkflowRun creation time,
freezing LLM, retrieval, web research, embedding index, and node-level override parameters.
Guarantees execution determinism across subsequent task executions, preventing runtime
configuration drift when workspace settings change mid-run.

Strict Security Boundaries:
- NEVER contains raw secrets, API keys, passwords, bearer tokens, or encrypted blobs.
- Enforces active platform policy allowlists: a captured snapshot CANNOT bypass revoked provider permissions.
- Revoked credentials fail closed.
- Preserves full backward compatibility for legacy runs lacking a snapshot.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

SNAPSHOT_SCHEMA_VERSION = 1

# Vetted and functional providers permitted by application policy
FUNCTIONAL_LLM_PROVIDERS: set[str] = {"groq", "mock"}
FUNCTIONAL_WEB_PROVIDERS: set[str] = {"tinyfish", "firecrawl"}

# Patterns matching sensitive secrets that must NEVER be persisted in snapshots
SECRET_PATTERNS = re.compile(r"(key|token|secret|password|auth|bearer|cookie|credential_val)", re.IGNORECASE)
ALLOWED_NON_SECRET_KEYS = {"tokens_spent", "token_count", "credential_ref", "max_tokens", "fencing_token"}


class SnapshotImmutabilityError(RuntimeError):
    """Raised when an attempt is made to mutate or overwrite an existing frozen runtime snapshot."""
    pass


class SnapshotPolicyRevocationError(RuntimeError):
    """Raised when a provider specified in a snapshot has been revoked by platform security policy."""
    pass


def sanitize_snapshot_value(data: Any) -> Any:
    """
    Recursively strips any credentials, tokens, or secret patterns from snapshot dictionaries.
    """
    if isinstance(data, dict):
        clean = {}
        for k, v in data.items():
            k_str = str(k).lower()
            if SECRET_PATTERNS.search(k_str) and k_str not in ALLOWED_NON_SECRET_KEYS:
                # Omit secret values entirely
                continue
            clean[k] = sanitize_snapshot_value(v)
        return clean
    elif isinstance(data, list):
        return [sanitize_snapshot_value(item) for item in data]
    return data


async def capture_runtime_snapshot(
    session: Optional[AsyncSession],
    workspace_id: Optional[uuid.UUID],
    workflow_id: Optional[uuid.UUID],
    workflow_version_id: Optional[uuid.UUID],
    steps_spec: Optional[list[Any]] = None,
    trigger_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Constructs an immutable configuration snapshot capturing the exact resolved
    AI, RAG, research, and node parameters at run initialization.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Default platform baseline
    llm_provider = "mock"
    llm_model = "mock-deterministic-v1"
    embedding_provider = "sentence_transformers"
    embedding_model = "all-MiniLM-L6-v2 (384 dim)"
    reranker_provider = "cross_encoder"
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    web_provider = "tinyfish"

    cfg: dict[str, Any] = {
        "temperature": 0.2,
        "max_tokens": 2048,
        "top_k": 8,
        "reranker_top_k": 5,
        "vector_search_enabled": True,
        "fts_enabled": True,
        "rrf_enabled": True,
        "reranker_enabled": True,
    }

    # 1. Resolve Workspace Provider Profile defaults if session and workspace_id provided
    if session is not None and workspace_id is not None:
        try:
            from rag_app.db.models import WorkspaceProviderProfile

            stmt = select(WorkspaceProviderProfile).where(
                WorkspaceProviderProfile.workspace_id == workspace_id,
                WorkspaceProviderProfile.is_active.is_(True),
            )
            profile = await session.scalar(stmt)
            if profile:
                if profile.llm_provider:
                    llm_provider = profile.llm_provider.strip().lower()
                if profile.llm_model:
                    llm_model = profile.llm_model.strip()
                if profile.embedding_provider:
                    embedding_provider = profile.embedding_provider.strip().lower()
                if profile.embedding_model:
                    embedding_model = profile.embedding_model.strip()
                if profile.reranker_provider:
                    reranker_provider = profile.reranker_provider.strip().lower()
                if profile.reranker_model:
                    reranker_model = profile.reranker_model.strip()
                if profile.web_provider:
                    web_provider = profile.web_provider.strip().lower()

                if profile.configuration and isinstance(profile.configuration, dict):
                    for k, v in profile.configuration.items():
                        if k in cfg:
                            cfg[k] = v
        except Exception as e:
            logger.warning(f"Could not load WorkspaceProviderProfile during snapshot capture: {e}")

    # 2. Extract validated node-level configurations
    node_configs: dict[str, Any] = {}
    if steps_spec and isinstance(steps_spec, list):
        for step in steps_spec:
            if isinstance(step, dict):
                node_id = step.get("id")
                inputs = step.get("inputs", {})
            elif hasattr(step, "id") and hasattr(step, "inputs"):
                node_id = getattr(step, "id")
                inputs = getattr(step, "inputs", {})
            else:
                continue

            if node_id and isinstance(inputs, dict):
                node_configs[node_id] = sanitize_snapshot_value(inputs)

    # 3. Assemble complete snapshot dictionary
    is_test_mode = False
    if isinstance(trigger_payload, dict):
        is_test_mode = bool(trigger_payload.get("is_test", False)) or trigger_payload.get("execution_mode") == "TEST"

    snapshot = {
        "snapshot_version": SNAPSHOT_SCHEMA_VERSION,
        "captured_at": now_iso,
        "workspace_id": str(workspace_id) if workspace_id else None,
        "workflow_id": str(workflow_id) if workflow_id else None,
        "workflow_version_id": str(workflow_version_id) if workflow_version_id else None,
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
            "temperature": float(cfg.get("temperature", 0.2)),
            "max_tokens": int(cfg.get("max_tokens", 2048)),
        },
        "retrieval": {
            "top_k": int(cfg.get("top_k", 8)),
            "reranker_top_k": int(cfg.get("reranker_top_k", 5)),
            "vector_search_enabled": bool(cfg.get("vector_search_enabled", True)),
            "fts_enabled": bool(cfg.get("fts_enabled", True)),
            "rrf_enabled": bool(cfg.get("rrf_enabled", True)),
            "reranker_enabled": bool(cfg.get("reranker_enabled", True)),
            "reranker_provider": reranker_provider,
            "reranker_model": reranker_model,
        },
        "research": {
            "web_provider": web_provider,
        },
        "knowledge_index": {
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimension": 384,
        },
        "node_configurations": node_configs,
        "policy_metadata": {
            "configuration_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "is_test": is_test_mode,
        },
    }

    # Final recursive pass guaranteeing strict secret sanitization
    return sanitize_snapshot_value(snapshot)


TERMINAL_WORKFLOW_STATUSES: set[str] = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "REJECTED",
    "EXPIRED",
}


def assert_snapshot_immutability(
    existing_snapshot: Optional[dict[str, Any]],
    updated_snapshot: Optional[dict[str, Any]],
) -> None:
    """
    Enforces application-level snapshot immutability.
    Prevents altering or clearing an already established snapshot.
    """
    if existing_snapshot is not None:
        if updated_snapshot is None or existing_snapshot != updated_snapshot:
            raise SnapshotImmutabilityError(
                "Cannot modify or clear an existing WorkflowRun provider_snapshot. Snapshots are immutable."
            )


async def ensure_run_runtime_snapshot(
    run: Any,
    session: Optional[AsyncSession],
) -> Optional[dict[str, Any]]:
    """
    Safe One-Time Compatibility Policy for Legacy WorkflowRuns (Phase 5E.1):
    1. If run.provider_snapshot already exists, return it immediately.
    2. If run is in a TERMINAL status (COMPLETED, FAILED, CANCELLED, REJECTED, EXPIRED):
       Preserve terminal historical runs without mutation; return None.
    3. If run is NON-TERMINAL (RUNNING, PENDING, PLANNING, AWAITING_APPROVAL, EXECUTING)
       and provider_snapshot is None:
       Capture effective workspace snapshot once, persist it to run.provider_snapshot,
       and freeze it permanently so subsequent nodes never repeatedly resolve mutable defaults.
    """
    existing = getattr(run, "provider_snapshot", None)
    if existing is not None:
        return existing

    status = str(getattr(run, "status", "")).upper()
    if status in TERMINAL_WORKFLOW_STATUSES:
        # Preserve historical terminal runs as-is without backfill or mutation
        return None

    # Non-terminal legacy run: apply safe one-time initialization
    workspace_id = getattr(run, "workspace_id", None)
    workflow_id = getattr(run, "workflow_id", None)
    workflow_version_id = getattr(run, "workflow_version_id", None)

    snapshot = await capture_runtime_snapshot(
        session=session,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
    )

    try:
        run.provider_snapshot = snapshot
        if session is not None:
            session.add(run)
            await session.flush()
        logger.info(
            "Phase 5E.1: Initialized one-time compatibility runtime snapshot for legacy non-terminal run %s",
            getattr(run, "id", "unknown"),
        )
    except Exception as exc:
        logger.warning(
            "Phase 5E.1: Failed to persist one-time compatibility snapshot on run %s: %s",
            getattr(run, "id", "unknown"),
            exc,
        )

    return snapshot


def sanitize_snapshot_for_api(
    snapshot: Optional[dict[str, Any]],
    user_role: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Sanitizes snapshot data before exposing it over public/client API responses (Phase 5E.1).
    Guarantees:
    1. Zero raw secrets, keys, tokens, or passwords.
    2. Internal credential references (e.g. 'env:GROQ_API_KEY') are masked as
       '[server-managed]' unless the requesting user is an 'ADMIN'.
    3. Safe configuration fields (model, temperature, top_k, etc.) remain inspectable.
    """
    if snapshot is None:
        return None

    import copy
    clean = copy.deepcopy(snapshot)

    # First pass: strip any matching secret patterns
    clean = sanitize_snapshot_value(clean)

    # Second pass: mask internal credential references for non-admin viewers
    is_admin = str(user_role).upper() == "ADMIN" if user_role else False

    def _mask_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                k_lower = str(k).lower()
                if "credential_ref" in k_lower and v is not None:
                    if not is_admin:
                        obj[k] = "[server-managed]"
                else:
                    obj[k] = _mask_refs(v)
            return obj
        elif isinstance(obj, list):
            return [_mask_refs(item) for item in obj]
        return obj

    return _mask_refs(clean)


async def resolve_workflow_llm(
    run: Any,
    session: Optional[AsyncSession],
    node_inputs: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Resolves the LLM provider for a workflow task:
    1. If run has an immutable provider_snapshot, respects the frozen provider and model.
    2. Enforces current platform policy allowlist (fails closed if provider was revoked).
    3. Applies valid node-level overrides (model_tier, temperature, max_tokens).
    4. Evaluates safe one-time compatibility snapshot for legacy non-terminal runs.
    5. Falls back safely for terminal historical runs without a snapshot.
    """
    node_inputs = node_inputs or {}
    snapshot = getattr(run, "provider_snapshot", None)

    # Phase 5E.1: If snapshot is missing, apply one-time compatibility policy for active legacy runs
    if snapshot is None and session is not None and getattr(run, "workspace_id", None):
        snapshot = await ensure_run_runtime_snapshot(run, session)

    if snapshot and isinstance(snapshot, dict) and "llm" in snapshot:
        llm_spec = snapshot["llm"]
        provider = str(llm_spec.get("provider", "mock")).lower()
        model = str(llm_spec.get("model", "mock-deterministic-v1"))

        # SECURITY GATE: Enforce active platform policy allowlist.
        # A historical snapshot CANNOT use a provider revoked from the platform allowlist.
        if provider not in FUNCTIONAL_LLM_PROVIDERS:
            raise SnapshotPolicyRevocationError(
                f"LLM provider '{provider}' from run snapshot is not permitted by current platform policy."
            )

        # Node-level overrides
        temp = node_inputs.get("temperature")
        if temp is None:
            temp = float(llm_spec.get("temperature", 0.2))
        else:
            temp = max(0.0, min(1.0, float(temp)))

        # Model tier mapping for Groq
        tier = node_inputs.get("model_tier")
        if tier and provider == "groq":
            tier_upper = str(tier).upper().strip()
            tier_map = {
                "FAST": "llama-3.1-8b-instant",
                "BALANCED": "llama-3.3-70b-versatile",
                "REASONING": "llama-3.3-70b-versatile",
            }
            if tier_upper in tier_map:
                model = tier_map[tier_upper]

        # Instantiate provider adapter
        if provider == "mock":
            from rag_app.providers.adapters.llm.mock import MockLLMAdapter
            return MockLLMAdapter(model=model, temperature=temp)
        elif provider == "groq":
            try:
                from rag_app.providers.adapters.llm.groq import GroqLLMAdapter
                return GroqLLMAdapter(model=model, temperature=temp)
            except Exception as e:
                logger.warning(f"Failed to instantiate Groq adapter for snapshot model '{model}': {e}. Falling back to mock.")
                from rag_app.providers.adapters.llm.mock import MockLLMAdapter
                return MockLLMAdapter(model=model, temperature=temp)
        else:
            from rag_app.providers.registry import get_provider
            return get_provider("llm", provider, model=model, temperature=temp)

    # BACKWARD COMPATIBILITY: Legacy runs without provider_snapshot
    if session is not None and getattr(run, "workspace_id", None):
        from rag_app.providers.resolver import resolve_llm_for_workspace
        return await resolve_llm_for_workspace(run.workspace_id, session)

    from rag_app.providers.adapters.llm.mock import MockLLMAdapter
    return MockLLMAdapter()


def resolve_workflow_retrieval_config(
    run: Any,
    node_inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Extracts frozen retrieval parameters from the run snapshot:
    vector_search_enabled, fts_enabled, rrf_enabled, reranker_enabled, top_k, reranker_top_k.
    """
    node_inputs = node_inputs or {}
    snapshot = getattr(run, "provider_snapshot", None)

    if snapshot and isinstance(snapshot, dict) and "retrieval" in snapshot:
        ret_spec = snapshot["retrieval"]
        rerank_node_val = node_inputs.get("rerank")
        rerank_effective = rerank_node_val if rerank_node_val is not None else ret_spec.get("reranker_enabled", True)

        return {
            "top_k": node_inputs.get("limit") or ret_spec.get("top_k", 8),
            "reranker_top_k": ret_spec.get("reranker_top_k", 5),
            "vector_search_enabled": ret_spec.get("vector_search_enabled", True),
            "fts_enabled": ret_spec.get("fts_enabled", True),
            "rrf_enabled": ret_spec.get("rrf_enabled", True),
            "reranker_enabled": rerank_effective,
        }

    return {}


def resolve_workflow_web_provider(
    run: Any,
    node_inputs: Optional[dict[str, Any]] = None,
) -> str:
    """
    Resolves the web research provider:
    1. Node override (if valid and permitted)
    2. Snapshot frozen web provider
    3. Default to 'tinyfish'
    """
    node_inputs = node_inputs or {}
    override = node_inputs.get("search_provider")
    if override and str(override).lower() in FUNCTIONAL_WEB_PROVIDERS:
        return str(override).lower()

    snapshot = getattr(run, "provider_snapshot", None)
    if snapshot and isinstance(snapshot, dict) and "research" in snapshot:
        res_provider = snapshot["research"].get("web_provider")
        if res_provider and str(res_provider).lower() in FUNCTIONAL_WEB_PROVIDERS:
            return str(res_provider).lower()

    return "tinyfish"

