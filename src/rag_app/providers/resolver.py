"""
Workspace-aware provider resolver.

Resolves RAG infrastructure providers for a given Workspace by:
1. Loading the Workspace's active WorkspaceProviderProfile (if any).
2. For each capability:
   - If the profile specifies a provider name → fetch it from the global registry.
   - Otherwise → fall back to the platform-wide default from the registry.

Business services consume capability interfaces (LLMProvider, EmbeddingProvider, …)
and MUST NOT branch on provider names internally.

Usage
-----
    resolved = await resolve_providers_for_workspace(workspace_id, session)
    llm = resolved["llm"]        # LLMProvider instance
    emb = resolved["embedding"]  # EmbeddingProvider instance
    ...

Or for a single capability:
    llm = await resolve_llm_for_workspace(workspace_id, session)
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.providers.registry import get_provider

logger = logging.getLogger(__name__)

# Supported capabilities and their profile field name
_CAPABILITY_FIELD_MAP: dict[str, str] = {
    "llm": "llm_provider",
    "embedding": "embedding_provider",
    "document_reader": "document_reader_provider",
    "reranker": "reranker_provider",
    "ocr": "ocr_provider",
    "web": "web_provider",
}


async def _load_profile(
    workspace_id: uuid.UUID,
    session: AsyncSession,
) -> Any | None:
    """
    Load the active WorkspaceProviderProfile for a workspace.
    Returns None if no profile exists.
    Avoids importing models at module level to prevent circular imports.
    """
    from rag_app.db.models import WorkspaceProviderProfile

    result = await session.scalar(
        select(WorkspaceProviderProfile).where(
            WorkspaceProviderProfile.workspace_id == workspace_id,
            WorkspaceProviderProfile.is_active.is_(True),
        )
    )
    return result


async def resolve_providers_for_workspace(
    workspace_id: uuid.UUID | None,
    session: AsyncSession,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Resolve all supported capability providers for a workspace.

    Resolution Order:
    1. Per-request Experiment Override (if present in overrides dict for capability)
    2. Active WorkspaceProviderProfile default for the workspace (if present)
    3. Platform-wide default registered provider

    Experiment overrides NEVER modify stored Workspace defaults.
    """
    profile = None
    if workspace_id is not None:
        profile = await _load_profile(workspace_id, session)

    resolved: dict[str, Any] = {}
    overrides = overrides or {}

    for capability, field_name in _CAPABILITY_FIELD_MAP.items():
        provider_name: str | None = None
        source = "platform_default"

        # 1. Experiment Override Priority
        if overrides.get(capability):
            provider_name = overrides[capability]
            source = "experiment_override"
        # 2. Workspace Default Profile Priority
        elif profile is not None:
            provider_name = getattr(profile, field_name, None)
            if provider_name:
                source = "workspace_default"

        if provider_name:
            try:
                resolved[capability] = get_provider(capability, provider_name)
                logger.debug(
                    "workspace_provider_resolved",
                    extra={
                        "workspace_id": str(workspace_id),
                        "capability": capability,
                        "provider": provider_name,
                        "source": source,
                    },
                )
            except Exception as exc:
                # If the configured provider is not available, fall back to default
                logger.warning(
                    "workspace_provider_fallback",
                    extra={
                        "workspace_id": str(workspace_id),
                        "capability": capability,
                        "configured_provider": provider_name,
                        "source": source,
                        "reason": str(exc),
                    },
                )
                resolved[capability] = get_provider(capability)
        else:
            resolved[capability] = get_provider(capability)
            logger.debug(
                "workspace_provider_resolved",
                extra={
                    "workspace_id": str(workspace_id),
                    "capability": capability,
                    "source": "platform_default",
                },
            )

    return resolved


async def resolve_llm_for_workspace(
    workspace_id: uuid.UUID | None,
    session: AsyncSession,
    override_provider: str | None = None,
) -> Any:
    """
    Convenience helper: resolve only the LLM provider for a workspace,
    supporting an optional runtime experiment override.
    """
    overrides = {"llm": override_provider} if override_provider else None
    resolved = await resolve_providers_for_workspace(workspace_id, session, overrides=overrides)
    return resolved["llm"]


async def resolve_embedding_for_workspace(
    workspace_id: uuid.UUID | None,
    session: AsyncSession,
    override_provider: str | None = None,
) -> Any:
    """
    Convenience helper: resolve only the Embedding provider for a workspace,
    supporting an optional runtime experiment override.
    """
    overrides = {"embedding": override_provider} if override_provider else None
    resolved = await resolve_providers_for_workspace(workspace_id, session, overrides=overrides)
    return resolved["embedding"]


async def resolve_document_reader_for_workspace(
    workspace_id: uuid.UUID | None,
    session: AsyncSession,
    override_provider: str | None = None,
) -> Any:
    """
    Convenience helper: resolve only the Document Reader provider for a workspace,
    supporting an optional runtime experiment override.
    """
    overrides = {"document_reader": override_provider} if override_provider else None
    resolved = await resolve_providers_for_workspace(workspace_id, session, overrides=overrides)
    return resolved["document_reader"]


async def resolve_reranker_for_workspace(
    workspace_id: uuid.UUID | None,
    session: AsyncSession,
    override_provider: str | None = None,
) -> Any:
    """
    Convenience helper: resolve only the Reranker provider for a workspace,
    supporting an optional runtime experiment override.
    """
    overrides = {"reranker": override_provider} if override_provider else None
    resolved = await resolve_providers_for_workspace(workspace_id, session, overrides=overrides)
    return resolved["reranker"]


async def resolve_ocr_for_workspace(
    workspace_id: uuid.UUID | None,
    session: AsyncSession,
    override_provider: str | None = None,
) -> Any:
    """
    Convenience helper: resolve only the OCR provider for a workspace,
    supporting an optional runtime experiment override.
    """
    overrides = {"ocr": override_provider} if override_provider else None
    resolved = await resolve_providers_for_workspace(workspace_id, session, overrides=overrides)
    return resolved["ocr"]


async def resolve_web_for_workspace(
    workspace_id: uuid.UUID | None,
    session: AsyncSession,
    override_provider: str | None = None,
) -> Any:
    """
    Convenience helper: resolve only the Web provider for a workspace,
    supporting an optional runtime experiment override.
    """
    overrides = {"web": override_provider} if override_provider else None
    resolved = await resolve_providers_for_workspace(workspace_id, session, overrides=overrides)
    return resolved["web"]


async def get_provider_registry_summary() -> dict[str, list[str]]:
    """
    Return a summary of all providers registered in the platform registry,
    grouped by capability. Used by the frontend to populate dropdowns.
    """
    from rag_app.providers.registry import provider_registry
    capabilities = list(_CAPABILITY_FIELD_MAP.keys())
    return {cap: provider_registry.list_providers(cap) for cap in capabilities}
