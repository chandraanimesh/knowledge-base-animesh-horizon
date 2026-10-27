"""
Diagnostics domain tools for FastMCP.
Exposes:
- run_diagnostics

Safe platform health inspection.
"""

from __future__ import annotations

from typing import Any, Optional

from rag_app.mcp.context import MCPRequestContext
from rag_app.mcp.diagnostics import get_public_health_probe, run_system_diagnostics
from rag_app.mcp.gateway import mcp_gateway


async def handle_run_diagnostics(
    include_rag_smoketest: bool = False,
    context: Optional[MCPRequestContext] = None,
) -> dict[str, Any]:
    """
    Diagnostics handler.
    If caller is unauthenticated (no verified context), exposes only the minimal public probe.
    If caller is authenticated with valid scope, executes full system diagnostics.
    """
    if context is None:
        return get_public_health_probe()
    res = await run_system_diagnostics(include_rag_smoketest=include_rag_smoketest)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def run_diagnostics(
    include_rag_smoketest: bool = False,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Run safe, non-destructive platform health checks (minimal probe if unauthenticated, detailed report if authorized)."""
    return await mcp_gateway.execute(
        "run_diagnostics",
        handle_run_diagnostics,
        include_rag_smoketest=include_rag_smoketest,
        auth_token=auth_token,
    )
