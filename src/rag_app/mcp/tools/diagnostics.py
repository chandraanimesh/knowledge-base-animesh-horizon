"""
Diagnostics domain tools for FastMCP.
Exposes:
- run_diagnostics

Safe platform health inspection.
"""

from __future__ import annotations

from typing import Any, Optional

from rag_app.mcp.diagnostics import run_system_diagnostics
from rag_app.mcp.gateway import mcp_gateway


async def handle_run_diagnostics(
    include_rag_smoketest: bool = False,
) -> dict[str, Any]:
    res = await run_system_diagnostics(include_rag_smoketest=include_rag_smoketest)
    return res.model_dump() if hasattr(res, "model_dump") else dict(res)


async def run_diagnostics(
    include_rag_smoketest: bool = False,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """Run safe, non-destructive platform health checks."""
    return await mcp_gateway.execute(
        "run_diagnostics",
        handle_run_diagnostics,
        include_rag_smoketest=include_rag_smoketest,
        auth_token=auth_token,
    )
