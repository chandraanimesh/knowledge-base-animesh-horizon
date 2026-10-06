"""
Tool Discovery and Search for FastMCP Interface.
Enables natural language tool discovery across the MCP tool surface while strictly
respecting role-aware visibility and authorization boundaries.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastmcp.server.transforms.search.bm25 import _BM25Index
from rag_app.db.models import User
from rag_app.mcp.context import MCPRequestContext
from rag_app.mcp.gateway import mcp_gateway
from rag_app.mcp.policies import TOOL_POLICIES, ToolPolicy

logger = logging.getLogger(__name__)


async def handle_search_tools(
    query: str,
    domain: Optional[str] = None,
    limit: int = 5,
    user: Optional[User] = None,
    context: Optional[MCPRequestContext] = None,
) -> dict[str, Any]:
    """
    Search and discover relevant MCP tools using FastMCP BM25 relevance ranking.
    Filters candidate tools by caller visibility (ADMIN vs standard USER).
    """
    is_admin = bool(user and user.role == "ADMIN")

    # 1. Filter candidate tools by caller visibility
    candidate_policies: list[ToolPolicy] = []
    for policy in TOOL_POLICIES.values():
        # If user is not admin, hide admin-only tools
        if not is_admin and policy.admin_only:
            continue
        # If domain filter is specified, filter by domain (case-insensitive substring)
        if domain and domain.strip().lower() not in policy.domain.lower():
            continue
        candidate_policies.append(policy)

    if not candidate_policies:
        return {
            "query": query,
            "domain_filter": domain,
            "total_matches": 0,
            "tools": [],
        }

    # 2. Build BM25 search corpus from candidate tool metadata
    documents = [
        f"{p.name} {p.domain} {p.description} {' '.join(p.tags)}"
        for p in candidate_policies
    ]

    bm25 = _BM25Index()
    bm25.build(documents)

    # 3. Query index
    matched_indices = bm25.query(query, max(1, min(limit, 20)))

    # If BM25 query yields no direct term matches, fall back to keyword / tag matching
    if not matched_indices:
        q_lower = query.lower()
        fallback_matches = []
        for i, p in enumerate(candidate_policies):
            if (
                q_lower in p.name.lower()
                or q_lower in p.domain.lower()
                or q_lower in p.description.lower()
                or any(q_lower in t.lower() for t in p.tags)
            ):
                fallback_matches.append(i)
                if len(fallback_matches) >= limit:
                    break
        matched_indices = fallback_matches

    # 4. Format search results
    matched_tools = []
    for idx in matched_indices:
        p = candidate_policies[idx]
        matched_tools.append({
            "name": p.name,
            "domain": p.domain,
            "description": p.description,
            "scope": p.scope,
            "visibility": p.visibility,
            "mutability": p.mutability,
            "tags": p.tags,
            "admin_only": p.admin_only,
        })

    return {
        "query": query,
        "domain_filter": domain,
        "total_matches": len(matched_tools),
        "tools": matched_tools,
    }


async def search_tools(
    query: str,
    domain: Optional[str] = None,
    limit: int = 5,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    """
    Search and discover available MCP tools using natural language or domain filtering.
    Discovered tools are filtered according to your authentication level and visibility.
    """
    return await mcp_gateway.execute(
        tool_name="search_tools",
        auth_token=auth_token,
        handler=handle_search_tools,
        query=query,
        domain=domain,
        limit=limit,
    )
