"""
FastMCP package for Production RAG.
Exposes the FastMCP server, gateway, context, policies, and schemas.
"""

from rag_app.mcp.context import MCPRequestContext, resolve_mcp_request_context
from rag_app.mcp.diagnostics import run_system_diagnostics
from rag_app.mcp.facade import MCPFacade
from rag_app.mcp.gateway import MCPGateway, MCPGatewayError, mcp_gateway
from rag_app.mcp.policies import TOOL_POLICIES, ToolPolicy
from rag_app.mcp.server import get_mcp_asgi_app, get_mcp_server, mcp, search_tools

__all__ = [
    "mcp",
    "get_mcp_server",
    "get_mcp_asgi_app",
    "search_tools",
    "mcp_gateway",
    "MCPGateway",
    "MCPGatewayError",
    "MCPRequestContext",
    "resolve_mcp_request_context",
    "TOOL_POLICIES",
    "ToolPolicy",
    "MCPFacade",
    "run_system_diagnostics",
]
