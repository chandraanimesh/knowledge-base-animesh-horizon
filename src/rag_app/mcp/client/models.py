"""
Data models, configurations, and exception hierarchy for the Generic Outbound MCP Client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MCPTransportType(str, Enum):
    """Supported MCP client transport types."""
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"


class MCPClientError(Exception):
    """Base exception for all outbound MCP client operations."""


class MCPSSRFError(MCPClientError):
    """Raised when an endpoint URL violates SSRF guard policies."""


class MCPConnectionError(MCPClientError):
    """Raised when the client cannot establish or maintain a connection to the remote MCP server."""


class MCPAuthenticationError(MCPClientError):
    """Raised when the remote MCP server rejects authentication or returns HTTP 401/403."""


class MCPTimeoutError(MCPClientError):
    """Raised when an MCP operation exceeds the configured timeout."""


class MCPProtocolError(MCPClientError):
    """Raised when a remote MCP server returns malformed protocol messages or JSON-RPC errors."""


class MCPPayloadTooLargeError(MCPClientError):
    """Raised when a response payload exceeds the configured byte size limit."""


class MCPToolExecutionError(MCPClientError):
    """Raised when an unrecoverable tool execution failure occurs."""


@dataclass
class MCPClientConfig:
    """Configuration for GenericMCPClient connection and bounds."""
    endpoint_url: str
    transport: MCPTransportType = MCPTransportType.STREAMABLE_HTTP
    auth_headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0
    max_response_bytes: int = 1_048_576  # 1 MB bounded limit
    max_pagination_pages: int = 10
    page_size: int = 50
    allow_private_hosts: bool = False  # SSRF protection: False by default; True only for dev/testing loopback


@dataclass
class MCPServerInfo:
    """Metadata regarding the connected remote MCP server."""
    name: str
    version: Optional[str] = None
    protocol_version: Optional[str] = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    instructions: Optional[str] = None


@dataclass
class MCPToolDefinition:
    """Preserved schema information for a discovered remote MCP tool."""
    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: Optional[dict[str, Any]] = None


@dataclass
class MCPToolResult:
    """Normalized application-level result from an MCP tool invocation."""
    name: str
    content: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    structured_content: Optional[Any] = None
    is_error: bool = False
    error_message: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
