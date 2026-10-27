"""
Unit and integration tests for Generic Outbound MCP Client Engine (Milestone 6B).

Covers:
1. SSRF prevention (cloud metadata, private IP rejection, scheme validation).
2. Credential sanitization (header redaction for logs).
3. MCP initialization and capability negotiation with official MCP SDK.
4. Tool discovery and input schema preservation.
5. Paginated tool listing handling.
6. Successful tool invocation and result normalization.
7. Tool error normalization (isError=True).
8. Protocol error and unknown tool handling.
9. Invalid argument validation.
10. Response-size bounds (MCPPayloadTooLargeError).
11. Bounded timeout handling (MCPTimeoutError).
12. Authentication failure handling (HTTP 401/403 -> MCPAuthenticationError).
13. Deterministic async context manager cleanup.
"""

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from fastmcp import FastMCP
import httpx2
import pytest
import pytest_asyncio

from rag_app.mcp.client import (
    GenericMCPClient,
    MCPAuthenticationError,
    MCPClientConfig,
    MCPClientError,
    MCPConnectionError,
    MCPPayloadTooLargeError,
    MCPProtocolError,
    MCPServerInfo,
    MCPSSRFError,
    MCPTimeoutError,
    MCPToolDefinition,
    MCPToolResult,
    MCPTransportType,
    sanitize_headers_for_logging,
    validate_mcp_url,
)


# -----------------------------------------------------------------------------
# 1. SSRF & Security Validation Tests
# -----------------------------------------------------------------------------

def test_ssrf_rejects_cloud_metadata():
    """Verify that cloud metadata hosts are strictly rejected."""
    for host in [
        "http://169.254.169.254/mcp",
        "https://169.254.169.254/mcp",
        "http://169.254.170.2/mcp",
        "http://metadata.google.internal/mcp",
        "http://metadata.azure.internal/mcp",
    ]:
        with pytest.raises(MCPSSRFError) as exc_info:
            validate_mcp_url(host, allow_private_hosts=True)
        assert "cloud metadata" in str(exc_info.value).lower()


def test_ssrf_rejects_private_hosts_by_default():
    """Verify that private and loopback networks are blocked by default."""
    private_endpoints = [
        "http://127.0.0.1:8000/mcp",
        "http://localhost:3000/mcp",
        "http://10.0.0.1/mcp",
        "http://192.168.1.1/mcp",
        "http://172.16.0.1/mcp",
        "http://app.local/mcp",
        "http://service.internal/mcp",
    ]
    for ep in private_endpoints:
        with pytest.raises(MCPSSRFError) as exc_info:
            validate_mcp_url(ep, allow_private_hosts=False)
        assert "forbidden" in str(exc_info.value).lower() or "prohibited" in str(exc_info.value).lower()


def test_ssrf_permits_private_hosts_when_explicitly_configured():
    """Verify that dev/testing loopback endpoints pass when allow_private_hosts=True."""
    # Should not raise
    validate_mcp_url("http://127.0.0.1:8000/mcp", allow_private_hosts=True)
    validate_mcp_url("http://localhost:3000/mcp", allow_private_hosts=True)


def test_ssrf_rejects_unsafe_schemes():
    """Verify that only http and https schemes are permitted."""
    for bad_url in [
        "file:///etc/passwd",
        "ftp://example.com/mcp",
        "gopher://example.com/mcp",
        "ws://example.com/mcp",
        "",
    ]:
        with pytest.raises(MCPSSRFError):
            validate_mcp_url(bad_url, allow_private_hosts=True)


def test_sanitize_headers_for_logging():
    """Verify that credentials and tokens are redacted from logging headers."""
    headers = {
        "Authorization": "Bearer super_secret_token_12345",
        "X-API-Key": "my-private-api-key",
        "Cookie": "session=xyz123",
        "Content-Type": "application/json",
        "User-Agent": "ProductionRAG/1.0",
    }
    sanitized = sanitize_headers_for_logging(headers)
    assert sanitized["Authorization"] == "[REDACTED]"
    assert sanitized["X-API-Key"] == "[REDACTED]"
    assert sanitized["Cookie"] == "[REDACTED]"
    assert sanitized["Content-Type"] == "application/json"
    assert sanitized["User-Agent"] == "ProductionRAG/1.0"
    # Ensure original dictionary is not mutated
    assert "super_secret" in headers["Authorization"]


# -----------------------------------------------------------------------------
# 2. In-Memory Mock Server Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_fastmcp_app():
    """Creates a FastMCP server application with various tool behaviors."""
    server = FastMCP(
        name="MockRemoteServer",
        instructions="Operational instructions for MockRemoteServer",
    )

    @server.tool(name="echo", description="Echoes back the message.")
    def echo(message: str) -> str:
        return f"echo: {message}"

    @server.tool(name="calculate", description="Performs basic arithmetic.")
    def calculate(a: int, b: int, op: str = "add") -> int:
        if op == "add":
            return a + b
        elif op == "multiply":
            return a * b
        raise ValueError(f"Unsupported op: {op}")

    @server.tool(name="faulty_tool", description="Always fails.")
    def faulty_tool() -> str:
        raise RuntimeError("Deterministic tool execution failure!")

    @server.tool(name="large_payload", description="Returns large text.")
    def large_payload() -> str:
        return "A" * 50_000

    return server.http_app(transport="streamable-http")


# -----------------------------------------------------------------------------
# 3. Client Capabilities & Lifecycle Integration Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_client_connect_and_negotiation(mock_fastmcp_app):
    """Verify successful initialization, capability negotiation, and server metadata."""
    async with mock_fastmcp_app.router.lifespan_context(mock_fastmcp_app):
        transport = httpx2.ASGITransport(app=mock_fastmcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                headers=headers,
                timeout=timeout,
            )

        config = MCPClientConfig(
            endpoint_url="http://testserver/mcp",
            transport=MCPTransportType.STREAMABLE_HTTP,
            allow_private_hosts=True,
        )

        async with GenericMCPClient(config, http_client_factory=client_factory) as client:
            assert client.is_connected is True
            info = client.server_info
            assert info is not None
            assert info.name == "MockRemoteServer"
            assert info.protocol_version is not None
            assert "Operational instructions" in (info.instructions or "")
            assert isinstance(info.capabilities, dict)

        # Connection should close deterministically upon exit
        assert client.is_connected is False


@pytest.mark.asyncio
async def test_client_tool_discovery(mock_fastmcp_app):
    """Verify tool discovery preserves tool names, descriptions, and input schemas."""
    async with mock_fastmcp_app.router.lifespan_context(mock_fastmcp_app):
        transport = httpx2.ASGITransport(app=mock_fastmcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                headers=headers,
                timeout=timeout,
            )

        config = MCPClientConfig(
            endpoint_url="http://testserver/mcp",
            allow_private_hosts=True,
        )

        async with GenericMCPClient(config, http_client_factory=client_factory) as client:
            tools = await client.list_tools()
            assert len(tools) >= 4
            tool_dict = {t.name: t for t in tools}

            assert "echo" in tool_dict
            assert tool_dict["echo"].description == "Echoes back the message."
            assert "message" in tool_dict["echo"].input_schema.get("properties", {})

            assert "calculate" in tool_dict
            calc_props = tool_dict["calculate"].input_schema.get("properties", {})
            assert "a" in calc_props
            assert "b" in calc_props


@pytest.mark.asyncio
async def test_client_successful_tool_invocation(mock_fastmcp_app):
    """Verify tool invocation executes correctly and normalizes results."""
    async with mock_fastmcp_app.router.lifespan_context(mock_fastmcp_app):
        transport = httpx2.ASGITransport(app=mock_fastmcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                headers=headers,
                timeout=timeout,
            )

        config = MCPClientConfig(
            endpoint_url="http://testserver/mcp",
            allow_private_hosts=True,
        )

        async with GenericMCPClient(config, http_client_factory=client_factory) as client:
            # 1. Echo invocation
            res1 = await client.call_tool("echo", {"message": "Hello MCP!"})
            assert isinstance(res1, MCPToolResult)
            assert res1.name == "echo"
            assert res1.is_error is False
            assert "echo: Hello MCP!" in res1.text

            # 2. Calculate invocation
            res2 = await client.call_tool("calculate", {"a": 7, "b": 6, "op": "multiply"})
            assert res2.is_error is False
            assert "42" in res2.text


@pytest.mark.asyncio
async def test_client_tool_execution_error_normalization(mock_fastmcp_app):
    """Verify that tool-level execution failures are cleanly normalized into is_error=True."""
    async with mock_fastmcp_app.router.lifespan_context(mock_fastmcp_app):
        transport = httpx2.ASGITransport(app=mock_fastmcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                headers=headers,
                timeout=timeout,
            )

        config = MCPClientConfig(
            endpoint_url="http://testserver/mcp",
            allow_private_hosts=True,
        )

        async with GenericMCPClient(config, http_client_factory=client_factory) as client:
            res = await client.call_tool("faulty_tool", {})
            assert isinstance(res, MCPToolResult)
            assert res.is_error is True
            assert res.error_message is not None
            assert "Deterministic tool execution failure" in res.error_message


@pytest.mark.asyncio
async def test_client_unknown_tool_invocation(mock_fastmcp_app):
    """Verify that calling an unknown tool returns an error result."""
    async with mock_fastmcp_app.router.lifespan_context(mock_fastmcp_app):
        transport = httpx2.ASGITransport(app=mock_fastmcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                headers=headers,
                timeout=timeout,
            )

        config = MCPClientConfig(
            endpoint_url="http://testserver/mcp",
            allow_private_hosts=True,
        )

        async with GenericMCPClient(config, http_client_factory=client_factory) as client:
            res = await client.call_tool("unregistered_tool_name", {})
            assert res.is_error is True
            assert "Unknown tool" in (res.error_message or "")


@pytest.mark.asyncio
async def test_client_invalid_tool_arguments():
    """Verify that non-dict tool arguments are rejected before transmission."""
    config = MCPClientConfig(
        endpoint_url="http://127.0.0.1:8000/mcp",
        allow_private_hosts=True,
    )
    client = GenericMCPClient(config)
    client._is_connected = True
    client._session = MagicMock()

    with pytest.raises(ValueError) as exc1:
        await client.call_tool("", {})
    assert "non-empty string" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        await client.call_tool("echo", "not-a-dict")  # type: ignore[arg-type]
    assert "must be a dictionary" in str(exc2.value)


@pytest.mark.asyncio
async def test_client_response_size_limit_exceeded(mock_fastmcp_app):
    """Verify that oversized tool responses trigger MCPPayloadTooLargeError."""
    async with mock_fastmcp_app.router.lifespan_context(mock_fastmcp_app):
        transport = httpx2.ASGITransport(app=mock_fastmcp_app)

        def client_factory(headers=None, timeout=None):
            return httpx2.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                headers=headers,
                timeout=timeout,
            )

        # Restrict max response bytes to 5 KB while large_payload returns 50 KB
        config = MCPClientConfig(
            endpoint_url="http://testserver/mcp",
            max_response_bytes=5_000,
            allow_private_hosts=True,
        )

        async with GenericMCPClient(config, http_client_factory=client_factory) as client:
            with pytest.raises(MCPPayloadTooLargeError) as exc_info:
                await client.call_tool("large_payload", {})
            assert "exceeded" in str(exc_info.value).lower() and "limit" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_client_auth_failure_mapping():
    """Verify that HTTP 401/403 status is mapped to MCPAuthenticationError."""
    # Mock transport raising HTTPStatusError 401
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    status_error = httpx2.HTTPStatusError("Unauthorized", request=MagicMock(), response=mock_resp)

    def failing_factory(headers=None, timeout=None):
        mock_client = AsyncMock(spec=httpx2.AsyncClient)
        mock_client.__aenter__.side_effect = status_error
        return mock_client

    config = MCPClientConfig(
        endpoint_url="http://127.0.0.1:8000/mcp",
        allow_private_hosts=True,
    )

    client = GenericMCPClient(config, http_client_factory=failing_factory)
    with pytest.raises(MCPAuthenticationError) as exc_info:
        await client.connect()
    assert "authentication failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_client_timeout_handling():
    """Verify that timeout errors are mapped to MCPTimeoutError."""
    timeout_err = httpx2.TimeoutException("Timed out connecting")

    def timing_out_factory(headers=None, timeout=None):
        mock_client = AsyncMock(spec=httpx2.AsyncClient)
        mock_client.__aenter__.side_effect = timeout_err
        return mock_client

    config = MCPClientConfig(
        endpoint_url="http://127.0.0.1:8000/mcp",
        timeout_seconds=0.1,
        allow_private_hosts=True,
    )

    client = GenericMCPClient(config, http_client_factory=timing_out_factory)
    with pytest.raises(MCPTimeoutError) as exc_info:
        await client.connect()
    assert "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_client_paginated_tool_discovery():
    """Verify that list_tools handles cursor-based pagination seamlessly."""
    config = MCPClientConfig(
        endpoint_url="http://127.0.0.1:8000/mcp",
        max_pagination_pages=5,
        allow_private_hosts=True,
    )
    client = GenericMCPClient(config)
    client._is_connected = True

    # Build mock paginated results
    tool1 = MagicMock()
    tool1.name = "tool_page1"
    tool1.description = "page 1 tool"
    tool1.input_schema = {}
    tool1.output_schema = None

    tool2 = MagicMock()
    tool2.name = "tool_page2"
    tool2.description = "page 2 tool"
    tool2.input_schema = {}
    tool2.output_schema = None

    res_page1 = MagicMock()
    res_page1.tools = [tool1]
    res_page1.next_cursor = "cursor_2"

    res_page2 = MagicMock()
    res_page2.tools = [tool2]
    res_page2.next_cursor = None

    mock_session = AsyncMock()
    mock_session.list_tools.side_effect = [res_page1, res_page2]
    client._session = mock_session

    tools = await client.list_tools()
    assert len(tools) == 2
    assert tools[0].name == "tool_page1"
    assert tools[1].name == "tool_page2"
    assert mock_session.list_tools.call_count == 2
