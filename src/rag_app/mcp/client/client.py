"""
Generic Outbound MCP Client Engine.

Provides a protocol-compliant, reusable MCP client utilizing the official MCP Python SDK
(mcp.ClientSession, streamable_http_client, sse_client).
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import logging
from typing import Any, Callable, Optional

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client
from mcp.shared._httpx_utils import create_mcp_http_client
import mcp.shared.exceptions as mcp_exc
import mcp.types as mcp_types

from rag_app.mcp.client.models import (
    MCPAuthenticationError,
    MCPClientConfig,
    MCPClientError,
    MCPConnectionError,
    MCPPayloadTooLargeError,
    MCPProtocolError,
    MCPServerInfo,
    MCPTimeoutError,
    MCPToolDefinition,
    MCPToolExecutionError,
    MCPToolResult,
    MCPTransportType,
)
from rag_app.mcp.client.security import (
    sanitize_headers_for_logging,
    validate_mcp_url,
)

logger = logging.getLogger(__name__)


class GenericMCPClient:
    """
    Protocol-compliant, provider-neutral outbound MCP Client.
    
    Supports:
    - Streamable HTTP (primary remote transport)
    - Legacy SSE (for compatibility with existing SSE MCP servers)
    - MCP initialization and capability negotiation
    - Paginated tool discovery
    - Validated tool invocation with normalized results
    - SSRF prevention, bounded timeouts, response size limits, and safe error handling
    - Deterministic async context manager lifecycle
    """

    def __init__(
        self,
        config: MCPClientConfig,
        http_client_factory: Optional[Callable[..., httpx2.AsyncClient]] = None,
    ):
        self.config = config
        self.http_client_factory = http_client_factory
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._server_info: Optional[MCPServerInfo] = None
        self._is_connected: bool = False

    @property
    def is_connected(self) -> bool:
        """Returns True if the client is currently connected and initialized."""
        return self._is_connected and self._session is not None

    @property
    def server_info(self) -> Optional[MCPServerInfo]:
        """Returns metadata regarding the connected MCP server."""
        return self._server_info

    async def __aenter__(self) -> "GenericMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> MCPServerInfo:
        """
        Establishes transport connection, creates ClientSession, and performs
        MCP initialization and capability negotiation.
        """
        if self._is_connected and self._session is not None:
            return self._server_info  # type: ignore[return-value]

        # 1. Validate endpoint against SSRF and prohibited networks
        validate_mcp_url(
            self.config.endpoint_url,
            allow_private_hosts=self.config.allow_private_hosts,
        )

        sanitized_headers = sanitize_headers_for_logging(self.config.auth_headers)
        logger.debug(
            "Connecting to MCP endpoint %s via %s (headers: %s)",
            self.config.endpoint_url,
            self.config.transport.value,
            list(sanitized_headers.keys()),
        )

        exit_stack = AsyncExitStack()
        try:
            # 2. Establish transport streams
            read_stream: Any = None
            write_stream: Any = None

            if self.config.transport == MCPTransportType.STREAMABLE_HTTP:
                client_timeout = httpx2.Timeout(
                    self.config.timeout_seconds,
                    read=self.config.read_timeout_seconds,
                )
                if self.http_client_factory:
                    http_client = self.http_client_factory(
                        headers=self.config.auth_headers,
                        timeout=client_timeout,
                    )
                else:
                    http_client = create_mcp_http_client(
                        headers=self.config.auth_headers,
                        timeout=client_timeout,
                    )
                await exit_stack.enter_async_context(http_client)

                streams_gen = streamable_http_client(
                    self.config.endpoint_url,
                    http_client=http_client,
                    max_sse_event_size=self.config.max_response_bytes,
                )
                read_stream, write_stream = await exit_stack.enter_async_context(streams_gen)

            elif self.config.transport == MCPTransportType.SSE:
                factory = self.http_client_factory or create_mcp_http_client
                streams_gen = sse_client(
                    self.config.endpoint_url,
                    headers=self.config.auth_headers,
                    timeout=self.config.timeout_seconds,
                    sse_read_timeout=self.config.read_timeout_seconds,
                    httpx_client_factory=factory,
                )
                read_stream, write_stream = await exit_stack.enter_async_context(streams_gen)
            else:
                raise MCPClientError(f"Unsupported transport type: {self.config.transport}")

            # 3. Create and initialize ClientSession
            session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                read_timeout_seconds=self.config.read_timeout_seconds,
            )
            await exit_stack.enter_async_context(session)

            # 4. Perform initialization handshake with bounded timeout
            init_res = await asyncio.wait_for(
                session.initialize(),
                timeout=self.config.timeout_seconds,
            )

            # 5. Capture negotiated protocol information and server metadata
            server_name = "Unknown"
            server_version = None
            if init_res.server_info:
                server_name = init_res.server_info.name
                server_version = init_res.server_info.version

            capabilities_dict: dict[str, Any] = {}
            if hasattr(init_res.capabilities, "model_dump"):
                capabilities_dict = init_res.capabilities.model_dump(by_alias=True)
            elif isinstance(init_res.capabilities, dict):
                capabilities_dict = init_res.capabilities

            self._server_info = MCPServerInfo(
                name=server_name,
                version=server_version,
                protocol_version=session.protocol_version,
                capabilities=capabilities_dict,
                instructions=init_res.instructions,
            )
            self._session = session
            self._exit_stack = exit_stack
            self._is_connected = True

            logger.info(
                "Connected to MCP server '%s' (protocol %s, version %s)",
                server_name,
                session.protocol_version,
                server_version,
            )
            return self._server_info

        except Exception as exc:
            await exit_stack.aclose()
            self._is_connected = False
            self._session = None
            self._exit_stack = None
            self._handle_transport_exception(exc, "MCP connection or initialization failed")
            raise  # pragma: no cover

    async def list_tools(self) -> list[MCPToolDefinition]:
        """
        Discovers tools registered on the connected MCP server.
        Handles bounded pagination preserving names, descriptions, and schemas.
        """
        if not self._is_connected or not self._session:
            raise MCPConnectionError("MCP client is not connected.")

        all_tools: list[MCPToolDefinition] = []
        cursor: Optional[str] = None
        pages_fetched = 0

        try:
            while pages_fetched < self.config.max_pagination_pages:
                params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
                result: mcp_types.ListToolsResult = await asyncio.wait_for(
                    self._session.list_tools(params=params),
                    timeout=self.config.timeout_seconds,
                )

                for tool in result.tools:
                    all_tools.append(
                        MCPToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            input_schema=tool.input_schema or {},
                            output_schema=getattr(tool, "output_schema", None),
                        )
                    )

                cursor = result.next_cursor
                pages_fetched += 1
                if not cursor:
                    break

            return all_tools

        except Exception as exc:
            self._handle_transport_exception(exc, "Tool discovery failed")
            raise  # pragma: no cover

    async def call_tool(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> MCPToolResult:
        """
        Invokes a selected tool on the remote server with validated request structure.
        Normalizes successful results, tool errors, and protocol errors into an MCPToolResult.
        Enforces size bounds and bounded execution timeouts.
        """
        if not self._is_connected or not self._session:
            raise MCPConnectionError("MCP client is not connected.")

        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string.")

        if arguments is not None and not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a dictionary or None.")

        tool_args = arguments if arguments is not None else {}

        try:
            raw_res = await asyncio.wait_for(
                self._session.call_tool(
                    name=name,
                    arguments=tool_args,
                    read_timeout_seconds=self.config.read_timeout_seconds,
                ),
                timeout=self.config.timeout_seconds,
            )

            # Normalize CallToolResult
            if isinstance(raw_res, mcp_types.CallToolResult):
                content_list: list[dict[str, Any]] = []
                text_parts: list[str] = []
                total_bytes = 0

                for item in raw_res.content:
                    if hasattr(item, "model_dump"):
                        dumped = item.model_dump(by_alias=True)
                    elif isinstance(item, dict):
                        dumped = item
                    else:
                        dumped = {"type": getattr(item, "type", "unknown")}

                    content_list.append(dumped)

                    item_text = getattr(item, "text", None) or dumped.get("text")
                    if item_text:
                        text_str = str(item_text)
                        text_parts.append(text_str)
                        total_bytes += len(text_str.encode("utf-8"))

                if total_bytes > self.config.max_response_bytes:
                    raise MCPPayloadTooLargeError(
                        f"MCP tool response payload ({total_bytes} bytes) exceeds limit ({self.config.max_response_bytes} bytes)."
                    )

                full_text = "\n".join(text_parts)
                is_error = bool(raw_res.is_error)
                error_message = None
                if is_error:
                    error_message = full_text or "Remote tool reported execution error."

                return MCPToolResult(
                    name=name,
                    content=content_list,
                    text=full_text,
                    structured_content=raw_res.structured_content,
                    is_error=is_error,
                    error_message=error_message,
                    meta=raw_res.meta,
                )

            # Fallback for unexpected result types
            return MCPToolResult(
                name=name,
                content=[{"raw": str(raw_res)}],
                text=str(raw_res),
                is_error=False,
            )

        except mcp_exc.MCPError as me:
            err_text = str(me)
            if "exceeded" in err_text.lower() and "limit" in err_text.lower():
                raise MCPPayloadTooLargeError(f"Response size exceeded limit: {err_text}") from me
            err_msg = str(getattr(getattr(me, "error", None), "message", None) or me)
            return MCPToolResult(
                name=name,
                is_error=True,
                error_message=f"MCP Error: {err_msg}",
            )

        except Exception as exc:
            self._handle_transport_exception(exc, f"Tool invocation for '{name}' failed")
            raise  # pragma: no cover

    async def close(self) -> None:
        """Deterministically closes the MCP session and HTTP transport streams."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.debug("Exception during MCP client cleanup: %s", e)
            finally:
                self._exit_stack = None
                self._session = None
                self._is_connected = False
                logger.debug("Closed MCP client connection to %s", self.config.endpoint_url)

    def _handle_transport_exception(self, exc: Exception, context_msg: str) -> None:
        """
        Maps underlying transport, network, and HTTP exceptions to the
        MCP exception hierarchy.
        """
        # Handle ExceptionGroup if present
        unwrapped: Exception = exc
        if hasattr(exc, "exceptions") and getattr(exc, "exceptions"):
            unwrapped = exc.exceptions[0]

        if isinstance(unwrapped, (MCPClientError, ValueError)):
            raise unwrapped

        if isinstance(unwrapped, httpx2.HTTPStatusError):
            status = unwrapped.response.status_code if unwrapped.response else 500
            if status in (401, 403):
                raise MCPAuthenticationError(
                    f"{context_msg}: authentication failed (HTTP {status})."
                ) from unwrapped
            raise MCPConnectionError(
                f"{context_msg}: HTTP error status {status}."
            ) from unwrapped

        if isinstance(unwrapped, (asyncio.TimeoutError, httpx2.TimeoutException)):
            raise MCPTimeoutError(
                f"{context_msg}: request timed out after configured limit."
            ) from unwrapped

        if isinstance(unwrapped, (httpx2.ConnectError, httpx2.NetworkError, ConnectionError)):
            raise MCPConnectionError(
                f"{context_msg}: network or connection error ({type(unwrapped).__name__})."
            ) from unwrapped

        if isinstance(unwrapped, (mcp_exc.JSONRPCError, mcp_exc.MCPError)):
            msg = str(unwrapped)
            if "exceeded" in msg.lower() and "limit" in msg.lower():
                raise MCPPayloadTooLargeError(
                    f"{context_msg}: payload size exceeded limit ({msg})."
                ) from unwrapped
            raise MCPProtocolError(
                f"{context_msg}: protocol error ({unwrapped})."
            ) from unwrapped

        raise MCPConnectionError(f"{context_msg}: {type(unwrapped).__name__} - {unwrapped}") from unwrapped
