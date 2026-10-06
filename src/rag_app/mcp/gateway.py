"""
MCP Gateway for Production RAG.
Centralized interception layer responsible for:
1. Request Context Resolution & Invariant Enforcement
2. Request ID Tracing & Structured Logging
3. Coarse MCP Scope & Admin Privilege Enforcement
4. Error Normalization (AUTHENTICATION_REQUIRED, FORBIDDEN, NOT_FOUND, INVALID_REQUEST)
5. Execution delegation to underlying domain handlers/services
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable, Coroutine, Optional, TypeVar

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.core.logging import new_request_id, set_request_id
from rag_app.db.session import AsyncSessionLocal
from rag_app.mcp.context import (
    MCPAuthenticationError,
    MCPForbiddenError,
    MCPRequestContext,
    resolve_mcp_request_context,
)
from rag_app.mcp.policies import TOOL_POLICIES, ToolPolicy

logger = logging.getLogger(__name__)

R = TypeVar("R")


class MCPGatewayError(Exception):
    """Normalized MCP Gateway error carrying error code and user-safe detail."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "status_code": self.status_code,
        }


class MCPGateway:
    """
    Gateway that wraps all FastMCP tool executions with authentication,
    scope validation, coarse admin checks, structured logging, and normalized errors.
    """

    def __init__(self):
        pass

    async def execute(
        self,
        tool_name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        auth_token: Optional[str] = None,
        mcp_request_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an MCP tool under gateway protection.
        """
        start_time = time.perf_counter()
        req_id = mcp_request_id or new_request_id()
        set_request_id(req_id)

        # 1. Lookup Tool Policy
        policy: Optional[ToolPolicy] = TOOL_POLICIES.get(tool_name)

        # 2. Open DB session if not provided
        owns_session = session is None
        db_session: AsyncSession = session or AsyncSessionLocal()

        try:
            # 3. Resolve Request Context (Identity + Scopes)
            context: Optional[MCPRequestContext] = None
            if tool_name == "run_diagnostics" and not auth_token:
                # Diagnostics allows non-authenticated health probes
                pass
            else:
                try:
                    context = await resolve_mcp_request_context(
                        session=db_session,
                        auth_token=auth_token,
                        request_id=req_id,
                    )
                except MCPAuthenticationError as auth_err:
                    logger.warning(
                        f"[MCP-GATEWAY] Auth failure for tool '{tool_name}' req_id={req_id}: {auth_err}"
                    )
                    raise MCPGatewayError(
                        code="AUTHENTICATION_REQUIRED",
                        message=str(auth_err),
                        status_code=401,
                    )

            # 4. Scope & Coarse Admin Verification
            if policy and context:
                # Scope check
                if policy.scope not in context.scopes:
                    logger.warning(
                        f"[MCP-GATEWAY] Scope check failed for user {context.user.id} on tool '{tool_name}'. "
                        f"Required: {policy.scope}, Possessed: {context.scopes}"
                    )
                    raise MCPGatewayError(
                        code="FORBIDDEN",
                        message=f"Access denied: missing required MCP scope '{policy.scope}'.",
                        status_code=403,
                    )

                # Admin check
                if policy.admin_only and context.user.role != "ADMIN":
                    logger.warning(
                        f"[MCP-GATEWAY] Admin check failed for non-admin user {context.user.id} on tool '{tool_name}'"
                    )
                    raise MCPGatewayError(
                        code="FORBIDDEN",
                        message="Access denied: administrative privileges required for this tool.",
                        status_code=403,
                    )

            # 5. Invoke the underlying handler with prepared context/session/kwargs
            sig = inspect.signature(handler)
            call_kwargs = dict(kwargs)

            if "session" in sig.parameters and "session" not in call_kwargs:
                call_kwargs["session"] = db_session
            if "context" in sig.parameters and "context" not in call_kwargs:
                call_kwargs["context"] = context
            if "user" in sig.parameters and "user" not in call_kwargs and context:
                call_kwargs["user"] = context.user
            if "auth_token" in sig.parameters and "auth_token" not in call_kwargs:
                call_kwargs["auth_token"] = auth_token

            # Filter out any kwargs not accepted by handler
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

            result = await handler(*args, **call_kwargs)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            user_log_id = str(context.user.id) if context else "anonymous"
            logger.info(
                f"[MCP-GATEWAY] Tool '{tool_name}' completed successfully in {elapsed_ms:.2f}ms "
                f"caller={user_log_id} req_id={req_id}"
            )
            return result

        except MCPGatewayError:
            raise

        except HTTPException as http_exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"[MCP-GATEWAY] HTTPException in tool '{tool_name}' ({http_exc.status_code}): {http_exc.detail} "
                f"req_id={req_id} ({elapsed_ms:.2f}ms)"
            )
            if http_exc.status_code == 401:
                raise MCPGatewayError(
                    code="AUTHENTICATION_REQUIRED",
                    message=str(http_exc.detail),
                    status_code=401,
                )
            elif http_exc.status_code == 403:
                raise MCPGatewayError(
                    code="FORBIDDEN",
                    message=str(http_exc.detail),
                    status_code=403,
                )
            elif http_exc.status_code == 404:
                raise MCPGatewayError(
                    code="NOT_FOUND",
                    message=str(http_exc.detail),
                    status_code=404,
                )
            else:
                raise MCPGatewayError(
                    code="INVALID_REQUEST",
                    message=str(http_exc.detail),
                    status_code=http_exc.status_code,
                )

        except ValueError as val_err:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"[MCP-GATEWAY] ValueError in tool '{tool_name}': {val_err} req_id={req_id} ({elapsed_ms:.2f}ms)"
            )
            raise MCPGatewayError(
                code="INVALID_REQUEST",
                message=str(val_err),
                status_code=400,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[MCP-GATEWAY] Internal error executing tool '{tool_name}': {type(exc).__name__}: {exc} "
                f"req_id={req_id} ({elapsed_ms:.2f}ms)",
                exc_info=True,
            )
            raise MCPGatewayError(
                code="INTERNAL_ERROR",
                message="An internal server error occurred while processing the request.",
                status_code=500,
            )

        finally:
            if owns_session:
                await db_session.close()


# Global Gateway Instance
mcp_gateway = MCPGateway()
