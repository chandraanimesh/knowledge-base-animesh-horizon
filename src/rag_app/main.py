
import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse

from rag_app.api.routes import router
from rag_app.core.config import get_settings
from rag_app.core.logging import new_request_id, set_request_id, setup_logging

settings = get_settings()
setup_logging(settings.log_level)

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager
from rag_app.retrieval.bm25 import GlobalBM25Index
from rag_app.db.session import AsyncSessionLocal
from starlette.types import Scope, Receive, Send
from starlette.responses import JSONResponse, Response

from rag_app.mcp.server import mcp, get_mcp_asgi_app

mcp_sse_app = get_mcp_asgi_app(transport="sse", path="/sse")
mcp_http_app = get_mcp_asgi_app(transport="streamable-http", path="/")

@asynccontextmanager
async def lifespan(app: FastAPI):
    bm25 = GlobalBM25Index.get_instance()
    try:
        async with AsyncSessionLocal() as session:
            await bm25.build(session)
    except Exception as bm25_err:
        is_prod = getattr(settings, "env", "dev").lower() in ("prod", "production")
        is_test_env = (
            getattr(settings, "env", "").lower() in ("test", "testing")
            or "test" in str(getattr(settings, "database_url", "")).lower()
        )
        if is_prod:
            logger.critical(
                "Fatal: BM25 index initialization failed in production schema: %s",
                bm25_err,
                exc_info=True,
            )
            raise RuntimeError(
                f"Production startup aborted: BM25 index initialization failed: {bm25_err}"
            ) from bm25_err
        elif is_test_env:
            logger.warning(
                "Gracefully tolerating missing or uninitialized BM25 index in test/disposable environment: %s",
                bm25_err,
            )
        else:
            logger.error(
                "BM25 index build failed during startup (check database migrations / schema): %s",
                bm25_err,
                exc_info=True,
            )

    from rag_app.research.registry import get_provider_registry, DuplicateProviderError
    from rag_app.research.providers.firecrawl import FirecrawlProvider
    from rag_app.research.providers.tinyfish import TinyFishProvider
    from rag_app.research.provider import ProviderRequestError, ProviderUnavailableError

    registry = get_provider_registry()
    try:
        fc_provider = FirecrawlProvider()
        registry.register("firecrawl", fc_provider)
        logger.info("Successfully registered 'firecrawl' WebProvider.")
    except (ProviderRequestError, ProviderUnavailableError) as e:
        logger.warning(f"Skipped registering 'firecrawl' WebProvider: {e}")
    except DuplicateProviderError:
        pass

    try:
        tf_provider = TinyFishProvider()
        registry.register("tinyfish", tf_provider)
        logger.info("Successfully registered 'tinyfish' WebProvider.")
    except (ProviderRequestError, ProviderUnavailableError) as e:
        logger.warning(f"Skipped registering 'tinyfish' WebProvider: {e}")
    except DuplicateProviderError:
        pass

    from rag_app.knowledge.scheduler import get_reconciliation_scheduler
    reconciliation_scheduler = get_reconciliation_scheduler(interval_seconds=settings.scheduler_interval_seconds)
    try:
        await reconciliation_scheduler.start()
        logger.info("Durable reconciliation scheduler started successfully.")
    except Exception as sched_err:
        logger.warning(f"Failed to start durable reconciliation scheduler: {sched_err}")

    from rag_app.core.tokenizer import log_effective_model_diagnostics
    try:
        log_effective_model_diagnostics()
    except Exception as diag_err:
        logger.warning(f"Failed to log effective model diagnostics: {diag_err}")

    from rag_app.mcp.server import mcp
    async with mcp.lifespan():
        async with mcp_sse_app.router.lifespan_context(mcp_sse_app):
            async with mcp_http_app.router.lifespan_context(mcp_http_app):
                yield

    try:
        await reconciliation_scheduler.stop()
        logger.info("Durable reconciliation scheduler stopped.")
    except Exception as stop_err:
        logger.warning(f"Error stopping durable reconciliation scheduler: {stop_err}")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8443", "http://localhost:5173"],
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestIdASGIMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path.startswith("/mcp"):
            # Completely bypass request tracking for /mcp streaming endpoints
            return await self.app(scope, receive, send)

        request_id = new_request_id()
        for header_name, header_val in scope.get("headers", []):
            if header_name.lower() == b"x-request-id":
                request_id = header_val.decode("latin1")
                break

        set_request_id(request_id)
        start = time.perf_counter()

        logger.info(
            "request_started",
            extra={
                "event": "request_started",
                "method": scope.get("method"),
                "path": path,
            },
        )

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin1")))
                headers.append((b"x-response-time-ms", str(latency_ms).encode("latin1")))
                message["headers"] = headers

                logger.info(
                    "request_finished",
                    extra={
                        "event": "request_finished",
                        "method": scope.get("method"),
                        "path": path,
                        "status_code": message.get("status"),
                        "latency_ms": latency_ms,
                    },
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)

app.add_middleware(RequestIdASGIMiddleware)

from rag_app.api.auth_routes import router as auth_router
from rag_app.api.admin_routes import router as admin_router, doc_admin_router
from rag_app.api.department_routes import router as department_router
from rag_app.api.access_request_routes import router as access_request_router
from rag_app.api.conversation_routes import router as conversation_router
from rag_app.api.workspace_routes import router as workspace_router
from rag_app.api.knowledge_routes import router as knowledge_router
from rag_app.api.memory_routes import router as memory_router
from rag_app.api.task_routes import router as task_router
from rag_app.api.connection_routes import router as connection_router
from rag_app.api.crm_routes import router as crm_router
from rag_app.api.artifact_routes import router as artifact_router
from rag_app.api.workflow_routes import router as workflow_router
from rag_app.api.workflow_definition_routes import router as workflow_definition_router
from rag_app.api.workflow_trigger_routes import router as workflow_trigger_router
from rag_app.api.workspace_mcp_routes import router as workspace_mcp_router
from rag_app.api.chunk_curation_routes import router as chunk_curation_router
from rag_app.api.document_note_routes import router as document_note_router
from rag_app.api.streaming_routes import router as streaming_router
from rag_app.api.registration_routes import router as registration_router
from rag_app.api.living_article_routes import router as living_article_router
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(doc_admin_router)
app.include_router(department_router)
app.include_router(access_request_router)
app.include_router(conversation_router)
app.include_router(workspace_router)
app.include_router(workspace_mcp_router)
app.include_router(knowledge_router)
app.include_router(connection_router)
app.include_router(crm_router)
app.include_router(memory_router)
app.include_router(memory_router, prefix="/api")
app.include_router(memory_router, prefix="/api/v1")
app.include_router(task_router)
app.include_router(artifact_router)
app.include_router(artifact_router, prefix="/api")
app.include_router(workflow_router)
app.include_router(workflow_definition_router)
app.include_router(workflow_trigger_router)
app.include_router(chunk_curation_router)
app.include_router(chunk_curation_router, prefix="/api/v1")
app.include_router(document_note_router)
app.include_router(streaming_router)
app.include_router(registration_router)
app.include_router(living_article_router)
app.include_router(router)

# ── Unified MCP Gateway Middleware ──────────────────────────────────────────
from rag_app.mcp.policies import TOOL_POLICIES

_cached_dev_token: str | None = None

async def resolve_dev_bearer_token() -> str | None:
    global _cached_dev_token
    if _cached_dev_token:
        return _cached_dev_token
    try:
        from rag_app.db.session import AsyncSessionLocal
        from rag_app.db.models import User
        from sqlalchemy import select
        from rag_app.security.authentication import create_access_token
        from datetime import timedelta
        async with AsyncSessionLocal() as session:
            admin_user = (await session.scalars(select(User).where(User.role == "ADMIN", User.is_active == True))).first()
            if admin_user:
                _cached_dev_token = create_access_token(subject=str(admin_user.id), expires_delta=timedelta(days=365))
                return _cached_dev_token
    except Exception as e:
        logger.warning(f"Failed to generate dev bearer token: {e}")
    return None

class MCPGatewayASGIMiddleware:
    """
    Unified MCP Gateway handling both Streamable HTTP (ChatGPT / Modern MCP)
    and SSE transports (Twenty CRM / Legacy MCP) under /mcp.
    
    Guarantees:
    - Zero buffering on SSE and streaming responses (Cache-Control: no-cache, no-transform; X-Accel-Buffering: no).
    - Single response.start protection against Starlette double-header assertions.
    - CORS preflight (OPTIONS) support for web clients.
    - Path normalization so ChatGPT succeeds whether configured with /mcp or /mcp/sse.
    """
    def __init__(self, asgi_app, sse_app, http_app):
        self.asgi_app = asgi_app
        self.sse_app = sse_app
        self.http_app = http_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.asgi_app(scope, receive, send)

        path = scope.get("path", "")
        if not path.startswith("/mcp"):
            return await self.asgi_app(scope, receive, send)

        method = scope.get("method", "GET").upper()
        norm_path = path.rstrip("/")

        # Check authorization header or dev fallback
        raw_headers = list(scope.get("headers", []))
        headers_dict = dict(raw_headers)
        if b"authorization" not in headers_dict:
            query_string = scope.get("query_string", b"").decode("latin1")
            token_val = None
            if "token=" in query_string or "auth_token=" in query_string:
                from urllib.parse import parse_qs
                qs = parse_qs(query_string)
                token_val = qs.get("token", qs.get("auth_token", [None]))[0]
            elif settings.env == "dev":
                token_val = await resolve_dev_bearer_token()

            if token_val:
                if not token_val.lower().startswith("bearer "):
                    token_val = f"Bearer {token_val}"
                raw_headers.append((b"authorization", token_val.encode("latin1")))
                scope["headers"] = raw_headers

        # Handle CORS preflight
        if method == "OPTIONS":
            async def cors_send(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    headers[b"access-control-allow-origin"] = b"*"
                    headers[b"access-control-allow-methods"] = b"GET, POST, OPTIONS, DELETE, HEAD"
                    headers[b"access-control-allow-headers"] = b"*"
                    message["headers"] = list(headers.items())
                await send(message)
            res = Response(status_code=204)
            await res(scope, receive, cors_send)
            return

        # Response wrapper ensuring unbuffered SSE and preventing duplicate http.response.start
        response_started = False
        async def unbuffered_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                if response_started:
                    return
                response_started = True
                headers = dict(message.get("headers", []))
                headers[b"cache-control"] = b"no-cache, no-transform"
                headers[b"x-accel-buffering"] = b"no"
                headers[b"access-control-allow-origin"] = b"*"
                message["headers"] = list(headers.items())
            elif message["type"] == "http.response.body":
                response_started = True
            await send(message)

        # 1. Base /mcp or /mcp/
        if norm_path == "/mcp":
            if method == "POST":
                # ChatGPT / Streamable HTTP POST initialization or JSON-RPC
                new_scope = dict(scope)
                new_scope["path"] = "/"
                new_scope["raw_path"] = b"/"
                return await self.http_app(new_scope, receive, unbuffered_send)
            elif method == "GET":
                headers = dict(scope.get("headers", []))
                accept = headers.get(b"accept", b"").decode("latin1")
                if "text/event-stream" in accept:
                    new_scope = dict(scope)
                    new_scope["path"] = "/"
                    new_scope["raw_path"] = b"/"
                    return await self.http_app(new_scope, receive, unbuffered_send)
                res = JSONResponse({
                    "status": "ready",
                    "name": "Production RAG MCP Server",
                    "transport": "streamable-http+sse",
                    "streamable_endpoint": "/mcp",
                    "sse_endpoint": "/mcp/sse",
                    "messages_endpoint": "/mcp/messages/",
                    "tools_count": len(TOOL_POLICIES),
                    "instructions": "Supports both Streamable HTTP at /mcp and SSE at /mcp/sse.",
                })
                return await res(scope, receive, unbuffered_send)
            elif method == "DELETE":
                new_scope = dict(scope)
                new_scope["path"] = "/"
                new_scope["raw_path"] = b"/"
                return await self.http_app(new_scope, receive, unbuffered_send)

        # 2. /mcp/sse endpoint
        if norm_path == "/mcp/sse":
            if method == "POST":
                # ChatGPT configured with URL ending in /sse sending POST JSON-RPC
                new_scope = dict(scope)
                new_scope["path"] = "/"
                new_scope["raw_path"] = b"/"
                return await self.http_app(new_scope, receive, unbuffered_send)
            elif method in ("GET", "HEAD"):
                # Twenty CRM or standard SSE client
                new_scope = dict(scope)
                new_scope["path"] = "/sse"
                new_scope["raw_path"] = b"/sse"
                return await self.sse_app(new_scope, receive, unbuffered_send)

        # 3. /mcp/messages (SSE POST messages)
        if path.startswith("/mcp/messages"):
            new_scope = dict(scope)
            new_scope["path"] = path.replace("/mcp", "", 1)
            new_scope["raw_path"] = new_scope["path"].encode("latin1")
            return await self.sse_app(new_scope, receive, unbuffered_send)

        # Default fallback to http_app
        return await self.http_app(scope, receive, unbuffered_send)

app.add_middleware(MCPGatewayASGIMiddleware, sse_app=mcp_sse_app, http_app=mcp_http_app)