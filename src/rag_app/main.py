
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
from rag_app.mcp.server import mcp, get_mcp_asgi_app

mcp_asgi_app = get_mcp_asgi_app(transport="sse")

@asynccontextmanager
async def lifespan(app: FastAPI):
    bm25 = GlobalBM25Index.get_instance()
    async with AsyncSessionLocal() as session:
        await bm25.build(session)

    from rag_app.research.registry import get_provider_registry, DuplicateProviderError
    from rag_app.research.providers.firecrawl import FirecrawlProvider
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

    from rag_app.knowledge.scheduler import get_reconciliation_scheduler
    reconciliation_scheduler = get_reconciliation_scheduler()
    try:
        await reconciliation_scheduler.start()
        logger.info("Durable reconciliation scheduler started successfully.")
    except Exception as sched_err:
        logger.warning(f"Failed to start durable reconciliation scheduler: {sched_err}")

    from rag_app.mcp.server import mcp
    async with mcp.lifespan():
        async with mcp_asgi_app.router.lifespan_context(mcp_asgi_app):
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


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or new_request_id()
    set_request_id(request_id)

    start = time.perf_counter()

    logger.info(
        "request_started",
        extra={
            "event": "request_started",
            "method": request.method,
            "path": request.url.path,
        },
    )

    response = await call_next(request)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = str(latency_ms)

    logger.info(
        "request_finished",
        extra={
            "event": "request_finished",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )

    return response

from rag_app.api.auth_routes import router as auth_router
from rag_app.api.admin_routes import router as admin_router, doc_admin_router
from rag_app.api.department_routes import router as department_router
from rag_app.api.access_request_routes import router as access_request_router
from rag_app.api.conversation_routes import router as conversation_router
from rag_app.api.workspace_routes import router as workspace_router
from rag_app.api.knowledge_routes import router as knowledge_router
from rag_app.api.memory_routes import router as memory_router
from rag_app.api.task_routes import router as task_router
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(doc_admin_router)
app.include_router(department_router)
app.include_router(access_request_router)
app.include_router(conversation_router)
app.include_router(workspace_router)
app.include_router(knowledge_router)
app.include_router(memory_router)
app.include_router(task_router)
app.include_router(router)

# ── MCP Interface Mounting ───────────────────────────────────────────────────
from rag_app.mcp.policies import TOOL_POLICIES

@app.get("/mcp", tags=["mcp"])
@app.get("/mcp/", tags=["mcp"])
async def mcp_discovery():
    """Discovery and health endpoint for local Twenty CRM and MCP clients."""
    return {
        "status": "ready",
        "name": "Production RAG MCP Server",
        "transport": "sse",
        "sse_endpoint": "/mcp/sse",
        "messages_endpoint": "/mcp/messages/",
        "local_base_url": "http://localhost:8000/mcp/sse",
        "tools_count": len(TOOL_POLICIES),
        "instructions": "Plug http://localhost:8000/mcp/sse into Twenty CRM or MCP client with Bearer Authorization header."
    }

app.mount("/mcp", mcp_asgi_app)