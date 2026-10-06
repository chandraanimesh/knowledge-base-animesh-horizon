
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

    yield

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.app_name,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8443", "http://localhost:5173"],
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
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(doc_admin_router)
app.include_router(department_router)
app.include_router(access_request_router)
app.include_router(conversation_router)
app.include_router(workspace_router)
app.include_router(router)

# Mount FastMCP Interface over Streamable HTTP at /mcp
from rag_app.mcp.server import get_mcp_asgi_app
app.mount("/mcp", get_mcp_asgi_app(transport="streamable-http"))