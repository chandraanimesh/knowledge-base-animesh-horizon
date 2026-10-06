"""
Controlled diagnostic service for Production RAG MCP interface.
Checks runtime, imports, database, pgvector, providers, and index health.
Does NOT expose arbitrary command/python execution, secrets, or sensitive env variables.
"""

from __future__ import annotations

import importlib
import logging
import platform
import sys
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, text
from rag_app.core.config import get_settings
from rag_app.db.session import AsyncSessionLocal
from rag_app.db.models import Document, DocumentChunk, Department, User
from rag_app.mcp.schemas import DiagnosticsResult

logger = logging.getLogger(__name__)

# Strictly allowlisted packages to check
ALLOWLISTED_PACKAGES = [
    "fastapi",
    "fastmcp",
    "sqlalchemy",
    "asyncpg",
    "pgvector",
    "pydantic",
    "sentence_transformers",
    "alembic",
    "groq",
    "openai",
    "google.generativeai",
]


async def run_system_diagnostics(include_rag_smoketest: bool = False) -> DiagnosticsResult:
    """
    Executes controlled system health checks using safe allowlists.
    """
    settings = get_settings()
    now_iso = datetime.now(timezone.utc).isoformat()
    overall_status = "healthy"

    # 1. Runtime info
    runtime_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "app_name": getattr(settings, "app_name", "production-rag-backend"),
        "environment": getattr(settings, "env", "dev"),
    }

    # 2. Package imports
    imports_status: dict[str, bool] = {}
    for pkg in ALLOWLISTED_PACKAGES:
        try:
            importlib.import_module(pkg)
            imports_status[pkg] = True
        except Exception:
            imports_status[pkg] = False
            if pkg in ("fastapi", "fastmcp", "sqlalchemy", "asyncpg", "pgvector"):
                overall_status = "degraded"

    # 3. Database & pgvector check
    db_status: dict[str, Any] = {
        "connected": False,
        "pgvector_extension": False,
        "tables": {},
        "error": None,
    }

    try:
        async with AsyncSessionLocal() as session:
            # Connectivity
            res = await session.execute(text("SELECT 1"))
            if res.scalar() == 1:
                db_status["connected"] = True

            # pgvector extension check
            ext_res = await session.execute(
                text("SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'")
            )
            vec_ver = ext_res.scalar()
            db_status["pgvector_extension"] = bool(vec_ver)
            if vec_ver:
                db_status["pgvector_version"] = str(vec_ver)

            # Table row counts
            doc_count = await session.scalar(select(text("count(*)")).select_from(Document))
            chunk_count = await session.scalar(select(text("count(*)")).select_from(DocumentChunk))
            dept_count = await session.scalar(select(text("count(*)")).select_from(Department))
            user_count = await session.scalar(select(text("count(*)")).select_from(User))

            db_status["tables"] = {
                "documents": int(doc_count or 0),
                "document_chunks": int(chunk_count or 0),
                "departments": int(dept_count or 0),
                "users": int(user_count or 0),
            }
    except Exception as exc:
        db_status["connected"] = False
        db_status["error"] = f"DB connection check failed: {type(exc).__name__}"
        overall_status = "unhealthy"

    # 4. Providers & Models check
    llm_model_name = getattr(settings, f"{settings.llm_provider}_model", getattr(settings, "groq_model", "unknown"))
    models_status: dict[str, Any] = {
        "embedding_provider": getattr(settings, "embedding_provider", "unknown"),
        "embedding_dim": getattr(settings, "embedding_dim", 384),
        "reranker_provider": None,
        "llm_provider": getattr(settings, "llm_provider", "unknown"),
        "llm_model": str(llm_model_name),
    }

    try:
        from rag_app.ingestion.embedding_v1 import get_embedding_provider
        emb = get_embedding_provider()
        models_status["embedding_provider"] = type(emb).__name__
    except Exception as exc:
        models_status["embedding_error"] = f"{type(exc).__name__}: {exc}"
        overall_status = "degraded"

    try:
        from rag_app.retrieval.reranker import get_reranker_provider
        reranker = get_reranker_provider()
        models_status["reranker_provider"] = type(reranker).__name__
    except Exception as exc:
        models_status["reranker_error"] = f"{type(exc).__name__}: {exc}"

    # 5. RAG & BM25 Index check
    rag_status: dict[str, Any] = {
        "bm25_index_ready": False,
        "smoke_test_run": include_rag_smoketest,
        "smoke_test_success": None,
    }

    try:
        from rag_app.retrieval.bm25 import GlobalBM25Index
        bm25 = GlobalBM25Index.get_instance()
        rag_status["bm25_index_ready"] = hasattr(bm25, "bm25") and bm25.bm25 is not None

        if include_rag_smoketest and db_status["connected"]:
            # Smoke test: test embedding provider vector generation
            from rag_app.ingestion.embedding_v1 import get_embedding_provider
            emb_provider = get_embedding_provider()
            vec = await emb_provider.embed_query("diagnostic smoke test")
            rag_status["smoke_test_success"] = len(vec) == settings.embedding_dim
    except Exception as exc:
        rag_status["error"] = f"{type(exc).__name__}: {exc}"
        if overall_status == "healthy":
            overall_status = "degraded"

    return DiagnosticsResult(
        status=overall_status,
        timestamp=now_iso,
        runtime=runtime_info,
        imports=imports_status,
        database=db_status,
        models=models_status,
        rag=rag_status,
    )
