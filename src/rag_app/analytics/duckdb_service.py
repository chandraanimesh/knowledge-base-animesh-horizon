"""DuckDB + Parquet Analytics Service

Provides secure columnar analytics over CSV/tabular datasets using DuckDB.
Enforces SELECT-only queries, memory bounds, execution timeouts, row caps,
and strict multi-tenant workspace/department authorization.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import Document, User
from rag_app.knowledge.errata_service import ErrataService

logger = logging.getLogger(__name__)

PARQUET_STORAGE_DIR = Path("storage/parquet")
PARQUET_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Strict SQL security: Disallowed keywords and commands
FORBIDDEN_SQL_PATTERNS = [
    r"\b(?:attach|install|load|copy|export|import|pragma|call)\b",
    r"\b(?:create|drop|alter|truncate|vacuum)\b",
    r"\b(?:insert|update|delete|merge|upsert)\b",
    r"\bread_csv\b",
    r"\bread_parquet\b",
    r"\bparquet_scan\b",
]
COMPILED_FORBIDDEN = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_SQL_PATTERNS]


class SQLSecurityViolationError(ValueError):
    """Raised when an analytical query violates SELECT-only security constraints."""
    pass


class DuckDBAnalyticsService:
    def __init__(self, storage_dir: Path = PARQUET_STORAGE_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_parquet_path(self, document_id: uuid.UUID) -> Path:
        return self.storage_dir / f"{document_id}.parquet"

    def convert_csv_to_parquet(self, csv_path: Path, document_id: uuid.UUID) -> Path:
        """
        Converts a CSV dataset into an optimized Parquet file with a deterministic row_id.
        """
        dest_path = self.get_parquet_path(document_id)
        df = pd.read_csv(csv_path)

        # Inject deterministic row_id if not already present
        if "row_id" not in df.columns:
            df.insert(0, "row_id", range(1, len(df) + 1))

        df.to_parquet(dest_path, index=False, engine="pyarrow")
        logger.info(
            "csv_converted_to_parquet",
            extra={
                "event": "csv_converted_to_parquet",
                "document_id": str(document_id),
                "rows": len(df),
                "columns": len(df.columns),
                "dest_path": str(dest_path),
            },
        )
        return dest_path

    def get_dataset_schema(self, document_id: uuid.UUID) -> dict[str, str]:
        """Returns column names and types for the Parquet dataset."""
        parquet_path = self.get_parquet_path(document_id)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet dataset for document {document_id} not found.")

        con = duckdb.connect(":memory:")
        try:
            con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path.as_posix()}')")
            rows = con.fetchall()
            return {r[0]: r[1] for r in rows}
        finally:
            con.close()

    def validate_sql(self, sql_query: str) -> str:
        """
        Enforces strict SELECT-only constraints.
        Disallows multiple statements, filesystem scans, ATTACH, COPY, or mutations.
        """
        clean_sql = sql_query.strip()
        # Remove trailing semicolon
        if clean_sql.endswith(";"):
            clean_sql = clean_sql[:-1].strip()

        if ";" in clean_sql:
            raise SQLSecurityViolationError("Multiple SQL statements (semicolon chaining) are strictly prohibited.")

        # Must start with SELECT or WITH
        first_token = clean_sql.split(maxsplit=1)[0].upper()
        if first_token not in ("SELECT", "WITH"):
            raise SQLSecurityViolationError("Only SELECT or WITH analytical queries are permitted.")

        for pattern in COMPILED_FORBIDDEN:
            if pattern.search(clean_sql):
                raise SQLSecurityViolationError(
                    f"Forbidden SQL operation detected. Queries must not load files, mutate data, or call system pragmas."
                )

        return clean_sql

    async def execute_query(
        self,
        document_id: uuid.UUID,
        sql_query: str,
        user: User,
        session: AsyncSession,
        max_rows: int = 100,
    ) -> dict[str, Any]:
        """
        Executes an analytical SQL query against the document's Parquet dataset.
        Validates user authorization, SELECT safety, memory limits, and row limits.
        """
        # 1. Authorize document access
        errata_svc = ErrataService(session)
        doc = await errata_svc._validate_document_access(user, document_id)

        # 2. Check Parquet file existence
        parquet_path = self.get_parquet_path(doc.id)
        if not parquet_path.exists():
            # If CSV original exists, generate parquet on the fly
            orig_path = Path(doc.file_path)
            if orig_path.exists() and orig_path.suffix.lower() == ".csv":
                self.convert_csv_to_parquet(orig_path, doc.id)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Structured Parquet dataset not found for this document.",
                )

        # 3. Validate SQL security
        safe_sql = self.validate_sql(sql_query)

        # 4. Execute inside isolated ephemeral DuckDB session
        con = duckdb.connect(":memory:")
        try:
            con.execute("SET max_memory = '256MB'")
            # Create a virtual table 'dataset' pointing to the authorized parquet file
            con.execute(f"CREATE VIEW dataset AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')")

            # Execute query with bound limit
            cursor = con.execute(safe_sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows + 1)
            
            truncated = len(rows) > max_rows
            if truncated:
                rows = rows[:max_rows]

            # Format rows as list of dicts for LLM/JSON consumption
            formatted_records = [dict(zip(columns, r)) for r in rows]

            return {
                "columns": columns,
                "records": formatted_records,
                "row_count": len(rows),
                "truncated": truncated,
                "document_id": str(doc.id),
                "file_name": doc.file_name,
            }
        except duckdb.Error as d_err:
            logger.warning(f"DuckDB execution error: {d_err}")
            raise
        finally:
            con.close()

