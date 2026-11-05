"""LangGraph Tool: query_csv_dataset

Enables analytical SQL queries over structured tabular datasets (Parquet/DuckDB)
with bounded self-healing loop for syntax and schema errors (up to 3 retries).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Coroutine, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.analytics.duckdb_service import DuckDBAnalyticsService, SQLSecurityViolationError
from rag_app.db.models import User

logger = logging.getLogger(__name__)


class QueryCSVDatasetInput(BaseModel):
    document_id: str = Field(..., description="UUID of the structured CSV/Parquet document to query")
    sql_query: str = Field(..., description="Analytical SQL query to run against the 'dataset' view (e.g. SELECT region, sum(sales) FROM dataset GROUP BY region)")


class QueryCSVDatasetTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "query_csv_dataset"
    description: str = (
        "Executes a fast analytical SQL query over a tabular CSV/Parquet dataset using DuckDB. "
        "The table name is always 'dataset'. Supports aggregation (SUM, AVG, COUNT, MIN, MAX), "
        "GROUP BY, ORDER BY, and WHERE clauses. Only SELECT queries are permitted."
    )
    args_schema: Type[BaseModel] = QueryCSVDatasetInput

    session: AsyncSession
    user: User
    llm_heal_fn: Callable[[str, str, dict[str, str]], Coroutine[Any, Any, str]] | None = None
    max_heal_retries: int = 3

    async def _arun(
        self,
        document_id: str,
        sql_query: str,
    ) -> str:
        try:
            doc_uuid = uuid.UUID(str(document_id).strip())
        except (ValueError, TypeError):
            return f"Error: Invalid document UUID '{document_id}'."

        service = DuckDBAnalyticsService()
        current_sql = sql_query.strip()

        # Bounded self-healing execution loop (max 3 retries)
        last_error = ""
        for attempt in range(self.max_heal_retries + 1):
            try:
                res = await service.execute_query(
                    document_id=doc_uuid,
                    sql_query=current_sql,
                    user=self.user,
                    session=self.session,
                )
                return json.dumps(res, default=str)
            except SQLSecurityViolationError as sec_err:
                # Security violations fail immediately without healing
                logger.warning(f"SQL security rejection: {sec_err}")
                return f"SQL Security Error: {sec_err}"
            except Exception as exc:
                last_error = str(exc)
                logger.info(
                    f"query_csv_dataset error on attempt {attempt + 1}: {exc}. "
                    f"Attempting self-healing..."
                )
                if attempt >= self.max_heal_retries or not self.llm_heal_fn:
                    break

                try:
                    schema = service.get_dataset_schema(doc_uuid)
                except Exception:
                    schema = {}

                # Feed error and schema back to LLM self-healing function
                try:
                    healed_sql = await self.llm_heal_fn(current_sql, last_error, schema)
                    if healed_sql and healed_sql.strip() != current_sql:
                        current_sql = healed_sql.strip()
                        continue
                    else:
                        break
                except Exception as heal_err:
                    logger.warning(f"Self-healing generation failed: {heal_err}")
                    break

        return (
            f"Execution failed after {attempt + 1} attempt(s). "
            f"Last error: {last_error}. "
            f"Please verify column names and SQL syntax."
        )

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use async _arun.")

