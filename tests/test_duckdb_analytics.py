import os
import tempfile
import uuid
from pathlib import Path
import pytest
from sqlalchemy import select

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.analytics.duckdb_service import DuckDBAnalyticsService, SQLSecurityViolationError
from rag_app.analytics.duckdb_tool import QueryCSVDatasetTool
from rag_app.db.session import get_db_session
from rag_app.db.models import User, Document


@pytest.mark.asyncio
async def test_duckdb_analytics_complete_flow():
    """Verify CSV to Parquet conversion, aggregations, security constraints, and self-healing loop."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("region,sales,reps\nNorth,100,5\nSouth,250,8\nNorth,150,6\nEast,300,10\n")
        sample_csv = Path(f.name)

    try:
        async for session in get_db_session():
            admin = await session.scalar(select(User).where(User.email == "admin_duckdb@local.dev"))
            if not admin:
                admin = User(email="admin_duckdb@local.dev", hashed_password="pw", role="ADMIN", is_active=True)
                session.add(admin)
                await session.flush()

            doc_id = uuid.uuid4()
            doc = Document(
                id=doc_id,
                file_name="sales.csv",
                file_path=str(sample_csv),
                content_hash=uuid.uuid4().hex + uuid.uuid4().hex,
                mime_type="text/csv",
            )
            session.add(doc)
            await session.flush()

            svc = DuckDBAnalyticsService()
            parquet_path = svc.convert_csv_to_parquet(sample_csv, doc_id)
            assert parquet_path.exists()

            # 1. Run aggregation SQL
            res = await svc.execute_query(
                document_id=doc_id,
                sql_query="SELECT region, sum(sales) as total_sales, count(row_id) as orders FROM dataset GROUP BY region ORDER BY total_sales DESC",
                user=admin,
                session=session,
            )

            assert res["row_count"] == 3
            assert res["columns"] == ["region", "total_sales", "orders"]
            records = {r["region"]: r["total_sales"] for r in res["records"]}
            assert records["North"] == 250
            assert records["East"] == 300
            assert records["South"] == 250

            # 2. Security Constraints: DROP, ATTACH, read_csv, and semicolon chaining must fail
            with pytest.raises(SQLSecurityViolationError):
                await svc.execute_query(doc_id, "DROP TABLE dataset", admin, session)

            with pytest.raises(SQLSecurityViolationError):
                await svc.execute_query(doc_id, "ATTACH 'malicious.db'", admin, session)

            with pytest.raises(SQLSecurityViolationError):
                await svc.execute_query(doc_id, "SELECT * FROM read_csv('/etc/passwd')", admin, session)

            with pytest.raises(SQLSecurityViolationError):
                await svc.execute_query(doc_id, "SELECT 1; DROP TABLE dataset;", admin, session)

            # 3. Tool with Self-Healing Loop
            heal_called = False

            async def mock_llm_heal(bad_sql: str, error: str, schema: dict[str, str]) -> str:
                nonlocal heal_called
                heal_called = True
                assert "non_existent_col" in bad_sql or "error" in error.lower()
                return "SELECT region, sum(sales) as total FROM dataset GROUP BY region"

            tool = QueryCSVDatasetTool(
                session=session,
                user=admin,
                llm_heal_fn=mock_llm_heal,
            )

            # Query with invalid column name triggers DuckDB compiler error -> self-healing
            bad_query = "SELECT non_existent_col, sum(sales) FROM dataset GROUP BY region"
            output = await tool._arun(document_id=str(doc_id), sql_query=bad_query)

            assert heal_called is True
            assert "total" in output
            assert "North" in output

            from rag_app.db.session import engine
            await engine.dispose()
            break
    finally:
        if sample_csv.exists():
            sample_csv.unlink()
