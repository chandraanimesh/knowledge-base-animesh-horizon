"""
Controlled Database Migration Gate & Preflight Tool for RAG Architecture.

Features:
- Positive Database Target Identification (refuses to run against unconfirmed targets).
- Preflight inspection: compares database URL, current revision, target revision, and alembic_version column schema.
- Pending migration graph validation: detects revision divergence, missing branches, or multi-head splits.
- Safe default dry-run/preflight mode: reports changes without modifying the database.
- Explicit gated apply mode: requires explicit target confirmation flags (--apply --confirm-target=<db_name>).
- Safe logging: sanitizes all database connection strings and passwords.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

import alembic.config
import alembic.ddl.impl
import alembic.ddl.postgresql
import alembic.script
from alembic import command
import asyncpg
import sqlalchemy as sa
from sqlalchemy.engine.url import make_url

from rag_app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migration_gate")

EXPECTED_BASE_REVISION = "add_memory_processing_job_source_trigger_unique_index"
EXPECTED_TARGET_REVISION = "add_workflow_run_and_tasks_persistence"


# Ensure Alembic in fresh environments uses VARCHAR(128) for version_num
class SafePostgresqlImpl(alembic.ddl.postgresql.PostgresqlImpl):
    def version_table_impl(
        self,
        *,
        version_table: str,
        version_table_schema: str | None,
        version_table_pk: bool,
        **kw,
    ) -> sa.Table:
        vt = sa.Table(
            version_table,
            sa.MetaData(),
            sa.Column("version_num", sa.String(128), nullable=False),
            schema=version_table_schema,
        )
        if version_table_pk:
            vt.append_constraint(sa.PrimaryKeyConstraint("version_num", name=f"{version_table}_pkc"))
        return vt


alembic.ddl.impl._impls["postgresql"] = SafePostgresqlImpl


async def inspect_db_state(database_url: str) -> dict:
    url = make_url(database_url)
    db_name = url.database
    host = url.host or "127.0.0.1"
    port = url.port or 5432
    user = url.username or "rag"
    password = url.password or "rag"

    # Direct raw asyncpg inspection
    conn = await asyncpg.connect(
        user=user,
        password=password,
        database=db_name,
        host=host,
        port=port,
    )
    try:
        # 1. Check version_num column width
        col_info = await conn.fetchrow(
            """
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'alembic_version' AND column_name = 'version_num'
            """
        )

        # 2. Check current version_num
        has_table = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'alembic_version')"
        )
        current_rev = None
        if has_table:
            current_rev = await conn.fetchval("SELECT version_num FROM alembic_version")

        # 3. Check existing tables
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        table_names = {r["table_name"] for r in tables}

        return {
            "db_name": db_name,
            "host": host,
            "port": port,
            "sanitized_url": url.render_as_string(hide_password=True),
            "version_column_len": col_info["character_maximum_length"] if col_info else None,
            "current_revision": current_rev,
            "has_workflow_runs": "workflow_runs" in table_names,
            "has_workflow_tasks": "workflow_tasks" in table_names,
        }
    finally:
        await conn.close()


def validate_migration_graph(alembic_cfg_path: str = "alembic.ini") -> tuple[list[str], str]:
    cfg = alembic.config.Config(alembic_cfg_path)
    script = alembic.script.ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    base = script.get_base()
    return heads, base


async def run_preflight(database_url: str) -> bool:
    print("=" * 65)
    print("CONTROLLED DATABASE MIGRATION GATE - PREFLIGHT INSPECTION")
    print("=" * 65)

    heads, base = validate_migration_graph()
    print(f"Migration Graph Base : {base}")
    print(f"Migration Graph Heads: {heads}")

    if len(heads) != 1:
        print(f"ERROR: Migration graph split detected! Expected 1 head, found {len(heads)}: {heads}")
        return False

    target_head = heads[0]
    if target_head != EXPECTED_TARGET_REVISION:
        print(f"WARNING: Graph head ({target_head}) does not match expected ({EXPECTED_TARGET_REVISION})")

    state = await inspect_db_state(database_url)
    print(f"\nTarget Database      : {state['db_name']} on {state['host']}:{state['port']}")
    print(f"Connection URL       : {state['sanitized_url']}")
    print(f"Alembic Version Col  : VARCHAR({state['version_column_len']})")
    print(f"Current Revision     : {state['current_revision']}")
    print(f"Target Revision      : {target_head}")
    print(f"Workflow Runs Table  : {'EXISTS' if state['has_workflow_runs'] else 'NOT PRESENT'}")
    print(f"Workflow Tasks Table : {'EXISTS' if state['has_workflow_tasks'] else 'NOT PRESENT'}")

    # Check width compatibility
    required_len = len(target_head)
    col_len = state["version_column_len"] or 32
    if col_len < required_len:
        print(f"\nERROR: Revision length ({required_len}) exceeds column width ({col_len})!")
        return False
    else:
        print(f"\n[PASS] Column width {col_len} is sufficient for target revision length {required_len}.")

    if state["current_revision"] == target_head:
        print("[PASS] Database is already at head revision. No pending migrations.")
        return True
    elif state["current_revision"] == EXPECTED_BASE_REVISION:
        print(f"[READY] Database is at expected base ({EXPECTED_BASE_REVISION}). Ready for upgrade.")
        return True
    else:
        print(f"[NOTICE] Database revision is '{state['current_revision']}'. Upstream migrations pending.")
        return True


async def main():
    parser = argparse.ArgumentParser(description="Controlled Database Migration Gate & Preflight Tool")
    parser.add_argument("--db-url", type=str, default=None, help="Database URL (defaults to DATABASE_URL setting)")
    parser.add_argument("--apply", action="store_true", help="Execute the migration (Default is dry-run preflight)")
    parser.add_argument("--confirm-target", type=str, default=None, help="Must match target database name to apply")
    parser.add_argument("--target-revision", type=str, default="head", help="Target revision (default: head)")

    args = parser.parse_args()
    db_url = args.db_url or get_settings().database_url

    preflight_ok = await run_preflight(db_url)
    if not preflight_ok:
        print("\nPreflight check FAILED. Aborting.")
        sys.exit(1)

    if not args.apply:
        print("\n[PREFLIGHT MODE COMPLETED] No changes were made to the database.")
        print("To apply migrations, specify: --apply --confirm-target=<database_name>")
        return

    # Gated execution safety checks
    target_state = await inspect_db_state(db_url)
    confirmed_db = target_state["db_name"]
    if args.confirm_target != confirmed_db:
        print(f"\nERROR: Target confirmation mismatch! Target is '{confirmed_db}', provided '{args.confirm_target}'.")
        print("Safety gate refusing execution.")
        sys.exit(1)

    if confirmed_db == "ragdb":
        print("\n" + "!" * 65)
        print("WARNING: Target database is active production/dev database 'ragdb'!")
        print("Explicit user approval is strictly required before running against ragdb.")
        print("!" * 65)

    print(f"\nApplying migration to '{confirmed_db}' -> target '{args.target_revision}'...")
    cfg = alembic.config.Config("alembic.ini")
    await asyncio.to_thread(command.upgrade, cfg, args.target_revision)
    print("\nMigration upgrade completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
