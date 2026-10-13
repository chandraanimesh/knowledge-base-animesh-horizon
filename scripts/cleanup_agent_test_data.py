"""
Audited cleanup script for confirmed agent-created mock departments and synthetic documents.

Removes:
1. Synthetic documents created by MCP RBAC / interface regression tests (mcp_*).
2. Mock departments created by personal memory regression tests (MD_*).
3. Associated mock workspaces (WS1_*, WS2_*) and test fixture users (u1_*, u2_*).

Preserves:
- All genuine documents (Sales report, Taber Connect spec, Review UI, Twenty CRM doc, OV2 plan).
- All genuine departments (Sales, Finance, HR, Engineering, Unclassified).
- All genuine users, notes, knowledge items, and evidence references.
- All genuine user conversations, messages, and workspaces.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from rag_app.core.config import get_settings
from rag_app.db.models import (
    ChatMessage,
    Conversation,
    Department,
    Document,
    DocumentChunk,
    DocumentSection,
    KnowledgeEvidence,
    User,
    Workspace,
    WorkspaceMember,
)
from rag_app.db.session import AsyncSessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def run_cleanup(dry_run: bool = True) -> dict[str, Any]:
    settings = get_settings()
    # 1. Confirm Environment Safety
    if settings.env != "dev":
        raise RuntimeError(
            f"Safety check failed: Cleanup can only be executed in 'dev' environment, current env is '{settings.env}'"
        )

    logger.info("Environment: %s | Database: %s", settings.env, settings.database_url.split("@")[-1] if "@" in settings.database_url else "configured")
    mode_str = "DRY-RUN" if dry_run else "EXECUTE"
    logger.info("Starting cleanup in mode: %s", mode_str)

    async with AsyncSessionLocal() as session:
        # =================================================================
        # 1. Audit Synthetic Documents
        # =================================================================
        stmt_mcp_docs = select(Document).where(Document.file_name.startswith("mcp_"))
        mcp_docs = (await session.execute(stmt_mcp_docs)).scalars().all()
        mcp_doc_ids = [d.id for d in mcp_docs]

        # Verify that none of these synthetic documents are referenced by KnowledgeEvidence
        ev_count = 0
        if mcp_doc_ids:
            stmt_ev = (
                select(KnowledgeEvidence)
                .join(DocumentChunk, KnowledgeEvidence.document_chunk_id == DocumentChunk.id)
                .where(DocumentChunk.document_id.in_(mcp_doc_ids))
            )
            ev_count = len((await session.execute(stmt_ev)).scalars().all())

        if ev_count > 0:
            raise RuntimeError(
                f"Safety check failed: {ev_count} knowledge evidence records reference mcp documents!"
            )

        mcp_inventory = [
            {
                "id": str(d.id),
                "file_name": d.file_name,
                "department_id": str(d.department_id) if d.department_id else None,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "provenance": "tests/test_mcp_*.py synthetic document ingest",
            }
            for d in mcp_docs
        ]

        # Check physical files for synthetic documents
        upload_dir = Path("uploaded_docs")
        deleted_physical_files = []
        for item in mcp_inventory:
            fpath = upload_dir / item["file_name"]
            if fpath.exists():
                deleted_physical_files.append(str(fpath))

        # =================================================================
        # 2. Audit Mock Departments (MD_*)
        # =================================================================
        stmt_md_depts = select(Department).where(Department.code.startswith("MD_"))
        md_depts = (await session.execute(stmt_md_depts)).scalars().all()
        md_dept_ids = [d.id for d in md_depts]

        md_dept_inventory = [
            {
                "id": str(d.id),
                "code": d.code,
                "name": d.name,
                "created_at": d.created_at.isoformat() if getattr(d, "created_at", None) else None,
                "provenance": "tests/test_personal_memory_lifecycle.py::mem_env fixture",
            }
            for d in md_depts
        ]

        # Associated mock workspaces
        stmt_ws = select(Workspace).where(Workspace.department_id.in_(md_dept_ids))
        md_workspaces = (await session.execute(stmt_ws)).scalars().all()
        ws_ids = [w.id for w in md_workspaces]

        ws_inventory = [
            {
                "id": str(w.id),
                "name": w.name,
                "code": w.code,
                "department_id": str(w.department_id),
                "provenance": "tests/test_personal_memory_lifecycle.py::mem_env fixture",
            }
            for w in md_workspaces
        ]

        # Audit conversations in mock workspaces
        conv_ids = []
        if ws_ids:
            stmt_conv_ids = select(Conversation.id).where(Conversation.workspace_id.in_(ws_ids))
            conv_ids = (await session.execute(stmt_conv_ids)).scalars().all()

        # Audit fixture test users u1_* and u2_*
        stmt_test_users = select(User).where(User.email.regexp_match(r"^u[12]_[0-9a-f]{8}@example\.com$"))
        test_users = (await session.execute(stmt_test_users)).scalars().all()
        test_user_inventory = [
            {
                "id": str(u.id),
                "email": u.email,
                "role": u.role,
                "provenance": "tests/test_personal_memory_lifecycle.py::mem_env fixture",
            }
            for u in test_users
        ]

        # Verify no genuine documents attached to mock departments
        if md_dept_ids:
            dept_docs = (await session.execute(select(Document).where(Document.department_id.in_(md_dept_ids)))).scalars().all()
            if dept_docs:
                raise RuntimeError(
                    f"Safety check failed: {len(dept_docs)} documents attached to candidate mock departments!"
                )

        if not dry_run:
            try:
                # 1. Delete synthetic documents if any
                for doc in mcp_docs:
                    await session.delete(doc)
                await session.flush()

                # Delete physical files
                for fpath_str in deleted_physical_files:
                    p = Path(fpath_str)
                    if p.exists():
                        p.unlink()

                # 2. Delete conversations and messages in mock workspaces
                if conv_ids:
                    await session.execute(delete(ChatMessage).where(ChatMessage.conversation_id.in_(conv_ids)))
                    await session.execute(delete(Conversation).where(Conversation.id.in_(conv_ids)))

                # 3. Delete workspace members and workspaces
                if ws_ids:
                    await session.execute(delete(WorkspaceMember).where(WorkspaceMember.workspace_id.in_(ws_ids)))
                    await session.execute(delete(Workspace).where(Workspace.id.in_(ws_ids)))
                await session.flush()

                # 4. Delete mock departments
                if md_dept_ids:
                    await session.execute(delete(Department).where(Department.id.in_(md_dept_ids)))
                await session.flush()

                # 5. Delete fixture test users u1_* and u2_*
                if test_users:
                    user_ids = [u.id for u in test_users]
                    await session.execute(delete(User).where(User.id.in_(user_ids)))
                await session.flush()

                await session.commit()
                logger.info("Database transaction committed successfully.")
            except Exception:
                await session.rollback()
                logger.exception("Cleanup failed, transaction rolled back.")
                raise

        # =================================================================
        # 3. Verify Genuine State
        # =================================================================
        remaining_docs = (await session.execute(select(Document))).scalars().all()
        remaining_depts = (await session.execute(select(Department))).scalars().all()
        remaining_workspaces = (await session.execute(select(Workspace))).scalars().all()
        remaining_convs = (await session.execute(select(Conversation))).scalars().all()

        return {
            "mode": mode_str,
            "target_env": settings.env,
            "target_database": settings.database_url.split("@")[-1] if "@" in settings.database_url else "configured",
            "proposed_synthetic_documents_count": len(mcp_inventory),
            "proposed_synthetic_documents": mcp_inventory,
            "proposed_physical_files": deleted_physical_files,
            "proposed_mock_departments_count": len(md_dept_inventory),
            "proposed_mock_departments": md_dept_inventory,
            "proposed_mock_workspaces_count": len(ws_inventory),
            "proposed_mock_workspaces": ws_inventory,
            "proposed_mock_users_count": len(test_user_inventory),
            "proposed_mock_users": test_user_inventory,
            "proposed_conversations_count": len(conv_ids),
            "remaining_genuine_documents_count": len(remaining_docs),
            "remaining_documents": [
                {"id": str(d.id), "file_name": d.file_name, "dept_id": str(d.department_id)}
                for d in remaining_docs
            ],
            "remaining_departments_count": len(remaining_depts),
            "remaining_departments": [
                {"code": d.code, "name": d.name, "active": d.is_active}
                for d in remaining_depts
            ],
            "remaining_workspaces_count": len(remaining_workspaces),
            "remaining_conversations_count": len(remaining_convs),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup agent regression test data.")
    parser.add_argument("--execute", action="store_true", help="Execute deletions (defaults to dry-run)")
    args = parser.parse_args()

    is_dry_run = not args.execute
    result = asyncio.run(run_cleanup(dry_run=is_dry_run))

    print(f"\n==========================================")
    print(f" CLEANUP REPORT [{result['mode']}]")
    print(f"==========================================")
    print(f"Environment: {result['target_env']}")
    print(f"Database: {result['target_database']}")
    print(f"Proposed Synthetic Documents to remove: {result['proposed_synthetic_documents_count']}")
    print(f"Proposed Mock Departments to remove:    {result['proposed_mock_departments_count']}")
    print(f"Proposed Mock Workspaces to remove:      {result['proposed_mock_workspaces_count']}")
    print(f"Proposed Mock Users to remove:           {result['proposed_mock_users_count']}")
    print(f"Proposed Conversations to remove:        {result['proposed_conversations_count']}")
    print(f"------------------------------------------")
    print(f"Remaining Genuine Documents: {result['remaining_genuine_documents_count']}")
    for d in result["remaining_documents"]:
        print(f"  - {d['file_name']} (id={d['id']})")
    print(f"Remaining Departments: {result['remaining_departments_count']}")
    for dept in result["remaining_departments"]:
        print(f"  - {dept['code']}: {dept['name']}")
    print(f"Remaining Genuine Workspaces: {result['remaining_workspaces_count']}")
    print(f"Remaining Genuine Conversations: {result['remaining_conversations_count']}")
    print(f"==========================================\n")
