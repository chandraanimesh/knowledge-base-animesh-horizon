"""
Admin Document Management domain tools for FastMCP.
Exposes:
- admin_update_document_department
- admin_get_document_permissions
- admin_grant_document_permission
- admin_revoke_document_permission
- admin_set_document_permissions

All tools require MCP scope 'admin.documents' and administrative privilege.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional
from fastapi import HTTPException
from sqlalchemy import and_, delete, exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import (
    Department,
    Document,
    DocumentChunk,
    DocumentPermission,
    KnowledgeEvidence,
    KnowledgeItem,
    Permission,
    User,
    UserDepartment,
    UserPermission,
)
from rag_app.mcp.gateway import mcp_gateway


# ── Underlying Handlers ───────────────────────────────────────────────────────

async def handle_admin_update_document_department(
    document_id: str,
    department_code: str,
    session: AsyncSession,
) -> dict[str, Any]:
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid document UUID: {document_id}")

    doc = await session.get(Document, doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    code = department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == code))
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{code}' not found.")
    if not dept.is_active:
        raise HTTPException(status_code=400, detail=f"Department '{code}' is inactive.")

    doc.department_id = dept.id

    chunk_subq = select(DocumentChunk.id).where(DocumentChunk.document_id == doc.id)
    ki_subq = select(KnowledgeEvidence.knowledge_item_id).where(
        KnowledgeEvidence.document_chunk_id.in_(chunk_subq)
    )
    async with session.begin_nested():
        try:
            await session.execute(
                update(KnowledgeItem)
                .where(KnowledgeItem.id.in_(ki_subq))
                .values(department_id=dept.id)
            )
        except Exception:
            pass

    await session.commit()
    await session.refresh(doc)

    return {
        "id": str(doc.id),
        "file_name": doc.file_name,
        "department_id": str(doc.department_id),
        "department_code": dept.code,
        "department_name": dept.name,
        "status": doc.status,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


async def handle_admin_get_document_permissions(
    document_id: str,
    session: AsyncSession,
) -> dict[str, Any]:
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid document UUID: {document_id}")

    doc = await session.get(Document, doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    dept = await session.get(Department, doc.department_id) if doc.department_id else None
    if not dept:
        raise HTTPException(status_code=404, detail="Document department not found.")

    users_stmt = (
        select(User)
        .join(UserDepartment, User.id == UserDepartment.user_id)
        .where(UserDepartment.department_id == dept.id, User.is_active == True)
        .order_by(User.email)
    )
    dept_users = (await session.scalars(users_stmt)).all()

    dept_perms_stmt = (
        select(UserPermission.user_id, Permission.code)
        .join(Permission, UserPermission.permission_id == Permission.id)
        .where(UserPermission.department_id == dept.id, Permission.is_active == True)
    )
    dept_perm_rows = (await session.execute(dept_perms_stmt)).all()
    dept_perm_map: dict[uuid.UUID, set[str]] = {}
    for uid, pcode in dept_perm_rows:
        dept_perm_map.setdefault(uid, set()).add(pcode)

    doc_perms_stmt = (
        select(DocumentPermission.user_id, Permission.code)
        .join(Permission, DocumentPermission.permission_id == Permission.id)
        .where(DocumentPermission.document_id == doc.id, Permission.is_active == True)
    )
    doc_perm_rows = (await session.execute(doc_perms_stmt)).all()
    doc_perm_map: dict[uuid.UUID, set[str]] = {}
    for uid, pcode in doc_perm_rows:
        doc_perm_map.setdefault(uid, set()).add(pcode)

    user_items = []
    for u in dept_users:
        dept_p = sorted(dept_perm_map.get(u.id, set()))
        doc_p = sorted(doc_perm_map.get(u.id, set()))
        eff_p = sorted(set(dept_p) | set(doc_p))
        user_items.append(
            {
                "user_id": str(u.id),
                "email": u.email,
                "full_name": u.email.split("@")[0].capitalize(),
                "department_permissions": dept_p,
                "document_permissions": doc_p,
                "effective_permissions": eff_p,
            }
        )

    return {
        "document_id": str(doc.id),
        "file_name": doc.file_name,
        "department_id": str(dept.id),
        "department_code": dept.code,
        "department_name": dept.name,
        "users": user_items,
    }


async def handle_admin_grant_document_permission(
    document_id: str,
    user_id: str,
    permission_code: str,
    session: AsyncSession,
) -> dict[str, str]:
    try:
        doc_uuid = uuid.UUID(document_id)
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document or user UUID.")

    doc = await session.get(Document, doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    dept = await session.get(Department, doc.department_id) if doc.department_id else None
    if not dept:
        raise HTTPException(status_code=404, detail="Document department not found.")
    if not dept.is_active:
        raise HTTPException(status_code=400, detail=f"Department '{dept.code}' is inactive.")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="User is inactive.")

    is_member = await session.scalar(
        select(
            exists().where(
                and_(
                    UserDepartment.user_id == user.id,
                    UserDepartment.department_id == dept.id,
                )
            )
        )
    )
    if not is_member:
        raise HTTPException(
            status_code=400,
            detail="User is not a member of the document's department. Cannot grant document permissions to non-members.",
        )

    pcode = permission_code.strip().upper()
    perm = await session.scalar(
        select(Permission).where(Permission.code == pcode, Permission.is_active == True)
    )
    if not perm:
        raise HTTPException(status_code=400, detail=f"Invalid or inactive permission code '{pcode}'.")

    existing = await session.scalar(
        select(DocumentPermission).where(
            DocumentPermission.document_id == doc.id,
            DocumentPermission.user_id == user.id,
            DocumentPermission.permission_id == perm.id,
        )
    )
    if not existing:
        session.add(
            DocumentPermission(
                document_id=doc.id,
                user_id=user.id,
                permission_id=perm.id,
            )
        )
        await session.commit()

    return {"message": f"Permission {pcode} granted to user {user.email} for document {doc.file_name}"}


async def handle_admin_revoke_document_permission(
    document_id: str,
    user_id: str,
    permission_code: str,
    session: AsyncSession,
) -> dict[str, str]:
    try:
        doc_uuid = uuid.UUID(document_id)
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document or user UUID.")

    doc = await session.get(Document, doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    pcode = permission_code.strip().upper()
    perm = await session.scalar(
        select(Permission).where(Permission.code == pcode, Permission.is_active == True)
    )
    if not perm:
        raise HTTPException(status_code=400, detail=f"Invalid or inactive permission code '{pcode}'.")

    await session.execute(
        delete(DocumentPermission).where(
            DocumentPermission.document_id == doc.id,
            DocumentPermission.user_id == uid,
            DocumentPermission.permission_id == perm.id,
        )
    )
    await session.commit()
    return {"message": f"Permission {pcode} revoked for user on document {doc.file_name}"}


async def handle_admin_set_document_permissions(
    document_id: str,
    user_id: str,
    permissions: list[str],
    session: AsyncSession,
) -> dict[str, str]:
    try:
        doc_uuid = uuid.UUID(document_id)
        uid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document or user UUID.")

    doc = await session.get(Document, doc_uuid)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    dept = await session.get(Department, doc.department_id) if doc.department_id else None
    if not dept:
        raise HTTPException(status_code=404, detail="Document department not found.")
    if not dept.is_active:
        raise HTTPException(status_code=400, detail=f"Department '{dept.code}' is inactive.")

    user = await session.get(User, uid)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="User is inactive.")

    is_member = await session.scalar(
        select(
            exists().where(
                and_(
                    UserDepartment.user_id == user.id,
                    UserDepartment.department_id == dept.id,
                )
            )
        )
    )
    if not is_member:
        raise HTTPException(
            status_code=400,
            detail="User is not a member of the document's department. Cannot grant document permissions to non-members.",
        )

    target_perm_codes = [p.strip().upper() for p in permissions]
    perm_records = []
    if target_perm_codes:
        perm_records = (
            await session.scalars(
                select(Permission).where(
                    Permission.code.in_(target_perm_codes),
                    Permission.is_active == True,
                )
            )
        ).all()
        found_codes = {p.code for p in perm_records}
        missing_codes = set(target_perm_codes) - found_codes
        if missing_codes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or inactive permissions: {', '.join(missing_codes)}",
            )

    await session.execute(
        delete(DocumentPermission).where(
            DocumentPermission.document_id == doc.id,
            DocumentPermission.user_id == user.id,
        )
    )

    for p in perm_records:
        session.add(
            DocumentPermission(
                document_id=doc.id,
                user_id=user.id,
                permission_id=p.id,
            )
        )

    await session.commit()
    return {"message": "Document permissions updated successfully"}


# ── MCP Tool Callers ─────────────────────────────────────────────────────────

async def admin_update_document_department(
    document_id: str,
    department_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "admin_update_document_department",
        handle_admin_update_document_department,
        document_id=document_id,
        department_code=department_code,
        auth_token=auth_token,
    )


async def admin_get_document_permissions(
    document_id: str,
    auth_token: Optional[str] = None,
) -> dict[str, Any]:
    return await mcp_gateway.execute(
        "admin_get_document_permissions",
        handle_admin_get_document_permissions,
        document_id=document_id,
        auth_token=auth_token,
    )


async def admin_grant_document_permission(
    document_id: str,
    user_id: str,
    permission_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "admin_grant_document_permission",
        handle_admin_grant_document_permission,
        document_id=document_id,
        user_id=user_id,
        permission_code=permission_code,
        auth_token=auth_token,
    )


async def admin_revoke_document_permission(
    document_id: str,
    user_id: str,
    permission_code: str,
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "admin_revoke_document_permission",
        handle_admin_revoke_document_permission,
        document_id=document_id,
        user_id=user_id,
        permission_code=permission_code,
        auth_token=auth_token,
    )


async def admin_set_document_permissions(
    document_id: str,
    user_id: str,
    permissions: list[str],
    auth_token: Optional[str] = None,
) -> dict[str, str]:
    return await mcp_gateway.execute(
        "admin_set_document_permissions",
        handle_admin_set_document_permissions,
        document_id=document_id,
        user_id=user_id,
        permissions=permissions,
        auth_token=auth_token,
    )
