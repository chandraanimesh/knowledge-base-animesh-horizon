from typing import Annotated, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from rag_app.db.session import get_db_session
from rag_app.db.models import User, Department, UserDepartment, Permission, UserPermission, Document
from rag_app.api.dependencies.auth import require_admin
from rag_app.schemas.users import (
    UserCreate, UserResponse, UserStatusUpdate, DepartmentResponse, PermissionResponse,
    PresetRequest, PRESET_PERMISSIONS, PermissionPreset, UserAccessSummary, DeptAccessSummary,
    DocumentDepartmentUpdate, DocumentAdminResponse
)
from rag_app.security.authentication import get_password_hash

router = APIRouter(prefix="/admin/users", tags=["admin-users"])
DbSession = Annotated[AsyncSession, Depends(get_db_session)]

@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    session: DbSession,
    _: User = Depends(require_admin)
):
    normalized_email = payload.email.lower().strip()
    
    # Check for existing email to return a clean 409
    existing = await session.scalar(select(User).where(User.email == normalized_email))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists."
        )

    dept = None
    perms_to_grant = []
    if payload.role == "USER" and payload.department_code:
        dept_code = payload.department_code.strip().upper()
        dept = await session.scalar(select(Department).where(Department.code == dept_code))
        if not dept:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Department '{dept_code}' not found."
            )
        if not dept.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Department '{dept_code}' is inactive."
            )
        
        if payload.permissions:
            requested_perm_codes = list(dict.fromkeys(payload.permissions))
            perm_records = (await session.scalars(
                select(Permission).where(
                    Permission.code.in_(requested_perm_codes),
                    Permission.is_active == True,
                )
            )).all()
            if len(perm_records) != len(requested_perm_codes):
                found_codes = {p.code for p in perm_records}
                missing_codes = [c for c in requested_perm_codes if c not in found_codes]
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid or inactive permissions: {missing_codes}"
                )
            perms_to_grant = perm_records

    new_user = User(
        email=normalized_email,
        hashed_password=get_password_hash(payload.password),
        role=payload.role,
        is_active=True
    )
    session.add(new_user)
    
    try:
        await session.flush()
        if dept:
            session.add(UserDepartment(user_id=new_user.id, department_id=dept.id))
            for perm in perms_to_grant:
                session.add(UserPermission(user_id=new_user.id, department_id=dept.id, permission_id=perm.id))
        
        await session.commit()
        await session.refresh(new_user)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists or constraint violation."
        )
    except Exception:
        await session.rollback()
        raise
        
    return new_user

@router.get("", response_model=List[UserResponse])
async def list_users(
    session: DbSession,
    _: User = Depends(require_admin)
):
    result = await session.scalars(select(User).order_by(User.created_at.desc()))
    return result.all()


@router.get("/with-access", response_model=list[UserAccessSummary])
async def list_users_with_access(
    session: DbSession,
    _: User = Depends(require_admin)
):
    """Return every user enriched with their department + permission info."""
    users = (await session.scalars(select(User).order_by(User.created_at.desc()))).all()

    summaries: list[UserAccessSummary] = []
    for u in users:
        dept_access: list[DeptAccessSummary] = []
        if u.role == "USER":
            dept_rows = await session.scalars(
                select(Department)
                .join(UserDepartment, Department.id == UserDepartment.department_id)
                .where(UserDepartment.user_id == u.id)
            )
            for d in dept_rows.all():
                perm_rows = await session.scalars(
                    select(Permission.code)
                    .join(UserPermission, Permission.id == UserPermission.permission_id)
                    .where(
                        UserPermission.user_id == u.id,
                        UserPermission.department_id == d.id
                    )
                )
                dept_access.append(DeptAccessSummary(
                    code=d.code, name=d.name, permissions=list(perm_rows.all())
                ))
        summaries.append(UserAccessSummary(
            id=u.id, email=u.email, role=u.role,
            is_active=u.is_active, created_at=u.created_at,
            departments=dept_access
        ))
    return summaries


@router.get("/departments", response_model=list[DepartmentResponse])
async def list_all_departments(
    session: DbSession,
    _: User = Depends(require_admin)
):
    """Return all active departments for the admin UI dropdowns."""
    result = await session.scalars(
        select(Department).where(Department.is_active == True).order_by(Department.code)
    )
    return result.all()

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    session: DbSession,
    _: User = Depends(require_admin)
):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )
    return user

@router.patch("/{user_id}/status", response_model=UserResponse)
async def update_user_status(
    user_id: UUID,
    payload: UserStatusUpdate,
    session: DbSession,
    _: User = Depends(require_admin)
):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )
        
    if payload.is_active is False and user.role == "ADMIN" and user.is_active is True:
        # Prevent deactivating the last active ADMIN
        count = await session.scalar(
            select(func.count(User.id)).where(User.role == "ADMIN", User.is_active == True)
        )
        if count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate the last active ADMIN."
            )
            
    user.is_active = payload.is_active
    await session.commit()
    await session.refresh(user)
    return user

@router.get("/{user_id}/departments", response_model=List[DepartmentResponse])
async def list_user_departments(
    user_id: UUID,
    session: DbSession,
    _: User = Depends(require_admin)
):
    # Verify user exists
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )
    
    # Query departments joined through user_departments
    stmt = (
        select(Department)
        .join(UserDepartment, Department.id == UserDepartment.department_id)
        .where(UserDepartment.user_id == user_id)
    )
    result = await session.scalars(stmt)
    return result.all()

@router.post("/{user_id}/departments/{department_code}", status_code=status.HTTP_201_CREATED)
async def assign_user_department(
    user_id: UUID,
    department_code: str,
    session: DbSession,
    _: User = Depends(require_admin)
):
    # Verify user exists
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        
    # Verify department exists and is active
    dept = await session.scalar(select(Department).where(Department.code == department_code))
    if not dept:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Department not found.")
        
    if not dept.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot assign inactive department.")
        
    # Add membership
    user_dept = UserDepartment(user_id=user.id, department_id=dept.id)
    session.add(user_dept)
    
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already belongs to this department.")
        
    return {"message": "Department assigned successfully"}

@router.delete("/{user_id}/departments/{department_code}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_user_department(
    user_id: UUID,
    department_code: str,
    session: DbSession,
    _: User = Depends(require_admin)
):
    # Verify user exists
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        
    # Verify department exists
    dept = await session.scalar(select(Department).where(Department.code == department_code))
    if not dept:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Department not found.")
        
    # Remove membership (cascade will handle user_permissions)
    stmt = delete(UserDepartment).where(
        UserDepartment.user_id == user.id,
        UserDepartment.department_id == dept.id
    )
    result = await session.execute(stmt)
    await session.commit()
    
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User is not a member of this department.")

@router.get("/{user_id}/departments/{department_code}/permissions", response_model=List[PermissionResponse])
async def list_user_permissions(
    user_id: UUID,
    department_code: str,
    session: DbSession,
    _: User = Depends(require_admin)
):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        
    dept = await session.scalar(select(Department).where(Department.code == department_code))
    if not dept:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Department not found.")
        
    # Check if user belongs to department
    is_member = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == user_id,
            UserDepartment.department_id == dept.id
        )
    )
    if not is_member:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User is not a member of this department.")
        
    stmt = (
        select(Permission)
        .join(UserPermission, Permission.id == UserPermission.permission_id)
        .where(
            UserPermission.user_id == user_id,
            UserPermission.department_id == dept.id
        )
    )
    result = await session.scalars(stmt)
    return result.all()

@router.post("/{user_id}/departments/{department_code}/permissions/{permission_code}", status_code=status.HTTP_201_CREATED)
async def grant_user_permission(
    user_id: UUID,
    department_code: str,
    permission_code: str,
    session: DbSession,
    _: User = Depends(require_admin)
):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        
    dept = await session.scalar(select(Department).where(Department.code == department_code))
    if not dept:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Department not found.")
        
    if not dept.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot assign permission in an inactive department.")
        
    perm = await session.scalar(select(Permission).where(Permission.code == permission_code))
    if not perm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found.")
        
    if not perm.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot assign an inactive permission.")
        
    # Check membership (application level verification before hitting DB fk constraint)
    is_member = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == user_id,
            UserDepartment.department_id == dept.id
        )
    )
    if not is_member:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User must belong to the department before permissions can be granted.")
        
    user_perm = UserPermission(user_id=user.id, department_id=dept.id, permission_id=perm.id)
    session.add(user_perm)
    
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Permission already granted for this department.")
        
    return {"message": "Permission granted successfully"}

@router.delete("/{user_id}/departments/{department_code}/permissions/{permission_code}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_user_permission(
    user_id: UUID,
    department_code: str,
    permission_code: str,
    session: DbSession,
    _: User = Depends(require_admin)
):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        
    dept = await session.scalar(select(Department).where(Department.code == department_code))
    if not dept:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Department not found.")
        
    perm = await session.scalar(select(Permission).where(Permission.code == permission_code))
    if not perm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found.")
        
    stmt = delete(UserPermission).where(
        UserPermission.user_id == user.id,
        UserPermission.department_id == dept.id,
        UserPermission.permission_id == perm.id
    )
    result = await session.execute(stmt)
    await session.commit()
    
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission is not granted for this user in this department.")

@router.post("/{user_id}/departments/{department_code}/permission-preset", status_code=status.HTTP_201_CREATED)
async def apply_permission_preset(
    user_id: UUID,
    department_code: str,
    payload: PresetRequest,
    session: DbSession,
    _: User = Depends(require_admin)
):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        
    dept = await session.scalar(select(Department).where(Department.code == department_code))
    if not dept:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Department not found.")
        
    if not dept.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot assign permissions in an inactive department.")
        
    # Check membership
    is_member = await session.scalar(
        select(UserDepartment).where(
            UserDepartment.user_id == user_id,
            UserDepartment.department_id == dept.id
        )
    )
    if not is_member:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User must belong to the department before presets can be applied.")
        
    required_codes = PRESET_PERMISSIONS[payload.preset]
    
    # Fetch active permissions that match the required codes
    perms = await session.scalars(
        select(Permission).where(
            Permission.code.in_(required_codes),
            Permission.is_active == True
        )
    )
    perm_map = {p.code: p.id for p in perms.all()}
    
    # Verify all required permissions exist and are active
    missing = set(required_codes) - set(perm_map.keys())
    if missing:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Required permissions {missing} for preset are missing or inactive.")
        
    # Fetch existing permissions to ensure idempotency
    existing_user_perms = await session.scalars(
        select(UserPermission.permission_id)
        .where(
            UserPermission.user_id == user.id,
            UserPermission.department_id == dept.id
        )
    )
    existing_perm_ids = set(existing_user_perms.all())
    
    new_perms = []
    for code in required_codes:
        pid = perm_map[code]
        if pid not in existing_perm_ids:
            new_perms.append(UserPermission(user_id=user.id, department_id=dept.id, permission_id=pid))
            
    if new_perms:
        session.add_all(new_perms)
        await session.commit()
        
    return {"message": f"Preset {payload.preset} applied successfully"}


from rag_app.db.models import (
    User, Department, UserDepartment, Permission, UserPermission, Document,
    DocumentChunk, KnowledgeItem, KnowledgeEvidence, DocumentPermission
)
from rag_app.schemas.users import (
    DocPermissionGrantRequest, DocPermissionRevokeRequest, DocPermissionSetRequest,
    DocUserAccessItem, DocumentAccessListResponse
)
from sqlalchemy import update, and_, exists

# ── Document Scope Administration Router ────────────────────────────────────

doc_admin_router = APIRouter(prefix="/admin/documents", tags=["admin-documents"])


@doc_admin_router.patch("/{document_id}/department", response_model=DocumentAdminResponse)
async def update_document_department(
    document_id: UUID,
    payload: DocumentDepartmentUpdate,
    session: DbSession,
    _: User = Depends(require_admin),
):
    """
    Minimal ADMIN-only operation to change document.department_id to an ACTIVE department.
    Also updates associated knowledge items if present.
    """
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    dept_code = payload.department_code.strip().upper()
    dept = await session.scalar(select(Department).where(Department.code == dept_code))
    if not dept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department '{dept_code}' not found."
        )
    if not dept.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Department '{dept_code}' is inactive."
        )

    doc.department_id = dept.id

    # Also update any knowledge items associated with this document's chunks if not duplicate
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

    return DocumentAdminResponse(
        id=doc.id,
        file_name=doc.file_name,
        department_id=doc.department_id,
        department_code=dept.code,
        department_name=dept.name,
        status=doc.status,
        created_at=doc.created_at,
    )


@doc_admin_router.get("/{document_id}/permissions", response_model=DocumentAccessListResponse)
async def get_document_permissions(
    document_id: UUID,
    session: DbSession,
    _: User = Depends(require_admin),
):
    """
    Fetch department members and their document-specific vs department-level permissions for a document.
    """
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    dept = await session.get(Department, doc.department_id) if doc.department_id else None
    if not dept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document department not found."
        )

    # Fetch active users in this department
    users_stmt = (
        select(User)
        .join(UserDepartment, User.id == UserDepartment.user_id)
        .where(
            UserDepartment.department_id == dept.id,
            User.is_active == True,
        )
        .order_by(User.email)
    )
    dept_users = (await session.scalars(users_stmt)).all()

    # Fetch all department permissions for these users
    dept_perms_stmt = (
        select(UserPermission.user_id, Permission.code)
        .join(Permission, UserPermission.permission_id == Permission.id)
        .where(
            UserPermission.department_id == dept.id,
            Permission.is_active == True,
        )
    )
    dept_perm_rows = (await session.execute(dept_perms_stmt)).all()
    dept_perm_map: dict[UUID, set[str]] = {}
    for uid, pcode in dept_perm_rows:
        dept_perm_map.setdefault(uid, set()).add(pcode)

    # Fetch all document-specific permissions for these users on this document
    doc_perms_stmt = (
        select(DocumentPermission.user_id, Permission.code)
        .join(Permission, DocumentPermission.permission_id == Permission.id)
        .where(
            DocumentPermission.document_id == doc.id,
            Permission.is_active == True,
        )
    )
    doc_perm_rows = (await session.execute(doc_perms_stmt)).all()
    doc_perm_map: dict[UUID, set[str]] = {}
    for uid, pcode in doc_perm_rows:
        doc_perm_map.setdefault(uid, set()).add(pcode)

    user_items = []
    for u in dept_users:
        dept_p = sorted(dept_perm_map.get(u.id, set()))
        doc_p = sorted(doc_perm_map.get(u.id, set()))
        eff_p = sorted(set(dept_p) | set(doc_p))
        user_items.append(
            DocUserAccessItem(
                user_id=u.id,
                email=u.email,
                full_name=u.email.split("@")[0].capitalize(),
                department_permissions=dept_p,
                document_permissions=doc_p,
                effective_permissions=eff_p,
            )
        )

    return DocumentAccessListResponse(
        document_id=doc.id,
        file_name=doc.file_name,
        department_id=dept.id,
        department_code=dept.code,
        department_name=dept.name,
        users=user_items,
    )


@doc_admin_router.post("/{document_id}/permissions/grant")
async def grant_document_permission(
    document_id: UUID,
    payload: DocPermissionGrantRequest,
    session: DbSession,
    _: User = Depends(require_admin),
):
    """
    Grant a document-specific permission (VIEW, UPLOAD, EDIT, DELETE) to a user in the document's department.
    """
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    dept = await session.get(Department, doc.department_id) if doc.department_id else None
    if not dept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document department not found."
        )
    if not dept.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Department '{dept.code}' is inactive."
        )

    user = await session.get(User, payload.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is inactive."
        )

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
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not a member of the document's department. Cannot grant document permissions to non-members."
        )

    perm_code = payload.permission_code.strip().upper()
    perm = await session.scalar(
        select(Permission).where(
            Permission.code == perm_code,
            Permission.is_active == True,
        )
    )
    if not perm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or inactive permission code '{perm_code}'."
        )

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

    return {"message": f"Permission {perm_code} granted to user {user.email} for document {doc.file_name}"}


@doc_admin_router.post("/{document_id}/permissions/revoke")
async def revoke_document_permission(
    document_id: UUID,
    payload: DocPermissionRevokeRequest,
    session: DbSession,
    _: User = Depends(require_admin),
):
    """
    Revoke a document-specific permission from a user on a document.
    """
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    perm_code = payload.permission_code.strip().upper()
    perm = await session.scalar(
        select(Permission).where(
            Permission.code == perm_code,
            Permission.is_active == True,
        )
    )
    if not perm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or inactive permission code '{perm_code}'."
        )

    await session.execute(
        delete(DocumentPermission).where(
            DocumentPermission.document_id == doc.id,
            DocumentPermission.user_id == payload.user_id,
            DocumentPermission.permission_id == perm.id,
        )
    )
    await session.commit()

    return {"message": f"Permission {perm_code} revoked for user on document {doc.file_name}"}


@doc_admin_router.put("/{document_id}/permissions")
async def set_document_permissions(
    document_id: UUID,
    payload: DocPermissionSetRequest,
    session: DbSession,
    _: User = Depends(require_admin),
):
    """
    Set exact list of document-specific permissions for a user on a document.
    """
    doc = await session.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    dept = await session.get(Department, doc.department_id) if doc.department_id else None
    if not dept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document department not found."
        )
    if not dept.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Department '{dept.code}' is inactive."
        )

    user = await session.get(User, payload.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is inactive."
        )

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
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not a member of the document's department. Cannot grant document permissions to non-members."
        )

    target_perm_codes = [p.strip().upper() for p in payload.permissions]
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
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or inactive permissions: {', '.join(missing_codes)}"
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



