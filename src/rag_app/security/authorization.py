import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, exists
from rag_app.db.models import User, Department, Permission, UserDepartment, UserPermission, Document, DocumentPermission

async def authorize(
    session: AsyncSession,
    user: User,
    permission_code: str,
    department_code: str
) -> bool:
    """
    Core deterministic server-side authorization check.
    Returns True if the user is authorized, False otherwise.
    Fails closed by default.
    """
    # 1. Is user active?
    if not user.is_active:
        return False
        
    # 2. Is user ADMIN? Global allow
    if user.role == "ADMIN":
        return True
        
    # 3. Unknown/unexpected role -> DENY
    if user.role != "USER":
        return False
        
    # 4. Department-scoped explicit permission check
    # We must ensure:
    # - The department exists and is active
    # - The permission exists and is active
    # - The user possesses the explicit permission for that department
    
    stmt = select(
        exists().where(
            and_(
                Department.code == department_code,
                Department.is_active == True,
                Permission.code == permission_code,
                Permission.is_active == True,
                UserPermission.user_id == user.id,
                UserPermission.department_id == Department.id,
                UserPermission.permission_id == Permission.id
            )
        )
    )
    
    result = await session.scalar(stmt)
    return bool(result)

async def get_allowed_departments(
    session: AsyncSession,
    user: User
) -> list[uuid.UUID] | None:
    if not user.is_active:
        return []
    if user.role == "ADMIN":
        return None
    if user.role != "USER":
        return []

    stmt = select(UserDepartment.department_id).join(
        Department, UserDepartment.department_id == Department.id
    ).join(
        UserPermission, 
        and_(
            UserPermission.user_id == user.id,
            UserPermission.department_id == Department.id
        )
    ).join(
        Permission,
        and_(
            Permission.id == UserPermission.permission_id,
            Permission.code == "VIEW",
            Permission.is_active == True
        )
    ).where(
        UserDepartment.user_id == user.id,
        Department.is_active == True
    )
    result = await session.scalars(stmt)
    return list(result.all())

async def authorize_document(
    session: AsyncSession,
    user: User,
    permission_code: str,
    document_id: uuid.UUID
) -> bool:
    """
    Additive document-level authorization check.
    Effective permission = Department permission OR Document permission.
    Valid only if user is a member of the document's active department.
    ADMIN has global access.
    """
    if not user.is_active:
        return False
    if user.role == "ADMIN":
        return True
    if user.role != "USER":
        return False

    # 1. Fetch document and verify active department
    doc = await session.get(Document, document_id)
    if not doc or not doc.department_id:
        return False

    dept = await session.get(Department, doc.department_id)
    if not dept or not dept.is_active:
        return False

    # 2. Must be a member of the document's department
    is_member = await session.scalar(
        select(
            exists().where(
                and_(
                    UserDepartment.user_id == user.id,
                    UserDepartment.department_id == doc.department_id,
                )
            )
        )
    )
    if not is_member:
        return False

    # 3. Check permission existence and active state
    perm = await session.scalar(
        select(Permission).where(
            Permission.code == permission_code,
            Permission.is_active == True,
        )
    )
    if not perm:
        return False

    # 4. Check if user has Department-level permission
    has_dept_perm = await session.scalar(
        select(
            exists().where(
                and_(
                    UserPermission.user_id == user.id,
                    UserPermission.department_id == doc.department_id,
                    UserPermission.permission_id == perm.id,
                )
            )
        )
    )
    if has_dept_perm:
        return True

    # 5. Check if user has Document-level permission
    has_doc_perm = await session.scalar(
        select(
            exists().where(
                and_(
                    DocumentPermission.document_id == doc.id,
                    DocumentPermission.user_id == user.id,
                    DocumentPermission.permission_id == perm.id,
                )
            )
        )
    )
    return bool(has_doc_perm)

async def get_allowed_document_ids(
    session: AsyncSession,
    user: User,
    permission_code: str = "VIEW",
) -> list[uuid.UUID] | None:
    """
    Returns list of document IDs that the user has explicit document-level permissions for,
    ensuring the document's department is active and the user is a member of that department.
    Returns None for ADMIN (global access).
    """
    if not user.is_active:
        return []
    if user.role == "ADMIN":
        return None
    if user.role != "USER":
        return []

    stmt = (
        select(DocumentPermission.document_id)
        .join(Document, DocumentPermission.document_id == Document.id)
        .join(Department, Document.department_id == Department.id)
        .join(
            UserDepartment,
            and_(
                UserDepartment.user_id == user.id,
                UserDepartment.department_id == Department.id,
            ),
        )
        .join(
            Permission,
            and_(
                Permission.id == DocumentPermission.permission_id,
                Permission.code == permission_code,
                Permission.is_active == True,
            ),
        )
        .where(
            DocumentPermission.user_id == user.id,
            Department.is_active == True,
        )
    )
    result = await session.scalars(stmt)
    return list(result.all())

