import asyncio
import os
import sys
import uuid
from sqlalchemy import select, delete, update, func

from rag_app.core.config import get_settings
from rag_app.db.session import AsyncSessionLocal
from rag_app.db.models import (
    User,
    Department,
    Permission,
    UserDepartment,
    UserPermission,
    AccessRequest,
    AccessRequestPermission,
    Document,
    KnowledgeItem,
    ResearchRun,
    Workspace,
)
from rag_app.security.authentication import get_password_hash, verify_password

CANONICAL_DEPARTMENTS = [
    ("UNCLASSIFIED", "Unclassified"),
    ("SALES", "Sales"),
    ("FINANCE", "Finance"),
    ("HR", "HR"),
]

CANONICAL_PERMISSIONS = [
    ("VIEW", "View access"),
    ("UPLOAD", "Upload access"),
    ("EDIT", "Edit access"),
    ("DELETE", "Delete access"),
]


async def seed_admin(
    custom_email: str | None = None,
    custom_password: str | None = None,
    enforce_safety: bool = True,
) -> dict:
    settings = get_settings()

    # 1. Environment Safety Check
    env = (os.getenv("ENV") or settings.env).lower().strip()
    if enforce_safety and env in ("prod", "production"):
        raise RuntimeError(
            f"Destructive seed/reset is disabled in production environment (ENV={env})."
        )

    # 2. Resolve Credentials
    admin_email = (
        custom_email
        or os.getenv("ADMIN_EMAIL")
        or settings.admin_email
    )
    admin_password = (
        custom_password
        or os.getenv("ADMIN_PASSWORD")
        or settings.admin_password
    )

    if not admin_email or not admin_password:
        raise ValueError(
            "ADMIN_EMAIL and ADMIN_PASSWORD must be configured in environment or .env file."
        )

    admin_email = admin_email.strip().lower()
    admin_password = admin_password.strip()

    async with AsyncSessionLocal() as session:
        # 3. Clean up user memberships, permissions, and access requests
        await session.execute(delete(AccessRequestPermission))
        await session.execute(delete(AccessRequest))
        await session.execute(delete(UserPermission))
        await session.execute(delete(UserDepartment))

        # 4. Remove all non-admin or stale user accounts
        await session.execute(
            delete(User).where(
                (User.email != admin_email) | (User.role != "ADMIN")
            )
        )

        # 5. Ensure single DEV ADMIN exists with up-to-date password & role
        existing_admin = await session.scalar(
            select(User).where(User.email == admin_email)
        )
        if existing_admin:
            existing_admin.role = "ADMIN"
            existing_admin.is_active = True
            existing_admin.hashed_password = get_password_hash(admin_password)
            existing_admin.deleted_at = None
        else:
            admin_user = User(
                id=uuid.uuid4(),
                email=admin_email,
                role="ADMIN",
                is_active=True,
                hashed_password=get_password_hash(admin_password),
            )
            session.add(admin_user)

        # 6. Ensure canonical departments exist and are active
        canonical_dept_map = {}
        for code, name in CANONICAL_DEPARTMENTS:
            dept = await session.scalar(
                select(Department).where(Department.code == code)
            )
            if dept:
                dept.name = name
                dept.is_active = True
                dept.deleted_at = None
            else:
                dept = Department(
                    id=uuid.uuid4(),
                    code=code,
                    name=name,
                    is_active=True,
                )
                session.add(dept)
            await session.flush()
            canonical_dept_map[code] = dept

        unclassified_dept_id = canonical_dept_map["UNCLASSIFIED"].id
        canonical_codes = list(canonical_dept_map.keys())

        # Remove knowledge items, research runs, and documents linked to non-canonical departments
        non_canonical_subq = select(Department.id).where(Department.code.not_in(canonical_codes))

        await session.execute(
            delete(KnowledgeItem).where(KnowledgeItem.department_id.in_(non_canonical_subq))
        )
        await session.execute(
            delete(ResearchRun).where(ResearchRun.department_id.in_(non_canonical_subq))
        )
        await session.execute(
            delete(Document).where(Document.department_id.in_(non_canonical_subq))
        )
        await session.execute(
            delete(Workspace).where(Workspace.department_id.in_(non_canonical_subq))
        )

        # Remove non-canonical departments
        await session.execute(
            delete(Department).where(Department.code.not_in(canonical_codes))
        )

        # 7. Ensure canonical permissions exist and are active
        for code, desc in CANONICAL_PERMISSIONS:
            perm = await session.scalar(
                select(Permission).where(Permission.code == code)
            )
            if perm:
                perm.description = desc
                perm.is_active = True
            else:
                session.add(
                    Permission(
                        id=uuid.uuid4(),
                        code=code,
                        description=desc,
                        is_active=True,
                    )
                )

        await session.commit()

        # 8. Verification assertions
        user_count = await session.scalar(select(func.count(User.id)))
        normal_users = await session.scalar(
            select(func.count(User.id)).where(User.role == "USER")
        )
        ud_count = await session.scalar(select(func.count(UserDepartment.user_id)))
        up_count = await session.scalar(select(func.count(UserPermission.user_id)))
        ar_count = await session.scalar(select(func.count(AccessRequest.id)))

        assert user_count == 1, f"Expected 1 user, found {user_count}"
        assert normal_users == 0, f"Expected 0 normal users, found {normal_users}"
        assert ud_count == 0, f"Expected 0 user_departments, found {ud_count}"
        assert up_count == 0, f"Expected 0 user_permissions, found {up_count}"
        assert ar_count == 0, f"Expected 0 access_requests, found {ar_count}"

    summary = (
        "========================================\n"
        "DEV ADMIN CREATED\n"
        f"Email: {admin_email}\n"
        f"Password: {admin_password}\n"
        "Role: ADMIN\n"
        "========================================"
    )
    print(summary)
    return {
        "email": admin_email,
        "password": admin_password,
        "role": "ADMIN",
    }


if __name__ == "__main__":
    try:
        asyncio.run(seed_admin())
    except Exception as e:
        print(f"Error seeding admin: {e}", file=sys.stderr)
        sys.exit(1)
