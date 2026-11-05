from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import User
from rag_app.db.session import get_db_session
from rag_app.security.authentication import verify_password, create_access_token

router = APIRouter(tags=["auth"])

DbSession = Annotated[AsyncSession, Depends(get_db_session)]

@router.post("/auth/login")
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: DbSession
):
    user = await session.scalar(
        select(User).where(User.email == form_data.username.lower().strip())
    )
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    reg_status = getattr(user, "registration_status", "APPROVED")
    if reg_status == "PENDING":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your registration is pending administrator approval.",
        )
    if reg_status == "REJECTED":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your registration request has been declined.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
        
    access_token = create_access_token(subject=str(user.id))
    return {"access_token": access_token, "token_type": "bearer"}


from pydantic import BaseModel, EmailStr, Field
from rag_app.core.config import get_settings
from rag_app.security.authentication import get_password_hash


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    requested_department: str | None = None
    reason: str | None = None


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    session: DbSession,
):
    """
    Self-registration endpoint. Creates an inactive user with status='PENDING'.
    Does not issue JWT tokens until approved by an administrator.
    """
    settings = get_settings()
    if not getattr(settings, "enable_self_registration", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-registration is disabled by system policy.",
        )

    email_clean = payload.email.lower().strip()
    existing = await session.scalar(select(User).where(User.email == email_clean))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )

    hashed_pw = get_password_hash(payload.password)
    user = User(
        email=email_clean,
        hashed_password=hashed_pw,
        role="USER",
        is_active=False,
        registration_status="PENDING",
        profile_json={
            "requested_department": payload.requested_department,
            "registration_reason": payload.reason,
        },
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    return {
        "status": "PENDING_APPROVAL",
        "message": "Your registration request has been submitted and is pending administrator approval.",
        "user_id": str(user.id),
        "email": user.email,
    }

from rag_app.api.dependencies.auth import get_current_user
from rag_app.schemas.users import UserResponse
from rag_app.db.models import Department, Permission, UserDepartment, UserPermission
from sqlalchemy import and_

@router.get("/auth/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    return user

@router.get("/auth/me/access")
async def get_my_access(
    session: DbSession,
    user: User = Depends(get_current_user)
):
    """
    Returns a dictionary of departments and permissions the user has access to.
    e.g. {"SALES": ["VIEW", "UPLOAD"], "FINANCE": ["VIEW"]}
    """
    if user.role == "ADMIN":
        # ADMIN has global access to all departments
        stmt = select(Department.code).where(Department.is_active == True)
        result = await session.execute(stmt)
        dept_codes = result.scalars().all()
        return {code: ["ALL"] for code in dept_codes}

    # Query all active departments and permissions for this user
    stmt = (
        select(Department.code, Permission.code)
        .join(UserDepartment, UserDepartment.department_id == Department.id)
        .join(
            UserPermission,
            and_(
                UserPermission.user_id == user.id,
                UserPermission.department_id == Department.id
            )
        )
        .join(Permission, Permission.id == UserPermission.permission_id)
        .where(
            UserDepartment.user_id == user.id,
            Department.is_active == True,
            Permission.is_active == True
        )
    )
    result = await session.execute(stmt)
    
    access = {}
    for dept_code, perm_code in result.all():
        if dept_code not in access:
            access[dept_code] = []
        access[dept_code].append(perm_code)
        
    return access
