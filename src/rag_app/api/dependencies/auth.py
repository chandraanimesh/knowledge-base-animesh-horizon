import uuid
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from rag_app.db.session import get_db_session
from rag_app.db.models import User
from rag_app.security.authorization import authorize

from fastapi.security import OAuth2PasswordBearer
import jwt
from rag_app.core.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
settings = get_settings()

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_db_session)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, 
            settings.jwt_secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        user_uuid = uuid.UUID(user_id)
    except (jwt.PyJWTError, ValueError, TypeError):
        raise credentials_exception
        
    user = await session.get(User, user_uuid)
    if user is None:
        raise credentials_exception
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
        
    return user

class RequirePermission:
    """
    FastAPI dependency factory for department-scoped authorization.
    Requires an endpoint to extract and provide 'department_code'.
    """
    def __init__(self, permission_code: str):
        self.permission_code = permission_code

    async def __call__(
        self,
        department_code: str,
        user: User = Depends(get_current_user),
        session: AsyncSession = Depends(get_db_session)
    ) -> User:
        is_authorized = await authorize(
            session=session,
            user=user,
            permission_code=self.permission_code,
            department_code=department_code
        )
        
        if not is_authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden"
            )
            
        return user

async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user
