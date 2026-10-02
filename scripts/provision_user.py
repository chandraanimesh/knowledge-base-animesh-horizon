import asyncio
import sys
from sqlalchemy import select
from rag_app.db.session import AsyncSessionLocal
from rag_app.db.models import User
from rag_app.security.authentication import get_password_hash

async def provision_user(email: str, password: str, is_active: bool = True):
    async with AsyncSessionLocal() as session:
        user = await session.scalar(
            select(User).where(User.email == email.lower().strip())
        )
        if not user:
            print(f"User {email} not found. Please create the user first.")
            return

        user.hashed_password = get_password_hash(password)
        user.is_active = is_active
        await session.commit()
        print(f"Successfully provisioned credentials for {email}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run scripts/provision_user.py <email> <password>")
        sys.exit(1)
    asyncio.run(provision_user(sys.argv[1], sys.argv[2]))
