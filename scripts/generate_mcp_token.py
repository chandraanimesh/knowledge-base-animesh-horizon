import asyncio
import sys
import uuid
from datetime import timedelta
from sqlalchemy import select

from rag_app.db.session import AsyncSessionLocal
from rag_app.db.models import User
from rag_app.security.authentication import create_access_token, get_password_hash


async def generate_mcp_token(email: str = "admin@example.com", days: int = 30) -> None:
    """Generate a valid JWT bearer token for local MCP client authorization."""
    async with AsyncSessionLocal() as session:
        user = await session.scalar(
            select(User).where(User.email == email.lower().strip())
        )
        if not user:
            # Fallback: look for any active admin user
            user = await session.scalar(
                select(User).where(User.role == "ADMIN", User.is_active == True).limit(1)
            )

        if not user:
            # Create a dedicated local dev admin user
            user = User(
                id=uuid.uuid4(),
                email=email.lower().strip(),
                hashed_password=get_password_hash("LocalDevPassword123!"),
                role="ADMIN",
                is_active=True,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

        expires_delta = timedelta(days=days)
        token = create_access_token(subject=str(user.id), expires_delta=expires_delta)

        print("\n" + "=" * 60)
        print(" PRODUCTION RAG - LOCAL MCP AUTHENTICATION TOKEN")
        print("=" * 60)
        print(f"User:  {user.email} (Role: {user.role}, ID: {user.id})")
        print(f"Valid: {days} days")
        print("-" * 60)
        print("Raw Token:")
        print(token)
        print("-" * 60)
        print("Authorization Header:")
        print(f"Bearer {token}")
        print("=" * 60)
        print("\nTwenty CRM / MCP Client Configuration Snippet:")
        print(f"""{{
  "mcpServers": {{
    "production-rag-local": {{
      "url": "http://localhost:8000/mcp/sse",
      "headers": {{
        "Authorization": "Bearer {token}"
      }}
    }}
  }}
}}""")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    email_arg = sys.argv[1] if len(sys.argv) > 1 else "admin@example.com"
    asyncio.run(generate_mcp_token(email=email_arg))
