import json
import os
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

os.environ["DATABASE_URL"] = "postgresql+asyncpg://rag:rag@127.0.0.1:5433/rag_disposable_test"

from rag_app.core.stream_cleaner import StreamToolCleaner
from rag_app.db.session import get_db_session
from rag_app.db.models import User, Department
from rag_app.main import app
from rag_app.security.authentication import create_access_token


def test_stream_tool_cleaner_isolates_tool_calls():
    """Verify tool call XML/JSON is scrubbed from prose tokens and emitted as tool events."""
    cleaner = StreamToolCleaner()

    # Chunk with leaked tool call syntax
    chunk1 = "Hello! Let me check that. <tool_call>{\"name\": \"search_docs\", \"arguments\": {\"q\": \"policy\"}}</tool_call> According to our records,"
    clean_text, events = cleaner.process_chunk(chunk1)

    assert "<tool_call>" not in clean_text
    assert "</tool_call>" not in clean_text
    assert "search_docs" not in clean_text
    assert clean_text.strip() == "Hello! Let me check that."
    assert len(events) == 1
    assert events[0]["tool"] == "search_docs"
    assert events[0]["input"] == {"q": "policy"}

    # Next chunk is normal prose
    chunk2 = " the refund window is 30 days."
    clean_text2, events2 = cleaner.process_chunk(chunk2)
    assert clean_text2 == " According to our records, the refund window is 30 days."
    assert len(events2) == 0


@pytest.mark.asyncio
async def test_chat_stream_sse_endpoint():
    """Verify POST /chat/stream emits standard SSE event stream."""
    async for session in get_db_session():
        admin = await session.scalar(select(User).where(User.email == "admin_stream@local.dev"))
        if not admin:
            admin = User(email="admin_stream@local.dev", hashed_password="pw", role="ADMIN", is_active=True)
            session.add(admin)
            await session.flush()

        dept = await session.scalar(select(Department).where(Department.code == "STREAM_TEST"))
        if not dept:
            dept = Department(code="STREAM_TEST", name="Stream Test Dept", is_active=True)
            session.add(dept)
            await session.commit()

        token = create_access_token(subject=str(admin.id))
        headers = {"Authorization": f"Bearer {token}"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat/stream",
                headers=headers,
                json={"query": "Hello", "top_k": 3},
            )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            body = resp.text
            assert "event: start" in body
            assert "event: token" in body
            assert "event: done" in body
        from rag_app.db.session import engine
        await engine.dispose()
        break

