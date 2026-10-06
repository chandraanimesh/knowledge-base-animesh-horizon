import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.db.models import ChatMessage, Conversation, User, Workspace, WorkspaceMember, Department
from rag_app.security.authorization import authorize, get_allowed_workspaces
from rag_app.schemas.rag import (
    ChatMessageResponse,
    ConversationDetailResponse,
    ConversationResponse,
)


def generate_deterministic_title(query: str, max_length: int = 50) -> str:
    """
    Deterministically creates a clean, lightweight, meaningful title from the first user query.
    No extra LLM call is required.
    """
    cleaned = " ".join(query.strip().split())
    if not cleaned:
        return "New Chat"

    # 1. Strip conversational interrogatives & prefixes at the start
    patterns = [
        r"^(?:can you|could you|please|would you)\s+(?:please\s+)?(?:tell me about|explain to me|explain|describe|give me|show me|find out|summarize|detail|share)\s+",
        r"^(?:what|how|why|when|where|who)\s+(?:were|was|are|is|did|does|do|happened with|happened regarding|about)\s+",
        r"^(?:tell me|explain to me|explain|describe|give me|show me|summarize|search for|find out)\s+(?:about|everything about|more about|the|a|an)?\s*",
    ]
    sub = cleaned
    for pat in patterns:
        sub = re.sub(pat, "", sub, flags=re.IGNORECASE).strip()

    # Strip leading articles
    sub = re.sub(r"^(?:the|a|an)\s+", "", sub, flags=re.IGNORECASE).strip()

    # 2. Compare X and Y pattern -> X vs Y
    compare_match = re.match(r"^compare\s+(.+?)\s+and\s+(.+)$", sub, re.IGNORECASE)
    if compare_match:
        part1 = compare_match.group(1).strip()
        part2 = compare_match.group(2).strip()
        sub = f"{part1} vs {part2}"

    # 3. What did we discuss with X -> X Discussion
    discuss_match = re.match(r"^(?:we\s+)?discuss(?:ed)?\s+(?:with\s+)?(.+)$", sub, re.IGNORECASE)
    if discuss_match:
        sub = f"{discuss_match.group(1).strip()} Discussion"

    # 4. Strip trailing punctuation & quotes
    sub = sub.strip("\"'?!.,:;` ")
    if not sub:
        sub = cleaned.strip("\"'?!.,:;` ")

    # 5. Capitalize words appropriately (Title Case while keeping acronyms like APAC, Q2, ABC uppercase)
    words = sub.split()
    minor_words = {"a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "from", "by", "in", "of", "vs", "with"}
    title_words = []
    for i, word in enumerate(words):
        # Preserve known acronyms (e.g. APAC, Q1, Q2, Q3, Q4, ABC, RAG, HR, EU, US, ID)
        if word.isupper() and len(word) > 1:
            title_words.append(word)
        elif i > 0 and word.lower() in minor_words:
            title_words.append(word.lower())
        else:
            title_words.append(word.capitalize())

    title = " ".join(title_words)

    if len(title) <= max_length:
        return title

    # Word-boundary truncation
    truncated = title[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > 20:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.- ") + "…"


class ConversationService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _validate_workspace_access(self, user: User, workspace_id: uuid.UUID) -> Workspace:
        """
        Validates workspace exists, is active, department access, and user membership.
        Supports resolving workspace by ID or by Department ID.
        """
        ws = await self.session.scalar(
            select(Workspace).where(
                Workspace.id == workspace_id,
                Workspace.deleted_at.is_(None),
                Workspace.is_active == True,
            )
        )
        if not ws:
            ws = await self.session.scalar(
                select(Workspace).where(
                    Workspace.department_id == workspace_id,
                    Workspace.deleted_at.is_(None),
                    Workspace.is_active == True,
                )
            )

        if not ws:
            # Auto-provision workspace for active department if valid department ID
            dept = await self.session.get(Department, workspace_id)
            if dept and dept.is_active:
                now = datetime.now(timezone.utc)
                ws = Workspace(
                    name=dept.name,
                    code=f"{dept.code.lower()}_ws",
                    department_id=dept.id,
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                )
                self.session.add(ws)
                await self.session.flush()
                self.session.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="OWNER" if user.role == "ADMIN" else "EDITOR"))
                await self.session.flush()

        if not ws:
            raise HTTPException(
                status_code=404,
                detail=f"Workspace '{workspace_id}' not found or inactive.",
            )

        if user.role != "ADMIN":
            dept = await self.session.get(Department, ws.department_id)
            if not dept or not (await authorize(self.session, user, "VIEW", dept.code)):
                raise HTTPException(
                    status_code=403,
                    detail=f"Forbidden: You do not have access to department '{dept.code if dept else ''}'.",
                )
            ws_member = await self.session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == ws.id,
                    WorkspaceMember.user_id == user.id,
                )
            )
            if not ws_member:
                # If this is the 1:1 default workspace for the department, auto-enroll authorized member
                if dept and (ws.code == f"{dept.code.lower()}_ws" or ws.code == dept.code.lower() or ws.name == dept.name):
                    self.session.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="EDITOR"))
                    await self.session.flush()
                else:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Forbidden: User is not a member of workspace '{ws.name}'.",
                    )

        return ws

    async def create_conversation(
        self,
        user: User,
        title: str | None = None,
        workspace_id: uuid.UUID | None = None,
    ) -> ConversationResponse:
        ws_name: str | None = None
        target_ws_id: uuid.UUID | None = None
        if workspace_id is not None:
            ws = await self._validate_workspace_access(user, workspace_id)
            ws_name = ws.name
            target_ws_id = ws.id

        now = datetime.now(timezone.utc)
        conv = Conversation(
            user_id=user.id,
            workspace_id=target_ws_id,
            title=title or "New Thread",
            created_at=now,
            updated_at=now,
        )
        self.session.add(conv)
        await self.session.commit()
        return ConversationResponse(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=0,
            workspace_id=conv.workspace_id,
            workspace_name=ws_name,
        )

    async def list_conversations(
        self,
        user: User,
        workspace_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[ConversationResponse]:
        """
        Lists conversations for the authenticated user, ordered by updated_at descending.
        If workspace_id is supplied, validates workspace authorization and scopes results.
        """
        target_ws_id: uuid.UUID | None = None
        if workspace_id is not None:
            ws = await self._validate_workspace_access(user, workspace_id)
            target_ws_id = ws.id

        stmt = (
            select(
                Conversation,
                func.count(ChatMessage.id).label("message_count"),
                Workspace.name.label("workspace_name"),
            )
            .outerjoin(ChatMessage, ChatMessage.conversation_id == Conversation.id)
            .outerjoin(Workspace, Workspace.id == Conversation.workspace_id)
            .where(Conversation.user_id == user.id)
        )

        if target_ws_id is not None:
            stmt = stmt.where(Conversation.workspace_id == target_ws_id)
        else:
            # Global Chat scope: strictly list only Global conversations where workspace_id is NULL
            stmt = stmt.where(Conversation.workspace_id.is_(None))

        stmt = (
            stmt.group_by(Conversation.id, Workspace.name)
            .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        rows = result.all()

        return [
            ConversationResponse(
                id=str(conv.id),
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=count,
                workspace_id=conv.workspace_id,
                workspace_name=ws_name,
            )
            for conv, count, ws_name in rows
        ]

    async def get_conversation_detail(
        self,
        conversation_id: uuid.UUID,
        user: User,
    ) -> ConversationDetailResponse:
        """
        Fetches full conversation history for the authenticated user.
        Validates workspace authorization if conversation belongs to a workspace.
        """
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        if user.role != "ADMIN":
            stmt = stmt.where(Conversation.user_id == user.id)

        conv = await self.session.scalar(stmt)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conv.workspace_id is not None:
            await self._validate_workspace_access(user, conv.workspace_id)

        messages_resp = [
            ChatMessageResponse(
                id=str(msg.id),
                conversation_id=str(msg.conversation_id),
                role=msg.role,
                content=msg.content,
                visualization=msg.visualization,
                sources=msg.sources,
                execution_metrics=msg.execution_metrics,
                created_at=msg.created_at,
            )
            for msg in conv.messages
        ]

        return ConversationDetailResponse(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            workspace_id=conv.workspace_id,
            messages=messages_resp,
        )

    async def delete_conversation(
        self,
        conversation_id: uuid.UUID,
        user: User,
    ) -> dict[str, str]:
        """
        Deletes a conversation and cascades to its messages.
        """
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        if user.role != "ADMIN":
            stmt = stmt.where(Conversation.user_id == user.id)

        conv = await self.session.scalar(stmt)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conv.workspace_id is not None:
            await self._validate_workspace_access(user, conv.workspace_id)

        await self.session.delete(conv)
        await self.session.commit()
        return {"status": "deleted", "id": str(conversation_id)}

    def is_placeholder_title(self, title: str | None) -> bool:
        """
        Returns True if the thread has a default/placeholder name that should be
        automatically replaced by a conversation-specific descriptive title.
        """
        if not title or not title.strip():
            return True
        clean = title.strip().lower()
        return (
            clean in ("new chat", "new thread", "untitled", "untitled thread", "query thread", "new conversation", "chat")
            or clean.startswith("thread ")
            or clean.startswith("new chat ")
            or clean.startswith("new thread ")
        )

    async def generate_llm_title(
        self,
        query: str,
        assistant_response: str | None = None,
        llm: Any = None,
    ) -> str:
        """
        Generates a concise 3-7 word conversation title using the existing LLM provider.
        Follows prompt rules: 3-7 words, descriptive, no quotes, no markdown, no 'Thread:'.
        Gracefully falls back to deterministic rule-based title on timeout or error.
        """
        import asyncio
        import logging
        logger = logging.getLogger(__name__)

        if llm is None:
            try:
                from rag_app.generation.llm_factory import get_llm
                llm = get_llm()
            except Exception as e:
                logger.debug(f"Could not initialize LLM for title generation: {e}")
                llm = None

        if llm is not None:
            try:
                conv_snippet = f"User: {query}"
                if assistant_response:
                    conv_snippet += f"\nAssistant: {assistant_response[:300]}"

                prompt = (
                    "Generate a concise title describing the main topic of this conversation.\n\n"
                    "Rules:\n"
                    "- 3–7 words\n"
                    "- descriptive\n"
                    "- no quotes\n"
                    "- no markdown\n"
                    "- no 'Thread:'\n"
                    "- no unnecessary punctuation\n"
                    "- do not invent information\n"
                    "- use only information present in the conversation\n\n"
                    f"Conversation:\n{conv_snippet}\n\n"
                    "Title:"
                )
                raw_title = await asyncio.wait_for(llm.generate(prompt), timeout=4.0)
                cleaned = raw_title.strip().strip("\"'#*` \n\r")
                cleaned = re.sub(r"^(?:title|thread):\s*", "", cleaned, flags=re.IGNORECASE).strip("\"'#*` ")
                # Strip wrapping quotes if any
                cleaned = cleaned.strip("\"' ")
                words = cleaned.split()
                if 2 <= len(words) <= 9:
                    return " ".join(words)
            except Exception as exc:
                logger.warning(f"LLM title generation failed or timed out ({exc}), falling back to deterministic title.")

        return generate_deterministic_title(query)

    async def get_or_create_conversation(
        self,
        user: User,
        conversation_id: uuid.UUID | None,
        initial_query: str,
        workspace_id: uuid.UUID | None = None,
    ) -> Conversation:
        """
        Ensures a conversation exists for the request.
        If conversation_id is passed, verifies ownership and workspace authorization.
        If not, creates a new one in the target workspace with a deterministic title from the query.
        """
        if conversation_id is not None:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            if user.role != "ADMIN":
                stmt = stmt.where(Conversation.user_id == user.id)

            conv = await self.session.scalar(stmt)
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Validate scope match between request and existing conversation
            if workspace_id is None:
                # Global Chat request: Conversation must be a global conversation (workspace_id is None)
                if conv.workspace_id is not None:
                    raise HTTPException(
                        status_code=400,
                        detail="Scope mismatch: Cannot use a workspace-scoped conversation in Global Chat.",
                    )
            else:
                # Workspace Chat request: Conversation must belong to the requested workspace
                if conv.workspace_id is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Scope mismatch: Cannot use a global conversation in Workspace Chat.",
                    )
                conv_ws = await self._validate_workspace_access(user, conv.workspace_id)
                req_ws = await self._validate_workspace_access(user, workspace_id)
                if conv_ws.id != req_ws.id:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Conversation belongs to workspace '{conv_ws.name}', not requested workspace '{req_ws.name}'.",
                    )

            # If the conversation has a placeholder title, assign a deterministic initial title
            if self.is_placeholder_title(conv.title):
                conv.title = generate_deterministic_title(initial_query)
                conv.updated_at = datetime.now(timezone.utc)
                await self.session.commit()

            return conv

        # Auto-create new conversation
        target_ws_id: uuid.UUID | None = None
        if workspace_id is not None:
            ws = await self._validate_workspace_access(user, workspace_id)
            target_ws_id = ws.id

        title = generate_deterministic_title(initial_query)
        now = datetime.now(timezone.utc)
        conv = Conversation(
            user_id=user.id,
            workspace_id=target_ws_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        self.session.add(conv)
        await self.session.commit()
        return conv

    async def search_workspace_threads(
        self,
        user: User,
        workspace_id: uuid.UUID | None = None,
        query: str = "",
        current_conversation_id: uuid.UUID | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieves relevant prior conversation messages within the authorized Workspace(s).
        If workspace_id is None, searches across all authorized workspaces for the user.
        """
        from rag_app.retrieval.workspace_thread_retriever import WorkspaceThreadRetriever
        return await WorkspaceThreadRetriever.retrieve_workspace_threads(
            session=self.session,
            user=user,
            workspace_id=workspace_id,
            query=query,
            current_conversation_id=current_conversation_id,
            limit=limit,
        )

    async def get_conversation_context_window(
        self,
        conversation_id: uuid.UUID,
        max_turns: int = 5,
    ) -> list[dict[str, str]]:
        """
        Retrieves the latest `max_turns` (default 5 turns = 10 messages)
        from DB for the LLM active context.
        Older messages remain safely preserved in DB.
        """
        max_messages = max_turns * 2
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at.asc())
        )
        result = await self.session.execute(stmt)
        all_messages = list(result.scalars().all())

        # Select latest max_messages (5 user-assistant turns)
        window = all_messages[-max_messages:] if len(all_messages) > max_messages else all_messages

        active_history = [
            {"role": msg.role, "content": msg.content}
            for msg in window
            if msg.role in ("user", "assistant")
        ]

        # Apply memory seam hook for future long-term retrieval
        return self.apply_memory_seam(conversation_id, active_history)

    def apply_memory_seam(
        self,
        conversation_id: uuid.UUID,
        active_history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        --- FUTURE SUPERMEMORY SEAM ---
        Clean architectural hook:
        When Supermemory is enabled, older conversation history / semantic memories
        can be extracted and dynamically prepended or injected here.
        Currently returns active_history unchanged.
        """
        return active_history

    async def save_user_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
    ) -> ChatMessage:
        msg = ChatMessage(
            conversation_id=conversation_id,
            role="user",
            content=content,
        )
        self.session.add(msg)
        await self.session.commit()
        return msg

    async def save_assistant_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        sources: list[dict] | None = None,
        execution_metrics: dict | None = None,
        visualization: dict | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            sources=sources,
            execution_metrics=execution_metrics,
            visualization=visualization,
        )
        self.session.add(msg)

        # Update conversation updated_at
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        conv = await self.session.scalar(stmt)
        if conv:
            conv.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        return msg


