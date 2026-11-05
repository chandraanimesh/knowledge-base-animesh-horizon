import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

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

    @staticmethod
    def _require_global_chat_admin(user: Any) -> None:
        """Keep organization-wide Global Chat available only to administrators."""
        user_role = None
        sa_state = getattr(user, "_sa_instance_state", None)
        if sa_state is not None and type(sa_state).__name__ == "InstanceState" and getattr(sa_state, "expired", False):
            loaded_dict = getattr(sa_state, "dict", {})
            user_role = loaded_dict.get("role")
        else:
            user_role = getattr(user, "role", None)

        if user_role != "ADMIN":
            raise HTTPException(
                status_code=403,
                detail="Global Chat is available to administrators only.",
            )

    async def _validate_workspace_access(self, user: Any, workspace_id: uuid.UUID | str) -> Workspace:
        """
        Validates workspace exists, is active, department access, and user membership.
        Supports resolving workspace by ID, Code, Name, Department ID, or Department Code.
        """
        if not workspace_id:
            raise HTTPException(status_code=400, detail="Workspace identifier cannot be empty.")

        user_id = None
        user_role = "USER"
        sa_state = getattr(user, "_sa_instance_state", None)
        if sa_state is not None and type(sa_state).__name__ == "InstanceState" and getattr(sa_state, "expired", False):
            if sa_state.key and len(sa_state.key) > 1 and sa_state.key[1]:
                user_id = sa_state.key[1][0]
            loaded_dict = getattr(sa_state, "dict", {})
            user_role = loaded_dict.get("role", "USER")
        else:
            user_id = getattr(user, "id", None)
            user_role = getattr(user, "role", "USER")

        ws_str = str(workspace_id).strip()
        ws_uuid: uuid.UUID | None = None
        try:
            ws_uuid = uuid.UUID(ws_str)
        except (ValueError, TypeError, AttributeError):
            ws_uuid = None

        ws: Workspace | None = None

        # 1. Look up by Workspace.id or Workspace.department_id if valid UUID
        if ws_uuid:
            ws = await self.session.scalar(
                select(Workspace).where(
                    Workspace.id == ws_uuid,
                    Workspace.deleted_at.is_(None),
                    Workspace.is_active == True,
                )
            )
            if not ws:
                ws = await self.session.scalar(
                    select(Workspace).where(
                        Workspace.department_id == ws_uuid,
                        Workspace.deleted_at.is_(None),
                        Workspace.is_active == True,
                    )
                )

        # 2. Look up by Workspace.code or Workspace.name (case-insensitive)
        if not ws:
            ws = await self.session.scalar(
                select(Workspace).where(
                    or_(
                        func.lower(Workspace.code) == ws_str.lower(),
                        func.lower(Workspace.name) == ws_str.lower(),
                    ),
                    Workspace.deleted_at.is_(None),
                    Workspace.is_active == True,
                )
            )

        # 3. Look up by Department.code or Department.name (case-insensitive)
        if not ws:
            dept = await self.session.scalar(
                select(Department).where(
                    or_(
                        func.lower(Department.code) == ws_str.lower(),
                        func.lower(Department.name) == ws_str.lower(),
                        *( [Department.id == ws_uuid] if ws_uuid else [] )
                    ),
                    Department.deleted_at.is_(None),
                    Department.is_active == True,
                )
            )
            if dept:
                ws = await self.session.scalar(
                    select(Workspace).where(
                        Workspace.department_id == dept.id,
                        Workspace.deleted_at.is_(None),
                        Workspace.is_active == True,
                    )
                )
                if not ws:
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
                    self.session.add(WorkspaceMember(workspace_id=ws.id, user_id=user_id, role="OWNER" if user_role == "ADMIN" else "EDITOR"))
                    await self.session.flush()

        # 4. Fallback: Auto-provision under active UNCLASSIFIED or first active department for new workspace code
        if not ws:
            default_dept = await self.session.scalar(
                select(Department).where(
                    func.lower(Department.code) == "unclassified",
                    Department.is_active == True,
                )
            )
            if not default_dept:
                default_dept = await self.session.scalar(
                    select(Department).where(Department.is_active == True).limit(1)
                )
            if default_dept:
                now = datetime.now(timezone.utc)
                clean_code = re.sub(r"[^a-zA-Z0-9_-]", "_", ws_str).lower()
                clean_name = ws_str.replace("_", " ").replace("-", " ").title()
                ws = Workspace(
                    name=clean_name,
                    code=clean_code,
                    department_id=default_dept.id,
                    is_active=True,
                    created_at=now,
                    updated_at=now,
                )
                self.session.add(ws)
                await self.session.flush()
                self.session.add(
                    WorkspaceMember(
                        workspace_id=ws.id,
                        user_id=user_id,
                        role="OWNER" if user_role == "ADMIN" else "EDITOR",
                    )
                )
                await self.session.flush()

        if not ws:
            raise HTTPException(
                status_code=404,
                detail=f"Workspace '{workspace_id}' not found or inactive.",
            )

        if user_role != "ADMIN":
            dept = await self.session.get(Department, ws.department_id)
            if not dept or not (await authorize(self.session, user, "VIEW", dept.code)):
                raise HTTPException(
                    status_code=403,
                    detail=f"Forbidden: You do not have access to department '{dept.code if dept else ''}'.",
                )
            ws_member = await self.session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == ws.id,
                    WorkspaceMember.user_id == user_id,
                )
            )
            if not ws_member:
                # Auto-enroll authorized department user in default department workspace
                self.session.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="EDITOR"))
                await self.session.flush()

        return ws

    async def create_conversation(
        self,
        user: User,
        title: str | None = None,
        workspace_id: uuid.UUID | str | None = None,
    ) -> ConversationResponse:
        ws_name: str | None = None
        target_ws_id: uuid.UUID | None = None
        if workspace_id is not None and str(workspace_id).strip():
            ws = await self._validate_workspace_access(user, workspace_id)
            ws_name = ws.name
            target_ws_id = ws.id
        else:
            self._require_global_chat_admin(user)

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
        workspace_id: uuid.UUID | str | None = None,
        limit: int = 50,
    ) -> list[ConversationResponse]:
        """
        Lists conversations for the authenticated user, ordered by updated_at descending.
        If workspace_id is supplied, validates workspace authorization and scopes results.
        """
        target_ws_id: uuid.UUID | None = None
        if workspace_id is not None and str(workspace_id).strip():
            ws = await self._validate_workspace_access(user, workspace_id)
            target_ws_id = ws.id
        else:
            self._require_global_chat_admin(user)

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
        conversation_id: uuid.UUID | str,
        user: User,
    ) -> ConversationDetailResponse:
        """
        Fetches full conversation history for the authenticated user.
        Validates workspace authorization if conversation belongs to a workspace.
        """
        conv_uuid: uuid.UUID | None = None
        try:
            conv_uuid = uuid.UUID(str(conversation_id))
        except (ValueError, TypeError, AttributeError):
            conv_uuid = None

        if not conv_uuid:
            raise HTTPException(status_code=404, detail="Conversation not found")

        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conv_uuid)
        )
        # Conversation history is private to its owner, including administrator threads.
        stmt = stmt.where(Conversation.user_id == user.id)

        conv = await self.session.scalar(stmt)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conv.workspace_id is None:
            self._require_global_chat_admin(user)
        else:
            await self._validate_workspace_access(user, conv.workspace_id)

        from rag_app.db.models import GeneratedArtifact
        from rag_app.schemas.artifacts import ArtifactMetadata, DownloadableArtifactFormat

        art_stmt = select(GeneratedArtifact).where(GeneratedArtifact.conversation_id == conv.id)
        artifacts_res = await self.session.scalars(art_stmt)
        all_artifacts = list(artifacts_res.all())

        now = datetime.now(timezone.utc)
        artifacts_by_msg: dict[str, list[ArtifactMetadata]] = {}
        unassigned_artifacts: list[ArtifactMetadata] = []
        for art in all_artifacts:
            meta = ArtifactMetadata(
                id=art.id,
                format=DownloadableArtifactFormat(art.format),
                filename=art.safe_filename,
                mime_type=art.mime_type,
                size_bytes=art.byte_size,
                download_url=f"/api/reports/{art.id}/download",
                created_at=art.created_at,
                expires_at=art.expires_at,
                is_expired=art.expires_at < now,
            )
            if art.message_id:
                artifacts_by_msg.setdefault(str(art.message_id), []).append(meta)
            else:
                unassigned_artifacts.append(meta)

        if unassigned_artifacts:
            latest_asst_id = None
            for msg in reversed(conv.messages):
                if msg.role == "assistant":
                    latest_asst_id = str(msg.id)
                    break
            if latest_asst_id:
                artifacts_by_msg.setdefault(latest_asst_id, []).extend(unassigned_artifacts)

        messages_resp = []
        for msg in conv.messages:
            m_metrics = msg.execution_metrics if isinstance(msg.execution_metrics, dict) else (
                msg.execution_metrics.model_dump() if hasattr(msg.execution_metrics, "model_dump") else {}
            ) or {}
            r_mode = m_metrics.get("routing_mode")
            r_reason = m_metrics.get("routing_reason")
            r_run_id = m_metrics.get("workflow_run_id")

            # Fallback for historical messages: if execution metrics exist on assistant message, mark as FAST_RAG
            if not r_mode and msg.role == "assistant" and (m_metrics.get("stages") or m_metrics.get("planned_strategies") or m_metrics.get("total_latency_ms")):
                r_mode = "FAST_RAG"
                r_reason = "Single-shot direct knowledge retrieval and grounded answer synthesis."

            messages_resp.append(
                ChatMessageResponse(
                    id=str(msg.id),
                    conversation_id=str(msg.conversation_id),
                    role=msg.role,
                    content=msg.content,
                    visualization=msg.visualization,
                    sources=msg.sources,
                    execution_metrics=msg.execution_metrics,
                    routing_mode=r_mode,
                    routing_reason=r_reason,
                    workflow_run_id=r_run_id,
                    artifacts=artifacts_by_msg.get(str(msg.id)),
                    created_at=msg.created_at,
                )
            )

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
        conversation_id: uuid.UUID | str,
        user: User,
    ) -> dict[str, str]:
        """
        Deletes a conversation and cascades to its messages.
        """
        conv_uuid: uuid.UUID | None = None
        try:
            conv_uuid = uuid.UUID(str(conversation_id))
        except (ValueError, TypeError, AttributeError):
            conv_uuid = None

        if not conv_uuid:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Administrators can administer the platform, but do not automatically own
        # or gain access to another user's private conversation history.
        stmt = select(Conversation).where(
            Conversation.id == conv_uuid,
            Conversation.user_id == user.id,
        )

        conv = await self.session.scalar(stmt)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conv.workspace_id is None:
            self._require_global_chat_admin(user)
        else:
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
        conversation_id: uuid.UUID | str | None,
        initial_query: str,
        workspace_id: uuid.UUID | str | None = None,
    ) -> Conversation:
        """
        Ensures a conversation exists for the request.
        If conversation_id is passed, verifies ownership and workspace authorization.
        If not found or not provided, creates a new one in the target workspace.
        """
        is_global_request = workspace_id is None or not str(workspace_id).strip()
        if is_global_request:
            self._require_global_chat_admin(user)

        # Resolve target workspace if specified
        target_ws: Workspace | None = None
        target_ws_id: uuid.UUID | None = None
        if not is_global_request:
            target_ws = await self._validate_workspace_access(user, workspace_id)
            target_ws_id = target_ws.id

        if conversation_id is not None and str(conversation_id).strip():
            conv_uuid: uuid.UUID | None = None
            try:
                conv_uuid = uuid.UUID(str(conversation_id))
            except (ValueError, TypeError, AttributeError):
                conv_uuid = None

            if not conv_uuid:
                raise HTTPException(status_code=404, detail="Conversation not found")

            stmt = select(Conversation).where(Conversation.id == conv_uuid)
            conv = await self.session.scalar(stmt)

            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Ownership authorization: Reject foreign conversation with HTTP 403 Forbidden
            if conv.user_id != user.id:
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: You do not have access to this conversation.",
                )

            if conv:
                # Validate scope match between request and existing conversation
                if is_global_request:
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
                    await self.session.refresh(conv)
                    if hasattr(user, "_sa_instance_state"):
                        await self.session.refresh(user)

                return conv

        # Auto-create new conversation
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
        await self.session.refresh(conv)
        if hasattr(user, "_sa_instance_state"):
            await self.session.refresh(user)
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
        Retrieves relevant prior conversation messages within an authorized Workspace.
        If workspace_id is None, returns empty list per Global Chat raw-message containment policy.
        """
        if workspace_id is None:
            return []
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
        query: Optional[str] = None,
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

        # Apply memory seam hook for persistent personal memory retrieval
        return await self.apply_memory_seam(conversation_id, active_history, query=query)

    async def apply_memory_seam(
        self,
        conversation_id: uuid.UUID,
        active_history: list[dict[str, str]],
        query: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Retrieves user personal memories within authorized scope and injects them
        into the active conversation context window under a strict token budget.
        """
        try:
            from rag_app.memory.retrieval import MemoryRetrievalService
            conv = await self.session.get(Conversation, conversation_id)
            if not conv:
                return active_history

            mem_resp = await MemoryRetrievalService.retrieve(
                session=self.session,
                user_id=conv.user_id,
                workspace_id=conv.workspace_id,
                query=query,
                limit_static=5,
                limit_dynamic=5,
                limit_relevant=3,
            )

            # Deduplicate by ID
            seen_ids = set()
            all_mems = []
            for mem in (mem_resp.static_memories + mem_resp.dynamic_memories + mem_resp.relevant_memories):
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    all_mems.append(mem)

            if not all_mems:
                return active_history

            # Format with strict token budget (max 300 tokens ~ 1200 characters)
            MAX_CHAR_BUDGET = 1200
            lines = []
            curr_len = 0
            for m in all_mems:
                m_type = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
                line = f"- [{m_type}] {m.content}"
                if curr_len + len(line) + 1 > MAX_CHAR_BUDGET:
                    break
                lines.append(line)
                curr_len += len(line) + 1

            if lines:
                mem_block = "\n".join(lines)
                memory_entry = {
                    "role": "system",
                    "content": f"[User Personal Preferences & Context]\n{mem_block}",
                    "name": "personal_memory",
                    "personal_memories": [
                        {
                            "id": str(m.id),
                            "content": m.content,
                            "canonical_key": m.canonical_key,
                            "memory_type": getattr(m.memory_type, "value", str(m.memory_type)),
                            "workspace_id": str(m.workspace_id) if m.workspace_id else None,
                            "user_id": str(m.user_id) if hasattr(m, "user_id") else str(conv.user_id),
                            "created_at": m.created_at.isoformat() if hasattr(m, "created_at") and m.created_at else None,
                        }
                        for m in all_mems
                    ],
                }
                return [memory_entry] + active_history
        except Exception as exc:
            logger.warning(f"apply_memory_seam_failed: {exc}")

        return active_history

    async def save_user_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        commit: bool = True,
    ) -> ChatMessage:
        msg = ChatMessage(
            conversation_id=conversation_id,
            role="user",
            content=content,
        )
        self.session.add(msg)
        if commit:
            await self.session.commit()
            await self.session.refresh(msg)
        else:
            await self.session.flush()
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
        await self.session.refresh(msg)
        return msg

