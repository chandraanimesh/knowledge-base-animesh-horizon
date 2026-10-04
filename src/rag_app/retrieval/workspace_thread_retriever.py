import logging
import re
import uuid
from typing import Any
from datetime import datetime

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.db.models import ChatMessage, Conversation, Department, Permission, User, UserDepartment, UserPermission, Workspace, WorkspaceMember
from rag_app.security.authorization import authorize, get_allowed_workspaces

logger = logging.getLogger(__name__)

STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "tell", "show", "give", "discuss", "discussed",
    "earlier", "previous", "prior", "last", "ki", "ke", "ka", "kya", "hai", "tha",
    "thi", "huye", "baat", "saath", "karein", "batao", "bataiye",
})


def extract_search_terms(query: str) -> list[str]:
    """
    Extracts informative keywords from query, filtering out common stop words.
    """
    raw_tokens = re.findall(r"[A-Za-z0-9_%'-]+", query)
    terms: list[str] = []
    for token in raw_tokens:
        clean = token.strip("'-_ %").lower()
        if len(clean) >= 2 and clean not in STOP_WORDS:
            terms.append(clean)
    return terms


class WorkspaceThreadRetriever:
    """
    Retrieves relevant conversation messages from prior threads within:
    - A single Workspace (Workspace Chat), or
    - Across all authorized Workspaces (Global Chat).

    Strictly applies deterministic server-side RBAC filters at the query level BEFORE ranking.
    """

    @classmethod
    async def retrieve_workspace_threads(
        cls,
        session: AsyncSession,
        user: User,
        workspace_id: uuid.UUID | None = None,
        query: str = "",
        current_conversation_id: uuid.UUID | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Searches previous threads in authorized workspace(s) for relevant messages.
        If workspace_id is provided: scopes search to that single workspace after verifying authorization.
        If workspace_id is None: searches across all workspaces the user is authorized to access.
        """
        if not user.is_active:
            return []

        authorized_workspaces: dict[uuid.UUID, str] = {}

        # -------------------------------------------------------------
        # 1. SCOPE RESOLUTION & AUTHORIZATION (BEFORE RETRIEVAL)
        # -------------------------------------------------------------
        if workspace_id is not None:
            # Workspace Chat: Validate single workspace authorization
            ws = await session.scalar(
                select(Workspace).where(
                    Workspace.id == workspace_id,
                    Workspace.deleted_at.is_(None),
                    Workspace.is_active == True,
                )
            )
            if not ws:
                ws = await session.scalar(
                    select(Workspace).where(
                        Workspace.department_id == workspace_id,
                        Workspace.deleted_at.is_(None),
                        Workspace.is_active == True,
                    )
                )
            if not ws:
                logger.warning(f"Workspace {workspace_id} not found or inactive during thread retrieval.")
                return []

            dept = await session.get(Department, ws.department_id)
            if not dept or not dept.is_active or dept.deleted_at is not None:
                logger.warning(f"Department for workspace {ws.name} is inactive or deleted.")
                return []

            if user.role != "ADMIN":
                if user.role != "USER":
                    return []
                # Department VIEW permission check
                is_dept_auth = await authorize(session, user, "VIEW", dept.code)
                if not is_dept_auth:
                    logger.warning(f"User {user.id} unauthorized for department {dept.code}.")
                    return []
                # Workspace membership check
                ws_member = await session.scalar(
                    select(WorkspaceMember).where(
                        WorkspaceMember.workspace_id == ws.id,
                        WorkspaceMember.user_id == user.id,
                    )
                )
                if not ws_member:
                    logger.warning(f"User {user.id} is not a member of workspace {ws.name}.")
                    return []

            authorized_workspaces[ws.id] = ws.name

        else:
            # Global Chat: Retrieve across all workspaces the user is authorized to access
            if user.role == "ADMIN":
                stmt = (
                    select(Workspace.id, Workspace.name)
                    .join(Department, Workspace.department_id == Department.id)
                    .where(
                        Workspace.is_active == True,
                        Workspace.deleted_at.is_(None),
                        Department.is_active == True,
                        Department.deleted_at.is_(None),
                    )
                )
                rows = (await session.execute(stmt)).all()
                authorized_workspaces = {ws_id: ws_name for ws_id, ws_name in rows}
            elif user.role == "USER":
                stmt = (
                    select(Workspace.id, Workspace.name)
                    .join(Department, Workspace.department_id == Department.id)
                    .join(
                        UserDepartment,
                        and_(
                            UserDepartment.user_id == user.id,
                            UserDepartment.department_id == Department.id,
                        ),
                    )
                    .join(
                        UserPermission,
                        and_(
                            UserPermission.user_id == user.id,
                            UserPermission.department_id == Department.id,
                        ),
                    )
                    .join(
                        Permission,
                        and_(
                            Permission.id == UserPermission.permission_id,
                            Permission.code == "VIEW",
                            Permission.is_active == True,
                        ),
                    )
                    .join(
                        WorkspaceMember,
                        and_(
                            WorkspaceMember.workspace_id == Workspace.id,
                            WorkspaceMember.user_id == user.id,
                        ),
                    )
                    .where(
                        Workspace.is_active == True,
                        Workspace.deleted_at.is_(None),
                        Department.is_active == True,
                        Department.deleted_at.is_(None),
                    )
                )
                rows = (await session.execute(stmt)).all()
                authorized_workspaces = {ws_id: ws_name for ws_id, ws_name in rows}
            else:
                return []

        if not authorized_workspaces:
            return []

        # -------------------------------------------------------------
        # 2. KEYWORD EXTRACTION
        # -------------------------------------------------------------
        terms = extract_search_terms(query)
        if not terms:
            return []

        # -------------------------------------------------------------
        # 3. DATABASE RETRIEVAL (STRICTLY CONSTRAINED TO AUTHORIZED WS)
        # -------------------------------------------------------------
        target_ws_ids = list(authorized_workspaces.keys())
        conv_stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.workspace_id.in_(target_ws_ids),
            )
        )
        if current_conversation_id is not None:
            conv_stmt = conv_stmt.where(Conversation.id != current_conversation_id)

        conv_result = await session.execute(conv_stmt)
        conversations = conv_result.scalars().all()
        if not conversations:
            return []

        # -------------------------------------------------------------
        # 4. RELEVANCE SCORING
        # -------------------------------------------------------------
        scored_messages: list[tuple[float, ChatMessage, Conversation, str]] = []

        for conv in conversations:
            ws_name = authorized_workspaces.get(conv.workspace_id, "")
            ws_name_lower = ws_name.lower()
            title_lower = conv.title.lower() if conv.title else ""
            title_match_count = sum(1 for term in terms if term in title_lower)
            ws_match_count = sum(1 for term in terms if term in ws_name_lower)

            for msg in conv.messages:
                if msg.role not in ("user", "assistant"):
                    continue
                content_lower = msg.content.lower()

                content_match_count = sum(1 for term in terms if term in content_lower)

                # A candidate must have at least one keyword match in its content or thread title.
                # Workspace name alone must not pull unrelated messages.
                if content_match_count == 0 and title_match_count == 0:
                    continue

                # Scoring formula:
                # Content match: 3.0 per matched term
                # Title match bonus: 2.0 per matched term
                # Workspace name match bonus: 1.0 per matched term (only for relevant messages)
                # Bigram / consecutive terms match bonus: 4.0
                score = (content_match_count * 3.0) + (title_match_count * 2.0) + (ws_match_count * 1.0)

                for i in range(len(terms) - 1):
                    bigram = f"{terms[i]} {terms[i+1]}"
                    if bigram in content_lower:
                        score += 4.0

                scored_messages.append((score, msg, conv, ws_name))

        if not scored_messages:
            return []

        # Sort by relevance score descending, then by creation date descending
        scored_messages.sort(
            key=lambda item: (item[0], item[1].created_at or datetime.min),
            reverse=True,
        )

        # In Global Chat across multiple workspaces, ensure balanced diversity so one workspace does not starve others
        if workspace_id is None and len(authorized_workspaces) > 1:
            per_ws_cap = max(2, limit // len(authorized_workspaces) + 1)
            ws_counts: dict[Any, int] = {}
            diverse_candidates = []
            for item in scored_messages:
                ws_id = item[2].workspace_id
                if ws_counts.get(ws_id, 0) < per_ws_cap:
                    diverse_candidates.append(item)
                    ws_counts[ws_id] = ws_counts.get(ws_id, 0) + 1
                if len(diverse_candidates) >= limit:
                    break
            if len(diverse_candidates) < limit:
                for item in scored_messages:
                    if item not in diverse_candidates:
                        diverse_candidates.append(item)
                        if len(diverse_candidates) >= limit:
                            break
            top_candidates = diverse_candidates
        else:
            top_candidates = scored_messages[:limit]

        retrieved: list[dict[str, Any]] = [
            {
                "source_type": "conversation",
                "workspace_id": str(conv.workspace_id) if conv.workspace_id else None,
                "workspace_name": ws_name,
                "thread_id": str(conv.id),
                "conversation_id": str(conv.id),
                "thread_title": conv.title,
                "message_id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "score": score,
            }
            for score, msg, conv, ws_name in top_candidates
        ]

        logger.info(
            f"Retrieved {len(retrieved)} relevant conversation thread messages across {len(authorized_workspaces)} authorized workspace(s)",
            extra={"authorized_workspaces_count": len(authorized_workspaces), "retrieved_count": len(retrieved)},
        )
        return retrieved
