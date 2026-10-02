import logging
import re
import uuid
from typing import Any
from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_app.db.models import ChatMessage, Conversation, Department, User, Workspace, WorkspaceMember
from rag_app.security.authorization import authorize

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
    Retrieves relevant conversation messages from prior threads within the SAME Workspace.
    Strictly scoped to the authorized Workspace boundary.
    """

    @classmethod
    async def retrieve_workspace_threads(
        cls,
        session: AsyncSession,
        user: User,
        workspace_id: uuid.UUID,
        query: str,
        current_conversation_id: uuid.UUID | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Searches previous threads in the given workspace for relevant messages.
        Validates user authorization for the workspace.
        """
        # 1. Authorize Workspace access
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

        if user.role != "ADMIN":
            dept = await session.get(Department, ws.department_id)
            if not dept or not (await authorize(session, user, "VIEW", dept.code)):
                logger.warning(f"User {user.id} unauthorized for department {dept.code if dept else 'None'}.")
                return []
            ws_member = await session.scalar(
                select(WorkspaceMember).where(
                    WorkspaceMember.workspace_id == ws.id,
                    WorkspaceMember.user_id == user.id,
                )
            )
            if not ws_member:
                logger.warning(f"User {user.id} is not a member of workspace {ws.name}.")
                return []

        effective_ws_id = ws.id

        # 2. Extract search terms
        terms = extract_search_terms(query)
        if not terms:
            return []

        # 3. Query other conversations within this workspace
        conv_stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(
                Conversation.workspace_id == effective_ws_id,
            )
        )
        if current_conversation_id is not None:
            conv_stmt = conv_stmt.where(Conversation.id != current_conversation_id)

        conv_result = await session.execute(conv_stmt)
        conversations = conv_result.scalars().all()
        if not conversations:
            return []

        # 4. Score messages based on keyword overlap & conversation title relevance
        scored_messages: list[tuple[float, ChatMessage, Conversation]] = []

        query_lower = query.lower()

        for conv in conversations:
            title_lower = conv.title.lower() if conv.title else ""
            title_match_count = sum(1 for term in terms if term in title_lower)

            for msg in conv.messages:
                if msg.role not in ("user", "assistant"):
                    continue
                content_lower = msg.content.lower()

                # Calculate term frequency / keyword overlap
                content_match_count = sum(1 for term in terms if term in content_lower)

                if content_match_count == 0 and title_match_count == 0:
                    continue

                # Scoring formula:
                # Content match: 2.0 per matched term
                # Title match bonus: 1.5 per matched term
                # Full phrase match bonus (if any 2+ consecutive terms appear): 3.0
                score = (content_match_count * 2.0) + (title_match_count * 1.5)

                for i in range(len(terms) - 1):
                    bigram = f"{terms[i]} {terms[i+1]}"
                    if bigram in content_lower:
                        score += 3.0

                scored_messages.append((score, msg, conv))

        if not scored_messages:
            return []

        # Sort by relevance score descending, then by creation date descending
        scored_messages.sort(
            key=lambda item: (item[0], item[1].created_at or datetime.min),
            reverse=True,
        )

        top_candidates = scored_messages[:limit]

        retrieved: list[dict[str, Any]] = [
            {
                "thread_id": str(conv.id),
                "thread_title": conv.title,
                "message_id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "score": score,
            }
            for score, msg, conv in top_candidates
        ]

        logger.info(
            f"Retrieved {len(retrieved)} relevant workspace thread messages for workspace '{ws.name}'",
            extra={"workspace_id": str(workspace_id), "count": len(retrieved)},
        )
        return retrieved
