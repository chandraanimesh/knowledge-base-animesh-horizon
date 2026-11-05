"""Living User Memory Profile Compiler (Milestone 9E)

Compiles, validates, and maintains the continuously updated user profile
stored in users.profile_json from active supermemory items.
Synchronizes procedural conventions, episodic milestones, semantic knowledge,
and stable preferences deterministically and idempotently.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_app.db.models import MemoryItem, User, MemoryTombstone

logger = logging.getLogger(__name__)


def _extract_category_and_data(
    canonical_key: str,
    content: str,
    structured_data: dict[str, Any] | None,
    memory_type: str = "DYNAMIC",
) -> tuple[str, str, Any]:
    """
    Deterministically maps memory attributes to (section, key, value).
    Sections:
      - user_preferences
      - work_context
      - stable_technical_preferences
      - procedural_conventions
    """
    struct = structured_data or {}
    key_norm = (canonical_key or "").lower()
    text_norm = (content or "").lower()
    m_type_norm = (memory_type or "").upper()

    # 1. Check structured_data first
    if "section" in struct and "key" in struct:
        return struct["section"], struct["key"], struct.get("value", content)

    # 2. Check PROCEDURAL type
    if m_type_norm == "PROCEDURAL" or "convention" in key_norm or "guideline" in key_norm:
        if "format" in key_norm or "response_format" in key_norm or "json" in text_norm:
            return "procedural_conventions", "response_format", content
        if "pep" in text_norm or "coding" in text_norm or "style" in text_norm or "naming" in text_norm:
            return "procedural_conventions", "coding_conventions", content
        return "procedural_conventions", "instructions", content

    # 3. Check EPISODIC type
    if m_type_norm == "EPISODIC" or "event" in key_norm or "milestone" in key_norm:
        return "work_context", "recent_milestones", content

    # 4. Canonical key prefixes
    if "communication_style" in key_norm or "response_format" in key_norm or "verbosity" in key_norm:
        val = struct.get("value") or content
        for opt in ["concise", "detailed", "bullet point", "bulletpoint", "brief"]:
            if opt in text_norm:
                val = opt
                break
        return "user_preferences", "communication_style", val

    if "tone" in key_norm:
        val = struct.get("value") or content
        for opt in ["direct", "formal", "casual", "friendly", "academic"]:
            if opt in text_norm:
                val = opt
                break
        return "user_preferences", "tone", val

    if "technical_depth" in key_norm:
        val = struct.get("value") or content
        for opt in ["senior", "expert", "beginner", "intermediate"]:
            if opt in text_norm:
                val = opt
                break
        return "user_preferences", "technical_depth", val

    if "framework" in key_norm or "preferred_frameworks" in key_norm:
        val = struct.get("value") or content
        return "stable_technical_preferences", "preferred_frameworks", val

    if "role" in key_norm:
        val = struct.get("value") or content
        if isinstance(val, str) and ("role is " in val.lower() or "role: " in val.lower()):
            val = re.split(r"role\s*(?:is|:)\s*", val, flags=re.IGNORECASE)[-1].strip(". ")
        elif isinstance(val, str) and "i am a " in val.lower():
            val = val.split("i am a ", 1)[-1].strip(". ")
        return "work_context", "role", val

    if "department" in key_norm:
        val = struct.get("value") or content
        return "work_context", "department", val

    if "project" in key_norm or "current_projects" in key_norm:
        val = struct.get("value") or content
        return "work_context", "current_projects", val

    if "database" in key_norm:
        val = struct.get("value") or content
        for opt in ["postgresql", "postgres", "mysql", "duckdb", "sqlite", "mongodb"]:
            if opt in text_norm:
                val = "PostgreSQL" if "postgres" in opt else opt.capitalize()
                break
        return "stable_technical_preferences", "database", val

    if "python" in key_norm or "python_version" in key_norm:
        val = struct.get("value") or content
        m = re.search(r"3\.\d+", text_norm)
        if m:
            val = m.group(0)
        return "stable_technical_preferences", "python_version", val

    # 5. Text pattern heuristics fallback
    if "database" in text_norm:
        val = "PostgreSQL" if "postgres" in text_norm else "Database"
        return "stable_technical_preferences", "database", val

    if "role" in text_norm or "i am a " in text_norm or "my title is " in text_norm:
        val = content
        if "role is " in text_norm:
            val = content.split("role is ", 1)[-1].strip(". ")
        elif "i am a " in text_norm:
            val = content.split("i am a ", 1)[-1].strip(". ")
        return "work_context", "role", val

    if "department" in text_norm:
        val = content
        if "department is " in text_norm:
            val = content.split("department is ", 1)[-1].strip(". ")
        return "work_context", "department", val

    if "project" in text_norm:
        val = content
        return "work_context", "current_projects", val

    if "concise" in text_norm or "brief" in text_norm:
        return "user_preferences", "communication_style", "concise"

    if "tone" in text_norm or "direct" in text_norm:
        return "user_preferences", "tone", "direct"

    # Default to user preferences general note
    return "user_preferences", "general_notes", content


async def compile_user_profile(
    user_id: uuid.UUID,
    session: AsyncSession,
) -> dict[str, Any]:
    """
    Compiles active user memories into a consolidated living profile
    and persists it into users.profile_json.
    Strictly excludes expired, purged, or tombstoned facts.
    """
    now = datetime.now(timezone.utc)

    # Subquery for tombstoned keys
    tombstone_subquery = (
        select(MemoryTombstone.canonical_key)
        .where(MemoryTombstone.user_id == user_id)
        .scalar_subquery()
    )

    stmt = (
        select(MemoryItem)
        .where(
            MemoryItem.user_id == user_id,
            MemoryItem.status == "ACTIVE",
            MemoryItem.canonical_key.not_in(tombstone_subquery),
            or_(
                MemoryItem.retention_expires_at.is_(None),
                MemoryItem.retention_expires_at > now,
            ),
            or_(
                MemoryItem.valid_until.is_(None),
                MemoryItem.valid_until > now,
            ),
        )
        .order_by(MemoryItem.created_at.asc())
    )
    memories = (await session.execute(stmt)).scalars().all()

    user_preferences: dict[str, Any] = {}
    work_context: dict[str, Any] = {}
    stable_technical_preferences: dict[str, Any] = {}
    procedural_conventions: dict[str, Any] = {}

    for mem in memories:
        section, key, val = _extract_category_and_data(
            canonical_key=mem.canonical_key,
            content=mem.content,
            structured_data=mem.structured_data,
            memory_type=mem.memory_type,
        )

        target_dict = user_preferences
        if section == "work_context":
            target_dict = work_context
        elif section == "stable_technical_preferences":
            target_dict = stable_technical_preferences
        elif section == "procedural_conventions":
            target_dict = procedural_conventions

        # List-accumulating fields
        if key in ("preferred_frameworks", "current_projects", "recent_milestones", "instructions"):
            if key not in target_dict or not isinstance(target_dict[key], list):
                target_dict[key] = []
            if isinstance(val, list):
                for item in val:
                    if item not in target_dict[key]:
                        target_dict[key].append(item)
            else:
                if val not in target_dict[key]:
                    target_dict[key].append(val)
        else:
            # Scalar override (latest active fact wins)
            target_dict[key] = val

    now_iso = datetime.now(timezone.utc).isoformat()
    profile = {
        "user_preferences": user_preferences,
        "work_context": work_context,
        "stable_technical_preferences": stable_technical_preferences,
        "procedural_conventions": procedural_conventions,
        "last_recompiled_at": now_iso,
        "memory_version_count": len(memories),
    }

    # Persist in User.profile_json
    user = await session.get(User, user_id)
    if user:
        user.profile_json = profile
        await session.flush()
        logger.info(
            "user_living_profile_compiled",
            extra={
                "event": "user_living_profile_compiled",
                "user_id": str(user_id),
                "active_memory_count": len(memories),
            },
        )

    return profile


async def invalidate_and_recompile_user_profile(
    user_id: uuid.UUID,
    session: AsyncSession,
) -> dict[str, Any]:
    """Forces an invalidation and immediate recompilation of the user profile."""
    user = await session.get(User, user_id)
    if user:
        user.profile_json = {}
        await session.flush()

    return await compile_user_profile(user_id=user_id, session=session)
