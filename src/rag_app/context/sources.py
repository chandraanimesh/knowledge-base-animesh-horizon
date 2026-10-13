import re
import uuid
from typing import Any, Optional


def format_personal_memory_label(memory_text: Optional[str], max_len: int = 120) -> str:
    """
    Formats the display label for a personal memory source item:
    🧠 Personal Memory: "<memory text>"

    Rules:
    - Exact prefix: '🧠 Personal Memory: "'
    - Safely formats quotes (internal double quotes replaced by single quotes)
    - Truncates long text gracefully with '...' before closing quote
    - Fallback for empty or missing memory text
    - Idempotent: will not double-prefix if already formatted
    """
    if not memory_text or not str(memory_text).strip():
        text = "Saved personal context"
    else:
        text = str(memory_text).strip()

    # Idempotent check: if already formatted with the exact prefix and closed with quote
    if text.startswith('🧠 Personal Memory: "') and text.endswith('"'):
        return text
    if text.startswith("🧠 Personal Memory:"):
        return text

    # Escape internal double quotes so outer quotes remain valid
    safe_inner = text.replace('"', "'")

    # Clean truncation if needed
    if len(safe_inner) > max_len:
        safe_inner = safe_inner[:max_len].rstrip() + "..."

    return f'🧠 Personal Memory: "{safe_inner}"'


def get_source_canonical_key(source: dict[str, Any], message_level: bool = False) -> tuple:
    """
    Derives a canonical stable identity key for deduplicating outward-facing source items.
    """
    s_type = str(source.get("type") or source.get("source_kind") or source.get("source_type") or "unknown").lower()

    if s_type in ("thread", "conversation"):
        t_id = source.get("thread_id") or source.get("conversation_id")
        m_id = source.get("message_id") or source.get("source_id")
        if message_level and m_id:
            return ("thread", str(t_id or ""), str(m_id))
        elif t_id:
            return ("thread", str(t_id))
        elif m_id:
            return ("thread", str(m_id))
        else:
            return ("thread", str(source.get("title") or ""))

    elif s_type in ("memory", "personal_memory"):
        mem_id = source.get("source_id") or source.get("id") or source.get("canonical_key")
        if mem_id:
            return ("memory", str(mem_id))
        return ("memory", str(source.get("title") or ""))

    elif s_type in ("document", "shared_knowledge"):
        chunk_id = source.get("chunk_id") or source.get("source_id") or source.get("id")
        if chunk_id:
            return (s_type, str(chunk_id))
        doc_id = source.get("document_id")
        p_num = source.get("page_number")
        if doc_id:
            return (s_type, str(doc_id), str(p_num or ""))
        return (s_type, str(source.get("title") or ""))

    else:
        # Fallback to specific IDs if present
        s_id = source.get("source_id") or source.get("id") or source.get("chunk_id") or source.get("thread_id")
        if s_id:
            return (s_type, str(s_id))
        return (s_type, str(source.get("title") or ""))


def deduplicate_sources_payload(
    sources: list[dict[str, Any]],
    message_level: bool = False,
) -> list[dict[str, Any]]:
    """
    Deduplicates repeated source items in outward-facing sources_payload:
    - Retains first-seen order.
    - Uses stable identity from get_source_canonical_key.
    - Preserves all metadata and original source dictionary attributes.
    - Idempotent: repeated runs produce identical output.
    - Does not alter underlying evidence passed to LLM.
    """
    if not sources:
        return []

    seen_keys: set[tuple] = set()
    deduped: list[dict[str, Any]] = []

    for item in sources:
        key = get_source_canonical_key(item, message_level=message_level)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)

    return deduped


def build_personal_memory_source_item(
    memory_id: Any,
    memory_text: str,
    user_id: Optional[Any] = None,
    workspace_id: Optional[Any] = None,
    canonical_key: Optional[str] = None,
    created_at: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Constructs a personal memory source item conforming to the existing sources_payload contract
    with the exact required display label.
    """
    formatted_label = format_personal_memory_label(memory_text)
    return {
        "type": "memory",
        "source_type": "personal_memory",
        "source_id": str(memory_id) if memory_id else str(uuid.uuid4()),
        "title": formatted_label,
        "excerpt": memory_text or "",
        "user_id": str(user_id) if user_id else None,
        "workspace_id": str(workspace_id) if workspace_id else None,
        "canonical_key": canonical_key,
        "created_at": created_at,
    }
