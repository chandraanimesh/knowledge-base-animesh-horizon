import re
import uuid
import zoneinfo
from datetime import datetime, time, timedelta, timezone
from typing import Optional
from pydantic import BaseModel


class TaskIntent(BaseModel):
    is_task_intent: bool = False
    intent_type: Optional[str] = None  # "QUERY_DUE_TODAY", "QUERY_OVERDUE", "QUERY_COMPANY", "QUERY_UPCOMING", "CREATE_TASK"
    company_name: Optional[str] = None
    target_date: Optional[datetime] = None
    action_text: Optional[str] = None
    is_ambiguous: bool = False
    clarification_message: Optional[str] = None


def detect_task_intent(query: str, user_timezone: str = "UTC") -> TaskIntent:
    """
    Deterministically detects structured task and follow-up intents from natural language.
    Resolves dates using the authenticated user's timezone.
    """
    q = query.strip()
    q_lower = q.lower()

    try:
        tz = zoneinfo.ZoneInfo(user_timezone)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")

    now_local = datetime.now(timezone.utc).astimezone(tz)

    # 1. DUE TODAY queries
    # e.g. "what do i need to follow up on today?", "what follow ups are due today", "tasks due today", "today's follow-ups"
    if (
        ("follow up" in q_lower or "follow-up" in q_lower or "task" in q_lower or "commitment" in q_lower or "remind" in q_lower)
        and ("today" in q_lower or "due today" in q_lower)
        and not q_lower.startswith("remind me")
    ):
        return TaskIntent(
            is_task_intent=True,
            intent_type="QUERY_DUE_TODAY",
        )

    # 2. OVERDUE queries
    # e.g. "which follow-ups are overdue?", "what is overdue", "overdue tasks", "overdue followups"
    if "overdue" in q_lower and ("follow" in q_lower or "task" in q_lower or "commitment" in q_lower or "what" in q_lower or "which" in q_lower):
        return TaskIntent(
            is_task_intent=True,
            intent_type="QUERY_OVERDUE",
        )

    # 3. UPCOMING / GENERAL FOLLOW-UPS
    # e.g. "what did Sales agree to follow up on?", "upcoming tasks", "next follow-ups"
    if (
        ("upcoming" in q_lower or "next" in q_lower or "agree to follow up" in q_lower or "agreed to follow up" in q_lower)
        and ("follow" in q_lower or "task" in q_lower or "agree" in q_lower)
        and not any(f" {k} " in q_lower for k in ["for", "about"])
    ):
        return TaskIntent(
            is_task_intent=True,
            intent_type="QUERY_UPCOMING",
            action_text="Sales" if "sales" in q_lower else None,
        )

    # 4. COMPANY COMMITMENTS, CRM STATUS & EVIDENCE FUSION queries
    # e.g. "what commitments did i make to ABC Company?", "what did Sales agree to do for ABC Company?",
    # "what is the current status of ABC Company?", "what was the latest activity for ABC Company?",
    # "what follow-up tasks are associated with ABC Company?", "summarize ABC Company's recent meetings and decisions",
    # "what is known about ABC Company, and which evidence supports it?"
    comp_match = re.search(
        r"(?:commitments?\s+(?:did\s+i\s+make\s+)?to|tasks?\s+(?:for|(?:are\s+)?associated\s+with)|follow-?ups?\s+(?:tasks?\s+)?(?:for|with|(?:are\s+)?associated\s+with)|agree(?:d)?\s+(?:to\s+(?:do|follow\s+up\s+on)\s+)?for|agree(?:d)?\s+to\s+follow\s+up\s+on\s+|known\s+about|current\s+status\s+of|status\s+of|latest\s+activity\s+(?:for|of))\s+([A-Za-z0-9\s\.\,\-]+?)(?:,|\?|$|\.|\n)",
        q,
        re.IGNORECASE,
    )
    if not comp_match:
        comp_match = re.search(
            r"(?:summarize\s+)([A-Za-z0-9\s\.\,\-]+?)(?:'s\s+recent|'s|\?|$|\.|\n)",
            q,
            re.IGNORECASE,
        )

    if comp_match:
        extracted_comp = comp_match.group(1).strip()
        # Clean trailing punctuation and common stop phrases
        extracted_comp = re.sub(r"[\?\.\!]$", "", extracted_comp).strip()
        extracted_comp = re.sub(r",?\s*(?:and\s+which\s+evidence\s+supports\s+it|and\s+decisions)$", "", extracted_comp, flags=re.IGNORECASE).strip()
        if extracted_comp and extracted_comp.lower() not in ("me", "today", "tomorrow", "this week", "it", "sales"):
            return TaskIntent(
                is_task_intent=True,
                intent_type="QUERY_COMPANY",
                company_name=extracted_comp,
            )

    # 5. CREATE TASK / REMINDER queries
    # e.g. "remind me to contact ABC tomorrow", "create a task to email John", "add task: call client"
    if q_lower.startswith("remind me") or q_lower.startswith("create task") or q_lower.startswith("add task"):
        # Check for ambiguity
        # e.g. "remind me", "remind me later", "remind me to follow up" without recipient/date
        reminder_body = re.sub(r"^(remind me to|remind me|create task to|create task|add task to|add task:?)\s*", "", q, flags=re.IGNORECASE).strip()
        if not reminder_body or reminder_body.lower() in ("later", "soon", "something", "follow up"):
            return TaskIntent(
                is_task_intent=True,
                intent_type="CREATE_TASK",
                is_ambiguous=True,
                clarification_message="Could you specify the exact task action and when you would like it to be due (e.g. `Remind me to contact ABC tomorrow`)?",
            )

        # Parse date if present
        target_due = None
        cleaned_action = reminder_body

        if "tomorrow" in reminder_body.lower():
            tomorrow_date = now_local.date() + timedelta(days=1)
            # Default to 9:00 AM local time tomorrow
            local_due = datetime.combine(tomorrow_date, time(hour=9, minute=0), tzinfo=tz)
            target_due = local_due.astimezone(timezone.utc)
            cleaned_action = re.sub(r"\b(tomorrow|by tomorrow)\b", "", cleaned_action, flags=re.IGNORECASE).strip()

        elif "today" in reminder_body.lower():
            # Default to 5:00 PM local time today
            local_due = datetime.combine(now_local.date(), time(hour=17, minute=0), tzinfo=tz)
            target_due = local_due.astimezone(timezone.utc)
            cleaned_action = re.sub(r"\b(today|by today)\b", "", cleaned_action, flags=re.IGNORECASE).strip()

        cleaned_action = cleaned_action.strip().rstrip(".,?!")

        # Extract optional company name
        comp_extracted = None
        comp_pat = re.search(r"(?:contact|call|email|reach out to|meet|follow up with)\s+([A-Za-z0-9\s]+?)(?:\s+by|\s+at|\s+on|$)", cleaned_action, re.IGNORECASE)
        if comp_pat:
            potential_comp = comp_pat.group(1).strip()
            if potential_comp and len(potential_comp) <= 50 and potential_comp.lower() not in ("him", "her", "them", "someone"):
                comp_extracted = potential_comp

        final_title = cleaned_action
        if not final_title:
            final_title = reminder_body.strip()

        formatted_title = (final_title[0].upper() + final_title[1:]) if final_title else ""

        return TaskIntent(
            is_task_intent=True,
            intent_type="CREATE_TASK",
            action_text=formatted_title,
            target_date=target_due,
            company_name=comp_extracted,
        )

    return TaskIntent(is_task_intent=False)
