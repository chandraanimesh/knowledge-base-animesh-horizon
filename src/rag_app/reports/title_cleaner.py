"""Automatic Export Title Cleaner

Derives clean, filesystem-safe, presentation-ready document titles from raw
user prompts by stripping conversational prefixes, format instructions,
and chatter.
"""

from __future__ import annotations

import re


CONVERSATIONAL_PREFIXES = [
    r"^(?:can\s+you\s+(?:please\s+)?)",
    r"^(?:could\s+you\s+(?:please\s+)?)",
    r"^(?:please\s+)",
    r"^(?:kindly\s+)",
    r"^(?:give\s+me\s+(?:\b(?:a|an|the)\b\s*)?)",
    r"^(?:show\s+me\s+(?:\b(?:a|an|the)\b\s*)?)",
    r"^(?:i\s+want\s+(?:to\s+see\s+)?(?:\b(?:a|an|the)\b\s*)?)",
    r"^(?:i\s+need\s+(?:\b(?:a|an|the)\b\s*)?)",
    r"^(?:generate\s+(?:\b(?:a|an|the)\b\s*)?(?:report|document|file|summary|overview)?\s*(?:on|about|for)?\s*)",
    r"^(?:create\s+(?:\b(?:a|an|the)\b\s*)?(?:report|document|file|summary|overview)?\s*(?:on|about|for)?\s*)",
    r"^(?:make\s+(?:\b(?:a|an|the)\b\s*)?(?:report|document|file|summary|overview)?\s*(?:on|about|for)?\s*)",
    r"^(?:export\s+(?:as|in|to)?\s*(?:pdf|csv|docx|json|md|markdown)?\s*(?:\b(?:a|an|the|this|that|these|those)\b\s*)?)",
    r"^(?:summarize\s+(?:\b(?:a|an|the)\b\s*)?)",
    r"^(?:summary\s+of\s+)",
    r"^(?:download\s+(?:as|in)?\s*(?:pdf|csv|docx|json|md|markdown)?\s*(?:\b(?:a|an|the|this|that)\b\s*)?)",
    r"^(?:(?:produce|prepare|draft)\s+(?:\b(?:a|an|the)\b\s*)?(?:report|document|file|summary|overview)?\s*(?:on|about|for)?\s*)",
    r"^(?:report\s+(?:on|about|for)\s*)",
]

FORMAT_SUFFIXES = [
    r"\b(?:as|in)\s+(?:a\s+)?(?:pdf|csv|docx|doc|word|excel|json|md|markdown)(?:\s+(?:format|file))?\b",
    r"\b(?:pdf|csv|docx|doc|word|excel|json|md|markdown)\s+format\b",
    r"\bformat\s+(?:pdf|csv|docx|doc|word|excel|json|md|markdown)\b",
    r"\b(?:file|export|download)\b",
]

TRIVIAL_WORDS = {
    "this", "that", "these", "those", "it", "them",
    "data", "info", "report", "file", "document", "everything"
}

SPECIAL_ACRONYMS = {
    "q1": "Q1",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Q4",
    "h1": "H1",
    "h2": "H2",
    "fy": "FY",
    "rag": "RAG",
    "llm": "LLM",
    "api": "API",
    "crm": "CRM",
    "pdf": "PDF",
    "csv": "CSV",
    "sql": "SQL",
    "hr": "HR",
    "roi": "ROI",
}


def derive_clean_title(prompt: str, max_chars: int = 60, fallback: str = "Report") -> str:
    """
    Cleans raw prompt chatter and derives a crisp, capitalized title.

    Examples:
      "Can you please give me a summary of Q3 revenue?" -> "Q3 Revenue"
      "Please export as docx the employee travel policies" -> "Employee Travel Policies"
      "can you export this?" -> "Report"
    """
    if not prompt or not prompt.strip():
        return fallback

    cleaned = prompt.strip()

    # 1. Repeatedly strip conversational leading phrases
    changed = True
    while changed:
        changed = False
        for prefix_pat in CONVERSATIONAL_PREFIXES:
            subbed = re.sub(prefix_pat, "", cleaned, flags=re.IGNORECASE).strip()
            if subbed != cleaned:
                cleaned = subbed
                changed = True

    # 2. Strip format markers from anywhere in text
    for fmt_pat in FORMAT_SUFFIXES:
        cleaned = re.sub(fmt_pat, "", cleaned, flags=re.IGNORECASE).strip()

    # 3. Strip trailing punctuation and questions
    cleaned = re.sub(r"[?!.,;:]+$", "", cleaned).strip()

    # 4. Strip illegal filesystem characters: / \ : * ? " < > |
    cleaned = re.sub(r'[/\\:*?"<>|]', " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 5. Fallback if empty, too short, or only trivial words
    words = cleaned.split()
    if not words or len(cleaned) < 2 or set(cleaned.lower().split()).issubset(TRIVIAL_WORDS):
        return fallback

    # 6. Smart Title Casing while preserving acronyms
    cased_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower in SPECIAL_ACRONYMS:
            cased_words.append(SPECIAL_ACRONYMS[w_lower])
        elif len(w) > 1 and w.isupper():
            cased_words.append(w)
        else:
            cased_words.append(w.capitalize())

    final_title = " ".join(cased_words)

    # 7. Bounded length
    if len(final_title) > max_chars:
        final_title = final_title[:max_chars].rsplit(" ", 1)[0]

    return final_title.strip() or fallback


def derive_clean_filename(prompt_or_title: str, format_ext: str, fallback: str = "Report") -> str:
    """Derives a filesystem-safe clean filename e.g. 'Q3_Revenue.pdf'."""
    title = derive_clean_title(prompt_or_title, fallback=fallback)
    clean_ext = format_ext.lower().lstrip(".")
    safe_name = re.sub(r"\s+", "_", title.strip())
    return f"{safe_name}.{clean_ext}"
