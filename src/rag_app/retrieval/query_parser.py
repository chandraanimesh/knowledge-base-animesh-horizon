import json
import logging
import re

from langsmith import traceable

from rag_app.providers.base.llm import LLMProvider
from rag_app.generation.llm_factory import get_llm
from rag_app.schemas.artifacts import DownloadableArtifactFormat, OutputPreferences
from rag_app.schemas.rag import (
    CompanyFilters,
    EvidenceNeed,
    InformationNeed,
    PresentationNeed,
    QueryPlan,
    RelationshipQuery,
)
from rag_app.core.tracing import track

logger = logging.getLogger(__name__)


# ============================================================
# CONVERSATIONAL QUERY DETECTION & DETERMINISTIC RELATIONSHIP EXTRACTION
# ============================================================

GREETING_PATTERNS = [
    re.compile(r"^(hi|hello|hey)(\s+[a-zA-Z0-9_-]+)?[!.?]*$", re.IGNORECASE),
    re.compile(r"^(hi|hello|hey)\s+there[!.?]*$", re.IGNORECASE),
    re.compile(r"^good\s+(morning|afternoon|evening|night)[!.?]*$", re.IGNORECASE),
    re.compile(r"^how\s+are\s+you[!.?]*$", re.IGNORECASE),
    re.compile(r"^how\s+are\s+you\s+doing[!.?]*$", re.IGNORECASE),
]


def is_greeting(query: str) -> bool:
    """
    Deterministically detects simple conversational greetings.
    """
    normalized = " ".join(query.strip().split())
    if not normalized:
        return False
    return any(pattern.fullmatch(normalized) for pattern in GREETING_PATTERNS)


HISTORY_PATTERNS = [
    re.compile(r"\b(?:what|which|how|why)\b.*\b(?:did\s+(?:i|we|you)\s+(?:ask|say|tell|mention|discuss|talk\s+about|decide)|(?:i|we|you)\s+(?:asked|said|told|mentioned|discussed|decided|talked\s+about))\b", re.IGNORECASE),
    re.compile(r"\b(?:what|which)\s+(?:was|were)\s+(?:my|your|our)\s+(?:previous|earlier|last|prior)\s+(?:question|message|turn|prompt|response|discussion|point)\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+did\s+(?:i|you|we)\s+(?:ask|say|tell|discuss|mention|talk\s+about|decide))\b", re.IGNORECASE),
    re.compile(r"\b(?:can\s+you\s+remind\s+me|remind\s+me|tell\s+me)\s+(?:what\s+(?:i|you|we)|about\s+what\s+(?:i|we))\s+(?:asked|said|discussed|mentioned|talked\s+about)\b", re.IGNORECASE),
    re.compile(r"\b(?:earlier|previously)\s+in\s+(?:this|our)\s+(?:chat|conversation|thread)\b", re.IGNORECASE),
    re.compile(r"\b(?:which\s+of\s+(?:those|these|them)\s+did\s+we\s+(?:discuss|talk\s+about|mention))\b", re.IGNORECASE),
]


def is_history_query(query: str) -> bool:
    """
    Deterministically detects conversation-history oriented queries.
    """
    normalized = " ".join(query.strip().split())
    if not normalized:
        return False
    if re.search(r"\b(?:before|after|in|during)\s+(?:19|20)\d{2}\b", normalized, re.IGNORECASE):
        return False
    if re.search(r"\b(?:before|after)\s+(?:launch|implementation|release|deployment|start|ready|completion|deadline|production)\b", normalized, re.IGNORECASE):
        return False
    if re.search(r"\b(?:last|previous|prior)\s+(?:quarter|year|month|week|fiscal|release|sprint|version|edition)\b", normalized, re.IGNORECASE):
        return False
    for pattern in HISTORY_PATTERNS:
        if pattern.search(normalized):
            return True
    return False


def is_substantive_format_topic(query: str, format_name: str) -> bool:
    """
    Returns True if the format word (e.g. 'pdf', 'json', 'csv', 'docx') is used as a substantive topic
    (e.g., 'policy on pdf', 'json schema validation') rather than an export/presentation instruction.
    """
    q_lower = query.lower()
    # If the query contains an explicit export action verb linked to the format, it's not merely a policy topic
    has_export_verb = bool(re.search(
        r"\b(?:generate|export|download|create|make|bana|kar\s+do|karo|do)\b.*?\b" + format_name + r"\b|\b" + format_name + r"\b.*?\b(?:generate|export|download|create|bana|kar\s+do|karo|do)\b",
        q_lower,
    ))
    if has_export_verb:
        return False

    topic_patterns = [
        rf"\b(?:policy\s+(?:on|about|for)|rules?\s+(?:on|for)|guidelines?\s+(?:on|for)|standard\s+(?:on|for)|retention\s+of|about|understanding)\s+.*?\b{format_name}\b",
        rf"\b{format_name}\s+(?:schema|retention|parser|validation|specification|format\s+rules|files?)\b",
    ]
    return any(bool(re.search(pat, q_lower)) for pat in topic_patterns)


def parse_output_preferences(query: str, explicit: OutputPreferences | None = None) -> OutputPreferences:
    """
    Deterministically extracts multi-output preferences from the user query.
    Precedence rule: If explicit API preferences are provided, explicit settings override inferred values.
    """
    if explicit is not None:
        return explicit.model_copy()

    q = (query or "").strip()
    q_lower = q.lower()

    include_chat_summary = True
    include_inline_json = False
    include_inline_table = False
    include_chart = False
    download_formats: list[DownloadableArtifactFormat] = []

    # Check inline JSON
    if not is_substantive_format_topic(q, "json"):
        if re.search(r"\b(?:inline\s+json|(?:detailed\s+)?records?\s+(?:in\s+)?json|json\s+format|json\s+mein)\b", q_lower):
            include_inline_json = True

    # Check inline table
    if re.search(r"\b(?:in|as)\s+(?:a\s+)?(?:markdown\s+)?table\b|\btable\b|\btabular\b|table\s+format|tabular\s+format|table\s+mein", q_lower):
        include_inline_table = True

    # Check chart
    if re.search(r"\b(?:bar|line|pie|scatter|donut|area)\s+chart\b|\bchart\b|\bgraph\b|\bvisualize\b", q_lower):
        include_chart = True

    # Check downloadable PDF
    if not is_substantive_format_topic(q, "pdf"):
        if re.search(r"\bpdf\s+(?:report|file|export|download)\b|\b(?:generate|export|download|create|bana)\s+.*?\bpdf\b|\bpdf\s+bhi\b|\bek\s+pdf\b", q_lower):
            if DownloadableArtifactFormat.PDF not in download_formats:
                download_formats.append(DownloadableArtifactFormat.PDF)

    # Check downloadable CSV
    if not is_substantive_format_topic(q, "csv"):
        if re.search(r"\bcsv\s+(?:report|file|export|download)\b|\b(?:generate|export|download|create|bana)\s+.*?\bcsv\b|\bcsv\s+mein\b|\bexport\s+(?:as\s+)?csv\b", q_lower):
            if DownloadableArtifactFormat.CSV not in download_formats:
                download_formats.append(DownloadableArtifactFormat.CSV)

    # Check downloadable DOCX / Word
    if not is_substantive_format_topic(q, "docx") and not is_substantive_format_topic(q, "word"):
        if re.search(r"\b(?:docx|word\s+doc(?:ument)?)\s+(?:report|file|export|download|document|doc)\b|\b(?:generate|export|download|create|bana)\s+.*?\b(?:docx|word\s+doc(?:ument)?)\b", q_lower):
            if DownloadableArtifactFormat.DOCX not in download_formats:
                download_formats.append(DownloadableArtifactFormat.DOCX)

    # Check downloadable JSON file
    if not is_substantive_format_topic(q, "json"):
        if re.search(
            r"\b(?:download|export|generate|create|bana)\s+.*?\bjson\b"
            r"|\b(?:download|export)\s+(?:as\s+)?json\b"
            r"|\bjson\s+(?:report|file|export|download|document)\b"
            r"|\bek\s+json\s+file\b",
            q_lower,
        ):
            if DownloadableArtifactFormat.JSON not in download_formats:
                download_formats.append(DownloadableArtifactFormat.JSON)

    # Check all formats wildcard
    if re.search(r"\b(?:all\s+(?:4\s+)?(?:downloadable\s+|file\s+)?formats|all\s+4\s+formats|all\s+formats|every\s+format)\b", q_lower):
        for f in (DownloadableArtifactFormat.PDF, DownloadableArtifactFormat.CSV, DownloadableArtifactFormat.DOCX, DownloadableArtifactFormat.JSON):
            if f not in download_formats:
                download_formats.append(f)

    return OutputPreferences(
        include_chat_summary=include_chat_summary,
        include_inline_json=include_inline_json,
        include_inline_table=include_inline_table,
        include_chart=include_chart,
        download_formats=download_formats,
    )


def strip_presentation_instructions(query: str) -> str:
    """
    Strips presentation, formatting, visualization, and chart instructions
    from the query string so semantic retrieval targets factual content.
    Generic, domain-agnostic, and preserves substantive topic keywords.
    """
    if not query:
        return query

    cleaned = query
    export_patterns = [
        r"(?:,\s*)?(?:aur\s+)?(?:ek\s+)?pdf\s+report\s+(?:bhi\s+)?(?:generate|bana|create|export)\s*(?:kar\s+do|karo|do)?",
        r"(?:,\s*)?(?:mujhe\s+)?summary\s+(?:chat\s+mein(?:\s+do)?|in\s+(?:the\s+)?chat|mein\s+do)",
        r"(?:,\s*)?(?:detailed\s+)?records?\s+(?:in\s+)?json\s+(?:mein(?:\s+do)?|format)?",
        r"\b(?:generate|export|create|download|make)\s+(?:a\s+)?(?:pdf|csv|docx|word|json)?\s*(?:report|file|document|export)\b",
        r"\b(?:in|as)\s+(?:a\s+)?(?:pdf|csv|docx|word)\s*(?:report|file|format)?\b",
        r"\b(?:detailed\s+)?records\s+in\s+json\b",
        r"\bsummary\s+in\s+chat\b",
        r"\b(?:show|render|visualize|display|draw|plot)\s+(?:this\s+)?(?:as\s+)?(?:a\s+)?(?:bar|line|pie|scatter|donut|area)?\s*(?:chart|graph|table|timeline)\b",
        r"\bas\s+(?:a\s+)?(?:bar|line|pie|scatter|donut|area)?\s*(?:chart|graph|table|timeline|markdown\s+table)\b",
        r"\bin\s+(?:a\s+)?(?:bar|line|pie|scatter|donut|area)?\s*(?:chart|graph|table|timeline|grid|markdown\s+table)\b",
        r"\b(?:make|generate|build|create)\s+(?:a\s+)?(?:bar|line|pie|scatter|donut|area)?\s*(?:chart|graph|table|timeline)\b",
        r"\b(?:as|in)\s+(?:a\s+)?(?:markdown\s+)?table\b",
        r"\b(?:visualize|plot)\b",
    ]
    for pat in export_patterns:
        match = re.search(pat, cleaned, flags=re.IGNORECASE)
        if match:
            matched_str = match.group(0).lower()
            if ("pdf" in matched_str and is_substantive_format_topic(query, "pdf")) or \
               ("json" in matched_str and is_substantive_format_topic(query, "json")):
                continue
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

    cleaned = " ".join(cleaned.strip().split())
    # Clean up leading/trailing prepositions or articles or punctuation left over
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+(?:as|in|with|using|like|aur|and)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("?.,!;: ")
    return cleaned if cleaned else query


def detect_web_search_intent(query: str) -> tuple[bool, str | None, bool]:
    """
    Deterministically evaluates whether web research is needed, why, and whether internal
    retrieval is also needed.
    Returns: (web_search_needed: bool, web_search_reason: str | None, internal_retrieval_needed: bool)
    """
    if not query:
        return False, None, True

    q_lower = query.lower().strip()

    # 1. Explicit web-search requests
    explicit_patterns = [
        r"\b(?:search\s+(?:the\s+)?web|web\s+search|google|look\s+up\s+online|search\s+online|browse\s+the\s+web|find\s+on\s+(?:the\s+)?(?:web|internet))\b",
        r"\b(?:search\s+internet|internet\s+search|check\s+online|find\s+online)\b",
    ]
    for pat in explicit_patterns:
        if re.search(pat, q_lower):
            has_internal_ref = bool(re.search(r"\b(?:our\s+docs?|internal|company|policy|our\s+files?|repo)\b", q_lower))
            return True, "explicit_request", has_internal_ref

    # 2. Freshness / current external information needed
    freshness_patterns = [
        r"\b(?:latest|current|recent|today(?:'s)?|news|breaking|trends?|updates?|live|as\s+of\s+today|as\s+of\s+now|this\s+week|this\s+month|right\s+now)\b",
    ]
    for pat in freshness_patterns:
        if re.search(pat, q_lower):
            return True, "freshness", True

    return False, None, True


# Regex to match leading web search trigger instructions
WEB_TRIGGER_PREFIX_PATTERN = re.compile(
    r"^(?:(?:can|could)\s+you\s+(?:please\s+)?|please\s+)?(?:search\s+(?:the\s+)?(?:web|internet)(?:\s+for)?|"
    r"web\s+search(?:\s+for)?|"
    r"google(?:\s+this)?(?:\s+for)?|"
    r"look\s+up\s+online(?:\s+for)?|"
    r"search\s+online(?:\s+for)?|"
    r"browse\s+the\s+web(?:\s+for)?|"
    r"find\s+on\s+(?:the\s+)?(?:web|internet)(?:\s+for)?|"
    r"check\s+online(?:\s+for)?|"
    r"find\s+online(?:\s+for)?)\s*[:.\-–—]*\s*",
    re.IGNORECASE,
)

CONVERSATIONAL_PREFIX_PATTERN = re.compile(
    r"^(?:(?:can|could)\s+you\s+(?:please\s+)?(?:tell\s+me|show\s+me|find|get|check|explain)|"
    r"please\s+(?:tell\s+me|show\s+me|find|get|check|explain)|"
    r"i\s+(?:want|need|would\s+like)\s+to\s+(?:know|find|see)|"
    r"tell\s+me\s+(?:about)?)\s*[:.\-–—]*\s*",
    re.IGNORECASE,
)


def strip_search_triggers(query: str) -> str:
    """Strips conversational prefixes and explicit search trigger phrases from the query."""
    if not query:
        return ""
    q = query.strip()
    prev = None
    while q != prev:
        prev = q
        q = CONVERSATIONAL_PREFIX_PATTERN.sub("", q).strip()
        q = WEB_TRIGGER_PREFIX_PATTERN.sub("", q).strip()
    return q


def format_web_search_query(query: str, query_plan: QueryPlan | None = None) -> tuple[str, str]:
    """
    Deterministically formats and cleans a query for external search engines while preserving
    intent, entities, locations, numbers, and technical terms.

    Returns:
        (formatted_query: str, source: str)
        where source is 'semantic_query', 'prefix_stripped', or 'raw_passthrough'.
    """
    if not query or not query.strip():
        return "", "empty"

    raw_query = query.strip()

    # 1. If QueryPlan provides an enriched, substantive semantic_query distinct from raw
    if query_plan and query_plan.semantic_query:
        sq = strip_search_triggers(query_plan.semantic_query)
        if sq and sq.lower() != raw_query.lower() and len(sq) >= 3:
            return sq, "semantic_query"

    # 2. Strip search trigger prefixes from raw query
    stripped = strip_search_triggers(raw_query)
    if stripped and stripped.lower() != raw_query.lower() and len(stripped) >= 3:
        # Also strip leading conversational pleasantries if safe
        conv_stripped = CONVERSATIONAL_PREFIX_PATTERN.sub("", stripped).strip()
        final_query = conv_stripped if len(conv_stripped) >= 3 else stripped
        return final_query, "prefix_stripped"

    # 3. Conversational query without explicit trigger prefix
    conv_stripped = CONVERSATIONAL_PREFIX_PATTERN.sub("", raw_query).strip()
    if conv_stripped and conv_stripped.lower() != raw_query.lower() and len(conv_stripped) >= 3:
        return conv_stripped, "prefix_stripped"

    # 4. Direct / short queries pass through safely
    return raw_query, "raw_passthrough"


def extract_deterministic_intent(query: str) -> tuple[InformationNeed, PresentationNeed, EvidenceNeed, bool, str]:
    """
    Extracts high-level information need, presentation need, evidence need,
    conversation_required flag, and reason deterministically from generic query patterns.
    """
    normalized = " ".join(query.strip().split())
    if not normalized:
        return InformationNeed.GENERAL, PresentationNeed.NONE, EvidenceNeed.PROSE, False, "empty_query"

    if is_greeting(normalized):
        return InformationNeed.CONVERSATION, PresentationNeed.NONE, EvidenceNeed.CONVERSATION, True, "greeting_detected"

    if is_history_query(normalized):
        return InformationNeed.CONVERSATION, PresentationNeed.NONE, EvidenceNeed.CONVERSATION, True, "conversation_history_referenced"

    q_lower = normalized.lower()

    # Presentation Need detection (independent from information need)
    pres_need = PresentationNeed.NONE
    if re.search(r"\b(?:bar|line|pie|scatter|donut|area)\s+chart\b|\bchart\b|\bgraph\b|\bplot\b|\bvisualize\b", q_lower):
        pres_need = PresentationNeed.CHART
    elif re.search(r"\btable\b|\bgrid\b|\bmatrix\b|\btabular\b", q_lower):
        pres_need = PresentationNeed.TABLE
    elif re.search(r"\btimeline\b|\bchronolog(?:y|ical)\b", q_lower):
        pres_need = PresentationNeed.TIMELINE
    elif re.search(r"\bnetwork\b|\bgraph\s+relationship\b|\bconnections\b", q_lower):
        pres_need = PresentationNeed.NETWORK

    # Information Need & Evidence Need detection
    if re.search(r"\b(what\s+is\s+this\s+document\s+about|overview|summary|summarize|document\s+summary|high-level\s+summary)\b", q_lower):
        return InformationNeed.OVERVIEW, pres_need, EvidenceNeed.HIERARCHICAL, False, "document_overview_requested"

    if re.search(r"^\s*list\b|\b(list\s+the|enumerate|bullet\s+points|items)\b", q_lower):
        return InformationNeed.LIST, pres_need, EvidenceNeed.MIXED, False, "list_enumeration_requested"

    if re.search(r"\b(compare|comparison|difference\s+between|differences\s+between|\bvs\b|\bversus\b)\b", q_lower):
        return InformationNeed.COMPARISON, pres_need, EvidenceNeed.MIXED, False, "comparison_analysis_requested"

    # REQUIREMENTS: explicit terminology + natural-language patterns
    if re.search(
        r"\b(requirement|requirements|specifications|mvp"
        r"|needs?\s+to\s+(?:be\s+)?(?:implemented|completed|built|done|changed|change|modified|modify|updated|update)"
        r"|need\s+to\s+implement"
        r"|(?:things?|what|work)\s+(?:that\s+)?(?:i\s+)?(?:need|must|have)\s+to\s+(?:implement|build|complete|do|finish)"
        r"|(?:what|which)\s+(?:things?\s+)?(?:do\s+i\s+)?need\s+to\s+implement"
        r"|implement(?:ation)?\s+requirement"
        r"|what\s+(?:needs|has)\s+to\s+(?:be\s+)?(?:implemented|built|completed|done|changed|change|fixed|fix)"
        r"|what\s+(?:is|work\s+is|information\s+is)\s+(?:still\s+)?(?:outstanding|required|remaining|needed|missing)"
        r"|(?:have\s+to|must|need\s+to)\s+(?:build|change|implement|complete)"
        r"|(?:ready|complete|finished|done)\s+(?:before|for)\b"
        r"|\bimplementation\s+(?:checklist|tasks?)\b"
        r"|covered\s+everything\s+that\s+needs"
        r")\b",
        q_lower,
    ):
        return InformationNeed.REQUIREMENTS, pres_need, EvidenceNeed.HIERARCHICAL, False, "requirements_evidence_requested"

    if re.search(r"\b(total|sum|average|by\s+value|breakdown|distribution|pipeline\s+by\s+value|statistics|aggregation|count\s+of)\b", q_lower) or pres_need in (PresentationNeed.CHART, PresentationNeed.TABLE):
        return InformationNeed.AGGREGATION, pres_need, EvidenceNeed.STRUCTURED, False, "numerical_aggregation_or_chart_requested"

    if re.search(r"\b(explain|how\s+does|why\s+is|how\s+to|what\s+causes|describe\s+how)\b", q_lower):
        return InformationNeed.EXPLANATION, pres_need, EvidenceNeed.PROSE, False, "explanation_requested"

    if re.search(r"\b(who\s+is|who\s+was|what\s+is|where\s+is|when\s+did|which\s+is|deadline)\b", q_lower):
        return InformationNeed.FACTUAL, pres_need, EvidenceNeed.PROSE, False, "factual_lookup_requested"

    return InformationNeed.GENERAL, pres_need, EvidenceNeed.PROSE, False, "general_query"


# ============================================================
# INTENT-BASED SEMANTIC ENRICHMENT
# ============================================================

# Bounded, deterministic synonym terms per InformationNeed category.
# These are generic, domain-agnostic terms that improve retrieval
# recall when the LLM-generated semantic_query strips intent signal.
_INTENT_SEMANTIC_TERMS: dict[InformationNeed, str] = {
    InformationNeed.REQUIREMENTS: "requirements implementation specifications tasks",
    InformationNeed.OVERVIEW: "overview summary introduction",
    InformationNeed.LIST: "list items enumeration",
    InformationNeed.COMPARISON: "comparison differences alternatives",
    InformationNeed.AGGREGATION: "total summary metrics statistics",
}


def enrich_semantic_query(
    semantic_query: str | None,
    information_need: InformationNeed,
) -> str | None:
    """
    Appends bounded, deterministic intent-derived synonym terms to the
    semantic query so that retrieval (vector + keyword + hierarchical)
    does not lose the intent signal when the LLM strips it.

    Only enriches when:
    1. A semantic_query exists.
    2. The intent has registered synonym terms.
    3. The semantic_query does not already contain a majority of those terms.

    The appended terms are kept short and bounded to avoid diluting the query.
    """
    if not semantic_query or not semantic_query.strip():
        return semantic_query

    enrichment = _INTENT_SEMANTIC_TERMS.get(information_need)
    if not enrichment:
        return semantic_query

    sq_lower = semantic_query.lower()
    enrichment_tokens = enrichment.split()
    # Only enrich if fewer than half the intent terms are already present
    present_count = sum(1 for t in enrichment_tokens if t in sq_lower)
    if present_count >= len(enrichment_tokens) / 2:
        return semantic_query

    # Append only missing terms
    missing = [t for t in enrichment_tokens if t not in sq_lower]
    if not missing:
        return semantic_query

    return f"{semantic_query.strip()} {' '.join(missing)}"


# Words that reliably signal the start of a new clause when they appear
# immediately after a conjunction ("and", "&"). Proper-noun fragments
# (e.g. "Gamble" in "Procter and Gamble") are never in this set.
_CLAUSE_STARTERS = frozenset({
    "what", "who", "how", "why", "where", "when", "which",
    "is", "are", "was", "were", "does", "do", "did",
    "has", "have", "had",
    "its", "their", "it", "the",
    "can", "could", "will", "would", "should", "may", "might",
})


def _truncate_at_compound_boundaries(text: str) -> str:
    """
    Stops entity capture at compound/conjunction boundaries.

    Unambiguous delimiters (" as well as", " also ", " plus ",
    " along with ", " in addition to ", ";") always trigger truncation.

    Ambiguous delimiters (" and ", " & ") trigger truncation ONLY when
    the first word following the delimiter is a clause-starter (question
    word or auxiliary verb). This preserves legitimate multi-word entity
    names such as "Procter and Gamble" or "Marks and Spencer".
    """
    if not text:
        return text

    text_lower = text.lower()
    first_idx = len(text)

    # --- Unambiguous delimiters: always truncate ---
    for delim in (" as well as ", " also ", " plus ", " along with ", " in addition to "):
        idx = text_lower.find(delim)
        if idx != -1 and idx < first_idx:
            first_idx = idx

    # Semicolons always truncate
    sc_idx = text.find(";")
    if sc_idx != -1 and sc_idx < first_idx:
        first_idx = sc_idx

    # --- Ambiguous delimiters: truncate only if followed by a clause-starter ---
    for delim in (" and ", " & "):
        start = 0
        while True:
            idx = text_lower.find(delim, start)
            if idx == -1:
                break
            after_fragment = text_lower[idx + len(delim):]
            first_word = after_fragment.split()[0].strip("?.,!\"' ") if after_fragment.split() else ""
            if first_word in _CLAUSE_STARTERS and idx < first_idx:
                first_idx = idx
                break
            start = idx + len(delim)

    if first_idx < len(text):
        return text[:first_idx].strip("?.,!\"' ")
    return text.strip("?.,!\"' ")


def extract_deterministic_relationship(query: str) -> RelationshipQuery | None:
    """
    Deterministically extracts relationship query intent from standard query patterns.
    """
    q = query.strip()
    q_lower = q.lower().rstrip("?.,! ")

    # "who works at <org>" or "who is at <org>"
    m = re.match(r"^who\s+(?:works?\s+at|is\s+at|is\s+working\s+at)\s+(.+)$", q_lower, re.IGNORECASE)
    if m:
        target = m.group(1).strip("?.,!\"' ")
        target = _truncate_at_compound_boundaries(target)
        if target:
            return RelationshipQuery(predicate="works_at", object_entity=target, direction="in", hop_limit=1)

    # "who founded <org>" or "who started <org>"
    m = re.match(r"^who\s+(?:founded|started|created|established)\s+(.+)$", q_lower, re.IGNORECASE)
    if m:
        target = m.group(1).strip("?.,!\"' ")
        target = _truncate_at_compound_boundaries(target)
        if target:
            return RelationshipQuery(predicate="founded", object_entity=target, direction="in", hop_limit=1)

    # "what does <person/org> work on"
    m = re.match(r"^what\s+does\s+(.+?)\s+work\s+on$", q_lower, re.IGNORECASE)
    if m:
        target = m.group(1).strip("?.,!\"' ")
        target = _truncate_at_compound_boundaries(target)
        if target:
            return RelationshipQuery(subject_entity=target, predicate="works_on", direction="out", hop_limit=1)

    # "what does <org> develop" / "what products does <org> build/develop/make"
    m = re.match(r"^(?:what|which\s+products?)\s+does\s+(.+?)\s+(?:develop|build|make|produce)$", q_lower, re.IGNORECASE)
    if m:
        target = m.group(1).strip("?.,!\"' ")
        target = _truncate_at_compound_boundaries(target)
        if target:
            return RelationshipQuery(subject_entity=target, predicate="develops", direction="out", hop_limit=1)

    # "which companies did <person> invest in" / "who did <person> invest in"
    m = re.match(r"^(?:which\s+companies\s+did|who\s+did)\s+(.+?)\s+invest\s+in$", q_lower, re.IGNORECASE)
    if m:
        target = m.group(1).strip("?.,!\"' ")
        target = _truncate_at_compound_boundaries(target)
        if target:
            return RelationshipQuery(subject_entity=target, predicate="invested_in", direction="out", hop_limit=1)

    # "who leads <org>" / "who manages <org>"
    m = re.match(r"^who\s+(?:leads|manages|heads)\s+(.+)$", q_lower, re.IGNORECASE)
    if m:
        target = m.group(1).strip("?.,!\"' ")
        target = _truncate_at_compound_boundaries(target)
        if target:
            return RelationshipQuery(predicate="leads", object_entity=target, direction="in", hop_limit=1)

    return None


def extract_entity_search_term(query_plan: QueryPlan) -> str | None:
    """
    Extract an entity name lookup term from the query plan if applicable.
    Skips generic procedural, conceptual, or long questions.
    """
    if query_plan.filters.company_name_contains:
        return query_plan.filters.company_name_contains.strip()
    if query_plan.filters.company_id:
        return query_plan.filters.company_id.strip()

    if not query_plan.semantic_query:
        return None

    sq = query_plan.semantic_query.strip()
    sq = _truncate_at_compound_boundaries(sq)
    
    generic_concepts = {"rag", "jwt", "oauth", "rest", "api", "json", "sql", "llm", "ai", "ml", "http", "https"}

    # Support entity questions like "Who is <person>?" and "What is <entity>?"
    sq_lower = sq.lower()
    if sq_lower.startswith("who is "):
        cand = sq[7:].strip("?.,!\"' ")
        if cand and cand.lower() not in generic_concepts and len(cand.split()) <= 3:
            return cand
    if sq_lower.startswith("who was "):
        cand = sq[8:].strip("?.,!\"' ")
        if cand and cand.lower() not in generic_concepts and len(cand.split()) <= 3:
            return cand
    if sq_lower.startswith("what is "):
        cand = sq[8:].strip("?.,!\"' ")
        if cand and cand.lower() not in generic_concepts and len(cand.split()) <= 3:
            return cand

    # "Where does <person> work" / "Where does <entity> work" — subject-entity extraction
    _where_work_m = re.match(r"^where\s+does\s+(.+?)\s+work\b", sq_lower)
    if _where_work_m:
        cand = sq[_where_work_m.start(1):_where_work_m.end(1)].strip("?.,!\"' ")
        if cand and cand.lower() not in generic_concepts and len(cand.split()) <= 3:
            return cand

    tokens = [t.strip("?.,!\"'").lower() for t in sq.split() if t.strip("?.,!\"'")]
    if not tokens:
        return None

    generic_question_starters = {"how", "why", "what", "where", "when", "who", "explain", "describe", "difference", "compare"}
    if any(token in generic_question_starters for token in tokens):
        return None

    if sq.lower() in generic_concepts or any(token in generic_concepts for token in tokens) or len(tokens) > 3:
        return None

    return sq


# ============================================================
# QUERY PARSER PROMPT
# ============================================================

QUERY_PARSER_SYSTEM_PROMPT = """
You are a retrieval query parser.

Convert the user's natural-language query into a retrieval plan.
DO NOT answer the user.

Return ONLY valid JSON with exactly this structure:

{
  "semantic_query": string or null,
  "information_need": "OVERVIEW" | "FACTUAL" | "REQUIREMENTS" | "LIST" | "COMPARISON" | "AGGREGATION" | "EXPLANATION" | "CONVERSATION" | "GENERAL",
  "presentation_need": "TEXT" | "TABLE" | "CHART" | "TIMELINE" | "NETWORK" | "NONE",
  "evidence_need": "PROSE" | "HIERARCHICAL" | "STRUCTURED" | "MIXED" | "CONVERSATION",
  "filters": {
    "company_id": string or null,
    "company_name_contains": string or null,
    "funding_status": "Funded" or "Non-Funded" or null,
    "city": string or null,
    "country": string or null
  },
  "relationship_query": {
    "subject_entity": string or null,
    "predicate": string or null,
    "object_entity": string or null,
    "direction": "out" or "in" or "both",
    "hop_limit": 1 or 2
  } or null
}

Rules:

1. Extract only values explicitly supported by the user's query.
   Never invent values.

2. information_need:
   Classify the primary information need:
   - "OVERVIEW": High-level document summary, general topic, what the document is about.
   - "REQUIREMENTS": Technical requirements, specifications, prerequisites, what needs to be implemented, MVP scope.
   - "LIST": Enumeration, list of items/features/rules.
   - "COMPARISON": Comparing two or more entities, approaches, or options.
   - "AGGREGATION": Numerical summaries, metrics, counts, totals, distributions, pipeline values.
   - "EXPLANATION": Conceptual or procedural explanations (how/why).
   - "FACTUAL": Point lookups, specific facts, dates, single entity queries.
   - "CONVERSATION": Greetings, chitchat, conversation history.
   - "GENERAL": General questions that do not fit the above.

3. presentation_need:
   Represent the output format desired by the user (independent from retrieval):
   - "CHART": User requests a chart, plot, bar graph, line graph, pie chart, or visualization.
   - "TABLE": User requests a table, grid, or tabular breakdown.
   - "TIMELINE": User requests a chronological sequence or timeline.
   - "NETWORK": User requests graph/relationship visual representation.
   - "TEXT": User requests prose or text output explicitly.
   - "NONE": Default when no presentation/visualization format is requested.

4. evidence_need:
   Represent the likely evidence structure:
   - "HIERARCHICAL": Document outline, section headers, overview abstracts, requirements structure.
   - "STRUCTURED": Tabular data, key-value attributes, metrics, numbers.
   - "PROSE": Paragraphs, explanations, articles.
   - "MIXED": Combination of structured/table and prose/hierarchy.
   - "CONVERSATION": Conversation turns or thread history.

5. semantic_query:
   Focus semantic_query purely on the underlying data, facts, entities, or concepts being requested.
   ALWAYS strip presentation, formatting, visualization, and chart instructions
   (e.g., 'visualize as a bar chart', 'plot line chart', 'render table', 'show as a pie chart', 'as a bar chart', 'draw a chart', 'in a graph')
   from semantic_query so retrieval targets the source document's factual content.

6. company_id & company_name_contains:
   Extract only when the user explicitly refers to a company identifier or company name search.
   Technical terms (JWT, FastAPI, SQLAlchemy, REST, etc.) remain semantic queries.

7. relationship_query:
   Extract when the user asks about relationships/connections between entities.

8. NEVER answer the user's question. NEVER generate SQL. Return JSON only.
""".strip()


# ============================================================
# QUERY PARSER
# ============================================================

class QueryParser:

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or get_llm()

    @track(name="parse_query", phase="retrieval")
    @traceable(run_type="llm")
    async def parse(self, query: str) -> QueryPlan:
        """
        Convert a user query into a structured retrieval plan.
        Greetings and conversation history are handled deterministically.
        """
        normalized_query = " ".join(query.strip().split())

        # Fast deterministic intent extraction
        det_info, det_pres, det_ev, det_conv, det_reason = extract_deterministic_intent(normalized_query)
        cleaned_semantic = strip_presentation_instructions(normalized_query)

        # --------------------------------------------------------
        # Do not spend an LLM call on greetings or pure history.
        # --------------------------------------------------------
        if is_greeting(normalized_query):
            logger.info(
                "query_parser_conversational_query",
                extra={
                    "event": "query_parser_conversational_query",
                    "query": normalized_query,
                },
            )
            return QueryPlan(
                semantic_query=None,
                filters=CompanyFilters(),
                information_need=InformationNeed.CONVERSATION,
                presentation_need=PresentationNeed.NONE,
                evidence_need=EvidenceNeed.CONVERSATION,
                conversation_required=True,
                reason="greeting",
            )

        if is_history_query(normalized_query):
            return QueryPlan(
                semantic_query=cleaned_semantic,
                filters=CompanyFilters(),
                information_need=InformationNeed.CONVERSATION,
                presentation_need=det_pres,
                evidence_need=EvidenceNeed.CONVERSATION,
                conversation_required=True,
                reason="conversation_history",
            )

        messages = [
            {
                "role": "system",
                "content": QUERY_PARSER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": normalized_query,
            },
        ]

        try:
            raw_response, _token_usage = await self.llm.generate(messages, stage="retrieval")
            return self._parse_response(
                raw_response=raw_response,
                original_query=normalized_query,
            )
        except Exception as exc:
            logger.warning(
                "query_parser_llm_failed_fallback_to_deterministic",
                extra={
                    "event": "query_parser_llm_failed",
                    "error": str(exc),
                    "query": normalized_query,
                },
            )
            plan = QueryPlan(
                semantic_query=enrich_semantic_query(cleaned_semantic, det_info),
                filters=CompanyFilters(),
                relationship_query=extract_deterministic_relationship(normalized_query),
                information_need=det_info,
                presentation_need=det_pres,
                evidence_need=det_ev,
                conversation_required=det_conv,
                reason=det_reason,
            )
            plan.entity_search_term = extract_entity_search_term(plan)
            return plan

    # ============================================================
    # RESPONSE PARSING
    # ============================================================

    def _parse_response(
        self,
        raw_response: str,
        original_query: str,
    ) -> QueryPlan:

        cleaned = raw_response.strip()

        # --------------------------------------------------------
        # Remove markdown JSON code fences if provider adds them.
        # --------------------------------------------------------
        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            re.DOTALL | re.IGNORECASE,
        )

        if match:
            cleaned = match.group(1).strip()
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

        det_info, det_pres, det_ev, det_conv, det_reason = extract_deterministic_intent(original_query)
        cleaned_semantic = strip_presentation_instructions(original_query)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(
                "query_parser_invalid_json",
                extra={
                    "event": "query_parser_invalid_json",
                    "raw_response": raw_response,
                    "original_query": original_query,
                },
            )
            web_needed, web_reason, int_needed = detect_web_search_intent(original_query)
            plan = QueryPlan(
                semantic_query=cleaned_semantic,
                filters=CompanyFilters(),
                relationship_query=extract_deterministic_relationship(original_query),
                information_need=det_info,
                presentation_need=det_pres,
                evidence_need=det_ev,
                conversation_required=det_conv,
                reason=det_reason,
                web_search_needed=web_needed,
                web_search_reason=web_reason,
                internal_retrieval_needed=int_needed,
            )
            plan.entity_search_term = extract_entity_search_term(plan)
            return plan

        try:
            plan = QueryPlan.model_validate(payload)
            if plan.semantic_query:
                plan.semantic_query = strip_presentation_instructions(plan.semantic_query)
            if plan.relationship_query is None:
                plan.relationship_query = extract_deterministic_relationship(original_query)
            if plan.entity_search_term is None:
                plan.entity_search_term = extract_entity_search_term(plan)
            # If LLM defaulted intent to GENERAL / NONE but query has clear deterministic pattern, enrich it
            if plan.information_need == InformationNeed.GENERAL and det_info != InformationNeed.GENERAL:
                plan.information_need = det_info
            if plan.presentation_need == PresentationNeed.NONE and det_pres != PresentationNeed.NONE:
                plan.presentation_need = det_pres
            if plan.evidence_need == EvidenceNeed.PROSE and det_ev != EvidenceNeed.PROSE:
                plan.evidence_need = det_ev

            plan.conversation_required = plan.information_need == InformationNeed.CONVERSATION or det_conv
            plan.reason = det_reason
            plan.output_preferences = parse_output_preferences(original_query)

            # Web search intent
            web_needed, web_reason, int_needed = detect_web_search_intent(original_query)
            plan.web_search_needed = web_needed
            plan.web_search_reason = web_reason
            plan.internal_retrieval_needed = int_needed

            # Enrich semantic_query with intent-derived terms
            plan.semantic_query = enrich_semantic_query(
                plan.semantic_query, plan.information_need,
            )

            return plan

        except Exception:
            logger.error(
                "query_parser_invalid_schema",
                extra={
                    "event": "query_parser_invalid_schema",
                    "payload": payload,
                    "original_query": original_query,
                },
            )

            web_needed, web_reason, int_needed = detect_web_search_intent(original_query)
            plan = QueryPlan(
                semantic_query=cleaned_semantic,
                filters=CompanyFilters(),
                relationship_query=extract_deterministic_relationship(original_query),
                information_need=det_info,
                presentation_need=det_pres,
                evidence_need=det_ev,
                conversation_required=det_conv,
                reason=det_reason,
                output_preferences=parse_output_preferences(original_query),
                web_search_needed=web_needed,
                web_search_reason=web_reason,
                internal_retrieval_needed=int_needed,
            )
            plan.entity_search_term = extract_entity_search_term(plan)
            return plan