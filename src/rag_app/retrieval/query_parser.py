import json
import logging
import re

from langsmith import traceable

from rag_app.providers.base.llm import LLMProvider
from rag_app.generation.llm_factory import get_llm
from rag_app.schemas.rag import CompanyFilters, QueryPlan, RelationshipQuery
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

    These queries should not consume:
    - query parser LLM tokens
    - embeddings
    - BM25 search
    - vector search
    - reranking
    """
    normalized = " ".join(query.strip().split())

    if not normalized:
        return False

    return any(
        pattern.fullmatch(normalized)
        for pattern in GREETING_PATTERNS
    )

HISTORY_PATTERNS = [
    re.compile(r"\b(what|remind|summarize|tell|can you remind).*(earlier|previous|last|before|discussed|talked|asked|said)\b(?!.*\b\d{4}\b)", re.IGNORECASE),
    re.compile(r"\b(what did i ask|what was my previous|what did you tell|what i asked)\b", re.IGNORECASE)
]

def is_history_query(query: str) -> bool:
    """
    Deterministically detects conversation-history oriented queries.
    Used to prevent the system from aggressively dropping conversational
    queries before they can reach the LLM with the injected history.
    """
    normalized = " ".join(query.strip().split())
    if not normalized:
        return False
        
    # False positive catchers for common document-retrieval patterns
    if re.search(r"\b(before|after|in|during)\s+(19|20)\d{2}\b", normalized, re.IGNORECASE):
        return False
        
    for pattern in HISTORY_PATTERNS:
        if pattern.search(normalized):
            return True
            
    return False



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

2. company_id:
   Extract only when the user explicitly refers to a company identifier.

3. company_name_contains:
   Extract only when the user explicitly searches, filters, or refers to
   a company name.

   Examples:
   "companies named Nexa"
   -> company_name_contains = "Nexa"

   "find Nexa company"
   -> company_name_contains = "Nexa"

   "companies containing Nexa in their name"
   -> company_name_contains = "Nexa"

   Technical terms such as JWT, FastAPI, SQLAlchemy, programming
   languages, libraries, protocols, databases, authentication
   mechanisms, and engineering concepts remain semantic queries unless
   the user explicitly indicates a company-name search.

4. If something could be both a company name and a technical concept,
   prefer semantic_query unless company-search intent is explicit.

   "JWT"
   -> semantic_query = "JWT"

   "information about JWT"
   -> semantic_query = "information about JWT"

   "company named JWT"
   -> company_name_contains = "JWT"

5. funding_status may ONLY be:
   "Funded", "Non-Funded", or null.

6. Extract city or country as structured filters ONLY when the user
   is explicitly searching for, filtering, or referring to companies
   or company records.

   For general knowledge, weather, news, travel, or other non-company
   queries, keep the location inside semantic_query and leave city/country null.

7. relationship_query:
   Extract when the user asks about relationships/connections between entities (e.g. who works at Acme, who founded Beta, what products does Gamma develop, which companies did Alice invest in).
   Examples:
   "Who works at Acme?" -> relationship_query: {"subject_entity": null, "predicate": "works_at", "object_entity": "Acme", "direction": "in", "hop_limit": 1}
   "Who founded Acme?" -> relationship_query: {"subject_entity": null, "predicate": "founded", "object_entity": "Acme", "direction": "in", "hop_limit": 1}
   "Which companies did Alice invest in?" -> relationship_query: {"subject_entity": "Alice", "predicate": "invested_in", "object_entity": null, "direction": "out", "hop_limit": 1}
   "What does Alice work on?" -> relationship_query: {"subject_entity": "Alice", "predicate": "works_on", "object_entity": null, "direction": "out", "hop_limit": 1}
   If no relationship query is present, emit relationship_query: null.

8. semantic_query represents the user's actual information need.

9. Purely structured requests may use semantic_query = null.

10. Technical, engineering, programming, documentation, policy, and
    general-knowledge queries should preserve their subject in
    semantic_query, even when only one or two words long.

11. For semantic_query retrieval rewriting, you may use:
    - synonyms
    - relevant terminology
    - section concepts
    - important entities
    - numerical concepts
    - policy terminology

    Do NOT invent facts or change filters.

12. NEVER answer the user's question.

13. NEVER generate SQL.

14. Return JSON only.

15. When uncertain between semantic interpretation and company-name
    interpretation, prefer semantic interpretation unless explicit
    company-search intent exists.
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

        Greetings are handled deterministically and never sent to the LLM.
        """

        normalized_query = " ".join(query.strip().split())

        # --------------------------------------------------------
        # IMPORTANT:
        # Do not spend an LLM call on greetings.
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
                semantic_query=normalized_query,
                filters=CompanyFilters(),
                relationship_query=extract_deterministic_relationship(normalized_query),
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
            # ----------------------------------------------------
            # Fallback: extract first JSON object.
            # ----------------------------------------------------
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

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

            # ----------------------------------------------------
            # Safe fallback:
            # treat original query as semantic retrieval query.
            # No filters are invented.
            # ----------------------------------------------------
            plan = QueryPlan(
                semantic_query=original_query,
                filters=CompanyFilters(),
                relationship_query=extract_deterministic_relationship(original_query),
            )
            plan.entity_search_term = extract_entity_search_term(plan)
            return plan

        try:
            plan = QueryPlan.model_validate(payload)
            if plan.relationship_query is None:
                plan.relationship_query = extract_deterministic_relationship(original_query)
            if plan.entity_search_term is None:
                plan.entity_search_term = extract_entity_search_term(plan)
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

            plan = QueryPlan(
                semantic_query=original_query,
                filters=CompanyFilters(),
                relationship_query=extract_deterministic_relationship(original_query),
            )
            plan.entity_search_term = extract_entity_search_term(plan)
            return plan

    # ============================================================
    # CODE FENCE HELPER
    # ============================================================

    @staticmethod
    def _remove_code_fence(text: str) -> str:
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()