import re
from typing import Optional, NamedTuple
from rag_app.memory.schemas import MemoryType, StabilityTier
from rag_app.memory.materializer import (
    extract_memory_topic_key,
    generate_memory_canonical_key,
    normalize_string,
)


class ParsedMemoryCommand(NamedTuple):
    is_command: bool
    command_name: str
    raw_payload: str
    is_empty: bool
    error_message: Optional[str]
    subject: Optional[str]
    predicate: Optional[str]
    content: str
    memory_type: MemoryType
    stability_tier: StabilityTier
    importance_signals: list[str]
    topic_key: str
    canonical_key: str
    is_explicit_update: bool


def parse_memory_command(text: str) -> ParsedMemoryCommand:
    """
    Deterministically parses /rem and /remember slash commands.
    Ensures empty or malformed commands do not create blank memories,
    derives stability tier from content, and extracts canonical identities.
    """
    clean = (text or "").strip()
    if not clean.startswith("/"):
        return ParsedMemoryCommand(
            is_command=False,
            command_name="",
            raw_payload="",
            is_empty=True,
            error_message="Not a slash command",
            subject=None,
            predicate=None,
            content="",
            memory_type=MemoryType.DYNAMIC,
            stability_tier=StabilityTier.EVOLVING,
            importance_signals=[],
            topic_key="",
            canonical_key="",
            is_explicit_update=False,
        )

    parts = clean.split(maxsplit=1)
    cmd_name = parts[0].lower()
    payload = parts[1].strip() if len(parts) > 1 else ""

    if cmd_name not in ("/rem", "/remember"):
        return ParsedMemoryCommand(
            is_command=False,
            command_name=cmd_name,
            raw_payload=payload,
            is_empty=not bool(payload),
            error_message=f"Unknown command {cmd_name}",
            subject=None,
            predicate=None,
            content="",
            memory_type=MemoryType.DYNAMIC,
            stability_tier=StabilityTier.EVOLVING,
            importance_signals=[],
            topic_key="",
            canonical_key="",
            is_explicit_update=False,
        )

    if not payload:
        return ParsedMemoryCommand(
            is_command=True,
            command_name=cmd_name,
            raw_payload="",
            is_empty=True,
            error_message="⚠️ Malformed command. Usage: `/rem <text to remember>`\n*Example:* `/rem I prefer responses in concise bullet points.`",
            subject=None,
            predicate=None,
            content="",
            memory_type=MemoryType.DYNAMIC,
            stability_tier=StabilityTier.EVOLVING,
            importance_signals=["explicit_rem"],
            topic_key="",
            canonical_key="",
            is_explicit_update=False,
        )

    p_norm = normalize_string(payload)

    # 1. Detect explicit update / correction / supersession signals
    update_patterns = [
        r"\b(?:now\s+prefer|changed\s+my|moved\s+to|no\s+longer|updated\s+my|switch(?:ed)?\s+to)\b",
        r"\b(?:i\s+am\s+now|instead\s+of|replaced\s+with)\b",
    ]
    is_explicit_update = any(re.search(pat, p_norm) for pat in update_patterns)

    # 2. Determine stability tier and memory type from content
    # Eternal: Durable persona, identity, permanent preferences, role
    eternal_patterns = [
        r"\b(?:my\s+name\s+is|call\s+me|refer\s+to\s+me\s+as)\b",
        r"\b(?:dark\s+mode|light\s+mode|format\s+monetary|in\s+usd|in\s+eur|in\s+inr)\b",
        r"\b(?:speak\s+in|respond\s+in|language\s+is|reply\s+in)\b",
        r"\b(?:always\s+format|concise\s+bullet\s+points?|bullet\s+points?)\b",
    ]

    # Stable: Architecture, technical conventions, durable team decisions
    stable_patterns = [
        r"\b(?:architecture|convention|standard|guideline|rule|framework|stack|postgres|fastapi|react)\b",
        r"\b(?:our\s+company|our\s+team|we\s+use|we\s+always|codebase\s+pattern)\b",
    ]

    # Experimental: Debugging, testing, temporary notes, short-lived hypotheses
    experimental_patterns = [
        r"\b(?:testing|debug(?:ging)?|temporary|workaround|hypothesis|for\s+now|quick\s+fix)\b",
    ]

    importance_signals = ["explicit_rem"]

    if any(re.search(pat, p_norm) for pat in eternal_patterns):
        stability_tier = StabilityTier.ETERNAL
        memory_type = MemoryType.STATIC
        importance_signals.append("core_preference")
    elif any(re.search(pat, p_norm) for pat in stable_patterns):
        stability_tier = StabilityTier.STABLE
        memory_type = MemoryType.SEMANTIC
        importance_signals.append("decision")
    elif any(re.search(pat, p_norm) for pat in experimental_patterns):
        stability_tier = StabilityTier.EXPERIMENTAL
        memory_type = MemoryType.DYNAMIC
        importance_signals.append("transient_note")
    else:
        # Default for /rem statements: evolving context
        stability_tier = StabilityTier.EVOLVING
        memory_type = MemoryType.DYNAMIC

    topic_key = extract_memory_topic_key(payload)
    canonical_key = generate_memory_canonical_key(
        memory_type.value,
        topic_key or payload,
    )

    # Extract rough subject/predicate
    subject = "user"
    predicate = topic_key

    return ParsedMemoryCommand(
        is_command=True,
        command_name=cmd_name,
        raw_payload=payload,
        is_empty=False,
        error_message=None,
        subject=subject,
        predicate=predicate,
        content=payload,
        memory_type=memory_type,
        stability_tier=stability_tier,
        importance_signals=importance_signals,
        topic_key=topic_key,
        canonical_key=canonical_key,
        is_explicit_update=is_explicit_update,
    )


def detect_memory_candidate_rules(text: str) -> tuple[bool, bool, dict]:
    """
    Deterministic rule-first classifier for conversational statements.
    Returns: (is_candidate, is_transient, candidate_hints)
    - is_candidate: True if matched deterministic memory patterns (preference, decision, task)
    - is_transient: True if matched greetings or casual chit-chat (produce no memory)
    - candidate_hints: dictionary with extracted topic, stability, importance if candidate
    """
    clean = (text or "").strip()
    if not clean or len(clean) < 3:
        return False, True, {}

    norm = normalize_string(clean)

    # 1. Check transient messages (greetings, chit-chat, simple queries)
    transient_patterns = [
        r"^(?:hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|good\s+night)\b",
        r"^(?:how\s+are\s+you|what(?:'s|\s+is)\s+up|thanks?|thank\s+you|ok|okay|bye|see\s+you)\b",
        r"^(?:what\s+is|who\s+is|where\s+is|explain|describe|summarize|search\s+for|tell\s+me\s+about)\b",
    ]
    if any(re.search(pat, norm) for pat in transient_patterns):
        return False, True, {}

    # 2. Check procedural instructions & guidelines (PROCEDURAL / STABLE)
    # Evaluated before general preferences so "I prefer step-by-step explanations" -> PROCEDURAL
    procedural_patterns = [
        r"\b(?:always\s+format|always\s+respond|always\s+generate|follow\s+pep|coding\s+standard|workflow\s+rule|naming\s+convention|response\s+format|in\s+json\s+format|in\s+markdown\s+table|step-by-step(?:\s+explanations?)?|bullet\s+points?)\b",
    ]
    if any(re.search(pat, norm) for pat in procedural_patterns):
        return True, False, {
            "memory_type": MemoryType.PROCEDURAL,
            "stability_tier": StabilityTier.STABLE,
            "importance_signals": ["instruction", "convention"],
            "topic_key": extract_memory_topic_key(clean),
        }

    # 3. Check clear preferences (STATIC / ETERNAL)
    preference_patterns = [
        (r"\b(?:i\s+prefer|i\s+like|i\s+always\s+want|my\s+preference\s+is)\b", "preference", StabilityTier.ETERNAL),
        (r"\b(?:my\s+role\s+is|i\s+am\s+a|i'm\s+a|i\s+work\s+as)\b", "role", StabilityTier.ETERNAL),
        (r"\b(?:my\s+name\s+is|call\s+me|refer\s+to\s+me\s+as)\b", "name", StabilityTier.ETERNAL),
        (r"\b(?:i\s+live\s+in|based\s+in|located\s+in)\b", "location", StabilityTier.ETERNAL),
    ]
    for pat, pref_type, tier in preference_patterns:
        if re.search(pat, norm):
            return True, False, {
                "memory_type": MemoryType.STATIC,
                "stability_tier": tier,
                "importance_signals": ["preference", pref_type],
                "topic_key": extract_memory_topic_key(clean),
            }

    # 4. Check commitments / active task context (DYNAMIC / EVOLVING)
    task_patterns = [
        r"\b(?:i\s+am\s+currently\s+working\s+on|working\s+on\s+ticket|reviewing\s+the)\b",
        r"\b(?:i\s+will\s+deliver|committed\s+to|planning\s+to|scheduled\s+to)\b",
    ]
    if any(re.search(pat, norm) for pat in task_patterns):
        return True, False, {
            "memory_type": MemoryType.DYNAMIC,
            "stability_tier": StabilityTier.EVOLVING,
            "importance_signals": ["commitment", "current_task"],
            "topic_key": extract_memory_topic_key(clean),
        }

    # 5. Check durable decisions / project conventions (SEMANTIC / STABLE)
    decision_patterns = [
        r"\b(?:(?:our\s+)?(?:backend|frontend|database|system|stack|app|service|server)\s+(?:uses|is\s+built\s+with|is\s+based\s+on)|we\s+decided\s+to|our\s+decision\s+is|agreed\s+to\s+use|we\s+adopt|architecture\s+is|stack\s+is)\b",
    ]
    if any(re.search(pat, norm) for pat in decision_patterns):
        return True, False, {
            "memory_type": MemoryType.SEMANTIC,
            "stability_tier": StabilityTier.STABLE,
            "importance_signals": ["decision", "project_architecture"],
            "topic_key": extract_memory_topic_key(clean),
        }

    # 6. Check episodic events & milestones (EPISODIC / EVOLVING)
    episodic_patterns = [
        r"\b(?:(?:yesterday|last\s+week|last\s+month|on\s+monday|on\s+friday)\s+(?:i|we)|(?:i|we)\s+attended\b|completed\s+(?:the\s+)?(?:migration|sprint|release|project|review|milestone)|deployed\s+version|incident\s+occurred|met\s+with\s+client|meeting\s+with\s+client)\b",
    ]
    if any(re.search(pat, norm) for pat in episodic_patterns):
        return True, False, {
            "memory_type": MemoryType.EPISODIC,
            "stability_tier": StabilityTier.EVOLVING,
            "importance_signals": ["event", "milestone"],
            "topic_key": extract_memory_topic_key(clean),
        }

    return False, False, {}
