import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyDecision:
    blocked: bool
    reason: str | None = None
    response: str | None = None


SENSITIVE_PII_PATTERNS = [
    re.compile(r"\baadha?ar\b", re.IGNORECASE),
    re.compile(r"\bpan\b", re.IGNORECASE),
    re.compile(r"\bpassport\b", re.IGNORECASE),
    re.compile(r"\bbank account\b", re.IGNORECASE),
    re.compile(r"\bcredit card\b", re.IGNORECASE),
    re.compile(r"\bphone number\b", re.IGNORECASE),
    re.compile(r"\bemail address\b", re.IGNORECASE),
]


def check_sensitive_request(
    query: str,
    *,
    can_view_pii: bool = False,
) -> SafetyDecision:
    """
    Blocks sensitive PII disclosure unless authorization
    was verified by the backend.

    Important:
    can_view_pii must come from trusted authentication/authorization,
    never from the user's request body.
    """
    requests_sensitive_data = any(
        pattern.search(query)
        for pattern in SENSITIVE_PII_PATTERNS
    )

    if requests_sensitive_data and not can_view_pii:
        return SafetyDecision(
            blocked=True,
            reason="unauthorized_sensitive_pii_request",
            response=(
                "I can’t provide sensitive personal information based only "
                "on this request. Government, legal, or emergency claims must "
                "be verified through an authorized access workflow. Please use "
                "the approved identity-verification process or contact an "
                "authorized administrator."
            ),
        )

    return SafetyDecision(blocked=False)