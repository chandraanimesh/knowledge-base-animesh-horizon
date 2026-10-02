from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Literal
ElementType = Literal[
        "text",
        "heading",
        "table",
        "figure",
        "record",
]

# fixed array structure ke slot = true, waise toh dictionar deta h n i.e default instance
@dataclass(slots=True)
class DocumentElement:
    """
    Normalised representaiion h source data ka..jisme sare 
    loaader ko DocumentElement produce kena chahie..
    """
    text: str
    source_path: str
    file_name: str
    file_type: str 
    element_type: ElementType
    page_number: int | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )