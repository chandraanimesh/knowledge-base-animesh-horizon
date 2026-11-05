"""Stream Cleaner & Tool Call Isolator

Provides structured isolation of tool call artifacts from streaming tokens.
Prevents internal <tool_call>...</tool_call> or Action: ... syntax leakages
from appearing in normal user-facing prose tokens.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|```json\s*({\s*\"tool\".*?})\s*```|Action:\s*([^\n]+)\nAction Input:\s*([^\n]+)",
    re.DOTALL | re.IGNORECASE,
)


class StreamTokenResult(NamedTuple):
    clean_text: str
    tool_events: list[dict]
    remaining_buffer: str


class StreamToolCleaner:
    def __init__(self):
        self.buffer = ""

    def process_chunk(self, chunk: str) -> tuple[str, list[dict]]:
        """
        Appends chunk to buffer, checks for complete or partial tool call signatures,
        emits clean user prose tokens and extracted structured tool events.
        """
        self.buffer += chunk
        tool_events = []

        # Check for completed tool call matches in the buffer
        match = TOOL_CALL_REGEX.search(self.buffer)
        while match:
            start, end = match.span()
            # Prose text before the tool call is safe to emit
            safe_prefix = self.buffer[:start]

            raw_matched = match.group(0)
            tool_name = "unknown"
            tool_input = {}

            if "<tool_call>" in raw_matched.lower():
                inner = match.group(1) or ""
                try:
                    parsed = json.loads(inner.strip())
                    tool_name = parsed.get("name") or parsed.get("tool") or "tool"
                    tool_input = parsed.get("arguments") or parsed.get("parameters") or parsed.get("input") or {}
                except Exception:
                    tool_name = "custom_tool"
                    tool_input = {"raw": inner.strip()}
            elif "action:" in raw_matched.lower():
                tool_name = (match.group(3) or "").strip()
                raw_in = (match.group(4) or "").strip()
                try:
                    tool_input = json.loads(raw_in)
                except Exception:
                    tool_input = {"input": raw_in}
            else:
                inner = match.group(2) or ""
                try:
                    parsed = json.loads(inner.strip())
                    tool_name = parsed.get("tool") or "tool"
                    tool_input = parsed.get("input") or {}
                except Exception:
                    tool_input = {"raw": inner.strip()}

            tool_events.append({
                "type": "tool_call",
                "tool": tool_name,
                "input": tool_input,
            })

            # Advance buffer past the tool call
            self.buffer = self.buffer[end:]
            match = TOOL_CALL_REGEX.search(self.buffer)
            return safe_prefix, tool_events

        # If buffer starts with a partial tag e.g. "<tool_", hold it back
        tag_markers = ["<tool_call", "```json", "Action:"]
        for marker in tag_markers:
            for i in range(1, len(marker)):
                partial = marker[:i]
                if self.buffer.endswith(partial):
                    emit_len = len(self.buffer) - len(partial)
                    to_emit = self.buffer[:emit_len]
                    self.buffer = self.buffer[emit_len:]
                    return to_emit, []

        to_emit = self.buffer
        self.buffer = ""
        return to_emit, []

    def flush(self) -> str:
        """Flushes any remaining text in the buffer at the end of the stream."""
        remaining = self.buffer
        self.buffer = ""
        # Scrub any leftover unmatched tag fragments
        cleaned = re.sub(r"</?tool_call>", "", remaining, flags=re.IGNORECASE)
        return cleaned

