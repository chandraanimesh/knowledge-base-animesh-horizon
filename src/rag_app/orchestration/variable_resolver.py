"""
Deterministic Variable Resolution Engine for Workflow DAGs.
Evaluates template expressions (e.g. {{trigger.company_name}} or {{step_1.output.id}})
against task execution context.
Enforces strict path validation, fail-closed rejection of missing context paths,
type preservation for standalone variables, and zero arbitrary code execution (no eval/exec).
"""

from __future__ import annotations

import re
from typing import Any


class VariableResolutionError(ValueError):
    """Raised when a template variable path cannot be resolved from the execution context."""
    pass


VARIABLE_REGEX = re.compile(r"\{\{\s*([a-zA-Z0-9_\-\.]+)\s*\}\}")
EXACT_VARIABLE_REGEX = re.compile(r"^\s*\{\{\s*([a-zA-Z0-9_\-\.]+)\s*\}\}\s*$")


def get_nested_value(path: str, context: dict[str, Any]) -> Any:
    """
    Safely retrieves a dot-separated path from a nested dictionary or object context.
    Raises VariableResolutionError if any segment is missing.
    """
    parts = path.strip().split(".")
    curr = context

    # Convenience support: if path starts with 'steps.<task_key>' and 'steps' key is not
    # explicitly present in context, resolve directly against the task key in context.
    if parts and parts[0] == "steps" and isinstance(curr, dict) and "steps" not in curr:
        if len(parts) > 1 and parts[1] in curr:
            parts = parts[1:]

    for idx, part in enumerate(parts):
        if curr is None:
            raise VariableResolutionError(
                f"Cannot resolve '{path}': segment '{part}' is None (path exhausted at index {idx})."
            )

        if isinstance(curr, dict):
            if part in curr:
                curr = curr[part]
            else:
                available = list(curr.keys())
                raise VariableResolutionError(
                    f"Variable path '{path}' failed: key '{part}' not found in dictionary. Available keys: {available}"
                )
        elif hasattr(curr, part):
            curr = getattr(curr, part)
        else:
            raise VariableResolutionError(
                f"Variable path '{path}' failed: attribute '{part}' not found on object of type '{type(curr).__name__}'."
            )

    return curr


def resolve_variables(value: Any, context: dict[str, Any]) -> Any:
    """
    Recursively resolves variable templates within strings, dicts, and lists.
    Preserves native data types (int, float, bool, list, dict) when a string contains
    exclusively a single template expression (e.g. '{{step.output.count}}').
    """
    if isinstance(value, str):
        # 1. Exact single variable expression -> preserve original typed object
        exact_match = EXACT_VARIABLE_REGEX.match(value)
        if exact_match:
            path = exact_match.group(1)
            return get_nested_value(path, context)

        # 2. Embedded variable expressions in prose -> string substitution
        if "{{" in value and "}}" in value:
            def _replacer(m: re.Match) -> str:
                path = m.group(1)
                resolved = get_nested_value(path, context)
                return str(resolved)

            return VARIABLE_REGEX.sub(_replacer, value)

        return value

    elif isinstance(value, dict):
        return {k: resolve_variables(v, context) for k, v in value.items()}

    elif isinstance(value, list):
        return [resolve_variables(item, context) for item in value]

    return value

