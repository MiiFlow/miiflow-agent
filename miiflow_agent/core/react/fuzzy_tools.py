"""Fuzzy matching of hallucinated tool names to registered ones.

Models occasionally emit near-miss tool names ("Add" for "Addition", case
variations, one-character typos). Auto-correcting the obvious cases saves a
round trip; anything ambiguous returns None and the model gets the standard
"tool not found, available: [...]" error instead.

Pure string logic, extracted from the orchestrator.
"""

from typing import Iterable, Optional


def find_similar_tool(
    requested_name: str, available_tools: Iterable[str]
) -> Optional[str]:
    """Best fuzzy match for ``requested_name`` among ``available_tools``.

    Strategy 1: case-insensitive substring containment, preferring the longer
    name ("Addition" over "Add"). Strategy 2: small edit distance (≤2).
    Returns None when nothing matches confidently.
    """
    if not requested_name:
        return None

    tools = list(available_tools)
    requested_lower = requested_name.lower()

    for tool_name in tools:
        tool_lower = tool_name.lower()
        if requested_lower in tool_lower or tool_lower in requested_lower:
            if len(tool_name) >= len(requested_name):
                return tool_name

    for tool_name in tools:
        if is_similar_enough(requested_name, tool_name):
            return tool_name

    return None


def is_similar_enough(s1: str, s2: str, threshold: int = 2) -> bool:
    """Cheap edit-distance-style similarity: same-length-ish strings whose
    positionwise differences (case-insensitive) fit within ``threshold``."""
    if abs(len(s1) - len(s2)) > threshold:
        return False
    s1_lower, s2_lower = s1.lower(), s2.lower()
    if s1_lower == s2_lower:
        return True
    differences = sum(1 for a, b in zip(s1_lower, s2_lower) if a != b)
    differences += abs(len(s1_lower) - len(s2_lower))
    return differences <= threshold
