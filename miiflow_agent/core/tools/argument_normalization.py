"""Which tool arguments are plain-text labels, and their repair.

Models sometimes hand a tool a name or title with an extra layer of HTML
escaping ("Monitor &amp; Optimizer"), and the tool stores it that way. Every
entry point for model-written arguments (the tool executor, the legacy agent
path, a host's MCP middleware) applies :func:`normalize_label_arguments`, so the
policy of which keys count as labels lives here, once. The decoding itself is
:func:`miiflow_agent.visualization.text_normalization.normalize_label`.
"""

from typing import Any, Set

from ...visualization.text_normalization import normalize_label

#: Tool-argument keys whose value is a plain-text label the tool stores and
#: shows: an entity's name or title. Lookup and match keys ("path",
#: "confirm_name", "name_contains") are deliberately absent, since rewriting
#: them would stop them matching what is already stored.
LABEL_KEYS: Set[str] = {"name", "title", "display_name", "brand_name"}

#: Argument keys holding a list of things that are each named: a table's
#: column definitions.
LABEL_CONTAINERS: Set[str] = {"columns"}


def _normalize_labels(arguments: dict) -> dict:
    return {
        key: normalize_label(item)
        if key in LABEL_KEYS and isinstance(item, str)
        else item
        for key, item in arguments.items()
    }


def normalize_label_arguments(arguments: Any) -> Any:
    """Apply :func:`normalize_label` to the label fields of one tool call.

    Closed on purpose: the call's own :data:`LABEL_KEYS`, and those of each
    item in a :data:`LABEL_CONTAINERS` list. A name key deeper down is often
    user data (a table row's "name" cell, a run input), and rewriting it would
    change the data and break matching against rows already stored. Anything
    else, including JSON passed as a string, is returned exactly as given, and
    the caller's dict is never edited.
    """
    if not isinstance(arguments, dict):
        return arguments
    normalized = _normalize_labels(arguments)
    for key in LABEL_CONTAINERS & normalized.keys():
        items = normalized[key]
        if isinstance(items, list):
            normalized[key] = [
                _normalize_labels(item) if isinstance(item, dict) else item
                for item in items
            ]
    return normalized
