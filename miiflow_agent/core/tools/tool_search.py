"""Provider-agnostic ToolSearch.

When a tool registry holds many tools, sending every schema to the LLM on every
turn wastes tokens (and on some providers degrades tool selection quality).
Instead, the registry can hide most tools behind a single ``tool_search``
meta-tool: the model calls ``tool_search`` with a query, gets back the top-N
matching tools, and those tools are then enabled for the remainder of the
current run.

This is implemented entirely client-side and does not depend on any
provider-specific feature flag (e.g. Anthropic's ``defer_loading``), so it
works across all LLM providers miiflow-agent supports.

Activation is gated by the registry (see ``ToolRegistry.should_use_tool_search``).
Per-run state (which tools the model has discovered so far) lives in a
``ContextVar`` so concurrent agent runs do not leak tool visibility into each
other.
"""

from __future__ import annotations

import contextlib
import logging
import re
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set

if TYPE_CHECKING:
    from .registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---- configuration constants -------------------------------------------------

#: Default tool count threshold above which ToolSearch activates.
DEFAULT_TOOL_SEARCH_THRESHOLD = 25

#: Default max number of tools returned per ``tool_search`` call.
DEFAULT_MAX_RESULTS = 5

#: Name of the built-in meta-tool exposed to the LLM.
TOOL_SEARCH_TOOL_NAME = "tool_search"


# ---- per-run session state ---------------------------------------------------

#: Contextvar holding the set of tool names the LLM has discovered (and thus
#: enabled) during the current agent run. ``None`` means "no session active",
#: which causes the registry to fall back to its legacy behaviour of exposing
#: all tools.
_enabled_tools: ContextVar[Optional[Set[str]]] = ContextVar(
    "miiflow_tool_search_enabled", default=None
)


def get_enabled_tool_names() -> Optional[Set[str]]:
    """Return the set of tool names enabled for the current run, or None."""
    return _enabled_tools.get()


def is_session_active() -> bool:
    """True if a ToolSearch session is currently active on this task."""
    return _enabled_tools.get() is not None


@contextlib.contextmanager
def tool_search_session(initial: Optional[Set[str]] = None) -> Iterator[Set[str]]:
    """Open a ToolSearch session for the current async task.

    The yielded set is the live set of enabled tool names; mutating it (or
    calling ``tool_search``) updates which tools the LLM sees on the next
    turn. The session is torn down on exit, restoring whatever state was
    visible to the caller.
    """
    enabled: Set[str] = set(initial) if initial else set()
    token = _enabled_tools.set(enabled)
    try:
        yield enabled
    finally:
        _enabled_tools.reset(token)


def mark_tools_enabled(names: List[str]) -> None:
    """Add tool names to the active session's enabled set, if any."""
    enabled = _enabled_tools.get()
    if enabled is None:
        return
    for name in names:
        enabled.add(name)


# ---- lexical scoring ---------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def score_tool(query_tokens: List[str], searchable_text: str) -> float:
    """Cheap lexical match score: token overlap weighted by rarity-of-match.

    Not BM25, but good enough as a default; the registry exposes a hook to
    swap in an embedding-based ranker later.
    """
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(searchable_text)
    if not text_tokens:
        return 0.0
    text_set = set(text_tokens)
    matches = sum(1 for t in query_tokens if t in text_set)
    if matches == 0:
        return 0.0
    # Normalise by query length so single-word queries don't dominate.
    coverage = matches / len(query_tokens)
    # Small bonus for substring match against the joined text (helps when the
    # query mentions a tool name fragment).
    joined = " ".join(text_tokens)
    substring_bonus = 0.25 if any(t in joined for t in query_tokens if len(t) >= 4) else 0.0
    return coverage + substring_bonus


def build_searchable_text(name: str, description: str, metadata: Dict[str, Any]) -> str:
    """Build the cached searchable string for a tool."""
    parts: List[str] = [name, description or ""]
    tags = metadata.get("tags") if metadata else None
    if tags:
        parts.extend(str(t) for t in tags)
    keywords = metadata.get("search_keywords") if metadata else None
    if keywords:
        parts.extend(str(k) for k in keywords)
    return " ".join(parts)


# ---- meta-tool factory -------------------------------------------------------

_TOOL_SEARCH_DESCRIPTION = (
    "Search the agent's tool catalog by keyword to discover additional tools that are "
    "not currently visible. Use this when none of the tools in your current toolset can "
    "accomplish the user's request. Pass a short natural-language query describing what "
    "you need (e.g. 'send email', 'read database', 'resize image'). Returns up to "
    "max_results tools with their names, descriptions, and parameter schemas; those "
    "tools become callable on subsequent turns."
)


def build_tool_search_tool(registry: "ToolRegistry"):
    """Build the ``tool_search`` FunctionTool bound to a specific registry.

    The returned tool, when called, scores the registry's tools against the
    query, marks the top matches as enabled in the active session, and returns
    their schemas to the model.
    """
    # Local import to avoid a circular import at module load time.
    from .function import FunctionTool
    from .schemas import ParameterSchema, ToolSchema
    from .types import ParameterType, ToolType

    async def _tool_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
        max_results = max(1, min(int(max_results or DEFAULT_MAX_RESULTS), 20))
        matches = registry.search(query, max_results=max_results)
        if not matches:
            return {
                "query": query,
                "results": [],
                "message": "No tools matched. Try different keywords.",
            }
        mark_tools_enabled([m["name"] for m in matches])
        return {
            "query": query,
            "results": matches,
            "message": (
                f"{len(matches)} tool(s) are now available to call directly on your next turn."
            ),
        }

    schema = ToolSchema(
        name=TOOL_SEARCH_TOOL_NAME,
        description=_TOOL_SEARCH_DESCRIPTION,
        tool_type=ToolType.FUNCTION,
        parameters={
            "query": ParameterSchema(
                name="query",
                type=ParameterType.STRING,
                description="Natural-language description of the capability you need.",
                required=True,
            ),
            "max_results": ParameterSchema(
                name="max_results",
                type=ParameterType.INTEGER,
                description=f"Maximum number of tools to return (default {DEFAULT_MAX_RESULTS}, max 20).",
                required=False,
                default=DEFAULT_MAX_RESULTS,
                minimum=1,
                maximum=20,
            ),
        },
    )
    schema.metadata["always_load"] = True
    schema.metadata["builtin"] = "tool_search"

    _tool_search._tool_schema = schema  # type: ignore[attr-defined]
    _tool_search._is_tool = True  # type: ignore[attr-defined]
    return FunctionTool(_tool_search, name=TOOL_SEARCH_TOOL_NAME, description=_TOOL_SEARCH_DESCRIPTION)
