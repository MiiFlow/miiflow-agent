"""Bridge-model progressive tool disclosure.

The problem this solves
=======================

``tool_search.py`` hides undiscovered tools and *enables* the ones the model
finds. Enabling means the next turn's tools array contains more schemas than
this turn's. That array is the first thing the provider hashes for prompt
caching, so growing it mid-loop invalidates the tools tier — and because the
tiers are nested (tools → system → messages), invalidating tools cascades and
the entire prefix is re-billed uncached. In production that showed up as a
~40s stall on a parent turn.

The workaround was to not use it: ``tool_search_threshold_for`` sets the
threshold to 100 on non-Anthropic providers, so ToolSearch effectively never
activates and the full ~32K-token schema array ships on every single turn.
That is the cost this module removes.

The bridge
==========

Three tools replace the meta-tool:

    tool_search(query)              -> candidate names + one-line descriptions
    tool_describe(name)             -> that tool's full parameter schema
    tool_call(name, arguments)      -> invoke it

Discovered tools are invoked *through* ``tool_call`` rather than being added to
the array. So the array is exactly ``always_load + pinned + 3`` and **never
grows**, no matter how many tools the model discovers. The cache prefix is
stable for the whole run.

The catalog listing — what tools exist at all — goes in ``tool_search``'s
*description*. Hermes puts its equivalent in the system prompt, but this
package does not own the system prompt (the host application assembles it and
passes it in as a message), and a tool description reaches the same place for
caching purposes: the tools tier, ahead of system and messages, byte-stable as
long as the catalog is. Putting it there keeps the whole change inside the
tool layer.

Rendering is deterministic — groups and tools sorted by name — because a
listing that reorders between assemblies would defeat the very caching this
exists to protect.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .registry import ToolRegistry

logger = logging.getLogger(__name__)


TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_CALL_NAME = "tool_call"

#: Reserved — a registered tool may not take one of these names, or the bridge
#: would shadow it (and `tool_call` would recurse into itself).
BRIDGE_TOOL_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_CALL_NAME})

#: Token budget for the catalog listing embedded in the search tool's
#: description. Past this we degrade rather than truncate mid-entry: a listing
#: cut off halfway through a tool name is worse than a shorter complete one,
#: because the model will confidently call the fragment.
DEFAULT_CATALOG_MAX_TOKENS = 4_000

#: Chars per token for the listing budget. Deliberately the same cheap
#: heuristic the activation gate uses — the two must agree or the listing can
#: be judged in-budget by one and over by the other.
_CHARS_PER_TOKEN = 3.5

#: Longest one-line description kept per tool in the full listing.
_SHORT_DESC_CHARS = 90

DEFAULT_MAX_RESULTS = 5


def _short_desc(description: str) -> str:
    text = " ".join((description or "").split())
    if len(text) <= _SHORT_DESC_CHARS:
        return text
    return text[: _SHORT_DESC_CHARS - 1].rstrip() + "…"


def _fits(text: str, max_tokens: int) -> bool:
    return len(text) / _CHARS_PER_TOKEN <= max_tokens


def _group_of(name: str) -> str:
    """Bucket a tool into a listing group.

    Grouping is by name prefix — ``google_ads_query`` and ``google_ads_mutate``
    land together. Crude, but it needs no metadata the registry may not carry,
    and the only thing riding on it is readability of the listing plus which
    block gets collapsed first under budget pressure.
    """
    if "__" in name:  # MCP tools: server__tool
        return name.split("__", 1)[0]
    parts = name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0] if parts else "other"


def build_catalog_listing(
    entries: List[Tuple[str, str]],
    *,
    max_tokens: int = DEFAULT_CATALOG_MAX_TOKENS,
) -> Tuple[str, str]:
    """Render the deferred-tool catalog, degrading to fit ``max_tokens``.

    ``entries`` is ``[(name, description), ...]``. Returns ``(text, form)``
    where ``form`` is one of:

      ``full``    name + short description per tool
      ``names``   names only, grouped, comma-joined
      ``mixed``   per-group degradation — oversized groups collapse to a
                  count line, small ones keep their names
      ``groups``  every group collapsed to ``name (N tools)``
      ``none``    nothing fits (only when the group summary itself is over)

    Degradation is **per group**, not global. One enormous group (an MCP server
    exposing hundreds of tools) must not cost a small co-registered group its
    listing — otherwise attaching one big server silently makes every other
    tool undiscoverable-by-browsing.
    """
    if not entries:
        return "", "none"

    groups: Dict[str, List[Tuple[str, str]]] = {}
    for name, description in entries:
        groups.setdefault(_group_of(name), []).append((name, _short_desc(description)))

    header = (
        "Tools available on demand (not loaded yet). "
        f"Use `{TOOL_DESCRIBE_NAME}` for a tool's parameters, "
        f"then `{TOOL_CALL_NAME}` to invoke it:\n"
    )

    def render(label: str, mode: str) -> str:
        tools = sorted(groups[label])
        if mode == "summary":
            return (
                f"{label} ({len(tools)} tools — names not listed; "
                f"discover with `{TOOL_SEARCH_NAME}`)"
            )
        lines = [f"{label} ({len(tools)}):"]
        if mode == "full":
            for name, desc in tools:
                lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        else:
            lines.append("  " + ", ".join(name for name, _ in tools))
        return "\n".join(lines)

    def assemble(modes: Dict[str, str]) -> str:
        return header + "\n".join(render(label, modes[label]) for label in sorted(groups))

    for uniform in ("full", "names"):
        text = assemble({label: uniform for label in groups})
        if _fits(text, max_tokens):
            return text, uniform

    # Per-group greedy fit: collapse the largest groups first, smallest last,
    # so the most tools stay individually visible.
    modes = {label: "summary" for label in groups}
    by_size = sorted(groups, key=lambda label: len(groups[label]))
    for label in by_size:
        trial = dict(modes)
        trial[label] = "names"
        if _fits(assemble(trial), max_tokens):
            modes = trial
        else:
            break

    text = assemble(modes)
    if _fits(text, max_tokens):
        return text, ("groups" if all(m == "summary" for m in modes.values()) else "mixed")

    # Even the all-summary form is over budget. Emit it anyway rather than
    # nothing: knowing WHICH domains are reachable is what lets the model form
    # a useful `tool_search` query, and that is strictly better than a bare
    # bridge with no hint at all.
    logger.warning(
        "[TOOL_BRIDGE] catalog summary exceeds %d tokens (%d groups, %d tools); "
        "emitting anyway",
        max_tokens,
        len(groups),
        len(entries),
    )
    return text, "none"


def build_bridge_tools(
    registry: "ToolRegistry",
    *,
    deferred: List[Tuple[str, str]],
    catalog_max_tokens: int = DEFAULT_CATALOG_MAX_TOKENS,
) -> List[Any]:
    """Build the three bridge tools for ``registry``.

    ``deferred`` is ``[(name, description), ...]`` for every tool hidden behind
    the bridge — used to render the catalog and to reject calls to tools that
    are not actually reachable.
    """
    from .function import FunctionTool
    from .schemas import ParameterSchema, ToolSchema
    from .types import ParameterType, ToolType

    catalog_text, catalog_form = build_catalog_listing(
        deferred, max_tokens=catalog_max_tokens
    )
    deferred_names = {name for name, _ in deferred}

    # ---- tool_search ---------------------------------------------------

    async def _tool_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
        max_results = max(1, min(int(max_results or DEFAULT_MAX_RESULTS), 20))
        matches = registry.search(query, max_results=max_results)
        # Deliberately NOT calling `mark_tools_enabled`. Enabling is what grew
        # the tools array and busted the cache; under the bridge the model
        # reaches a discovered tool through `tool_call` instead.
        if not matches:
            return {
                "query": query,
                "results": [],
                "message": (
                    "No tools matched. Try different keywords, or check the "
                    f"catalog in the `{TOOL_SEARCH_NAME}` description."
                ),
            }
        return {
            "query": query,
            "results": [
                {
                    "name": m.get("name"),
                    "description": _short_desc(m.get("description", "")),
                }
                for m in matches
            ],
            "message": (
                f"{len(matches)} tool(s) found. Call `{TOOL_DESCRIBE_NAME}` for "
                f"parameters, then `{TOOL_CALL_NAME}` to invoke."
            ),
        }

    search_description = (
        "Search the tool catalog by keyword when you need a capability that is "
        "not in your currently loaded tools. Returns matching tool names and "
        f"short descriptions; use `{TOOL_DESCRIBE_NAME}` for parameters and "
        f"`{TOOL_CALL_NAME}` to invoke.\n\n" + catalog_text
    )

    search_schema = ToolSchema(
        name=TOOL_SEARCH_NAME,
        description=search_description,
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
                description=f"Maximum results (default {DEFAULT_MAX_RESULTS}, max 20).",
                required=False,
                default=DEFAULT_MAX_RESULTS,
                minimum=1,
                maximum=20,
            ),
        },
    )

    # ---- tool_describe -------------------------------------------------

    async def _tool_describe(name: str) -> Dict[str, Any]:
        schema = registry._get_universal_schema(name)
        if schema is None:
            suggestion = _closest(name, deferred_names)
            return {
                "error": f"No tool named {name!r}.",
                "hint": (
                    f"Did you mean {suggestion!r}?"
                    if suggestion
                    else f"Use `{TOOL_SEARCH_NAME}` to find the right name."
                ),
            }
        return {
            "name": schema.get("name"),
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {}),
            "usage": (
                f'Invoke with {TOOL_CALL_NAME}(name="{schema.get("name")}", '
                'arguments={...}) using the parameters above.'
            ),
        }

    describe_schema = ToolSchema(
        name=TOOL_DESCRIBE_NAME,
        description=(
            "Get the full parameter schema for a tool by exact name. Required "
            f"before `{TOOL_CALL_NAME}` unless you already know the tool's "
            "parameters."
        ),
        tool_type=ToolType.FUNCTION,
        parameters={
            "name": ParameterSchema(
                name="name",
                type=ParameterType.STRING,
                description=f"Exact tool name, as returned by `{TOOL_SEARCH_NAME}`.",
                required=True,
            ),
        },
    )

    # ---- tool_call -----------------------------------------------------

    async def _tool_call(name: str, arguments: Optional[Dict[str, Any]] = None, **kwargs):
        """Unreachable by design — see :func:`unwrap_bridge_call`.

        ``tool_call`` is rewritten into a direct call to the named tool by the
        tool executor *before* dispatch, so that approval gating, context
        injection, PRE/POST_TOOL_USE callbacks, dedup, observation recording
        and result truncation all see the real tool name and behave exactly as
        they would for a directly-loaded tool. Dispatching here instead would
        silently skip every one of those — most seriously the approval gate,
        which would make every deferred tool self-approving.

        If this handler ever runs, the unwrap did not happen. Fail loudly
        rather than quietly executing on the wrong path.
        """
        logger.error(
            "[TOOL_BRIDGE] %s handler reached for %r — unwrap did not run; "
            "the call was NOT executed",
            TOOL_CALL_NAME,
            name,
        )
        return {
            "error": (
                f"Internal routing error: `{TOOL_CALL_NAME}` was not unwrapped. "
                "The call was not executed."
            )
        }

    call_schema = ToolSchema(
        name=TOOL_CALL_NAME,
        description=(
            "Invoke a tool from the catalog by exact name. Pass its parameters "
            "as the `arguments` object, matching the schema from "
            f"`{TOOL_DESCRIBE_NAME}`."
        ),
        tool_type=ToolType.FUNCTION,
        parameters={
            "name": ParameterSchema(
                name="name",
                type=ParameterType.STRING,
                description="Exact tool name to invoke.",
                required=True,
            ),
            "arguments": ParameterSchema(
                name="arguments",
                type=ParameterType.OBJECT,
                description="Arguments for the tool, matching its schema.",
                required=True,
            ),
        },
    )

    tools = []
    for fn, schema, name, description in (
        (_tool_search, search_schema, TOOL_SEARCH_NAME, search_description),
        (_tool_describe, describe_schema, TOOL_DESCRIBE_NAME, describe_schema.description),
        (_tool_call, call_schema, TOOL_CALL_NAME, call_schema.description),
    ):
        schema.metadata["always_load"] = True
        schema.metadata["builtin"] = "tool_bridge"
        fn._tool_schema = schema  # type: ignore[attr-defined]
        fn._is_tool = True  # type: ignore[attr-defined]
        tools.append(FunctionTool(fn, name=name, description=description))

    logger.info(
        "[TOOL_BRIDGE] built bridge over %d deferred tools (catalog form=%s, "
        "~%d catalog tokens)",
        len(deferred),
        catalog_form,
        int(len(catalog_text) / _CHARS_PER_TOKEN),
    )
    return tools


def unwrap_bridge_call(
    tool_name: str, inputs: Dict[str, Any]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Rewrite a ``tool_call`` invocation into the real call it stands for.

    Returns ``(real_name, real_args)``, or ``None`` when ``tool_name`` is not
    ``tool_call`` (the overwhelmingly common case — this runs on every tool
    dispatch, so the miss path is a single string compare).

    Called by the tool executor *before* the approval gate, so everything
    downstream — gating, context injection, callbacks, dedup, tracing, the
    activity feed — sees the underlying tool rather than the bridge. A user
    approving ``google_ads_mutate`` must be shown ``google_ads_mutate``, not
    ``tool_call``.

    Returns ``(tool_name, inputs)`` unchanged when the payload is malformed
    (missing/blank ``name``, or a name that is itself a bridge tool). Letting
    the real handler run then produces a clear error message the model can act
    on, which beats raising here — a raise mid-dispatch surfaces as a step
    failure with no usable guidance.
    """
    if tool_name != TOOL_CALL_NAME:
        return None

    target = inputs.get("name")
    if not isinstance(target, str) or not target.strip():
        return None
    target = target.strip()
    if target in BRIDGE_TOOL_NAMES:
        # A confused model can otherwise drive tool_call into itself and burn
        # the whole iteration budget on a no-op recursion.
        return None

    arguments = inputs.get("arguments")
    args: Dict[str, Any] = dict(arguments) if isinstance(arguments, dict) else {}

    if isinstance(arguments, str):
        # Some models emit `arguments` as a JSON *string* rather than an
        # object. Recovering it costs one parse and saves a wasted turn.
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                args = parsed
        except (ValueError, TypeError):
            pass

    # Models sometimes flatten the arguments object into the call itself
    # (`tool_call(name="x", customer_id="123")`). Accept both shapes rather
    # than failing on a formatting slip the model can't see from the error.
    for key, value in inputs.items():
        if key not in ("name", "arguments"):
            args.setdefault(key, value)

    return target, args


def _closest(name: str, candidates) -> Optional[str]:
    """Nearest catalog name, for a typo hint. ``None`` when nothing is close.

    Returning a bad suggestion is worse than returning none — the model will
    take it and call the wrong tool — so the cutoff is deliberately strict.
    """
    import difflib

    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.7)
    return matches[0] if matches else None
