"""Session-scoped data reference cache for render tools.

When data tools (e.g. google_ads_query, meta_ads_insights) return large
result sets, the LLM has to re-emit the data inline whenever it wants to
render it — `render_table(rows=[...several hundred rows...])`. At
realistic dataset sizes this blows the model's max_tokens budget and
produces truncated tool calls (the original flood of "missing required
parameters" errors).

This module stashes the data on the RunContext under an opaque id. Data
tools call ``put_render_data`` after producing a dataset and surface the
id to the model (typically as ``data_id`` in their tool result). Render
tools accept the id as an alternative to inline rows; when set, they
fetch from the cache and the model never re-emits the data.

The cache is session-scoped (lives on RunContext.metadata, GC'd when the
session ends) and FIFO-bounded so a long-running session can't grow it
without limit.
"""

import secrets
from typing import Any, Dict, List, Optional

_CACHE_KEY = "_render_cache"
_MAX_ENTRIES = 50


def put_render_data(
    ctx: Any,
    rows: List[Dict[str, Any]],
    extras: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Cache ``rows`` on the RunContext and return an opaque id.

    Returns None if ``ctx`` has no ``metadata`` attribute (e.g. the tool
    is being unit-tested without a real RunContext).
    """
    if rows is None or not hasattr(ctx, "metadata"):
        return None

    cache: Dict[str, Dict[str, Any]] = ctx.metadata.setdefault(_CACHE_KEY, {})

    while len(cache) >= _MAX_ENTRIES:
        oldest = next(iter(cache))
        cache.pop(oldest, None)

    data_id = "ref_" + secrets.token_urlsafe(8)
    entry: Dict[str, Any] = {"rows": list(rows)}
    if extras:
        entry.update(extras)
    cache[data_id] = entry
    return data_id


def get_render_data(ctx: Any, data_id: str) -> Optional[Dict[str, Any]]:
    """Look up cached data by id. Returns None if missing or untrusted ctx."""
    if not data_id or not hasattr(ctx, "metadata"):
        return None
    cache = ctx.metadata.get(_CACHE_KEY)
    if not cache:
        return None
    return cache.get(data_id)
