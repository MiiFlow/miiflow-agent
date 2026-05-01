"""Tests for the session-scoped data reference cache.

Used by data tools (e.g. google_ads_query) to stash large result sets so
render_table can reference them by id, avoiding the LLM re-emitting
hundreds of rows in the render_table(rows=[...]) parameter — which was
the dominant max_tokens-truncation cause in the original flood.
"""

from types import SimpleNamespace

from miiflow_agent.core.data_reference import (
    _MAX_ENTRIES,
    get_render_data,
    put_render_data,
)


def _ctx():
    """Minimal RunContext-shaped object — only `metadata` is read."""
    return SimpleNamespace(metadata={})


def test_put_returns_unique_id():
    ctx = _ctx()
    a = put_render_data(ctx, [{"x": 1}])
    b = put_render_data(ctx, [{"x": 2}])
    assert a and b and a != b
    assert a.startswith("ref_")


def test_get_round_trips_rows():
    ctx = _ctx()
    rows = [{"name": "A", "spend": 100}, {"name": "B", "spend": 200}]
    data_id = put_render_data(ctx, rows)
    cached = get_render_data(ctx, data_id)
    assert cached is not None
    assert cached["rows"] == rows


def test_get_returns_none_for_missing_id():
    ctx = _ctx()
    assert get_render_data(ctx, "ref_nope") is None


def test_get_returns_none_for_empty_id():
    ctx = _ctx()
    assert get_render_data(ctx, "") is None
    assert get_render_data(ctx, None) is None


def test_extras_are_carried_through():
    ctx = _ctx()
    data_id = put_render_data(
        ctx, [{"x": 1}], extras={"source": "google_ads_query", "total_rows": 1}
    )
    cached = get_render_data(ctx, data_id)
    assert cached["source"] == "google_ads_query"
    assert cached["total_rows"] == 1


def test_put_returns_none_when_ctx_has_no_metadata():
    """Tools should be safe to call even with a stub ctx in unit tests."""
    bare_ctx = object()
    assert put_render_data(bare_ctx, [{"x": 1}]) is None


def test_fifo_eviction_caps_memory():
    """Long-running sessions can't grow the cache without bound."""
    ctx = _ctx()
    ids = [put_render_data(ctx, [{"i": i}]) for i in range(_MAX_ENTRIES + 5)]
    # First 5 should have been evicted.
    for evicted in ids[:5]:
        assert get_render_data(ctx, evicted) is None
    # Last _MAX_ENTRIES are retained.
    for kept in ids[5:]:
        assert get_render_data(ctx, kept) is not None


def test_put_no_op_on_none_rows():
    ctx = _ctx()
    assert put_render_data(ctx, None) is None
