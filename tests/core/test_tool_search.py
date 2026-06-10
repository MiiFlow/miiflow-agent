"""Tests for the provider-agnostic ToolSearch system."""

import asyncio

import pytest

from miiflow_agent.core.tools import (
    ToolRegistry,
    get_enabled_tool_names,
    tool,
    tool_search_session,
    TOOL_SEARCH_TOOL_NAME,
)
from miiflow_agent.core.tools.decorators import get_tool_from_function


def _make_dummy_tool(idx: int, *, always_load: bool = False, keywords=None):
    @tool(
        name=f"tool_{idx}",
        description=f"Dummy tool number {idx} for testing.",
        always_load=always_load,
        search_keywords=keywords,
    )
    def _fn(x: int = 0) -> int:
        return x + idx

    return get_tool_from_function(_fn)


def _make_registry(n: int) -> ToolRegistry:
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=10)
    for i in range(n):
        reg.register(_make_dummy_tool(i))
    return reg


def test_threshold_inactive_for_small_registry():
    reg = _make_registry(5)
    assert reg.should_use_tool_search() is False
    schemas = reg.get_schemas("anthropic")
    # No meta-tool injected.
    assert all(
        (s.get("name") if "name" in s else s.get("function", {}).get("name"))
        != TOOL_SEARCH_TOOL_NAME
        for s in schemas
    )


def test_threshold_active_for_large_registry():
    reg = _make_registry(50)
    assert reg.should_use_tool_search() is True


def test_filtered_schemas_only_meta_and_always_load_initially():
    reg = _make_registry(50)
    # Mark a couple of tools as always-loaded.
    reg.register(_make_dummy_tool(100, always_load=True))
    reg.register(_make_dummy_tool(101, always_load=True))

    schemas = reg.get_filtered_schemas("anthropic", enabled_names=set())
    names = {s.get("name") for s in schemas}
    assert TOOL_SEARCH_TOOL_NAME in names
    assert "tool_100" in names
    assert "tool_101" in names
    # A random non-always-load tool is hidden.
    assert "tool_3" not in names


def test_search_returns_relevant_tools():
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=2)

    @tool(description="Send an email to a recipient", search_keywords=["mail", "smtp"])
    def send_email(to: str, body: str) -> str:
        return "ok"

    @tool(description="Read a row from the postgres database")
    def read_db(table: str) -> str:
        return "row"

    @tool(description="Resize an image to a target size")
    def resize_image(path: str) -> str:
        return "ok"

    reg.register(get_tool_from_function(send_email))
    reg.register(get_tool_from_function(read_db))
    reg.register(get_tool_from_function(resize_image))

    results = reg.search("send mail", max_results=3)
    assert results, "expected at least one result for 'send mail'"
    assert results[0]["name"] == "send_email"

    results = reg.search("database row", max_results=3)
    assert results[0]["name"] == "read_db"


def test_tool_search_session_enables_discovered_tools():
    reg = _make_registry(50)

    async def go():
        with tool_search_session() as enabled:
            assert len(enabled) == 0
            meta = reg.get_tool_search_tool()
            # Invoke the meta-tool with a query that matches "tool_3".
            result = await meta.acall(query="dummy 3", max_results=2)
            assert result.success, result.error
            # The discovered tool names are now in the session's enabled set.
            # (get_enabled_tool_names returns a fresh set, not the live object.)
            assert get_enabled_tool_names() == set(enabled)
            assert any(n.startswith("tool_") for n in enabled)

            schemas = reg.get_filtered_schemas("anthropic", enabled_names=enabled)
            names = {s.get("name") for s in schemas}
            assert TOOL_SEARCH_TOOL_NAME in names
            # At least one discovered tool is now visible.
            assert names & set(enabled)

    asyncio.run(go())


def test_session_isolation_between_runs():
    reg = _make_registry(50)

    async def run_one(query: str):
        with tool_search_session() as enabled:
            meta = reg.get_tool_search_tool()
            await meta.acall(query=query, max_results=1)
            return set(enabled)

    async def go():
        a, b = await asyncio.gather(run_one("dummy 1"), run_one("dummy 7"))
        # Each session should have its own enabled set; they should not be merged.
        assert a and b
        # Outside any session, no enabled set is visible.
        assert get_enabled_tool_names() is None

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Schema-emission cap: the enabled set must never grow the SENT tool schema
# past the provider-safe budget, no matter how many tool_search calls run.
# Regression for the Anthropic "Schema is too complex for compilation" 400.
# ---------------------------------------------------------------------------

from miiflow_agent.core.tools import get_enabled_tool_names_ordered, mark_tools_enabled


def _names(schemas):
    return {s.get("name") for s in schemas}


def test_filtered_schemas_caps_to_provider_budget():
    # threshold 6 → non-meta budget = max(always, 6-1) = 5; +meta = 6 total.
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=6)
    for i in range(20):
        reg.register(_make_dummy_tool(i))
    reg.register(_make_dummy_tool(100, always_load=True))
    reg.register(_make_dummy_tool(101, always_load=True))

    # Simulate two tool_search calls enabling 5 tools each (oldest → newest).
    discovered = [f"tool_{i}" for i in range(10)]  # tool_0 .. tool_9
    schemas = reg.get_filtered_schemas("anthropic", enabled_names=discovered)

    names = _names(schemas)
    # meta + 2 always_load + 3 most-recent discovered = 6, never the full 12.
    assert len(schemas) == 6
    assert TOOL_SEARCH_TOOL_NAME in names
    assert {"tool_100", "tool_101"} <= names  # always_load pinned
    # Most-recent discovered kept; oldest evicted from the schema.
    assert {"tool_9", "tool_8", "tool_7"} <= names
    assert "tool_0" not in names and "tool_5" not in names


def test_always_load_floor_never_capped_away():
    # 8 always_load tools but a tiny threshold — they must ALL still be sent
    # (core tools can't be hidden), discovered get 0 slots.
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=4)
    for i in range(8):
        reg.register(_make_dummy_tool(i, always_load=True))
    for i in range(8, 20):
        reg.register(_make_dummy_tool(i))

    schemas = reg.get_filtered_schemas("anthropic", enabled_names=[f"tool_{i}" for i in range(8, 14)])
    names = _names(schemas)
    for i in range(8):
        assert f"tool_{i}" in names  # every always_load survives
    # No discovered tool fit under the always_load floor.
    assert not any(f"tool_{i}" in names for i in range(8, 20))


def test_mark_tools_enabled_is_lru_ordered():
    async def go():
        with tool_search_session():
            mark_tools_enabled(["a", "b", "c"])
            assert get_enabled_tool_names_ordered() == ["a", "b", "c"]
            # Re-marking 'a' moves it to the most-recent end.
            mark_tools_enabled(["a"])
            assert get_enabled_tool_names_ordered() == ["b", "c", "a"]

    asyncio.run(go())


def test_cap_keeps_lru_most_recent_after_rediscovery():
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=4)  # budget 3, no always_load → 3 discovered
    for i in range(10):
        reg.register(_make_dummy_tool(i))

    async def go():
        with tool_search_session():
            mark_tools_enabled(["tool_0", "tool_1", "tool_2", "tool_3", "tool_4"])
            mark_tools_enabled(["tool_0"])  # bump tool_0 to newest
            ordered = get_enabled_tool_names_ordered()
            schemas = reg.get_filtered_schemas("anthropic", enabled_names=ordered)
            names = _names(schemas)
            # budget = max(0, 4-1) = 3 most-recent: tool_3, tool_4, tool_0.
            assert "tool_0" in names and "tool_4" in names and "tool_3" in names
            assert "tool_1" not in names and "tool_2" not in names

    asyncio.run(go())


def test_recent_tool_call_names_seeds_approved_tool():
    """A resumed turn must keep the just-used tool visible: the seed includes
    the tool the model called in its last assistant turn (e.g. the approved
    mutation the reconstruction tells it to 'call again')."""
    from miiflow_agent.core.agent import _recent_tool_call_names
    from miiflow_agent.core.message import Message, MessageRole

    msgs = [
        Message(role=MessageRole.ASSISTANT, content="", tool_calls=[
            {"id": "t1", "type": "function", "function": {"name": "ask_user_clarification", "arguments": {}}}]),
        Message(role=MessageRole.USER, content="answers"),
        Message(role=MessageRole.ASSISTANT, content="", tool_calls=[
            {"id": "t2", "type": "function", "function": {"name": "google_ads_mutate", "arguments": {}}}]),
        Message(role=MessageRole.TOOL, content="approved — call it again", tool_call_id="t2"),
    ]
    assert _recent_tool_call_names(msgs) == {"google_ads_mutate"}
    assert _recent_tool_call_names(msgs, max_assistant_turns=2) == {
        "ask_user_clarification", "google_ads_mutate"}
    assert _recent_tool_call_names([]) == set()


def test_pinned_tool_never_evicted_by_cap():
    """A pinned continuation tool stays visible even when the discovered set is
    full and would otherwise push it past the cap."""
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=5)  # non-meta budget 4
    for i in range(20):
        reg.register(_make_dummy_tool(i))
    reg.register(_make_dummy_tool(99))  # the "approved" tool to continue

    discovered = [f"tool_{i}" for i in range(10)]
    schemas = reg.get_filtered_schemas(
        "anthropic", enabled_names=discovered, pinned_names=["tool_99"]
    )
    names = _names(schemas)
    assert "tool_99" in names  # pinned → always visible despite a full discovered set
    non_meta = [n for n in names if n != TOOL_SEARCH_TOOL_NAME]
    assert len(non_meta) == 4  # budget honored: pinned(1) + 3 most-recent discovered


def test_session_pinned_seed_survives_later_discovery():
    """End-to-end: a tool pinned at session open (as the agent seeds it from the
    last assistant turn) survives even after several tool_search calls fill the
    discovered set — the model can always call the tool it was told to."""
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=5)
    for i in range(20):
        reg.register(_make_dummy_tool(i))
    reg.register(_make_dummy_tool(99))

    async def go():
        with tool_search_session(initial={"tool_99"}):
            # Simulate the model running several tool_search calls first.
            mark_tools_enabled([f"tool_{i}" for i in range(8)])
            schemas = reg.get_filtered_schemas(
                "anthropic", enabled_names=get_enabled_tool_names_ordered()
            )
            assert "tool_99" in _names(schemas)  # pinned via session seed, not evicted

    asyncio.run(go())


def _make_heavy_tool(idx: int):
    """A tool with a large schema (~4KB) to exercise the SIZE cap."""
    @tool(name=f"heavy_{idx}", description="X" * 4000)
    def _fn(a: str = "") -> str:
        return a
    return get_tool_from_function(_fn)


def test_filtered_schemas_size_cap_evicts_heavy_tools():
    """A few param/description-heavy tools must be size-capped BELOW the count
    budget so the combined schema can't trip the provider compiler."""
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=6)  # size budget 6*1500=9000
    for i in range(5):
        reg.register(_make_heavy_tool(i))

    discovered = [f"heavy_{i}" for i in range(5)]  # oldest → newest
    schemas = reg.get_filtered_schemas("anthropic", enabled_names=discovered, pinned_names=[])
    non_meta = [n for n in _names(schemas) if n != TOOL_SEARCH_TOOL_NAME]

    # Count budget alone would allow 5; size budget (9000B / ~4.2KB each) caps fewer.
    assert 1 <= len(non_meta) < 5
    assert "heavy_4" in _names(schemas)  # most-recent always kept (no search-loop)


def test_size_cap_inactive_for_small_schemas():
    """When tools are small, the COUNT budget binds and size never evicts."""
    reg = ToolRegistry(enable_logging=False, tool_search_threshold=8)
    for i in range(20):
        reg.register(_make_dummy_tool(i))  # ~small schemas
    discovered = [f"tool_{i}" for i in range(20)]
    schemas = reg.get_filtered_schemas("anthropic", enabled_names=discovered, pinned_names=[])
    non_meta = [n for n in _names(schemas) if n != TOOL_SEARCH_TOOL_NAME]
    # Bound by count budget (threshold-1 = 7), not size.
    assert len(non_meta) == 7
