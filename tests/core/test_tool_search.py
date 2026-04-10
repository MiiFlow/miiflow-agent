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
            assert enabled == set()
            meta = reg.get_tool_search_tool()
            # Invoke the meta-tool with a query that matches "tool_3".
            result = await meta.acall(query="dummy 3", max_results=2)
            assert result.success, result.error
            # The session set should now contain the discovered tool names.
            assert get_enabled_tool_names() is enabled
            assert any(n.startswith("tool_") for n in enabled)

            schemas = reg.get_filtered_schemas("anthropic", enabled_names=enabled)
            names = {s.get("name") for s in schemas}
            assert TOOL_SEARCH_TOOL_NAME in names
            # At least one discovered tool is now visible.
            assert names & set(enabled)

    asyncio.get_event_loop().run_until_complete(go())


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

    asyncio.get_event_loop().run_until_complete(go())
