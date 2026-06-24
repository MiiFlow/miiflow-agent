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


# ── Agent-level pre-load threshold override ──────────────────────────────────
# Adapters (e.g. the Django layer) can force a modest, stable tool surface to be
# pre-loaded in full instead of deferred behind tool_search, by passing
# `tool_search_threshold=` to Agent(...). Pre-loading keeps the native tools
# array byte-identical across ReAct iterations so Anthropic's prompt-cache
# breakpoints keep hitting. The override must win over provider calibration and
# must NOT leak across constructions that share a cached client/registry.


def _agent(provider: str, **kwargs):
    from unittest.mock import AsyncMock, MagicMock

    from miiflow_agent.core.agent import Agent, AgentType
    from miiflow_agent.core.client import LLMClient

    provider_client = MagicMock()
    provider_client.provider_name = provider
    provider_client.achat = AsyncMock()
    provider_client.convert_schema_to_provider_format = MagicMock(side_effect=lambda s: s)
    return Agent(LLMClient(provider_client), agent_type=AgentType.REACT, **kwargs)


def test_agent_calibrates_threshold_to_provider_by_default():
    # Anthropic's strict grammar compiler trips early → low ceiling.
    assert _agent("anthropic").tool_registry.tool_search_threshold == 12
    assert _agent("openai").tool_registry.tool_search_threshold == 25


def test_agent_tool_search_threshold_override_wins_over_calibration():
    agent = _agent("anthropic", tool_search_threshold=100)
    assert agent.tool_registry.tool_search_threshold == 100


def test_agent_threshold_override_does_not_leak_across_shared_registry():
    # Same client/registry reused (LlmRegistry may cache clients). An override
    # on one construction must not stick to the next non-override construction.
    from unittest.mock import AsyncMock, MagicMock

    from miiflow_agent.core.agent import Agent, AgentType
    from miiflow_agent.core.client import LLMClient

    provider_client = MagicMock()
    provider_client.provider_name = "anthropic"
    provider_client.achat = AsyncMock()
    provider_client.convert_schema_to_provider_format = MagicMock(side_effect=lambda s: s)
    shared = LLMClient(provider_client)

    overridden = Agent(shared, agent_type=AgentType.REACT, tool_search_threshold=100)
    assert overridden.tool_registry.tool_search_threshold == 100

    plain = Agent(shared, agent_type=AgentType.REACT)
    assert plain.tool_registry.tool_search_threshold == 12


def test_tool_search_result_is_compact_no_full_schema():
    """The tool_search observation returns only name/description/param-names —
    NOT the full JSON-Schema. Full schemas are re-emitted next turn by
    get_filtered_schemas, so echoing them here is redundant and inflates the
    cached tool_result (the ~36KB prod regression)."""
    import json

    reg = _make_registry(50)

    async def go():
        with tool_search_session():
            meta = reg.get_tool_search_tool()
            result = await meta.acall(query="dummy 3", max_results=2)
            assert result.success, result.error
            out = result.output
            assert out["results"], "expected matches"
            for entry in out["results"]:
                # Compact shape: name + description + param NAMES only.
                assert set(entry) <= {"name", "description", "params", "required"}
                assert "parameters" not in entry  # no full schema
                if "params" in entry:
                    assert all(isinstance(p, str) for p in entry["params"])
                    assert entry["params"] == ["x"]  # the dummy tool's only param
            # Sanity on size: a 2-match result must stay tiny.
            assert len(json.dumps(out)) < 800

    asyncio.run(go())


# ── Native Anthropic server-side tool search (defer_loading) ─────────────────
# When the registry warrants deferral AND the provider is first-party Anthropic,
# _build_native_tool_schemas sends the FULL tool list with defer_loading on the
# non-core tools and appends Anthropic's server search tool — replacing the
# in-process meta-tool. The API strips deferred tools from the prompt, so the
# tools cache prefix stays stable. (Live-API behavior verified separately.)

def _executor(provider: str, n_tools: int, threshold: int, always_load_idx=()):
    from unittest.mock import AsyncMock, MagicMock

    from miiflow_agent.core.agent import Agent, AgentType
    from miiflow_agent.core.client import LLMClient
    from miiflow_agent.core.react.tool_executor import AgentToolExecutor
    from miiflow_agent.core.tools.decorators import get_tool_from_function

    provider_client = MagicMock()
    provider_client.provider_name = provider
    provider_client.achat = AsyncMock()
    # Passthrough conversion: return a minimal provider dict carrying the name.
    provider_client.convert_schema_to_provider_format = MagicMock(
        side_effect=lambda s: {"name": s["name"], "description": s.get("description", ""),
                               "input_schema": s.get("parameters", {})}
    )
    tools = []
    for i in range(n_tools):
        def _fn(x: int = 0, _i=i):
            return x + _i
        _fn.__name__ = f"nt_{i}"
        tools.append(get_tool_from_function(
            tool(name=f"nt_{i}", description=f"tool {i}", always_load=(i in always_load_idx))(_fn)
        ))
    agent = Agent(client=LLMClient(provider_client), agent_type=AgentType.REACT,
                  tools=tools, tool_search_threshold=threshold)
    return AgentToolExecutor(agent)


def _native_names(schemas):
    return [s.get("name") for s in schemas if isinstance(s, dict) and "name" in s]


def test_native_tool_search_anthropic_defers_and_appends_search_tool():
    from miiflow_agent.core.react.tool_executor import NATIVE_TOOL_SEARCH_TOOL

    ex = _executor("anthropic", n_tools=5, threshold=2, always_load_idx={0})
    schemas = ex._build_native_tool_schemas()
    # Server search tool is appended LAST and is never deferred.
    assert schemas[-1] == NATIVE_TOOL_SEARCH_TOOL
    assert "defer_loading" not in schemas[-1]
    # All 5 regular tools are still sent (full list), not hidden.
    assert {f"nt_{i}" for i in range(5)} <= set(_native_names(schemas))
    # always_load tool (nt_0) is NOT deferred; the rest ARE.
    by_name = {s["name"]: s for s in schemas if "name" in s}
    assert "defer_loading" not in by_name["nt_0"]
    assert all(by_name[f"nt_{i}"].get("defer_loading") is True for i in range(1, 5))
    # The in-process meta-tool is NOT emitted on the native path.
    assert "tool_search" not in _native_names(schemas)
    # cache_control never co-exists with defer_loading (would 400).
    assert not any(s.get("defer_loading") and "cache_control" in s for s in schemas)


def test_native_tool_search_only_for_anthropic():
    # Non-Anthropic provider keeps the in-process meta-tool path: no defer_loading,
    # and the native server search tool is absent.
    ex = _executor("openai", n_tools=5, threshold=2)
    schemas = ex._build_native_tool_schemas()
    assert not any(isinstance(s, dict) and s.get("defer_loading") for s in schemas)
    assert not any(isinstance(s, dict) and s.get("type", "").startswith("tool_search_tool") for s in schemas)


def test_native_tool_search_off_below_threshold():
    # Corpus at/below threshold → no deferral at all (pre-load everything),
    # even on Anthropic. No defer flags, no server search tool.
    ex = _executor("anthropic", n_tools=3, threshold=10)
    schemas = ex._build_native_tool_schemas()
    assert not any(isinstance(s, dict) and s.get("defer_loading") for s in schemas)
    assert not any(isinstance(s, dict) and s.get("type", "").startswith("tool_search_tool") for s in schemas)
