"""``_build_native_tool_schemas`` is memoized per tool surface within a run.

It converts every registered tool through the provider's (deep-copying)
normalizer and ran at least twice per step — once to size the request, once
for the call — plus once per loop step even though the surface almost never
changes within a run. The memo must hit when nothing changed, rebuild when the
surface changes, and never hand out the dicts it holds (callers add
``cache_control`` / ``defer_loading`` in place).
"""

from unittest.mock import MagicMock

import pytest

from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import ToolRegistry, tool, tool_search_session
from miiflow_agent.core.tools.decorators import get_tool_from_function
from miiflow_agent.core.tools.tool_search import mark_tools_enabled

pytestmark = pytest.mark.unit


def _make_dummy_tool(idx: int):
    @tool(name=f"tool_{idx}", description=f"Dummy tool number {idx}.")
    def _fn(x: int = 0) -> int:
        return x + idx

    return get_tool_from_function(_fn)


def _executor(n_tools: int, provider: str = "anthropic", threshold: int = 10):
    registry = ToolRegistry(enable_logging=False, tool_search_threshold=threshold)
    for i in range(n_tools):
        registry.register(_make_dummy_tool(i))
    model_client = MagicMock()
    model_client.provider_name = provider
    model_client.convert_schema_to_provider_format = MagicMock(side_effect=lambda x: dict(x))
    llm_client = MagicMock()
    llm_client.client = model_client
    llm_client.tool_registry = registry
    agent = MagicMock()
    agent.client = llm_client
    agent.tool_registry = registry
    agent._tools = []
    return AgentToolExecutor(agent), registry, model_client.convert_schema_to_provider_format


def test_second_build_is_served_from_the_memo():
    executor, _registry, convert = _executor(6)
    first = executor._build_native_tool_schemas()
    calls_after_first = convert.call_count
    assert calls_after_first == 6

    second = executor._build_native_tool_schemas()
    assert convert.call_count == calls_after_first
    assert second == first


def test_memo_hands_out_copies_not_its_own_dicts():
    executor, _registry, _convert = _executor(4)
    first = executor._build_native_tool_schemas()
    second = executor._build_native_tool_schemas()
    assert all(a is not b for a, b in zip(first, second))

    # What a provider client does to the list it sends must not persist.
    first[-1]["cache_control"] = {"type": "ephemeral"}
    first[0]["defer_loading"] = True
    third = executor._build_native_tool_schemas()
    assert "cache_control" not in third[-1]
    assert "defer_loading" not in third[0]


def test_discovery_within_a_session_rebuilds():
    executor, _registry, convert = _executor(15, provider="openai")
    with tool_search_session():
        executor._build_native_tool_schemas()
        before = convert.call_count
        mark_tools_enabled(["tool_3"])
        schemas = executor._build_native_tool_schemas()
    assert convert.call_count > before
    assert "tool_3" in {s.get("name") for s in schemas if isinstance(s, dict)}


def test_always_load_change_rebuilds():
    executor, registry, convert = _executor(15)
    with tool_search_session():
        executor._build_native_tool_schemas()
        before = convert.call_count
        registry.mark_always_load(["tool_4"])
        schemas = executor._build_native_tool_schemas()
    assert convert.call_count > before
    by_name = {s["name"]: s for s in schemas if isinstance(s, dict) and "name" in s}
    assert not by_name["tool_4"].get("defer_loading")


def test_registration_rebuilds():
    executor, registry, convert = _executor(3)
    executor._build_native_tool_schemas()
    before = convert.call_count
    registry.register(_make_dummy_tool(99))
    schemas = executor._build_native_tool_schemas()
    assert convert.call_count > before
    assert "tool_99" in {s.get("name") for s in schemas if isinstance(s, dict)}


def test_instance_built_without_init_still_builds():
    """Tests construct executors via __new__; the class default must not break them."""
    executor = AgentToolExecutor.__new__(AgentToolExecutor)
    assert executor._schema_memo is None
