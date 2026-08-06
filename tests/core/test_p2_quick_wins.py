"""Tests for the P2 quick wins: framework POST_TOOL_USE emission (opt-in)
and schema-size memoization."""

from unittest.mock import MagicMock

import pytest

from miiflow_agent.core.callbacks import (
    CallbackEventType,
    get_global_registry,
)
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import ToolRegistry, ToolResult, tool


def _executor_with_registry(registry):
    agent = MagicMock()
    agent.tool_registry = registry
    agent.client = MagicMock()
    executor = AgentToolExecutor(agent)
    executor._tool_registry = registry
    return executor


@tool(name="echo_tool")
async def echo_tool(text: str) -> str:
    """Echo the input text."""
    return f"echo: {text}"


class TestFrameworkPostToolUse:
    async def test_disabled_by_default(self):
        registry = ToolRegistry()
        registry.register(echo_tool)
        executor = _executor_with_registry(registry)
        assert executor._emit_post_tool_use is False

        fired = []
        get_global_registry().register(
            CallbackEventType.POST_TOOL_USE, lambda e: fired.append(e)
        )
        try:
            result = await executor.execute_tool("echo_tool", {"text": "hi"})
        finally:
            get_global_registry().clear(CallbackEventType.POST_TOOL_USE)

        assert result.success
        assert fired == []

    async def test_opt_in_emits_and_applies_transform(self):
        registry = ToolRegistry()
        registry.register(echo_tool)
        executor = _executor_with_registry(registry)
        executor._emit_post_tool_use = True

        def transform(event):
            if event.tool_name == "echo_tool":
                event.transformed_output = event.tool_output + " [enriched]"
                event.output_transformed = True

        get_global_registry().register(CallbackEventType.POST_TOOL_USE, transform)
        try:
            result = await executor.execute_tool("echo_tool", {"text": "hi"})
        finally:
            get_global_registry().clear(CallbackEventType.POST_TOOL_USE)

        assert result.success
        assert result.output == "echo: hi [enriched]"

    async def test_hook_failure_does_not_fail_the_tool(self):
        registry = ToolRegistry()
        registry.register(echo_tool)
        executor = _executor_with_registry(registry)
        executor._emit_post_tool_use = True

        def broken(event):
            raise RuntimeError("hook exploded")

        get_global_registry().register(CallbackEventType.POST_TOOL_USE, broken)
        try:
            result = await executor.execute_tool("echo_tool", {"text": "hi"})
        finally:
            get_global_registry().clear(CallbackEventType.POST_TOOL_USE)

        # emit() contains callback errors; the tool result is unaffected.
        assert result.success
        assert result.output == "echo: hi"


class TestSchemaSizeCache:
    def test_size_is_memoized_and_invalidated_on_registration(self):
        registry = ToolRegistry()
        registry.register(echo_tool)

        first = registry._schema_size("echo_tool")
        assert first > 0
        assert registry._schema_size_cache["echo_tool"] == first
        # Cached value is served (poison the cache to prove it).
        registry._schema_size_cache["echo_tool"] = 42
        assert registry._schema_size("echo_tool") == 42

        # Any registration invalidates.
        @tool(name="other_tool")
        async def other_tool(x: str) -> str:
            """Other."""
            return x

        registry.register(other_tool)
        assert "echo_tool" not in registry._schema_size_cache
        assert registry._schema_size("echo_tool") == first
