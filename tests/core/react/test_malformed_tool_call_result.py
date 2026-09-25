"""A malformed tool call from the model is a failed ToolResult, not a crash.

The single-tool path used to ``raise Exception`` for an unknown tool name or a
non-dict input; the generic handler logged that with ``exc_info`` and every
hallucinated tool name became a PostHog Error Tracking issue (prod 2026-09-24:
``Tool 'get_ai_visibility' not found``). The registry and the batch path
already answered the same condition with a failed ``ToolResult``.
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.core.react.tool_actions import ToolActionHandler


def _handler(known_tools, schema=None):
    executor = SimpleNamespace(
        has_tool=lambda name: name in known_tools,
        list_tools=lambda: sorted(known_tools),
        get_tool_schema=lambda name: schema or {},
    )
    return ToolActionHandler(SimpleNamespace(tool_executor=executor))


def _step(action, action_input):
    return SimpleNamespace(action=action, action_input=action_input, error=None)


def test_unknown_tool_is_a_failed_result():
    result = asyncio.run(
        _handler({"get_performance"}).execute_tool(_step("get_metrics", {"x": 1}), None)
    )

    assert result.success is False
    assert result.error == "Tool 'get_metrics' not found. Available: ['get_performance']"
    assert result.metadata == {"error_type": "tool_not_found"}


def test_non_dict_input_for_a_multi_param_tool_is_a_failed_result():
    schema = {"parameters": {"properties": {"a": {}, "b": {}}}}
    result = asyncio.run(
        _handler({"two_params"}, schema).execute_tool(_step("two_params", "oops"), None)
    )

    assert result.success is False
    assert "expects dict input" in result.error
    assert result.metadata == {"error_type": "invalid_input"}
