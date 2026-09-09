"""Pinning the previous turn's tools must not change the tools array on ordinary turns.

The ToolSearch session used to be seeded from the last assistant turn's tool
calls on EVERY run. A pinned tool is sent resident (no ``defer_loading``), so
two consecutive turns whose last tool sets differed produced two different
tools arrays — and the tools array is the first prompt-cache tier, so that one
flag flip re-billed the whole prefix. Now the seed is empty unless the run is a
resumed one (or ``MIIFLOW_PIN_RECENT_TOOLS=always``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miiflow_agent.core.agent import (
    RunContext,
    _context_is_resume,
    _recent_tools_to_pin,
)
from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import ToolRegistry, tool, tool_search_session
from miiflow_agent.core.tools.decorators import get_tool_from_function

pytestmark = pytest.mark.unit


def _make_dummy_tool(idx: int):
    @tool(name=f"tool_{idx}", description=f"Dummy tool number {idx}.")
    def _fn(x: int = 0) -> int:
        return x + idx

    return get_tool_from_function(_fn)


def _history_calling(tool_name: str):
    """A paused turn: the last assistant message is the tool call awaiting
    execution, followed by the user's approval — the shape a resume replays."""
    return [
        Message(role=MessageRole.USER, content="do the thing"),
        Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {"id": "call_1", "function": {"name": tool_name, "arguments": "{}"}}
            ],
        ),
        Message(role=MessageRole.TOOL, content="paused for approval", tool_call_id="call_1"),
        Message(role=MessageRole.USER, content="approved, go ahead"),
    ]


def _anthropic_executor(n_tools: int = 15) -> AgentToolExecutor:
    registry = ToolRegistry(enable_logging=False, tool_search_threshold=10)
    for i in range(n_tools):
        registry.register(_make_dummy_tool(i))
    model_client = MagicMock()
    model_client.provider_name = "anthropic"
    model_client.convert_schema_to_provider_format = MagicMock(side_effect=lambda x: x)
    llm_client = MagicMock()
    llm_client.client = model_client
    llm_client.tool_registry = registry
    agent = MagicMock()
    agent.client = llm_client
    agent.tool_registry = registry
    agent._tools = []
    return AgentToolExecutor(agent)


def _by_name(schemas):
    return {s["name"]: s for s in schemas if isinstance(s, dict) and "name" in s}


class TestPolicy:
    def test_default_policy_pins_only_on_resume(self, monkeypatch):
        monkeypatch.delenv("MIIFLOW_PIN_RECENT_TOOLS", raising=False)
        history = _history_calling("tool_3")
        assert _recent_tools_to_pin(history, is_resume=None) == set()
        assert _recent_tools_to_pin(history, is_resume=False) == set()
        assert _recent_tools_to_pin(history, is_resume=True) == {"tool_3"}

    def test_always_restores_previous_behaviour(self, monkeypatch):
        monkeypatch.setenv("MIIFLOW_PIN_RECENT_TOOLS", "always")
        assert _recent_tools_to_pin(_history_calling("tool_3"), is_resume=False) == {
            "tool_3"
        }

    def test_never_disables_pinning_even_on_resume(self, monkeypatch):
        monkeypatch.setenv("MIIFLOW_PIN_RECENT_TOOLS", "never")
        assert _recent_tools_to_pin(_history_calling("tool_3"), is_resume=True) == set()

    def test_unknown_policy_reads_as_default(self, monkeypatch):
        monkeypatch.setenv("MIIFLOW_PIN_RECENT_TOOLS", "sometimes")
        history = _history_calling("tool_3")
        assert _recent_tools_to_pin(history, is_resume=False) == set()
        assert _recent_tools_to_pin(history, is_resume=True) == {"tool_3"}


class TestContextIsResume:
    def test_plain_context_is_not_a_resume(self):
        assert _context_is_resume(RunContext(deps=None)) is False

    def test_resume_command_marks_a_resume(self):
        ctx = RunContext(deps=None, resume=SimpleNamespace(kind="approval"))
        assert _context_is_resume(ctx) is True

    def test_host_flag_wins_both_ways(self):
        assert _context_is_resume(RunContext(deps=None, pin_recent_tools=True)) is True
        ctx = RunContext(
            deps=None, resume=SimpleNamespace(kind="approval"), pin_recent_tools=False
        )
        assert _context_is_resume(ctx) is False


class TestToolsArrayStability:
    def test_ordinary_turns_send_a_byte_identical_tools_array(self, monkeypatch):
        """Two turns whose previous tool calls differ must build the same array."""
        monkeypatch.delenv("MIIFLOW_PIN_RECENT_TOOLS", raising=False)
        executor = _anthropic_executor()

        with tool_search_session(
            initial=_recent_tools_to_pin(_history_calling("tool_3"), is_resume=False)
        ):
            turn_a = executor._build_native_tool_schemas()
        with tool_search_session(
            initial=_recent_tools_to_pin(_history_calling("tool_7"), is_resume=False)
        ):
            turn_b = executor._build_native_tool_schemas()

        assert turn_a == turn_b
        # Both non-core tools are deferred on both turns — nothing was pinned.
        assert _by_name(turn_a)["tool_3"].get("defer_loading") is True
        assert _by_name(turn_b)["tool_7"].get("defer_loading") is True

    def test_resumed_turn_still_keeps_the_last_tool_resident(self, monkeypatch):
        monkeypatch.delenv("MIIFLOW_PIN_RECENT_TOOLS", raising=False)
        executor = _anthropic_executor()
        with tool_search_session(
            initial=_recent_tools_to_pin(_history_calling("tool_7"), is_resume=True)
        ):
            schemas = executor._build_native_tool_schemas()
        by_name = _by_name(schemas)
        assert not by_name["tool_7"].get("defer_loading")
        assert by_name["tool_3"].get("defer_loading") is True
