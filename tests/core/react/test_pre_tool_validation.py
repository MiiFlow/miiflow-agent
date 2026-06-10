"""PRE_TOOL_USE validation channel (CallbackEvent.validation_error).

A callback that rejects a call's INPUTS must produce a FAILED ToolResult the
model can fix — never a ToolApprovalRequired modal. Validation outranks the
approval block: the user is never asked to approve a call that can't succeed.
"""

import asyncio

import pytest

from miiflow_agent.core.callbacks import (
    CallbackEventType,
    get_global_registry,
)
from miiflow_agent.core.react.exceptions import ToolApprovalRequired
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import ToolRegistry, tool


@tool(name="mutate_thing", description="test mutation", strict=False)
def mutate_thing(value: str = "") -> dict:
    return {"did": value}


@pytest.fixture
def executor():
    registry = ToolRegistry(tool_search_enabled=False)
    registry.register(mutate_thing)

    class _Agent:
        pass

    ex = AgentToolExecutor.__new__(AgentToolExecutor)
    ex._tool_registry = registry
    ex.tool_filter = None
    ex.agent = _Agent()
    return ex


@pytest.fixture(autouse=True)
def _clean_callbacks():
    get_global_registry().clear(CallbackEventType.PRE_TOOL_USE)
    yield
    get_global_registry().clear(CallbackEventType.PRE_TOOL_USE)


def test_validation_error_returns_failed_result_not_modal(executor):
    async def gate(event):
        event.validation_error = "Preflight failed: need 3 headlines."

    get_global_registry().register(CallbackEventType.PRE_TOOL_USE, gate)

    result = asyncio.run(executor.execute_tool("mutate_thing", {"value": "x"}))
    assert result.success is False
    assert "3 headlines" in result.error


def test_validation_outranks_approval_block(executor):
    async def gate(event):
        event.validation_error = "invalid inputs"
        event.blocked = True
        event.block_reason = "Tool requires user approval"

    get_global_registry().register(CallbackEventType.PRE_TOOL_USE, gate)

    # No ToolApprovalRequired escapes — the failed result wins.
    result = asyncio.run(executor.execute_tool("mutate_thing", {"value": "x"}))
    assert result.success is False
    assert "invalid inputs" in result.error


def test_block_without_validation_still_raises_approval(executor):
    async def gate(event):
        event.blocked = True
        event.block_reason = "Tool requires user approval"

    get_global_registry().register(CallbackEventType.PRE_TOOL_USE, gate)

    with pytest.raises(ToolApprovalRequired):
        asyncio.run(executor.execute_tool("mutate_thing", {"value": "x"}))


def test_no_callback_executes_normally(executor):
    result = asyncio.run(executor.execute_tool("mutate_thing", {"value": "ok"}))
    assert result.success is True
