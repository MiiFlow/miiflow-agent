"""Tests for read-aware staged scheduling in execute_many.

The all-or-nothing rule serialized the dominant real batch shape — N reads
plus one writer — whenever a single member wasn't parallelizable. Mixed
batches now run as ordered stages: consecutive gather-safe calls
(parallelizable or read-only) overlap; writers, approval-required tools, and
control-flow tools run serially at their original positions.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from miiflow_agent.core.react.tool_executor import AgentToolExecutor, ToolCall
from miiflow_agent.core.react.exceptions import (
    PlanApprovalRequired,
    ToolApprovalRequired,
)
from miiflow_agent.core.tools import ToolResult


def _fake_tool(parallelizable=False, require_approval=False, is_read_only=False):
    mock = MagicMock()
    mock.schema = MagicMock(
        parallelizable=parallelizable,
        require_approval=require_approval,
        is_read_only=is_read_only,
    )
    return mock


def _executor(tools_by_name):
    agent = MagicMock()
    registry = MagicMock()
    registry.tools = tools_by_name
    registry._tool_search_tool = None
    agent.tool_registry = registry
    agent.client = MagicMock()
    return AgentToolExecutor(agent)


def _result(name):
    return ToolResult(name=name, input={}, output=f"{name}-ok", success=True)


def _tracking_execute(events, delay=0.03):
    async def fake_execute_tool(name, inputs, context=None):
        events.append(("start", name, time.monotonic()))
        await asyncio.sleep(delay)
        events.append(("end", name, time.monotonic()))
        return _result(name)

    return fake_execute_tool


async def test_reads_overlap_around_a_writer():
    """[read, read, write, read]: the two leading reads overlap; the writer
    runs after both finish and before the trailing read starts."""
    executor = _executor(
        {
            "read_a": _fake_tool(is_read_only=True),
            "read_b": _fake_tool(is_read_only=True),
            "write_c": _fake_tool(),
            "read_d": _fake_tool(is_read_only=True),
        }
    )
    executor._readonly_parallel = True  # explicit opt-in (default off)
    events = []
    executor.execute_tool = _tracking_execute(events)

    batch = [
        ToolCall(tool_call_id="1", name="read_a", inputs={}),
        ToolCall(tool_call_id="2", name="read_b", inputs={}),
        ToolCall(tool_call_id="3", name="write_c", inputs={}),
        ToolCall(tool_call_id="4", name="read_d", inputs={}),
    ]
    results = await executor.execute_many(batch)

    assert [r.output for r in results] == [
        "read_a-ok", "read_b-ok", "write_c-ok", "read_d-ok",
    ]

    starts = {n: t for kind, n, t in events if kind == "start"}
    ends = {n: t for kind, n, t in events if kind == "end"}
    # The two reads overlapped: b started before a ended.
    assert starts["read_b"] < ends["read_a"]
    # The writer waited for both reads and ran alone.
    assert starts["write_c"] >= max(ends["read_a"], ends["read_b"])
    assert starts["read_d"] >= ends["write_c"]


async def test_all_writer_batch_stays_fully_serial():
    executor = _executor(
        {"w1": _fake_tool(), "w2": _fake_tool(), "w3": _fake_tool()}
    )
    events = []
    executor.execute_tool = _tracking_execute(events, delay=0.01)

    batch = [
        ToolCall(tool_call_id=str(i), name=f"w{i}", inputs={}) for i in (1, 2, 3)
    ]
    await executor.execute_many(batch)

    order = [n for kind, n, _ in events]
    assert order == ["start", "end", "start", "end", "start", "end"] or [
        (k, n) for k, n, _ in events
    ] == [
        ("start", "w1"), ("end", "w1"),
        ("start", "w2"), ("end", "w2"),
        ("start", "w3"), ("end", "w3"),
    ]


async def test_readonly_overlap_is_off_by_default():
    """is_read_only was declared for the plan-mode gate, not as a
    concurrency contract — overlapping such tools is opt-in
    (MIIFLOW_READONLY_PARALLEL=1)."""
    executor = _executor(
        {
            "read_a": _fake_tool(is_read_only=True),
            "read_b": _fake_tool(is_read_only=True),
            "write_c": _fake_tool(),
        }
    )
    assert executor._readonly_parallel is False
    events = []
    executor.execute_tool = _tracking_execute(events, delay=0.01)

    batch = [
        ToolCall(tool_call_id="1", name="read_a", inputs={}),
        ToolCall(tool_call_id="2", name="read_b", inputs={}),
        ToolCall(tool_call_id="3", name="write_c", inputs={}),
    ]
    await executor.execute_many(batch)

    starts = {n: t for kind, n, t in events if kind == "start"}
    ends = {n: t for kind, n, t in events if kind == "end"}
    assert starts["read_b"] >= ends["read_a"]


async def test_approval_required_read_stays_serial_and_raises():
    executor = _executor(
        {
            "read_a": _fake_tool(is_read_only=True),
            "gated": _fake_tool(is_read_only=True, require_approval=True),
        }
    )

    async def fake_execute_tool(name, inputs, context=None):
        if name == "gated":
            raise ToolApprovalRequired(
                tool_name="gated", tool_inputs=inputs, reason="approve?"
            )
        return _result(name)

    executor.execute_tool = fake_execute_tool
    batch = [
        ToolCall(tool_call_id="1", name="read_a", inputs={}),
        ToolCall(tool_call_id="2", name="gated", inputs={}),
    ]
    with pytest.raises(ToolApprovalRequired):
        await executor.execute_many(batch)


async def test_plan_mode_tools_never_gather():
    """exit_plan_mode is is_read_only=True but raises PlanApprovalRequired —
    it must stay off the parallel path so the pause propagates."""
    executor = _executor(
        {
            "read_a": _fake_tool(is_read_only=True),
            "exit_plan_mode": _fake_tool(is_read_only=True),
        }
    )
    executor._readonly_parallel = True
    assert executor._is_gather_safe(
        ToolCall(tool_call_id="1", name="read_a", inputs={})
    )
    assert not executor._is_gather_safe(
        ToolCall(tool_call_id="2", name="exit_plan_mode", inputs={})
    )


async def test_unknown_tool_is_not_gather_safe():
    executor = _executor({"read_a": _fake_tool(is_read_only=True)})
    assert not executor._is_gather_safe(
        ToolCall(tool_call_id="1", name="ghost", inputs={})
    )


async def test_parallel_path_reraises_control_flow_exceptions():
    """Backstop: a control-flow exception inside a gather must escape, not
    flatten into a failed ToolResult."""
    executor = _executor(
        {"a": _fake_tool(parallelizable=True), "b": _fake_tool(parallelizable=True)}
    )

    async def fake_execute_tool(name, inputs, context=None):
        if name == "b":
            raise PlanApprovalRequired(plan_text="the plan")
        return _result(name)

    executor.execute_tool = fake_execute_tool
    batch = [
        ToolCall(tool_call_id="1", name="a", inputs={}),
        ToolCall(tool_call_id="2", name="b", inputs={}),
    ]
    with pytest.raises(PlanApprovalRequired):
        await executor._execute_parallel(batch, context=None)
