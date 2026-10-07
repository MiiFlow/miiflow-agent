"""A stopped run abandons the tool calls it has in flight.

Production thread_Kvralz4QBB0RlP53CLJ8Mae2: a run superseded by the person's
next message kept a specialist dispatch going for minutes (the loop checked
`cancel_event` only between steps), and that dispatch then raised a second
approval card for a change already queued. Drives the REAL
`AgentToolExecutor.execute_tool`, with the registry call standing in for the
slow tool.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import ToolRegistry, ToolResult


def _executor(slow_tool):
    client = MagicMock()
    client.tool_registry = ToolRegistry()
    agent = MagicMock()
    agent.client = client
    agent.tool_registry = client.tool_registry
    # Unregistered names take the context-free path; stub both legs.
    agent.tool_registry.execute_safe_with_context = slow_tool
    agent.tool_registry.execute_safe = slow_tool
    return AgentToolExecutor(agent)


@pytest.mark.asyncio
async def test_stop_cancels_a_call_in_flight_and_reports_it_failed():
    started, finished = asyncio.Event(), []

    async def slow_tool(*args, **kwargs):
        started.set()
        await asyncio.sleep(30)
        finished.append(True)
        return ToolResult(name="dispatch_assistant", input={}, output="late", success=True)

    context = SimpleNamespace(cancel_event=asyncio.Event(), deps={})
    call = asyncio.create_task(
        _executor(slow_tool).execute_tool("dispatch_assistant", {}, context=context)
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    context.cancel_event.set()
    result = await asyncio.wait_for(call, timeout=2)

    assert result.success is False
    assert "stopped" in result.error
    assert finished == []  # the tool body never ran to completion


@pytest.mark.asyncio
async def test_a_call_after_stop_never_starts():
    ran = []

    async def tool(*args, **kwargs):
        ran.append(True)
        return ToolResult(name="t", input={}, output="x", success=True)

    context = SimpleNamespace(cancel_event=asyncio.Event(), deps={})
    context.cancel_event.set()
    result = await _executor(tool).execute_tool("t", {}, context=context)

    assert result.success is False and ran == []


@pytest.mark.asyncio
async def test_a_call_that_finishes_is_unaffected_and_leaves_no_cancel_behind():
    async def tool(*args, **kwargs):
        return ToolResult(name="t", input={}, output="ok", success=True)

    context = SimpleNamespace(cancel_event=asyncio.Event(), deps={})
    result = await _executor(tool).execute_tool("t", {}, context=context)
    # A later stop must not cancel the caller after the call returned.
    context.cancel_event.set()
    await asyncio.sleep(0.01)

    assert result.success is True and result.output == "ok"
    assert asyncio.current_task().cancelling() == 0


@pytest.mark.asyncio
async def test_the_hosts_own_cancel_still_propagates():
    """Only the stop's cancel is absorbed; a shutdown drain must still unwind."""
    started = asyncio.Event()

    async def slow_tool(*args, **kwargs):
        started.set()
        await asyncio.sleep(30)

    context = SimpleNamespace(cancel_event=asyncio.Event(), deps={})
    call = asyncio.create_task(_executor(slow_tool).execute_tool("t", {}, context=context))
    await asyncio.wait_for(started.wait(), timeout=2)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_parallel_siblings_are_each_stopped():
    """The parallel batch runs each call in its own task; every one stops."""
    started = []

    async def slow_tool(*args, **kwargs):
        started.append(True)
        await asyncio.sleep(30)

    executor = _executor(slow_tool)
    context = SimpleNamespace(cancel_event=asyncio.Event(), deps={})
    batch = asyncio.gather(
        executor.execute_tool("a", {}, context=context),
        executor.execute_tool("b", {}, context=context),
    )
    for _ in range(200):
        if len(started) == 2:
            break
        await asyncio.sleep(0.01)
    context.cancel_event.set()
    results = await asyncio.wait_for(batch, timeout=2)

    assert [r.success for r in results] == [False, False]


@pytest.mark.asyncio
async def test_a_write_in_flight_is_let_finish_and_recorded_truthfully():
    """Abandoning the await would not stop a write already sent."""
    from miiflow_agent.core.react.tool_executor import _unless_run_stopped

    started, finished = asyncio.Event(), []

    async def write():
        started.set()
        await asyncio.sleep(0.05)
        finished.append(True)
        return ToolResult(name="google_ads_mutate", input={}, output="paused", success=True)

    context = SimpleNamespace(cancel_event=asyncio.Event())
    call = asyncio.create_task(_unless_run_stopped(write(), "google_ads_mutate", {}, context, writes=True))
    await asyncio.wait_for(started.wait(), timeout=2)
    context.cancel_event.set()
    result = await asyncio.wait_for(call, timeout=2)

    assert result.success is True and finished == [True]


@pytest.mark.asyncio
async def test_a_host_cancel_in_the_same_tick_as_the_stop_is_not_swallowed():
    from miiflow_agent.core.react.tool_executor import _unless_run_stopped

    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(30)

    context = SimpleNamespace(cancel_event=asyncio.Event())
    call = asyncio.create_task(_unless_run_stopped(slow(), "t", {}, context))
    await asyncio.wait_for(started.wait(), timeout=2)
    context.cancel_event.set()
    call.cancel()  # a shutdown drain, same tick
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_a_tool_that_turns_our_cancel_into_an_error_leaves_no_cancel_behind():
    from miiflow_agent.core.react.tool_executor import _unless_run_stopped

    started = asyncio.Event()

    async def converts():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise RuntimeError("client aborted")

    async def run():
        context = SimpleNamespace(cancel_event=asyncio.Event())
        call = _unless_run_stopped(converts(), "t", {}, context)
        waiter = asyncio.ensure_future(started.wait())
        waiter.add_done_callback(lambda _: context.cancel_event.set())
        with pytest.raises(RuntimeError):
            await call
        await asyncio.sleep(0)  # a leaked cancel would fire here
        # Task.cancelling() is 3.11+; a 3.10 task has no counter to leak into,
        # so the sleep above is the whole check there (same guard as the SDK).
        return getattr(asyncio.current_task(), "cancelling", lambda: 0)()

    assert await asyncio.wait_for(asyncio.create_task(run()), timeout=2) == 0
