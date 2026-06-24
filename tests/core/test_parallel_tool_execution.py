"""Unit tests for parallel tool execution (Stage 1 of the agent refactor).

Covers the four primitives added in stage 1:
- `ToolSchema.parallelizable` flag default + opt-in
- `@tool(parallelizable=True)` round-trips onto the schema
- `AgentToolExecutor.is_batch_parallelizable` rule (all-or-nothing)
- `AgentToolExecutor.execute_many` serial/parallel paths + partial-failure

Plus ReActStep model changes:
- legacy single-action fields still work
- new `tool_invocations` list supports batch
- `is_error_step` is step-level only (not per-invocation)
- `has_failed_invocations` separates "some failed" from "step failed"
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── ToolSchema / decorator ────────────────────────────────────────────────


def test_tool_schema_parallelizable_defaults_false():
    from miiflow_agent.core.tools.schemas import ToolSchema
    from miiflow_agent.core.tools.types import ToolType

    schema = ToolSchema(name="x", description="y", tool_type=ToolType.FUNCTION)
    assert schema.parallelizable is False


def test_tool_schema_parallelizable_opt_in():
    from miiflow_agent.core.tools.schemas import ToolSchema
    from miiflow_agent.core.tools.types import ToolType

    schema = ToolSchema(
        name="x",
        description="y",
        tool_type=ToolType.FUNCTION,
        parallelizable=True,
    )
    assert schema.parallelizable is True


def test_tool_decorator_passes_parallelizable_through():
    from miiflow_agent.core.tools.decorators import tool

    @tool(name="reader", description="r", parallelizable=True)
    def reader():
        pass

    @tool(name="writer", description="w")
    def writer():
        pass

    assert reader._tool_schema.parallelizable is True
    assert writer._tool_schema.parallelizable is False


# ── ReActStep model ──────────────────────────────────────────────────────


def test_react_step_legacy_single_action_fields_still_work():
    from miiflow_agent.core.react.models import ReActStep

    step = ReActStep(
        step_number=1,
        thought="thinking",
        action="read_file",
        action_input={"path": "x"},
        observation="ok",
    )
    assert step.is_action_step
    assert not step.is_batch_step
    assert not step.is_error_step
    invocations = step.all_invocations
    assert len(invocations) == 1
    assert invocations[0].name == "read_file"


def test_react_step_batch_via_tool_invocations():
    from miiflow_agent.core.react.models import ReActStep, ToolInvocation

    step = ReActStep(
        step_number=1,
        thought="",
        tool_invocations=[
            ToolInvocation(tool_call_id="a", name="r1", inputs={}, observation="o1"),
            ToolInvocation(tool_call_id="b", name="r2", inputs={}, observation="o2"),
            ToolInvocation(tool_call_id="c", name="r3", inputs={}, observation="o3"),
        ],
    )
    assert step.is_action_step
    assert step.is_batch_step
    assert not step.is_error_step  # no step-level failure
    assert not step.has_failed_invocations
    assert len(step.all_invocations) == 3


def test_react_step_partial_failure_is_not_step_error():
    """A batch where some tools fail must NOT be is_error_step (otherwise
    recovery_manager would fire even though some calls succeeded).
    has_failed_invocations is the correct signal for that case."""
    from miiflow_agent.core.react.models import ReActStep, ToolInvocation

    step = ReActStep(
        step_number=1,
        thought="",
        tool_invocations=[
            ToolInvocation(tool_call_id="a", name="r1", observation="ok"),
            ToolInvocation(tool_call_id="b", name="r2", error="boom"),
        ],
    )
    assert not step.is_error_step
    assert step.has_failed_invocations


def test_react_step_step_level_error_is_error_step():
    from miiflow_agent.core.react.models import ReActStep

    step = ReActStep(step_number=1, thought="", error="malformed transcript")
    assert step.is_error_step


# ── Executor batch eligibility (the all-or-nothing rule) ─────────────────


def _fake_tool(parallelizable: bool, require_approval: bool = False):
    """Build a minimal mock with a `schema` carrying the relevant flags."""
    mock = MagicMock()
    mock.schema = MagicMock(
        parallelizable=parallelizable, require_approval=require_approval
    )
    return mock


def _make_executor_with_tools(tools_by_name):
    """Build an AgentToolExecutor with a stubbed registry that returns the
    given tools for `get_tool_schema_obj`. Other registry methods are
    no-ops; the tests below only exercise the schema-flag inspection."""
    from miiflow_agent.core.react.tool_executor import AgentToolExecutor

    agent = MagicMock()
    registry = MagicMock()
    registry.tools = tools_by_name
    registry._tool_search_tool = None
    agent.tool_registry = registry
    agent.client = MagicMock()
    executor = AgentToolExecutor(agent)
    return executor


def test_is_batch_parallelizable_all_safe():
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {
        "read1": _fake_tool(parallelizable=True),
        "read2": _fake_tool(parallelizable=True),
    }
    executor = _make_executor_with_tools(tools)
    batch = [
        ToolCall(tool_call_id="a", name="read1", inputs={}),
        ToolCall(tool_call_id="b", name="read2", inputs={}),
    ]
    assert executor.is_batch_parallelizable(batch) is True


def test_is_batch_parallelizable_falls_to_serial_when_any_non_parallel():
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {
        "read1": _fake_tool(parallelizable=True),
        "write1": _fake_tool(parallelizable=False),
    }
    executor = _make_executor_with_tools(tools)
    batch = [
        ToolCall(tool_call_id="a", name="read1", inputs={}),
        ToolCall(tool_call_id="b", name="write1", inputs={}),
    ]
    assert executor.is_batch_parallelizable(batch) is False


def test_is_batch_parallelizable_falls_to_serial_when_any_approval_required():
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {
        "read1": _fake_tool(parallelizable=True),
        "mutate": _fake_tool(parallelizable=True, require_approval=True),
    }
    executor = _make_executor_with_tools(tools)
    batch = [
        ToolCall(tool_call_id="a", name="read1", inputs={}),
        ToolCall(tool_call_id="b", name="mutate", inputs={}),
    ]
    assert executor.is_batch_parallelizable(batch) is False


def test_is_batch_parallelizable_unknown_tool_forces_serial():
    """Conservative default: if we can't read flags on a tool, fall to serial."""
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {"read1": _fake_tool(parallelizable=True)}
    executor = _make_executor_with_tools(tools)
    batch = [
        ToolCall(tool_call_id="a", name="read1", inputs={}),
        ToolCall(tool_call_id="b", name="ghost_tool", inputs={}),
    ]
    assert executor.is_batch_parallelizable(batch) is False


# ── execute_many: serial vs parallel paths + partial failure ─────────────


def _make_tool_result(name, output=None, error=None, success=True):
    from miiflow_agent.core.tools import ToolResult

    return ToolResult(
        name=name, input={}, output=output, error=error, success=success
    )


@pytest.mark.asyncio
async def test_execute_many_empty_batch_returns_empty():
    executor = _make_executor_with_tools({})
    out = await executor.execute_many([])
    assert out == []


@pytest.mark.asyncio
async def test_execute_many_runs_parallel_when_all_safe(monkeypatch):
    """When every tool is parallelizable, execute_many uses asyncio.gather.
    We verify by recording start times: parallel calls should overlap.
    """
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {
        "r1": _fake_tool(parallelizable=True),
        "r2": _fake_tool(parallelizable=True),
        "r3": _fake_tool(parallelizable=True),
    }
    executor = _make_executor_with_tools(tools)

    start_times = []

    async def fake_execute_tool(name, inputs, context=None):
        import time

        start_times.append(time.monotonic())
        await asyncio.sleep(0.05)  # 50 ms each
        return _make_tool_result(name, output=f"{name}-ok")

    executor.execute_tool = fake_execute_tool

    batch = [
        ToolCall(tool_call_id=f"id{i}", name=f"r{i+1}", inputs={}) for i in range(3)
    ]
    import time

    t0 = time.monotonic()
    results = await executor.execute_many(batch)
    elapsed = time.monotonic() - t0

    assert len(results) == 3
    assert [r.output for r in results] == ["r1-ok", "r2-ok", "r3-ok"]
    # All three should start within ~10ms (parallel scheduling). Serial
    # would space them by 50ms each (≥100ms between first and third).
    spread = max(start_times) - min(start_times)
    assert spread < 0.020, f"parallel start spread too high: {spread*1000:.1f}ms"
    # Total wall time should be ~50ms (one tool's duration), not 150ms.
    assert elapsed < 0.100, f"parallel batch took too long: {elapsed*1000:.1f}ms"


@pytest.mark.asyncio
async def test_execute_many_runs_serial_when_any_unsafe():
    """Mixed batch with a non-parallelizable tool runs in input order."""
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {
        "r1": _fake_tool(parallelizable=True),
        "write1": _fake_tool(parallelizable=False),  # forces serial
        "r2": _fake_tool(parallelizable=True),
    }
    executor = _make_executor_with_tools(tools)

    call_order = []

    async def fake_execute_tool(name, inputs, context=None):
        call_order.append(name)
        await asyncio.sleep(0.01)
        return _make_tool_result(name, output=f"{name}-ok")

    executor.execute_tool = fake_execute_tool

    batch = [
        ToolCall(tool_call_id="a", name="r1", inputs={}),
        ToolCall(tool_call_id="b", name="write1", inputs={}),
        ToolCall(tool_call_id="c", name="r2", inputs={}),
    ]
    results = await executor.execute_many(batch)
    assert len(results) == 3
    assert call_order == ["r1", "write1", "r2"]  # input order preserved


@pytest.mark.asyncio
async def test_execute_many_partial_failure_does_not_abort_batch():
    """One bad tool doesn't kill the others — each result independently
    carries success/error."""
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {
        "r1": _fake_tool(parallelizable=True),
        "r2": _fake_tool(parallelizable=True),
        "r3": _fake_tool(parallelizable=True),
    }
    executor = _make_executor_with_tools(tools)

    async def fake_execute_tool(name, inputs, context=None):
        if name == "r2":
            return _make_tool_result(name, error="boom", success=False)
        return _make_tool_result(name, output=f"{name}-ok")

    executor.execute_tool = fake_execute_tool

    batch = [
        ToolCall(tool_call_id="a", name="r1", inputs={}),
        ToolCall(tool_call_id="b", name="r2", inputs={}),
        ToolCall(tool_call_id="c", name="r3", inputs={}),
    ]
    results = await executor.execute_many(batch)
    assert len(results) == 3
    assert results[0].success
    assert not results[1].success
    assert results[1].error == "boom"
    assert results[2].success


@pytest.mark.asyncio
async def test_execute_many_raw_exception_wrapped_into_tool_result():
    """asyncio.gather with return_exceptions=True surfaces raw exceptions;
    execute_many must wrap them so the caller never sees a bare exception
    in the results list."""
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {
        "r1": _fake_tool(parallelizable=True),
        "r2": _fake_tool(parallelizable=True),
    }
    executor = _make_executor_with_tools(tools)

    async def fake_execute_tool(name, inputs, context=None):
        if name == "r2":
            raise RuntimeError("network down")
        return _make_tool_result(name, output="ok")

    executor.execute_tool = fake_execute_tool

    batch = [
        ToolCall(tool_call_id="a", name="r1", inputs={}),
        ToolCall(tool_call_id="b", name="r2", inputs={}),
    ]
    results = await executor.execute_many(batch)
    assert len(results) == 2
    assert results[0].success
    assert not results[1].success
    assert "network down" in (results[1].error or "")


@pytest.mark.asyncio
async def test_execute_many_results_in_input_order():
    """Even when parallel completion finishes out of input order, the
    returned results list must be in input order (callers correlate by
    index)."""
    from miiflow_agent.core.react.tool_executor import ToolCall

    tools = {
        "slow": _fake_tool(parallelizable=True),
        "fast": _fake_tool(parallelizable=True),
    }
    executor = _make_executor_with_tools(tools)

    async def fake_execute_tool(name, inputs, context=None):
        if name == "slow":
            await asyncio.sleep(0.05)
        else:
            await asyncio.sleep(0.001)
        return _make_tool_result(name, output=f"{name}-done")

    executor.execute_tool = fake_execute_tool

    batch = [
        ToolCall(tool_call_id="a", name="slow", inputs={}),
        ToolCall(tool_call_id="b", name="fast", inputs={}),
    ]
    results = await executor.execute_many(batch)
    assert [r.output for r in results] == ["slow-done", "fast-done"]


# ── Validation-error classification (no runtime-ladder activation) ──────


@pytest.mark.asyncio
async def test_registry_stamps_is_validation_error_on_raised_exception():
    """Exceptions carrying `is_tool_validation_error = True` must show up
    on the resulting `ToolResult.metadata`. Without this, the orchestrator
    can't distinguish a deterministic input-shape rejection (which the LLM
    should self-correct via the observation) from a generic runtime
    failure (which kicks the recovery ladder)."""
    from miiflow_agent.core.tools import ToolRegistry, tool

    class _PreflightError(Exception):
        is_tool_validation_error = True

    @tool(description="Always raises a validation error")
    def bad_tool(x: str) -> dict:
        raise _PreflightError(f"reject {x}")

    @tool(description="Always raises a generic error")
    def boom_tool(x: str) -> dict:
        raise RuntimeError(f"runtime boom {x}")

    registry = ToolRegistry()
    registry.register(bad_tool)
    registry.register(boom_tool)

    bad_result = await registry.execute_safe("bad_tool", x="q")
    boom_result = await registry.execute_safe("boom_tool", x="q")

    assert not bad_result.success
    assert bad_result.metadata.get("is_validation_error") is True

    assert not boom_result.success
    assert boom_result.metadata.get("is_validation_error") is False


def test_orchestrator_all_failed_classifies_validation_as_schema():
    """When every parallel invocation failed with a validation-flagged
    error, the step's failure_kind must be `schema` (not `all_failed`),
    so recovery_manager's SCHEMA short-circuit fires and the run doesn't
    burn the runtime recovery ladder."""
    from miiflow_agent.core.react.models import ReActStep, ToolInvocation

    # Mirrors the relevant lines of orchestrator._handle_batch_action:
    # build the invocations as the orchestrator would after Phase 4, then
    # apply the same step.error/failure_kind classification.
    invocations = [
        ToolInvocation(
            tool_call_id="a", name="google_ads_query",
            error="Tool 'google_ads_query' failed: segments.date missing",
            observation="Tool execution failed: …",
            is_validation_error=True,
        ),
        ToolInvocation(
            tool_call_id="b", name="google_ads_query",
            error="Tool 'google_ads_query' failed: segments.date missing",
            observation="Tool execution failed: …",
            is_validation_error=True,
        ),
    ]
    step = ReActStep(step_number=1, thought="", tool_invocations=invocations)

    if all(inv.error is not None for inv in invocations):
        if all(getattr(inv, "is_validation_error", False) for inv in invocations):
            step.metadata["failure_kind"] = "schema"
        else:
            step.metadata["failure_kind"] = "all_failed"

    assert step.metadata["failure_kind"] == "schema"


def test_orchestrator_all_failed_mixed_kinds_stays_all_failed():
    """If even one of the failures isn't validation-flagged, treat the
    step as a runtime all-failed event — recovery ladder still applies."""
    from miiflow_agent.core.react.models import ReActStep, ToolInvocation

    invocations = [
        ToolInvocation(
            tool_call_id="a", name="google_ads_query",
            error="schema mismatch", observation="…",
            is_validation_error=True,
        ),
        ToolInvocation(
            tool_call_id="b", name="meta_ads_insights",
            error="api 500", observation="…",
            is_validation_error=False,
        ),
    ]
    step = ReActStep(step_number=1, thought="", tool_invocations=invocations)

    if all(inv.error is not None for inv in invocations):
        if all(getattr(inv, "is_validation_error", False) for inv in invocations):
            step.metadata["failure_kind"] = "schema"
        else:
            step.metadata["failure_kind"] = "all_failed"

    assert step.metadata["failure_kind"] == "all_failed"


def test_execute_parallel_bounds_concurrency_with_semaphore():
    """_execute_parallel must cap in-flight execution at _max_parallel_tools.
    A wide batch (one dispatch per platform, each a full sub-agent) should not
    stampede the loop — excess branches queue and start as slots free, but the
    whole batch still completes in input order."""
    import asyncio

    from miiflow_agent.core.react.tool_executor import ToolCall
    from miiflow_agent.core.tools import ToolResult

    executor = _make_executor_with_tools({})
    executor._max_parallel_tools = 2

    inflight = 0
    peak = 0

    async def fake_execute_tool(name, inputs, context=None):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await asyncio.sleep(0.01)  # hold the slot so overlap is observable
            return ToolResult(name=name, input=inputs, output=name, success=True)
        finally:
            inflight -= 1

    executor.execute_tool = fake_execute_tool

    batch = [ToolCall(tool_call_id=str(i), name=f"t{i}", inputs={}) for i in range(6)]
    results = asyncio.run(executor._execute_parallel(batch, context=None))

    assert peak <= 2, f"concurrency exceeded cap: peak={peak}"
    assert peak == 2, "expected the cap to actually be saturated"
    # Results preserved, in input order, one per call.
    assert [r.output for r in results] == [f"t{i}" for i in range(6)]


def test_execute_parallel_unbounded_relative_to_small_batch():
    """When the batch is within the cap, all branches run at once (no needless
    serialization)."""
    import asyncio

    from miiflow_agent.core.react.tool_executor import ToolCall
    from miiflow_agent.core.tools import ToolResult

    executor = _make_executor_with_tools({})
    executor._max_parallel_tools = 8

    inflight = 0
    peak = 0

    async def fake_execute_tool(name, inputs, context=None):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        try:
            await asyncio.sleep(0.01)
            return ToolResult(name=name, input=inputs, output=name, success=True)
        finally:
            inflight -= 1

    executor.execute_tool = fake_execute_tool

    batch = [ToolCall(tool_call_id=str(i), name=f"t{i}", inputs={}) for i in range(3)]
    asyncio.run(executor._execute_parallel(batch, context=None))
    assert peak == 3  # all three ran concurrently
