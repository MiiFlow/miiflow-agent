"""A tool that declines is unsuccessful, and its answer is still what the model reads.

Production (2026-11-15, Microsoft Ads Specialist): an approved
`microsoft_ads_mutate` came back `{"error": "Edit 1: each negative keyword
needs ...", "error_type": "invalid_document"}`. The host's result adapter
calls that a failure, but `execute_safe_with_context` -- the path every
context-taking tool runs through -- never applied the adapter, so the call
was recorded as a success: the request list then said "1 change made since
this request started is kept" although nothing had changed.

A refusal (`ToolFailure.refusal`) is unsuccessful for every record of what
was done (ledger, observation, events) but it is the tool's answer, not a
crash: the model reads the same payload, and no step error enters the
recovery ladder.
"""

from unittest.mock import MagicMock

import pytest

from miiflow_agent import Agent, RunContext
from miiflow_agent.core.checkpoint import Checkpoint
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.factory import ReActFactory
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.tools import FunctionTool, ToolFailure, ToolRegistry

REFUSED = {"error": "Edit 1: needs a match type", "error_type": "invalid_document"}


def _adapter(output):
    if isinstance(output, dict) and output.get("error"):
        return ToolFailure(error=output["error"], output=output, refusal=True)
    return output


@pytest.mark.asyncio
async def test_a_tool_without_context_answers_through_the_same_rule():
    """`FunctionTool.acall` and the context path shape a result one way."""

    async def add_without_context(text: str) -> dict:
        return REFUSED

    result = await FunctionTool(add_without_context, result_adapter=_adapter).acall(text="bad")

    assert result.is_refusal is True
    assert result.output == REFUSED


async def add_keywords(ctx, text: str) -> dict:
    """Add a keyword."""
    return REFUSED if text == "bad" else {"added": [text]}


def _registry():
    registry = ToolRegistry(tool_search_enabled=False)
    registry.register(FunctionTool(add_keywords, result_adapter=_adapter))
    return registry


def _orchestrator(registry):
    agent = MagicMock(spec=Agent)
    agent.client = MagicMock(provider_name="anthropic")
    agent.tool_registry = registry
    agent._tools = []
    return ReActFactory.create_orchestrator(
        agent=agent, max_steps=5, max_budget=None, max_time_seconds=None
    )


@pytest.mark.asyncio
async def test_a_context_tool_answers_through_its_result_adapter():
    registry = _registry()

    refused = await registry.execute_safe_with_context("add_keywords", "ctx", text="bad")
    added = await registry.execute_safe_with_context("add_keywords", "ctx", text="free")

    assert refused.success is False
    assert refused.is_refusal is True
    assert refused.output == REFUSED
    assert refused.error == "Edit 1: needs a match type"
    assert registry.get_stats()["add_keywords"]["failures"] == 1
    assert added.success is True and added.is_refusal is False


def test_an_explicit_failure_is_not_a_refusal_unless_it_says_so():
    from miiflow_agent.core.tools import ToolResult

    crashed = ToolResult(name="t", input={}).apply_failure(ToolFailure(error="boom"))

    assert crashed.success is False
    assert crashed.is_refusal is False


@pytest.mark.asyncio
async def test_a_refused_call_is_recorded_unsuccessful_but_read_as_its_answer():
    orch = _orchestrator(_registry())
    checkpoint = Checkpoint()
    context = RunContext(deps={}, messages=[], checkpoint=checkpoint)
    step = ReActStep(step_number=1, thought="", action="add_keywords", action_input={"text": "bad"})

    await orch._handle_tool_action(step, context, ExecutionState(current_step=1), tool_call_id="call_1")

    assert step.error is None  # not a crash: no recovery ladder
    assert "Edit 1: needs a match type" in step.observation
    assert "invalid_document" in step.observation  # the whole payload, as before
    assert "Tool execution failed" not in step.observation
    [entry] = [e for e in checkpoint.dispatch_ledger if e.tool_name == "add_keywords"]
    assert entry.success is False
    [invocation] = step.all_invocations
    assert invocation.is_success is False


@pytest.mark.asyncio
async def test_a_refused_call_in_a_batch_is_recorded_unsuccessful_but_read_as_its_answer():
    orch = _orchestrator(_registry())
    checkpoint = Checkpoint()
    context = RunContext(deps={}, messages=[], checkpoint=checkpoint)
    calls = {
        0: {"id": "call_0", "function": {"name": "add_keywords", "arguments": {"text": "bad"}}},
        1: {"id": "call_1", "function": {"name": "add_keywords", "arguments": {"text": "free"}}},
    }
    step = ReActStep(step_number=1, thought="")

    await orch._handle_parallel_tool_batch(step, context, ExecutionState(current_step=1), calls, "")

    refused, added = step.tool_invocations
    assert refused.error is None
    assert refused.is_success is False and added.is_success is True
    assert "Edit 1: needs a match type" in refused.observation
    assert "invalid_document" in refused.observation
    assert "Tool execution failed" not in refused.observation
    successes = {e.inputs_hash: e.success for e in checkpoint.dispatch_ledger if e.tool_name == "add_keywords"}
    assert sorted(successes.values()) == [False, True]
    assert added.error is None


PARTIAL = {"status": "failed", "error": "stopped at the ad group", "created": {"campaign": "c/3"}}


def _partial_adapter(output):
    if isinstance(output, dict) and output.get("error"):
        return ToolFailure(
            error=output["error"], output=output, refusal=True, partial=bool(output.get("created"))
        )
    return output


@pytest.mark.asyncio
async def test_a_refusal_that_changed_something_is_recorded_as_partial():
    async def create_campaign(ctx, name: str) -> dict:
        """Create."""
        return PARTIAL

    registry = ToolRegistry(tool_search_enabled=False)
    registry.register(FunctionTool(create_campaign, result_adapter=_partial_adapter))
    orch = _orchestrator(registry)
    checkpoint = Checkpoint()
    context = RunContext(deps={}, messages=[], checkpoint=checkpoint)
    step = ReActStep(step_number=1, thought="", action="create_campaign", action_input={"name": "x"})

    await orch._handle_tool_action(step, context, ExecutionState(current_step=1), tool_call_id="c1")

    [entry] = [e for e in checkpoint.dispatch_ledger if e.tool_name == "create_campaign"]
    assert entry.success is False and entry.partial is True
    assert "c/3" in step.observation
    restored = type(entry).from_dict(entry.to_dict())
    assert restored.partial is True


@pytest.mark.asyncio
async def test_a_refusal_that_changed_nothing_is_not_partial():
    orch = _orchestrator(_registry())
    checkpoint = Checkpoint()
    context = RunContext(deps={}, messages=[], checkpoint=checkpoint)
    step = ReActStep(step_number=1, thought="", action="add_keywords", action_input={"text": "bad"})

    await orch._handle_tool_action(step, context, ExecutionState(current_step=1), tool_call_id="c1")

    [entry] = [e for e in checkpoint.dispatch_ledger if e.tool_name == "add_keywords"]
    assert entry.partial is False


def test_a_ledger_entry_stored_before_partial_reads_as_not_partial():
    from miiflow_agent.core.checkpoint import DispatchLedgerEntry

    old = {"kind": "tool_call", "success": False, "tool_name": "t", "inputs_hash": "h"}
    assert DispatchLedgerEntry.from_dict(old).partial is False


@pytest.mark.asyncio
async def test_the_plain_agent_loop_shows_a_refusal_its_own_answer():
    """`Agent._execute_tool_calls` (no ReAct) put only the error message in
    the tool message, so the model lost what the tool said; one rule now."""
    from types import SimpleNamespace

    context = RunContext(deps={}, messages=[])
    call = {"id": "c1", "function": {"name": "add_keywords", "arguments": {"text": "bad"}}}

    await Agent._execute_tool_calls(SimpleNamespace(tool_registry=_registry()), [call], context)

    [message] = context.messages
    assert "invalid_document" in str(message.content)
