"""An unknown tool name in a batch is answered as unknown, before any gate.

Production (2026-10-14, Microsoft Ads Specialist): a made-up
`load_skill("microsoft_ads")` issued in a batch reached the host's permission
gate, which fails closed on a name with no declared facts, and came back
"Permission denied. Ask a Member or Admin to perform this action." — sending
the model after a role instead of a tool that does not exist. The single-tool
path already answered "not found" first; the batch path now does too.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from miiflow_agent import Agent, RunContext
from miiflow_agent.core.callbacks import CallbackEventType, get_global_registry
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.factory import ReActFactory
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.tools import ToolRegistry, tool


@tool(name="list_campaigns", description="list campaigns", strict=False, writes=False)
def list_campaigns() -> dict:
    return {"campaigns": []}


@pytest.fixture(autouse=True)
def _clean_callbacks():
    get_global_registry().clear(CallbackEventType.PRE_TOOL_USE)
    yield
    get_global_registry().clear(CallbackEventType.PRE_TOOL_USE)


@pytest.mark.asyncio
async def test_an_unknown_name_in_a_batch_never_reaches_the_gate():
    registry = ToolRegistry(tool_search_enabled=False)
    registry.register(list_campaigns)
    agent = MagicMock(spec=Agent)
    agent.client = MagicMock(provider_name="anthropic")
    agent.tool_registry = registry
    agent._tools = []
    orch = ReActFactory.create_orchestrator(
        agent=agent, max_steps=5, max_budget=None, max_time_seconds=None
    )

    gated = []

    async def gate(event):
        gated.append(event.tool_name)
        event.validation_error = "Permission denied. Ask a Member or Admin to perform this action."

    get_global_registry().register(CallbackEventType.PRE_TOOL_USE, gate)

    calls = {
        0: {"id": "call_0", "function": {"name": "load_skill", "arguments": {"name": "microsoft_ads"}}},
        1: {"id": "call_1", "function": {"name": "list_campaigns", "arguments": {}}},
    }
    step = ReActStep(step_number=1, thought="")
    await orch._handle_parallel_tool_batch(
        step, RunContext(deps={}, messages=[]), ExecutionState(current_step=1), calls, ""
    )

    unknown, known = step.tool_invocations
    assert "not found" in unknown.error
    assert "Permission denied" not in unknown.error
    # The real tool still went through the gate.
    assert gated == ["list_campaigns"]
