"""A child's clarification from inside a parallel batch pauses the parent.

The dispatch layer used to publish a child's question straight up the parent
bus, before the batch returned and without any interrupt recorded -- the only
way a question from a batched dispatch reached the person, and one that could
be answered before a pause existed. The dispatch layer no longer publishes it
(thread_Kvralz4QBB0RlP53CLJ8Mae2), so the batch path must pause on the marker
the way the single-call path always has.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from miiflow_agent import Agent, RunContext
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.factory import ReActFactory
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.tools.clarification import child_clarification_observation
from miiflow_agent.core.tools.schemas import ToolResult


@pytest.mark.asyncio
async def test_a_batched_child_question_pauses_once_with_an_interrupt():
    agent = MagicMock(spec=Agent)
    agent.client = MagicMock(provider_name="anthropic")
    agent.tool_registry = MagicMock()
    agent.tool_registry.list_tools.return_value = ["dispatch_assistant"]
    agent._tools = []
    orch = ReActFactory.create_orchestrator(agent=agent, max_steps=5, max_budget=None, max_time_seconds=None)
    question = child_clarification_observation(
        [{"question": "Which geo?", "options": ["US", "EU"], "key": "geo"}],
        "needed to size the budget",
        {"handle": "google_ads_specialist", "subagent_path": ["sub_1"]},
    )
    orch.tool_executor.execute_many = AsyncMock(return_value=[
        ToolResult(name="dispatch_assistant", input={}, output=question),
        ToolResult(name="dispatch_assistant", input={}, output="report: all good"),
    ])
    orch.tool_executor.has_tool = MagicMock(return_value=True)
    orch.tool_executor.get_tool_schema = MagicMock(return_value={})
    events = []
    orch.event_bus.subscribe(lambda e: events.append(e))
    ctx, state = RunContext(deps={}, messages=[]), ExecutionState(current_step=1)
    calls = {
        i: {"id": f"call_{i}", "function": {"name": "dispatch_assistant", "arguments": {}}}
        for i in range(2)
    }

    await orch._handle_parallel_tool_batch(ReActStep(step_number=1, thought=""), ctx, state, calls, "")

    asked = [e for e in events if e.event_type == ReActEventType.CLARIFICATION_NEEDED]
    assert state.needs_clarification is True
    assert len(asked) == 1
    assert asked[0].data["questions"][0]["question"] == "Which geo?"
    assert asked[0].data["interrupt_id"]  # recorded, so an answer has something to resume
    assert state.clarification_data["tool_call_id"] == "call_0"
