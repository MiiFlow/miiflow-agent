"""Two specialists that each pause for approval in one parallel batch.

Both dispatch results are stored, at pause time, as "waiting on the user's
approval of <tool>; do not dispatch again" — the old "Tool execution paused -
waiting for user approval." read as a stuck task, and the resume turn for the
first card re-dispatched the second specialist from scratch
(thread_474iZNmonj4UxDnb1ZugJAmw). Written once so later turns replay the same
bytes (a request must be a byte prefix of the next).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from miiflow_agent import Agent, RunContext
from miiflow_agent.core.message import MessageRole
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.factory import ReActFactory
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.react.orchestrator import waiting_pause_observation
from miiflow_agent.core.tools.clarification import child_tool_approval_observation
from miiflow_agent.core.tools.schemas import ToolResult


def _marker(tool_name, call):
    return child_tool_approval_observation(
        tool_name=tool_name,
        tool_inputs={"status": "PAUSED"},
        tool_call_id=f"child_{call}",
        interrupt_id=f"int_{call}",
        dispatch_meta={"handle": tool_name.split("_")[0] + "_ads_specialist", "subagent_path": [f"sub_{call}"]},
    )


@pytest.mark.asyncio
async def test_each_paused_sibling_stores_a_waiting_result_naming_its_tool():
    agent = MagicMock(spec=Agent)
    agent.client = MagicMock(provider_name="anthropic")
    agent.tool_registry = MagicMock()
    agent.tool_registry.list_tools.return_value = ["dispatch_assistant"]
    agent._tools = []
    orch = ReActFactory.create_orchestrator(agent=agent, max_steps=5, max_budget=None, max_time_seconds=None)
    orch.tool_executor.execute_many = AsyncMock(return_value=[
        ToolResult(name="dispatch_assistant", input={}, output=_marker("google_ads_mutate", "g")),
        ToolResult(name="dispatch_assistant", input={}, output=_marker("meta_ads_mutate", "m")),
    ])
    orch.tool_executor.has_tool = MagicMock(return_value=True)
    orch.tool_executor.get_tool_schema = MagicMock(return_value={})
    ctx, state = RunContext(deps={}, messages=[]), ExecutionState(current_step=1)
    calls = {
        i: {"id": f"call_{name}", "function": {"name": "dispatch_assistant", "arguments": {}}}
        for i, name in enumerate(("google", "meta"))
    }

    await orch._handle_parallel_tool_batch(ReActStep(step_number=1, thought=""), ctx, state, calls, "")

    assert state.needs_clarification is True
    results = {
        m.tool_call_id: m.content for m in ctx.messages if m.role == MessageRole.TOOL
    }
    # Each result may carry its citation ref line first.
    assert results["call_google"].endswith(waiting_pause_observation("google_ads_mutate"))
    assert results["call_meta"].endswith(waiting_pause_observation("meta_ads_mutate"))
    assert "Do not call or dispatch it again" in results["call_meta"]
    assert all("Tool execution paused" not in text for text in results.values())
