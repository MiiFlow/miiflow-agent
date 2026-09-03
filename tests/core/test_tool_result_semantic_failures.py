"""Regression coverage for explicit tool outcomes.

The SDK must not infer execution semantics from application payload keys.
Hosts with legacy output conventions can opt into a result adapter, while new
tools and callbacks return ``ToolFailure`` when the invocation itself failed.
"""

import pytest

from miiflow_agent.core.callbacks import CallbackEvent, CallbackEventType
from miiflow_agent.core.react.tool_executor import AgentToolExecutor
from miiflow_agent.core.tools import FunctionTool, ToolFailure, ToolResult


@pytest.mark.parametrize(
    "output",
    [
        {"job_id": "j1", "status": "failed", "message": "The job failed"},
        {"record_id": "r1", "error_type": "validation", "message": "A category"},
        {"success": False, "name": "stored result"},
        {"ok": False, "status": "rejected"},
        {"error": "A historical error attached to this record"},
    ],
)
def test_application_payloads_are_opaque(output):
    result = ToolResult(name="read_record", input={}, output=output, success=True)

    assert result.success is True
    assert result.error is None
    assert result.output == output
    assert result.metadata == {}


@pytest.mark.asyncio
async def test_function_tool_converts_explicit_failure_to_result():
    payload = {
        "status": "rejected",
        "error_kind": "policy_rejected",
        "error": "Policy rejected the result",
    }

    async def rejected():
        return ToolFailure(
            error="Policy rejected the result",
            output=payload,
            error_type="policy_rejected",
            metadata={"policy": "workspace"},
        )

    result = await FunctionTool(rejected).acall()

    assert result.success is False
    assert result.error == "Policy rejected the result"
    assert result.output == payload
    assert result.metadata["error_type"] == "policy_rejected"
    assert result.metadata["policy"] == "workspace"


@pytest.mark.asyncio
async def test_host_can_opt_into_a_legacy_result_adapter():
    payload = {"error": "Not found", "error_type": "not_found"}

    async def legacy_read():
        return payload

    def adapter(output):
        if isinstance(output, dict) and output.get("error"):
            return ToolFailure(
                error=output["error"],
                output=output,
                error_type=output.get("error_type"),
            )
        return output

    result = await FunctionTool(legacy_read, result_adapter=adapter).acall()

    assert result.success is False
    assert result.output == payload
    assert result.metadata["error_type"] == "not_found"


@pytest.mark.asyncio
async def test_post_tool_failure_is_finalized_before_tool_executed(monkeypatch):
    """Bookkeeping callbacks must see the outcome returned to the caller."""

    class Registry:
        async def execute_safe(self, _tool_name, **_kwargs):
            return ToolResult(name="read", input={}, output={"data": []})

    class Agent:
        tool_registry = Registry()
        client = None

    class CallbackRegistry:
        def __init__(self):
            self.events = []

        async def emit(self, event: CallbackEvent):
            if event.event_type == CallbackEventType.POST_TOOL_USE:
                event.output_transformed = True
                event.transformed_output = ToolFailure(
                    error="Policy rejected the result",
                    output={
                        "status": "rejected",
                        "error_kind": "policy_rejected",
                    },
                    error_type="policy_rejected",
                )
            self.events.append((event.event_type, event.success, event.tool_output))

    callbacks = CallbackRegistry()
    monkeypatch.setattr(
        "miiflow_agent.core.react.tool_executor.get_active_registry",
        lambda: callbacks,
    )

    executor = AgentToolExecutor(Agent())
    executor._emit_post_tool_use = True
    result = await executor.execute_tool("read", {})

    assert result.success is False
    assert result.error == "Policy rejected the result"
    assert result.metadata["error_type"] == "policy_rejected"
    assert [event[0] for event in callbacks.events] == [
        CallbackEventType.PRE_TOOL_USE,
        CallbackEventType.POST_TOOL_USE,
        CallbackEventType.TOOL_EXECUTED,
    ]
    executed = callbacks.events[-1]
    assert executed[1] is False
    assert executed[2] == {
        "status": "rejected",
        "error_kind": "policy_rejected",
    }
