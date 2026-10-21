"""A sub-agent's approved write is a step in its task, not the end of it.

Before: on approval the orchestrator ran the child's one approved call and
handed the turn to the parent model, which knew neither what the child still
had to do nor what it had read. Production 2026-10-21
(thread_UBBhoBvFSOQAPmF4VRjE06Wn): of five requested Google Ads edits, the
child's first approved batch covered one; the other four were never sent, the
requested read-back never ran, and the parent told the user a list id the
child had verified was a guess. Now the adapter's resumer may hand back a
continuation call, which runs here as a deterministic step.
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.core.checkpoint import AgentFrame, Checkpoint, PendingInterrupt
from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.orchestrator import ReActOrchestrator

CONTINUATION = {
    "tool_name": "dispatch_assistant",
    "inputs": {
        "handle": "google_ads_specialist",
        "task": "Continue the task.",
        "intent_summary": "Edit RRTN campaigns",
        "mode": "transfer",
    },
}


class _Registry:
    def _has_registered_tool(self, name):
        return False  # the approved tool belongs to the child


class _Executor:
    _tool_registry = _Registry()

    def __init__(self, tools):
        self._tools = set(tools)

    def has_tool(self, name):
        return name in self._tools


class _Bus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


def _orch(tools, on_dispatch):
    orch = SimpleNamespace(tool_executor=_Executor(tools), event_bus=_Bus())
    orch._record_tool_observation = ReActOrchestrator._record_tool_observation.__get__(
        orch, SimpleNamespace
    )
    orch._steer_report_after_approved_action_failure = (
        ReActOrchestrator._steer_report_after_approved_action_failure.__get__(
            orch, SimpleNamespace
        )
    )
    orch._publish_final_answer_event = ReActOrchestrator._publish_final_answer_event.__get__(
        orch, SimpleNamespace
    )
    orch.dispatched = []

    async def _handle_tool_action(step, context, state, tool_call_id=None, tool_description=None):
        orch.dispatched.append((step.action, step.action_input, tool_call_id))
        on_dispatch(context, state)
        context.messages.append(
            Message(role=MessageRole.TOOL, content="dispatched", tool_call_id=tool_call_id)
        )

    orch._handle_tool_action = _handle_tool_action
    return orch


def _run(orch, continuation=CONTINUATION):
    cp = Checkpoint()
    interrupt = PendingInterrupt(
        interrupt_id="int_child",
        kind="tool_approval",
        raised_by_path=["root", "sub_1"],
        tool_call_id="child_call",
        payload={"tool_name": "google_ads_mutate"},
    )
    cp.agent_frames["root/sub_1"] = AgentFrame(
        path=["root", "sub_1"],
        pending_interrupt=interrupt,
        metadata={"child_thread_id": "thread_child", "handle": "google_ads_specialist"},
    )

    async def resumer(*, action, frame, context):
        outcome = {"success": True, "observation": '{"succeeded": 2}'}
        if continuation is not None:
            outcome["continuation"] = continuation
        return outcome

    deps = {
        "pending_approved_action": {
            "tool_name": "google_ads_mutate",
            "tool_call_id": "child_call",
            "inputs": {"operation": "update_audience_targeting"},
            "interrupt_id": "int_child",
            "raised_by_path": ["root", "sub_1"],
        },
        "child_approval_resumer": resumer,
    }
    messages = [Message(role=MessageRole.TOOL, content="paused", tool_call_id="child_call")]
    ctx = SimpleNamespace(deps=deps, messages=messages, checkpoint=cp)
    state = SimpleNamespace(
        current_step=1,
        is_running=True,
        final_answer="",
        failure_metadata=None,
        needs_clarification=False,
        steps=[],
    )
    asyncio.run(ReActOrchestrator._execute_pending_approved_action(orch, ctx, state))
    return ctx, state


def test_continuation_runs_as_a_paired_tool_step():
    orch = _orch({"dispatch_assistant"}, lambda context, state: None)
    ctx, state = _run(orch)

    (name, inputs, call_id), = orch.dispatched
    assert name == "dispatch_assistant"
    assert inputs == CONTINUATION["inputs"]
    calls = [m for m in ctx.messages if m.role == MessageRole.ASSISTANT]
    assert calls[-1].tool_calls[0]["id"] == call_id
    assert ctx.messages[-1].tool_call_id == call_id
    assert len(state.steps) == 1
    # A report continuation hands back to the model to read the report, with
    # no "report it now" steer contradicting the dispatch.
    assert state.is_running is True
    assert not any(m.role == MessageRole.USER for m in ctx.messages)


def test_a_second_pause_inside_the_continuation_stops_the_run():
    def pause(context, state):
        state.needs_clarification = True

    orch = _orch({"dispatch_assistant"}, pause)
    _, state = _run(orch)
    assert state.is_running is False
    assert state.final_answer == ""


def test_a_transferred_continuation_ends_the_turn_with_the_child_reply():
    def transfer(context, state):
        context.deps["pending_transfer"] = {
            "answer": "All five edits are in place; read back below.",
            "handle": "google_ads_specialist",
            "child_assistant_id": "assistant_child",
        }

    orch = _orch({"dispatch_assistant"}, transfer)
    ctx, state = _run(orch)
    assert state.is_running is False
    assert state.final_answer == "All five edits are in place; read back below."
    assert "pending_transfer" not in ctx.deps
    (step,) = state.steps
    assert step.metadata["transferred_to"] == "google_ads_specialist"
    assert any(e.event_type == ReActEventType.FINAL_ANSWER for e in orch.event_bus.events)


def test_no_continuation_keeps_the_old_report_path():
    orch = _orch({"dispatch_assistant"}, lambda context, state: None)
    ctx, state = _run(orch, continuation=None)
    assert orch.dispatched == []
    assert state.is_running is True
    assert ctx.messages[-1].role == MessageRole.USER  # the report-only steer


def test_an_unavailable_continuation_tool_is_skipped_with_the_report_steer():
    orch = _orch(set(), lambda context, state: None)
    ctx, state = _run(orch)
    assert orch.dispatched == []
    assert state.is_running is True
    assert ctx.messages[-1].role == MessageRole.USER  # the report-only steer
