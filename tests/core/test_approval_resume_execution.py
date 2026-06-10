"""Deterministic approval-resume execution (foundational fix).

The orchestrator, on resume, EXECUTES the approved tool itself instead of asking
the model to re-emit it — eliminating the 'approve → model re-asks/loses the
proposal' class. Tested in isolation by invoking the method on a stand-in self.
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.core.checkpoint import (
    AgentFrame,
    Checkpoint,
    PendingInterrupt,
    ResumeCommand,
)
from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.orchestrator import ReActOrchestrator


class _FakeRegistry:
    def __init__(self, names):
        self._names = set(names)

    def _has_registered_tool(self, name):
        return name in self._names


class _FakeExecutor:
    def __init__(self, registry, result=None, raises=None):
        self._tool_registry = registry
        self._result = result
        self._raises = raises
        self.calls = []

    async def execute_tool(self, name, inputs, context=None):
        self.calls.append((name, dict(inputs)))
        if self._raises:
            raise self._raises
        return self._result


class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, e):
        self.events.append(e)


def _orch(executor):
    orch = SimpleNamespace(tool_executor=executor, event_bus=_FakeBus())
    orch._steer_report_after_approved_action_failure = (
        ReActOrchestrator._steer_report_after_approved_action_failure.__get__(
            orch, SimpleNamespace
        )
    )
    return orch


def test_executes_approved_tool_and_replaces_placeholder():
    async def go():
        reg = _FakeRegistry(["google_ads_mutate"])
        result = SimpleNamespace(
            success=True, output={"created": {"campaign_id": "1002"}}
        )
        ex = _FakeExecutor(reg, result=result)
        orch = _orch(ex)
        messages = [
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "X",
                        "type": "function",
                        "function": {"name": "google_ads_mutate", "arguments": {}},
                    }
                ],
            ),
            Message(
                role=MessageRole.TOOL,
                content="approved — call it again",
                tool_call_id="X",
            ),
        ]
        deps = {
            "pending_approved_action": {
                "tool_name": "google_ads_mutate",
                "tool_call_id": "X",
                "inputs": {"operation": "create_search_campaign"},
            }
        }
        ctx = SimpleNamespace(deps=deps, messages=messages)
        state = SimpleNamespace(
            current_step=1, is_running=True, final_answer="", failure_metadata=None
        )
        await ReActOrchestrator._execute_pending_approved_action(orch, ctx, state)

        assert ex.calls == [
            ("google_ads_mutate", {"operation": "create_search_campaign"})
        ]
        assert (
            "1002" in messages[1].content and "call it again" not in messages[1].content
        )
        assert deps["pending_approved_action"] is None  # consumed
        assert orch.event_bus.events  # observation emitted
        assert state.is_running is True

    asyncio.run(go())


def test_child_owned_tool_without_frame_or_hook_fails_closed():
    async def go():
        ex = _FakeExecutor(_FakeRegistry([]))  # parent lacks the child's tool
        orch = _orch(ex)
        deps = {
            "pending_approved_action": {
                "tool_name": "google_ads_mutate",
                "tool_call_id": "X",
                "inputs": {},
                "raised_by_path": ["root", "sub_1"],
            }
        }
        ctx = SimpleNamespace(deps=deps, messages=[], checkpoint=Checkpoint())
        state = SimpleNamespace(
            current_step=1, is_running=True, final_answer="", failure_metadata=None
        )
        await ReActOrchestrator._execute_pending_approved_action(orch, ctx, state)
        assert ex.calls == []  # NOT executed
        assert deps["pending_approved_action"] is None  # consumed; no legacy free-run
        assert state.is_running is False
        assert state.failure_metadata["stop_reason"] == "child_approval_resume_missing"

    asyncio.run(go())


def test_child_owned_tool_routes_to_child_resumer_hook():
    async def go():
        ex = _FakeExecutor(_FakeRegistry([]))  # parent lacks the child's tool
        orch = _orch(ex)
        cp = Checkpoint()
        interrupt = PendingInterrupt(
            interrupt_id="int_tool_child",
            kind="tool_approval",
            raised_by_path=["root", "sub_1"],
            tool_call_id="child_tool_call",
            payload={"tool_name": "google_ads_mutate"},
        )
        cp.agent_frames["root/sub_1"] = AgentFrame(
            path=["root", "sub_1"],
            pending_interrupt=interrupt,
            metadata={"child_thread_id": "thread_child"},
        )
        cp.pending_approved_action = None
        calls = []

        async def child_resumer(*, action, frame, context):
            calls.append((action["tool_name"], frame.metadata["child_thread_id"]))
            return {"success": True, "observation": '{"created": true}'}

        deps = {
            "pending_approved_action": {
                "tool_name": "google_ads_mutate",
                "tool_call_id": "child_tool_call",
                "inputs": {"operation": "create_search_campaign"},
                "interrupt_id": "int_tool_child",
                "raised_by_path": ["root", "sub_1"],
            },
            "child_approval_resumer": child_resumer,
        }
        messages = [
            Message(
                role=MessageRole.TOOL,
                content="approved — call it again",
                tool_call_id="child_tool_call",
            ),
        ]
        ctx = SimpleNamespace(deps=deps, messages=messages, checkpoint=cp)
        state = SimpleNamespace(
            current_step=1, is_running=True, final_answer="", failure_metadata=None
        )

        await ReActOrchestrator._execute_pending_approved_action(orch, ctx, state)

        assert ex.calls == []
        assert calls == [("google_ads_mutate", "thread_child")]
        assert deps["pending_approved_action"] is None
        assert '{"created": true}' == messages[0].content
        assert orch.event_bus.events
        assert state.is_running is True

    asyncio.run(go())


def test_legacy_approval_descriptor_recovers_child_path_from_checkpoint():
    async def go():
        ex = _FakeExecutor(_FakeRegistry([]))
        orch = _orch(ex)
        cp = Checkpoint()
        interrupt = PendingInterrupt(
            interrupt_id="int_tool_child",
            kind="tool_approval",
            raised_by_path=["root", "sub_1"],
            tool_call_id="child_tool_call",
            payload={"tool_name": "google_ads_mutate", "tool_inputs": {"budget": 50}},
        )
        cp.set_interrupt(interrupt)
        cp.agent_frames["root/sub_1"] = AgentFrame(
            path=["root", "sub_1"],
            pending_interrupt=interrupt,
            metadata={"child_thread_id": "thread_child"},
        )
        calls = []

        async def child_resumer(*, action, frame, context):
            calls.append(
                (
                    action["tool_name"],
                    action["raised_by_path"],
                    frame.metadata["child_thread_id"],
                )
            )
            return {"success": True, "observation": '{"created": true}'}

        deps = {
            # Legacy descriptor from message metadata: no interrupt_id/path.
            "pending_approved_action": {
                "tool_name": "google_ads_mutate",
                "tool_call_id": "child_tool_call",
                "inputs": {"budget": 50},
            },
            "child_approval_resumer": child_resumer,
        }
        messages = [
            Message(
                role=MessageRole.TOOL,
                content="approved — call it again",
                tool_call_id="child_tool_call",
            ),
        ]
        ctx = SimpleNamespace(deps=deps, messages=messages, checkpoint=cp)
        state = SimpleNamespace(
            current_step=1, is_running=True, final_answer="", failure_metadata=None
        )

        await ReActOrchestrator._execute_pending_approved_action(orch, ctx, state)

        assert ex.calls == []
        assert calls == [("google_ads_mutate", ["root", "sub_1"], "thread_child")]
        assert deps["pending_approved_action"] is None
        assert cp.active_interrupt() is None
        assert "root/sub_1" not in cp.agent_frames
        assert '{"created": true}' == messages[0].content
        assert state.is_running is True

    asyncio.run(go())


def test_execution_error_steers_reporting_turn_without_halting():
    async def go():
        ex = _FakeExecutor(_FakeRegistry(["t"]), raises=RuntimeError("boom"))
        orch = _orch(ex)
        messages = [
            Message(role=MessageRole.TOOL, content="placeholder", tool_call_id="X")
        ]
        deps = {
            "pending_approved_action": {
                "tool_name": "t",
                "tool_call_id": "X",
                "inputs": {},
            }
        }
        ctx = SimpleNamespace(deps=deps, messages=messages)
        state = SimpleNamespace(
            current_step=1, is_running=True, final_answer="", failure_metadata=None
        )
        await ReActOrchestrator._execute_pending_approved_action(orch, ctx, state)
        assert "boom" in messages[0].content  # error surfaced as observation
        assert deps["pending_approved_action"] is None
        # Run continues: the model gets a steered reporting turn instead of the
        # templated halt that shipped raw API errors to the user.
        assert state.is_running is True
        assert state.failure_metadata is None
        steer = messages[-1]
        assert steer.role == MessageRole.USER
        assert "FAILED" in steer.content and "plain language" in steer.content
        assert not any(
            e.event_type == ReActEventType.FINAL_ANSWER for e in orch.event_bus.events
        )

    asyncio.run(go())


def test_approved_mutation_status_failed_steers_reporting_turn():
    async def go():
        reg = _FakeRegistry(["google_ads_mutate"])
        result = SimpleNamespace(
            success=True,
            output={
                "status": "failed",
                "failed_step": "create_campaign",
                "error": "API error: duplicate campaign name",
                "created": {"budget_resource_name": "customers/1/campaignBudgets/9"},
                "steps": [{"step": "create_campaign_budget", "status": "completed"}],
            },
        )
        ex = _FakeExecutor(reg, result=result)
        orch = _orch(ex)
        messages = [
            Message(role=MessageRole.TOOL, content="placeholder", tool_call_id="X")
        ]
        deps = {
            "pending_approved_action": {
                "tool_name": "google_ads_mutate",
                "tool_call_id": "X",
                "inputs": {"operation": "create_search_campaign"},
            }
        }
        ctx = SimpleNamespace(deps=deps, messages=messages)
        state = SimpleNamespace(
            current_step=1, is_running=True, final_answer="", failure_metadata=None
        )

        await ReActOrchestrator._execute_pending_approved_action(orch, ctx, state)

        assert ex.calls == [
            ("google_ads_mutate", {"operation": "create_search_campaign"})
        ]
        assert deps["pending_approved_action"] is None
        # The failure result stays in the transcript for the model, and a
        # steering note follows it; the run continues to a reporting turn.
        assert state.is_running is True
        assert state.failure_metadata is None
        assert "duplicate campaign name" in messages[0].content
        steer = messages[-1]
        assert steer.role == MessageRole.USER
        assert "google_ads_mutate" in steer.content
        assert "create_campaign" in steer.content  # failed step carried into steer
        assert "fresh user approval" in steer.content
        observation_events = [
            e
            for e in orch.event_bus.events
            if e.event_type == ReActEventType.OBSERVATION
        ]
        assert observation_events[-1].data["success"] is False
        assert not any(
            e.event_type == ReActEventType.FINAL_ANSWER for e in orch.event_bus.events
        )

    asyncio.run(go())


def test_child_approved_mutation_status_failed_steers_reporting_turn():
    async def go():
        ex = _FakeExecutor(_FakeRegistry([]))
        orch = _orch(ex)
        cp = Checkpoint()
        interrupt = PendingInterrupt(
            interrupt_id="int_tool_child",
            kind="tool_approval",
            raised_by_path=["root", "sub_1"],
            tool_call_id="child_tool_call",
            payload={"tool_name": "google_ads_mutate"},
        )
        cp.agent_frames["root/sub_1"] = AgentFrame(
            path=["root", "sub_1"],
            pending_interrupt=interrupt,
            metadata={"child_thread_id": "thread_child"},
        )

        async def child_resumer(*, action, frame, context):
            return {
                "success": True,
                "observation": (
                    '{"status":"failed","failed_step":"create_campaign",'
                    '"error":"duplicate campaign name",'
                    '"created":{"budget_resource_name":"customers/1/campaignBudgets/9"}}'
                ),
            }

        deps = {
            "pending_approved_action": {
                "tool_name": "google_ads_mutate",
                "tool_call_id": "child_tool_call",
                "inputs": {"operation": "create_search_campaign"},
                "interrupt_id": "int_tool_child",
                "raised_by_path": ["root", "sub_1"],
            },
            "child_approval_resumer": child_resumer,
        }
        messages = [
            Message(
                role=MessageRole.TOOL,
                content="placeholder",
                tool_call_id="child_tool_call",
            )
        ]
        ctx = SimpleNamespace(deps=deps, messages=messages, checkpoint=cp)
        state = SimpleNamespace(
            current_step=1, is_running=True, final_answer="", failure_metadata=None
        )

        await ReActOrchestrator._execute_pending_approved_action(orch, ctx, state)

        assert ex.calls == []
        assert deps["pending_approved_action"] is None
        # Run continues into a steered reporting turn; the raw failure JSON
        # stays in the transcript (for the model), not in a user-facing answer.
        assert state.is_running is True
        assert state.failure_metadata is None
        assert "duplicate campaign name" in messages[0].content
        steer = messages[-1]
        assert steer.role == MessageRole.USER
        assert "google_ads_mutate" in steer.content
        assert "create_campaign" in steer.content
        observation_events = [
            e
            for e in orch.event_bus.events
            if e.event_type == ReActEventType.OBSERVATION
        ]
        assert observation_events[-1].data["success"] is False
        assert not any(
            e.event_type == ReActEventType.FINAL_ANSWER for e in orch.event_bus.events
        )

    asyncio.run(go())


def test_no_descriptor_is_noop():
    async def go():
        ex = _FakeExecutor(_FakeRegistry(["t"]))
        orch = _orch(ex)
        ctx = SimpleNamespace(deps={}, messages=[])
        await ReActOrchestrator._execute_pending_approved_action(
            orch, ctx, SimpleNamespace(current_step=1)
        )
        assert ex.calls == []

    asyncio.run(go())


def test_child_success_resolves_dispatch_placeholder_and_steers_report():
    """Regression: thread_SipzgTCTP4CJ9pnjtDGFSREw — after a SUCCESSFUL
    deterministic child resume, the model saw only a bare success result next
    to the pause turn's frozen 'waiting for user approval' dispatch
    observation, concluded the work was unconfirmed, and re-dispatched the
    identical mutation (another approval modal). Success must rewrite the
    stale dispatch observation AND steer an explicit report-only turn."""

    async def go():
        ex = _FakeExecutor(_FakeRegistry([]))
        orch = _orch(ex)
        cp = Checkpoint()
        interrupt = PendingInterrupt(
            interrupt_id="int_tool_child",
            kind="tool_approval",
            raised_by_path=["root", "sub_1"],
            tool_call_id="child_tool_call",
            payload={"tool_name": "google_ads_mutate"},
        )
        cp.agent_frames["root/sub_1"] = AgentFrame(
            path=["root", "sub_1"],
            pending_interrupt=interrupt,
            metadata={"child_thread_id": "thread_child"},
        )

        async def child_resumer(*, action, frame, context):
            return {
                "success": True,
                "observation": '{"status":"completed","created":{"campaign_id":"1"}}',
            }

        deps = {
            "pending_approved_action": {
                "tool_name": "google_ads_mutate",
                "tool_call_id": "child_tool_call",
                "inputs": {},
                "interrupt_id": "int_tool_child",
                "raised_by_path": ["root", "sub_1"],
                "parent_tool_call_id": "parent_dispatch_call",
            },
            "child_approval_resumer": child_resumer,
        }
        messages = [
            Message(
                role=MessageRole.TOOL,
                content="Tool execution paused - waiting for user approval.",
                tool_call_id="parent_dispatch_call",
            ),
            Message(
                role=MessageRole.TOOL,
                content="approved — call it again",
                tool_call_id="child_tool_call",
            ),
        ]
        ctx = SimpleNamespace(deps=deps, messages=messages, checkpoint=cp)
        state = SimpleNamespace(
            current_step=1, is_running=True, final_answer="", failure_metadata=None
        )

        await ReActOrchestrator._execute_pending_approved_action(orch, ctx, state)

        # The frozen dispatch observation no longer claims to be waiting.
        assert "waiting for user approval" not in messages[0].content
        assert "approved" in messages[0].content and "do not repeat" in messages[0].content
        # The child result replaced its placeholder.
        assert '"completed"' in messages[1].content
        # Success steer appended: explicit ALREADY-executed + report-only.
        steer = messages[-1]
        assert steer.role == MessageRole.USER
        assert "ALREADY been executed" in steer.content
        assert "do not re-dispatch" in steer.content.lower()
        # Run continues to the reporting turn.
        assert state.is_running is True

    asyncio.run(go())


def test_rejected_resume_acknowledges_deterministically_without_llm():
    async def go():
        orch = SimpleNamespace(event_bus=_FakeBus())
        orch._acknowledge_rejected_approval = (
            ReActOrchestrator._acknowledge_rejected_approval.__get__(
                orch, SimpleNamespace
            )
        )
        resume = ResumeCommand(
            interrupt_id="int_tool_tc_1",
            kind="tool_approval",
            decision="rejected",
            value={"reason": "Holding off — confirm budget with finance first."},
        )
        ctx = SimpleNamespace(deps={}, messages=[])
        state = SimpleNamespace(
            current_step=0, is_running=True, final_answer="", failure_metadata=None
        )

        handled = await orch._acknowledge_rejected_approval(resume, ctx, state)

        assert handled is True
        assert state.is_running is False
        assert "cancelled" in state.final_answer
        assert "Holding off" in state.final_answer  # user-authored reason mirrored
        assert "What would you like to do instead" in state.final_answer
        from miiflow_agent.core.react.enums import ReActEventType

        assert any(
            e.event_type == ReActEventType.FINAL_ANSWER for e in orch.event_bus.events
        )

    asyncio.run(go())


def test_rejected_resume_default_ui_reason_is_not_mirrored():
    async def go():
        orch = SimpleNamespace(event_bus=_FakeBus())
        orch._acknowledge_rejected_approval = (
            ReActOrchestrator._acknowledge_rejected_approval.__get__(
                orch, SimpleNamespace
            )
        )
        resume = ResumeCommand(
            interrupt_id="int_tool_tc_1",
            kind="tool_approval",
            decision="rejected",
            value={"reason": "User declined this change"},
        )
        state = SimpleNamespace(
            current_step=0, is_running=True, final_answer="", failure_metadata=None
        )
        await orch._acknowledge_rejected_approval(
            resume, SimpleNamespace(deps={}, messages=[]), state
        )
        assert "User declined this change" not in state.final_answer
        assert state.is_running is False

    asyncio.run(go())


def test_approved_or_absent_resume_does_not_acknowledge():
    async def go():
        orch = SimpleNamespace(event_bus=_FakeBus())
        orch._acknowledge_rejected_approval = (
            ReActOrchestrator._acknowledge_rejected_approval.__get__(
                orch, SimpleNamespace
            )
        )
        state = SimpleNamespace(
            current_step=0, is_running=True, final_answer="", failure_metadata=None
        )
        ctx = SimpleNamespace(deps={}, messages=[])
        approved = ResumeCommand(
            interrupt_id="i", kind="tool_approval", decision="approved", value={}
        )
        assert await orch._acknowledge_rejected_approval(approved, ctx, state) is False
        assert await orch._acknowledge_rejected_approval(None, ctx, state) is False
        clar = ResumeCommand(
            interrupt_id="i", kind="clarification", decision="answered", value={}
        )
        assert await orch._acknowledge_rejected_approval(clar, ctx, state) is False
        assert state.is_running is True

    asyncio.run(go())


def test_apply_resume_command_consumes_stored_resume():
    cp = Checkpoint()
    cp.set_interrupt(
        PendingInterrupt(
            interrupt_id="int_tool_tc_1",
            kind="tool_approval",
            tool_call_id="tc_1",
            payload={"tool_name": "google_ads_mutate", "tool_inputs": {}},
        )
    )
    cp.resume = ResumeCommand(
        interrupt_id="int_tool_tc_1",
        kind="tool_approval",
        decision="rejected",
        value={"reason": "no"},
    )
    ctx = SimpleNamespace(deps={}, checkpoint=cp, resume=None)

    returned = ReActOrchestrator._apply_resume_command(SimpleNamespace(), ctx)

    assert returned is not None and returned.decision == "rejected"
    assert cp.resume is None  # consumed — can never replay on a later turn
    assert cp.active_interrupt() is None
    # A second call (next turn) finds nothing to act on.
    assert ReActOrchestrator._apply_resume_command(SimpleNamespace(), ctx) is None


def test_resume_command_creates_checkpoint_pending_approved_action():
    cp = Checkpoint()
    cp.set_interrupt(
        PendingInterrupt(
            interrupt_id="int_tool_tc_1",
            kind="tool_approval",
            tool_call_id="tc_1",
            payload={
                "tool_name": "google_ads_mutate",
                "tool_inputs": {"budget": 50},
            },
        )
    )
    resume = ResumeCommand(
        interrupt_id="int_tool_tc_1",
        kind="tool_approval",
        decision="approved",
        value={"modified_inputs": {"budget": 40}},
    )
    ctx = SimpleNamespace(deps={}, checkpoint=cp, resume=resume)

    ReActOrchestrator._apply_resume_command(SimpleNamespace(), ctx)

    assert cp.active_interrupt() is None
    assert cp.pending_approved_action.tool_name == "google_ads_mutate"
    assert cp.pending_approved_action.inputs == {"budget": 40}
    assert ctx.deps["pending_approved_action"]["tool_call_id"] == "tc_1"
