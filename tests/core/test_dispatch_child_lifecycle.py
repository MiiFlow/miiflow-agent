"""A dispatched child's run ends when its dispatch ends, and its answer's media render.

Regression for 2026-09-24 (thread_MZJLsF4nUnmokoxmOsaKvyug): every transfer
dispatch whose child narrated before a tool call raised NameError in the
forwarder; the abandoned children kept generating and overwriting a storyboard
for minutes, their ContextVar resets raised "created in a different Context"
from the garbage collector's task, and the final answer's `[MEDIA:memfile_…]`
refs rendered "Preview unavailable".
"""
from __future__ import annotations

import asyncio
import contextvars
from typing import List
from unittest.mock import MagicMock

import pytest


def _make_event_bus():
    from miiflow_agent.core.react.events.bus import EventBus

    bus = EventBus()
    received: List = []
    bus.subscribe(lambda ev: received.append(ev))
    return bus, received


def _make_client():
    from miiflow_agent.core.tools import ToolRegistry

    client = MagicMock()
    client.tool_registry = ToolRegistry()
    return client


# ── transfer forwarding ──────────────────────────────────────────────────


def test_transfer_forwards_child_narration_with_its_own_step():
    """ASSISTANT_TEXT from a transferred child reaches the parent stream keyed
    by the CHILD's step — the branch that raised NameError on `event`."""
    from miiflow_agent.core.react.dispatch import forward_subagent_events
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.react.react_events import ReActEvent

    async def child():
        yield ReActEvent(
            event_type=ReActEventType.ASSISTANT_TEXT,
            step_number=4,
            data={"text": "Generating frame 1 now."},
        )

    bus, received = _make_event_bus()
    asyncio.run(
        forward_subagent_events(
            child(),
            parent_event_bus=bus,
            parent_step_number=2,
            subagent_id="sub_abc",
            own_path=["sub_abc"],
            promote_to_final=True,
        )
    )

    assert [e.event_type for e in received] == [ReActEventType.ASSISTANT_TEXT]
    assert received[0].step_number == 2
    assert received[0].data["transcript_step"] == "sub_abc:4"
    assert received[0].data["text"] == "Generating frame 1 now."


# ── the dispatch owns the child's run ─────────────────────────────────────


_CHILD_VAR: contextvars.ContextVar = contextvars.ContextVar("child_var", default=None)


class _ChildHoldingContextVar:
    """A child that, like AssistantInvoker.stream, sets a ContextVar before
    its first yield and resets it in `finally` — which only works when the
    stream is closed from the task that advanced it."""

    handle = "creative_strategist"
    name = "Creative Strategist"
    description = ""
    when_to_use = ""

    def __init__(self):
        self.closed_in_task = None
        self.close_error = None
        self.events_after_close = 0

    async def stream(self, handoff):
        from miiflow_agent.core.react.enums import ReActEventType
        from miiflow_agent.core.react.react_events import ReActEvent

        token = _CHILD_VAR.set("child")
        try:
            for step in range(100):
                yield ReActEvent(
                    event_type=ReActEventType.FINAL_ANSWER_CHUNK,
                    step_number=step,
                    data={"delta": f"chunk {step} "},
                )
        finally:
            self.closed_in_task = asyncio.current_task()
            try:
                _CHILD_VAR.reset(token)
            except ValueError as exc:  # the production PostHog error
                self.close_error = exc

    def final_result(self):
        from miiflow_agent.core.subagent import SubAgentResult

        return SubAgentResult(answer="", status="completed")


def test_forwarder_failure_closes_child_in_the_dispatching_task(monkeypatch):
    from miiflow_agent.core.react import dispatch as dispatch_mod
    from miiflow_agent.core.react.dispatch import DispatchCounter, dispatch_subagent
    from miiflow_agent.core.subagent import SubAgentHandoff

    async def exploding_forwarder(child_events, **_kwargs):
        async for _event in child_events:
            raise NameError("name 'event' is not defined")

    monkeypatch.setattr(dispatch_mod, "forward_subagent_events", exploding_forwarder)
    child = _ChildHoldingContextVar()
    bus, _ = _make_event_bus()

    async def run():
        result = await dispatch_subagent(
            child,
            SubAgentHandoff(task="t", intent_summary="s"),
            parent_event_bus=bus,
            parent_step_number=1,
            parent_assistant_id="parent",
            child_id="child",
            counter=DispatchCounter(),
            transfer=True,
        )
        return result, asyncio.current_task()

    result, dispatching_task = asyncio.run(run())

    assert result.status == "failed"
    assert "event" in (result.error or "")
    # Closed before dispatch_subagent returned, in the same task, so its
    # ContextVar reset succeeded.
    assert child.closed_in_task is dispatching_task
    assert child.close_error is None


def test_cancelled_parent_closes_child_in_the_dispatching_task():
    from miiflow_agent.core.react.dispatch import DispatchCounter, dispatch_subagent
    from miiflow_agent.core.subagent import SubAgentHandoff

    child = _ChildHoldingContextVar()
    received_first = asyncio.Event()

    class _Bus:
        async def publish(self, event):
            if event.data.get("sub_event") == "progress":
                received_first.set()
                await asyncio.sleep(3600)

    async def run():
        task = asyncio.create_task(
            dispatch_subagent(
                child,
                SubAgentHandoff(task="t", intent_summary="s"),
                parent_event_bus=_Bus(),
                parent_step_number=1,
                parent_assistant_id="parent",
                child_id="child",
                counter=DispatchCounter(),
            )
        )
        await received_first.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task

    dispatching_task = asyncio.run(run())

    assert child.closed_in_task is dispatching_task
    assert child.close_error is None


# ── an abandoned stream stops its run ─────────────────────────────────────


def test_closing_agent_stream_cancels_the_orchestrator_run(monkeypatch):
    """The orchestrator runs in its own task; a consumer that stops reading
    must stop the run, not leave it calling tools with nobody listening."""
    from miiflow_agent.core.agent import Agent, AgentType, RunContext
    from miiflow_agent.core.react import ReActFactory
    from miiflow_agent.core.react.events.bus import EventBus
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.react.react_events import ReActEvent

    run_state = {"steps": 0, "cancelled": False}

    class _Orchestrator:
        def __init__(self):
            self.event_bus = EventBus()

        async def execute(self, query, context):
            try:
                while True:
                    run_state["steps"] += 1
                    await self.event_bus.publish(
                        ReActEvent(
                            event_type=ReActEventType.THINKING_CHUNK,
                            step_number=run_state["steps"],
                            data={"delta": "working"},
                        )
                    )
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                run_state["cancelled"] = True
                raise

    monkeypatch.setattr(
        ReActFactory, "create_orchestrator", staticmethod(lambda **_: _Orchestrator())
    )
    agent = Agent(_make_client(), agent_type=AgentType.REACT)

    async def consume_one_then_close():
        stream = agent._stream_react("q", RunContext(deps={}, messages=[]))
        await stream.__anext__()
        await stream.aclose()
        steps_at_close = run_state["steps"]
        await asyncio.sleep(0.1)
        return steps_at_close

    steps_at_close = asyncio.run(consume_one_then_close())

    assert run_state["cancelled"] is True
    assert run_state["steps"] == steps_at_close


# ── the answer's media refs are presented ────────────────────────────────


def test_final_answer_presents_referenced_media_the_run_never_showed():
    from miiflow_agent.core.agent import Agent, AgentType
    from miiflow_agent.core.react import ReActFactory
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.react.execution import ExecutionState

    agent = Agent(_make_client(), agent_type=AgentType.REACT)
    orchestrator = ReActFactory.create_orchestrator(agent=agent, max_steps=1)
    received: List = []
    orchestrator.event_bus.subscribe(lambda ev: received.append(ev))

    state = ExecutionState()
    state.final_answer = (
        "Frame 4: [MEDIA:memfile_frame4]\n\n"
        "| Frame | Preview |\n| --- | --- |\n| 5 | ![CTA](media_ref:memfile_frame5) |\n\n"
        "Earlier: [MEDIA:gen_shown] and again [MEDIA:memfile_frame4]. "
        "Missing: [MEDIA:memfile_unknown]. Video: [MEDIA:clip]"
    )
    state.media_store = {
        "memfile_frame4": "https://s3.test/bucket/a/frame4.png?X-Amz-Signature=1",
        "memfile_frame5": "https://s3.test/bucket/a/frame5.png",
        "gen_shown": "https://s3.test/bucket/a/gen.png",
        "clip": "https://s3.test/bucket/a/clip.mp4",
    }
    state.presented_media_ids = {"gen_shown"}

    asyncio.run(orchestrator._build_result(state, None))

    media = [e.data["media"] for e in received if e.event_type == ReActEventType.MEDIA]
    assert [m["id"] for m in media] == ["memfile_frame4", "clip", "memfile_frame5"]
    assert media[0]["url"] == state.media_store["memfile_frame4"]
    assert [m["media_type"] for m in media] == ["image", "video", "image"]
    assert {"memfile_frame4", "memfile_frame5", "clip"} <= state.presented_media_ids


def test_tool_media_is_not_presented_twice():
    """Media a tool already published this run is marked presented, so the
    closing pass does not re-emit it."""
    from miiflow_agent.core.agent import Agent, AgentType
    from miiflow_agent.core.react import ReActFactory
    from miiflow_agent.core.react.enums import ReActEventType
    from miiflow_agent.core.react.execution import ExecutionState

    agent = Agent(_make_client(), agent_type=AgentType.REACT)
    orchestrator = ReActFactory.create_orchestrator(agent=agent, max_steps=1)
    received: List = []
    orchestrator.event_bus.subscribe(lambda ev: received.append(ev))
    state = ExecutionState()

    asyncio.run(
        orchestrator._tool_actions.process_result(
            {"__media__": True, "media_type": "image", "id": "gen_1", "url": "https://s3.test/bucket/g.png"},
            state,
            "generate_ad_image",
        )
    )
    state.final_answer = "Here it is: [MEDIA:gen_1]"
    asyncio.run(orchestrator._build_result(state, None))

    media_ids = [e.data["media"]["id"] for e in received if e.event_type == ReActEventType.MEDIA]
    assert media_ids == ["gen_1"]
