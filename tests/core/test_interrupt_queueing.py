"""Concurrent-interrupt queueing (parallel child pauses must not strand).

``dispatch_assistant`` is parallelizable, so two children can pause in ONE
batch. Before the queueing fix, ``_record_interrupt`` → ``set_interrupt``
silently replaced the active interrupt: the later pause became active and the
earlier one was stranded in ``interrupts`` forever (never in
``interrupt_queue``, never re-activated). Now the prior active — when raised
by this same run — is demoted into the queue, and resolving the active one
promotes it back (flagged for UI re-surfacing via ``resurface_interrupt_id``).
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.core.checkpoint import Checkpoint, PendingInterrupt, ResumeCommand
from miiflow_agent.core.react.execution import ExecutionState
from miiflow_agent.core.react.orchestrator import ReActOrchestrator


class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, e):
        self.events.append(e)


def _orch():
    return SimpleNamespace(event_bus=_FakeBus())


def _interrupt(iid, kind="tool_approval", tool_call_id=None):
    return PendingInterrupt(
        interrupt_id=iid,
        kind=kind,
        raised_by_path=["root"],
        payload={"tool_name": "google_ads_mutate", "tool_inputs": {}},
        tool_call_id=tool_call_id or f"tc_{iid}",
    )


def test_set_interrupt_removes_new_active_from_queue():
    cp = Checkpoint(thread_id="t")
    a, b = _interrupt("int_a"), _interrupt("int_b")
    cp.set_interrupt(a)
    cp.interrupt_queue.append(b.interrupt_id)
    cp.interrupts[b.interrupt_id] = b
    # Re-activating a queued interrupt must not leave it queued too.
    cp.set_interrupt(b)
    assert cp.active_interrupt_id == "int_b"
    assert "int_b" not in cp.interrupt_queue


def test_record_interrupt_queues_prior_active_raised_this_run():
    async def go():
        orch = _orch()
        cp = Checkpoint(thread_id="t")
        ctx = SimpleNamespace(checkpoint=cp)
        state = ExecutionState()

        first = await ReActOrchestrator._record_interrupt(
            orch,
            ctx,
            state,
            kind="tool_approval",
            payload={"tool_name": "google_ads_mutate", "tool_inputs": {"a": 1}},
            tool_call_id="tc_1",
        )
        second = await ReActOrchestrator._record_interrupt(
            orch,
            ctx,
            state,
            kind="tool_approval",
            payload={"tool_name": "google_ads_mutate", "tool_inputs": {"b": 2}},
            tool_call_id="tc_2",
        )

        # Later pause wins the (single) UI slot; the earlier one is QUEUED,
        # not stranded.
        assert cp.active_interrupt_id == second.interrupt_id
        assert cp.interrupt_queue == [first.interrupt_id]
        assert first.interrupt_id in cp.interrupts

        # Resolving the active one promotes the queued one.
        cp.clear_active_interrupt()
        assert cp.active_interrupt_id == first.interrupt_id
        assert cp.interrupt_queue == []

    asyncio.run(go())


def test_record_interrupt_does_not_resurrect_stale_active():
    """An active interrupt persisted by an OLD turn keeps the historical
    replace-and-forget behavior — only same-run pauses are queued."""

    async def go():
        orch = _orch()
        cp = Checkpoint(thread_id="t")
        stale = _interrupt("int_stale")
        cp.set_interrupt(stale)
        ctx = SimpleNamespace(checkpoint=cp)
        state = ExecutionState()  # fresh run: raised_interrupt_ids is empty

        fresh = await ReActOrchestrator._record_interrupt(
            orch,
            ctx,
            state,
            kind="clarification",
            payload={"questions": []},
            tool_call_id="tc_new",
        )
        assert cp.active_interrupt_id == fresh.interrupt_id
        assert cp.interrupt_queue == []

    asyncio.run(go())


def test_apply_resume_command_promotes_queued_interrupt_and_flags_resurface():
    async def go():
        orch = _orch()
        cp = Checkpoint(thread_id="t")
        answered = _interrupt("int_answered", tool_call_id="tc_1")
        queued = _interrupt("int_queued", tool_call_id="tc_2")
        cp.set_interrupt(answered)
        cp.interrupts[queued.interrupt_id] = queued
        cp.interrupt_queue.append(queued.interrupt_id)
        cp.resume = ResumeCommand(
            interrupt_id="int_answered",
            kind="tool_approval",
            decision="approved",
            value={"tool_inputs": {}},
        )
        ctx = SimpleNamespace(checkpoint=cp, deps={}, resume=None)

        consumed = ReActOrchestrator._apply_resume_command(orch, ctx)

        assert consumed is not None and consumed.interrupt_id == "int_answered"
        assert cp.resume is None
        # The queued pause was promoted and flagged for UI re-surfacing —
        # the frontend only ever rendered the answered one.
        assert cp.active_interrupt_id == "int_queued"
        assert cp.extra.get("resurface_interrupt_id") == "int_queued"

    asyncio.run(go())


def test_resume_leaves_stored_tool_results_byte_identical():
    """A request must be a byte prefix of the next: resuming one approval of a
    paused batch must not rewrite the waiting sibling's stored result (the
    first fix for thread_474iZNmonj4UxDnb1ZugJAmw relabelled it on resume,
    which broke the cached prefix and kept a stale label in history). The
    waiting text is written once, at pause time — see
    `tests/core/react/test_batch_child_approval.py`."""
    from miiflow_agent.core.message import Message, MessageRole
    from miiflow_agent.core.react.orchestrator import waiting_pause_observation

    def child_pause(iid, tool, parent):
        interrupt = _interrupt(iid, tool_call_id=f"child_{iid}")
        interrupt.payload = {
            "tool_name": tool,
            "tool_inputs": {},
            "handle": tool.split("_")[0] + "_ads_specialist",
            "parent_tool_call_id": parent,
        }
        return interrupt

    cp = Checkpoint(thread_id="t")
    meta = child_pause("int_meta", "meta_ads_mutate", "dispatch_meta")
    google = child_pause("int_google", "google_ads_mutate", "dispatch_google")
    cp.interrupts[meta.interrupt_id] = meta
    cp.set_interrupt(google)
    cp.interrupt_queue.append(meta.interrupt_id)
    cp.resume = ResumeCommand(
        interrupt_id="int_google", kind="tool_approval", decision="approved", value={}
    )
    stored = [
        waiting_pause_observation("meta_ads_mutate"),
        waiting_pause_observation("google_ads_mutate"),
    ]
    messages = [
        Message(role=MessageRole.TOOL, content=stored[0], tool_call_id="dispatch_meta"),
        Message(role=MessageRole.TOOL, content=stored[1], tool_call_id="dispatch_google"),
    ]
    ctx = SimpleNamespace(checkpoint=cp, deps={}, resume=None, messages=messages)

    ReActOrchestrator._apply_resume_command(_orch(), ctx)

    assert [m.content for m in messages] == stored
    # The queued sibling is promoted and flagged for re-surfacing instead.
    assert cp.active_interrupt_id == "int_meta"
    assert cp.extra.get("resurface_interrupt_id") == "int_meta"
    # And this run is marked as resuming the batch (dispatch guard reads it).
    from miiflow_agent.core.react.approval_flow import RESUMED_APPROVAL_DEP

    assert ctx.deps[RESUMED_APPROVAL_DEP] == "int_google"
    assert [i.interrupt_id for i in cp.waiting_tool_approvals()] == ["int_meta"]


def test_waiting_tool_approvals_lists_active_then_queued_approvals_only():
    cp = Checkpoint(thread_id="t")
    a, b = _interrupt("int_a"), _interrupt("int_b")
    question = _interrupt("int_q", kind="clarification")
    cp.set_interrupt(a)
    for queued in (question, b):
        cp.interrupts[queued.interrupt_id] = queued
        cp.interrupt_queue.append(queued.interrupt_id)
    assert [i.interrupt_id for i in cp.waiting_tool_approvals()] == ["int_a", "int_b"]
