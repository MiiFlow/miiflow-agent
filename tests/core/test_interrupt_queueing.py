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
