"""Round-trip + reducer tests for the durable run Checkpoint (Phase 0).

The checkpoint is the single serializable source of truth for pause/resume +
multi-agent context. These tests pin its two load-bearing guarantees:

  1. Faithful JSON round-trip (it persists into ``thread.metadata`` as plain JSON),
     including forward-compat preservation of unknown keys.
  2. Deterministic helpers Phases 1/3 depend on: established-facts exact-match lookup
     + upsert, and the dispatch-ledger dedupe reducer.
"""

import json

from miiflow_agent.core.checkpoint import (
    CHECKPOINT_VERSION,
    AgentFrame,
    Checkpoint,
    DispatchLedgerEntry,
    EstablishedFact,
    PendingApprovedAction,
    PendingInterrupt,
    ResumeCommand,
)
from miiflow_agent.core.message import Message, MessageRole


def _full_checkpoint() -> Checkpoint:
    cp = Checkpoint(
        thread_id="thread_abc",
        established_facts=[
            EstablishedFact(
                key="daily_budget",
                answer="$50/day",
                question_text="What daily budget?",
                options=["$25/day", "$50/day"],
                answered_by_path=["root"],
                turn=1,
            )
        ],
        pending_interrupt=PendingInterrupt(
            interrupt_id="int_1",
            kind="tool_approval",
            raised_by_path=["root", "tc_dispatch_9"],
            payload={"tool_name": "google_ads_mutate", "tool_inputs": {"budget": 50}},
            tool_call_id="tc_2",
        ),
        pending_approved_action=PendingApprovedAction(
            tool_name="google_ads_mutate",
            tool_call_id="tc_2",
            inputs={"budget": 40},
        ),
        resume=ResumeCommand(
            interrupt_id="int_1",
            kind="tool_approval",
            decision="approved",
            value={"modified_inputs": {"budget": 40}},
        ),
        dispatch_ledger=[
            DispatchLedgerEntry(
                kind="tool_call",
                tool_name="search_memory",
                inputs_hash="h1",
                digest="brand voice = bold",
                observation_ref="agent_obs_h1",
                produced_at=1234.5,
                turn_index=1,
                produced_by_path=["root", "tc_a"],
            )
        ],
        agent_frames={
            "root/tc_dispatch_9": AgentFrame(
                path=["root", "tc_dispatch_9"],
                transcript=[Message.user("subtask")],
                step_pointer=3,
                pending_interrupt=PendingInterrupt(
                    interrupt_id="int_1",
                    kind="clarification",
                    raised_by_path=["root", "tc_dispatch_9"],
                    payload={"questions": [{"key": "goal", "prompt": "leads or sales?"}]},
                ),
            )
        },
    )
    return cp


def test_checkpoint_round_trips_through_json():
    cp = _full_checkpoint()
    blob = json.dumps(cp.to_dict())  # must be plain-JSON serializable
    restored = Checkpoint.from_dict(json.loads(blob))

    assert restored.version == CHECKPOINT_VERSION
    assert restored.thread_id == "thread_abc"
    # Legacy blobs may still carry a "transcript" key (the field never had a
    # writer and was removed); it must be silently dropped, not preserved.
    legacy = json.loads(blob)
    legacy["transcript"] = [{"role": "user", "content": "old"}]
    assert "transcript" not in Checkpoint.from_dict(legacy).extra

    fact = restored.fact("daily_budget")
    assert fact is not None and fact.answer == "$50/day"
    assert fact.options == ["$25/day", "$50/day"]

    assert restored.pending_interrupt.kind == "tool_approval"
    assert restored.pending_interrupt.raised_by_path == ["root", "tc_dispatch_9"]
    assert restored.active_interrupt().interrupt_id == "int_1"
    assert restored.resume.decision == "approved"
    assert restored.pending_approved_action.inputs == {"budget": 40}

    frame = restored.agent_frames["root/tc_dispatch_9"]
    assert frame.step_pointer == 3
    assert frame.pending_interrupt.payload["questions"][0]["key"] == "goal"


def test_empty_and_none_round_trip():
    assert Checkpoint.from_dict(None).established_facts == []
    assert Checkpoint.from_dict({}).pending_interrupt is None
    cp = Checkpoint()
    assert Checkpoint.from_dict(cp.to_dict()).version == CHECKPOINT_VERSION


def test_unknown_keys_preserved_for_forward_compat():
    # A newer binary wrote a field this (older) one doesn't know. We must not destroy it.
    raw = Checkpoint().to_dict()
    raw["future_field"] = {"some": "state"}
    restored = Checkpoint.from_dict(raw)
    assert restored.extra["future_field"] == {"some": "state"}
    assert restored.to_dict()["future_field"] == {"some": "state"}


def test_established_fact_upsert_supersedes_same_key():
    cp = Checkpoint()
    cp.upsert_fact(EstablishedFact(key="geo", answer="California"))
    cp.upsert_fact(EstablishedFact(key="geo", answer="Texas"))
    assert len([f for f in cp.established_facts if f.key == "geo"]) == 1
    assert cp.fact("geo").answer == "Texas"


def test_dispatch_ledger_reducer_dedupes_by_key():
    cp = Checkpoint()
    cp.merge_ledger(
        [
            DispatchLedgerEntry(kind="tool_call", tool_name="search_memory", inputs_hash="h1", success=False),
            DispatchLedgerEntry(kind="dispatch", handle="opportunity_investigator", task_hash="t1"),
        ]
    )
    # Re-running the same tool_call (same identity) supersedes, doesn't duplicate.
    cp.merge_ledger(
        [
            DispatchLedgerEntry(
                kind="tool_call", tool_name="search_memory", inputs_hash="h1",
                success=True, digest="found",
            )
        ]
    )
    tool_entries = [e for e in cp.dispatch_ledger if e.kind == "tool_call"]
    assert len(tool_entries) == 1
    assert tool_entries[0].success is True and tool_entries[0].digest == "found"
    assert len(cp.dispatch_ledger) == 2  # the dispatch entry is untouched


def test_canonical_interrupt_helpers_keep_legacy_mirror():
    cp = Checkpoint()
    interrupt = PendingInterrupt(
        interrupt_id="int_clarification_tc_1",
        kind="clarification",
        payload={"questions": [{"key": "goal"}]},
        tool_call_id="tc_1",
    )
    cp.set_interrupt(interrupt)

    assert cp.active_interrupt_id == "int_clarification_tc_1"
    assert cp.active_interrupt() is interrupt
    assert cp.pending_interrupt is interrupt

    restored = Checkpoint.from_dict(cp.to_dict())
    assert restored.active_interrupt().payload["questions"][0]["key"] == "goal"
    assert restored.pending_interrupt.interrupt_id == "int_clarification_tc_1"

    restored.clear_active_interrupt()
    assert restored.active_interrupt() is None
    assert restored.pending_interrupt is None


def test_fact_precedence_user_beats_tool():
    """The precedence lattice: a tool-promoted fact never overwrites a user answer."""
    cp = Checkpoint()
    cp.upsert_fact(
        EstablishedFact(key="k", answer="user said A", question_text="q", source="user")
    )
    cp.upsert_fact(
        EstablishedFact(key="k", answer="tool derived B", question_text="q", source="tool")
    )
    assert cp.fact("k").answer == "user said A"

    # Tool facts refresh freely among themselves…
    cp.upsert_fact(
        EstablishedFact(key="tool:x", answer="v1", source="tool")
    )
    cp.upsert_fact(
        EstablishedFact(key="tool:x", answer="v2", source="tool")
    )
    assert cp.fact("tool:x").answer == "v2"

    # …and a user answer replaces a tool fact for the same key.
    cp.upsert_fact(EstablishedFact(key="tool:x", answer="user override", source="user"))
    assert cp.fact("tool:x").answer == "user override"


def test_facts_cap_drops_tool_facts_first():
    cp = Checkpoint()
    cp.upsert_fact(EstablishedFact(key="user:0", answer="keep", source="user"))
    for i in range(Checkpoint.FACTS_MAX + 10):
        cp.upsert_fact(EstablishedFact(key=f"tool:{i}", answer=str(i), source="tool"))
    assert len(cp.established_facts) <= Checkpoint.FACTS_MAX
    assert cp.fact("user:0") is not None  # user answers survive


def test_fact_source_round_trips():
    fact = EstablishedFact(key="tool:a", answer="x", source="tool")
    assert EstablishedFact.from_dict(fact.to_dict()).source == "tool"
    # Legacy dicts (no source) default to "user" — old clarification answers
    # keep their protected status.
    legacy = fact.to_dict()
    del legacy["source"]
    assert EstablishedFact.from_dict(legacy).source == "user"
