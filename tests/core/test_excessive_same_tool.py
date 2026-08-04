"""ExcessiveSameToolCondition — caps a single tool's per-turn call count.

Guards the failure mode where a model flails (e.g. ~20 search_memory calls with slightly
varied queries, all fruitless) — which RepeatedActionsCondition misses because the args
differ — burning the turn and then echoing the last empty tool result as its answer.
"""

from miiflow_agent.core.react.models import ReActStep, ToolInvocation
from miiflow_agent.core.react.safety import (
    ExcessiveSameToolCondition,
    RepeatedActionsCondition,
    SafetyManager,
)


def _step(action, **inp):
    return ReActStep(step_number=0, thought="", action=action, action_input=inp, observation="0 results")


def _batch_step(*calls):
    """A step the model filled with N parallel tool_use blocks.

    Mirrors `_finalize_batch_step`: `tool_invocations` holds every call and the
    legacy `action`/`action_input` fields mirror the FIRST one only.
    """
    invocations = [
        ToolInvocation(tool_call_id=f"toolu_{i}", name=name, inputs=inputs, observation="ok")
        for i, (name, inputs) in enumerate(calls)
    ]
    return ReActStep(
        step_number=0,
        thought="",
        action=invocations[0].name,
        action_input=invocations[0].inputs,
        observation="ok",
        tool_invocations=invocations,
    )


def test_fires_on_same_tool_with_varying_args():
    cond = ExcessiveSameToolCondition(max_same_tool=8)
    # 8 search_memory calls, each a DIFFERENT query — RepeatedActions wouldn't catch these.
    steps = [_step("search_memory", q=f"query {i}") for i in range(8)]
    assert cond.should_stop(steps, current_step=8) is True
    # RepeatedActionsCondition (identical-args) does NOT catch them — proves the gap.
    assert RepeatedActionsCondition(max_repeats=3).should_stop(steps, current_step=8) is False


def test_under_cap_does_not_fire():
    cond = ExcessiveSameToolCondition(max_same_tool=8)
    steps = [_step("search_memory", q=f"query {i}") for i in range(5)]
    assert cond.should_stop(steps, current_step=5) is False


def test_mixed_tools_under_cap_do_not_fire():
    cond = ExcessiveSameToolCondition(max_same_tool=8)
    # 7 search_memory + 7 read_file interleaved — neither hits 8.
    steps = []
    for i in range(7):
        steps.append(_step("search_memory", q=str(i)))
        steps.append(_step("read_file", p=str(i)))
    assert cond.should_stop(steps, current_step=len(steps)) is False


def test_dispatch_assistant_is_exempt():
    cond = ExcessiveSameToolCondition(max_same_tool=8)
    # dispatch_assistant has its own DispatchCounter — must not trip this guard.
    steps = [_step("dispatch_assistant", handle=f"h{i}") for i in range(20)]
    assert cond.should_stop(steps, current_step=20) is False


def test_counts_every_call_in_a_parallel_batch_not_just_the_first():
    """The cap is expressed in CALLS; a batch step contributes all of them.

    Reading `step.action` (the first-invocation mirror) counted 2 turns here
    instead of 4 calls, so a cap of 4 could never fire on parallel batches.
    """
    cond = ExcessiveSameToolCondition(max_same_tool=4)
    steps = [
        _batch_step(("search_memory", {"q": "a"}), ("search_memory", {"q": "b"})),
        _batch_step(("search_memory", {"q": "c"}), ("search_memory", {"q": "d"})),
    ]
    assert cond.should_stop(steps, current_step=2) is True
    # Probe: the same four calls spread one-per-turn must fire too, so a True
    # here is about the count and not about batching per se.
    flat = [_step("search_memory", q=q) for q in "abcd"]
    assert cond.should_stop(flat, current_step=4) is True


def test_exemption_applies_to_batched_calls_too():
    cond = ExcessiveSameToolCondition(max_same_tool=4)
    steps = [
        _batch_step(("dispatch_assistant", {"h": "a"}), ("dispatch_assistant", {"h": "b"}))
        for _ in range(5)
    ]
    assert cond.should_stop(steps, current_step=5) is False


def test_production_schema_exploration_is_not_halted():
    """Regression: thread_tzYBJVGhXe2LaRVV9gjj5idL, 2026-08-04.

    11 LLM turns exploring an unfamiliar Postgres schema, 22 tool calls, every
    one successful — halted at turn 12 by this condition, discarding the answer
    the model had already assembled. Turns are reconstructed from the message's
    persisted `execution_timeline`, grouped by identical timestamp.
    """
    Q, D = "postgres_run_query", "postgres_describe_table"
    turns = [
        [(Q, "use_case cols"), (D, "workflow_agenttokenusage")],
        [(Q, "tables noderun"), (D, "assistant_assistant")],
        [(D, "workflow_agentnoderun")],
        [(Q, "agentrun tables"), (Q, "node_id counts")],
        [(D, "workflow_agentrun"), (D, "workflow_agentassistantnode")],
        [(Q, "use_case count"), (Q, "30d attribution")],
        [(Q, "agent_id counts"), (Q, "agentnode cols")],
        [(Q, "workflow_agent tables"), (Q, "workflow_agent cols")],
        [(Q, "usecase costs"), (Q, "30d totals"), (Q, "unattributed")],
        [(D, "finance_externalserviceusagerecord"), (D, "assistanttoolexecution")],
        [(Q, "usage_session counts"), (D, "finance_usagesession")],
    ]
    manager = SafetyManager()
    steps = []
    for i, turn in enumerate(turns, start=1):
        # Checked at the TOP of each iteration, as the orchestrator does.
        assert manager.should_stop(steps, i) is None, f"halted before turn {i}"
        steps.append(_batch_step(*[(tool, {"q": q}) for tool, q in turn]))
    assert manager.should_stop(steps, len(turns) + 1) is None

    # The guard must still bound this shape — it just has to let a real
    # investigation finish first. Five more turns of DISTINCT queries (so
    # RepeatedActionsCondition stays out of it and this asserts the cap) trip it.
    for i in range(5):
        steps.append(_batch_step((Q, {"q": f"more {i}"}), (Q, {"q": f"and more {i}"})))
    fired = manager.should_stop(steps, len(steps) + 1)
    assert isinstance(fired, ExcessiveSameToolCondition)
