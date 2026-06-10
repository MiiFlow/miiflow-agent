"""ExcessiveSameToolCondition — caps a single tool's per-turn call count.

Guards the failure mode where a model flails (e.g. ~20 search_memory calls with slightly
varied queries, all fruitless) — which RepeatedActionsCondition misses because the args
differ — burning the turn and then echoing the last empty tool result as its answer.
"""

from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.react.safety import (
    ExcessiveSameToolCondition,
    RepeatedActionsCondition,
)


def _step(action, **inp):
    return ReActStep(step_number=0, thought="", action=action, action_input=inp, observation="0 results")


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
