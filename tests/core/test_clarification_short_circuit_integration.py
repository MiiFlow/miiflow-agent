"""End-to-end behaviour test for the clarification established-facts short-circuit (R4).

Drives the REAL orchestrator method ``_handle_tool_action`` on a stand-in self (same
pattern as test_approval_resume_execution) so the actual wiring runs: a clarification
ToolResult flows through the real clarification-detection block, reads
``context.deps["established_facts"]``, runs ``decide_clarification``, and either pauses or
short-circuits. This proves the behaviour without a live LLM — no manual staging needed.
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.orchestrator import ReActOrchestrator
from miiflow_agent.core.tools.clarification import CLARIFICATION_MARKER


class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, e):
        self.events.append(e)


def _standin_orch(execute_tool):
    """Stand-in self for _handle_tool_action with the real helper methods bound."""
    orch = SimpleNamespace(
        tool_executor=SimpleNamespace(has_tool=lambda n: True),
        event_bus=_FakeBus(),
        _find_similar_tool=lambda n: None,
        _execute_tool=execute_tool,
    )
    for helper in (
        "_handle_tool_approval_marker_result",
        "_record_tool_ledger_entry",
        "_record_interrupt",
    ):
        setattr(
            orch, helper, getattr(ReActOrchestrator, helper).__get__(orch, SimpleNamespace)
        )
    return orch


def _clarification_result(questions):
    return SimpleNamespace(
        name="ask_user_clarification",
        success=True,
        output={"marker": CLARIFICATION_MARKER, "questions": questions, "context": "need info"},
        cost=0.0,
        execution_time=0.0,
    )


def _drive(established_facts, questions, clarification_round=0):
    """Run the real _handle_tool_action with a stubbed tool execution; return (state, bus)."""

    async def go():
        result = _clarification_result(questions)

        async def _execute_tool(step, context, state):
            return result

        orch = _standin_orch(_execute_tool)
        step = SimpleNamespace(
            action="ask_user_clarification",
            action_input={"questions": questions},
            observation="",
            cost=0.0,
            execution_time=0.0,
        )
        state = SimpleNamespace(
            current_step=1,
            needs_clarification=False,
            clarification_data=None,
            pending_llm_blocks=[],
            media_store={},
        )
        context = SimpleNamespace(
            deps={
                "established_facts": established_facts,
                "clarification_round": clarification_round,
            },
            messages=[],
        )
        await ReActOrchestrator._handle_tool_action(
            orch, step, context, state, tool_call_id="tc_1"
        )
        return state, step, orch.event_bus

    return asyncio.run(go())


def _clar_events(bus):
    return [e for e in bus.events if e.event_type == ReActEventType.CLARIFICATION_NEEDED]


def test_short_circuits_when_question_already_settled():
    facts = [{"key": "daily_budget", "answer": "$50/day", "question_text": "What daily budget?"}]
    state, step, bus = _drive(
        facts, [{"key": "daily_budget", "question": "What daily budget?", "options": ["$25", "$50"]}]
    )
    # No pause, and the model is handed the known answer instead of re-asking.
    assert state.needs_clarification is False
    assert "$50/day" in step.observation
    assert _clar_events(bus) == []  # never surfaced to the user


def test_pauses_when_no_matching_fact():
    state, step, bus = _drive(
        [], [{"key": "geo", "question": "Which geos?", "options": ["CA", "TX"]}]
    )
    assert state.needs_clarification is True
    events = _clar_events(bus)
    assert len(events) == 1
    assert [q["key"] for q in events[0].data["questions"]] == ["geo"]


def test_pauses_only_on_unresolved_subset():
    facts = [{"key": "daily_budget", "answer": "$50/day"}]
    state, step, bus = _drive(
        facts,
        [
            {"key": "daily_budget", "question": "What daily budget?", "options": ["$25", "$50"]},
            {"key": "geo", "question": "Which geos?", "options": ["CA", "TX"]},
        ],
    )
    assert state.needs_clarification is True
    events = _clar_events(bus)
    assert len(events) == 1
    # Only the genuinely-new question reaches the user; the settled one is dropped.
    assert [q["key"] for q in events[0].data["questions"]] == ["geo"]


def test_circuit_breaker_forces_proceed_after_too_many_rounds():
    # Genuinely-unresolved question, but the consecutive-round count has hit the cap:
    # the orchestrator stops pausing and forces the model to proceed (content-free).
    state, step, bus = _drive(
        [],
        [{"key": "geo", "question": "Which geos?", "options": ["CA", "TX"]}],
        clarification_round=5,
    )
    assert state.needs_clarification is False
    assert "Do NOT ask again" in step.observation
    assert _clar_events(bus) == []


def test_legacy_path_unchanged_when_no_facts_in_deps():
    # deps without established_facts ⇒ byte-for-byte legacy: full set pauses.
    async def go():
        result = _clarification_result(
            [{"key": "a", "question": "A?", "options": ["x"]}, {"key": "b", "question": "B?", "options": ["y"]}]
        )

        async def _execute_tool(step, context, state):
            return result

        orch = _standin_orch(_execute_tool)
        step = SimpleNamespace(action="ask_user_clarification", action_input={}, observation="", cost=0.0, execution_time=0.0)
        state = SimpleNamespace(current_step=1, needs_clarification=False, clarification_data=None, pending_llm_blocks=[], media_store={})
        context = SimpleNamespace(deps={}, messages=[])  # no established_facts key
        await ReActOrchestrator._handle_tool_action(orch, step, context, state, tool_call_id="tc_1")
        return state, orch.event_bus

    state, bus = asyncio.run(go())
    assert state.needs_clarification is True
    events = _clar_events(bus)
    assert len(events) == 1
    assert [q["key"] for q in events[0].data["questions"]] == ["a", "b"]
