"""An empty model turn is not an answer.

Drives the REAL ``_execute_reasoning_step_native`` on a stand-in self (same
pattern as test_clarification_short_circuit_integration) with a stream that
closes having emitted nothing.

Production fingerprint (2026-07-31): three consecutive google_ads_specialist
dispatches returned the generic "I wasn't able to produce a complete answer
from this run" fallback — the ``_generate_fallback_answer`` variant chosen
when the last step is NOT an error step — with no ``failure`` in the dispatch
envelope, i.e. no safety condition fired. That is the signature of the loop
breaking on ``is_final_step`` (``answer is not None``) after an empty turn:
``step.answer = ""`` ends the run, ``_build_result`` sees a falsy final
answer, and neither the user nor the dispatching parent learns anything. The
parent then burned its 3-per-handle dispatch budget and reported "Google Ads
data could not be retrieved this turn".

It also made ``EmptyResponseCondition`` unreachable: the condition counts
steps with no thought/action/answer, but the loop had already ended.

Production fingerprint (2026-08-02), the SEQUEL: the fix above landed and the
condition became reachable — the traces now say ``forced_stop: LLM returned 2
consecutive empty responses`` with a full ``failure`` envelope. The user-visible
outcome did not change, because labelling the stop and answering from the run's
work are different code paths. ``thread_Pc8FLSWfGlukiiTWQdrCt60p`` spent 1771s
across three ``google_ads_specialist`` dispatches; the first alone ran 738s and
completed six ``google_ads_query`` calls holding the entire requested dataset,
then returned the fixed apology because ``_generate_fallback_answer`` ignores
``steps``. So the second half of this file guards the other direction: a halted
run must REPORT what it retrieved, to the user and to its dispatching parent.
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.core.message import MessageRole
from miiflow_agent.core.react.models import ReActStep, ToolInvocation
from miiflow_agent.core.react.orchestrator import (
    ReActOrchestrator,
    _extract_partial_results,
)
from miiflow_agent.core.react.safety import EmptyResponseCondition


class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, e):
        self.events.append(e)


def _chunk(delta="", finish_reason=None):
    return SimpleNamespace(
        delta=delta,
        thinking_delta=None,
        tool_calls=None,
        finish_reason=finish_reason,
        usage=None,
        cost=0.0,
        tokens_used=0,
    )


def _drive(chunks):
    """Run the real reasoning step against a canned stream; return (step, ctx)."""

    async def go():
        async def stream_with_tools(messages=None, prebuilt_tools=None):
            for c in chunks:
                yield c

        orch = SimpleNamespace(
            event_bus=_FakeBus(),
            tool_executor=SimpleNamespace(
                _build_native_tool_schemas=lambda: [],
                stream_with_tools=stream_with_tools,
                agent=SimpleNamespace(temperature=0.0, max_tokens=8192),
            ),
        )
        # Bind the real error handler: a stand-in that swallowed failures
        # would let a broken harness masquerade as a passing assertion.
        orch._handle_step_error = ReActOrchestrator._handle_step_error.__get__(
            orch, SimpleNamespace
        )
        state = SimpleNamespace(
            current_step=3,
            needs_clarification=False,
            clarification_data=None,
            pending_llm_blocks=[],
            media_store={},
            steps=[],
        )
        context = SimpleNamespace(deps={}, messages=[])
        step = await ReActOrchestrator._execute_reasoning_step_native(
            orch, context, state
        )
        return step, context, orch.event_bus

    return asyncio.run(go())


class TestEmptyTurnDoesNotEndTheRun:
    def test_empty_stream_leaves_no_answer(self):
        step, _ctx, _bus = _drive([_chunk(delta="", finish_reason="stop")])

        # The load-bearing assertion: answer stays None, so is_final_step is
        # False and the main loop keeps going instead of falling through to
        # the contentless fallback.
        assert step.answer is None
        assert step.is_final_step is False
        assert step.error is None

    def test_whitespace_only_turn_is_treated_as_empty(self):
        step, _ctx, _bus = _drive([_chunk(delta="\n  \n", finish_reason="stop")])

        assert step.answer is None

    def test_nudge_is_appended_and_ends_on_a_user_turn(self):
        """Anthropic rejects a trailing assistant message as prefill, so the
        recovery message must be a USER turn — the same invariant the
        max_tokens branch maintains."""
        _step, ctx, _bus = _drive([_chunk(delta="", finish_reason="stop")])

        assert ctx.messages, "an empty turn must leave a nudge behind"
        assert ctx.messages[-1].role == MessageRole.USER
        assert "empty" in ctx.messages[-1].content.lower()
        # No empty assistant message added alongside it.
        assert not [
            m for m in ctx.messages if m.role == MessageRole.ASSISTANT
        ]

    def test_real_answer_still_ends_the_run(self):
        """The guard must not swallow ordinary answers."""
        step, ctx, _bus = _drive(
            [_chunk(delta="Spend was £405.77.", finish_reason="stop")]
        )

        assert step.answer == "Spend was £405.77."
        assert step.is_final_step is True
        assert ctx.messages[-1].role == MessageRole.ASSISTANT


def _step(number, invocations):
    step = ReActStep(step_number=number, thought="")
    step.tool_invocations = invocations
    return step


def _google_ads_run():
    """The 2026-08-02 shape: real pulls, one rejected query, then empty turns."""
    return [
        _step(
            1,
            [
                ToolInvocation(
                    name="google_ads_query",
                    inputs={"customer_id": "4447141884", "query": "SELECT ..."},
                    observation="{'results': [{'metrics': {'cost_micros': 1044.27}}]}",
                    observation_ref="agent_obs_YWiV6QkDmOXaCCYShe6eDDYn",
                    description="Pull account totals for June-July 2026",
                ),
                ToolInvocation(
                    name="google_ads_query",
                    inputs={"customer_id": "4447141884", "query": "SELECT ..."},
                    error="Tool 'google_ads_query' failed: API error: "
                    "PROHIBITED_SEGMENT_WITH_METRIC_IN_SELECT_OR_WHERE_CLAUSE",
                ),
            ],
        ),
        _step(
            2,
            [
                ToolInvocation(
                    name="google_ads_query",
                    inputs={"customer_id": "4447141884", "query": "SELECT ..."},
                    observation="{'results': [{'segments': "
                    "{'conversion_action_name': 'Paid subscription'}}]}",
                    observation_ref="agent_obs_paidsubs",
                )
            ],
        ),
        # The turns that killed the run: no thought, no action, no answer.
        ReActStep(step_number=3, thought=""),
        ReActStep(step_number=4, thought=""),
    ]


class TestPartialResultsSurviveAHaltedRun:
    def test_successful_invocations_are_extracted_with_their_refs(self):
        partial = _extract_partial_results(_google_ads_run())

        assert [e["tool"] for e in partial] == ["google_ads_query"] * 2
        assert [e.get("observation_ref") for e in partial] == [
            "agent_obs_YWiV6QkDmOXaCCYShe6eDDYn",
            "agent_obs_paidsubs",
        ]

    def test_the_failed_invocation_is_not_reported_as_work_done(self):
        """Telling a parent a rejected query succeeded is worse than silence."""
        partial = _extract_partial_results(_google_ads_run())

        assert len(partial) == 2
        assert not any("PROHIBITED_SEGMENT" in str(e) for e in partial)

    def test_soft_error_observations_are_excluded_too(self):
        """A tool that reports failure in its payload rather than via `error`."""
        steps = [
            _step(
                1,
                [
                    ToolInvocation(
                        name="google_ads_query",
                        observation="Tool execution failed: API error: nope",
                    )
                ],
            )
        ]

        assert _extract_partial_results(steps) == []

    def test_fallback_answer_names_the_work_instead_of_apologizing_only(self):
        orch = SimpleNamespace()
        answer = ReActOrchestrator._generate_fallback_answer(
            orch, _google_ads_run()
        )

        assert "agent_obs_YWiV6QkDmOXaCCYShe6eDDYn" in answer
        assert "Pull account totals for June-July 2026" in answer
        assert "read_observation" in answer

    def test_fallback_probe_a_run_with_no_successes_still_apologizes(self):
        """The negative case. Without this, the assertion above could pass on a
        method that always emitted the same boilerplate section."""
        steps = [
            _step(
                1,
                [ToolInvocation(name="google_ads_query", error="API error: nope")],
            )
        ]
        answer = ReActOrchestrator._generate_fallback_answer(SimpleNamespace(), steps)

        assert "read_observation" not in answer
        assert "wasn't able to finish" in answer

    def test_error_tail_branch_also_reports_the_work(self):
        """Both branches, not just the one that fired in production. The
        error-tail branch withholds raw tool-error TEXT; that was never a reason
        to withhold the results."""
        steps = _google_ads_run()
        steps[-1].error = "recovery exhausted"
        steps[-2].error = "recovery exhausted"

        answer = ReActOrchestrator._generate_fallback_answer(SimpleNamespace(), steps)

        assert "repeated issues" in answer
        assert "agent_obs_paidsubs" in answer
        # The raw tool error must not ride along.
        assert "PROHIBITED_SEGMENT" not in answer

    def test_results_are_bounded(self):
        many = [
            _step(
                i,
                [
                    ToolInvocation(
                        name="google_ads_query",
                        inputs={"query": "S" * 5000},
                        observation="x" * 10_000,
                    )
                ],
            )
            for i in range(30)
        ]
        partial = _extract_partial_results(many)

        assert len(partial) == 12
        assert all(len(e["excerpt"]) <= 400 for e in partial)
        assert all(len(e["inputs"]["query"]) < 400 for e in partial)


class TestEmptyResponseConditionIsNowReachable:
    def test_fires_once_the_budget_of_consecutive_empty_steps_is_spent(self):
        """Derived from the condition's own budget, not a pinned literal — the
        default is a tuning knob (raised 2 → 4 after 2026-08-02, since each
        empty turn gets a nudge and N allows only N-1 recoveries) and a test
        that hardcodes it fails on every retune without finding a defect."""
        condition = EmptyResponseCondition()
        budget = condition.max_empty_responses
        steps = [ReActStep(step_number=i, thought="") for i in range(1, budget + 1)]

        assert condition.should_stop(steps, current_step=budget) is True
        assert "empty" in condition.get_description().lower()

    def test_does_not_fire_while_retries_remain(self):
        condition = EmptyResponseCondition()
        one_short = [
            ReActStep(step_number=i, thought="")
            for i in range(1, condition.max_empty_responses)
        ]

        assert condition.should_stop(one_short, len(one_short)) is False

    def test_does_not_fire_once_the_model_answers(self):
        condition = EmptyResponseCondition()
        steps = [
            ReActStep(step_number=i, thought="")
            for i in range(1, condition.max_empty_responses)
        ]
        answered = ReActStep(step_number=condition.max_empty_responses, thought="")
        answered.answer = "Spend was £405.77."
        steps.append(answered)

        assert condition.should_stop(steps, len(steps)) is False
