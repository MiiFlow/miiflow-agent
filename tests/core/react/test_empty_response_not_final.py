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
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.core.message import MessageRole
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.react.orchestrator import ReActOrchestrator
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


class TestEmptyResponseConditionIsNowReachable:
    def test_fires_after_two_consecutive_empty_steps(self):
        condition = EmptyResponseCondition()
        steps = [ReActStep(step_number=1, thought=""), ReActStep(step_number=2, thought="")]

        assert condition.should_stop(steps, current_step=2) is True
        assert "empty" in condition.get_description().lower()

    def test_does_not_fire_once_the_model_answers(self):
        condition = EmptyResponseCondition()
        answered = ReActStep(step_number=2, thought="")
        answered.answer = "Spend was £405.77."

        assert (
            condition.should_stop(
                [ReActStep(step_number=1, thought=""), answered], 2
            )
            is False
        )
