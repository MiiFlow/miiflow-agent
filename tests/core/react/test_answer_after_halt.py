"""A safety halt must report the work in hand, not discard it.

Drives the REAL ``_answer_after_halt`` on a stand-in self (same pattern as
test_empty_response_not_final).

Production fingerprint (2026-08-04, thread_tzYBJVGhXe2LaRVV9gjj5idL): 11 LLM
turns and 22 successful tool calls exploring a Postgres schema, the answer
already assembled in the transcript, halted by ExcessiveSameToolCondition at
turn 12. The loop broke with no ``final_answer``, ``_build_result`` fell
through to ``_generate_fallback_answer``, and the user received "I wasn't able
to produce a complete answer from this run." — 100s of wait and every tool
result thrown away.
"""

import asyncio
from types import SimpleNamespace

import pytest

from miiflow_agent.core.message import MessageRole
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.react.orchestrator import ReActOrchestrator


class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, e):
        self.events.append(e)


def _chunk(delta="", usage=None):
    return SimpleNamespace(
        delta=delta,
        thinking_delta=None,
        tool_calls=None,
        finish_reason=None,
        usage=usage,
        cost=0.0,
        tokens_used=0,
    )


def _run(chunks, *, steps=None, final_answer=None, cancelled=False, raises=None):
    """Run the real wrap-up against a canned stream.

    Returns (produced, state, context, bus, calls) where `calls` records the
    kwargs each stream_with_tools invocation received — that is how the
    tool-free property is asserted rather than assumed.
    """
    calls = []

    async def go():
        async def stream_with_tools(messages=None, prebuilt_tools=None):
            calls.append({"messages": messages, "prebuilt_tools": prebuilt_tools})
            if raises is not None:
                raise raises
            for c in chunks:
                yield c

        orch = SimpleNamespace(
            event_bus=_FakeBus(),
            tool_executor=SimpleNamespace(
                stream_with_tools=stream_with_tools,
                agent=SimpleNamespace(temperature=0.0, max_tokens=8192),
            ),
        )
        state = SimpleNamespace(
            current_step=12,
            final_answer=final_answer,
            steps=[ReActStep(step_number=1, thought="")] if steps is None else steps,
            needs_clarification=False,
            halt_description="Called one tool 20+ times in a single turn",
        )
        context = SimpleNamespace(messages=[], is_cancelled=cancelled)
        produced = await ReActOrchestrator._answer_after_halt(orch, context, state)
        return produced, state, context, orch.event_bus, calls

    return asyncio.run(go())


class TestWrapUpProducesTheAnswer:
    def test_streamed_text_becomes_the_final_answer(self):
        produced, state, _ctx, bus, _calls = _run(
            [_chunk("8 orgs onboarded, "), _chunk("total 30d cost $2,263.71.")]
        )

        assert produced is True
        assert state.final_answer == "8 orgs onboarded, total 30d cost $2,263.71."
        # Recorded as a step so cost/token totals in _build_result include it.
        assert state.steps[-1].answer == state.final_answer

    def test_the_wrap_up_turn_carries_no_tools(self):
        """The model must be UNABLE to call a tool, not merely asked not to.

        An empty list is how every provider client here drops the `tools` key;
        `tool_choice: "none"` is spelled differently per provider.
        """
        _produced, _state, _ctx, _bus, calls = _run([_chunk("done")])

        assert len(calls) == 1
        assert calls[0]["prebuilt_tools"] == []

    def test_guidance_is_a_user_turn_naming_the_halt(self):
        _produced, _state, ctx, _bus, _calls = _run([_chunk("done")])

        assert ctx.messages[-1].role == MessageRole.USER
        content = ctx.messages[-1].content
        assert "Called one tool 20+ times" in content
        assert "halted" in content.lower()

    def test_answer_is_streamed_and_closed(self):
        _produced, _state, _ctx, bus, _calls = _run([_chunk("a"), _chunk("b")])

        kinds = [e.event_type for e in bus.events]
        assert kinds.count(ReActEventType.FINAL_ANSWER_CHUNK) == 2
        assert kinds[-1] == ReActEventType.FINAL_ANSWER

    def test_usage_is_assigned_not_summed(self):
        """Usage chunks carry the running total for the call; adding them up
        would over-report the wrap-up's tokens."""
        _p, state, _ctx, _bus, _calls = _run(
            [
                _chunk("a", usage=SimpleNamespace(total_tokens=100)),
                _chunk("b", usage=SimpleNamespace(total_tokens=140)),
            ]
        )

        assert state.steps[-1].tokens_used == 140


class TestWrapUpDegradesToTheCannedFallback:
    def test_empty_stream_does_not_invent_an_answer(self):
        produced, state, _ctx, _bus, _calls = _run([_chunk("   \n ")])

        assert produced is False
        assert not state.final_answer

    def test_provider_failure_is_absorbed(self):
        produced, state, _ctx, _bus, _calls = _run([], raises=RuntimeError("503"))

        assert produced is False
        assert not state.final_answer

    def test_cancellation_propagates(self):
        """A cancelled turn must not look like one that had nothing to say."""
        with pytest.raises(asyncio.CancelledError):
            _run([], raises=asyncio.CancelledError())


class TestWrapUpDoesNotRun:
    def test_skipped_when_an_answer_already_exists(self):
        produced, state, _ctx, _bus, calls = _run(
            [_chunk("overwrite me")], final_answer="the real answer"
        )

        assert produced is False
        assert state.final_answer == "the real answer"
        assert calls == [], "must not spend an LLM call when the run answered"

    def test_skipped_when_there_is_nothing_to_summarize(self):
        produced, _state, _ctx, _bus, calls = _run([_chunk("hi")], steps=[])

        assert produced is False
        assert calls == []

    def test_skipped_when_the_user_cancelled(self):
        produced, _state, _ctx, _bus, calls = _run([_chunk("hi")], cancelled=True)

        assert produced is False
        assert calls == []

    def test_no_context_is_a_no_op_not_a_crash(self):
        """`context` is Optional on the surrounding call path; appending the
        guidance message to None would turn a halt into an AttributeError."""

        async def go():
            orch = SimpleNamespace(event_bus=_FakeBus(), tool_executor=None)
            state = SimpleNamespace(
                current_step=12,
                final_answer=None,
                steps=[ReActStep(step_number=1, thought="")],
                needs_clarification=False,
                halt_description="whatever",
            )
            return await ReActOrchestrator._answer_after_halt(orch, None, state)

        assert asyncio.run(go()) is False
