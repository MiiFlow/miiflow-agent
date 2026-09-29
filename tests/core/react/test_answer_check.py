"""The host's final-answer check (`ANSWER_CHECK_DEP`).

Production thread_4ObeDb8ULVRugMLIfWdnELJe: asked to "generate documents", the
model made no tool call and answered "Both PDFs are generated" with two made-up
file ids. The loop accepted any text as the answer, so nothing could send it
back. Drives the REAL `_execute_reasoning_step_native` (same stand-in pattern as
test_empty_response_not_final).
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.artifacts import ANSWER_CHECK_DEP, MAX_ANSWER_CHECK_FAILURES
from miiflow_agent.core.message import MessageRole
from miiflow_agent.core.react.enums import ReActEventType
from miiflow_agent.core.react.orchestrator import ReActOrchestrator


class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, e):
        self.events.append(e)


def _chunk(delta="", finish_reason=None):
    return SimpleNamespace(
        delta=delta, thinking_delta=None, tool_calls=None, finish_reason=finish_reason,
        usage=None, cost=0.0, tokens_used=0,
    )


def _drive(answer, hook, *, failed_before=0):
    async def go():
        async def stream_with_tools(messages=None, prebuilt_tools=None):
            yield _chunk(delta=answer, finish_reason="stop")

        orch = SimpleNamespace(
            event_bus=_FakeBus(),
            tool_executor=SimpleNamespace(
                _build_native_tool_schemas=lambda: [],
                stream_with_tools=stream_with_tools,
                agent=SimpleNamespace(temperature=0.0, max_tokens=8192),
            ),
        )
        orch._handle_step_error = ReActOrchestrator._handle_step_error.__get__(orch, SimpleNamespace)
        state = SimpleNamespace(
            current_step=2, needs_clarification=False, clarification_data=None,
            pending_llm_blocks=[], media_store={}, steps=[], answer_checks_failed=failed_before,
        )
        deps = {ANSWER_CHECK_DEP: hook} if hook else {}
        context = SimpleNamespace(deps=deps, messages=[])
        step = await ReActOrchestrator._execute_reasoning_step_native(orch, context, state)
        return step, context, orch.event_bus, state

    return asyncio.run(go())


def _retractions(bus):
    return [e for e in bus.events if e.event_type == ReActEventType.ANSWER_RETRACTED]


class TestAnswerCheck:
    def test_a_refused_answer_is_retracted_and_the_run_continues(self):
        seen = []

        async def hook(answer, context):
            seen.append(answer)
            return "Your answer names file id x, but no such file exists."

        step, ctx, bus, state = _drive("Both PDFs are generated: [ARTIFACT:x]", hook)

        assert seen == ["Both PDFs are generated: [ARTIFACT:x]"]
        assert step.answer is None and step.is_final_step is False
        assert len(_retractions(bus)) == 1
        assert _retractions(bus)[0].data.get("reason") == "answer_check"
        # The model keeps its own words and gets the correction as the last turn.
        assert ctx.messages[-2].role == MessageRole.ASSISTANT
        assert ctx.messages[-1].role == MessageRole.USER
        assert "no such file exists" in ctx.messages[-1].content
        assert state.answer_checks_failed == 1

    def test_an_accepted_answer_ends_the_run(self):
        step, _ctx, bus, _state = _drive("Here it is: [ARTIFACT:real]", lambda answer, context: None)

        assert step.answer == "Here it is: [ARTIFACT:real]"
        assert not _retractions(bus)

    def test_the_check_is_bounded(self):
        """A check the model cannot satisfy ends in an answer, never a loop."""
        calls = []

        def hook(answer, context):
            calls.append(answer)
            return "still wrong"

        step, _ctx, bus, _state = _drive("x", hook, failed_before=MAX_ANSWER_CHECK_FAILURES)

        assert step.answer == "x"
        assert calls == []
        assert not _retractions(bus)

    def test_no_hook_changes_nothing(self):
        step, _ctx, _bus, _state = _drive("plain answer", None)
        assert step.answer == "plain answer"
