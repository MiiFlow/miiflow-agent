"""Turning a finished (or halted) run into what the caller receives.

Owns the terminal paths of a ReAct run: publishing the closing final-answer
event, building the success/crash results, the one tool-free wrap-up turn
after a safety halt (_answer_after_halt — report the work in hand instead of
discarding it), and the canned fallback when even that fails. Methods were
moved verbatim from ReActOrchestrator (which keeps thin delegates);
``self._orch`` is the orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, List

from ..message import Message, MessageRole
from .enums import StopReason
from .events import EventFactory
from .models import ReActResult, ReActStep
from .orchestrator import _extract_partial_results, _preview

if TYPE_CHECKING:
    from ..agent import RunContext
    from .execution import ExecutionState
    from .orchestrator import ReActOrchestrator

logger = logging.getLogger(__name__)


class AnswerSynthesis:
    """Terminal paths: results, halt wrap-up, fallback answer."""

    def __init__(self, orch: "ReActOrchestrator"):
        self._orch = orch

    async def publish_final_answer_event(
        self, step: ReActStep, state: "ExecutionState"
    ):
        """Publish the closing final_answer event for the complete answer.

        Answer chunks are streamed live as they arrive in the streaming loop;
        this single event signals completion and gives consumers like
        agent.run() the full answer string in one place.
        """
        if step.answer:
            await self._orch.event_bus.publish(
                EventFactory.final_answer(state.current_step, step.answer)
            )

    async def build_result(
        self, state: "ExecutionState", context: RunContext = None
    ) -> ReActResult:
        """Build successful result."""
        # Determine stop reason
        if state.needs_clarification:
            stop_reason = StopReason.NEEDS_CLARIFICATION
            # Don't generate fallback - we're waiting for user input
            state.final_answer = ""
        elif context is not None and context.is_cancelled:
            stop_reason = StopReason.USER_CANCELLED
        elif state.final_answer:
            stop_reason = StopReason.ANSWER_COMPLETE
        else:
            stop_reason = StopReason.FORCED_STOP
            state.final_answer = self._orch._generate_fallback_answer(state.steps)
            logger.warning(
                "[ORCH] FALLBACK answer generated (no step produced a final answer). steps=%d final_answer=%s",
                len(state.steps),
                _preview(state.final_answer, 300),
            )
            # Publish fallback as FINAL_ANSWER event so streaming_service captures it
            if state.final_answer:
                await self._orch.event_bus.publish(
                    EventFactory.final_answer(state.current_step, state.final_answer)
                )

        logger.info(
            "[ORCH] result stop_reason=%s steps=%d final_answer=%s",
            getattr(stop_reason, "value", stop_reason),
            len(state.steps),
            _preview(state.final_answer, 300),
        )

        # Calculate totals
        total_time = time.time() - state.start_time
        total_cost = sum(step.cost for step in state.steps)
        total_tokens = sum(step.tokens_used for step in state.steps)

        result = ReActResult(
            steps=state.steps,
            final_answer=state.final_answer,
            stop_reason=stop_reason,
            total_cost=total_cost,
            total_execution_time=total_time,
            total_tokens=total_tokens,
        )

        # Attach clarification data if present
        if state.clarification_data:
            result.clarification_data = state.clarification_data

        if context is not None:
            checkpoint = getattr(context, "checkpoint", None)
            interrupt = (
                checkpoint.active_interrupt()
                if checkpoint is not None and hasattr(checkpoint, "active_interrupt")
                else None
            )
            if interrupt is not None:
                result.metadata["pending_interrupt"] = interrupt.to_dict()

        # Carry the structured failure (set by ``_should_stop`` when a
        # safety condition halts the loop) so the dispatch envelope can
        # surface a real cause to the parent agent.
        if state.failure_metadata is not None:
            result.metadata["failure"] = state.failure_metadata

        # ...and its counterpart: what the run DID retrieve. A parent that
        # dispatched this child should be able to finish from the child's work
        # instead of re-dispatching into the same wall until its per-handle
        # budget is gone. Only attached when the run ended without an answer of
        # its own — a successful run's answer already carries its findings.
        if stop_reason is not StopReason.ANSWER_COMPLETE:
            partial = _extract_partial_results(state.steps)
            if partial:
                result.metadata["partial_results"] = partial

        return result

    def build_error_result(
        self, state: "ExecutionState", error: Exception
    ) -> ReActResult:
        """Build the result for a run that CRASHED (top-level exception).

        The final_answer is an error string for display, but callers must not
        have to parse it to learn that the run crashed rather than answered —
        ``metadata["error"]`` carries the machine-readable cause, and
        ``error_type`` preserves the typed classification (rate_limited,
        timeout_error, ...) when the exception was a MiiflowLLMError.
        """
        error_type = getattr(error, "error_type", None)
        return ReActResult(
            steps=state.steps,
            final_answer=f"Error occurred during execution: {str(error)}",
            stop_reason=StopReason.FORCED_STOP,
            metadata={
                "error": {
                    "crashed": True,
                    "exception": type(error).__name__,
                    "message": str(error)[:500],
                    "error_type": getattr(error_type, "value", None),
                }
            },
        )

    async def answer_after_halt(
        self, context: RunContext, state: "ExecutionState"
    ) -> bool:
        """One tool-free LLM turn, so a safety halt reports instead of discarding.

        A safety condition halting the loop says "stop doing work", NOT "throw the
        work away" — but that is what happened: the loop broke with no
        `final_answer`, `_build_result` fell through to `_generate_fallback_answer`,
        and the user got a canned apology. Production 2026-08-04
        (thread_tzYBJVGhXe2LaRVV9gjj5idL): 11 turns, 22 successful tool calls, the
        answer already assembled in the transcript, ~100s of user wait — all
        replaced by "I wasn't able to produce a complete answer from this run."
        The same is true of the error-shaped halts: the model explaining WHICH
        tool failed and what would unblock it beats a generic apology, which is
        what the platform prompt's Rule 17 asks for and could never deliver while
        the string was generated in Python.

        Tools are withheld by passing an EMPTY schema list — every provider client
        here skips the `tools` key when it is falsy — rather than by asking nicely
        or by sending `tool_choice: "none"`, whose spelling differs per provider.
        The model therefore *cannot* call anything, which is also why the deltas
        stream unconditionally: no tool_use block can arrive to retract them.

        Cost: an empty tools array is a different array, so this one call re-bills
        the prompt uncached (~$0.2 at a 64k-token context on Sonnet). It buys back
        a turn that was otherwise 100% waste, and only ever runs on a halt.

        Returns True when it produced an answer. On any failure the caller still
        has `_generate_fallback_answer`, so this degrades to the old behaviour.
        """
        if context is None or state.final_answer or not state.steps:
            return False
        if state.needs_clarification or context.is_cancelled:
            return False

        reason = state.halt_description or "a safety limit was reached"
        context.messages.append(
            Message(
                role=MessageRole.USER,
                content=(
                    f"SYSTEM: This run has been halted — {reason}. No further tool "
                    "calls are possible; none are available to you on this turn.\n\n"
                    "Write your final answer NOW, using only what is already in "
                    "this conversation. Report everything you did establish, with "
                    "the concrete numbers and findings from the tool results above "
                    "— partial results are valuable and must not be withheld. Then "
                    "state plainly what you could not finish and what would let you "
                    "finish it. Do not apologise generically, do not claim you were "
                    "unable to produce an answer, and do not ask to try again."
                ),
            )
        )

        step = ReActStep(step_number=state.current_step, thought="")
        started = time.time()
        buffer = ""
        try:
            await self._orch.event_bus.publish(
                EventFactory.step_started(state.current_step)
            )
            async for chunk in self._orch.tool_executor.stream_with_tools(
                messages=context.messages, prebuilt_tools=[]
            ):
                if getattr(chunk, "thinking_delta", None):
                    await self._orch.event_bus.publish(
                        EventFactory.thinking_chunk(
                            state.current_step, chunk.thinking_delta, buffer
                        )
                    )
                if chunk.delta:
                    buffer += chunk.delta
                    await self._orch.event_bus.publish(
                        EventFactory.final_answer_chunk(
                            state.current_step, chunk.delta, buffer
                        )
                    )
                if getattr(chunk, "usage", None):
                    # Assigned, not accumulated — usage chunks carry the running
                    # total for the call, as `_execute_reasoning_step_native` does.
                    step.tokens_used = chunk.usage.total_tokens or 0
        except asyncio.CancelledError:
            # Cancellation is not a failed wrap-up: it must reach the caller,
            # or a cancelled turn looks like one that merely had nothing to say.
            raise
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad and deliberately terminal: only result-building
            # remains after this, so absorbing even a Celery soft-time-limit
            # here costs milliseconds rather than the whole answer.
            logger.warning(
                "[ORCH] wrap-up turn after halt (%s) failed: %s — falling back to "
                "the canned answer",
                reason,
                exc,
            )
            return False

        answer = buffer.strip()
        # Log the OUTCOME, not just the exception path: "the wrap-up ran and the
        # model said nothing" and "the wrap-up never ran" are different bugs and
        # must not look identical in the logs.
        if not answer:
            logger.warning(
                "[ORCH] wrap-up turn after halt (%s) returned empty text; "
                "falling back to the canned answer",
                reason,
            )
            return False

        step.answer = answer
        step.execution_time = time.time() - started
        state.steps.append(step)
        state.final_answer = answer
        logger.info(
            "[ORCH] wrap-up turn after halt (%s) produced an answer: %s",
            reason,
            _preview(answer, 300),
        )
        await self._orch.event_bus.publish(
            EventFactory.final_answer(state.current_step, answer)
        )
        return True

    def generate_fallback_answer(self, steps) -> str:
        """Compose an answer for a run that stopped without producing one.

        A run that force-stops has almost always DONE something first — the
        2026-08-02 incident lost six completed `google_ads_query` calls holding
        the whole requested dataset because this method returned a constant. So
        the work comes first and the apology second, on BOTH branches: the
        error-tail case leaks raw tool-error text if you let it, but that is an
        argument for withholding the *error*, never for withholding the results.
        The structured cause travels separately in `result.metadata["failure"]`.
        """
        if not steps:
            return "No reasoning steps were completed."

        last_step = steps[-1]

        # The "halted on consecutive tool errors" case. The last observation is
        # a raw tool-execution error string; surfacing it exposes internals like
        # parameter names and schema mismatches, so the apology stays generic.
        recent_errors = [s for s in steps[-3:] if getattr(s, "is_error_step", False)]
        if recent_errors and getattr(last_step, "is_error_step", False):
            apology = (
                "I ran into repeated issues while trying to fulfill this request "
                "and wasn't able to finish it. Please try rephrasing your "
                "question, or try again in a moment."
            )
        else:
            apology = (
                "I wasn't able to finish this run. Please try again, or narrow "
                "the request if it involves a lot of work."
            )

        partial = _extract_partial_results(steps)
        if not partial:
            return apology

        lines = [
            apology,
            "",
            "Before stopping I did complete the following, so this work does not "
            "need to be redone:",
        ]
        for entry in partial:
            label = entry.get("description") or entry["tool"]
            ref = entry.get("observation_ref")
            suffix = f' — read_observation(ref="{ref}") for the full result' if ref else ""
            lines.append(f"- {label}{suffix}")
        return "\n".join(lines)
