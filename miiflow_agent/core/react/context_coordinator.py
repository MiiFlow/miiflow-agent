"""Context-engine coordination for the ReAct loop.

The orchestrator decides WHEN the context engine runs (before every LLM
call, on overflow recovery, on usage arrival); this collaborator owns HOW —
building the request shape from the same schema list that goes on the wire,
interpreting verdicts, applying compacted messages, and feeding real usage
back for calibration. Methods were moved verbatim from ReActOrchestrator
(which keeps thin delegates); ``self._orch`` is the orchestrator.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from ..context import CompressionVerdict
from .events import EventFactory

if TYPE_CHECKING:
    from ..agent import RunContext
    from .execution import ExecutionState
    from .orchestrator import ReActOrchestrator

logger = logging.getLogger(__name__)


class ContextCoordinator:
    """How compaction/calibration happens; the loop decides when."""

    def __init__(self, orch: "ReActOrchestrator"):
        self._orch = orch

    def reconcile_context_usage(self, usage, state: "ExecutionState") -> None:
        """Hand the provider's real usage to the context engine.

        Consumes ``state.last_estimated_prompt_tokens`` — the uncorrected
        estimate for the request that produced this usage — and clears it, so
        a later call can never be calibrated against a stale estimate from an
        earlier, differently-shaped request.
        """
        engine = self._orch.context_compressor
        if engine is None or not hasattr(engine, "update_from_response"):
            return
        estimated = state.last_estimated_prompt_tokens
        state.last_estimated_prompt_tokens = None
        try:
            engine.update_from_response(usage, estimated_prompt_tokens=estimated)
        except Exception as exc:  # noqa: BLE001 — telemetry must not fail a run
            logger.debug("[ORCH] context usage reconciliation failed: %s", exc)

    async def compress_for_recovery(
        self, context: RunContext, overflow: bool = False
    ) -> bool:
        """Forced compaction after a provider-confirmed overflow.

        Injected into the RecoveryManager as ``compress_fn``. Unlike
        ``_maybe_compress`` this does not consult ``should_compress``: the
        provider just rejected the request as too large, and that verdict
        outranks any local estimate (which has, by definition, just been
        proven wrong). ``overflow=True`` additionally teaches the engine the
        real window (observe_overflow), so the trigger self-corrects instead
        of re-hitting the same wall every turn. Returns True if the message
        history actually shrank.
        """
        engine = self._orch.context_compressor
        if engine is None:
            return False

        # Legacy compressor: message-only sizing, no notion of force.
        if not hasattr(engine, "should_compress"):
            result = await engine.compress_if_needed(
                context.messages, preserve_recent=6
            )
            if result.was_compressed:
                context.messages = result.messages
                logger.info(
                    "[ORCH] recovery context compression: %d -> %d messages",
                    result.original_count,
                    result.compressed_count,
                )
            return result.was_compressed

        try:
            shape = self._orch.tool_executor.build_request_shape(context.messages)
        except Exception as exc:  # noqa: BLE001 — recovery must not raise
            logger.warning(
                "[ORCH] recovery compaction could not build request shape: %s", exc
            )
            return False

        if overflow and hasattr(engine, "observe_overflow"):
            try:
                engine.observe_overflow(shape)
            except Exception as exc:  # noqa: BLE001 — learning is best-effort
                logger.debug("[ORCH] observe_overflow failed: %s", exc)

        try:
            # Older/external engines may predate the ``force`` keyword; fall
            # back to a plain pass rather than failing the recovery attempt.
            if "force" in inspect.signature(engine.compress).parameters:
                outcome = await engine.compress(shape, force=True)
            else:
                outcome = await engine.compress(shape)
        except Exception as exc:  # noqa: BLE001 — recovery must not raise
            logger.warning("[ORCH] recovery compaction failed: %s", exc)
            return False

        if outcome.was_compressed:
            context.messages = outcome.shape.messages
            logger.info(
                "[ORCH] recovery context compression: %d -> %d messages, "
                "%d -> %d tokens (%s)",
                outcome.messages_before,
                outcome.messages_after,
                outcome.tokens_before,
                outcome.tokens_after,
                outcome.reason,
            )
        return outcome.was_compressed

    async def maybe_compress(
        self, context: RunContext, phase: str, state: "ExecutionState" = None
    ) -> None:
        """Run the context engine over the request that is about to go out.

        Accepts either a ``ContextEngine`` or the legacy ``ContextCompressor``
        in ``self._orch.context_compressor``. The legacy path is kept because
        adapters and tests construct one directly; it takes messages only, so
        it stays blind to the tool schemas and behaves exactly as before.
        Anything constructed through ``core.context`` gets the shape-aware
        path.
        """
        engine = self._orch.context_compressor
        if engine is None:
            return

        # Legacy compressor: no shape, no verdicts, message-only sizing.
        if not hasattr(engine, "should_compress"):
            result = await engine.compress_if_needed(context.messages)
            if result.was_compressed:
                context.messages = result.messages
                logger.info(
                    "[ORCH] %s context compression: %d -> %d messages",
                    phase,
                    result.original_count,
                    result.compressed_count,
                )
            return

        # Build the shape from the SAME schema list the next call will send.
        # Sizing a freshly-rebuilt list risks measuring a different tool set
        # than the one that actually goes on the wire.
        try:
            shape = self._orch.tool_executor.build_request_shape(context.messages)
        except Exception as exc:  # noqa: BLE001 — sizing must not fail a run
            logger.warning("[ORCH] could not build request shape: %s", exc)
            return

        decision = engine.should_compress(shape)
        step_number = state.current_step if state is not None else 0
        await self._orch.event_bus.publish(
            EventFactory.context_breakdown(decision.to_dict(), step_number)
        )

        def record_estimate(sized_shape) -> None:
            """Stash the uncorrected estimate for the request we're about to
            send, so the next response's usage can calibrate against it."""
            if state is None:
                return
            try:
                from ..context import get_counter

                counter = get_counter(sized_shape.provider, sized_shape.model)
                state.last_estimated_prompt_tokens = counter.raw_total(sized_shape)
            except Exception:  # noqa: BLE001 — calibration is best-effort
                state.last_estimated_prompt_tokens = None

        if decision.verdict is CompressionVerdict.FLOOR_EXCEEDED:
            # Surfaced rather than swallowed: the actionable fix is upstream
            # (fewer tools, shorter system prompt, bigger window), and the run
            # is about to fail or degrade in a way that looks like a model
            # problem unless we say otherwise.
            logger.error("[ORCH] %s %s", phase, decision.reason)
            record_estimate(shape)
            return

        if not decision.should_compress:
            record_estimate(shape)
            return

        outcome = await engine.compress(shape)
        if outcome.was_compressed:
            context.messages = outcome.shape.messages
            logger.info(
                "[ORCH] %s context compression: %d -> %d messages, "
                "%d -> %d tokens (%s)",
                phase,
                outcome.messages_before,
                outcome.messages_after,
                outcome.tokens_before,
                outcome.tokens_after,
                outcome.reason,
            )
        # Calibrate against the shape actually sent — the compacted one.
        record_estimate(outcome.shape)
