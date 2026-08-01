"""The built-in context engine: threshold-triggered conversation compaction.

Behaviourally this is the compressor that shipped before, with three defects
fixed:

  1. **It can see the whole request.** The old ``compress_if_needed(messages)``
     was blind to the system prompt and tool schemas, so on a tool-heavy
     assistant it was ignoring the largest tier. The budget handed to
     truncation is now ``threshold - floor``, not ``threshold``.

  2. **The window is per-model.** ``max_context_tokens or 128000`` compacted at
     96K on a 1M-token model. Resolved from the model registry instead.

  3. **It knows when compaction cannot help.** If system + tools alone exceed
     the threshold, no amount of conversation compaction brings the request
     under it. That case now returns ``FLOOR_EXCEEDED`` instead of burning a
     summarization call per turn, forever.

The message-rewriting algorithms themselves are unchanged and still live in
``core.context_compression`` — the tool_use/tool_result grouping and the
oversized-message clamp encode real production incidents, and there was no
reason to re-derive them here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from ..context_compression import (
    CompressionStrategy,
    ContextCompressor,
)
from ..message import Message
from ..metrics import TokenCount
from .budget import DEFAULT_COMPRESSION_THRESHOLD, ContextBudget
from .engine import (
    CompressionDecision,
    CompressionOutcome,
    CompressionVerdict,
    ContextEngine,
    EngineStats,
)
from .shape import RequestShape, TokenBreakdown
from .tokens import calibrator, get_counter

logger = logging.getLogger(__name__)


#: How many consecutive ineffective compactions before the engine latches off.
#: One is not enough — a single turn can append a huge tool result right after
#: compaction and legitimately push the request back over. Two consecutive
#: means the strategy genuinely isn't working.
_INEFFECTIVE_LATCH_THRESHOLD = 2


class DefaultContextEngine(ContextEngine):
    """Threshold-triggered compaction with floor awareness and calibration."""

    name = "compressor"

    def __init__(
        self,
        client=None,
        max_context_tokens: Optional[int] = None,
        compression_threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
        strategy: CompressionStrategy = CompressionStrategy.AUTO,
        preserve_recent: int = 4,
    ):
        self.client = client
        self._explicit_window = max_context_tokens
        self._threshold_ratio = compression_threshold
        self._strategy = strategy
        self._preserve_recent = preserve_recent

        self._stats = EngineStats()

        # Anti-thrash state. `_awaiting_verdict` means we compacted and have
        # not yet seen a real provider count for the result, so the next
        # `update_from_response` is the one that judges whether it worked.
        self._awaiting_verdict = False
        self._consecutive_ineffective = 0
        self._latched_off = False

        # Set on the first shape we see, so `update_from_response` can attribute
        # usage to the right (provider, model) without the caller threading it.
        self._provider: Optional[str] = None
        self._model: Optional[str] = None

    # -- budgeting -------------------------------------------------------

    def _budget_for(self, shape: RequestShape) -> ContextBudget:
        return ContextBudget.resolve(
            provider=shape.provider or self._provider,
            model=shape.model or self._model,
            max_context_tokens=self._explicit_window,
            threshold_ratio=self._threshold_ratio,
        )

    def _counter(self, shape: RequestShape):
        return get_counter(
            shape.provider or self._provider, shape.model or self._model
        )

    def breakdown(self, shape: RequestShape) -> TokenBreakdown:
        return self._counter(shape).breakdown(shape)

    # -- the decision ----------------------------------------------------

    def should_compress(self, shape: RequestShape) -> CompressionDecision:
        self._remember_identity(shape)
        budget = self._budget_for(shape)
        breakdown = self.breakdown(shape)

        def decide(verdict: CompressionVerdict, reason: str) -> CompressionDecision:
            return CompressionDecision(
                verdict=verdict, breakdown=breakdown, budget=budget, reason=reason
            )

        if self._latched_off:
            return decide(
                CompressionVerdict.INEFFECTIVE,
                f"compaction latched off after {self._consecutive_ineffective} "
                "consecutive ineffective passes",
            )

        if breakdown.total <= budget.threshold:
            return decide(
                CompressionVerdict.NOT_NEEDED,
                f"{breakdown.total} tokens under threshold {budget.threshold}",
            )

        # The request is over budget. The question is whether the conversation
        # is the reason — compaction only ever touches messages, so if the
        # floor alone is already over the line, compacting is provably wasted
        # work. Reporting this as a distinct verdict is what stops the loop.
        if breakdown.floor >= budget.threshold:
            self._stats.floor_exceeded_events += 1
            logger.warning(
                "[CONTEXT] floor exceeds threshold — compaction cannot help. "
                "system=%d tools=%d floor=%d threshold=%d window=%d (%s). "
                "Reduce the tool surface or raise the window.",
                breakdown.system,
                breakdown.tools,
                breakdown.floor,
                budget.threshold,
                budget.window,
                budget.source,
            )
            return decide(
                CompressionVerdict.FLOOR_EXCEEDED,
                f"system+tools ({breakdown.floor}) alone exceed threshold "
                f"({budget.threshold}); compaction cannot bring this under",
            )

        return decide(
            CompressionVerdict.COMPRESS,
            f"{breakdown.total} tokens exceeds threshold {budget.threshold} "
            f"(floor {breakdown.floor}, conversation {breakdown.messages})",
        )

    # -- compaction ------------------------------------------------------

    async def compress(self, shape: RequestShape) -> CompressionOutcome:
        budget = self._budget_for(shape)
        counter = self._counter(shape)
        before = counter.breakdown(shape)

        # This subtraction is the whole point of the refactor. Truncation gets
        # the budget available *to the conversation*, which is what's left
        # after the incompressible floor — not the full threshold, which is
        # what the old code passed and why compaction under-delivered on
        # tool-heavy assistants.
        message_budget = max(0, budget.threshold - before.floor)

        compressor = ContextCompressor(
            client=self.client,
            max_context_tokens=message_budget,
            # The budget is already the exact ceiling for messages, so the
            # inner compressor must not scale it down again.
            compression_threshold=1.0,
            strategy=self._strategy,
            # Exclude SYSTEM-role messages: the floor already accounts for
            # them, so counting them again inside the message budget would
            # charge the conversation twice for the system prompt and
            # over-truncate by exactly its size.
            token_fn=lambda msgs: counter.count_messages(msgs, include_system=False),
        )

        result = await compressor.compress_if_needed(
            shape.messages, preserve_recent=self._preserve_recent
        )

        compacted = shape.with_messages(result.messages)
        after = counter.breakdown(compacted)

        if result.was_compressed:
            self._stats.compressions += 1
            self._awaiting_verdict = True
            logger.info(
                "[CONTEXT] compacted %d→%d messages, %d→%d tokens "
                "(floor %d, message budget %d, threshold %d)",
                result.original_count,
                result.compressed_count,
                before.total,
                after.total,
                before.floor,
                message_budget,
                budget.threshold,
            )

        return CompressionOutcome(
            shape=compacted,
            was_compressed=result.was_compressed,
            verdict=(
                CompressionVerdict.COMPRESS
                if result.was_compressed
                else CompressionVerdict.NOT_NEEDED
            ),
            tokens_before=before.total,
            tokens_after=after.total,
            messages_before=result.original_count,
            messages_after=result.compressed_count,
            reason=f"message budget {message_budget} (threshold {budget.threshold} "
            f"− floor {before.floor})",
        )

    # -- reconciliation --------------------------------------------------

    def update_from_response(
        self,
        usage: Optional[TokenCount],
        estimated_prompt_tokens: Optional[int] = None,
    ) -> None:
        """Fold the provider's real prompt count back into engine state.

        Two things happen here, and both are free — the numbers already came
        back on the response:

        * **Calibration.** ``(estimated, actual)`` trains the local estimator
          toward this provider's real tokenizer.
        * **The anti-thrash verdict.** This is the only place that sees the
          provider's real count *for the just-compacted conversation*, which
          is what makes it the right place to judge whether compaction worked.
          ``should_compress`` cannot do it: it runs before the call, on an
          estimate.
        """
        if usage is None:
            return

        self._stats.calls += 1

        # NOT `usage.prompt_tokens` unconditionally — on Anthropic that field
        # is the uncached remainder, which on a cache hit is near-zero. See
        # TokenCount.billed_prompt_tokens.
        actual = usage.billed_prompt_tokens
        if actual <= 0:
            return

        self._stats.last_actual_prompt_tokens = actual
        self._stats.last_cache_read_tokens = usage.cache_read_tokens

        if estimated_prompt_tokens:
            self._stats.last_estimated_prompt_tokens = estimated_prompt_tokens
            calibrator.observe(
                self._provider, self._model, estimated_prompt_tokens, actual
            )

        if not self._awaiting_verdict:
            return
        self._awaiting_verdict = False

        budget = ContextBudget.resolve(
            provider=self._provider,
            model=self._model,
            max_context_tokens=self._explicit_window,
            threshold_ratio=self._threshold_ratio,
        )

        if actual <= budget.threshold:
            # Compaction worked. Clear the streak — an isolated ineffective
            # pass between two effective ones is noise, not a trend.
            self._consecutive_ineffective = 0
            return

        self._consecutive_ineffective += 1
        self._stats.ineffective_events += 1
        logger.warning(
            "[CONTEXT] compaction ineffective: provider reported %d prompt "
            "tokens, still over threshold %d (streak %d/%d)",
            actual,
            budget.threshold,
            self._consecutive_ineffective,
            _INEFFECTIVE_LATCH_THRESHOLD,
        )
        if self._consecutive_ineffective >= _INEFFECTIVE_LATCH_THRESHOLD:
            self._latched_off = True
            logger.error(
                "[CONTEXT] latching compaction off — %d consecutive passes "
                "failed to bring the prompt under %d. Continuing to compact "
                "would burn a summarization call per turn without converging.",
                self._consecutive_ineffective,
                budget.threshold,
            )

    # -- misc ------------------------------------------------------------

    def _remember_identity(self, shape: RequestShape) -> None:
        if shape.provider and not self._provider:
            self._provider = shape.provider
        if shape.model and not self._model:
            self._model = shape.model

    async def on_session_start(self, shape: RequestShape) -> None:
        self._remember_identity(shape)
        # A new session gets a clean anti-thrash slate: the latch is a
        # statement about one conversation's shape, not about the deployment.
        self._awaiting_verdict = False
        self._consecutive_ineffective = 0
        self._latched_off = False

    def stats(self) -> EngineStats:
        return self._stats

    def observability_snapshot(self, shape: Optional[RequestShape] = None) -> Dict[str, Any]:
        """Everything a UI or trace would want about context state."""
        snapshot: Dict[str, Any] = {
            "engine": self.name,
            "stats": self._stats.to_dict(),
            "latched_off": self._latched_off,
            "consecutive_ineffective": self._consecutive_ineffective,
        }
        if shape is not None:
            snapshot["tokens"] = self.breakdown(shape).to_dict()
            snapshot["budget"] = self._budget_for(shape).to_dict()
        return snapshot
