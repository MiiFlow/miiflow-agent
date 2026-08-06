"""Pluggable context management.

A context engine owns one question: *what should go on the wire for this
call, given a token budget?* Everything downstream of that — when to compact,
how to compact, whether compacting is even capable of helping — is policy, and
policy is exactly the thing that differs between deployments.

Previously this was a single concrete class the orchestrator constructed
directly, which made two things impossible: swapping the policy per assistant,
and answering "is compaction actually working" at runtime.

The lifecycle mirrors the request loop:

  1. ``on_session_start``  — once, when a run begins
  2. ``should_compress``   — before each LLM call
  3. ``compress``          — when (2) says so
  4. ``update_from_response`` — after each call, with the provider's real usage
  5. ``on_session_end``    — at a real session boundary, not per turn

Step 4 is what makes the engine self-correcting. The provider reports exactly
how many tokens it processed; folding that back is free and turns a local
estimate into a measurement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..message import Message
from ..metrics import TokenCount
from .budget import ContextBudget
from .shape import RequestShape, TokenBreakdown

logger = logging.getLogger(__name__)


class CompressionVerdict(Enum):
    """Why the engine decided to compact — or not to.

    The distinction that matters most is ``FLOOR_EXCEEDED`` vs ``COMPRESS``.
    Both mean "the request is over budget", but only one of them is a problem
    compaction can solve. Collapsing them is what produces a thrash loop:
    every pass shrinks messages by a healthy margin, the request stays over
    because the system prompt and tool schemas alone exceed the threshold, and
    the next turn tries again. Forever.
    """

    #: Under threshold. Send as-is.
    NOT_NEEDED = "not_needed"

    #: Over threshold, and the conversation is the reason. Compact it.
    COMPRESS = "compress"

    #: Over threshold, but system + tools alone already exceed it. Compaction
    #: cannot bring this under the line — it can only burn a summarization
    #: call to prove that. The fix is upstream: fewer tools, a shorter system
    #: prompt, or a bigger window.
    FLOOR_EXCEEDED = "floor_exceeded"

    #: Compaction ran, the provider's real count came back still over
    #: threshold, and the floor did not explain it. Latch off rather than
    #: retry — something is wrong with the strategy, and looping makes it
    #: worse, not better.
    INEFFECTIVE = "ineffective"

    #: Engine is disabled for this run.
    DISABLED = "disabled"


@dataclass
class CompressionDecision:
    """Outcome of :meth:`ContextEngine.should_compress`."""

    verdict: CompressionVerdict
    breakdown: TokenBreakdown
    budget: ContextBudget
    reason: str = ""

    @property
    def should_compress(self) -> bool:
        return self.verdict is CompressionVerdict.COMPRESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "tokens": self.breakdown.to_dict(),
            "budget": self.budget.to_dict(),
        }


@dataclass
class CompressionOutcome:
    """Result of :meth:`ContextEngine.compress`.

    Carries the full :class:`RequestShape` rather than a bare message list so
    an engine is free to touch the other tiers — an engine that trims the tool
    surface, or swaps in a shorter system prompt under pressure, needs
    somewhere to put that.
    """

    shape: RequestShape
    was_compressed: bool
    verdict: CompressionVerdict = CompressionVerdict.NOT_NEEDED
    tokens_before: int = 0
    tokens_after: int = 0
    messages_before: int = 0
    messages_after: int = 0
    reason: str = ""

    @property
    def messages(self) -> List[Message]:
        """Convenience accessor — most callers only want the messages."""
        return self.shape.messages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "was_compressed": self.was_compressed,
            "verdict": self.verdict.value,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "reason": self.reason,
        }


@dataclass
class EngineStats:
    """Running counters, for the breakdown event and debugging."""

    calls: int = 0
    compressions: int = 0
    floor_exceeded_events: int = 0
    ineffective_events: int = 0
    last_actual_prompt_tokens: int = 0
    last_estimated_prompt_tokens: int = 0
    last_cache_read_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        drift = None
        if self.last_actual_prompt_tokens and self.last_estimated_prompt_tokens:
            drift = round(
                (self.last_estimated_prompt_tokens - self.last_actual_prompt_tokens)
                / self.last_actual_prompt_tokens
                * 100.0,
                2,
            )
        return {
            "calls": self.calls,
            "compressions": self.compressions,
            "floor_exceeded_events": self.floor_exceeded_events,
            "ineffective_events": self.ineffective_events,
            "last_actual_prompt_tokens": self.last_actual_prompt_tokens,
            "last_estimated_prompt_tokens": self.last_estimated_prompt_tokens,
            "last_cache_read_tokens": self.last_cache_read_tokens,
            "estimate_drift_pct": drift,
        }


class ContextEngine(ABC):
    """Base class for context management policies.

    Implementations must be safe to call from the orchestrator's hot path.
    Concretely: no network I/O in ``should_compress`` or ``breakdown``.
    ``compress`` may do I/O (the default engine calls an LLM to summarize) —
    it only runs when the engine has already decided the alternative is a
    failed request.
    """

    name: str = "base"

    # -- lifecycle -------------------------------------------------------

    async def on_session_start(self, shape: RequestShape) -> None:
        """Called once when a run begins. Default: nothing."""

    async def on_session_end(self, messages: List[Message]) -> None:
        """Called at a real session boundary — not per turn. Default: nothing.

        This is the hook an engine would use to extract durable knowledge from
        a finished conversation. Deliberately *not* called per turn: doing
        extraction work on every turn is how a context engine becomes the
        thing that makes turns slow.
        """

    # -- required policy -------------------------------------------------

    @abstractmethod
    def breakdown(self, shape: RequestShape) -> TokenBreakdown:
        """Per-tier token estimate for ``shape``."""

    @abstractmethod
    def should_compress(self, shape: RequestShape) -> CompressionDecision:
        """Decide whether ``shape`` needs compaction before going on the wire."""

    @abstractmethod
    async def compress(
        self, shape: RequestShape, *, force: bool = False
    ) -> CompressionOutcome:
        """Compact ``shape``. Only called when ``should_compress`` says so.

        ``force=True`` means the provider itself rejected the request as too
        large. That signal outranks any local estimate — a forced pass must
        actually shrink the conversation even when the engine's own sizing
        says the request fits, because the sizing has just been proven wrong.
        """

    # -- optional -------------------------------------------------------

    def update_from_response(
        self, usage: Optional[TokenCount], estimated_prompt_tokens: Optional[int] = None
    ) -> None:
        """Fold the provider's real usage back into the engine's state.

        ``estimated_prompt_tokens`` must be the *uncorrected* local estimate
        for the request that produced ``usage`` — see
        ``tokens.calibration.Calibrator.observe``.
        """

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Tools this engine wants exposed to the model.

        An engine that indexes compacted history and offers a retrieval tool
        over it would return that tool's schema here. The default engine has
        none.
        """
        return []

    def stats(self) -> EngineStats:
        """Running counters. Default: empty."""
        return EngineStats()


class NullContextEngine(ContextEngine):
    """Engine that never compacts.

    The honest representation of "context management is off" — previously
    expressed as a ``None`` compressor, which meant every call site needed a
    truthiness check and the ones that forgot silently skipped the breakdown
    event too.
    """

    name = "null"

    def __init__(self, budget: Optional[ContextBudget] = None):
        self._budget = budget or ContextBudget.resolve()

    def breakdown(self, shape: RequestShape) -> TokenBreakdown:
        from .tokens import get_counter

        return get_counter(shape.provider, shape.model).breakdown(shape)

    def should_compress(self, shape: RequestShape) -> CompressionDecision:
        return CompressionDecision(
            verdict=CompressionVerdict.DISABLED,
            breakdown=self.breakdown(shape),
            budget=self._budget,
            reason="context engine disabled",
        )

    async def compress(
        self, shape: RequestShape, *, force: bool = False
    ) -> CompressionOutcome:
        return CompressionOutcome(
            shape=shape,
            was_compressed=False,
            verdict=CompressionVerdict.DISABLED,
            messages_before=len(shape.messages),
            messages_after=len(shape.messages),
        )
