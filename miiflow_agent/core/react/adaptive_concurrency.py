"""Rate-limit-adaptive concurrency for parallel tool batches.

The parallel batch path used a fixed Semaphore(8). When a branch hits a
provider rate limit — the dominant case being a wide ``dispatch_assistant``
fan-out where every branch is a full sub-agent making LLM calls — launching
the remaining branches at full width just converts one 429 into many.

Patterned after kimi-code's subagent-batch scheduler, reduced to what a tool
batch needs:

* on a rate-limit signal, halve the effective limit (floor 1), at most once
  per ``shrink_cooldown`` so one burst of 429s from sibling branches counts
  as one event, not five;
* after ``recovery_interval`` without a rate-limit signal, restore capacity
  one slot at a time up to the configured maximum.

The limiter lives on the executor (one per run), so learned pressure carries
across steps of the same run but never leaks between runs.
"""

import asyncio
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

#: Bursts of 429s from sibling branches within this window count as ONE
#: shrink event.
DEFAULT_SHRINK_COOLDOWN_S = 2.0

#: How long the pool must stay rate-limit-free before capacity grows back,
#: one slot at a time.
DEFAULT_RECOVERY_INTERVAL_S = 30.0

_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "too many requests")

# \b429\b, not the bare substring: entity ids, counts, and offsets carry
# these digits constantly ("campaign 84291 not found" must not shrink the
# pool). Digit-adjacent occurrences (4291, 84290) have no word boundary and
# don't match; a standalone status code ("HTTP 429", "status: 429") does.
_STANDALONE_429 = re.compile(r"\b429\b")


def looks_like_rate_limit(outcome) -> bool:
    """Whether a branch outcome (exception or ToolResult) signals a 429.

    Tool failures arrive flattened into ``ToolResult(success=False, error=str)``,
    so exception-type checks alone see nothing — the error string is the
    portable signal.
    """
    if outcome is None:
        return False
    if isinstance(outcome, BaseException):
        if type(outcome).__name__ == "RateLimitError":
            return True
        text = str(outcome)
    else:
        if getattr(outcome, "success", True):
            return False
        text = str(getattr(outcome, "error", "") or "")
    lowered = text.lower()
    if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
        return True
    return _STANDALONE_429.search(lowered) is not None


class AdaptiveConcurrencyLimiter:
    """A resizable concurrency gate with shrink-on-429 / grow-on-quiet."""

    def __init__(
        self,
        max_concurrency: int,
        min_concurrency: int = 1,
        shrink_cooldown_s: float = DEFAULT_SHRINK_COOLDOWN_S,
        recovery_interval_s: float = DEFAULT_RECOVERY_INTERVAL_S,
    ):
        self._max = max(1, int(max_concurrency))
        self._min = max(1, min(int(min_concurrency), self._max))
        self._limit = self._max
        self._active = 0
        self._cond = asyncio.Condition()
        self._shrink_cooldown = shrink_cooldown_s
        self._recovery_interval = recovery_interval_s
        self._last_shrink_at = 0.0
        self._last_rate_limit_at: Optional[float] = None

    @property
    def limit(self) -> int:
        return self._limit

    def _maybe_recover(self, now: float) -> None:
        """Grow one slot when the pool has been quiet long enough.

        Called with the condition lock held. ``_last_shrink_at`` doubles as
        the last-growth stamp so recovery is also one-slot-per-interval.
        """
        if self._limit >= self._max or self._last_rate_limit_at is None:
            return
        quiet_for = now - self._last_rate_limit_at
        since_change = now - self._last_shrink_at
        if quiet_for >= self._recovery_interval and since_change >= self._recovery_interval:
            self._limit += 1
            self._last_shrink_at = now
            logger.info(
                "[ADAPTIVE] concurrency recovered to %d/%d after %.0fs without "
                "rate limits",
                self._limit,
                self._max,
                quiet_for,
            )

    async def acquire(self) -> None:
        async with self._cond:
            self._maybe_recover(time.monotonic())
            while self._active >= self._limit:
                await self._cond.wait()
            self._active += 1

    async def release(self) -> None:
        async with self._cond:
            self._active -= 1
            self._cond.notify_all()

    def report_rate_limit(self) -> None:
        """A branch hit a 429. Halve capacity (cooldown-guarded).

        Sync on purpose: callers sit in hot result paths; mutating the ints
        from event-loop coroutines is safe, and waiters re-check the limit on
        the next release's notify.
        """
        now = time.monotonic()
        self._last_rate_limit_at = now
        if now - self._last_shrink_at < self._shrink_cooldown:
            return
        new_limit = max(self._min, self._limit // 2)
        if new_limit < self._limit:
            self._last_shrink_at = now
            self._limit = new_limit
            logger.warning(
                "[ADAPTIVE] rate limit reported by a parallel branch; "
                "shrinking tool concurrency to %d (max %d)",
                new_limit,
                self._max,
            )
