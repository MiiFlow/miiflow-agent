"""Token estimation for context management.

Local-only by design: estimation runs on the hot path, so it must never make a
network call. Accuracy comes from content-aware ratios plus an EWMA correction
learned from the token counts providers already report on every response.

    from miiflow_agent.core.context.tokens import get_counter, calibrator

    counter = get_counter("anthropic", "claude-opus-5")
    breakdown = counter.breakdown(shape)      # per-tier, corrected
    raw = counter.raw_total(shape)            # uncorrected — feed back below
    ...
    calibrator.observe("anthropic", "claude-opus-5", raw, usage.prompt_tokens)
"""

from .calibration import CalibrationState, Calibrator, calibrator
from .counter import (
    LocalTokenCounter,
    TiktokenCounter,
    TokenCounter,
    get_counter,
)

__all__ = [
    "CalibrationState",
    "Calibrator",
    "calibrator",
    "LocalTokenCounter",
    "TiktokenCounter",
    "TokenCounter",
    "get_counter",
]
