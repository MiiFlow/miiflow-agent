"""Self-calibrating correction factor for local token estimates.

Local estimation is cheap but approximate. Every provider response, however,
reports the *real* prompt token count it billed. That number is ground truth
for everything we sent, and it costs nothing extra to read.

This module closes the loop: after each call we compare what we estimated
against what the provider actually counted, and fold the ratio into an EWMA
correction factor keyed by ``(provider, model)``. Subsequent estimates are
scaled by that factor, so the estimator converges on the provider's real
tokenizer within a handful of calls — without ever making a ``count_tokens``
round-trip on the hot path.

Why an EWMA rather than a running mean: the ratio is not stationary. A run
whose messages are mostly prose has a different chars-per-token profile from
one dominated by JSON tool results, and a conversation drifts between the two
as it goes. An EWMA tracks that drift; a mean averages it away.

The factor is deliberately clamped. A single pathological turn (an enormous
base64 image block estimated at a flat cost, say) must not be able to poison
the factor for the rest of the process.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


#: Weight given to the newest observation. Low enough that one outlier turn
#: barely moves the factor, high enough to converge in ~5-10 calls.
_EWMA_ALPHA = 0.25

#: Hard bounds on the correction factor. Our base ratios are within ~35% of
#: every real tokenizer we target, so a factor outside this range means the
#: observation was garbage (a truncated response, a provider reporting usage
#: for a different request shape, an estimate of zero), not that the estimator
#: is genuinely off by 2x. Clamp rather than trust it.
_FACTOR_MIN = 0.5
_FACTOR_MAX = 2.0

#: Ignore observations below this many actual tokens. Tiny requests are
#: dominated by fixed per-request overhead we don't model, so their ratio is
#: noise and would drag the factor around for no benefit.
_MIN_OBSERVATION_TOKENS = 200


@dataclass
class CalibrationState:
    """Per-``(provider, model)`` correction factor and its provenance."""

    factor: float = 1.0
    observations: int = 0
    last_estimated: int = 0
    last_actual: int = 0

    @property
    def is_grounded(self) -> bool:
        """True once at least one real provider count has been folded in.

        Callers that surface context numbers to users (or that gate an
        expensive decision on them) can use this to distinguish "estimated"
        from "estimated, then corrected against the provider".
        """
        return self.observations > 0

    def drift_pct(self) -> Optional[float]:
        """Signed error of the most recent estimate, as a percentage.

        Positive means we over-estimated. Returns ``None`` before the first
        observation. This is the metric to watch when flipping a provider
        onto exact counting — it should collapse toward zero.
        """
        if not self.observations or not self.last_actual:
            return None
        return (self.last_estimated - self.last_actual) / self.last_actual * 100.0


class Calibrator:
    """Thread-safe registry of correction factors.

    One instance is shared process-wide (see :data:`calibrator`). Concurrent
    agent runs against the same model deliberately share a factor — they are
    hitting the same tokenizer, so pooling their observations converges faster
    than isolating them would.
    """

    def __init__(self) -> None:
        self._states: Dict[Tuple[str, str], CalibrationState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(provider: Optional[str], model: Optional[str]) -> Tuple[str, str]:
        return ((provider or "unknown").lower(), (model or "unknown").lower())

    def factor_for(self, provider: Optional[str], model: Optional[str]) -> float:
        """Current correction factor. 1.0 when nothing has been observed yet."""
        with self._lock:
            state = self._states.get(self._key(provider, model))
            return state.factor if state else 1.0

    def state_for(
        self, provider: Optional[str], model: Optional[str]
    ) -> CalibrationState:
        """Snapshot of the calibration state (a copy — safe to read freely)."""
        with self._lock:
            state = self._states.get(self._key(provider, model))
            if state is None:
                return CalibrationState()
            return CalibrationState(
                factor=state.factor,
                observations=state.observations,
                last_estimated=state.last_estimated,
                last_actual=state.last_actual,
            )

    def observe(
        self,
        provider: Optional[str],
        model: Optional[str],
        estimated: int,
        actual: int,
    ) -> float:
        """Fold one ``(estimated, actual)`` pair into the factor.

        ``estimated`` must be the *uncorrected* estimate for the exact request
        that produced ``actual`` — that is, what the estimator returned before
        the current factor was applied. Feeding back a corrected estimate makes
        the loop converge on 1.0 regardless of accuracy, which silently
        disables calibration.

        ``actual`` must be the provider's total prompt tokens. On Anthropic
        that is ``input_tokens + cache_creation_input_tokens +
        cache_read_input_tokens``, NOT ``input_tokens`` alone — that field is
        only the uncached remainder, so on a cache hit it is near-zero and
        would make the estimator think it is over-counting by 10x.

        Returns the factor in effect after the update.
        """
        if estimated <= 0 or actual < _MIN_OBSERVATION_TOKENS:
            return self.factor_for(provider, model)

        ratio = actual / estimated
        key = self._key(provider, model)

        with self._lock:
            state = self._states.get(key)
            if state is None:
                # Seed from the first observation rather than easing into it
                # from 1.0 — the first real number is strictly better
                # information than the default, so there is nothing to blend.
                state = CalibrationState(factor=ratio)
                self._states[key] = state
            else:
                state.factor = (1 - _EWMA_ALPHA) * state.factor + _EWMA_ALPHA * ratio

            state.factor = max(_FACTOR_MIN, min(_FACTOR_MAX, state.factor))
            state.observations += 1
            state.last_estimated = estimated
            state.last_actual = actual
            factor = state.factor
            observations = state.observations

        logger.debug(
            "[TOKENS] calibrate provider=%s model=%s est=%d actual=%d "
            "ratio=%.3f factor=%.3f n=%d",
            key[0],
            key[1],
            estimated,
            actual,
            ratio,
            factor,
            observations,
        )
        return factor

    def reset(self) -> None:
        """Drop all learned factors. Tests only."""
        with self._lock:
            self._states.clear()

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """All factors, for metrics export / debugging."""
        with self._lock:
            return {
                f"{provider}:{model}": {
                    "factor": state.factor,
                    "observations": state.observations,
                    "drift_pct": state.drift_pct() or 0.0,
                }
                for (provider, model), state in self._states.items()
            }


#: Process-wide calibrator. Shared on purpose — see :class:`Calibrator`.
calibrator = Calibrator()
