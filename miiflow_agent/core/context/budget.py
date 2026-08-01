"""Context-window budgets, resolved from the model registry.

The compressor previously hard-coded ``max_context_tokens or 128000``. That
default is wrong for essentially every model this package actually runs:
Claude is 200K–1M, GPT-5 is 400K, Gemini is 1M–2M. On a 1M-token model it
means compaction fires at 96K — summarizing away context the model could have
held eight times over, and paying an LLM call to do it.

The per-model numbers already exist in ``miiflow_agent.models`` as
``ModelConfig.maximum_context_tokens``. This module reads them instead of
maintaining a second, divergent table.

Resolution order, most specific first:

  1. An explicit ``max_context_tokens`` passed by the caller. Always wins —
     a deployment may want a smaller working window than the model allows
     (to bound cost, or to leave room for an output budget).
  2. The model registry entry for ``(provider, model)``.
  3. A conservative per-provider floor, for models newer than this package.

Step 3 is deliberately conservative rather than optimistic. Guessing too high
means the request 400s at the provider; guessing too low means compaction runs
earlier than it needed to. The second failure is cheaper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


#: Fallback window per provider, used when the model isn't in the registry
#: (a release newer than this package). Chosen as the *smallest* current
#: window for that provider's family, so an unknown model is under-estimated
#: rather than over-estimated.
_PROVIDER_FALLBACK_WINDOW = {
    "anthropic": 200_000,
    "bedrock": 200_000,
    "openai": 128_000,
    "azure": 128_000,
    "gemini": 1_000_000,
    "google": 1_000_000,
    "groq": 128_000,
    "mistral": 128_000,
    "deepseek": 128_000,
    "xai": 128_000,
    "openrouter": 128_000,
    "ollama": 32_000,
}

#: Used when even the provider is unknown.
_GLOBAL_FALLBACK_WINDOW = 128_000

#: Fraction of the window at which compaction fires. Below this the request
#: goes out untouched. 0.75 leaves headroom for the model's own output plus
#: whatever the current turn is about to append (a large tool result can land
#: between the check and the call).
DEFAULT_COMPRESSION_THRESHOLD = 0.75


def _registry_window(provider: Optional[str], model: Optional[str]) -> Optional[int]:
    """Look up ``maximum_context_tokens`` from the model registry.

    Import is local and defensive: the registry is a large module graph and
    budget resolution must never be the thing that fails a request.
    """
    if not model:
        return None
    try:
        from ... import models as model_registry
    except Exception:  # noqa: BLE001
        return None

    tables = {
        "anthropic": "ANTHROPIC_MODELS",
        "bedrock": "ANTHROPIC_MODELS",
        "openai": "OPENAI_MODELS",
        "azure": "OPENAI_MODELS",
        "gemini": "GOOGLE_MODELS",
        "google": "GOOGLE_MODELS",
        "groq": "GROQ_MODELS",
        "mistral": "MISTRAL_MODELS",
        "deepseek": "DEEPSEEK_MODELS",
        "xai": "XAI_MODELS",
        "openrouter": "OPENROUTER_MODELS",
        "ollama": "OLLAMA_MODELS",
    }

    candidates = []
    table_name = tables.get((provider or "").lower())
    if table_name:
        candidates.append(table_name)
    else:
        # Unknown provider — scan every table. A model id is distinctive
        # enough that a cross-provider collision is not a real concern.
        candidates.extend(dict.fromkeys(tables.values()))

    for name in candidates:
        table = getattr(model_registry, name, None)
        if not isinstance(table, dict):
            continue
        config = table.get(model)
        if config is None:
            # Model ids often carry a deployment suffix ("claude-opus-5-v2",
            # "gpt-4o-2024-08-06"). Fall back to the longest registry key the
            # id starts with, so a dated snapshot resolves to its family.
            matches = [key for key in table if model.startswith(key)]
            if matches:
                config = table[max(matches, key=len)]
        window = getattr(config, "maximum_context_tokens", 0) if config else 0
        if window:
            return int(window)
    return None


@dataclass
class ContextBudget:
    """The token budget for one run, and where it came from.

    ``source`` is carried so a surprising compaction can be traced back to a
    bad window rather than to the compressor — "why did this compact at 96K on
    a 1M model" is answered by ``source == 'default'``.
    """

    window: int
    threshold_ratio: float = DEFAULT_COMPRESSION_THRESHOLD
    source: str = "default"

    @property
    def threshold(self) -> int:
        """Token count at which compaction fires."""
        return int(self.window * self.threshold_ratio)

    @classmethod
    def resolve(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        threshold_ratio: float = DEFAULT_COMPRESSION_THRESHOLD,
    ) -> "ContextBudget":
        if max_context_tokens:
            return cls(
                window=int(max_context_tokens),
                threshold_ratio=threshold_ratio,
                source="explicit",
            )

        window = _registry_window(provider, model)
        if window:
            return cls(window=window, threshold_ratio=threshold_ratio, source="registry")

        window = _PROVIDER_FALLBACK_WINDOW.get(
            (provider or "").lower(), _GLOBAL_FALLBACK_WINDOW
        )
        logger.debug(
            "[CONTEXT] no registry window for provider=%s model=%s; "
            "falling back to %d",
            provider,
            model,
            window,
        )
        return cls(
            window=window,
            threshold_ratio=threshold_ratio,
            source="provider_fallback" if provider else "default",
        )

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "threshold": self.threshold,
            "threshold_ratio": self.threshold_ratio,
            "source": self.source,
        }
