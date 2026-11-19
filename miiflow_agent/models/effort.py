"""One reading of a configured ``effort`` for every provider client.

``effort`` is a cross-provider config key: an assistant's stored
``model_config`` is spread into whichever client its model needs
(``AnthropicClient``, ``OpenAIClient``, ``GeminiClient``, ``XAIClient``), and
the server does not validate the key on write. Each client used to read it its
own way -- two refused an unknown name with ``ValueError``, case-sensitively,
one tolerated it, one dropped it at DEBUG -- so the same stored ``"Medium"``
ran on one provider and failed every turn on another after a model swap.

The contract, owned here and nowhere else:

- :func:`normalize_effort` canonicalises the value (trimmed, lower-cased,
  empty -> None). Every client and the server's ``applied_effort`` apply it.
- A configured effort is never refused. One the model does not take -- off its
  scale, or a name no provider knows -- is not sent, and the model runs at its
  default; :func:`warn_effort_not_sent` says so once per process.

A model that takes no effort setting at all skips a configured one silently on
every provider -- one config is shared across model swaps -- except Gemini,
whose empty scale means *uncatalogued* (every Gemini 3 model thinks), so it
warns. An explicit per-call ``reasoning_effort`` passed by calling code is not
stored configuration and is outside this contract (``XAIClient`` raises on an
unsupported one; ``OpenAIClient`` forwards it to the API).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


def normalize_effort(value: Any) -> Optional[str]:
    """The canonical form of a configured ``effort``: trimmed and lower-cased,
    or None when nothing is configured. ``" Medium "`` -> ``"medium"``."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


@lru_cache(maxsize=1)
def known_effort_names() -> frozenset:
    """Every ``effort`` name some provider client in this package sends,
    derived from the catalogues rather than copied (xAI's levels are a subset
    of OpenAI's). Imported lazily: the catalogue modules import this one."""
    from .anthropic import EFFORT_LEVELS
    from .google import THINKING_LEVELS
    from .openai import REASONING_EFFORT_VALUES

    return frozenset(REASONING_EFFORT_VALUES) | frozenset(EFFORT_LEVELS) | frozenset(THINKING_LEVELS)


def warn_effort_not_sent(provider: str, model: str, effort: str, levels: Optional[Sequence[str]]) -> None:
    """Log, once per process for each (provider, model, effort), that a
    configured effort is not sent. Clients are built per turn, and an assistant
    left on ``max`` after a swap from Claude would otherwise log on every one.
    ``levels`` may be any sequence; it is frozen here so the cache key is
    hashable and logging can never fail the turn it reports on."""
    _warn_once(provider, model, effort, tuple(levels or ()))


@lru_cache(maxsize=512)
def _warn_once(provider: str, model: str, effort: str, levels: tuple) -> None:
    if effort not in known_effort_names():
        reason = f"no provider knows that level; known: {sorted(known_effort_names())}"
    elif levels:
        reason = f"it accepts {list(levels)}"
    else:
        reason = "its effort levels are not catalogued"
    logger.warning(
        "Not sending effort=%r to %s model %s; it runs at its default (%s)",
        effort,
        provider,
        model,
        reason,
    )


warn_effort_not_sent.cache_clear = _warn_once.cache_clear  # type: ignore[attr-defined]
