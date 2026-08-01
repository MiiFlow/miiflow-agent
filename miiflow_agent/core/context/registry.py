"""Engine selection by name.

This is the seam that makes the engine swappable per assistant rather than
per deployment. A host application resolves a config value — ``context.engine``
— to a constructor here, so shipping a second policy (server-side compaction
on providers that offer it, a retrieval-backed engine, a no-op for cheap
single-hop agents) needs no orchestrator change.

Registration is by name rather than by import path so a host can register its
own engine without this package knowing it exists.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from .engine import ContextEngine, NullContextEngine

logger = logging.getLogger(__name__)

EngineFactory = Callable[..., ContextEngine]

_REGISTRY: Dict[str, EngineFactory] = {}


def register_engine(name: str, factory: EngineFactory, *, replace: bool = False) -> None:
    """Register an engine constructor under ``name``.

    Refuses to shadow an existing name unless ``replace=True``. Silent
    override is how two plugins both "work" in isolation and one of them
    silently stops mattering once they're loaded together.
    """
    key = name.strip().lower()
    if key in _REGISTRY and not replace:
        raise ValueError(
            f"context engine {name!r} is already registered; "
            "pass replace=True to override deliberately"
        )
    _REGISTRY[key] = factory


def list_engines() -> List[str]:
    return sorted(_REGISTRY)


def get_engine(name: str = "compressor", **kwargs: Any) -> ContextEngine:
    """Construct the engine registered under ``name``.

    An unknown name falls back to the default engine with a warning rather
    than raising. A typo in a config value should degrade to working
    compaction, not take the agent down.
    """
    key = (name or "compressor").strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        logger.warning(
            "[CONTEXT] unknown engine %r (known: %s); falling back to 'compressor'",
            name,
            ", ".join(list_engines()) or "none",
        )
        factory = _REGISTRY["compressor"]
    return factory(**kwargs)


def _default_factory(**kwargs: Any) -> ContextEngine:
    # Imported lazily: compressor imports from this package's siblings, and a
    # module-level import here would close the cycle.
    from .compressor import DefaultContextEngine

    return DefaultContextEngine(**kwargs)


def _null_factory(**kwargs: Any) -> ContextEngine:
    return NullContextEngine()


register_engine("compressor", _default_factory)
register_engine("none", _null_factory)
register_engine("null", _null_factory)
