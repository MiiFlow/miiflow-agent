"""Pluggable context management.

    from miiflow_agent.core.context import (
        RequestShape, DefaultContextEngine, get_engine,
    )

    engine = get_engine("compressor", client=client)
    shape = RequestShape(messages=msgs, system=sys, tools=tools,
                         provider="anthropic", model="claude-opus-5")

    decision = engine.should_compress(shape)
    if decision.should_compress:
        shape = (await engine.compress(shape)).shape
    ...
    engine.update_from_response(response.usage, estimated_prompt_tokens=raw)

The engine sees ``system`` and ``tools``, not just messages — that is the
point. See ``shape.py`` for why.
"""

from .budget import DEFAULT_COMPRESSION_THRESHOLD, ContextBudget
from .compressor import DefaultContextEngine
from .engine import (
    CompressionDecision,
    CompressionOutcome,
    CompressionVerdict,
    ContextEngine,
    EngineStats,
    NullContextEngine,
)
from .registry import get_engine, list_engines, register_engine
from .shape import RequestShape, TokenBreakdown
from .tokens import TokenCounter, calibrator, get_counter

__all__ = [
    "ContextBudget",
    "DEFAULT_COMPRESSION_THRESHOLD",
    "CompressionDecision",
    "CompressionOutcome",
    "CompressionVerdict",
    "ContextEngine",
    "DefaultContextEngine",
    "EngineStats",
    "NullContextEngine",
    "RequestShape",
    "TokenBreakdown",
    "TokenCounter",
    "calibrator",
    "get_counter",
    "get_engine",
    "list_engines",
    "register_engine",
]
