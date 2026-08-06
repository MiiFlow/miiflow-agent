"""Tests for context-overflow detection and short-circuit recovery."""

import asyncio

import pytest

from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.recovery import (
    RecoveryManager,
    RecoveryStrategy,
    is_context_overflow_error,
)


class _FakeContext:
    def __init__(self, messages):
        self.messages = messages


class _FakeCompressor:
    def __init__(self):
        self.calls = 0

    async def compress_if_needed(self, messages, preserve_recent: int = 6):
        self.calls += 1

        class _Result:
            was_compressed = True
            original_count = len(messages)
            messages_attr = messages[-preserve_recent:] if preserve_recent else messages

        result = _Result()
        result.messages = result.messages_attr
        result.compressed_count = len(result.messages)
        return result


def test_detects_anthropic_style_message():
    err = ValueError("prompt is too long: 250000 tokens > 200000 maximum context")
    assert is_context_overflow_error(err) is True


def test_detects_openai_style_message():
    err = RuntimeError("This model's maximum context length is 8192 tokens")
    assert is_context_overflow_error(err) is True


def test_detects_max_output_tokens():
    err = Exception("Reached max_output_tokens before completing")
    assert is_context_overflow_error(err) is True


def test_does_not_match_unrelated_errors():
    assert is_context_overflow_error(ValueError("connection refused")) is False
    assert is_context_overflow_error(None) is False


def test_recovery_short_circuits_to_compression_on_overflow():
    compressor = _FakeCompressor()
    rm = RecoveryManager(context_compressor=compressor)
    ctx = _FakeContext([Message(role=MessageRole.USER, content=f"msg {i}") for i in range(20)])

    err = RuntimeError("prompt is too long; exceeded the context window")

    action = asyncio.run(
        rm.attempt_recovery(error=err, context=ctx, step=None, tool_name=None)
    )

    assert action.strategy_used == RecoveryStrategy.COMPRESS_AND_RETRY
    assert action.should_continue is True
    assert compressor.calls == 1
    # Compression should have shrunk the message list.
    assert len(ctx.messages) <= 6


def test_recovery_short_circuit_skipped_without_compressor():
    rm = RecoveryManager(context_compressor=None)
    ctx = _FakeContext([])
    err = RuntimeError("prompt is too long")

    action = asyncio.run(
        rm.attempt_recovery(error=err, context=ctx, step=None, tool_name=None)
    )
    # Without a compressor we fall through to the normal ladder, which on
    # attempt 1 is RETRY_WITH_GUIDANCE.
    assert action.strategy_used == RecoveryStrategy.RETRY_WITH_GUIDANCE


# ---------------------------------------------------------------------------
# Regression tests for the ContextEngine / compress_if_needed API mismatch.
#
# RecoveryManager was written against the legacy ContextCompressor. When Agent
# migrated to the pluggable ContextEngine, recovery kept calling
# compress_if_needed() — a method engines don't have — and the AttributeError
# was swallowed by a blanket `except Exception`, silently disabling the
# COMPRESS_AND_RETRY leg in the default configuration. These tests drive the
# real DefaultContextEngine (not a fake) so the mismatch can never hide again.
# ---------------------------------------------------------------------------


def _make_engine(**kwargs):
    from miiflow_agent.core.context.compressor import DefaultContextEngine
    from miiflow_agent.core.context_compression import CompressionStrategy

    kwargs.setdefault("strategy", CompressionStrategy.TRUNCATE)
    return DefaultContextEngine(client=None, **kwargs)


class _StubToolExecutor:
    """Just enough executor for _compress_for_recovery to build a shape."""

    def build_request_shape(self, messages, tools=None):
        from miiflow_agent.core.context import RequestShape

        return RequestShape(messages=messages, tools=[], provider=None, model=None)


def _make_orchestrator(engine, recovery_manager=None):
    from miiflow_agent.core.react.orchestrator import ReActOrchestrator

    return ReActOrchestrator(
        tool_executor=_StubToolExecutor(),
        event_bus=None,
        safety_manager=None,
        recovery_manager=recovery_manager,
        context_compressor=engine,
    )


def test_engine_without_compress_fn_is_loud_not_silent(caplog):
    """A ContextEngine handed straight to RecoveryManager must not silently
    no-op — it logs an error and recovery continues without compaction."""
    import logging

    rm = RecoveryManager(context_compressor=_make_engine())
    messages = [Message(role=MessageRole.USER, content=f"msg {i}") for i in range(20)]
    ctx = _FakeContext(list(messages))

    with caplog.at_level(logging.ERROR):
        action = asyncio.run(
            rm.attempt_recovery(
                error=RuntimeError("prompt is too long"),
                context=ctx,
                step=None,
                tool_name=None,
            )
        )

    assert action.strategy_used == RecoveryStrategy.COMPRESS_AND_RETRY
    assert action.should_continue is True
    assert ctx.messages == messages  # nothing compressed...
    assert any("COMPRESS_AND_RETRY cannot run" in r.message for r in caplog.records)
    # ...and the guidance must not claim the context was refreshed.
    assert "refreshed" not in (action.guidance_message or "")


def test_orchestrator_wires_compress_fn_into_recovery_manager():
    rm = RecoveryManager(context_compressor=_make_engine())
    orchestrator = _make_orchestrator(_make_engine(), recovery_manager=rm)
    assert rm.compress_fn == orchestrator._compress_for_recovery


def test_compress_for_recovery_shrinks_history_with_default_engine():
    """End-to-end: overflow recovery through the orchestrator's compress_fn
    actually shrinks the history using a real DefaultContextEngine."""
    engine = _make_engine(max_context_tokens=200)
    rm = RecoveryManager(context_compressor=engine)
    orchestrator = _make_orchestrator(engine, recovery_manager=rm)

    messages = [
        Message(role=MessageRole.USER, content="x" * 400) for _ in range(30)
    ]
    ctx = _FakeContext(list(messages))

    action = asyncio.run(
        rm.attempt_recovery(
            error=RuntimeError("prompt is too long; exceeded the context window"),
            context=ctx,
            step=None,
            tool_name=None,
        )
    )

    assert action.strategy_used == RecoveryStrategy.COMPRESS_AND_RETRY
    assert action.should_continue is True
    assert len(ctx.messages) < len(messages)
    assert "refreshed" in action.guidance_message


def test_default_engine_force_compresses_even_under_threshold():
    """force=True is the provider-confirmed-overflow path: it must shrink the
    conversation even when the local estimate says the request fits."""
    engine = _make_engine(max_context_tokens=1_000_000)

    from miiflow_agent.core.context import RequestShape

    messages = [
        Message(role=MessageRole.USER, content="hello world " * 40) for _ in range(20)
    ]
    shape = RequestShape(messages=messages, tools=[], provider=None, model=None)

    # Sanity: the engine's own sizing says no compaction is needed.
    assert not engine.should_compress(shape).should_compress

    outcome = asyncio.run(engine.compress(shape, force=True))
    assert outcome.was_compressed is True
    assert len(outcome.shape.messages) < len(messages)
