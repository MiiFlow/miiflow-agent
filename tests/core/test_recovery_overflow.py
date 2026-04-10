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

    action = asyncio.get_event_loop().run_until_complete(
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

    action = asyncio.get_event_loop().run_until_complete(
        rm.attempt_recovery(error=err, context=ctx, step=None, tool_name=None)
    )
    # Without a compressor we fall through to the normal ladder, which on
    # attempt 1 is RETRY_WITH_GUIDANCE.
    assert action.strategy_used == RecoveryStrategy.RETRY_WITH_GUIDANCE
