"""Tests for the provider-agnostic overflow/structural recovery ladder:
adaptive window learning, the capped overflow loop, and one-shot repair of
structurally invalid histories.
"""

import asyncio

import pytest

from miiflow_agent.core.context import RequestShape
from miiflow_agent.core.context.compressor import DefaultContextEngine
from miiflow_agent.core.context_compression import CompressionStrategy
from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.react.message_repair import (
    is_structural_message_error,
    repair_tool_pairing,
)
from miiflow_agent.core.react.recovery import RecoveryManager, RecoveryStrategy


def _engine(**kwargs):
    kwargs.setdefault("strategy", CompressionStrategy.TRUNCATE)
    return DefaultContextEngine(client=None, **kwargs)


def _shape(n=10, size=400):
    messages = [
        Message(role=MessageRole.USER, content="x" * size) for _ in range(n)
    ]
    return RequestShape(messages=messages, tools=[], provider=None, model=None)


class TestObserveOverflow:
    def test_rejection_caps_the_window(self):
        engine = _engine(max_context_tokens=1_000_000)
        shape = _shape(n=40, size=2_000)

        before = engine._budget_for(shape).window
        engine.observe_overflow(shape)
        after_budget = engine._budget_for(shape)

        assert after_budget.window < before
        assert after_budget.source == "observed_overflow"
        # 85% of the estimate for the rejected request.
        estimated = engine.breakdown(shape).total
        assert after_budget.window == max(1_000, int(estimated * 0.85))

    def test_window_only_ratchets_down(self):
        engine = _engine(max_context_tokens=1_000_000)
        small, big = _shape(n=10), _shape(n=40, size=2_000)

        engine.observe_overflow(small)
        first = engine._budget_for(small).window
        engine.observe_overflow(big)  # larger estimate: must not raise the cap
        assert engine._budget_for(small).window == first

    async def test_session_start_keeps_learned_window(self):
        engine = _engine(max_context_tokens=1_000_000)
        shape = _shape(n=40, size=2_000)
        engine.observe_overflow(shape)
        capped = engine._budget_for(shape).window

        await engine.on_session_start(shape)
        assert engine._budget_for(shape).window == capped


class TestOverflowAttemptCap:
    async def test_overflow_recovery_stops_after_cap(self):
        compress_calls = {"n": 0}

        async def fake_compress(context, overflow=False):
            compress_calls["n"] += 1
            return True

        rm = RecoveryManager(compress_fn=fake_compress)
        err = RuntimeError("prompt is too long; exceeded the context window")

        actions = [
            await rm.attempt_recovery(error=err, context=object(), step=None)
            for _ in range(rm.max_overflow_attempts + 1)
        ]

        for action in actions[:-1]:
            assert action.should_continue is True
            assert action.strategy_used == RecoveryStrategy.COMPRESS_AND_RETRY
        assert actions[-1].should_continue is False
        assert compress_calls["n"] == rm.max_overflow_attempts

    async def test_success_resets_overflow_budget(self):
        async def fake_compress(context, overflow=False):
            return True

        rm = RecoveryManager(compress_fn=fake_compress)
        err = RuntimeError("prompt is too long")

        for _ in range(rm.max_overflow_attempts):
            await rm.attempt_recovery(error=err, context=object(), step=None)
        rm.record_success()

        action = await rm.attempt_recovery(error=err, context=object(), step=None)
        assert action.should_continue is True

    async def test_overflow_flag_reaches_compress_fn(self):
        seen = {}

        async def fake_compress(context, overflow=False):
            seen["overflow"] = overflow
            return True

        rm = RecoveryManager(compress_fn=fake_compress)
        await rm.attempt_recovery(
            error=RuntimeError("prompt is too long"), context=object(), step=None
        )
        assert seen["overflow"] is True


class TestStructuralErrorDetection:
    def test_anthropic_wordings(self):
        assert is_structural_message_error(
            Exception("400: tool_use ids were found without tool_result blocks")
        )
        assert is_structural_message_error(
            Exception("unexpected tool_use_id found in tool_result blocks")
        )

    def test_openai_wording(self):
        assert is_structural_message_error(
            Exception(
                "An assistant message with 'tool_calls' must be followed by "
                "tool messages responding to each 'tool_call_id'"
            )
        )

    def test_unrelated_errors_do_not_match(self):
        assert not is_structural_message_error(Exception("rate limited"))
        assert not is_structural_message_error(Exception("prompt is too long"))
        assert not is_structural_message_error(None)


def _call(call_id, name="lookup"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


class TestRepairToolPairing:
    def test_valid_history_is_untouched(self):
        messages = [
            Message(role=MessageRole.USER, content="go"),
            Message(role=MessageRole.ASSISTANT, content="", tool_calls=[_call("a")]),
            Message(role=MessageRole.TOOL, content="ok", tool_call_id="a"),
            Message(role=MessageRole.ASSISTANT, content="done"),
        ]
        repaired, anomalies = repair_tool_pairing(messages)
        assert anomalies == []
        assert repaired == messages

    def test_orphan_tool_result_is_dropped(self):
        messages = [
            Message(role=MessageRole.USER, content="go"),
            Message(role=MessageRole.TOOL, content="stale", tool_call_id="ghost"),
            Message(role=MessageRole.ASSISTANT, content="done"),
        ]
        repaired, anomalies = repair_tool_pairing(messages)
        assert len(anomalies) == 1 and "orphan" in anomalies[0]
        assert all(m.role != MessageRole.TOOL for m in repaired)

    def test_missing_tool_result_is_synthesized(self):
        messages = [
            Message(role=MessageRole.USER, content="go"),
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[_call("a"), _call("b")],
            ),
            Message(role=MessageRole.TOOL, content="ok", tool_call_id="a"),
            Message(role=MessageRole.ASSISTANT, content="continuing"),
        ]
        repaired, anomalies = repair_tool_pairing(messages)
        assert len(anomalies) == 1 and "b" in anomalies[0]
        # The synthesized result sits before the next assistant message.
        roles = [m.role for m in repaired]
        assert roles == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
            MessageRole.TOOL,
            MessageRole.ASSISTANT,
        ]
        synthesized = repaired[3]
        assert synthesized.tool_call_id == "b"
        assert "interrupted" in synthesized.content

    def test_duplicate_tool_result_is_dropped(self):
        messages = [
            Message(role=MessageRole.ASSISTANT, content="", tool_calls=[_call("a")]),
            Message(role=MessageRole.TOOL, content="ok", tool_call_id="a"),
            Message(role=MessageRole.TOOL, content="ok again", tool_call_id="a"),
        ]
        repaired, anomalies = repair_tool_pairing(messages)
        assert len(anomalies) == 1
        assert sum(1 for m in repaired if m.role == MessageRole.TOOL) == 1

    def test_dangling_calls_at_end_are_closed(self):
        messages = [
            Message(role=MessageRole.ASSISTANT, content="", tool_calls=[_call("a")]),
        ]
        repaired, anomalies = repair_tool_pairing(messages)
        assert len(anomalies) == 1
        assert repaired[-1].role == MessageRole.TOOL
        assert repaired[-1].tool_call_id == "a"
