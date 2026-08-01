"""Tests for the context engine.

Three behaviours are load-bearing and each corresponds to a real defect:

  * the engine sizes **tool schemas**, not just messages (the old compressor
    was blind to them, which on a tool-heavy assistant meant ignoring the
    largest tier);
  * the window comes from the **model registry**, not a hard-coded 128000
    (which compacted at 96K on a 1M-token model);
  * ``FLOOR_EXCEEDED`` is distinct from ``COMPRESS``, so a request that
    compaction provably cannot rescue does not get compacted every turn
    forever.
"""

import json

import pytest

from miiflow_agent.core.context import (
    CompressionVerdict,
    ContextBudget,
    NullContextEngine,
    RequestShape,
    get_engine,
    list_engines,
)
from miiflow_agent.core.context.tokens import calibrator
from miiflow_agent.core.message import Message, MessageRole
from miiflow_agent.core.metrics import TokenCount

pytestmark = pytest.mark.unit


def _tool(i: int) -> dict:
    return {
        "name": f"ads_tool_{i}",
        "description": "Query a Google Ads account and return performance rows. " * 8,
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "x" * 150},
                "query": {"type": "string", "description": "y" * 150},
            },
            "required": ["customer_id"],
        },
    }


def _tool_result(n: int) -> Message:
    return Message(
        role=MessageRole.TOOL,
        content=json.dumps([{"campaign": "Brand", "cost": 12.5}] * n),
        tool_call_id="t1",
    )


@pytest.fixture(autouse=True)
def _clean_calibration():
    calibrator.reset()
    yield
    calibrator.reset()


class TestBudgetResolution:
    def test_window_comes_from_model_registry(self):
        budget = ContextBudget.resolve("anthropic", "claude-opus-5")
        assert budget.window == 1_000_000
        assert budget.source == "registry"

    def test_explicit_override_wins(self):
        budget = ContextBudget.resolve(
            "anthropic", "claude-opus-5", max_context_tokens=50_000
        )
        assert budget.window == 50_000
        assert budget.source == "explicit"

    def test_unknown_model_falls_back_conservatively(self):
        """Guessing high means a provider 400; guessing low means compaction
        runs earlier than needed. The second failure is cheaper."""
        budget = ContextBudget.resolve("anthropic", "claude-something-unreleased")
        assert budget.window == 200_000
        assert budget.source == "provider_fallback"

    def test_dated_snapshot_resolves_to_family(self):
        budget = ContextBudget.resolve("openai", "gpt-4o-mini-2024-07-18")
        assert budget.window > 0

    def test_unknown_provider_uses_global_fallback(self):
        budget = ContextBudget.resolve("nobody", "nothing")
        assert budget.window == 128_000

    def test_threshold_is_a_fraction_of_the_window(self):
        budget = ContextBudget.resolve(max_context_tokens=100_000, threshold_ratio=0.75)
        assert budget.threshold == 75_000


class TestVerdicts:
    def test_small_request_not_needed(self):
        engine = get_engine("compressor")
        shape = RequestShape(
            messages=[Message.user("hi")],
            tools=[_tool(0)],
            provider="anthropic",
            model="claude-opus-5",
        )
        assert engine.should_compress(shape).verdict is CompressionVerdict.NOT_NEEDED

    def test_tool_schemas_alone_can_trigger_floor_exceeded(self):
        """The defect this whole change exists for: schemas are the biggest
        tier and the old compressor could not see them at all."""
        engine = get_engine("compressor", max_context_tokens=8_000)
        shape = RequestShape(
            messages=[Message.user("hi")],
            tools=[_tool(i) for i in range(60)],
            provider="anthropic",
            model="claude-opus-5",
        )
        decision = engine.should_compress(shape)
        assert decision.verdict is CompressionVerdict.FLOOR_EXCEEDED
        assert decision.breakdown.tools > decision.breakdown.messages

    def test_long_conversation_triggers_compress(self):
        engine = get_engine("compressor", max_context_tokens=20_000)
        shape = RequestShape(
            messages=[Message.user("q")] + [_tool_result(400) for _ in range(20)],
            tools=[_tool(0)],
            provider="anthropic",
            model="claude-opus-5",
        )
        decision = engine.should_compress(shape)
        assert decision.verdict is CompressionVerdict.COMPRESS

    def test_floor_exceeded_is_not_compress(self):
        """Collapsing these two verdicts is what produces the thrash loop."""
        engine = get_engine("compressor", max_context_tokens=8_000)
        shape = RequestShape(
            messages=[Message.user("hi")],
            tools=[_tool(i) for i in range(60)],
            provider="anthropic",
            model="claude-opus-5",
        )
        assert not engine.should_compress(shape).should_compress


class TestCompression:
    @pytest.mark.asyncio
    async def test_message_budget_subtracts_the_floor(self):
        """Truncation must get `threshold - floor`, not `threshold` — passing
        the full threshold is what made compaction under-deliver on
        tool-heavy assistants."""
        engine = get_engine("compressor", max_context_tokens=20_000)
        shape = RequestShape(
            messages=[Message.user("q")] + [_tool_result(400) for _ in range(20)],
            tools=[_tool(i) for i in range(5)],
            provider="anthropic",
            model="claude-opus-5",
        )
        floor = engine.breakdown(shape).floor
        outcome = await engine.compress(shape)
        assert outcome.was_compressed
        assert str(15_000 - floor) in outcome.reason

    @pytest.mark.asyncio
    async def test_compression_preserves_the_floor(self):
        engine = get_engine("compressor", max_context_tokens=20_000)
        shape = RequestShape(
            messages=[Message.user("q")] + [_tool_result(400) for _ in range(20)],
            tools=[_tool(i) for i in range(5)],
            provider="anthropic",
            model="claude-opus-5",
        )
        before = engine.breakdown(shape)
        outcome = await engine.compress(shape)
        assert engine.breakdown(outcome.shape).floor == before.floor
        assert outcome.tokens_after < outcome.tokens_before

    @pytest.mark.asyncio
    async def test_system_prompt_survives_compaction(self):
        engine = get_engine("compressor", max_context_tokens=20_000)
        system = Message.system("You are an ads assistant. " * 50)
        shape = RequestShape(
            messages=[system, Message.user("q")]
            + [_tool_result(400) for _ in range(20)],
            provider="anthropic",
            model="claude-opus-5",
        )
        outcome = await engine.compress(shape)
        assert any(m.role == MessageRole.SYSTEM for m in outcome.shape.messages)


class TestAntiThrash:
    @pytest.mark.asyncio
    async def test_latches_off_after_repeated_ineffective_passes(self):
        engine = get_engine("compressor", max_context_tokens=20_000)
        shape = RequestShape(
            messages=[Message.user("q")] + [_tool_result(400) for _ in range(20)],
            provider="anthropic",
            model="claude-opus-5",
        )
        for _ in range(2):
            await engine.compress(shape)
            engine.update_from_response(TokenCount(prompt_tokens=999_999))
        assert engine.should_compress(shape).verdict is CompressionVerdict.INEFFECTIVE

    @pytest.mark.asyncio
    async def test_one_bad_pass_does_not_latch(self):
        """A single turn can append a huge tool result right after compaction
        and legitimately push the request back over."""
        engine = get_engine("compressor", max_context_tokens=20_000)
        shape = RequestShape(
            messages=[Message.user("q")] + [_tool_result(400) for _ in range(20)],
            provider="anthropic",
            model="claude-opus-5",
        )
        await engine.compress(shape)
        engine.update_from_response(TokenCount(prompt_tokens=999_999))
        assert engine.should_compress(shape).verdict is CompressionVerdict.COMPRESS

    @pytest.mark.asyncio
    async def test_effective_pass_clears_the_streak(self):
        engine = get_engine("compressor", max_context_tokens=20_000)
        shape = RequestShape(
            messages=[Message.user("q")] + [_tool_result(400) for _ in range(20)],
            provider="anthropic",
            model="claude-opus-5",
        )
        await engine.compress(shape)
        engine.update_from_response(TokenCount(prompt_tokens=999_999))
        await engine.compress(shape)
        engine.update_from_response(TokenCount(prompt_tokens=100))  # worked
        await engine.compress(shape)
        engine.update_from_response(TokenCount(prompt_tokens=999_999))
        assert engine.should_compress(shape).verdict is CompressionVerdict.COMPRESS


class TestReconciliation:
    def test_calibrates_from_real_usage(self):
        engine = get_engine("compressor")
        shape = RequestShape(
            messages=[Message.user("hi " * 500)],
            provider="anthropic",
            model="claude-opus-5",
        )
        engine.should_compress(shape)  # binds provider/model
        engine.update_from_response(
            TokenCount(prompt_tokens=1_500), estimated_prompt_tokens=1_000
        )
        assert calibrator.factor_for("anthropic", "claude-opus-5") == pytest.approx(1.5)

    def test_anthropic_cache_split_does_not_poison_calibration(self):
        """Anthropic's input_tokens is the uncached remainder. Calibrating
        against it would make the estimator think it over-counts by 10x."""
        engine = get_engine("compressor")
        shape = RequestShape(messages=[], provider="anthropic", model="claude-opus-5")
        engine.should_compress(shape)
        engine.update_from_response(
            TokenCount(prompt_tokens=0, cache_read_tokens=30_000),
            estimated_prompt_tokens=30_000,
        )
        assert calibrator.factor_for("anthropic", "claude-opus-5") == pytest.approx(1.0)

    def test_none_usage_is_ignored(self):
        engine = get_engine("compressor")
        engine.update_from_response(None)  # must not raise


class TestRegistry:
    def test_known_engines(self):
        assert "compressor" in list_engines()
        assert "none" in list_engines()

    def test_unknown_name_falls_back_rather_than_raising(self):
        """A typo in a config value should degrade to working compaction,
        not take the agent down."""
        engine = get_engine("does-not-exist")
        assert engine.name == "compressor"

    def test_null_engine_never_compresses(self):
        engine = get_engine("none")
        assert isinstance(engine, NullContextEngine)
        shape = RequestShape(messages=[Message.user("x" * 100_000)])
        assert engine.should_compress(shape).verdict is CompressionVerdict.DISABLED

    def test_duplicate_registration_is_rejected(self):
        from miiflow_agent.core.context import register_engine

        with pytest.raises(ValueError, match="already registered"):
            register_engine("compressor", lambda **kw: None)
