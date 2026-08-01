"""Tests for local token estimation and calibration.

The properties that matter here are not "the estimate equals N" — the estimate
is approximate by construction. What must hold is:

  * tool schemas and tool_calls are counted at all (the old estimator missed
    both, which is what made it blind to the largest tier);
  * the floor is separable from the conversation (the anti-thrash signal
    depends on it);
  * calibration moves the estimate toward the provider's real number and
    cannot be poisoned by one bad observation.
"""

import json

import pytest

from miiflow_agent.core.context.shape import RequestShape, TokenBreakdown
from miiflow_agent.core.context.tokens import get_counter
from miiflow_agent.core.context.tokens.calibration import Calibrator
from miiflow_agent.core.message import Message, MessageRole

pytestmark = pytest.mark.unit


def _tool(name: str) -> dict:
    return {
        "name": name,
        "description": "Run a GAQL query against a Google Ads account." * 4,
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Account id"},
                "query": {"type": "string", "description": "A GAQL query string"},
                "limit": {"type": "integer", "description": "Row cap"},
            },
            "required": ["customer_id", "query"],
        },
    }


@pytest.fixture
def calibrator():
    """A fresh calibrator — the module-level one is process-wide state."""
    return Calibrator()


class TestTierCounting:
    def test_tool_schemas_are_counted(self):
        counter = get_counter("anthropic", "claude-opus-5")
        without = RequestShape(messages=[Message.user("hi")], provider="anthropic")
        with_tools = RequestShape(
            messages=[Message.user("hi")],
            tools=[_tool(f"t{i}") for i in range(40)],
            provider="anthropic",
        )
        assert counter.breakdown(without).tools == 0
        # 40 schemas of this size are thousands of tokens; the point is that
        # they are not zero, which is what the message-only estimator saw.
        assert counter.breakdown(with_tools).tools > 3_000

    def test_system_prompt_is_counted(self):
        counter = get_counter("anthropic", "claude-opus-5")
        shape = RequestShape(
            messages=[Message.user("hi")],
            system="You are a helpful ads assistant. " * 200,
            provider="anthropic",
        )
        assert counter.breakdown(shape).system > 500

    def test_anthropic_style_system_block_list(self):
        counter = get_counter("anthropic", "claude-opus-5")
        shape = RequestShape(
            messages=[],
            system=[{"type": "text", "text": "prompt text " * 100}],
            provider="anthropic",
        )
        assert counter.breakdown(shape).system > 100

    def test_tool_calls_are_counted(self):
        """tool_calls live off `content`; a counter that only walks content
        misses every argument payload the assistant emitted."""
        counter = get_counter("anthropic", "claude-opus-5")
        bare = Message(role=MessageRole.ASSISTANT, content="")
        with_calls = Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {
                    "id": "t1",
                    "function": {
                        "name": "google_ads_query",
                        "arguments": json.dumps({"query": "SELECT " + "x," * 500}),
                    },
                }
            ],
        )
        assert counter.count_messages([with_calls]) > counter.count_messages([bare]) + 200

    def test_unknown_block_type_is_not_free(self):
        """Counting an unrecognized block as zero is how a request sails past
        the threshold and 400s."""
        counter = get_counter("anthropic", "claude-opus-5")
        msg = Message(
            role=MessageRole.USER,
            content=[{"type": "something_new", "payload": "x" * 4000}],
        )
        assert counter.count_messages([msg]) > 500

    def test_json_counts_denser_than_prose(self):
        """The flat-4 estimator under-counted JSON, which is the dangerous
        direction — under-counting means compaction fires too late."""
        counter = get_counter("anthropic", "claude-opus-5")
        payload = json.dumps([{"campaign": "Brand", "cost": 1234.5}] * 100)
        prose = "the quick brown fox jumps over the lazy dog " * 60
        # Same character budget, different token cost.
        prose = prose[: len(payload)]
        assert counter.count_text(payload) > counter.count_text(prose)


class TestBreakdown:
    def test_floor_excludes_messages(self):
        counter = get_counter("anthropic", "claude-opus-5")
        shape = RequestShape(
            messages=[Message.user("hello " * 500)],
            system="sys " * 200,
            tools=[_tool("t1")],
            provider="anthropic",
        )
        b = counter.breakdown(shape)
        assert b.floor == b.system + b.tools
        assert b.total == b.floor + b.messages
        assert b.messages > 0

    def test_with_messages_preserves_floor(self):
        """Compaction rewrites messages and must leave the floor untouched."""
        counter = get_counter("anthropic", "claude-opus-5")
        shape = RequestShape(
            messages=[Message.user("x " * 1000)],
            system="sys " * 100,
            tools=[_tool("t1")],
            provider="anthropic",
        )
        compacted = shape.with_messages([Message.user("summary")])
        assert counter.breakdown(compacted).floor == counter.breakdown(shape).floor
        assert counter.breakdown(compacted).messages < counter.breakdown(shape).messages

    def test_to_dict_is_serializable(self):
        b = TokenBreakdown(system=10, tools=20, messages=30, calibration_factor=1.2)
        d = b.to_dict()
        json.dumps(d)  # must not raise
        assert d["floor"] == 30
        assert d["total"] == 60


class TestCalibration:
    def test_converges_toward_provider_count(self, calibrator):
        raw = 10_000
        for _ in range(10):
            calibrator.observe("anthropic", "claude-opus-5", raw, 11_500)
        factor = calibrator.factor_for("anthropic", "claude-opus-5")
        assert factor == pytest.approx(1.15, abs=0.02)

    def test_first_observation_seeds_directly(self, calibrator):
        """Easing in from 1.0 would waste the first real datapoint, which is
        strictly better information than the default."""
        calibrator.observe("openai", "gpt-4o", 1_000, 1_300)
        assert calibrator.factor_for("openai", "gpt-4o") == pytest.approx(1.3)

    def test_outlier_cannot_escape_the_clamp(self, calibrator):
        calibrator.observe("anthropic", "claude-opus-5", 1_000, 1_100)
        for _ in range(50):
            calibrator.observe("anthropic", "claude-opus-5", 1_000, 100_000)
        assert calibrator.factor_for("anthropic", "claude-opus-5") <= 2.0

    def test_tiny_requests_are_ignored(self, calibrator):
        """Small requests are dominated by fixed overhead we don't model, so
        their ratio is noise."""
        calibrator.observe("anthropic", "claude-opus-5", 10, 150)
        assert calibrator.factor_for("anthropic", "claude-opus-5") == 1.0

    def test_zero_estimate_does_not_divide_by_zero(self, calibrator):
        calibrator.observe("anthropic", "claude-opus-5", 0, 5_000)
        assert calibrator.factor_for("anthropic", "claude-opus-5") == 1.0

    def test_providers_do_not_share_factors(self, calibrator):
        calibrator.observe("anthropic", "claude-opus-5", 1_000, 1_400)
        assert calibrator.factor_for("openai", "gpt-4o") == 1.0

    def test_drift_reported_signed(self, calibrator):
        calibrator.observe("anthropic", "claude-opus-5", 1_150, 1_000)
        state = calibrator.state_for("anthropic", "claude-opus-5")
        assert state.drift_pct() == pytest.approx(15.0, abs=0.1)  # over-estimated
        assert state.is_grounded

    def test_ungrounded_state_reports_no_drift(self, calibrator):
        assert calibrator.state_for("anthropic", "claude-opus-5").drift_pct() is None


class TestCounterSelection:
    def test_openai_uses_tiktoken(self):
        from miiflow_agent.core.context.tokens import TiktokenCounter

        assert isinstance(get_counter("openai", "gpt-4o-mini"), TiktokenCounter)

    def test_other_providers_use_ratios(self):
        from miiflow_agent.core.context.tokens import LocalTokenCounter, TiktokenCounter

        for provider in ("anthropic", "gemini", "groq", "mistral"):
            counter = get_counter(provider, "some-model")
            assert isinstance(counter, LocalTokenCounter)
            assert not isinstance(counter, TiktokenCounter)

    def test_counter_identity_wins_over_shape(self):
        """A tiktoken counter must not apply another provider's residual
        error on top of exact tokenization."""
        from miiflow_agent.core.context.tokens import calibrator as global_calibrator

        global_calibrator.reset()
        global_calibrator.observe("anthropic", "m", 1_000, 1_500)
        shape = RequestShape(
            messages=[Message.user("hi " * 200)], provider="anthropic", model="m"
        )
        b = get_counter("openai", "gpt-4o").breakdown(shape)
        assert b.calibration_factor == 1.0
        assert not b.calibrated
        global_calibrator.reset()

    def test_tiktoken_is_exact_for_known_text(self):
        import tiktoken

        counter = get_counter("openai", "gpt-4o")
        text = "The quick brown fox jumps over the lazy dog."
        expected = len(tiktoken.get_encoding("o200k_base").encode(text))
        assert counter.count_text(text) == expected
