"""Contract tests for Anthropic model capabilities and resolution.

Both classes here pin a defect that a *second model in an existing family*
makes reachable, which is why they exist as tests rather than comments: neither
failure raises anything. One resolves a model to the wrong catalog entry and
bills against it; the other reports thinking as off on a model that always
thinks, so the caller's `max_tokens` is silently shared with a thinking block.
"""

import pytest

from miiflow_agent.models.anthropic import (
    ANTHROPIC_MODELS,
    _resolve_model_name,
    effort_levels,
    supports_structured_outputs,
    supports_temperature,
    thinking_disable_param,
    thinks_by_default,
)


class TestFableResolutionIsNotShadowedByItsPredecessor:
    """`claude-fable-5` is a substring of every Fable 5.1 identifier."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "claude-fable-5.1",
            "claude-fable-5-1",
            "anthropic.claude-fable-5-1",
            "us.anthropic.claude-fable-5-1",
        ],
    )
    def test_every_spelling_resolves_to_fable_51(self, spelling):
        # The third resolution tier is a substring match over the catalog in
        # insertion order, so Fable 5.1 must be entered first. Reversed, the
        # Bedrock and Vertex spellings resolve to Fable 5 and bill cache reads
        # at 4x the real rate — with no error anywhere.
        assert _resolve_model_name(spelling) == "claude-fable-5.1"

    @pytest.mark.parametrize(
        "spelling", ["claude-fable-5", "anthropic.claude-fable-5"]
    )
    def test_fable_5_still_resolves_to_itself(self, spelling):
        assert _resolve_model_name(spelling) == "claude-fable-5"

    def test_fable_51_is_ordered_before_fable_5(self):
        # The property the resolution above depends on, asserted directly so a
        # reordering fails here rather than as a mispriced request.
        keys = list(ANTHROPIC_MODELS)
        assert keys.index("claude-fable-5.1") < keys.index("claude-fable-5")


class TestFable51Contract:
    def test_cache_reads_bill_at_the_reduced_rate(self):
        # 2.5% of input, not the 10% every other Claude model charges. This is
        # the entire point of the 5.1 release.
        config = ANTHROPIC_MODELS["claude-fable-5.1"]
        assert config.cache_read_cost_hint == 0.25
        assert config.input_cost_hint == 10.0
        assert config.output_cost_hint == 50.0
        assert ANTHROPIC_MODELS["claude-fable-5"].cache_read_cost_hint == 1.0

    def test_thinking_is_on_and_cannot_be_turned_off(self):
        # The check this replaces was `name == "claude-fable-5"`, which answered
        # False here. A caller told thinking is off budgets max_tokens for text
        # alone and gets stop_reason=max_tokens with no text at all.
        assert thinks_by_default("claude-fable-5.1")
        assert thinks_by_default("claude-fable-5")
        assert thinking_disable_param("claude-fable-5.1") is None

    def test_structured_outputs_stay_native(self):
        # The tool-based JSON fallback forces `tool_choice`, and forced tool use
        # returns an error on this model — so declaring False here would make
        # every schema request fail rather than merely loosen the schema.
        assert supports_structured_outputs("claude-fable-5.1")

    def test_sampling_parameters_are_refused(self):
        assert not supports_temperature("claude-fable-5.1")

    def test_all_five_effort_levels_are_accepted(self):
        assert effort_levels("claude-fable-5.1") == (
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )
