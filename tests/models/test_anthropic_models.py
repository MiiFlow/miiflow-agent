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
    supports_thinking,
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


class TestOpus55ResolutionIsNotShadowedByItsPredecessor:
    """`claude-opus-5` is a substring of every Opus 5.5 identifier."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "claude-opus-5.5",
            "claude-opus-5-5",
            "anthropic.claude-opus-5-5",
            "us.anthropic.claude-opus-5-5",
        ],
    )
    def test_every_spelling_resolves_to_opus_55(self, spelling):
        # Same trap as Fable 5.1 above, one tier down. Reversed, an Opus 5.5
        # request bills at Opus 5's $5/$25 and, worse, is told thinking can be
        # disabled — which Opus 5.5 rejects with a 400 at every effort level.
        assert _resolve_model_name(spelling) == "claude-opus-5.5"

    @pytest.mark.parametrize(
        "spelling", ["claude-opus-5", "anthropic.claude-opus-5"]
    )
    def test_opus_5_still_resolves_to_itself(self, spelling):
        assert _resolve_model_name(spelling) == "claude-opus-5"

    def test_opus_55_is_ordered_before_opus_5(self):
        keys = list(ANTHROPIC_MODELS)
        assert keys.index("claude-opus-5.5") < keys.index("claude-opus-5")


class TestOpus55Contract:
    def test_cache_reads_bill_at_the_reduced_rate(self):
        # 5% of input, between Fable 5.1's 2.5% and the 10% every other Claude
        # model charges.
        config = ANTHROPIC_MODELS["claude-opus-5.5"]
        assert config.cache_read_cost_hint == 0.20
        assert config.input_cost_hint == 4.0
        assert config.output_cost_hint == 20.0
        assert ANTHROPIC_MODELS["claude-opus-5"].cache_read_cost_hint == 0.5

    def test_thinking_is_on_and_cannot_be_turned_off_at_any_effort(self):
        # The difference from Opus 5, and the one that 400s: Opus 5 accepts
        # "disabled" below xhigh, Opus 5.5 at no level at all.
        assert thinks_by_default("claude-opus-5.5")
        for effort in (None, "low", "medium", "high", "xhigh", "max"):
            assert thinking_disable_param("claude-opus-5.5", effort) is None
        assert thinking_disable_param("claude-opus-5", "low") == {"type": "disabled"}

    def test_structured_outputs_stay_native(self):
        # Forced tool use returns an error here, so the forced-`tool_choice`
        # JSON fallback would fail every schema request rather than loosen it.
        assert supports_structured_outputs("claude-opus-5.5")

    def test_sampling_parameters_are_refused(self):
        assert not supports_temperature("claude-opus-5.5")

    def test_all_five_effort_levels_are_accepted(self):
        assert effort_levels("claude-opus-5.5") == (
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )


class TestSonnet55ResolutionIsNotShadowedByItsPredecessor:
    """`claude-sonnet-5` is a substring of every Sonnet 5.5 identifier."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "claude-sonnet-5.5",
            "claude-sonnet-5-5",
            "anthropic.claude-sonnet-5-5",
            "us.anthropic.claude-sonnet-5-5",
        ],
    )
    def test_every_spelling_resolves_to_sonnet_55(self, spelling):
        # The third trap of this shape in one catalog. Reversed, a Sonnet 5.5
        # request resolves to Sonnet 5 and is handed `{"type": "disabled"}` to
        # turn thinking off — which Sonnet 5.5 rejects with a 400, so the whole
        # request fails rather than merely costing the wrong amount.
        assert _resolve_model_name(spelling) == "claude-sonnet-5.5"

    @pytest.mark.parametrize(
        "spelling", ["claude-sonnet-5", "anthropic.claude-sonnet-5"]
    )
    def test_sonnet_5_still_resolves_to_itself(self, spelling):
        assert _resolve_model_name(spelling) == "claude-sonnet-5"

    def test_sonnet_55_is_ordered_before_sonnet_5(self):
        keys = list(ANTHROPIC_MODELS)
        assert keys.index("claude-sonnet-5.5") < keys.index("claude-sonnet-5")


class TestSonnet55Contract:
    def test_prices_match_sonnet_5(self):
        # Sonnet 5.5 is the rare successor that changes nothing about price:
        # the gain is speed and fewer tool calls, not a cheaper rate card. A
        # "newer must cost more" assumption here would overbill every request.
        config = ANTHROPIC_MODELS["claude-sonnet-5.5"]
        assert config.input_cost_hint == 2.0
        assert config.output_cost_hint == 10.0
        assert config.cache_read_cost_hint == 0.2
        assert config.cache_write_cost_hint == 2.5

    def test_thinking_is_off_via_between_tools_not_disabled(self):
        # The breaking change from Sonnet 5, and the one that 400s: "disabled"
        # is rejected outright and `between_tools` replaces it. Returning
        # Sonnet 5's value here fails every request that asks for a cheap
        # text-only completion.
        assert thinks_by_default("claude-sonnet-5.5")
        for effort in (None, "low", "medium", "high"):
            assert thinking_disable_param("claude-sonnet-5.5", effort) == {
                "type": "between_tools"
            }

    def test_between_tools_is_withheld_at_the_top_two_efforts(self):
        # `between_tools` at xhigh or max is itself a 400, so at those levels
        # there is no off switch at all and the caller must budget max_tokens
        # for a thinking block it cannot suppress.
        for effort in ("xhigh", "max"):
            assert thinking_disable_param("claude-sonnet-5.5", effort) is None

    def test_between_tools_is_sent_bare(self):
        # `display`, `budget_tokens` or `block_binding` alongside it is a 400,
        # so the returned dict must stay single-keyed.
        assert thinking_disable_param("claude-sonnet-5.5", "medium") == {
            "type": "between_tools"
        }

    def test_manual_thinking_budgets_are_refused(self):
        # `thinking: {"type": "enabled", "budget_tokens": N}` is a 400 here, so
        # the model must stay out of the extended-thinking parameter's
        # supported list.
        assert not supports_thinking("claude-sonnet-5.5")

    def test_structured_outputs_stay_native(self):
        # Forced tool use returns an error here as it does on Opus 5.5, so the
        # forced-`tool_choice` JSON fallback would fail every schema request.
        assert supports_structured_outputs("claude-sonnet-5.5")

    def test_sampling_parameters_are_refused(self):
        assert not supports_temperature("claude-sonnet-5.5")

    def test_all_five_effort_levels_are_accepted(self):
        assert effort_levels("claude-sonnet-5.5") == (
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )
