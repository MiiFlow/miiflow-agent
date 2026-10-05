"""Contract tests for Gemini model pricing and parameter behaviour."""

import pytest

from miiflow_agent.models.google import (
    GOOGLE_MODELS,
    get_token_param_name,
    supports_temperature,
)

# Google's standard context-caching multiplier: a cache read costs a tenth of
# the model's base input rate. Google publishes a per-model figure rather than a
# ratio, and the ratio is what every catalogued figure was cross-checked against
# when the official pricing table could not be fetched — a new model whose
# quoted cache rate does not hold it is usually the batch or the
# post-introductory column misread.
#
# It is a default, NOT an invariant, which is why the exceptions are a mapping
# rather than something to bend a rate to fit. A future audit that finds a real
# non-conforming rate (Anthropic already publishes 0.025x and 0.05x tiers, and
# Google prices 3.1 Flash-Lite's AUDIO cache reads at 0.1x a different base)
# records it here and keeps the true number in the catalog. Entering a wrong
# figure to keep this test green is the one thing it must not cause.
_CACHE_READ_RATIO = 0.1
_CACHE_READ_RATIO_EXCEPTIONS: dict[str, float] = {}


@pytest.mark.parametrize("name", sorted(GOOGLE_MODELS))
def test_cached_input_holds_googles_standard_ratio(name):
    config = GOOGLE_MODELS[name]
    ratio = _CACHE_READ_RATIO_EXCEPTIONS.get(name, _CACHE_READ_RATIO)
    assert config.cache_read_cost_hint == pytest.approx(
        config.input_cost_hint * ratio
    ), name


def test_tiered_and_conditional_cache_rates_are_not_hidden_by_the_flat_hint():
    # gemini-3.1-pro's $0.20 is the under-200K tier; above it Google charges
    # $0.40, and nothing in ModelConfig can express the tier — the same gap its
    # $2/$12 input price already has, since get_long_context_pricing_multipliers
    # is applied for OpenAI only. Pinned so the figure reads as the lower tier
    # rather than as the model's only rate.
    assert GOOGLE_MODELS["gemini-3.1-pro"].cache_read_cost_hint == 0.20
    assert GOOGLE_MODELS["gemini-3.1-pro"].input_cost_hint == 2.0


@pytest.mark.parametrize("name", sorted(GOOGLE_MODELS))
def test_every_model_declares_a_cache_read_rate(name):
    # 0.0 means "undeclared", which bills cached tokens at the full input rate.
    assert GOOGLE_MODELS[name].cache_read_cost_hint > 0, name


def test_the_three_current_flash_models_share_one_price():
    flash = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash")
    prices = {
        (
            GOOGLE_MODELS[name].input_cost_hint,
            GOOGLE_MODELS[name].output_cost_hint,
            GOOGLE_MODELS[name].cache_read_cost_hint,
        )
        for name in flash
    }
    # $0.75 / $3.75 / $0.075 introductory through December 31, 2026 — the input
    # and cache-read halves both double on January 1, 2027.
    assert prices == {(0.75, 3.75, 0.075)}


def test_gemini_3_5_flash_is_not_carried_at_its_batch_rate():
    # Google runs the Batch API at a flat 50% off, so $0.75/$4.50 is this
    # model's batch rate and has twice been mistaken for a price cut.
    config = GOOGLE_MODELS["gemini-3.5-flash"]
    assert (config.input_cost_hint, config.output_cost_hint) == (1.50, 9.00)


def test_sampling_params_are_dropped_from_gemini_3_6_onward():
    for name in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"):
        assert supports_temperature(name) is False, name
    for name in ("gemini-3.5-flash", "gemini-3.1-pro", "gemini-3.1-flash-lite"):
        assert supports_temperature(name) is True, name


def test_every_model_uses_max_output_tokens():
    for name, config in GOOGLE_MODELS.items():
        assert config.token_param_name == "max_output_tokens", name
        assert get_token_param_name(name) == "max_output_tokens", name


def test_gemini_4_argon_is_not_catalogued_while_it_is_fairwind_only():
    assert not any("argon" in name for name in GOOGLE_MODELS)
    assert not any("cyber" in name for name in GOOGLE_MODELS)
