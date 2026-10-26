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
_CACHE_READ_RATIO_EXCEPTIONS: dict[str, float] = {
    # Gemini 4 Argon is the first Google model here to break the 0.1x default,
    # and it is a published figure, not a misread column: Vertex's global
    # standard tier lists $2 input against $0.10 cached through November 30,
    # 2026 and $4 against $0.20 from December 1 — 0.05x on both sides of the
    # step-up, so the ratio survives the price change even though the rates do
    # not. Google describes the same thing as "95% off" on the Gemini API.
    "gemini-4-argon": 0.05,
}


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


def test_the_current_flash_models_share_one_price():
    # 3.7 Flash used to be the third of these. Google deprecated it on
    # October 8, 2026 and routes its id to 3.8 Flash, so it is no longer
    # carried; 3.6 Flash stays because it was not deprecated.
    flash = ("gemini-3.8-flash", "gemini-3.6-flash")
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


def test_flash_models_are_not_carried_at_their_batch_rate():
    # Google runs the Batch API at a flat 50% off input and output, so every
    # rate here has a halved twin that reads like a plausible standard rate.
    # Two audits have already mistaken one for a price cut (the September 30,
    # 2026 audit carried the now-removed gemini-3.5-flash at its batch rate and
    # read it as evidence of a cut that never happened). The Flash tier is the
    # easiest to get wrong, because its introductory $0.75/$3.75 is itself
    # within a factor of two of the $1.50/$7.50 it reverts to.
    for name in ("gemini-3.8-flash", "gemini-3.6-flash"):
        config = GOOGLE_MODELS[name]
        assert (config.input_cost_hint, config.output_cost_hint) == (0.75, 3.75), name


def test_auto_routed_model_ids_are_not_in_the_catalog():
    # Google deprecated both of these on October 8, 2026 WITHOUT retiring the
    # endpoints: a request naming either succeeds and is silently answered by
    # another model (3.7 Flash -> 3.8 Flash, 3.5 Flash -> 3.6 Flash). A
    # ModelConfig promises "this id gets you this model at these rates", which
    # is false for a routed id — gemini-3.5-flash in particular was carried at
    # $1.50/$9.00 with temperature support while actually being served by a
    # model that bills $0.75/$3.75 and ignores sampling parameters. A routed id
    # is worse to carry than a retired one, which at least fails loudly, so
    # neither may come back without the routing being gone.
    for name in ("gemini-3.7-flash", "gemini-3.5-flash"):
        assert name not in GOOGLE_MODELS, name


def test_routed_ids_still_report_sampling_params_as_ineffective():
    # The other half of the removal, and the easy half to get wrong: both ids
    # are still CALLABLE (Google routes rather than rejects), and
    # supports_temperature() defaults to True for a model it does not
    # recognise. So taking them out of GOOGLE_MODELS without leaving them in
    # _NO_SAMPLING_PARAMS flips them False -> True and makes the client send a
    # `temperature` the answering model ignores. For gemini-3.5-flash this is
    # also stricter than the catalog entry was: it declared
    # supports_temperature=True, which stopped being true once its requests
    # began being served by 3.6 Flash.
    for name in ("gemini-3.7-flash", "gemini-3.5-flash"):
        assert supports_temperature(name) is False, name


def test_sampling_params_are_dropped_from_gemini_3_6_onward():
    for name in ("gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"):
        assert supports_temperature(name) is False, name
    # The 3.1 pair are the last models here that a temperature actually
    # reaches; gemini-3.5-flash was the third until it was removed.
    for name in ("gemini-3.1-pro", "gemini-3.1-flash-lite"):
        assert supports_temperature(name) is True, name


def test_every_model_uses_max_output_tokens():
    for name, config in GOOGLE_MODELS.items():
        assert config.token_param_name == "max_output_tokens", name
        assert get_token_param_name(name) == "max_output_tokens", name


def test_gemini_4_argon_is_catalogued_at_its_introductory_rate():
    # Argon went generally available on October 13, 2026 and is reachable with
    # an ordinary paid API key, which is what three earlier audits were waiting
    # for. The rates below are the INTRODUCTORY tier and expire: from
    # December 1, 2026 Google charges $4 / $20 with cached input at $0.20, so
    # the first audit after November 30 doubles all three or every Argon
    # request bills at half cost.
    config = GOOGLE_MODELS["gemini-4-argon"]
    assert config.model_identifier == "models/gemini-4-argon"
    assert (config.input_cost_hint, config.output_cost_hint) == (2.0, 10.0)
    assert config.cache_read_cost_hint == 0.10


def test_gemini_4_argon_caps_output_at_the_non_streaming_limit():
    # Three numbers, and the catalog carries the smallest on purpose. A
    # non-streaming direct-HTTP request returns at most 32,768 output tokens
    # (Gemini 3.x allowed 65,536 there); streaming reaches 65,536; and
    # 1,048,576 — which is also the shared context window — needs long decode
    # and a `continuation_token` loop. GeminiClient's non-streaming path is a
    # direct POST to `:generateContent`, so 65,536 here would advertise a
    # ceiling that path cannot reach.
    config = GOOGLE_MODELS["gemini-4-argon"]
    assert config.maximum_context_tokens == 1048576
    assert config.maximum_output_tokens == 32768


def test_fairwind_gated_cyber_variants_stay_out_of_the_catalog():
    # gemini-3.8-flash-cyber and gemini-4-argon-cyber-permissive are served
    # only to Google's Fairwind programme, so they would fail for every key
    # this catalog serves — Argon's own GA does not bring its cyber sibling in.
    assert not any("cyber" in name for name in GOOGLE_MODELS)
