"""Contract tests for Gemini model pricing and parameter behaviour."""

import pytest

from miiflow_agent.models.google import (
    GOOGLE_MODELS,
    THINKING_LEVELS,
    get_token_param_name,
    supports_temperature,
    thinking_level_for,
    thinking_levels,
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
    # 0.05x IS THE WHOLE GEMINI 4 LINE, NOT AN ARGON QUIRK. Argon was the first
    # Google model here to break the 0.1x default, and when Carbon and Hydrogen
    # were catalogued on November 16, 2026 both arrived at the same ratio —
    # so read this as "Gemini 4 reads cache at 5%, Gemini 3.x at 10%", and
    # expect the next Gemini 4 model to need an entry here too.
    #
    # Each is a published figure rather than a misread column:
    #   * gemini-4-argon  — $2 input against $0.10 cached through November 30,
    #     2026, and $4 against $0.20 from December 1. 0.05x on both sides of the
    #     step-up, so the RATIO survives the price change even though the rates
    #     do not. Google describes the same thing as "95% off" on the Gemini API.
    #   * gemini-4-carbon — same rates and the same December 1 step-up as
    #     Argon, so the same ratio holds through it.
    #   * gemini-4-hydrogen — $1.50 input against $0.075 cached, and these
    #     are STANDARD rates with no announced step-up, so unlike the two above
    #     this pair of numbers is not expected to move at all.
    #
    # Note the collision this test cannot see: Hydrogen's $0.075 cached rate is
    # the same figure as gemini-3.8-flash's and gemini-3.6-flash's, but it is
    # 0.05x of Hydrogen's $1.50 input and 0.1x of their $0.75. Identical cache
    # rate, different ratio, different tier.
    "gemini-4-argon": 0.05,
    "gemini-4-carbon": 0.05,
    "gemini-4-hydrogen": 0.05,
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
    for name in ("gemini-4-carbon", "gemini-4-hydrogen", "gemini-4-argon",
                 "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"):
        assert supports_temperature(name) is False, name
    # The 3.1 pair are the last models here that a temperature actually
    # reaches; gemini-3.5-flash was the third until it was removed.
    for name in ("gemini-3.1-pro", "gemini-3.1-flash-lite"):
        assert supports_temperature(name) is True, name


def test_every_model_uses_max_output_tokens():
    for name, config in GOOGLE_MODELS.items():
        assert config.token_param_name == "max_output_tokens", name
        assert get_token_param_name(name) == "max_output_tokens", name


def test_gemini_4_carbon_is_catalogued_at_its_introductory_rate():
    # Carbon went generally available on the PAID Gemini API on November 10,
    # 2026, which is what four earlier audits were waiting for — it was
    # Fairwind-gated before that, and a published rate card was deliberately
    # NOT treated as grounds to catalogue it, because a priced id nobody can
    # call is a 404 with a number attached. The rates below are the
    # INTRODUCTORY tier and expire on the same date as Argon's: from
    # December 1, 2026 Google charges $4 / $20 with cached input at $0.20. The
    # first audit after November 30 must double all three on BOTH models.
    config = GOOGLE_MODELS["gemini-4-carbon"]
    assert config.model_identifier == "models/gemini-4-carbon"
    assert (config.input_cost_hint, config.output_cost_hint) == (2.0, 10.0)
    assert config.cache_read_cost_hint == 0.10


def test_gemini_4_carbon_shares_argons_non_streaming_output_cap():
    # Same three-number shape as Argon, same reason for carrying the smallest:
    # GeminiClient's non-streaming path is a direct POST to
    # `:generateContent`, which tops out at 32,768.
    config = GOOGLE_MODELS["gemini-4-carbon"]
    assert config.maximum_context_tokens == 1048576
    assert config.maximum_output_tokens == 32768


def test_gemini_4_hydrogen_carries_standard_not_introductory_rates():
    # THE ONE GEMINI 4 MODEL WITH NO SCHEDULED STEP-UP. Argon and Carbon both
    # double on December 1, 2026 and the 3.x Flash tier doubles on January 1,
    # 2027; Hydrogen's $1.50/$7.50 is its standard price. The audit that
    # finally doubles the dated rates must leave this entry alone, which is
    # what this test exists to catch.
    config = GOOGLE_MODELS["gemini-4-hydrogen"]
    assert config.model_identifier == "models/gemini-4-hydrogen"
    assert (config.input_cost_hint, config.output_cost_hint) == (1.50, 7.50)
    assert config.cache_read_cost_hint == 0.075


def test_gemini_4_hydrogen_allows_twice_the_output_of_carbon_and_argon():
    # Hydrogen does NOT share the Gemini 4 non-streaming cap: Google caps each
    # generation segment at 65,536 whether streaming or not and returns a
    # continuation token past it, where Argon and Carbon stop a non-streaming
    # response at 32,768. Assuming family uniformity here would halve the
    # usable output of every Hydrogen call.
    hydrogen = GOOGLE_MODELS["gemini-4-hydrogen"]
    assert hydrogen.maximum_context_tokens == 1048576
    assert hydrogen.maximum_output_tokens == 65536
    for sibling in ("gemini-4-argon", "gemini-4-carbon"):
        assert GOOGLE_MODELS[sibling].maximum_output_tokens == 32768, sibling


def test_hydrogen_cache_rate_collides_with_the_flash_tier_at_a_different_ratio():
    # $0.075 is the cached rate on Hydrogen AND on both 3.x Flash models, but
    # it is 0.05x of Hydrogen's $1.50 input and 0.1x of their $0.75. The equal
    # figures are a coincidence of two different tiers, so neither entry may be
    # "corrected" to match the other.
    hydrogen = GOOGLE_MODELS["gemini-4-hydrogen"]
    flash = GOOGLE_MODELS["gemini-3.8-flash"]
    assert hydrogen.cache_read_cost_hint == flash.cache_read_cost_hint
    assert hydrogen.input_cost_hint == 2 * flash.input_cost_hint


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


# --- thinking levels ---------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GOOGLE_MODELS))
def test_every_catalogued_model_declares_its_thinking_levels(name):
    # A model missing here gets no thinkingLevel sent, which on 3.1 Pro or
    # Hydrogen means full-depth thinking whatever `effort` says.
    levels = thinking_levels(name)
    assert levels, f"{name} has no thinking levels"
    assert set(levels) <= set(THINKING_LEVELS)
    assert list(levels) == sorted(levels, key=THINKING_LEVELS.index)


@pytest.mark.parametrize(
    "model, expected",
    [
        # What Kopperfield's panel reader calls: the preview id resolves to
        # 3.1 Pro, which has no `minimal`.
        ("gemini-3.1-pro-preview", ("low", "medium", "high")),
        ("models/gemini-3.1-pro", ("low", "medium", "high")),
        ("GEMINI-4-ARGON", ("low", "medium", "high")),
        # Routed ids take the scale of the model that answers them.
        ("gemini-3.7-flash", ("low", "medium", "high")),
        ("gemini-3.5-flash", ("minimal", "low", "medium", "high")),
        # Release suffixes name the same model, routed ids included.
        ("gemini-3.1-pro-001", ("low", "medium", "high")),
        ("gemini-4-argon-latest", ("low", "medium", "high")),
        ("gemini-3.1-pro-preview-customtools", ("low", "medium", "high")),
        ("gemini-3.5-flash-preview-09-2026", ("minimal", "low", "medium", "high")),
    ],
)
def test_thinking_levels_resolve_aliases_previews_and_routed_ids(model, expected):
    assert thinking_levels(model) == expected


@pytest.mark.parametrize(
    "model",
    [
        # Contains a catalog name but takes only minimal/high: a substring
        # match would hand it `low`, which is a 400.
        "gemini-3.1-flash-lite-image",
        "gemini-9-unknown",
        "",
    ],
)
def test_uncatalogued_models_have_no_thinking_levels(model):
    assert thinking_levels(model) == ()


def test_only_hydrogen_takes_xhigh_and_minimal_stays_off_pro_and_gemini_4():
    takes_xhigh = {name for name in GOOGLE_MODELS if "xhigh" in thinking_levels(name)}
    assert takes_xhigh == {"gemini-4-hydrogen"}
    for name in ("gemini-3.1-pro", "gemini-3.8-flash", "gemini-4-argon", "gemini-4-carbon",
                 "gemini-4-hydrogen"):
        assert "minimal" not in thinking_levels(name)


@pytest.mark.parametrize(
    "model, effort, expected",
    [
        ("gemini-3.1-pro", "medium", "medium"),
        ("gemini-3.1-pro", "minimal", "low"),  # below the scale: its floor
        ("gemini-3.1-pro", "none", "low"),
        ("gemini-3.6-flash", "none", "minimal"),
        ("gemini-3.1-pro", "xhigh", None),  # above the scale: not sent
        ("gemini-3.1-pro", "max", None),
        ("gemini-9-unknown", "low", None),  # uncatalogued: not sent
        ("gemini-3.1-pro", None, None),
    ],
)
def test_thinking_level_for_maps_effort_onto_the_scale(model, effort, expected):
    assert thinking_level_for(model, effort) == expected
