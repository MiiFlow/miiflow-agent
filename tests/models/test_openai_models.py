"""Contract tests for OpenAI model capabilities, parameters, and pricing."""

import pytest

from miiflow_agent.models.openai import (
    OPENAI_MODELS,
    get_long_context_pricing_multipliers,
    get_parameters_for_model,
    get_token_param_name,
    is_gpt6_model,
    supports_json_mode,
    supports_temperature,
    tools_require_responses_api,
    unsupported_request_params,
)


def _parameter(model: str, field_name: str):
    return next(
        parameter
        for parameter in get_parameters_for_model(model)
        if parameter.field_name == field_name
    )


def _parameter_names(model: str) -> set[str]:
    return {parameter.field_name for parameter in get_parameters_for_model(model)}


def test_gpt56_pro_is_a_mode_not_a_model_slug():
    assert "gpt-5.6-sol-pro" not in OPENAI_MODELS
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert _parameter(model, "reasoning_mode").options == ["standard", "pro"]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            "gpt-5.6-sol",
            ["none", "low", "medium", "high", "xhigh", "max"],
        ),
        (
            "gpt-5.5",
            ["none", "low", "medium", "high", "xhigh"],
        ),
        (
            "gpt-5.4-mini",
            ["none", "low", "medium", "high", "xhigh"],
        ),
        ("gpt-5.5-pro", ["medium", "high", "xhigh"]),
        ("gpt-5.4-pro", ["medium", "high", "xhigh"]),
    ],
)
def test_reasoning_effort_options_are_model_specific(model, expected):
    assert _parameter(model, "reasoning_effort").options == expected


def test_sampling_penalties_are_only_exposed_for_gpt41():
    assert {"frequency_penalty", "presence_penalty"} <= _parameter_names("gpt-4.1")
    assert "frequency_penalty" not in _parameter_names("gpt-5.6-terra")
    assert "presence_penalty" not in _parameter_names("gpt-5.5")


def test_pro_capabilities_match_endpoint_contract():
    gpt55_pro = OPENAI_MODELS["gpt-5.5-pro"]
    assert gpt55_pro.api_path == "/responses"
    assert gpt55_pro.support_streaming is False
    assert gpt55_pro.supports_json_mode is True

    gpt54_pro = OPENAI_MODELS["gpt-5.4-pro"]
    assert gpt54_pro.api_path == "/responses"
    assert gpt54_pro.support_streaming is True
    assert gpt54_pro.supports_json_mode is False
    assert supports_json_mode("gpt-5.4-pro-2026-03-05") is False


def test_openai_cache_prices_cover_gpt56_writes_and_gpt55_pro_no_discount():
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        config = OPENAI_MODELS[model]
        assert config.cache_read_cost_hint == pytest.approx(
            config.input_cost_hint * 0.1
        )
        assert config.cache_write_cost_hint == pytest.approx(
            config.input_cost_hint * 1.25
        )

    assert OPENAI_MODELS["gpt-5.5-pro"].cache_read_cost_hint == 0


def test_gpt41_context_window_uses_exact_documented_limit():
    for model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"):
        assert OPENAI_MODELS[model].maximum_context_tokens == 1_047_576


@pytest.mark.parametrize(
    ("model", "input_tokens", "expected"),
    [
        ("gpt-5.6-sol", 272_000, (1.0, 1.0)),
        ("gpt-5.6-terra", 272_001, (2.0, 1.5)),
        ("gpt-5.5-pro", 500_000, (2.0, 1.5)),
        ("gpt-5.4", 300_000, (2.0, 1.5)),
        ("gpt-5.4-mini", 300_000, (1.0, 1.0)),
        ("gpt-4.1", 500_000, (1.0, 1.0)),
    ],
)
def test_long_context_pricing_multipliers(model, input_tokens, expected):
    assert get_long_context_pricing_multipliers(model, input_tokens) == expected


class TestGpt6Astra:
    """GPT-6 Astra's contract differs from GPT-5.6's in ways that fail as 400s."""

    @pytest.mark.parametrize(
        "spelling",
        ["gpt-6-astra", "gpt-6", "gpt-6-astra-fast", "gpt-6-fast", "gpt-6-astra-2026-09-03"],
    )
    def test_every_spelling_answers_the_same(self, spelling):
        # A latency suffix, the family alias and a dated snapshot are all the
        # same model. Answering them per-helper is how `-fast` came to be
        # recognised as GPT-6 while escaping the long-context surcharge.
        assert is_gpt6_model(spelling)
        assert not supports_temperature(spelling)
        assert get_token_param_name(spelling) == "max_completion_tokens"
        assert tools_require_responses_api(spelling)
        assert get_long_context_pricing_multipliers(spelling, 300_000) == (2.0, 1.5)
        assert unsupported_request_params(spelling) == {
            "logprobs",
            "prompt_cache_retention",
            "temperature",
            "top_logprobs",
            "top_p",
        }

    def test_effort_scale_drops_none(self):
        # Astra rejects `none`/`minimal`; GPT-5.6 still takes `none`. Offering a
        # level the model refuses puts a 400 behind a UI dropdown.
        assert _parameter("gpt-6-astra", "reasoning_effort").options == [
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        assert "none" in _parameter("gpt-5.6-sol", "reasoning_effort").options

    def test_temperature_is_not_offered(self):
        assert "temperature" not in _parameter_names("gpt-6-astra")
        assert "max_tokens" not in _parameter_names("gpt-6-astra")

    def test_models_with_no_removals_report_none(self):
        # The drop list must not leak onto models that still accept these.
        for model in ("gpt-5.6-sol", "gpt-5.4", "gpt-4.1"):
            assert unsupported_request_params(model) == frozenset()

    def test_pricing_matches_published_rates(self):
        config = OPENAI_MODELS["gpt-6-astra"]
        assert (config.input_cost_hint, config.output_cost_hint) == (10.0, 50.0)
        assert config.cache_read_cost_hint == 1.0
        assert config.cache_write_cost_hint == 12.5
        assert config.maximum_context_tokens == 1_050_000
        assert config.maximum_output_tokens == 128_000


class TestGpt6SolAndLuna:
    """Sol and Luna share Astra's contract everywhere EXCEPT the effort scale."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-6-sol-fast",
            "gpt-6-luna-fast",
            "gpt-6-sol-2026-09-22",
        ],
    )
    def test_every_spelling_answers_the_same(self, spelling):
        # Same reasoning as the Astra case: the latency suffix and a dated
        # snapshot are the same model, and a helper that disagrees is how a
        # request goes out carrying a parameter the model rejects.
        assert is_gpt6_model(spelling)
        assert not supports_temperature(spelling)
        assert get_token_param_name(spelling) == "max_completion_tokens"
        assert tools_require_responses_api(spelling)
        assert get_long_context_pricing_multipliers(spelling, 300_000) == (2.0, 1.5)
        assert unsupported_request_params(spelling) == {
            "logprobs",
            "prompt_cache_retention",
            "temperature",
            "top_logprobs",
            "top_p",
        }

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
    def test_effort_scale_keeps_none_unlike_astra(self, model):
        # The one place the family is NOT uniform. `none` is load-bearing here:
        # Chat Completions will only call function tools at effort `none`, so
        # withholding it forces every tool call onto the Responses API. Astra
        # returns a 400 on the same value, which is why the two are separate
        # sets rather than one.
        assert _parameter(model, "reasoning_effort").options == [
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        assert "none" not in _parameter("gpt-6-astra", "reasoning_effort").options

    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
    def test_sampling_and_completion_params_match_the_family(self, model):
        names = _parameter_names(model)
        assert "temperature" not in names
        assert "max_tokens" not in names
        assert "max_completion_tokens" in names
        # Pro is a mode on these ids, not a separate slug.
        assert _parameter(model, "reasoning_mode").options == ["standard", "pro"]

    def test_pro_is_a_mode_not_a_model_slug(self):
        assert "gpt-6-sol-pro" not in OPENAI_MODELS
        assert "gpt-6-luna-pro" not in OPENAI_MODELS

    def test_there_is_no_gpt6_terra(self):
        # OpenAI did not carry the GPT-5.6 mid tier into this generation. A
        # `gpt-6-terra` entry would be an id that 404s.
        assert "gpt-6-terra" not in OPENAI_MODELS

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-6-sol", (2.0, 10.0, 0.20, 2.5)),
            ("gpt-6-luna", (0.10, 0.50, 0.01, 0.125)),
        ],
    )
    def test_pricing_matches_published_rates(self, model, expected):
        config = OPENAI_MODELS[model]
        assert (
            config.input_cost_hint,
            config.output_cost_hint,
            config.cache_read_cost_hint,
            config.cache_write_cost_hint,
        ) == expected
        assert config.maximum_context_tokens == 1_050_000
        assert config.maximum_output_tokens == 128_000

    @pytest.mark.parametrize(
        ("newer", "older"),
        [("gpt-6-sol", "gpt-5.6-sol"), ("gpt-6-luna", "gpt-5.6-luna")],
    )
    def test_each_undercuts_the_gpt56_tier_it_succeeds(self, newer, older):
        # The reason both GPT-5.6 entries are now marked legacy. If a future
        # reprice inverts this, the descriptions saying "half the price" are
        # wrong and this fails rather than going stale silently.
        assert OPENAI_MODELS[newer].input_cost_hint < OPENAI_MODELS[older].input_cost_hint
        assert (
            OPENAI_MODELS[newer].output_cost_hint < OPENAI_MODELS[older].output_cost_hint
        )
