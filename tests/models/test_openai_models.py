"""Contract tests for OpenAI model capabilities, parameters, and pricing."""

import pytest

from miiflow_agent.models.openai import (
    OPENAI_MODELS,
    get_long_context_pricing_multipliers,
    get_parameters_for_model,
    supports_json_mode,
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
