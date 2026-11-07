"""The public Grok catalog matches the direct API's current capabilities."""

from miiflow_agent.models import XAI_MODELS, XAI_PARAMETERS
from miiflow_agent.models.xai import prompt_length_multiplier, reasoning_effort_levels


def test_current_models_and_pricing():
    assert "grok-4.7" in XAI_MODELS
    flagship = XAI_MODELS["grok-4.7"]
    assert flagship.maximum_context_tokens == 500000
    assert (
        flagship.input_cost_hint,
        flagship.output_cost_hint,
        flagship.cache_read_cost_hint,
    ) == (2.0, 6.0, 0.5)
    assert flagship.support_images and flagship.supports_structured_outputs
    assert flagship.supports_tool_call and flagship.support_streaming
    assert "grok-2-mini" not in XAI_MODELS
    assert "grok-4-1-fast-reasoning" not in XAI_MODELS


def test_parameters_match_model_limits_and_effort():
    limits = next(p for p in XAI_PARAMETERS if p.field_name == "max_tokens")
    for model, config in XAI_MODELS.items():
        assert limits.max_value[model] == config.maximum_output_tokens
        effort = next(
            p
            for p in XAI_PARAMETERS
            if p.field_name == "reasoning_effort" and model in p.supported_models
        )
        assert effort.options == reasoning_effort_levels(model)
    assert "none" not in reasoning_effort_levels("grok-4.7")
    assert "none" in reasoning_effort_levels("grok-4.3")


def test_long_prompt_tier_uses_inclusive_boundary_and_known_models_only():
    for model in ("grok-4.3", "grok-4.6", "grok-4.7"):
        assert prompt_length_multiplier(model, None) == 1.0
        assert prompt_length_multiplier(model, 199_999) == 1.0
        assert prompt_length_multiplier(model, 200_000) == 2.0
        assert prompt_length_multiplier(model, 500_000) == 2.0
    assert prompt_length_multiplier("grok-unknown", 500_000) == 1.0
