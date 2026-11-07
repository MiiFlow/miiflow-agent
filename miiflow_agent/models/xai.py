"""Grok models served directly by xAI (verified November 6, 2026).

Sources: https://docs.x.ai/developers/models/grok-4.7 (also 4.6 and 4.3)
and https://docs.x.ai/developers/grok-4-7.
Prices are USD per million tokens below 200K prompt tokens. At >=200K,
xAI doubles all three rates; prompt_length_multiplier exposes that tier.
Grok has no separate text output limit: the context budget is the ceiling,
shared with input and reasoning. Files are not advertised by this Chat
Completions adapter. Native search, media generation and multi-agent endpoints
are outside its scope. Retired Grok 2/3/4 slugs are deliberately not advertised;
callers can still pass an older slug explicitly, subject to xAI redirects.
"""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

XAI_MODELS: Dict[str, ModelConfig] = {
    model: ModelConfig(
        model_identifier=model,
        name=model,
        description=description,
        support_images=True,
        support_files=False,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=context,
        maximum_output_tokens=context,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=input_cost,
        output_cost_hint=output_cost,
        cache_read_cost_hint=cache_cost,
    )
    for model, description, context, input_cost, output_cost, cache_cost in [
        (
            "grok-4.7",
            "Grok 4.7: coding, agentic tasks and knowledge work.",
            500000,
            2.0,
            6.0,
            0.50,
        ),
        (
            "grok-4.6",
            "Grok 4.6: previous generation for pinned workloads.",
            500000,
            2.0,
            6.0,
            0.50,
        ),
        (
            "grok-4.3",
            "Grok 4.3: fast tool calling with configurable reasoning.",
            1000000,
            1.25,
            2.50,
            0.20,
        ),
    ]
}


def prompt_length_multiplier(model: str, input_tokens: int | None) -> float:
    """Whole-request rate multiplier, using inclusive prompt/cache tokens."""
    if model in {"grok-4.3", "grok-4.6", "grok-4.7"} and (input_tokens or 0) >= 200_000:
        return 2.0
    return 1.0


def reasoning_effort_levels(model: str) -> list[str]:
    """Return only the effort levels documented for this model."""
    if model == "grok-4.3":
        return ["none", "low", "medium", "high", "xhigh"]
    if model in {"grok-4.6", "grok-4.7"}:
        return ["low", "medium", "high", "xhigh"]
    return []


XAI_PARAMETERS: list[ParameterConfig] = [
    ParameterConfig(
        field_name="temperature",
        display_name="Temperature",
        parameter_type=ParameterType.NUMBER,
        default_value=0.7,
        min_value=0,
        max_value=2,
        step=0.1,
    ),
    ParameterConfig(
        field_name="top_p",
        display_name="Top P",
        parameter_type=ParameterType.NUMBER,
        default_value=1.0,
        min_value=0,
        max_value=1,
        step=0.1,
    ),
    ParameterConfig(
        field_name="max_tokens",
        display_name="Max Tokens",
        description="Output budget, including reasoning; also bounded by remaining context.",
        parameter_type=ParameterType.NUMBER,
        default_value=4096,
        min_value=1,
        max_value={
            model: config.maximum_output_tokens for model, config in XAI_MODELS.items()
        },
        step=1,
    ),
    ParameterConfig(
        field_name="reasoning_effort",
        display_name="Reasoning Effort",
        parameter_type=ParameterType.SELECT,
        options=["low", "medium", "high", "xhigh"],
        supported_models=["grok-4.6", "grok-4.7"],
    ),
    ParameterConfig(
        field_name="reasoning_effort",
        display_name="Reasoning Effort",
        parameter_type=ParameterType.SELECT,
        options=reasoning_effort_levels("grok-4.3"),
        supported_models=["grok-4.3"],
    ),
]
