"""OpenRouter model configurations and model-family whitelist.

OpenRouter is a gateway service that provides access to models from multiple providers.
Miiflow intentionally exposes only the DeepSeek, GLM, and Grok families. All models use
the OpenAI-compatible API with the ``max_tokens`` parameter.
"""

import re
from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# Keep the provider slug and family prefix coupled so a model merely based on one of
# these families is not admitted. Concrete model slugs may use ``:free``; other
# OpenRouter routing modifiers (for example ``:online``) are intentionally excluded.
# Moving aliases use the documented ``~provider/family-latest`` shape. All suffixes are
# slash-free so another provider or endpoint cannot be smuggled into an allowed prefix.
_OPENROUTER_MODEL_SUFFIX = r"(?:[-.][a-z0-9](?:[a-z0-9._-]*[a-z0-9])?)?"
OPENROUTER_MODEL_WHITELIST: tuple[str, ...] = (
    rf"\A(?:~deepseek(?:-ai)?/deepseek-latest|"
    rf"deepseek(?:-ai)?/deepseek{_OPENROUTER_MODEL_SUFFIX}(?::free)?)\Z",
    rf"\A(?:~(?:z-ai|thudm)/glm-latest|(?:z-ai|thudm)/glm{_OPENROUTER_MODEL_SUFFIX}(?::free)?)\Z",
    rf"\A(?:~x-ai/grok-latest|x-ai/grok{_OPENROUTER_MODEL_SUFFIX}(?::free)?)\Z",
)

_OPENROUTER_MODEL_WHITELIST_REGEX = tuple(
    re.compile(pattern) for pattern in OPENROUTER_MODEL_WHITELIST
)


def is_openrouter_model_allowed(model_identifier: str) -> bool:
    """Return whether an OpenRouter model ID belongs to an allowed family."""
    if not isinstance(model_identifier, str):
        return False
    return any(pattern.fullmatch(model_identifier) for pattern in _OPENROUTER_MODEL_WHITELIST_REGEX)


# Fallback configurations used when OpenRouter's model catalog is unavailable. The
# normal path dynamically imports every catalog entry that passes the whitelist.
OPENROUTER_MODELS: Dict[str, ModelConfig] = {
    "deepseek/deepseek-v4-pro-0813": ModelConfig(
        model_identifier="deepseek/deepseek-v4-pro-0813",
        name="DeepSeek V4 Pro 0813",
        description="DeepSeek V4 Pro 0813 via OpenRouter.",
        support_images=False,
        support_files=False,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1048576,
        maximum_output_tokens=384000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=0.66,
        output_cost_hint=1.98,
        cache_read_cost_hint=0.022,
    ),
    "z-ai/glm-5.3": ModelConfig(
        model_identifier="z-ai/glm-5.3",
        name="GLM 5.3",
        description="Z.ai GLM 5.3 via OpenRouter.",
        support_images=False,
        support_files=False,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1310720,
        maximum_output_tokens=131072,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=1.40,
        output_cost_hint=4.40,
        cache_read_cost_hint=0.26,
    ),
    "x-ai/grok-4.6": ModelConfig(
        model_identifier="x-ai/grok-4.6",
        name="Grok 4.6",
        description="xAI Grok 4.6 via OpenRouter.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=500000,
        maximum_output_tokens=450000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=2.0,
        output_cost_hint=6.0,
        cache_read_cost_hint=0.5,
    ),
}


OPENROUTER_PARAMETERS: list[ParameterConfig] = [
    ParameterConfig(
        field_name="temperature",
        display_name="Temperature",
        description="Controls randomness in responses.",
        parameter_type=ParameterType.NUMBER,
        default_value=0.7,
        min_value=0,
        max_value=2,
        step=0.1,
    ),
    ParameterConfig(
        field_name="top_p",
        display_name="Top P",
        description="Nucleus sampling parameter.",
        parameter_type=ParameterType.NUMBER,
        default_value=1.0,
        min_value=0,
        max_value=1,
        step=0.1,
    ),
    ParameterConfig(
        field_name="max_tokens",
        display_name="Max Tokens",
        description="Maximum number of tokens to generate.",
        parameter_type=ParameterType.NUMBER,
        default_value=4096,
        min_value=1,
        max_value=450000,
        step=4,
    ),
    ParameterConfig(
        field_name="frequency_penalty",
        display_name="Frequency Penalty",
        description="Penalizes tokens based on frequency.",
        parameter_type=ParameterType.NUMBER,
        default_value=0,
        min_value=-2,
        max_value=2,
        step=0.1,
    ),
    ParameterConfig(
        field_name="presence_penalty",
        display_name="Presence Penalty",
        description="Penalizes tokens based on presence.",
        parameter_type=ParameterType.NUMBER,
        default_value=0,
        min_value=-2,
        max_value=2,
        step=0.1,
    ),
]
