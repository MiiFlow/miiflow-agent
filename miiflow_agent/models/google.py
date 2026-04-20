"""Google Gemini model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# All Gemini models use max_output_tokens
# and all current models support temperature

GOOGLE_MODELS: Dict[str, ModelConfig] = {
    "gemini-3.1-pro-preview": ModelConfig(
        model_identifier="models/gemini-3.1-pro-preview",
        name="gemini-3.1-pro-preview",
        description="Google's latest and most advanced multimodal AI model. Features improved reasoning, agentic capabilities, and 1M token context window. Preview version.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=64000,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=2.0,
        output_cost_hint=12.0,
    ),
    "gemini-3-flash-preview": ModelConfig(
        model_identifier="models/gemini-3-flash-preview",
        name="gemini-3-flash-preview",
        description="Google's fast and efficient Gemini 3 Flash model with strong multimodal capabilities and 1M token context. Optimized for speed and cost-efficiency. Preview version.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=64000,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=0.50,
        output_cost_hint=3.00,
    ),
    "gemini-2.5-pro": ModelConfig(
        model_identifier="models/gemini-2.5-pro",
        name="gemini-2.5-pro",
        description="Google's capable thinking model for complex problems. Earliest possible sunset June 17, 2026.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=1048576,
        maximum_output_tokens=65536,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=1.25,
        output_cost_hint=10.0,
    ),
    "gemini-2.5-flash": ModelConfig(
        model_identifier="models/gemini-2.5-flash",
        name="gemini-2.5-flash",
        description="Best price-performance model with thinking capabilities. Earliest possible sunset June 17, 2026.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=1048576,
        maximum_output_tokens=65536,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=0.30,
        output_cost_hint=2.50,
    ),
    "gemini-2.5-flash-lite": ModelConfig(
        model_identifier="models/gemini-2.5-flash-lite",
        name="gemini-2.5-flash-lite",
        description="Entry-level thinking model with exceptional cost-efficiency. Fastest Flash variant optimized for high-throughput, high-volume applications with strong quality.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=1048576,
        maximum_output_tokens=65536,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=0.10,
        output_cost_hint=0.40,
    ),
    "gemini-3.1-flash-lite-preview": ModelConfig(
        model_identifier="models/gemini-3.1-flash-lite-preview",
        name="gemini-3.1-flash-lite-preview",
        description="Google's frontier-class performance at reduced cost. Fast and efficient Gemini 3.1 Flash Lite model optimized for high-throughput, cost-sensitive applications. Preview version.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=64000,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=0.25,
        output_cost_hint=1.50,
    ),
}


GOOGLE_PARAMETERS: list[ParameterConfig] = [
    ParameterConfig(
        field_name="temperature",
        display_name="Temperature",
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
        parameter_type=ParameterType.NUMBER,
        default_value=0.5,
        min_value=0,
        max_value=2,
        step=0.1,
    ),
    ParameterConfig(
        field_name="top_p",
        display_name="Top P",
        description="The cumulative probability cutoff for token selection. Tokens are selected in descending probability order until the sum of their probabilities equals this value.",
        parameter_type=ParameterType.NUMBER,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
    ),
    ParameterConfig(
        field_name="top_k",
        display_name="Top K",
        description="The maximum number of top tokens to consider when sampling.",
        parameter_type=ParameterType.NUMBER,
        default_value=40,
        min_value=1,
        max_value=100,
        step=1,
    ),
    ParameterConfig(
        field_name="max_output_tokens",
        display_name="Max Output Tokens",
        description="The maximum number of tokens to generate in the response.",
        parameter_type=ParameterType.NUMBER,
        default_value=4096,
        min_value=1,
        max_value={
            "gemini-3.1-pro-preview": 64000,
            "gemini-3.1-flash-lite-preview": 64000,
            "gemini-3-flash-preview": 64000,
            "gemini-2.5-pro": 65536,
            "gemini-2.5-flash": 65536,
            "gemini-2.5-flash-lite": 65536,
            "default": 65536,
        },
        step=1,
    ),
]


def get_token_param_name(model: str) -> str:
    """Get the correct token parameter name for a Google model.

    All Gemini models use max_output_tokens.

    Args:
        model: The model identifier

    Returns:
        The API parameter name to use for max tokens
    """
    return "max_output_tokens"
