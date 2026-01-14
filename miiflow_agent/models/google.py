"""Google Gemini model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# All Gemini models use max_output_tokens
# and all current models support temperature

GOOGLE_MODELS: Dict[str, ModelConfig] = {
    "gemini-3-pro-preview": ModelConfig(
        model_identifier="models/gemini-3-pro-preview",
        name="gemini-3-pro-preview",
        description="Google's most advanced multimodal AI model released November 2025. Features state-of-the-art reasoning, agentic capabilities, and 1M token context. Leads in math, science, and multimodal benchmarks. Preview version.",
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
    "gemini-2.5-pro": ModelConfig(
        model_identifier="models/gemini-2.5-pro",
        name="gemini-2.5-pro",
        description="Google's highly capable thinking model with state-of-the-art reasoning for complex problems in code, math, and STEM. Supports 1M token context window.",
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
        description="Best model for price-performance with thinking capabilities. Optimized for large-scale processing, low-latency applications, and agentic use cases. Faster than 2.5 Pro with excellent efficiency.",
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
    "gemini-2.0-flash": ModelConfig(
        model_identifier="models/gemini-2.0-flash",
        name="gemini-2.0-flash",
        description="Fast, efficient Gemini model with native tool use and multimodal capabilities (text, images, video). Strong performance for everyday tasks at low cost.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=False,
        maximum_context_tokens=1048576,
        maximum_output_tokens=8192,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=0.10,
        output_cost_hint=0.40,
    ),
    "gemini-2.0-flash-lite": ModelConfig(
        model_identifier="models/gemini-2.0-flash-lite",
        name="gemini-2.0-flash-lite",
        description="Ultra cost-efficient Gemini model for high-volume applications. Simplified pricing with strong multimodal performance for routine tasks.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=False,
        maximum_context_tokens=1048576,
        maximum_output_tokens=8192,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=0.075,
        output_cost_hint=0.30,
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
            "gemini-3-pro-preview": 64000,
            "gemini-2.5-pro": 65536,
            "gemini-2.5-flash": 65536,
            "gemini-2.5-flash-lite": 65536,
            "gemini-2.0-flash": 8192,
            "gemini-2.0-flash-lite": 8192,
            "default": 8192,
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
