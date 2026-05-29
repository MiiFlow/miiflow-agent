"""Google Gemini model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# All Gemini models use max_output_tokens
# and all current models support temperature

GOOGLE_MODELS: Dict[str, ModelConfig] = {
    "gemini-3.5-flash": ModelConfig(
        model_identifier="models/gemini-3.5-flash",
        name="gemini-3.5-flash",
        description="Google's most capable Flash model (released May 19, 2026). Outperforms Gemini 3.1 Pro on coding and agentic benchmarks at Flash-tier pricing. 1M token context window.",
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
        input_cost_hint=1.50,
        output_cost_hint=9.00,
    ),
    "gemini-3.1-pro-preview": ModelConfig(
        model_identifier="models/gemini-3.1-pro-preview",
        name="gemini-3.1-pro-preview",
        description="Google's advanced Pro model with strong reasoning and 2x boost over Gemini 3 Pro. 1M token context window. Preview version.",
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
        input_cost_hint=2.0,
        output_cost_hint=12.0,
    ),
    "gemini-3-flash-preview": ModelConfig(
        model_identifier="models/gemini-3-flash-preview",
        name="gemini-3-flash-preview",
        description="Legacy — succeeded by Gemini 3.5 Flash (May 2026). Preview-era Flash model with 1M token context. Optimized for speed and cost-efficiency.",
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
        input_cost_hint=0.50,
        output_cost_hint=3.00,
    ),
    "gemini-2.5-pro": ModelConfig(
        model_identifier="models/gemini-2.5-pro",
        name="gemini-2.5-pro",
        description="Legacy — succeeded by Gemini 3.1 Pro. Capable thinking model for complex problems. Sunset June 17, 2026.",
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
        description="Legacy — succeeded by Gemini 3.5 Flash. Best price-performance model with thinking capabilities. Sunset June 17, 2026.",
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
        description="Legacy — succeeded by Gemini 3.1 Flash Lite. Entry-level thinking model with exceptional cost-efficiency. Sunset June 17, 2026.",
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
    "gemini-3.1-flash-lite": ModelConfig(
        model_identifier="models/gemini-3.1-flash-lite",
        name="gemini-3.1-flash-lite",
        description="Google's most cost-efficient Gemini 3 series model. Optimized for high-throughput, low-latency, cost-sensitive applications. GA as of May 2026.",
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
            "gemini-3.5-flash": 65536,
            "gemini-3.1-pro-preview": 65536,
            "gemini-3.1-flash-lite": 65536,
            "gemini-3-flash-preview": 65536,
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
