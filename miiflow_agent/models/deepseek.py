"""DeepSeek model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# DeepSeek uses max_completion_tokens parameter (like OpenAI reasoning models)

DEEPSEEK_MODELS: Dict[str, ModelConfig] = {
    "deepseek-chat": ModelConfig(
        model_identifier="deepseek-chat",
        name="deepseek-v-3",
        description="High-performance model with strong capabilities in reasoning, coding, and general language understanding.",
        support_images=False,
        support_files=False,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=False,
        supports_structured_outputs=False,
        reasoning=False,
        maximum_context_tokens=128000,
        maximum_output_tokens=8000,
        token_param_name="max_completion_tokens",
        supports_temperature=True,
        input_cost_hint=0.07,
        output_cost_hint=1.10,
    ),
}


DEEPSEEK_PARAMETERS: list[ParameterConfig] = [
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
        field_name="frequency_penalty",
        display_name="Frequency Penalty",
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
        parameter_type=ParameterType.NUMBER,
        default_value=0,
        min_value=-2,
        max_value=2,
        step=0.1,
    ),
    ParameterConfig(
        field_name="presence_penalty",
        display_name="Presence Penalty",
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
        parameter_type=ParameterType.NUMBER,
        default_value=0,
        min_value=-2,
        max_value=2,
        step=0.1,
    ),
    ParameterConfig(
        field_name="max_completion_tokens",
        display_name="Max Completion Tokens",
        description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.",
        parameter_type=ParameterType.NUMBER,
        default_value=8000,
        min_value=1,
        max_value={"deepseek-chat": 8000, "default": 8000},
        step=10,
    ),
]


def get_token_param_name(model: str) -> str:
    """Get the correct token parameter name for a DeepSeek model.

    DeepSeek uses max_completion_tokens similar to OpenAI reasoning models.

    Args:
        model: The model identifier

    Returns:
        The API parameter name to use for max tokens
    """
    return "max_completion_tokens"
