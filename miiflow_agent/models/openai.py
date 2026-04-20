"""OpenAI model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# Models that use max_completion_tokens instead of max_tokens
_REASONING_MODELS = {
    "o3",
    "o3-pro",
    "o3-mini",
    "o4-mini",
}

_GPT5_MODELS = {
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
}

# Models that don't support temperature parameter
_NO_TEMPERATURE_MODELS = _REASONING_MODELS | _GPT5_MODELS


OPENAI_MODELS: Dict[str, ModelConfig] = {
    # GPT-5.4 series (released March 2026) — current flagship line
    "gpt-5.4": ModelConfig(
        model_identifier="gpt-5.4",
        name="gpt-5.4",
        description="GPT-5.4 is OpenAI's flagship model (March 2026) with 1M context window, built-in computer use, and improved deep research. 33% fewer factual errors than GPT-5.2 and 75% on OSWorld-Verified.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=True,
        maximum_context_tokens=1050000,
        maximum_output_tokens=128000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=2.50,
        output_cost_hint=15.0,
    ),
    "gpt-5.4-pro": ModelConfig(
        model_identifier="gpt-5.4-pro",
        name="gpt-5.4-pro",
        description="GPT-5.4 Pro is OpenAI's highest-accuracy model for mission-critical agentic tasks. 1M context window with built-in computer use.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=True,
        maximum_context_tokens=1050000,
        maximum_output_tokens=128000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=30.0,
        output_cost_hint=180.0,
    ),
    "gpt-5.4-mini": ModelConfig(
        model_identifier="gpt-5.4-mini",
        name="gpt-5.4-mini",
        description="GPT-5.4 Mini is a smaller, faster GPT-5.4 variant with strong reasoning at lower cost. 400K context window. Released March 17, 2026.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=True,
        maximum_context_tokens=400000,
        maximum_output_tokens=128000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=0.75,
        output_cost_hint=4.50,
    ),
    "gpt-5.4-nano": ModelConfig(
        model_identifier="gpt-5.4-nano",
        name="gpt-5.4-nano",
        description="GPT-5.4 Nano is the most cost-effective GPT-5.4 variant, optimized for latency. 400K context window. Released March 17, 2026.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=True,
        maximum_context_tokens=400000,
        maximum_output_tokens=128000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=0.20,
        output_cost_hint=1.25,
    ),
    # GPT-4o series (standard models, use max_tokens)
    "gpt-4o": ModelConfig(
        model_identifier="gpt-4o",
        name="gpt-4o",
        description="Legacy — retired from ChatGPT February 13, 2026, still available via API. Multimodal model with vision and audio. Succeeded by GPT-5.4.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=False,
        maximum_context_tokens=128000,
        maximum_output_tokens=16384,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=2.5,
        output_cost_hint=10.0,
    ),
    "gpt-4o-mini": ModelConfig(
        model_identifier="gpt-4o-mini",
        name="gpt-4o-mini",
        description="Legacy — succeeded by GPT-5.4 Mini. Cost-efficient multimodal model, still available via API.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=False,
        maximum_context_tokens=128000,
        maximum_output_tokens=16384,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=0.15,
        output_cost_hint=0.60,
    ),
    # GPT-4.1 series (standard models, use max_tokens)
    "gpt-4.1": ModelConfig(
        model_identifier="gpt-4.1",
        name="gpt-4.1",
        description="Legacy — succeeded by GPT-5.4. General-purpose model with 1M token context. API cutoff October 14, 2026.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=False,
        maximum_context_tokens=1000000,
        maximum_output_tokens=32768,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=2.0,
        output_cost_hint=8.0,
    ),
    "gpt-4.1-mini": ModelConfig(
        model_identifier="gpt-4.1-mini",
        name="gpt-4.1-mini",
        description="Legacy — succeeded by GPT-5.4 Mini. Smaller GPT-4.1 with 1M token context.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=False,
        maximum_context_tokens=1000000,
        maximum_output_tokens=32768,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=0.40,
        output_cost_hint=1.60,
    ),
    "gpt-4.1-nano": ModelConfig(
        model_identifier="gpt-4.1-nano",
        name="gpt-4.1-nano",
        description="Legacy — succeeded by GPT-5.4 Nano. Smallest, fastest 4.1 variant.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=False,
        maximum_context_tokens=1000000,
        maximum_output_tokens=32768,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=0.10,
        output_cost_hint=0.40,
    ),
    # O-series reasoning models (use max_completion_tokens)
    "o3": ModelConfig(
        model_identifier="o3",
        name="o3",
        description="OpenAI's advanced reasoning model with strong math, science, and coding performance.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=100000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=2.00,
        output_cost_hint=8.00,
    ),
    "o3-mini": ModelConfig(
        model_identifier="o3-mini",
        name="o3-mini",
        description="o3-mini is a fast, cost-efficient reasoning model optimized for math, coding, and science.",
        support_images=False,
        support_files=False,
        support_streaming=True,
        supports_json_mode=False,
        supports_tool_call=True,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=100000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=1.10,
        output_cost_hint=4.40,
    ),
    "o3-pro": ModelConfig(
        model_identifier="o3-pro",
        name="o3-pro",
        description="OpenAI's premium reasoning model for maximum accuracy on the hardest problems in math, science, and coding.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=100000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=20.00,
        output_cost_hint=80.00,
    ),
    "o4-mini": ModelConfig(
        model_identifier="o4-mini",
        name="o4-mini",
        description="OpenAI's compact yet powerful reasoning model with excellent cost-efficiency for math, coding, and science tasks.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=100000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=1.10,
        output_cost_hint=4.40,
    ),
}


OPENAI_PARAMETERS: list[ParameterConfig] = [
    ParameterConfig(
        field_name="temperature",
        display_name="Temperature",
        description="Controls randomness in responses. Higher values (e.g., 0.8) make output more random.",
        parameter_type=ParameterType.NUMBER,
        default_value=1.0,
        min_value=0,
        max_value=2,
        step=0.1,
        unsupported_models=list(_NO_TEMPERATURE_MODELS),
    ),
    ParameterConfig(
        field_name="max_tokens",
        display_name="Max Tokens",
        description="Maximum tokens in response (for standard models).",
        parameter_type=ParameterType.NUMBER,
        min_value=1,
        max_value={
            "gpt-4o": 16384,
            "gpt-4o-mini": 16384,
            "gpt-4.1": 32768,
            "gpt-4.1-mini": 32768,
            "gpt-4.1-nano": 32768,
            "default": 16384,
        },
        unsupported_models=list(_NO_TEMPERATURE_MODELS),
    ),
    ParameterConfig(
        field_name="max_completion_tokens",
        display_name="Max Completion Tokens",
        description="Maximum tokens in response including reasoning tokens (for reasoning models).",
        parameter_type=ParameterType.NUMBER,
        min_value=1,
        max_value={
            "gpt-5.4": 128000,
            "gpt-5.4-pro": 128000,
            "gpt-5.4-mini": 128000,
            "gpt-5.4-nano": 128000,
            "o3": 100000,
            "o3-pro": 100000,
            "o3-mini": 100000,
            "o4-mini": 100000,
            "default": 100000,
        },
        supported_models=list(_NO_TEMPERATURE_MODELS),
    ),
    ParameterConfig(
        field_name="frequency_penalty",
        display_name="Frequency Penalty",
        description="Penalize new tokens based on their frequency in the text so far.",
        parameter_type=ParameterType.NUMBER,
        default_value=0,
        min_value=-2,
        max_value=2,
        step=0.1,
    ),
    ParameterConfig(
        field_name="presence_penalty",
        display_name="Presence Penalty",
        description="Penalize new tokens based on whether they appear in the text so far.",
        parameter_type=ParameterType.NUMBER,
        default_value=0,
        min_value=-2,
        max_value=2,
        step=0.1,
    ),
    ParameterConfig(
        field_name="reasoning_effort",
        display_name="Reasoning Effort",
        description="Constrains effort on reasoning for reasoning models.",
        parameter_type=ParameterType.SELECT,
        default_value="medium",
        options=["minimal", "low", "medium", "high"],
        supported_models=list(_REASONING_MODELS | _GPT5_MODELS),
    ),
    ParameterConfig(
        field_name="verbosity",
        display_name="Verbosity",
        description="Controls the verbosity of the model's response. Available for GPT-5 models to adjust response detail level.",
        parameter_type=ParameterType.SELECT,
        default_value="medium",
        options=["low", "medium", "high"],
        supported_models=list(_GPT5_MODELS),
    ),
]


def get_token_param_name(model: str) -> str:
    """Get the correct token parameter name for a model.

    OpenAI uses different parameter names for different model families:
    - Standard models (GPT-4o, GPT-4.1, etc.): max_tokens
    - Reasoning models (o1, o3, GPT-5): max_completion_tokens

    Args:
        model: The model identifier (e.g., "gpt-4o", "o1-mini")

    Returns:
        The API parameter name to use for max tokens
    """
    model_lower = model.lower()

    # Check exact match first
    if model_lower in OPENAI_MODELS:
        return OPENAI_MODELS[model_lower].token_param_name

    # Check prefix for versioned models (e.g., "o1-2024-12-17", "gpt-5.2-turbo")
    for prefix in ("o1", "o3", "o4", "gpt-5"):
        if model_lower.startswith(prefix):
            return "max_completion_tokens"

    return "max_tokens"


def supports_temperature(model: str) -> bool:
    """Check if model supports temperature parameter.

    Reasoning models and GPT-5 series don't support temperature.

    Args:
        model: The model identifier

    Returns:
        True if model supports temperature, False otherwise
    """
    model_lower = model.lower()

    # Check exact match first
    if model_lower in OPENAI_MODELS:
        return OPENAI_MODELS[model_lower].supports_temperature

    # Check prefix for versioned/unknown models
    for prefix in ("o1", "o3", "o4", "gpt-5"):
        if model_lower.startswith(prefix):
            return False

    return True


def supports_native_mcp(model: str) -> bool:
    """Check if model supports native MCP via the Responses API.

    Native MCP allows the OpenAI API to connect directly to MCP servers
    and execute tools server-side via the Responses API endpoint.

    Note: This requires using the Responses API (/v1/responses) instead
    of the Chat Completions API (/v1/chat/completions).

    Args:
        model: The model identifier

    Returns:
        True if model supports native MCP (most OpenAI models do)
    """
    # Most OpenAI models support native MCP via Responses API
    model_lower = model.lower()

    # Check exact match first
    if model_lower in OPENAI_MODELS:
        return True

    # Check common OpenAI model prefixes
    openai_prefixes = ("gpt-4", "gpt-5", "o1", "o3", "o4")
    for prefix in openai_prefixes:
        if model_lower.startswith(prefix):
            return True

    return False
