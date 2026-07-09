"""OpenAI model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# O-series reasoning models that use max_completion_tokens instead of max_tokens.
# Currently empty: o3 was retired after its June 2026 deprecation; the GPT-5.x
# reasoning models are tracked separately in _GPT5_MODELS below.
_REASONING_MODELS: set[str] = set()

# GPT-5.x reasoning models (use max_completion_tokens, no temperature). The
# GPT-5.6 Sol / Terra / Luna family reached general availability on July 9, 2026
# with confirmed API model ids; Sol is the current flagship. Sol Pro is Sol
# served with reasoning.mode=pro at the same per-token price.
_GPT5_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-sol-pro",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
}

# Models that don't support temperature parameter
_NO_TEMPERATURE_MODELS = _REASONING_MODELS | _GPT5_MODELS


OPENAI_MODELS: Dict[str, ModelConfig] = {
    # GPT-5.6 series (Sol / Terra / Luna) — generally available July 9, 2026.
    # Sol is the current flagship; gpt-5.6 is an API alias for gpt-5.6-sol.
    "gpt-5.6-sol": ModelConfig(
        model_identifier="gpt-5.6-sol",
        name="gpt-5.6-sol",
        description="GPT-5.6 Sol is OpenAI's flagship model (generally available July 9, 2026) and the highest-intelligence tier of the GPT-5.6 family, built for complex coding, reasoning, and long-horizon agentic work. 1M context window. Available in the API as gpt-5.6-sol (alias: gpt-5.6).",
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
        input_cost_hint=5.0,
        output_cost_hint=30.0,
    ),
    "gpt-5.6-sol-pro": ModelConfig(
        model_identifier="gpt-5.6-sol-pro",
        name="gpt-5.6-sol-pro",
        description="GPT-5.6 Sol Pro is GPT-5.6 Sol served with reasoning.mode=pro for maximum accuracy on the hardest agentic and reasoning tasks (July 9, 2026). Same per-token price as Sol, with higher latency and reasoning-token usage. 1M context window.",
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
        input_cost_hint=5.0,
        output_cost_hint=30.0,
    ),
    "gpt-5.6-terra": ModelConfig(
        model_identifier="gpt-5.6-terra",
        name="gpt-5.6-terra",
        description="GPT-5.6 Terra is the balanced mid-tier of the GPT-5.6 family (July 9, 2026), delivering strong reasoning and agentic performance at roughly half the cost of Sol. 1M context window.",
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
    "gpt-5.6-luna": ModelConfig(
        model_identifier="gpt-5.6-luna",
        name="gpt-5.6-luna",
        description="GPT-5.6 Luna is the fastest and most cost-efficient tier of the GPT-5.6 family (July 9, 2026), optimized for high-throughput, latency-sensitive workloads. 1M context window.",
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
        input_cost_hint=1.0,
        output_cost_hint=6.0,
    ),
    # GPT-5.5 series (released April 23, 2026) — succeeded by the GPT-5.6 family
    "gpt-5.5": ModelConfig(
        model_identifier="gpt-5.5",
        name="gpt-5.5",
        description="GPT-5.5 (released April 23, 2026) — previous flagship, succeeded by GPT-5.6 Sol. Strong coding and agentic model (82.7% on Terminal-Bench 2.0, 58.6% on SWE-Bench Pro) with a 1M context window, still available as a lower-cost alternative to the GPT-5.6 line.",
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
        input_cost_hint=5.0,
        output_cost_hint=30.0,
    ),
    "gpt-5.5-pro": ModelConfig(
        model_identifier="gpt-5.5-pro",
        name="gpt-5.5-pro",
        description="GPT-5.5 Pro (April 23, 2026) — high-accuracy model for mission-critical agentic and reasoning tasks; succeeded by GPT-5.6 Sol Pro. 1M context window.",
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
    # GPT-5.4 series (released March 2026) — two generations old, superseded by the GPT-5.6 family
    "gpt-5.4": ModelConfig(
        model_identifier="gpt-5.4",
        name="gpt-5.4",
        description="GPT-5.4 (March 2026) — two generations old, superseded by the GPT-5.6 family. Still available as a lower-cost option with 1M context window, built-in computer use, and improved deep research.",
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
        description="GPT-5.4 Pro (March 2026) — superseded by GPT-5.6 Sol Pro. High-accuracy model for mission-critical agentic tasks. 1M context window with built-in computer use.",
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
    # GPT-4.1 series (standard models, use max_tokens)
    "gpt-4.1": ModelConfig(
        model_identifier="gpt-4.1",
        name="gpt-4.1",
        description="Legacy — succeeded by GPT-5.4. General-purpose model with 1M token context. Retired from ChatGPT Feb 13, 2026; still available via the API.",
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
            "gpt-4.1": 32768,
            "gpt-4.1-mini": 32768,
            "gpt-4.1-nano": 32768,
            "default": 32768,
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
            "gpt-5.6-sol": 128000,
            "gpt-5.6-sol-pro": 128000,
            "gpt-5.6-terra": 128000,
            "gpt-5.6-luna": 128000,
            "gpt-5.5": 128000,
            "gpt-5.5-pro": 128000,
            "gpt-5.4": 128000,
            "gpt-5.4-pro": 128000,
            "gpt-5.4-mini": 128000,
            "gpt-5.4-nano": 128000,
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
        model: The model identifier (e.g., "gpt-5.5", "o4-mini")

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
