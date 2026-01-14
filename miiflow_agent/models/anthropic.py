"""Anthropic model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType


ANTHROPIC_MODELS: Dict[str, ModelConfig] = {
    "claude-sonnet-4.5": ModelConfig(
        model_identifier="claude-sonnet-4-5-20250929",
        name="claude-sonnet-4.5",
        description="Anthropic's best model for complex agents and coding. Highest intelligence across most tasks with extended thinking capabilities.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=64000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=3.0,
        output_cost_hint=15.0,
    ),
    "claude-haiku-4.5": ModelConfig(
        model_identifier="claude-haiku-4-5-20251001",
        name="claude-haiku-4.5",
        description="Our fastest and most intelligent Haiku model. Delivers Sonnet-4-level coding performance at one-third the cost and more than twice the speed.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=64000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=1.0,
        output_cost_hint=5.0,
    ),
    "claude-opus-4.1": ModelConfig(
        model_identifier="claude-opus-4-1-20250805",
        name="claude-opus-4.1",
        description="Exceptional model for specialized complex tasks. Superior reasoning capabilities for advanced coding projects and deep research tasks.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=32000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=15.0,
        output_cost_hint=75.0,
    ),
    "claude-opus-4": ModelConfig(
        model_identifier="claude-opus-4-20250514",
        name="claude-opus-4",
        description="Highly capable model with exceptional reasoning and advanced coding capabilities. Sets new standards in complex reasoning.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=32000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=15.0,
        output_cost_hint=75.0,
    ),
    "claude-sonnet-4": ModelConfig(
        model_identifier="claude-sonnet-4-20250514",
        name="claude-sonnet-4",
        description="High-performance model with exceptional reasoning capabilities and efficiency. Excellent for coding and complex reasoning tasks.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=64000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=3.0,
        output_cost_hint=15.0,
    ),
    "claude-3-7-sonnet": ModelConfig(
        model_identifier="claude-3-7-sonnet-20250219",
        name="claude-3-7-sonnet",
        description="High-performance model with extended thinking capabilities. Excellent balance of intelligence and speed.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=64000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=3.0,
        output_cost_hint=15.0,
    ),
    "claude-3-5-haiku": ModelConfig(
        model_identifier="claude-3-5-haiku-20241022",
        name="claude-3-5-haiku",
        description="Fast model with intelligence at blazing speeds. Optimized for quick responses and cost-efficiency with vision support.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=False,
        maximum_context_tokens=200000,
        maximum_output_tokens=8192,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=0.8,
        output_cost_hint=4.0,
    ),
}


ANTHROPIC_PARAMETERS: list[ParameterConfig] = [
    ParameterConfig(
        field_name="temperature",
        display_name="Temperature",
        description="Amount of randomness injected into the response.",
        parameter_type=ParameterType.NUMBER,
        default_value=0.7,
        min_value=0,
        max_value=1,
        step=0.1,
    ),
    ParameterConfig(
        field_name="max_tokens",
        display_name="Max Tokens",
        description="An upper bound for the number of tokens that can be generated for a completion.",
        parameter_type=ParameterType.NUMBER,
        default_value=4096,
        min_value=1,
        max_value={
            "claude-sonnet-4.5": 64000,
            "claude-haiku-4.5": 64000,
            "claude-opus-4.1": 32000,
            "claude-opus-4": 32000,
            "claude-sonnet-4": 64000,
            "claude-3-7-sonnet": 64000,
            "claude-3-5-haiku": 8192,
            "default": 8192,
        },
        step=4,
    ),
]


def _get_thinking_models() -> list[str]:
    """Get list of models that support extended thinking (reasoning=True)."""
    return [name for name, config in ANTHROPIC_MODELS.items() if config.reasoning]


def _get_structured_output_models() -> list[str]:
    """Get list of models that support structured outputs."""
    models = []
    for name, config in ANTHROPIC_MODELS.items():
        if config.supports_structured_outputs:
            models.append(name)
            models.append(config.model_identifier)
    return models


# Add thinking_enabled parameter with dynamically derived supported models
ANTHROPIC_PARAMETERS.append(
    ParameterConfig(
        field_name="thinking_enabled",
        display_name="Extended Thinking",
        description="Enable extended thinking mode for deeper reasoning.",
        parameter_type=ParameterType.BOOLEAN,
        default_value=False,
        supported_models=_get_thinking_models(),
    )
)


def supports_structured_outputs(model: str) -> bool:
    """Check if model supports native structured outputs.

    Checks the model's supports_structured_outputs field from ANTHROPIC_MODELS.

    Args:
        model: The model identifier (can be full identifier or alias)

    Returns:
        True if model supports native structured outputs
    """
    # Check exact match first
    if model in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[model].supports_structured_outputs

    # Check if model identifier matches any config's model_identifier
    for config in ANTHROPIC_MODELS.values():
        if config.model_identifier == model:
            return config.supports_structured_outputs

    # Check partial match (for versioned models like claude-sonnet-4-5-20250929)
    for name, config in ANTHROPIC_MODELS.items():
        if name in model or config.model_identifier in model:
            return config.supports_structured_outputs

    return False


def supports_thinking(model: str) -> bool:
    """Check if model supports extended thinking mode.

    Checks the model's reasoning field from ANTHROPIC_MODELS.

    Args:
        model: The model identifier

    Returns:
        True if model supports extended thinking
    """
    # Check exact match first
    if model in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[model].reasoning

    # Check if model identifier matches any config's model_identifier
    for config in ANTHROPIC_MODELS.values():
        if config.model_identifier == model:
            return config.reasoning

    # Check partial match
    model_lower = model.lower()
    for name, config in ANTHROPIC_MODELS.items():
        if name in model_lower or config.model_identifier in model_lower:
            return config.reasoning

    return False


def supports_native_mcp(model: str) -> bool:
    """Check if model supports native MCP via the beta API.

    Native MCP allows the Anthropic API to connect directly to MCP servers
    and execute tools server-side, rather than requiring client-side handling.

    All Claude models support native MCP via the mcp-client-2025-04-04 beta.

    Args:
        model: The model identifier

    Returns:
        True if model supports native MCP (all Claude models do)
    """
    # All Claude models support native MCP via beta API
    # Check if it's a known Claude model
    if model in ANTHROPIC_MODELS:
        return True

    # Check if model identifier matches any config
    for config in ANTHROPIC_MODELS.values():
        if config.model_identifier == model:
            return True

    # Check partial match for Claude models
    model_lower = model.lower()
    if "claude" in model_lower:
        return True

    return False
