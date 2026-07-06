"""Anthropic model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

ANTHROPIC_MODELS: Dict[str, ModelConfig] = {
    "claude-fable-5": ModelConfig(
        model_identifier="claude-fable-5",
        name="claude-fable-5",
        description="Anthropic's most capable widely released model (generally available since June 9, 2026; API access was briefly suspended June 12–July 1, 2026 under a US export-control directive and has since been restored). Built for demanding reasoning and long-horizon agentic work, with always-on adaptive thinking, structured outputs, and a 1M context window.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=False,
        input_cost_hint=10.0,
        output_cost_hint=50.0,
    ),
    "claude-opus-4.8": ModelConfig(
        model_identifier="claude-opus-4-8",
        name="claude-opus-4.8",
        description="Powerful reasoning and coding model (released May 28, 2026). Improved reasoning, coding, and agentic performance over Opus 4.7. Features adaptive thinking, structured outputs, and fast mode. 1M context window.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=False,
        input_cost_hint=5.0,
        output_cost_hint=25.0,
    ),
    "claude-opus-4.7": ModelConfig(
        model_identifier="claude-opus-4-7",
        name="claude-opus-4.7",
        description="Legacy — succeeded by Claude Opus 4.8 (May 2026). Strong coding, reasoning, and agentic performance with adaptive thinking. 1M context window.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=False,
        input_cost_hint=5.0,
        output_cost_hint=25.0,
    ),
    "claude-opus-4.6": ModelConfig(
        model_identifier="claude-opus-4-6",
        name="claude-opus-4.6",
        description="Legacy — succeeded by Claude Opus 4.7 (April 2026). Supports adaptive thinking and 128K max output tokens. 1M context window.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=5.0,
        output_cost_hint=25.0,
    ),
    "claude-sonnet-5": ModelConfig(
        model_identifier="claude-sonnet-5",
        name="claude-sonnet-5",
        description="Anthropic's most agentic Sonnet model (released June 30, 2026), succeeding Sonnet 4.6 and closing much of the gap with Opus 4.8 on reasoning, tool use, and coding. Adaptive thinking is on by default; manual extended thinking and non-default temperature/top_p/top_k are rejected. 1M context window. Introductory pricing of $2/$10 per 1M input/output tokens applies through August 31, 2026, reverting to $3/$15 on September 1, 2026.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=False,
        input_cost_hint=2.0,
        output_cost_hint=10.0,
    ),
    "claude-sonnet-4.6": ModelConfig(
        model_identifier="claude-sonnet-4-6",
        name="claude-sonnet-4.6",
        description="Legacy — succeeded by Claude Sonnet 5 (June 2026). High-performance model with excellent balance of intelligence, speed, and cost. Features extended thinking, adaptive thinking, and structured outputs. 1M context window.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=3.0,
        output_cost_hint=15.0,
    ),
    "claude-haiku-4.5": ModelConfig(
        model_identifier="claude-haiku-4-5-20251001",
        name="claude-haiku-4.5",
        description="Anthropic's fastest and most intelligent Haiku model. Delivers Sonnet-4-level coding performance at one-third the cost and more than twice the speed.",
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
            "claude-fable-5": 128000,
            "claude-opus-4.8": 128000,
            "claude-opus-4.7": 128000,
            "claude-opus-4.6": 128000,
            "claude-sonnet-5": 128000,
            "claude-sonnet-4.6": 128000,
            "claude-haiku-4.5": 64000,
            "default": 8192,
        },
        step=4,
    ),
]


_NO_EXTENDED_THINKING = {
    "claude-fable-5",
    "claude-opus-4.8",
    "claude-opus-4.7",
    "claude-sonnet-5",
}


def _get_thinking_models() -> list[str]:
    """Get list of models that support extended thinking.

    Opus 4.7 uses adaptive thinking (always-on) instead of the explicit
    extended-thinking API parameter, so it is excluded here.
    """
    return [
        name
        for name, config in ANTHROPIC_MODELS.items()
        if config.reasoning and name not in _NO_EXTENDED_THINKING
    ]


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

    # Check partial match (for versioned models like claude-sonnet-4-6)
    for name, config in ANTHROPIC_MODELS.items():
        if name in model or config.model_identifier in model:
            return config.supports_structured_outputs

    return False


def supports_thinking(model: str) -> bool:
    """Check if model supports the explicit extended-thinking API parameter.

    Opus 4.7 uses adaptive thinking (always-on) and does NOT accept the
    ``thinking`` request parameter, so this returns False for it.

    Args:
        model: The model identifier

    Returns:
        True if model supports the extended-thinking parameter
    """

    def _check(name: str, config: ModelConfig) -> bool:
        return config.reasoning and name not in _NO_EXTENDED_THINKING

    # Check exact match first
    if model in ANTHROPIC_MODELS:
        return _check(model, ANTHROPIC_MODELS[model])

    # Check if model identifier matches any config's model_identifier
    for name, config in ANTHROPIC_MODELS.items():
        if config.model_identifier == model:
            return _check(name, config)

    # Check partial match
    model_lower = model.lower()
    for name, config in ANTHROPIC_MODELS.items():
        if name in model_lower or config.model_identifier in model_lower:
            return _check(name, config)

    return False


def supports_temperature(model: str) -> bool:
    """Check whether a model accepts the `temperature` request parameter.

    Anthropic deprecated `temperature` for Opus 4.7 (and likely future models);
    sending it returns HTTP 400 `"temperature is deprecated for this model"`.
    Callers should omit `temperature` from the request_params when this is
    False.

    Args:
        model: The model identifier (alias or full identifier).

    Returns:
        True when the model accepts `temperature`. Defaults to True for
        unknown models so behavior matches the previous implicit default.
    """
    if model in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[model].supports_temperature
    for config in ANTHROPIC_MODELS.values():
        if config.model_identifier == model:
            return config.supports_temperature
    model_lower = model.lower()
    for name, config in ANTHROPIC_MODELS.items():
        if name in model_lower or config.model_identifier in model_lower:
            return config.supports_temperature
    return True


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
