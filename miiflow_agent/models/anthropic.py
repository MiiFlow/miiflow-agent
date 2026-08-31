"""Anthropic model configurations."""

from typing import Dict, Optional

from .base import ModelConfig, ParameterConfig, ParameterType

ANTHROPIC_MODELS: Dict[str, ModelConfig] = {
    "claude-fable-5": ModelConfig(
        model_identifier="claude-fable-5",
        name="claude-fable-5",
        description="Anthropic's most capable widely released model (generally available since June 9, 2026; API access was briefly suspended June 12–July 1, 2026 under a US export-control directive and has since been restored). Built for demanding reasoning and long-horizon agentic work, with always-on adaptive thinking, structured outputs, and a 1M context window. Claude Mythos 5 shares its $10/$50 pricing but is limited-availability (Project Glasswing partners only), so Fable 5 is the top tier reachable with a standard API key.",
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
        cache_read_cost_hint=1.0,  # 0.1x input
        cache_write_cost_hint=12.5,  # 1.25x input (5-min TTL)
    ),
    "claude-opus-5": ModelConfig(
        model_identifier="claude-opus-5",
        name="claude-opus-5",
        description="Anthropic's most capable Opus model (released July 24, 2026), delivering near-Fable 5 performance at half the token price. Features always-on adaptive thinking with an xhigh reasoning-effort mode, a Fast Mode (2.5x faster at 2x the price), structured outputs, and a safety fallback that routes to Opus 4.8. 1M context window.",
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
        cache_read_cost_hint=0.5,  # 0.1x input
        cache_write_cost_hint=6.25,  # 1.25x input (5-min TTL)
    ),
    "claude-opus-4.8": ModelConfig(
        model_identifier="claude-opus-4-8",
        name="claude-opus-4.8",
        description="Legacy — succeeded by Claude Opus 5 (July 24, 2026). Powerful reasoning and coding model with adaptive thinking, structured outputs, and fast mode; remains available and serves as Opus 5's safety fallback. 1M context window.",
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
        cache_read_cost_hint=0.5,  # 0.1x input
        cache_write_cost_hint=6.25,  # 1.25x input (5-min TTL)
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
        cache_read_cost_hint=0.5,  # 0.1x input
        cache_write_cost_hint=6.25,  # 1.25x input (5-min TTL)
    ),
    "claude-opus-4.6": ModelConfig(
        model_identifier="claude-opus-4-6",
        name="claude-opus-4.6",
        description="Legacy — succeeded by Claude Opus 4.7 (April 2026). The oldest model here that still accepts the sampling parameters, and the last before manual extended thinking was removed (it still works but is deprecated). Effort tops out at max — it predates the xhigh level and rejects it. 128K max output tokens, 1M context window.",
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
        cache_read_cost_hint=0.5,  # 0.1x input
        cache_write_cost_hint=6.25,  # 1.25x input (5-min TTL)
    ),
    "claude-sonnet-5": ModelConfig(
        model_identifier="claude-sonnet-5",
        name="claude-sonnet-5",
        description="Anthropic's most agentic Sonnet model (released June 30, 2026), succeeding Sonnet 4.6 and closing much of the gap with Opus 4.8 on reasoning, tool use, and coding. Adaptive thinking is on by default; manual extended thinking and non-default temperature/top_p/top_k are rejected. Supports all five effort levels. 1M context window. $2/$10 per 1M input/output tokens is the standard price — the launch rate was announced as introductory through August 31, 2026, and Anthropic then cancelled the scheduled September 1, 2026 increase to $3/$15.",
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
        cache_read_cost_hint=0.2,  # 0.1x input
        cache_write_cost_hint=2.5,  # 1.25x input (5-min TTL)
    ),
    "claude-sonnet-4.6": ModelConfig(
        model_identifier="claude-sonnet-4-6",
        name="claude-sonnet-4.6",
        description="Legacy — succeeded by Claude Sonnet 5 (June 2026), which is both stronger and cheaper ($2/$10 vs $3/$15), so this is kept for pinned workloads only. Supports adaptive thinking, structured outputs and the sampling parameters; manual extended thinking still works but is deprecated. Effort tops out at max — it predates the xhigh level and rejects it. 1M context window.",
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
        cache_read_cost_hint=0.3,  # 0.1x input
        cache_write_cost_hint=3.75,  # 1.25x input (5-min TTL)
    ),
    "claude-haiku-4.5": ModelConfig(
        model_identifier="claude-haiku-4-5-20251001",
        name="claude-haiku-4.5",
        description="Anthropic's fastest model with near-frontier intelligence, delivering Sonnet-4-level coding performance at one-third the cost and more than twice the speed. The only model here on extended thinking only: it rejects adaptive thinking and the effort parameter with a 400. 200K context window; retirement not sooner than October 15, 2026.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        # Native structured outputs are supported on Haiku 4.5 (Claude API and
        # the legacy Bedrock endpoint). This was False here, which silently
        # routed every Haiku 4.5 schema request through the loosened-schema
        # fallback instead of the strict native format.
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=200000,
        maximum_output_tokens=64000,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=1.0,
        output_cost_hint=5.0,
        cache_read_cost_hint=0.1,  # 0.1x input
        cache_write_cost_hint=1.25,  # 1.25x input (5-min TTL)
    ),
}


def _resolve_model_name(model: str) -> Optional[str]:
    """Resolve any spelling of a model to its key in ``ANTHROPIC_MODELS``.

    One implementation of the match, because every per-model capability lookup
    below needs the same three tiers and a fourth copy of it is how they drift:
    the catalog key (``claude-sonnet-4.6``), the API identifier
    (``claude-sonnet-4-6``), then a substring match so a Bedrock inference
    profile (``us.anthropic.claude-sonnet-4-6``) or a dated snapshot resolves
    too. Returns None for a model this catalog does not know; each caller owns
    what that means, since the safe default differs per capability.
    """
    if not model:
        return None
    if model in ANTHROPIC_MODELS:
        return model
    for name, config in ANTHROPIC_MODELS.items():
        if config.model_identifier == model:
            return name
    model_lower = model.lower()
    for name, config in ANTHROPIC_MODELS.items():
        if name in model_lower or config.model_identifier in model_lower:
            return name
    return None


# Anthropic deprecated `temperature` (and `top_p` / `top_k`) on Claude Opus 4.7
# and later: a non-default value returns HTTP 400 "temperature is deprecated for
# this model". Derived from the model configs rather than restated, so a new
# model declaring supports_temperature=False is excluded automatically — the
# request-time gate is `supports_temperature()` below, and this keeps the
# parameter the UI OFFERS in step with the one the API will actually take.
_NO_TEMPERATURE_MODELS = [
    name for name, config in ANTHROPIC_MODELS.items() if not config.supports_temperature
]


ANTHROPIC_PARAMETERS: list[ParameterConfig] = [
    ParameterConfig(
        field_name="temperature",
        display_name="Temperature",
        description="Amount of randomness injected into the response. Not accepted by Claude Opus 4.7 and later, which reject a non-default value with a 400.",
        parameter_type=ParameterType.NUMBER,
        default_value=0.7,
        min_value=0,
        max_value=1,
        step=0.1,
        unsupported_models=_NO_TEMPERATURE_MODELS,
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
            "claude-opus-5": 128000,
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


# Models that reject `thinking: {"type": "enabled", "budget_tokens": N}` with a
# 400. Claude 4.7 and later removed the manual extended-thinking mode entirely;
# Opus 4.6 and Sonnet 4.6 still accept it but Anthropic marks it deprecated
# there, and Haiku 4.5 is extended-thinking ONLY (it 400s on adaptive), so both
# stay out of this set and keep the parameter.
_NO_EXTENDED_THINKING = {
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4.8",
    "claude-opus-4.7",
    "claude-sonnet-5",
}

# Models that THINK BY DEFAULT when the request omits `thinking` (adaptive is
# the default, not off) and that accept `thinking: {"type": "disabled"}` at
# effort <= high. Fable 5 also thinks by default but rejects "disabled" with a
# 400, so it is deliberately absent; Opus 4.8/4.7/4.6 and Sonnet 4.6 default to
# no thinking, so there is nothing to disable.
_THINKING_ON_BY_DEFAULT_DISABLEABLE = {
    "claude-opus-5",
    "claude-sonnet-5",
}


# `output_config.effort` — the GA knob that scales adaptive thinking (and overall
# token spend) on the models that think by default. `budget_tokens` is rejected
# on Sonnet 5 / Opus 5 / 4.7 / 4.8 / Fable 5, so this is the ONLY way to bound
# their deliberation short of disabling thinking (which is discouraged: with
# thinking off these models sometimes write a tool call into visible text).
# Haiku 4.5 and older models 400 on the parameter.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# The levels are NOT uniform across the models that accept the parameter, so a
# single flat tuple is not the contract: `xhigh` is a newer level, and Opus 4.6
# / Sonnet 4.6 support `max` but return a 400 on `xhigh`. This map is the wire
# contract per model; EFFORT_LEVELS above stays the validation vocabulary a
# caller may configure, so one config survives a model swap and the per-model
# projection happens at request time (see `effort_levels`).
_EFFORT_LEVELS_BY_MODEL: Dict[str, tuple] = {
    "claude-fable-5": EFFORT_LEVELS,
    "claude-opus-5": EFFORT_LEVELS,
    "claude-opus-4.8": EFFORT_LEVELS,
    "claude-opus-4.7": EFFORT_LEVELS,
    "claude-sonnet-5": EFFORT_LEVELS,
    "claude-opus-4.6": ("low", "medium", "high", "max"),
    "claude-sonnet-4.6": ("low", "medium", "high", "max"),
}

# Models that reject `thinking: {"type": "disabled"}` once effort is `xhigh` or
# `max` — the combination is a 400, enforced per request. Anthropic documents
# the restriction as applying to Claude Opus 5 and later; Sonnet 5 predates it
# and accepts "disabled" at every level.
_DISABLED_REJECTED_AT_EFFORT = frozenset({"xhigh", "max"})
_DISABLED_EFFORT_GATED_MODELS = frozenset({"claude-opus-5"})


def effort_levels(model: str) -> tuple:
    """The `output_config.effort` values `model` accepts, or `()` if none."""
    return _EFFORT_LEVELS_BY_MODEL.get(_resolve_model_name(model) or "", ())


def supports_effort(model: str) -> bool:
    """True when `model` accepts `output_config: {"effort": ...}`."""
    return bool(effort_levels(model))


def thinking_disable_param(
    model: str, effort: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """The `thinking` request value that turns thinking OFF for `model`, or None.

    Callers that want a short, deterministic, cheap completion (a compaction
    handoff note, a title, a classifier) must pass this explicitly on the
    thinking-by-default models: otherwise adaptive thinking runs first and
    `max_tokens` — a hard cap on thinking PLUS text — can be consumed entirely
    by the thinking block, returning `stop_reason=max_tokens` with no text at
    all. Returns None where nothing needs sending (defaults to off), where the
    API would reject "disabled" outright (Fable 5), or where the model rejects
    it at the effort level this request carries (Opus 5 at `xhigh` / `max`) —
    hence `effort`: the answer depends on the whole request, not the model
    alone, and returning the parameter without it is a guaranteed 400.
    """
    name = _resolve_model_name(model)
    if name not in _THINKING_ON_BY_DEFAULT_DISABLEABLE:
        return None
    if (
        name in _DISABLED_EFFORT_GATED_MODELS
        and (effort or "").lower() in _DISABLED_REJECTED_AT_EFFORT
    ):
        return None
    return {"type": "disabled"}


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
    name = _resolve_model_name(model)
    if name is None:
        return False
    return ANTHROPIC_MODELS[name].supports_structured_outputs


def supports_thinking(model: str) -> bool:
    """Check if model supports the explicit extended-thinking API parameter.

    Opus 4.7 uses adaptive thinking (always-on) and does NOT accept the
    ``thinking`` request parameter, so this returns False for it.

    Args:
        model: The model identifier

    Returns:
        True if model supports the extended-thinking parameter
    """

    name = _resolve_model_name(model)
    if name is None:
        return False
    return ANTHROPIC_MODELS[name].reasoning and name not in _NO_EXTENDED_THINKING


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
    name = _resolve_model_name(model)
    if name is None:
        return True
    return ANTHROPIC_MODELS[name].supports_temperature


def supports_native_mcp(model: str) -> bool:
    """Check if model supports native MCP via the beta API.

    Native MCP allows the Anthropic API to connect directly to MCP servers
    and execute tools server-side, rather than requiring client-side handling.

    All Claude models support native MCP via the mcp-client beta (see AnthropicClient.MCP_CONNECTOR_BETA).

    Args:
        model: The model identifier

    Returns:
        True if model supports native MCP (all Claude models do)
    """
    # All Claude models support native MCP via beta API
    if _resolve_model_name(model) is not None:
        return True

    # An unknown model is still a Claude model if it is named like one.
    return "claude" in (model or "").lower()
