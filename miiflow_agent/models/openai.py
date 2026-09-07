"""OpenAI model configurations."""

from dataclasses import replace
from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# O-series reasoning models that use max_completion_tokens instead of max_tokens.
# Currently empty: o3 was retired after its June 2026 deprecation; the GPT-5.x
# reasoning models are tracked separately in _GPT5_MODELS below.
_REASONING_MODELS: set[str] = set()

# GPT-5.x reasoning models (use max_completion_tokens, no temperature). GPT-5.6
# Pro is an execution mode on these model ids, not a separate model slug.
#
# Deliberately absent: gpt-5.6-cyber ($12.50/$75). It is a purpose-trained
# cybersecurity model gated behind OpenAI's Daybreak program — provisioned to
# approved defenders, with no self-serve route — so it would 404 for every key
# this catalog serves. Add it only if an org is onboarded to Daybreak.
_GPT5_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
}

# GPT-6 (Astra), generally available September 3, 2026. Kept as its own set
# because its parameter contract is NOT the GPT-5.6 one: it drops `temperature`,
# `top_p`, `top_logprobs`, `logprobs` and `prompt_cache_retention` outright, and
# its effort scale loses `none`/`minimal`. `-fast` is a latency variant of the
# same id (2x speed, 2x price), not a separate model; pro is `reasoning.mode`,
# the same shape GPT-5.6 uses, so neither gets its own catalog entry.
_GPT6_MODELS = {
    "gpt-6-astra",
}
# The family alias. Kept beside the catalog entry rather than only inside
# `is_gpt6_model`, because every per-model answer (the surcharge, the rejected
# parameters) has to agree about it: a spelling one helper calls GPT-6 and
# another does not is how a request goes out carrying a parameter the model
# rejects.
_GPT6_ALIASES = {"gpt-6"}
_GPT6_ALL = _GPT6_MODELS | _GPT6_ALIASES

# Models whose reasoning-tier contract applies: max_completion_tokens, no
# temperature, a reasoning_effort scale, verbosity.
_REASONING_TIER_MODELS = _GPT5_MODELS | _GPT6_MODELS

_GPT56_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}
_GPT55_STANDARD_MODELS = {"gpt-5.5"}
_GPT54_STANDARD_MODELS = {"gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}
_GPT_PRO_MODELS = {"gpt-5.5-pro", "gpt-5.4-pro"}
_GPT41_MODELS = {"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"}

# OpenAI applies a request-wide surcharge once a 1.05M-context GPT-5 model's
# input exceeds this threshold. Finance imports these constants so the request
# pricing rule stays beside the model catalog rather than becoming a second,
# drifting model list.
LONG_CONTEXT_PRICING_THRESHOLD = 272_000
LONG_CONTEXT_INPUT_MULTIPLIER = 2.0
LONG_CONTEXT_OUTPUT_MULTIPLIER = 1.5
_LONG_CONTEXT_SURCHARGE_MODELS = {
    "gpt-5.6",
    *_GPT56_MODELS,
    *_GPT55_STANDARD_MODELS,
    *_GPT_PRO_MODELS,
    "gpt-5.4",
    # Astra: past 272K input the whole request bills at $20 / $2 / $75, i.e. the
    # same 2x input / 1.5x output multipliers the GPT-5 models carry.
    *_GPT6_ALL,
}

# Models that don't support temperature parameter
_NO_TEMPERATURE_MODELS = _REASONING_MODELS | _REASONING_TIER_MODELS

# Prefixes that identify a reasoning model this catalog has not seen yet — a
# dated snapshot (`gpt-6-astra-2026-09-03`) or a family member released after
# this file was written. The capability helpers below fall back to these so an
# unknown reasoning model gets `max_completion_tokens` and no `temperature`
# rather than the standard-chat-model defaults, which would 400. One tuple,
# because it was spelled out at four call sites and the "gpt-6" entry would
# otherwise have had to be added to each of them.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")

# Request parameters a model rejects outright, beyond the temperature gate
# above. These are PASSTHROUGH kwargs — the client copies them onto the request
# whenever a caller supplies one — so a model that 400s on them needs the drop
# to happen here rather than at each call site. GPT-6 Astra removed the
# sampling/logprob controls and replaced `prompt_cache_retention` with
# `prompt_cache_options.ttl`.
_UNSUPPORTED_REQUEST_PARAMS: Dict[str, frozenset[str]] = {
    model: frozenset(
        {"logprobs", "prompt_cache_retention", "temperature", "top_logprobs", "top_p"}
    )
    for model in _GPT6_ALL
}


OPENAI_MODELS: Dict[str, ModelConfig] = {
    # GPT-6 (Astra) — generally available September 3, 2026. Current flagship.
    "gpt-6-astra": ModelConfig(
        model_identifier="gpt-6-astra",
        name="gpt-6-astra",
        description="GPT-6 Astra is OpenAI's flagship model (generally available September 3, 2026), a single dense reasoning model for complex coding, reasoning, and long-horizon agentic work. 1.05M context window, 128K max output. $10/$50 per 1M input/output tokens, with cached input at $1 and cache writes at $12.50; past 272K input tokens the whole request bills at $20/$2/$75. Effort is the biggest cost dial — Artificial Analysis measures a 3.6x swing from low to max. Its parameter contract differs from GPT-5.6: temperature, top_p, top_logprobs (and logprobs on Chat Completions) are removed, prompt_cache_retention is replaced by prompt_cache_options.ttl, and the effort scale drops none/minimal. Tool calling requires the Responses API; `max` effort is Responses-only. Appending -fast to the model id runs it at up to 2x speed for 2x the price.",
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
        input_cost_hint=10.0,
        output_cost_hint=50.0,
        cache_read_cost_hint=1.0,  # OpenAI cached input: 10% of input rate
        cache_write_cost_hint=12.5,  # Cache writes: 1.25x input
    ),
    # GPT-5.6 series (Sol / Terra / Luna) — generally available July 9, 2026.
    # gpt-5.6 is an API alias for gpt-5.6-sol.
    "gpt-5.6-sol": ModelConfig(
        model_identifier="gpt-5.6-sol",
        name="gpt-5.6-sol",
        description="Legacy — succeeded by GPT-6 Astra (September 3, 2026) as OpenAI's flagship. The highest-intelligence tier of the GPT-5.6 family, built for complex coding, reasoning, and long-horizon agentic work, and still much cheaper than Astra ($4/$20 vs $10/$50). 1M context window. Available in the API as gpt-5.6-sol (alias: gpt-5.6). An August 21, 2026 price cut lowered it to $4/$20 per 1M input/output tokens (from $5/$30) — a promotional rate running through at least November 21, 2026 — undercutting Claude Opus 5 on both input and output.",
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
        input_cost_hint=4.0,
        output_cost_hint=20.0,
        cache_read_cost_hint=0.4,  # OpenAI cached input: 10% of input rate
        cache_write_cost_hint=5.0,  # Explicit GPT-5.6 cache writes: 1.25x input
    ),
    "gpt-5.6-terra": ModelConfig(
        model_identifier="gpt-5.6-terra",
        name="gpt-5.6-terra",
        description="GPT-5.6 Terra is the balanced mid-tier of the GPT-5.6 family (July 9, 2026), delivering strong reasoning and agentic performance at well under half the cost of Sol. A July 30, 2026 price cut lowered it to $2/$12 per 1M input/output tokens (from $2.50/$15). 1M context window.",
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
        input_cost_hint=2.0,
        output_cost_hint=12.0,
        cache_read_cost_hint=0.2,  # OpenAI cached input: 10% of input rate
        cache_write_cost_hint=2.5,  # Explicit GPT-5.6 cache writes: 1.25x input
    ),
    "gpt-5.6-luna": ModelConfig(
        model_identifier="gpt-5.6-luna",
        name="gpt-5.6-luna",
        description="GPT-5.6 Luna is the fastest and most cost-efficient tier of the GPT-5.6 family (July 9, 2026), optimized for high-throughput, latency-sensitive workloads. A July 30, 2026 price cut lowered it ~80% to $0.20/$1.20 per 1M input/output tokens (from $1/$6). 1M context window.",
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
        input_cost_hint=0.20,
        output_cost_hint=1.20,
        cache_read_cost_hint=0.02,  # OpenAI cached input: 10% of input rate
        cache_write_cost_hint=0.25,  # Explicit GPT-5.6 cache writes: 1.25x input
    ),
    # GPT-5.5 series (released April 23, 2026) — succeeded by the GPT-5.6 family
    "gpt-5.5": ModelConfig(
        model_identifier="gpt-5.5",
        name="gpt-5.5",
        description="GPT-5.5 (released April 23, 2026) — previous flagship, superseded by GPT-5.6 Sol. Strong coding and agentic model (82.7% on Terminal-Bench 2.0, 58.6% on SWE-Bench Pro) with a 1M context window. Note that it is no longer the cheaper choice: after Sol's August 21, 2026 cut to $4/$20, GPT-5.5 at $5/$30 costs more than the newer and more capable Sol. Kept for pinned workloads only.",
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
        cache_read_cost_hint=0.5,  # OpenAI cached input: 10% of input rate
    ),
    "gpt-5.5-pro": ModelConfig(
        model_identifier="gpt-5.5-pro",
        name="gpt-5.5-pro",
        description="GPT-5.5 Pro (April 23, 2026) — high-accuracy model for mission-critical agentic and reasoning tasks. Responses API, 1M context window.",
        support_images=True,
        support_files=True,
        support_streaming=False,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=True,
        api_path="/responses",
        maximum_context_tokens=1050000,
        maximum_output_tokens=128000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=30.0,
        output_cost_hint=180.0,
        # GPT-5.5 Pro has no cached-input discount. Leaving the cache-read
        # rate undeclared makes finance bill cached tokens at the input rate.
    ),
    # GPT-5.4 series (released March 2026) — two generations old, superseded by the GPT-5.6 family
    "gpt-5.4": ModelConfig(
        model_identifier="gpt-5.4",
        name="gpt-5.4",
        description="GPT-5.4 (March 5, 2026) — two generations old, superseded by the GPT-5.6 family. 1M context window, built-in computer use, and improved deep research. No longer a cost saving: at $2.50/$15 it is priced above GPT-5.6 Terra ($2/$12), which is both newer and stronger. Kept for pinned workloads only.",
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
        cache_read_cost_hint=0.25,  # OpenAI cached input: 10% of input rate
    ),
    "gpt-5.4-pro": ModelConfig(
        model_identifier="gpt-5.4-pro",
        name="gpt-5.4-pro",
        description="GPT-5.4 Pro (March 2026) — high-accuracy Responses API model for mission-critical agentic tasks. 1M context window with built-in computer use.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=False,
        supports_tool_call=True,
        reasoning=True,
        api_path="/responses",
        maximum_context_tokens=1050000,
        maximum_output_tokens=128000,
        token_param_name="max_completion_tokens",
        supports_temperature=False,
        input_cost_hint=30.0,
        output_cost_hint=180.0,
        cache_read_cost_hint=3.0,  # OpenAI cached input: 10% of input rate
    ),
    "gpt-5.4-mini": ModelConfig(
        model_identifier="gpt-5.4-mini",
        name="gpt-5.4-mini",
        description="GPT-5.4 Mini is a smaller, faster GPT-5.4 variant with strong reasoning at lower cost. 400K context window. Released March 17, 2026. Superseded by GPT-5.6 Luna, which is cheaper ($0.20/$1.20 vs $0.75/$4.50) and carries a 1M context window; Mini remains the current small model on the 5.4 line.",
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
        cache_read_cost_hint=0.075,  # OpenAI cached input: 10% of input rate
    ),
    "gpt-5.4-nano": ModelConfig(
        model_identifier="gpt-5.4-nano",
        name="gpt-5.4-nano",
        description="GPT-5.4 Nano is the most cost-effective GPT-5.4 variant, optimized for latency. 400K context window. Released March 17, 2026. GPT-5.6 Luna matches its input price at a lower output price ($1.20 vs $1.25) with a 1M context window.",
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
        cache_read_cost_hint=0.02,  # OpenAI cached input: 10% of input rate
    ),
    # GPT-4.1 series (standard models, use max_tokens)
    "gpt-4.1": ModelConfig(
        model_identifier="gpt-4.1",
        name="gpt-4.1",
        description="Deprecated — migrate to GPT-5.6 Terra. General-purpose model with 1M token context. Retired from ChatGPT Feb 13, 2026; still available via the API, but the API shutdown is scheduled for Oct 14, 2026, after which requests will fail.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=False,
        maximum_context_tokens=1047576,
        maximum_output_tokens=32768,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=2.0,
        output_cost_hint=8.0,
        cache_read_cost_hint=0.50,  # OpenAI cached input: 25% of input for gpt-4.1
    ),
    "gpt-4.1-mini": ModelConfig(
        model_identifier="gpt-4.1-mini",
        name="gpt-4.1-mini",
        description="Deprecated — migrate to GPT-5.6 Luna or GPT-5.4 Mini. Smaller GPT-4.1 with 1M token context. API shutdown scheduled for Oct 14, 2026, after which requests will fail.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=False,
        maximum_context_tokens=1047576,
        maximum_output_tokens=32768,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=0.40,
        output_cost_hint=1.60,
        cache_read_cost_hint=0.10,  # OpenAI cached input: 25% of input for gpt-4.1
    ),
    "gpt-4.1-nano": ModelConfig(
        model_identifier="gpt-4.1-nano",
        name="gpt-4.1-nano",
        description="Deprecated — migrate to GPT-5.6 Luna or GPT-5.4 Nano. Smallest, fastest 4.1 variant. API shutdown scheduled for Oct 14, 2026, after which requests will fail.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        reasoning=False,
        maximum_context_tokens=1047576,
        maximum_output_tokens=32768,
        token_param_name="max_tokens",
        supports_temperature=True,
        input_cost_hint=0.10,
        output_cost_hint=0.40,
        cache_read_cost_hint=0.025,  # OpenAI cached input: 25% of input for gpt-4.1
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
            "gpt-6-astra": 128000,
            "gpt-5.6-sol": 128000,
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
        supported_models=list(_GPT41_MODELS),
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
        supported_models=list(_GPT41_MODELS),
    ),
    ParameterConfig(
        field_name="reasoning_effort",
        display_name="Reasoning Effort",
        description="Constrains effort on reasoning for reasoning models.",
        parameter_type=ParameterType.SELECT,
        default_value="medium",
        options=["none", "low", "medium", "high", "xhigh", "max"],
        supported_models=list(_REASONING_MODELS | _REASONING_TIER_MODELS),
    ),
    ParameterConfig(
        field_name="reasoning_mode",
        display_name="Reasoning Mode",
        description="Use pro mode for additional model work on difficult tasks. Pro mode requires the Responses API.",
        parameter_type=ParameterType.SELECT,
        default_value="standard",
        options=["standard", "pro"],
        supported_models=list(_GPT56_MODELS | _GPT6_MODELS),
    ),
    ParameterConfig(
        field_name="verbosity",
        display_name="Verbosity",
        description="Controls the verbosity of the model's response. Available for GPT-5 models to adjust response detail level.",
        parameter_type=ParameterType.SELECT,
        default_value="medium",
        options=["low", "medium", "high"],
        supported_models=list(_REASONING_TIER_MODELS),
    ),
]


_REASONING_EFFORT_OPTIONS = {
    # Astra dropped `none` (and `minimal`); `max` is Responses-only.
    **{model: ["low", "medium", "high", "xhigh", "max"] for model in _GPT6_MODELS},
    **{
        model: ["none", "low", "medium", "high", "xhigh", "max"]
        for model in _GPT56_MODELS
    },
    **{
        model: ["none", "low", "medium", "high", "xhigh"]
        for model in _GPT55_STANDARD_MODELS | _GPT54_STANDARD_MODELS
    },
    **{model: ["medium", "high", "xhigh"] for model in _GPT_PRO_MODELS},
}


def get_parameters_for_model(model: str) -> list[ParameterConfig]:
    """Return an exact parameter contract for one OpenAI model.

    The provider importer persists parameter options per model. Keeping this
    projection in the SDK avoids a single union enum advertising values that
    OpenAI rejects for Pro and older GPT-5 models.
    """
    model_lower = model.lower()
    parameters: list[ParameterConfig] = []
    for parameter in OPENAI_PARAMETERS:
        if parameter.supported_models and model_lower not in parameter.supported_models:
            continue
        if parameter.unsupported_models and model_lower in parameter.unsupported_models:
            continue
        if parameter.field_name == "reasoning_effort":
            options = _REASONING_EFFORT_OPTIONS.get(model_lower)
            if not options:
                continue
            default = "medium" if "medium" in options else options[0]
            parameters.append(
                replace(parameter, options=options, default_value=default)
            )
            continue
        parameters.append(replace(parameter))
    return parameters


def normalize_model_name(model: str) -> str:
    """Map retired local aliases to valid OpenAI model identifiers."""
    if model.lower() == "gpt-5.6-sol-pro":
        return "gpt-5.6-sol"
    return model


def _base_model_name(model: str) -> str:
    """The catalog identity behind a request-time model spelling.

    Strips OpenAI's `-fast` latency suffix, which selects a service tier rather
    than a different model: `gpt-6-astra-fast` has the same context window,
    parameter contract and long-context surcharge as `gpt-6-astra`, only at 2x
    speed and 2x price. Stripping it in the ONE family matcher every capability
    predicate goes through is what keeps them from disagreeing — answering the
    suffix separately in each is how `-fast` came to be billed without the
    long-context surcharge while still being recognised as a GPT-6 model.
    """
    return normalize_model_name(model).lower().removesuffix("-fast")


def _matches_model_family(model: str, families: set[str]) -> bool:
    model_lower = _base_model_name(model)
    return any(
        model_lower == family or model_lower.startswith(f"{family}-20")
        for family in families
    )


def is_gpt56_model(model: str) -> bool:
    """Return whether ``model`` is a GPT-5.6 standard model or alias."""
    normalized = normalize_model_name(model).lower()
    return normalized == "gpt-5.6" or _matches_model_family(normalized, _GPT56_MODELS)


def is_gpt6_model(model: str) -> bool:
    """Return whether ``model`` is a GPT-6 model, alias, or latency variant."""
    return _matches_model_family(model, _GPT6_ALL)


def tools_require_responses_api(model: str) -> bool:
    """Return whether tool calling on ``model`` must go through /v1/responses.

    True for GPT-5.6 (Chat Completions cannot combine tools with the reasoning
    controls) and for GPT-6 Astra, which OpenAI documents as supporting Chat
    Completions for plain turns but requiring Responses for tool calling.
    """
    return is_gpt56_model(model) or is_gpt6_model(model)


def unsupported_request_params(model: str) -> frozenset[str]:
    """Request parameters ``model`` rejects, so the client can drop them.

    Empty for a model with no removals. Keyed off the catalog rather than a
    literal at the call site: these are passthrough kwargs, and a caller that
    sets one on a model that removed it gets a 400 rather than a warning.
    """
    for family, params in _UNSUPPORTED_REQUEST_PARAMS.items():
        if _matches_model_family(model, {family}):
            return params
    return frozenset()


def requires_responses_api(model: str) -> bool:
    """Return whether OpenAI documents the model as Responses-only/preferred."""
    if model.lower() == "gpt-5.6-sol-pro":
        return True
    return _matches_model_family(model, _GPT_PRO_MODELS)


def supports_sampling_penalties(model: str) -> bool:
    """Frequency/presence penalties are exposed only for GPT-4.1 here."""
    return _matches_model_family(model, _GPT41_MODELS)


def supports_verbosity(model: str) -> bool:
    """Return whether the model accepts OpenAI's verbosity control.

    Astra keeps it: its published removals are the sampling/logprob controls
    and `prompt_cache_retention`, and it otherwise carries GPT-5.6's API
    surface.
    """
    return _matches_model_family(model, _GPT5_MODELS) or is_gpt6_model(model)


def supports_streaming(model: str) -> bool:
    """Return the catalog's streaming capability for a model or snapshot."""
    normalized = normalize_model_name(model).lower()
    if normalized in OPENAI_MODELS:
        return OPENAI_MODELS[normalized].support_streaming
    if _matches_model_family(normalized, {"gpt-5.5-pro"}):
        return False
    return True


def supports_json_mode(model: str) -> bool:
    """Return the catalog's structured-output capability for a model."""
    normalized = normalize_model_name(model).lower()
    if normalized in OPENAI_MODELS:
        return OPENAI_MODELS[normalized].supports_json_mode
    if _matches_model_family(normalized, {"gpt-5.4-pro"}):
        return False
    return True


def get_long_context_pricing_multipliers(
    model: str | None, input_tokens: int | None
) -> tuple[float, float]:
    """Return request-wide input/output multipliers for long-context GPT-5.

    OpenAI applies the surcharge to the full request once prompt input exceeds
    272K tokens. Unknown models and requests at the boundary keep base rates.
    """
    if not model or not input_tokens or input_tokens <= LONG_CONTEXT_PRICING_THRESHOLD:
        return 1.0, 1.0
    if not _matches_model_family(model, _LONG_CONTEXT_SURCHARGE_MODELS):
        return 1.0, 1.0
    return LONG_CONTEXT_INPUT_MULTIPLIER, LONG_CONTEXT_OUTPUT_MULTIPLIER


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
    for prefix in _REASONING_MODEL_PREFIXES:
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
    for prefix in _REASONING_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return False

    return True


def supports_reasoning_effort(model: str) -> bool:
    """Check if model accepts the ``reasoning_effort`` parameter.

    Only the reasoning models (o-series) and the GPT-5 family expose this
    control; sending it to a standard chat model (e.g. gpt-4.1) is a 400.

    Note: even for supported models, OpenAI rejects ``reasoning_effort`` when
    it is combined with function ``tools`` on the Chat Completions endpoint
    ("Function tools with reasoning_effort are not supported ... in
    /v1/chat/completions"). Callers must additionally gate on the absence of
    tools — this helper only answers model-level support. See
    ``OpenAIClient._apply_reasoning_effort``.

    Args:
        model: The model identifier

    Returns:
        True if the model supports reasoning_effort, False otherwise
    """
    model_lower = model.lower()

    if model_lower in _NO_TEMPERATURE_MODELS:
        return True

    # Prefix match for versioned/unknown reasoning models.
    for prefix in _REASONING_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return True

    return False


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
    openai_prefixes = ("gpt-4", *_REASONING_MODEL_PREFIXES)
    for prefix in openai_prefixes:
        if model_lower.startswith(prefix):
            return True

    return False
