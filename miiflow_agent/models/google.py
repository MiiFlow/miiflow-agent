"""Google Gemini model configurations."""

from typing import Dict

from .base import ModelConfig, ParameterConfig, ParameterType

# All Gemini models use max_output_tokens.
#
# Sampling parameters (temperature / top_p / top_k) are DEPRECATED from Gemini
# 3.6 Flash and Gemini 3.5 Flash-Lite onward: the API accepts them and silently
# IGNORES them — no error, no warning — so a value set here would look applied
# while changing nothing. The older models (3.5 Flash, 3.1 Pro, 3.1 Flash-Lite)
# still honour them. Google's replacement knob is `thinking_level`, which
# GeminiClient does not send yet. Note the scale itself changed: 3.8 Flash
# documents LOW/MEDIUM/HIGH with MEDIUM as the default, dropping the `minimal`
# level the earlier generations offered.
_NO_SAMPLING_PARAMS = {
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
}

# Deliberately absent: gemini-3.8-flash-cyber. It is the same core model behind
# a restricted access envelope — trusted defenders admitted to Google's Fairwind
# programme, with no self-serve route — so it would fail for every key this
# catalog serves. Same reasoning as gpt-5.6-cyber in the OpenAI catalog.
#
# Also deliberately absent, all released September 15, 2026: gemini-3.8-live and
# gemini-3.8-live-extended-thinking (native speech-to-speech over the Live API's
# WebSocket, billed per MINUTE — $0.005/min in, $0.018/min out) and
# gemini-3.5-transcribe (a dedicated speech-to-text model). None of them takes a
# generateContent request or has per-token prices, so a ModelConfig — whose whole
# contract is per-token pricing plus a token_param_name — cannot describe them.
# Same reasoning as gpt-live-1 in the OpenAI catalog. gemini-omni-1.1-flash (GA
# September 2026) is excluded for the same reason: it is a video generation and
# editing model billed per second of video, not per token.
#
# Also deliberately absent: gemini-4-argon, announced September 30, 2026.
# Gemini 4 Argon is Google's new frontier model and it genuinely beats
# everything in this catalog — Artificial Analysis puts it level with GPT-6
# Astra on their Intelligence Index at roughly 60% of Astra's cost per task —
# but NO KEY THIS CATALOG SERVES CAN CALL IT. Launch-day access is Google's
# Fairwind programme (vetted cyber defenders, who get it without the cyber
# guardrails) plus Google's own teams; next comes the US government's voluntary
# pre-release evaluation; only after that do paid API customers and Google AI
# Ultra subscribers get it, and developers and the public after them. Google has
# published no date for any of those later phases and no API model id, so an
# entry here would fail on every request. Same reasoning as
# gemini-3.8-flash-cyber above and gpt-5.6-cyber in the OpenAI catalog — but
# note the reason is ACCESS alone, not missing prices: unlike gpt-6-astra-law,
# Argon's pricing is published, so access is the only thing being waited on.
#
# Unlike the -cyber variants this is not a permanently gated model, so it is the
# most likely Google addition at the next audit. Recording the specs here to
# make that a short change rather than another research pass: 1M token context
# window; 1M MAX OUTPUT TOKENS, which is not a typo and is 16x the 64K ceiling
# every model below it carries — `maximum_output_tokens` has never held a value
# that large here, so budget/truncation maths wants checking rather than
# assuming it drops in; introductory pricing $2/$10 per 1M input/output tokens
# with cached input 95% off at $0.10, rising to $4/$20 when the introductory
# period ends (Google has not said when). Argon also settles the question the
# gemini-3.1-pro entry below spent three audits tracking: Gemini 4 now exists,
# with a name, a price and a context window.
#
# PRICES HERE ARE STANDARD (pay-as-you-go) RATES, never the batch rate. Google
# runs the Batch API at a flat 50% off input and output for every model, so each
# published price has a halved twin that looks like a plausible standard rate and
# is not one. The September 30, 2026 audit recorded exactly that mistake:
# gemini-3.5-flash was carried at $0.75/$4.50, which is its batch rate, and read
# as evidence Google had cut the model from $1.50/$9.00. Google had not — 3.5
# Flash has never had an introductory rate and has never been cut, and the entry
# under-stated its cost by half. If a "price cut" is only ever half the previous
# figure, check the batch column before believing it.
#
# Cached-input rates are wired for gemini-3.5-flash only. The rest of the catalog
# leaves cache_read_cost_hint at 0.0, which bills cached tokens at the full input
# rate — an over-statement, so safe, but still wrong. Google's own pricing pages
# (ai.google.dev, cloud.google.com) were unreachable from the audit environment
# and the third-party figures for the 3.6/3.7/3.8 Flash and Flash-Lite tiers
# disagreed with each other, so they were left unset rather than guessed. Fill
# them in from the official pricing table when it is reachable.
GOOGLE_MODELS: Dict[str, ModelConfig] = {
    "gemini-3.8-flash": ModelConfig(
        model_identifier="models/gemini-3.8-flash",
        name="gemini-3.8-flash",
        description="Google's newest and most capable GENERALLY AVAILABLE Gemini model (released September 2, 2026), and the right default here. It is no longer Google's best model outright: Gemini 4 Argon, announced September 30, 2026, is stronger than anything in this catalog but is restricted to Google's Fairwind programme and cannot be called with an ordinary API key, so it is deliberately absent (see the note above GOOGLE_MODELS). Google's top self-serve tier is a Flash model because Gemini 3.5 Pro was never released — Gemini 3.1 Pro remains the Pro tier. Improves on Gemini 3.7 Flash for reasoning, coding, and agentic workflows, with native grounding, computer use, and multimodal (text, image, video, audio, PDF) input. 1M token context window, 64K max output. Thinking depth is controlled with thinking_level (LOW/MEDIUM/HIGH, default MEDIUM); temperature, top_p, top_k, candidate_count and the frequency/presence penalties are accepted and silently ignored. Introductory pricing of $0.75/$3.75 per 1M input/output tokens (output includes thinking tokens) applies through December 31, 2026, rising to $1.50/$7.50 on January 1, 2027.",
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
        supports_temperature=False,
        input_cost_hint=0.75,
        output_cost_hint=3.75,
    ),
    "gemini-3.7-flash": ModelConfig(
        model_identifier="models/gemini-3.7-flash",
        name="gemini-3.7-flash",
        description="Legacy — succeeded by Gemini 3.8 Flash (September 2, 2026), which is stronger at identical pricing, so this is kept for pinned workloads only. Released August 13, 2026. Strong coding and reasoning with native grounding, computer use, and multimodal (text, image, video, audio, PDF) input. 1M token context window; thinking depth is controlled with thinking_level rather than sampling parameters. Introductory pricing of $0.75/$3.75 per 1M input/output tokens applies through December 31, 2026, rising to $1.50/$7.50 on January 1, 2027.",
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
        supports_temperature=False,
        input_cost_hint=0.75,
        output_cost_hint=3.75,
    ),
    "gemini-3.6-flash": ModelConfig(
        model_identifier="models/gemini-3.6-flash",
        name="gemini-3.6-flash",
        description="Legacy — succeeded by Gemini 3.7 Flash (August 2026). Strong coding and agentic performance at Flash-tier pricing, with native grounding and multimodal (text, image, video, audio, PDF) input. 1M token context window; the first generation in which temperature/top_p/top_k are ignored in favour of thinking_level. Introductory pricing of $0.75/$3.75 per 1M input/output tokens applies through December 31, 2026, rising to $1.50/$7.50 on January 1, 2027.",
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
        supports_temperature=False,
        input_cost_hint=0.75,
        output_cost_hint=3.75,
    ),
    "gemini-3.5-flash-lite": ModelConfig(
        model_identifier="models/gemini-3.5-flash-lite",
        name="gemini-3.5-flash-lite",
        description="Google's current-generation Flash-Lite model (released July 21, 2026). The fastest and cheapest model in the Gemini 3.5 line (~490 output tokens/sec), built for high-volume, low-reasoning workloads such as search, document processing, and translation. Defaults to minimal thinking; temperature/top_p/top_k are ignored. 1M token context window.",
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
        supports_temperature=False,
        input_cost_hint=0.30,
        output_cost_hint=2.50,
    ),
    "gemini-3.5-flash": ModelConfig(
        model_identifier="models/gemini-3.5-flash",
        name="gemini-3.5-flash",
        description="Legacy — succeeded by Gemini 3.6 Flash (July 2026), and superseded twice over since. Released May 19, 2026 at $1.50/$9.00 per 1M input/output tokens and STILL AT THAT PRICE: unlike the 3.6/3.7/3.8 Flash generations it never carried an introductory rate, and Google has never cut it. That makes it the most expensive Flash in this catalog — twice the input and 2.4x the output of the $0.75/$3.75 introductory rate the three newer Flash models share, every one of which also beats it on capability — so no cost or quality argument remains for it. Kept for pinned workloads only, where it is the last Flash generation to honour temperature/top_p/top_k. 1M token context window, 64K max output, cached input $0.15 per 1M.",
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
        cache_read_cost_hint=0.15,  # 0.1x input
    ),
    "gemini-3.1-pro": ModelConfig(
        model_identifier="models/gemini-3.1-pro",
        name="gemini-3.1-pro",
        description="Google's Pro tier, with strong reasoning and the largest context window in this catalog at 2M tokens (GA since May 2026). It is no longer Google's frontier model — Gemini 4 Argon shipped on September 30, 2026 and is stronger than anything here — but Argon is Fairwind-only and uncallable with an ordinary API key (see the note above GOOGLE_MODELS), so this remains the Pro tier in practice and the only 2M-context option. The long wait is over rather than ongoing: Gemini 3.5 Pro slipped repeatedly and was never released, so there is no 3.5 Pro to migrate to, and Argon is the successor to plan for once its access opens to paid API customers. Unlike the 3.6/3.7 Flash generation, it still honours temperature/top_p/top_k. Tiered pricing: $2/$12 per 1M input/output tokens for prompts up to 200K tokens, rising to $4/$18 above 200K.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=2097152,
        maximum_output_tokens=65536,
        token_param_name="max_output_tokens",
        supports_temperature=True,
        input_cost_hint=2.0,
        output_cost_hint=12.0,
    ),
    "gemini-3.1-flash-lite": ModelConfig(
        model_identifier="models/gemini-3.1-flash-lite",
        name="gemini-3.1-flash-lite",
        description="Legacy — succeeded by Gemini 3.5 Flash-Lite (July 2026) for current-generation quality, but remains available at the lowest price point ($0.25/$1.50 per 1M tokens). Optimized for high-throughput, low-latency, cost-sensitive applications. GA as of May 2026. The only model here with an announced end date: Google's deprecations page lists a shutdown of May 7, 2027, with gemini-3.5-flash-lite as the recommended replacement. (The separate gemini-3.1-flash-lite-preview endpoint was shut down May 25, 2026 and is not this model.)",
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
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Deprecated and ignored from Gemini 3.6 Flash / 3.5 Flash-Lite onward.",
        parameter_type=ParameterType.NUMBER,
        default_value=0.5,
        min_value=0,
        max_value=2,
        step=0.1,
        unsupported_models=sorted(_NO_SAMPLING_PARAMS),
    ),
    ParameterConfig(
        field_name="top_p",
        display_name="Top P",
        description="The cumulative probability cutoff for token selection. Tokens are selected in descending probability order until the sum of their probabilities equals this value. Deprecated and ignored from Gemini 3.6 Flash / 3.5 Flash-Lite onward.",
        parameter_type=ParameterType.NUMBER,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        unsupported_models=sorted(_NO_SAMPLING_PARAMS),
    ),
    ParameterConfig(
        field_name="top_k",
        display_name="Top K",
        description="The maximum number of top tokens to consider when sampling. Deprecated and ignored from Gemini 3.6 Flash / 3.5 Flash-Lite onward.",
        parameter_type=ParameterType.NUMBER,
        default_value=40,
        min_value=1,
        max_value=100,
        step=1,
        unsupported_models=sorted(_NO_SAMPLING_PARAMS),
    ),
    ParameterConfig(
        field_name="max_output_tokens",
        display_name="Max Output Tokens",
        description="The maximum number of tokens to generate in the response.",
        parameter_type=ParameterType.NUMBER,
        default_value=4096,
        min_value=1,
        max_value={
            "gemini-3.8-flash": 65536,
            "gemini-3.7-flash": 65536,
            "gemini-3.6-flash": 65536,
            "gemini-3.5-flash-lite": 65536,
            "gemini-3.5-flash": 65536,
            "gemini-3.1-pro": 65536,
            "gemini-3.1-flash-lite": 65536,
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


def supports_temperature(model: str) -> bool:
    """Check whether a Gemini model still honours the `temperature` parameter.

    Gemini 3.6 Flash and later (and Gemini 3.5 Flash-Lite) deprecated the
    sampling parameters. The API does NOT reject them — it accepts and silently
    ignores them — so nothing fails loudly when we send one; the request simply
    carries a field that reads as configuration and is not. Callers should omit
    `temperature` when this is False.

    Args:
        model: The model identifier (alias, or the "models/..." identifier).

    Returns:
        True when the model honours `temperature`. Defaults to True for unknown
        models so behavior matches the previous implicit default.
    """
    if model in GOOGLE_MODELS:
        return GOOGLE_MODELS[model].supports_temperature
    for config in GOOGLE_MODELS.values():
        if config.model_identifier == model:
            return config.supports_temperature
    model_lower = (model or "").lower()
    for name, config in GOOGLE_MODELS.items():
        if name in model_lower or config.model_identifier in model_lower:
            return config.supports_temperature
    return True
