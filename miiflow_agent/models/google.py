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
# level the earlier generations offered, and Gemini 4 Argon keeps that scale —
# `minimal`, `xhigh` and `max` are all a 400 there.
#
# Gemini 4 Argon is in this set for the same reason the Flash models are, but
# it is NOT silent about everything: `thinking_budget` — which the 3.1 Pro
# generation takes — returns a 400 on Argon rather than being ignored. The set
# only governs the three sampling parameters, so that one is called out in the
# Argon entry's description instead of being inferred from membership here.
_NO_SAMPLING_PARAMS = {
    "gemini-4-argon",
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
# Also deliberately absent: the Nano Banana image models (gemini-nano-banana-2.1,
# gemini-3.1-flash-image, gemini-3.1-flash-lite-image, gemini-3-pro-image).
# They output images, billed per image token at a different rate from their
# text and thinking tokens, which a text ModelConfig cannot describe. miiflow-web
# catalogs them with their rates in server/core/gemini_image_models.py.
#
# gemini-4-argon is NO LONGER ABSENT. Three audits tracked it as announced
# (September 30, 2026) but Fairwind-only and uncallable with an ordinary key;
# it went generally available on October 13, 2026 with the stable model id
# `gemini-4-argon`, and it is catalogued below as the first entry. It is not on
# the Gemini API free tier or the AI Pro plan in AI Studio, but a paid API key
# reaches it, which is the bar every other entry here is held to.
#
# Two things about it that a flat ModelConfig cannot say, both of which the
# entry's description repeats for the reader who never gets here:
#   * ITS PRICE STEPS UP ON DECEMBER 1, 2026. $2/$10 per 1M input/output with
#     cached input at $0.10 runs through November 30, 2026; from December 1 the
#     rates are $4/$20 with cached input at $0.20. The hints below are the
#     introductory tier, so an audit after November 30 must double all three or
#     every Argon request bills at half cost. Note the MECHANISM differs by
#     platform and only the first-party rate is what this catalog serves: on the
#     Gemini API it is introductory pricing (the $2 is the price), while Vertex
#     bills the $4 list rate and returns 50% as credits, so a Vertex-sourced
#     invoice will not show $2 per 1M at all.
#   * MAX OUTPUT IS THREE NUMBERS, and the one this catalog carries is the
#     SMALLEST. A non-streaming direct-HTTP request returns at most 32,768
#     output tokens — down from 65,536 on Gemini 3.x, and what Argon's own
#     Models API entry reports as `outputTokenLimit`; streaming reaches 65,536;
#     and only "long decode" — setting max_output_tokens above the cap and
#     following `continuation_token` across requests, automatic in the Google
#     GenAI SDK and manual over raw REST — reaches the 1,048,576 the headlines
#     quote. `maximum_output_tokens` is 32,768 because GeminiClient's
#     non-streaming path IS a non-streaming direct-HTTP POST to
#     `:generateContent` (providers/gemini_client.py), so 65,536 would be a
#     ceiling this code cannot reach: a maxOutputTokens above 32,768 with no
#     continuation token is reported to 400 there. Under-promising costs a
#     streaming caller some headroom; over-promising fails the request. The 1M
#     figure is the shared context window, which input, thinking and output all
#     draw from, not a per-request output allowance.
# Explicit context caching is NOT supported on Argon (implicit only, 6,144-token
# minimum), so the cache-read rate below applies to cache hits Google decides on
# rather than to a prefix this code pins.
#
# Argon also settles the question the gemini-3.1-pro entry below spent three
# audits tracking: Gemini 4 now exists, with a name, a price and a context
# window, and 3.1 Pro is no longer the Pro tier by default.
#
# Still deliberately absent from the Gemini 4 line: gemini-4-argon-cyber-permissive,
# which the Vertex pricing table lists beside Argon at the post-introductory
# $4/$20 (the promotion explicitly does not apply to it). It is the same core
# model without the cyber guardrails, served to Fairwind-programme defenders
# only, so it would fail for every key this catalog serves — the same reasoning
# as gemini-3.8-flash-cyber above and gpt-5.6-cyber in the OpenAI catalog.
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
# Cached-input rates are now wired for every model here (the October 5, 2026
# audit; previously only gemini-3.5-flash carried one, which billed every other
# model's cached tokens at the full input rate). They were second-hand when
# written — quotations of the pricing table, cross-checked against the 0.1x
# ratio the catalog already knew — and the October 12, 2026 audit CONFIRMED
# every one of them against the primary source: cloud.google.com is reachable
# from this environment (ai.google.dev is not), and Vertex AI's global
# standard-tier column matches this file exactly on all seven models, input,
# output and cached input alike — $0.75/$3.75/$0.075 on the three newer Flash
# models, $1.50/$9.00/$0.15 on 3.5 Flash, $0.30/$2.50/$0.03 on 3.5 Flash-Lite,
# $2/$12/$0.20 under 200K and $4/$18/$0.40 above it on 3.1 Pro, and
# $0.25/$1.50/$0.025 on 3.1 Flash-Lite. Read the GLOBAL column: Vertex adds 10%
# for non-global endpoints and Batch halves everything again, so one page shows
# three plausible-looking rate sets per model.
# Two caveats a flat hint cannot express: the three Flash models' $0.075 is the
# same INTRODUCTORY rate their input price is, and doubles to $0.15 on January 1,
# 2027 alongside it; and gemini-3.1-pro's $0.20 is the under-200K tier, $0.40
# above it, mirroring its tiered input price. Google also bills cache STORAGE per
# token-hour ($0.50/1M/hour on the Flash tier through December 31, 2026, $1.00
# after; $4.50 on 3.1 Pro), which has no field in ModelConfig at all and is
# therefore not captured here — explicit caching costs more than these rates say.
# The October 19, 2026 audit re-read the same Vertex table and found all seven
# of those models unchanged, and added gemini-4-argon's row to the list it
# confirms: $2/$10/$0.10 global standard through November 30, 2026.
GOOGLE_MODELS: Dict[str, ModelConfig] = {
    "gemini-4-argon": ModelConfig(
        model_identifier="models/gemini-4-argon",
        name="gemini-4-argon",
        description="Google's frontier model and the strongest thing in this catalog (announced September 30, 2026, generally available October 13, 2026). It replaces Gemini 3.8 Flash as the default here for hard reasoning, long-horizon agentic work and coding; keep 3.8 Flash for latency- and cost-sensitive volume, where it is still a third of Argon's price. Three audits carried it as uncallable — launch access was Google's Fairwind programme only — and that is over: a paid API key reaches it, though it is not on the free tier or the AI Pro plan in AI Studio. PRICING IS INTRODUCTORY AND STEPS UP: $2/$10 per 1M input/output tokens with cached input at $0.10 through November 30, 2026, then $4/$20 with cached input at $0.20 from December 1, 2026 — the hints here are the introductory tier, so costing after that date doubles. On the Gemini API the $2 is the price; Vertex instead bills $4 and returns 50% as credits, so the two platforms' invoices do not look alike. 1M token context window shared by input, thinking AND output, so a large prompt leaves less room to answer in. Max output is three different numbers and the catalog carries the smallest: 32,768 on a non-streaming direct-HTTP request (down from 65,536 on Gemini 3.x, and what Argon's Models API entry reports as its outputTokenLimit), 65,536 when streaming, and 1,048,576 only through long decode, where max_output_tokens is set above the cap and `continuation_token` is followed across requests — automatic in the Google GenAI SDK, manual over raw REST. This client's non-streaming path is a direct HTTP POST, so 32,768 is the ceiling a plain call here can actually reach. Thinking depth is thinking_level (LOW/MEDIUM/HIGH, default MEDIUM); `minimal` is gone and so are `xhigh`/`max`, which 400. Sampling parameters are accepted and silently ignored, as on 3.6 Flash and later, but thinking_budget is a hard 400 here rather than ignored — a 3.1 Pro request carrying one fails rather than degrades. Structured outputs, function calling and multimodal (text, image, video, audio, PDF) input are supported; context caching is IMPLICIT ONLY, with a 6,144-token minimum and no explicit cache to pin.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        supports_structured_outputs=False,
        reasoning=True,
        maximum_context_tokens=1048576,
        # The NON-STREAMING per-request ceiling, which is the path
        # GeminiClient's `chat()` takes — not the 65,536 a streaming call
        # reaches, and not the 1M long decode reaches. See the note above
        # GOOGLE_MODELS.
        maximum_output_tokens=32768,
        token_param_name="max_output_tokens",
        supports_temperature=False,
        input_cost_hint=2.0,
        output_cost_hint=10.0,
        cache_read_cost_hint=0.10,  # 0.05x input; $0.20 from December 1, 2026
    ),
    "gemini-3.8-flash": ModelConfig(
        model_identifier="models/gemini-3.8-flash",
        name="gemini-3.8-flash",
        description="Google's best Flash model (released September 2, 2026), and the right default for latency- and cost-sensitive volume here. It is no longer the most capable model in this catalog, nor the default for hard work: Gemini 4 Argon went generally available on October 13, 2026 and is catalogued above — but Argon costs $2/$10 against this model's introductory $0.75/$3.75, so the two split the work rather than one replacing the other. Improves on Gemini 3.7 Flash for reasoning, coding, and agentic workflows, with native grounding, computer use, and multimodal (text, image, video, audio, PDF) input. 1M token context window, 64K max output. Thinking depth is controlled with thinking_level (LOW/MEDIUM/HIGH, default MEDIUM); temperature, top_p, top_k, candidate_count and the frequency/presence penalties are accepted and silently ignored. Introductory pricing of $0.75/$3.75 per 1M input/output tokens (output includes thinking tokens) applies through December 31, 2026, rising to $1.50/$7.50 on January 1, 2027.",
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
        cache_read_cost_hint=0.075,  # 0.1x input; $0.15 from January 1, 2027
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
        cache_read_cost_hint=0.075,  # 0.1x input; $0.15 from January 1, 2027
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
        cache_read_cost_hint=0.075,  # 0.1x input; $0.15 from January 1, 2027
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
        cache_read_cost_hint=0.03,  # 0.1x input
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
        description="Legacy — succeeded by Gemini 4 Argon, which went generally available on October 13, 2026 and is callable with an ordinary paid API key, so the three audits that kept this entry as \"the Pro tier in practice\" are over and new work belongs on Argon. It is stronger on every axis and CHEAPER on output at the introductory rate ($2/$10 against the $2/$12 below), and it stays cheaper on input above 200K prompts, where this model steps to $4/$18. Two reasons to keep it pinned: it is the only 2M-context option here, twice Argon's 1M window; and it still honours temperature/top_p/top_k, which Argon accepts and ignores. There is no 3.5 Pro between them — it slipped repeatedly and was never released. GA since May 2026. Tiered pricing: $2/$12 per 1M input/output tokens for prompts up to 200K tokens, rising to $4/$18 above 200K. Migration is not a model-id swap: Argon rejects thinking_budget with a 400 rather than ignoring it, so drop it before switching.",
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
        cache_read_cost_hint=0.20,  # 0.1x input below 200K; $0.40 above it
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
        cache_read_cost_hint=0.025,  # 0.1x input (text/image/video; audio $0.05)
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
            # Argon's NON-STREAMING cap, the one GeminiClient's `chat()` path
            # is held to. Streaming reaches 65,536 and long decode 1,048,576 by
            # following `continuation_token`, neither of which is a bound this
            # parameter can express.
            "gemini-4-argon": 32768,
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
