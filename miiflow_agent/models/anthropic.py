"""Anthropic model configurations."""

from typing import Dict, Optional

from .base import ModelConfig, ParameterConfig, ParameterType

# EFFECTIVE COST, NOT JUST PER-TOKEN COST: Claude Opus 4.7 and later — that is
# Opus 4.7 / 4.8 / 5 / 5.5, Sonnet 5 and 5.5, Fable 5 and Fable 5.1 — use a newer
# tokenizer that emits roughly 30% MORE tokens for the same text than the one
# Opus 4.6, Sonnet 4.6 and Haiku 4.5 use. The prices below are the published
# per-token rates and are correct as written, but a like-for-like comparison
# across that boundary has to scale the newer models' token counts up by ~30%:
# Sonnet 5 at $2/$10 is not simply "cheaper than Sonnet 4.6 at $3/$15" on the
# same prompt. The exact increase depends on the content and workload shape,
# so this is a caveat for cost modelling, not a multiplier to hard-code.

ANTHROPIC_MODELS: Dict[str, ModelConfig] = {
    # Ordered newest-first on purpose: `_resolve_model_name`'s third tier is a
    # SUBSTRING match, and "claude-fable-5" is a substring of every Fable 5.1
    # identifier. Fable 5.1 must be reached first or a Bedrock/Vertex spelling
    # of it (`anthropic.claude-fable-5-1`) resolves to Fable 5 and gets Fable
    # 5's capabilities and prices. The same collision exists one tier down:
    # "claude-opus-5" is a substring of every Opus 5.5 identifier, so Opus 5.5
    # is listed above Opus 5, and "claude-sonnet-5" is a substring of every
    # Sonnet 5.5 identifier, so Sonnet 5.5 is listed above Sonnet 5. Getting
    # that order wrong is not cosmetic — it bills the newer model at the older
    # one's rates and tells callers thinking can be disabled with
    # `{"type": "disabled"}` on a model that 400s on it.
    "claude-fable-5.1": ModelConfig(
        model_identifier="claude-fable-5-1",
        name="claude-fable-5.1",
        description="Anthropic's most capable widely released model (released September 1, 2026), for demanding reasoning, long-horizon agentic coding, multistep research, and document/spreadsheet/slide work. Same input and output prices as Fable 5, with cache reads at a quarter of the rate ($0.25 vs $1.00 per 1M) — 2.5% of input, where every other Claude model charges 10% — which Anthropic estimates at ~25% lower cost on typical token-billed workloads and up to ~45% on highly agentic ones. Always-on adaptive thinking (default effort high), structured outputs, 1M context window. Forced tool use is NOT supported: it returns an error, so structured output must go through the native path, never the forced-`json_tool` fallback. Claude Mythos 5.1 shares its specifications and pricing but is invitation-only (Project Glasswing), so Fable 5.1 is the top tier reachable with a standard API key. No longer Anthropic's newest model — Opus 5.5 shipped September 22, 2026 and Sonnet 5.5 on September 28 — but still the most capable one, and the model to reach for when evals on Opus 5.5 at higher effort fall short. Retirement not sooner than September 1, 2027.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        # Must stay True: the tool-based JSON fallback in AnthropicClient forces
        # `tool_choice`, which this model rejects outright.
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=False,
        input_cost_hint=10.0,
        output_cost_hint=50.0,
        cache_read_cost_hint=0.25,  # 0.025x input — the Fable 5.1 / Mythos 5.1 rate
        cache_write_cost_hint=12.5,  # 1.25x input (5-min TTL); $20 at 1h
    ),
    "claude-fable-5": ModelConfig(
        model_identifier="claude-fable-5",
        name="claude-fable-5",
        description="Legacy — succeeded by Claude Fable 5.1 (September 1, 2026), which carries the same specifications and per-token price but bills cache reads at a quarter of the rate, so this is kept for pinned workloads only. Generally available since June 9, 2026 (API access was briefly suspended June 12–July 1, 2026 under a US export-control directive and has since been restored). Always-on adaptive thinking, structured outputs, 1M context window. Retirement not sooner than June 9, 2027.",
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
    "claude-opus-5.5": ModelConfig(
        model_identifier="claude-opus-5-5",
        name="claude-opus-5.5",
        description="Anthropic's newest Opus (released September 22, 2026) and the recommended default for most workloads, built for long-running agentic coding and knowledge work. Undercuts Opus 5 on price ($4/$20 vs $5/$25) while beating it, and reads cached tokens at 5% of input ($0.20 vs $0.50 per 1M) where every Claude model outside the Fable line charges 10%. Always-on adaptive thinking that CANNOT be disabled: `thinking: {\"type\": \"disabled\"}` is a 400 at every effort level, unlike Opus 5 where it is rejected only at xhigh/max. Default effort is `medium`, not `high` — a request that omits `effort` runs one level lower than it did on Opus 5. Forced tool use is NOT supported (it returns an error), so structured output must go through the native path, never the forced-`json_tool` fallback. All five effort levels, structured outputs, 1M context window, 128K max output. Fast mode runs at $8/$40. The first of the Claude 5.5 family, and no longer the newest of it: Claude Sonnet 5.5 shipped September 28, 2026 and is catalogued here. Claude Haiku 5.5, named at this launch as following \"in the coming weeks\", still has no date, price, benchmark or model card at the October 5, 2026 audit — it is absent from both the models overview and the deprecations table — and so cannot be catalogued yet; two weeks on from being named, it remains the most likely Anthropic addition at the next audit. Note the effort default still differs from Sonnet 5.5, which defaults to high. Retirement not sooner than September 22, 2027.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        # Must stay True: the tool-based JSON fallback in AnthropicClient forces
        # `tool_choice`, which this model rejects outright.
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=False,
        input_cost_hint=4.0,
        output_cost_hint=20.0,
        cache_read_cost_hint=0.20,  # 0.05x input — the Opus 5.5 rate
        cache_write_cost_hint=5.0,  # 1.25x input (5-min TTL); $8 at 1h
    ),
    "claude-opus-5": ModelConfig(
        model_identifier="claude-opus-5",
        name="claude-opus-5",
        description="Legacy — succeeded by Claude Opus 5.5 (September 22, 2026), which is both stronger and cheaper ($4/$20 vs $5/$25) and reads cached tokens at 5% of input rather than 10%, so this is kept for pinned workloads only. Released July 24, 2026. Always-on adaptive thinking with an xhigh reasoning-effort mode, a Fast Mode (2.5x faster at 2x the price), structured outputs, and a safety fallback that routes to Opus 4.8. Thinking can still be disabled here at effort <= high, which Opus 5.5 rejects outright. 1M context window. Retirement not sooner than July 24, 2027.",
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
        description="Legacy — succeeded by Claude Opus 5 (July 24, 2026). Powerful reasoning and coding model with adaptive thinking, structured outputs, and fast mode; remains available and serves as Opus 5's safety fallback. 1M context window. Retirement not sooner than May 28, 2027.",
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
        description="Legacy — succeeded by Claude Opus 4.8 (May 2026). Strong coding, reasoning, and agentic performance with adaptive thinking. 1M context window. Retirement not sooner than April 16, 2027.",
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
        description="Legacy — succeeded by Claude Opus 4.7 (April 2026). The oldest model here that still accepts the sampling parameters, and the last before manual extended thinking was removed (it still works but is deprecated). Effort tops out at max — it predates the xhigh level and rejects it. 128K max output tokens, 1M context window. Retirement not sooner than February 5, 2027.",
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
    "claude-sonnet-5.5": ModelConfig(
        model_identifier="claude-sonnet-5-5",
        name="claude-sonnet-5.5",
        description="Anthropic's newest model (released September 28, 2026) and the best combination of speed and intelligence, succeeding Sonnet 5 at identical prices ($2/$10, cache reads $0.20, cache writes $2.50). Anthropic measures output more than 30% faster and up to 30% lower cost per task than Sonnet 5, from the speed and from fewer tool calls; strongest on well-scoped everyday work, bug fixes, and polished documents, slides and spreadsheets. Same tokenizer as Sonnet 5, so identical text bills the same token count. Adaptive thinking on by default, DEFAULT EFFORT high (Opus 5.5 defaults to medium), all five effort levels, 1M context window, 128K max output. Three parameter changes from Sonnet 5, each a 400: `thinking: {\"type\": \"disabled\"}` is rejected and replaced by `thinking: {\"type\": \"between_tools\"}`, which is accepted ONLY at low/medium/high effort and takes no other field; manual `budget_tokens` thinking is rejected; and forced tool use (`tool_choice` of `any` or `tool`) is rejected, so structured output must go through the native path rather than the forced-`json_tool` fallback. Effort levels are recalibrated — an effort sweep carried over from Sonnet 5 does not produce the same amount of thinking. Prompt caching needs only a 512-token prefix here, against 1,024 on Sonnet 5. Its thinking blocks are bound to the model, the conversation and the account, so keep histories append-only. On Amazon Bedrock, structured outputs (including strict tool use) are NOT available for this model. Retirement not sooner than September 28, 2027.",
        support_images=True,
        support_files=True,
        support_streaming=True,
        supports_json_mode=True,
        supports_tool_call=True,
        # Must stay True: the tool-based JSON fallback in AnthropicClient forces
        # `tool_choice`, which this model rejects outright.
        supports_structured_outputs=True,
        reasoning=True,
        maximum_context_tokens=1000000,
        maximum_output_tokens=128000,
        token_param_name="max_tokens",
        supports_temperature=False,
        input_cost_hint=2.0,
        output_cost_hint=10.0,
        cache_read_cost_hint=0.2,  # 0.1x input
        cache_write_cost_hint=2.5,  # 1.25x input (5-min TTL); $4 at 1h
    ),
    "claude-sonnet-5": ModelConfig(
        model_identifier="claude-sonnet-5",
        name="claude-sonnet-5",
        description="Legacy — succeeded by Claude Sonnet 5.5 (September 28, 2026), which is stronger and faster at exactly the same prices, so this is kept for pinned workloads only. It is still the right target for two things Sonnet 5.5 cannot do: forced tool use (`tool_choice` of `any` or `tool`), and turning thinking off with `thinking: {\"type\": \"disabled\"}` — Sonnet 5.5 400s on both. It is also the server-side fallback Sonnet 5.5 retries \"cyber\" and \"frontier_llm\" refusals onto. Released June 30, 2026, succeeding Sonnet 4.6 and closing much of the gap with Opus 4.8 on reasoning, tool use, and coding. Adaptive thinking is on by default; manual extended thinking and non-default temperature/top_p/top_k are rejected. All five effort levels, 1M context window, 1,024-token minimum cacheable prompt (Sonnet 5.5 needs only 512). $2/$10 per 1M input/output tokens is the standard price — the launch rate was announced as introductory through August 31, 2026, and Anthropic then cancelled the scheduled September 1, 2026 increase to $3/$15, which has passed with the $2/$10 rate standing (re-confirmed on the pricing page at the October 5, 2026 audit, where the footnote still calls $2/$10 the standard price and still records that the increase will not occur). Retirement not sooner than June 30, 2027.",
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
        description="Legacy — succeeded by Claude Sonnet 5 (June 2026), which is both stronger and cheaper ($2/$10 vs $3/$15), so this is kept for pinned workloads only. Supports adaptive thinking, structured outputs and the sampling parameters; manual extended thinking still works but is deprecated. Effort tops out at max — it predates the xhigh level and rejects it. 1M context window. Retirement not sooner than February 17, 2027.",
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
        description="Anthropic's fastest model with near-frontier intelligence, delivering Sonnet-4-level coding performance at one-third the cost and more than twice the speed. The only model here on extended thinking only: it rejects adaptive thinking and the effort parameter with a 400. 200K context window. Retirement not sooner than October 15, 2026 — the nearest retirement floor in this catalog, and TEN DAYS out as of the October 5, 2026 audit. Read it as a floor, not a shutdown date: Anthropic gives at least 60 days' notice before retiring a public model, the deprecations table still lists this model as Active with no deprecation date, and no notice had been sent as of October 5, so the earliest it can actually retire is around December 4, 2026 and the floor will move. The notice Anthropic did send on September 30, 2026 — Sonnet 4.5 deprecated for a November 30, 2026 retirement, 61 days out, replacement claude-sonnet-5-5 — is that 60-day commitment being honoured in practice, and is the shape to expect here when it comes. A newer Haiku still exists on paper only: Anthropic named Claude Haiku 5.5 at the Opus 5.5 launch on September 22, 2026 as shipping \"in the coming weeks\", and at the October 5 audit it remains unannounced — no date, price, benchmark or model card, and absent from both the models overview and the deprecations table — so it cannot be catalogued yet, two weeks after it was named. Until it does ship, the fallback for this tier is Sonnet 5.5 at low effort.",
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
            "claude-fable-5.1": 128000,
            "claude-fable-5": 128000,
            "claude-opus-5.5": 128000,
            "claude-opus-5": 128000,
            "claude-opus-4.8": 128000,
            "claude-opus-4.7": 128000,
            "claude-opus-4.6": 128000,
            "claude-sonnet-5.5": 128000,
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
    "claude-fable-5.1",
    "claude-fable-5",
    "claude-opus-5.5",
    "claude-opus-5",
    "claude-opus-4.8",
    "claude-opus-4.7",
    "claude-sonnet-5.5",
    "claude-sonnet-5",
}

# Models that THINK BY DEFAULT when the request omits `thinking` (adaptive is
# the default, not off) and that accept `thinking: {"type": "disabled"}` at
# effort <= high. The Fable line and Opus 5.5 also think by default but reject
# "disabled" with a 400 at every level, so they live in
# `_THINKING_ON_BY_DEFAULT_LOCKED` below instead; Sonnet 5.5 rejects it too but
# has a REPLACEMENT rather than nothing, so it lives in
# `_THINKING_OFF_VIA_BETWEEN_TOOLS`; Opus 4.8/4.7/4.6 and Sonnet 4.6 default to
# no thinking, so there is nothing to disable.
_THINKING_ON_BY_DEFAULT_DISABLEABLE = {
    "claude-opus-5",
    "claude-sonnet-5",
}

# Thinks by default, rejects `{"type": "disabled"}` with a 400, and turns
# up-front thinking off through `{"type": "between_tools"}` instead — the
# lowest thinking setting on the model. Kept apart from the two sets around it
# because it is neither: the off switch exists, but it is spelled differently
# and it is gated on effort the opposite way round from Opus 5. `between_tools`
# is accepted at low/medium/high and 400s at xhigh/max, and it takes NO other
# field — `display`, `budget_tokens` or `block_binding` alongside it is also a
# 400 — so the value returned below has to stay a bare one-key dict.
_THINKING_OFF_VIA_BETWEEN_TOOLS = {
    "claude-sonnet-5.5",
}

# The effort levels at which `between_tools` itself is refused. The same pair
# of levels gates "disabled" on Opus 5, by a different mechanism: there the
# request succeeds and the model keeps thinking, here the request fails
# outright unless the caller falls back to adaptive thinking.
_BETWEEN_TOOLS_REJECTED_AT_EFFORT = frozenset({"xhigh", "max"})

# Thinks by default AND refuses to be turned off. Spelled as a set rather than
# a name compared inline in `thinks_by_default`, because that comparison was
# `name == "claude-fable-5"` and silently answered False for Fable 5.1 — a
# model with exactly the same behaviour. A caller told thinking is off budgets
# `max_tokens` for text alone and gets `stop_reason=max_tokens` with no text.
# Opus 5.5 belongs here and NOT in the effort-gated set below: Anthropic
# documents `thinking: {"type": "disabled"}` as a 400 on it at every effort
# level, where Opus 5 rejects it only at xhigh/max.
_THINKING_ON_BY_DEFAULT_LOCKED = {
    "claude-fable-5.1",
    "claude-fable-5",
    "claude-opus-5.5",
}


# `output_config.effort` — the GA knob that scales adaptive thinking (and overall
# token spend) on the models that think by default. `budget_tokens` is rejected
# on Sonnet 5 / Opus 5.5 / Opus 5 / 4.7 / 4.8 / Fable 5 / Fable 5.1, so this is
# the ONLY way to bound their deliberation short of disabling thinking (which is
# discouraged: with thinking off these models sometimes write a tool call into
# visible text, and Opus 5.5 and the Fable line refuse it outright).
# Haiku 4.5 and older models 400 on the parameter.
#
# Note the DEFAULT differs: Opus 5.5 defaults to `medium`, every other model
# here — Sonnet 5.5 included — to `high`. A request that omits effort therefore
# runs one level lower on Opus 5.5 than the same request did on Opus 5, and at
# the same level on Sonnet 5.5 as it did on Sonnet 5. Sonnet 5.5 has however
# RECALIBRATED what each level spends, so an effort sweep tuned on Sonnet 5
# does not carry over unchanged.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# The levels are NOT uniform across the models that accept the parameter, so a
# single flat tuple is not the contract: `xhigh` is a newer level, and Opus 4.6
# / Sonnet 4.6 support `max` but return a 400 on `xhigh`. This map is the wire
# contract per model; EFFORT_LEVELS above stays the validation vocabulary a
# caller may configure, so one config survives a model swap and the per-model
# projection happens at request time (see `effort_levels`).
_EFFORT_LEVELS_BY_MODEL: Dict[str, tuple] = {
    "claude-fable-5.1": EFFORT_LEVELS,
    "claude-fable-5": EFFORT_LEVELS,
    "claude-opus-5.5": EFFORT_LEVELS,
    "claude-opus-5": EFFORT_LEVELS,
    "claude-opus-4.8": EFFORT_LEVELS,
    "claude-opus-4.7": EFFORT_LEVELS,
    "claude-sonnet-5.5": EFFORT_LEVELS,
    "claude-sonnet-5": EFFORT_LEVELS,
    "claude-opus-4.6": ("low", "medium", "high", "max"),
    "claude-sonnet-4.6": ("low", "medium", "high", "max"),
}

# Models that reject `thinking: {"type": "disabled"}` once effort is `xhigh` or
# `max` — the combination is a 400, enforced per request. Only Opus 5 lands
# here: Sonnet 5 predates the restriction and accepts "disabled" at every
# level, and Opus 5.5 went further and rejects it at every level, so it is in
# `_THINKING_ON_BY_DEFAULT_LOCKED` instead and never reaches this gate.
_DISABLED_REJECTED_AT_EFFORT = frozenset({"xhigh", "max"})
_DISABLED_EFFORT_GATED_MODELS = frozenset({"claude-opus-5"})


def effort_levels(model: str) -> tuple:
    """The `output_config.effort` values `model` accepts, or `()` if none."""
    return _EFFORT_LEVELS_BY_MODEL.get(_resolve_model_name(model) or "", ())


def supports_effort(model: str) -> bool:
    """True when `model` accepts `output_config: {"effort": ...}`."""
    return bool(effort_levels(model))


def thinks_by_default(model: str) -> bool:
    """True when `model` runs adaptive thinking unless the request disables it.

    A superset of `_THINKING_ON_BY_DEFAULT_DISABLEABLE`: the Fable models think
    by default but reject `thinking: {"type": "disabled"}`, so there are models
    where thinking is on and cannot be turned off, and Sonnet 5.5 thinks by
    default but spells the off switch `between_tools`. Callers that budget
    `max_tokens` for text alone need this to know when the budget is shared.
    """
    name = _resolve_model_name(model) or ""
    return (
        name in _THINKING_ON_BY_DEFAULT_DISABLEABLE
        or name in _THINKING_ON_BY_DEFAULT_LOCKED
        or name in _THINKING_OFF_VIA_BETWEEN_TOOLS
    )


def thinking_disable_param(
    model: str, effort: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """The `thinking` request value that turns thinking OFF for `model`, or None.

    Callers that want a short, deterministic, cheap completion (a compaction
    handoff note, a title, a classifier) must pass this explicitly on the
    thinking-by-default models: otherwise adaptive thinking runs first and
    `max_tokens` — a hard cap on thinking PLUS text — can be consumed entirely
    by the thinking block, returning `stop_reason=max_tokens` with no text at
    all. The VALUE is per-model, not a constant: Sonnet 5.5 renamed the setting
    to `{"type": "between_tools"}` and 400s on "disabled", so a caller that
    hard-codes either spelling is wrong on part of the catalog. Returns None
    where nothing needs sending (defaults to off), where the API would reject
    every off switch outright (the Fable models, Opus 5.5), or where the model
    rejects the one it has at the effort level this request carries (Opus 5 and
    Sonnet 5.5 at `xhigh` / `max`) — hence `effort`: the answer depends on the
    whole request, not the model alone, and returning the parameter without it
    is a guaranteed 400.
    """
    name = _resolve_model_name(model)
    if name in _THINKING_OFF_VIA_BETWEEN_TOOLS:
        if (effort or "").lower() in _BETWEEN_TOOLS_REJECTED_AT_EFFORT:
            return None
        return {"type": "between_tools"}
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
