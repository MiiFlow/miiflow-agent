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
#
# Also deliberately absent: gpt-live-1 (released September 10, 2026). It is a
# full-duplex VOICE front end billed per MINUTE ($0.05/min) that delegates the
# reasoning to whatever model sits behind it — it has no token prices and does
# not take a chat-completions request, so a ModelConfig here (whose whole
# contract is per-token pricing plus a token_param_name) cannot describe it.
#
# Also deliberately absent: gpt-6-astra-law (Astra for Law, launched
# September 17, 2026, re-checked October 9, 2026 and still API-unlisted). It is GPT-6 Astra plus legal instructions and a Legal
# Search Index tool — a configuration of Astra, not a new base model — reached
# today through ChatGPT and a "Trusted Access" programme for law firms. OpenAI
# names `gpt-6-astra-law` as the coming API id but has published neither a date
# nor a price, so there is nothing to price and the id would 404. Re-checked at
# the October 9, 2026 audit and still unlisted — it does not appear in the API
# model list under that id or any other; with GPT-6.1 Astra cancelled (below)
# it remains the most likely OpenAI addition at the next one.
#
# Also deliberately absent: gpt-6-luna-pro and gpt-6-sol-pro. Third-party
# catalogues list these as models; OpenAI does not. They are the same ids served
# with `reasoning.mode: "pro"`, the same shape GPT-5.6 Pro uses, so they are an
# execution mode on `gpt-6-sol` / `gpt-6-luna` rather than slugs to send.
#
# Also deliberately absent: the ULTRAFAST speed tier announced at DevDay on
# September 29, 2026 (faster in the API at 6x the standard price — $60/$300 on
# Astra, $12/$60 on GPT-6.1 Sol). It is `service_tier: "ultrafast"` on an
# existing model id, not a model. Note this makes THREE speed tiers with three
# different shapes: standard, the `-fast` SUFFIX on the id (2x/2x, stripped by
# `_base_model_name`), and this `service_tier` value. A
# `gpt-6-astra-ultrafast` or `gpt-6.1-sol-ultrafast` id would 404.
#
# Both are now live: GPT-6 Astra Ultrafast shipped at DevDay, and GPT-6.1 Sol
# Ultrafast — "promised in the coming days" at the October 5, 2026 audit —
# rolled out on October 8, 2026 across the API, Codex and ChatGPT Work at
# $12/$60 per 1M (OpenAI quotes up to 8x standard Sol throughput). API access
# is open to developers; the ChatGPT and Codex side is gated to Pro 500,
# enterprise and education. Because the tier multiplies BOTH prices by exactly
# 6 on both models, a consumer that wants to bill it can scale the catalog
# hints rather than carry duplicate entries — the same shape as
# LONG_CONTEXT_*_MULTIPLIER below, and the reason this stays out of the
# catalog rather than becoming two more ModelConfigs.
#
# gpt-5.4-nano is on a shutdown clock: deprecated October 1, 2026, removed from
# the API April 1, 2027, replacement gpt-6-luna. It stays in this set (and in the
# catalog) until then so pinned workloads keep resolving, and comes out of both —
# plus the `max_completion_tokens` map below — at the first audit after that date.
# gpt-5.4-cyber, retired in the same window, was never catalogued here.
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

# GPT-6 shares one parameter contract, which is NOT the GPT-5.6 one: the family
# drops `temperature`, `top_p`, `top_logprobs`, `logprobs` and
# `prompt_cache_retention` outright (the replacement is
# `prompt_cache_options.ttl`, e.g. "30m"). `-fast` is a latency variant of the
# same id (2x speed, 2x price), not a separate model; pro is `reasoning.mode`,
# the same shape GPT-5.6 uses, so neither gets its own catalog entry.
#
# The EFFORT SCALE is the one thing that is not uniform across the family, which
# is why Astra, the Sol/Luna tier and GPT-6.1 are separate sets rather than one:
# Astra dropped `none` (and `minimal`) and returns a 400 on either, while Sol and
# Luna — released nineteen days later — accept `none` and in fact REQUIRE it to
# call function tools over Chat Completions. Folding them into one set would
# advertise `none` on Astra (a guaranteed 400) or withhold it from Sol/Luna
# (which forces every tool call onto the Responses API for no reason).
_GPT6_ASTRA_MODELS = {
    "gpt-6-astra",
}
# GPT-6 Sol and GPT-6 Luna, generally available September 22, 2026. There is no
# GPT-6 Terra: OpenAI did not carry the GPT-5.6 mid tier into this generation,
# so the family is Astra (flagship) > Sol > Luna.
_GPT6_SOL_LUNA_MODELS = {
    "gpt-6-sol",
    "gpt-6-luna",
}
# GPT-6.1, generally available September 29, 2026. One model: `gpt-6.1-sol`.
# There is no GPT-6.1 Astra — OpenAI cancelled it over internal safety findings
# (deceptive behaviour), confirmed at DevDay on September 29 — and no GPT-6.1
# Luna or Terra, so GPT-6 Astra stays the capability ceiling and `gpt-6-luna`
# stays the cheap tier. Either id would 404.
#
# It takes the GPT-6 parameter contract but ASTRA's effort scale, not the one
# belonging to the tier whose name it shares: `none` and `minimal` are rejected,
# so it is neither `_GPT6_SOL_LUNA_MODELS` (which would advertise a 400) nor
# folded into `_GPT6_ASTRA_MODELS` (which would put it in a set named for a
# different model and quietly claim `reasoning.mode: "pro"` works here, which
# OpenAI does not document for 6.1). Because `none` is what Chat Completions
# needs to call function tools, dropping it means tool calling on this model is
# Responses-only in practice, exactly as on Astra.
_GPT61_MODELS = {
    "gpt-6.1-sol",
}
_GPT6_MODELS = _GPT6_ASTRA_MODELS | _GPT6_SOL_LUNA_MODELS | _GPT61_MODELS
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
# The only NON-reasoning models left in this catalog.
#
# gpt-4.1-nano WAS a third member here and was removed at the September 29,
# 2026 audit: OpenAI's deprecations page lists it (and
# `gpt-4.1-nano-2025-04-14`) with an API shutdown of October 23, 2026 and
# `gpt-5.6-luna` as the replacement. It is the only 4.1 model on that page —
# `gpt-4.1` and `gpt-4.1-mini` are not on it and are NOT deprecated. This
# catalog steers to `gpt-6-luna` rather than OpenAI's suggested
# `gpt-5.6-luna`: same tier, newer generation, half the price ($0.10/$0.50 vs
# $0.20/$1.20).
#
# The distinction matters because two different October dates circulate for
# the 4.1 family and NEITHER is an API shutdown of gpt-4.1 or gpt-4.1-mini:
# the February 13, 2026 retirement notice covers ChatGPT only and says the API
# is unchanged, and Microsoft Foundry retires FINE-TUNED gpt-4.1 and
# gpt-4.1-mini deployments on October 14, 2027 — a different platform and a
# different year. So the two models left here are superseded, not expiring:
# keep steering new work to GPT-6.1 Sol / GPT-6 Luna, but do not tell anyone
# their requests stop working this month. Re-check the deprecations page every
# audit; developers.openai.com was not reachable from the audit environment,
# so the October 23 date rests on search results quoting that page.
#
# Re-checked September 30, 2026 and unchanged: `gpt-4.1` still has a live model
# page and no API shutdown date, and the October 14, 2026 date that third-party
# "GPT-4.1 is retiring" write-ups keep repeating still traces to Azure/Foundry
# deployments, not the OpenAI API. One thing that IS newly on the page:
# `gpt-5.4-cyber` is removed from the API on October 1, 2026, replaced by
# `gpt-5.6-cyber`. Neither is in this catalog (both are Daybreak-gated), so
# there is nothing to remove here.
#
# When these two do go, so does every OpenAI model that takes `temperature`,
# `max_tokens`, `frequency_penalty` and `presence_penalty` — the audit that
# removes them must also drop those four ParameterConfigs rather than leave
# the UI offering controls no remaining model accepts.
_GPT41_MODELS = {"gpt-4.1", "gpt-4.1-mini"}

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
    # The whole GPT-6 family carries the same 2x input / 1.5x output multipliers
    # as the GPT-5 models past 272K input: Astra bills $20 / $2 / $75 and Sol
    # $4 / $15 over the line. It applies to the FULL request, not the overage.
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
# to happen here rather than at each call site. The GPT-6 family — Astra, Sol,
# Luna and GPT-6.1 Sol alike — removed the sampling/logprob controls and
# replaced `prompt_cache_retention` with `prompt_cache_options.ttl`.
#
# Sol and Luna document the sampling drops as conditional ("remove temperature,
# top_p and top_logprobs when effort is not none"), but they are listed here
# unconditionally on purpose: this catalog cannot see the effort a given request
# will carry, every call site here runs at the medium default rather than
# `none`, and providers already return a flat 400 on temperature for both
# (reported against Bedrock's Converse path). Dropping a parameter that would
# have been accepted at effort=none costs nothing; sending one that is rejected
# fails the request.
_UNSUPPORTED_REQUEST_PARAMS: Dict[str, frozenset[str]] = {
    model: frozenset(
        {"logprobs", "prompt_cache_retention", "temperature", "top_logprobs", "top_p"}
    )
    for model in _GPT6_ALL
}


OPENAI_MODELS: Dict[str, ModelConfig] = {
    # Ordered newest-release-first. Unlike the Anthropic catalog, order here is
    # presentational only: every lookup goes through `_matches_model_family`,
    # which matches the id exactly or as a dated snapshot (`-20…`) and never as a
    # substring, so "gpt-6-sol" cannot capture "gpt-6.1-sol".
    #
    # GPT-6.1 Sol — generally available September 29, 2026. OpenAI's newest
    # model, but NOT its most capable: GPT-6 Astra below is still the ceiling.
    "gpt-6.1-sol": ModelConfig(
        model_identifier="gpt-6.1-sol",
        name="gpt-6.1-sol",
        description="OpenAI's newest model (generally available September 29, 2026, announced at DevDay) and the recommended default for most workloads: an upgrade to GPT-6 Sol for agentic coding, computer use, and professional work that OpenAI positions as near-Astra intelligence at a fifth of Astra's token price. Priced identically to GPT-6 Sol at $2/$10 per 1M input/output tokens, with cache writes also unchanged at $2.50 — the one price that moved is CACHED INPUT, halved from $0.20 to $0.10, i.e. 5% of the input rate where the rest of the family reads at 10%. Past 272K input tokens the whole request bills at 2x input / 1.5x output ($4/$15). 1.05M context window (922K of it input), 128K max output, text and image input (plus PDF), structured outputs, prompt caching. Same parameter contract as the rest of GPT-6: temperature, top_p, top_logprobs and logprobs are removed, and prompt_cache_retention is replaced by prompt_cache_options.ttl. But the EFFORT SCALE is Astra's, not GPT-6 Sol's: effort runs low through max and defaults to medium, and `none`/`minimal` return a 400 — so unlike GPT-6 Sol there is no effort level at which Chat Completions will call function tools, and tool calling is Responses-only. `reasoning.mode: \"pro\"` is NOT documented for this tier and is therefore not offered here. There is no GPT-6.1 Astra: OpenAI cancelled it over internal safety findings. An Ultrafast speed tier (service_tier, 6x price) is promised for this model \"in the coming days\" and is not served yet.",
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
        output_cost_hint=10.0,
        cache_read_cost_hint=0.10,  # 5% of input — half the rest of GPT-6's 10%
        cache_write_cost_hint=2.5,  # Cache writes: 1.25x input
    ),
    # GPT-6 (Astra) — generally available September 3, 2026. Current flagship.
    "gpt-6-astra": ModelConfig(
        model_identifier="gpt-6-astra",
        name="gpt-6-astra",
        description="GPT-6 Astra is OpenAI's flagship and most capable model (generally available September 3, 2026), a single dense reasoning model for complex coding, reasoning, and long-horizon agentic work. It is no longer OpenAI's newest model — GPT-6 Sol and GPT-6 Luna shipped September 22, 2026 beneath it and GPT-6.1 Sol on September 29 — so reach for Astra when GPT-6.1 Sol at high effort falls short, not by default. 1.05M context window, 128K max output. $10/$50 per 1M input/output tokens, with cached input at $1 and cache writes at $12.50; past 272K input tokens the whole request bills at $20/$2/$75. Effort is the biggest cost dial — Artificial Analysis measures a 3.6x swing from low to max. Its parameter contract differs from GPT-5.6: temperature, top_p, top_logprobs (and logprobs on Chat Completions) are removed, prompt_cache_retention is replaced by prompt_cache_options.ttl, and the effort scale drops none/minimal — unlike Sol and Luna, Astra returns a 400 on `none` (GPT-6.1 Sol shares that restriction). Tool calling requires the Responses API; `max` effort is Responses-only. Appending -fast to the model id runs it at up to 2x speed for 2x the price, and the Ultrafast service tier (announced at DevDay, live on this model) runs up to 6x faster at 6x the price — $60/$300. It also stays the flagship for longer than expected: OpenAI cancelled GPT-6.1 Astra, its planned October successor, over internal safety findings, confirmed publicly at DevDay on September 29, 2026 — so do not hold migrations for it.",
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
    # GPT-6 Sol / Luna — generally available September 22, 2026. Luna is still
    # the cheapest route to GPT-6-generation reasoning; Sol was superseded by
    # GPT-6.1 Sol seven days after it shipped.
    "gpt-6-sol": ModelConfig(
        model_identifier="gpt-6-sol",
        name="gpt-6-sol",
        description="Legacy — succeeded by GPT-6.1 Sol (September 29, 2026), which is stronger at the SAME $2/$10 per 1M input/output tokens and reads cached input at half the price ($0.10 vs $0.20), so this is kept for pinned workloads only. Generally available September 22, 2026 alongside GPT-6 Luna. It is still the right target for one thing GPT-6.1 Sol cannot do: effort `none`, which is what Chat Completions requires to call function tools — GPT-6.1 Sol 400s on `none`, putting every tool call on the Responses API. It is also the tier that documents `reasoning.mode: \"pro\"`, which OpenAI has not documented for 6.1. Carries the GPT-6 advances into a cost-efficient high-end tier, reaching an estimated 90-95% of Astra's practical capability at roughly a fifth the cost per task. Cache writes $2.50. These are standard rates, not a promotion. Past 272K input tokens the whole request bills at 2x input / 1.5x output ($4/$15). 1.05M context window (922K of it input), 128K max output, text and image input. Effort runs none through max and defaults to medium. Same parameter contract as the rest of GPT-6: temperature, top_p, top_logprobs and logprobs are removed, and prompt_cache_retention is replaced by prompt_cache_options.ttl (e.g. \"30m\"). Pro is `reasoning.mode`, not a separate model id.",
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
        output_cost_hint=10.0,
        cache_read_cost_hint=0.20,  # GPT-6 cached input: 10% of input rate
        cache_write_cost_hint=2.5,  # Cache writes: 1.25x input
    ),
    "gpt-6-luna": ModelConfig(
        model_identifier="gpt-6-luna",
        name="gpt-6-luna",
        description="The fast, cost-efficient tier of the GPT-6 family (generally available September 22, 2026), for high-volume, latency-sensitive work: extraction, classification, structured summarisation and lightweight agentic steps. Still current: the September 29, 2026 GPT-6.1 release covered the Sol tier only, so there is no GPT-6.1 Luna and this remains the cheap tier to steer to. OpenAI positions it as matching the previous generation's flagship quality at roughly a tenth of its cost. Priced at $0.10/$0.50 per 1M input/output tokens — half GPT-5.6 Luna's $0.20/$1.20 and the cheapest model in this catalog — with cached input at $0.01 and cache writes at $0.125. These are standard rates, not a promotion. Past 272K input tokens the whole request bills at 2x input / 1.5x output. 1.05M context window (922K of it input), 128K max output, text and image input. Effort runs none through max and defaults to medium; `none` is what Chat Completions requires to call function tools, and anything above it needs the Responses API. Same parameter contract as the rest of GPT-6: temperature, top_p, top_logprobs and logprobs are removed, and prompt_cache_retention is replaced by prompt_cache_options.ttl. Pro is `reasoning.mode`, not a separate model id.",
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
        input_cost_hint=0.10,
        output_cost_hint=0.50,
        cache_read_cost_hint=0.01,  # GPT-6 cached input: 10% of input rate
        cache_write_cost_hint=0.125,  # Cache writes: 1.25x input
    ),
    # GPT-5.6 series (Sol / Terra / Luna) — generally available July 9, 2026.
    # gpt-5.6 is an API alias for gpt-5.6-sol.
    "gpt-5.6-sol": ModelConfig(
        model_identifier="gpt-5.6-sol",
        name="gpt-5.6-sol",
        description="Legacy — succeeded by GPT-6 Astra (September 3, 2026) as OpenAI's flagship and then undercut by GPT-6 Sol (September 22, 2026), which is both stronger and half the price ($2/$10 vs $4/$20). Kept for pinned workloads only; there is no longer a cost argument for it. The highest-intelligence tier of the GPT-5.6 family, built for complex coding, reasoning, and long-horizon agentic work. 1M context window. Available in the API as gpt-5.6-sol (alias: gpt-5.6). Its $4/$20 rate is itself promotional — an August 21, 2026 cut from $5/$30, running through at least November 21, 2026 — so the gap to GPT-6 Sol may widen when it lapses. OpenAI has announced no API deprecation date for the GPT-5.6 family.",
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
        description="GPT-5.6 Terra is the balanced mid-tier of the GPT-5.6 family (July 9, 2026), delivering strong reasoning and agentic performance at well under half the cost of GPT-5.6 Sol. A July 30, 2026 price cut lowered it to $2/$12 per 1M input/output tokens (from $2.50/$15). 1M context window. The mid tier has no GPT-6 successor — OpenAI shipped Astra, Sol and Luna and no Terra — but GPT-6 Sol matches its input price with a lower output price ($2/$10) and a newer generation behind it, so new work belongs there. Terra remains available with no announced deprecation date.",
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
        description="Legacy — succeeded by GPT-6 Luna (September 22, 2026), which is stronger and half the price ($0.10/$0.50 vs $0.20/$1.20), so this is kept for pinned workloads only. The fastest and most cost-efficient tier of the GPT-5.6 family (July 9, 2026), optimized for high-throughput, latency-sensitive workloads. A July 30, 2026 price cut lowered it ~80% to $0.20/$1.20 per 1M input/output tokens (from $1/$6). 1M context window. Both remain available: OpenAI has announced no API deprecation date for the GPT-5.6 family.",
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
        description="GPT-5.5 (released April 23, 2026) — previous flagship, superseded by GPT-5.6 Sol. Strong coding and agentic model (82.7% on Terminal-Bench 2.0, 58.6% on SWE-Bench Pro) with a 1M context window. Note that it is no longer the cheaper choice: at $5/$30 it costs more than GPT-5.6 Sol ($4/$20 since the August 21, 2026 cut) and 2.5x more than GPT-6 Sol ($2/$10), both of which are newer and more capable. Kept for pinned workloads only. Do NOT delete it over the October 14, 2026 date in the news: that retires GPT-5.5 from ChatGPT, ChatGPT Work and Codex only — the API model is untouched and appears nowhere on OpenAI's deprecations page.",
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
        description="GPT-5.4 (March 5, 2026) — two generations old, superseded by the GPT-5.6 family. 1M context window, built-in computer use, and improved deep research. No longer a cost saving: at $2.50/$15 it is priced above both GPT-5.6 Terra ($2/$12) and GPT-6 Sol ($2/$10), each newer and stronger. Kept for pinned workloads only.",
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
        description="GPT-5.4 Mini is a smaller, faster GPT-5.4 variant with strong reasoning at lower cost. 400K context window. Released March 17, 2026. Superseded on price and capability by GPT-6 Luna ($0.10/$0.50 vs $0.75/$4.50) and GPT-5.6 Luna ($0.20/$1.20), both of which carry a 1M context window; Mini remains the current small model on the 5.4 line.",
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
        description="DEPRECATED, WITH A SHUTDOWN DATE — OpenAI announced the retirement of GPT-5.4 Nano on October 1, 2026 and removes it from the API on April 1, 2027, naming gpt-6-luna as the replacement. It is the only model in this catalog with an announced end date, so treat it as a migration target and do not start new work on it; it is kept here only so pinned workloads keep resolving until then. Released March 17, 2026 as the most cost-effective GPT-5.4 variant, optimized for latency. 400K context window. The named replacement is also the cheaper one on both sides: GPT-6 Luna at $0.10/$0.50 is half its input price and a fifth of its output price, with a 1M context window (GPT-5.6 Luna is the other way out, matching its input price at a lower output price, $1.20 vs $1.25). OpenAI deprecated gpt-5.1 and gpt-5.3-codex in the same October 1, 2026 notice, for the same April 1, 2027 shutdown and both pointed at gpt-6-sol; neither is catalogued here. Six months is OpenAI's stated minimum notice for a generally available model — but it is not always honoured: gpt-5.4-cyber was announced on September 11, 2026 and removed on October 1, twenty days later.",
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
        description="Superseded — migrate to GPT-6 Sol. General-purpose non-reasoning model with a 1M token context window, retired from ChatGPT on February 13, 2026 with no change to the API at that time. At $2/$8 it is no longer a saving: GPT-6 Sol matches its input price at $2/$10 and GPT-6 Luna is a twentieth of it at $0.10/$0.50, both newer and stronger. At the October 5, 2026 audit OpenAI's deprecations page still listed NO API shutdown for it; the only 4.1 model on that page is gpt-4.1-nano, which shuts down October 23, 2026 and has been dropped from this catalog. NOT re-verified at the October 9, 2026 audit: platform.openai.com and developers.openai.com were both unreachable from the audit environment, so this is the October 5 finding carried forward rather than a fresh check, and it is the first thing to re-read from an environment that can reach the page. The October 14, 2026 date an earlier audit recorded for this model could not be reconfirmed and appears to trace to Microsoft Foundry's October 14, 2027 retirement of fine-tuned gpt-4.1 deployments. Migrate on capability and price, not on a deadline. Note that OpenAI began winding down the self-serve fine-tuning platform on May 8, 2026 — existing fine-tunes still serve until their base model is deprecated, but new ones are no longer generally available.",
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
        description="Superseded — migrate to GPT-6 Luna. Smaller GPT-4.1 with a 1M token context window, retired from ChatGPT on February 13, 2026 with no change to the API. GPT-6 Luna undercuts it four-fold ($0.10/$0.50 vs $0.40/$1.60) and reasons besides. At the October 5, 2026 audit OpenAI had published no API shutdown date for it — unlike gpt-4.1-nano, which shuts down October 23, 2026 and has been dropped from this catalog. Carried forward unverified at the October 9, 2026 audit, which could not reach the deprecations page (see `gpt-4.1` above). The October 14, 2026 date an earlier audit recorded for this model could not be reconfirmed (see the note on _GPT41_MODELS).",
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
            "gpt-6.1-sol": 128000,
            "gpt-6-astra": 128000,
            "gpt-6-sol": 128000,
            "gpt-6-luna": 128000,
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
        # Deliberately NOT `_GPT6_MODELS`: OpenAI documents `reasoning.mode` on
        # GPT-5.6, Astra and GPT-6 Sol/Luna, but not on the GPT-6.1 tier. An
        # undocumented mode offered in a dropdown is a 400 waiting to be picked,
        # so 6.1 is excluded until OpenAI publishes it.
        supported_models=list(
            _GPT56_MODELS | _GPT6_ASTRA_MODELS | _GPT6_SOL_LUNA_MODELS
        ),
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
    **{
        model: ["low", "medium", "high", "xhigh", "max"]
        for model in _GPT6_ASTRA_MODELS
    },
    # GPT-6.1 Sol follows Astra here, not the GPT-6 Sol it succeeds: OpenAI's
    # model page lists low through max and marks `none` and `minimal` as not
    # supported. Offering `none` because the tier name matches GPT-6 Sol's would
    # put a 400 behind a UI dropdown.
    **{
        model: ["low", "medium", "high", "xhigh", "max"]
        for model in _GPT61_MODELS
    },
    # Sol and Luna put `none` back — and it is load-bearing, not cosmetic:
    # Chat Completions will only call function tools at effort `none`, so
    # withholding it forces every tool call onto the Responses API.
    **{
        model: ["none", "low", "medium", "high", "xhigh", "max"]
        for model in _GPT6_SOL_LUNA_MODELS
    },
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
    Completions for plain turns but requiring Responses for tool calling. GPT-6.1
    Sol lands here by the same route as Astra: Chat Completions will only call
    function tools at effort `none`, and 6.1 rejects `none`.
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
    surface. GPT-6.1 Sol carries the same surface — OpenAI routes it exactly as
    GPT-6 (Responses API, xhigh/max effort, reasoning context, verbosity).
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
