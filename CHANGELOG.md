# Changelog

All notable changes to miiflow-agent will be documented here.

## [1.14.0] - 2026-07-24

### Added
- **Claude Opus 5** (`models/anthropic.py`, `core/llms/bedrock.py`): Registered Anthropic's newest Opus model (released July 23, 2026, model id `claude-opus-5`), which delivers near-Fable 5 performance at half the token price ($5/$25 per 1M input/output, same as Opus 4.8). Always-on adaptive thinking with an xhigh reasoning-effort mode, a Fast Mode (2.5x faster at 2x price), structured outputs, and a safety fallback that routes to Opus 4.8. 1M context, 128K max output. Registered with `supports_temperature=False` and in `_NO_EXTENDED_THINKING` (adaptive thinking, no manual extended-thinking param), matching the other current flagship models. Matching Bedrock entry (`us.anthropic.claude-opus-5`) added.
- **Gemini 3.6 Flash** (`models/google.py`): Registered Google's newest and most capable Flash model (released July 21, 2026, model id `models/gemini-3.6-flash`). It beats Gemini 3.1 Pro on coding at roughly 25% lower cost, with native grounding and multimodal (text, image, video, audio, PDF) input. 1M-token context window (1,048,576), 65,536 max output, `max_output_tokens` accounting, full tool/image/file/streaming/JSON-mode support, and per-token pricing hints of $1.50/$7.50 per 1M input/output tokens.

### Changed
- **Claude Opus 4.8 → legacy** (`models/anthropic.py`, `core/llms/bedrock.py`): Marked as succeeded by Claude Opus 5. It remains available (unchanged pricing/limits) and serves as Opus 5's safety fallback. The Adlyse orchestrator (`assistant/configs/platform_assistants.py`) and the whole-account structural strategist specialist (`assistant/configs/specialists.py`) — the two deepest-reasoning, once-per-run Opus roles — were rolled from `claude-opus-4.8` to `claude-opus-5` (same price, better reasoning).
- **Gemini 3.5 Flash → legacy** (`models/google.py`): Marked as succeeded by Gemini 3.6 Flash. It is no longer the most capable Flash model; pricing/limits are unchanged and it remains available. Server-side runtime references that pinned `gemini-3.5-flash` for creative/video tagging and swipe-file inference were rolled to `gemini-3.6-flash` (a cheaper, better drop-in with the same multimodal surface).

## [1.13.0] - 2026-07-22

### Added
- **Caller-preseeded run media store (`core/react/orchestrator.py`)**: `execute()` now merges a `media_ref -> URL` map the caller pre-installs on `ctx.deps["media_store"]` (or `ctx.run_state.media_store`) into the run's `ExecutionState.media_store` at setup, and points both ctx surfaces at it — mirroring how `event_bus` / `dispatch_counter` are provisioned. This lets an adapter rehydrate media the model referenced in earlier turns so a `media_ref:<id>` from a prior turn still resolves, instead of failing with "Unknown media_ref … must come from a tool call earlier in this run" and forcing a fresh (non-identical) regeneration. Backward compatible: runs that don't preseed get the same empty store as before, and media generated during the run merges into the same dict.

## [1.12.0] - 2026-07-15

### Added
- **GPT-5.6 Sol / Terra / Luna models** (`models/openai.py`): Registered the generally-available GPT-5.6 family — `gpt-5.6-sol` (flagship), `gpt-5.6-sol-pro` (served with `reasoning.mode=pro`), `gpt-5.6-terra` (balanced mid-tier at ~half Sol's cost), and `gpt-5.6-luna` (fastest, most cost-efficient). Each carries a 1M-token context window, 128K max output, `max_completion_tokens` token accounting, full tool/image/file/streaming/JSON-mode support, and per-token pricing hints. GPT-5.6 is now the current flagship line; GPT-5.5 and GPT-5.4 descriptions were updated to reflect their legacy status.
- **`reasoning_effort` is now forwarded to OpenAI** (`models/openai.py`, `providers/openai_client.py`): A configured `reasoning_effort` is actually sent on Chat Completions requests for the o-series and GPT-5 families — previously it was declared as a model parameter but silently dropped before reaching the API. A new `supports_reasoning_effort()` helper gates the parameter to models that accept it, so it is never sent to a standard chat model (which would 400).

### Fixed
- **`reasoning_effort` + function tools 400 on the GPT-5 family** (`providers/openai_client.py`): OpenAI rejects `reasoning_effort` when it is combined with function `tools` on `/v1/chat/completions` ("Function tools with reasoning_effort are not supported for gpt-5.6-terra in /v1/chat/completions"). The client now omits `reasoning_effort` on tool-calling turns — falling back to the model's default effort — instead of raising, on both the streaming and non-streaming paths.

## [1.11.1] - 2026-07-08

### Fixed
- **Symbolic `media_ref` passthrough in the orchestrator** (`core/react/orchestrator.py`): `ReActOrchestrator` no longer eagerly resolves `media_ref:<id>` inputs to URLs for tool parameters that are declared to consume the symbolic reference directly. Some tools must recover the backing asset from `media_store` themselves — for example before saving a generated image into a workspace — and URL substitution broke that path. URL resolution is unchanged for every other parameter, and media-ref resolution now runs consistently across both single-step and batched (parallel) tool invocations.

## [1.11.0] - 2026-07-07

### Added
- **Durable multi-agent handoff foundation**: A coordinated set of changes that moves dispatch-tree state out of the model's hands and into durable, code-owned storage (durable-execution model, cf. LangGraph checkpoints), so what one agent learns and does reliably reaches the agents that come after it.
  - **Canonical observation store** (`core/observation.py`, `core/__init__.py`): every ReAct tool execution now persists exactly one full observation record through a new `ObservationSink` port, awaited inline at the orchestrator's observation seams. Every other surface — execution-timeline items, SSE frames, sub-agent traces — carries a policy-bounded excerpt plus the record's ref instead of duplicating multi-hundred-KB payloads, and a new universal `read_observation` tool fetches full depth on demand. New package-root exports: `ObservationSink`, `ObservationRecord`, `StoredObservation`, `get_observation_sink`, `OBSERVATION_SINK_DEPS_KEY`.
  - **Tree-wide durable checkpoint + blackboard** (`core/checkpoint.py`): the root thread's checkpoint becomes the single durable store for the whole dispatch tree. Compare-and-set writes with a commutative merge (bounded ledger reducer + keyed fact upsert) replace the full-blob rewrite that silently dropped a concurrent turn's state under parallel dispatch; the ledger is bounded to short digests + observation refs (no raw payloads) and pruned on every persist.
  - **Read-through dedupe gate** (`core/react/dedupe.py`, `core/tools/schemas.py`, `core/tools/decorators.py`): opt-in serve contracts on `ToolSchema` (`idempotency_class` + `dedupe_scope_dims`) let identical reads be served from the ledger by code rather than prompt guidance, with a validity predicate that never serves business-error payloads and single-flight coalescing that closes the parallel-sibling duplicate-read race.
  - **Tool-derived fact promotion with precedence** (`core/checkpoint.py`): `EstablishedFact` gains a `source` precedence lattice enforced in `upsert_fact` — user answers are never overwritten by tool-promoted facts — so structural knowledge one agent discovers (e.g. connected ad accounts: platform, id, name, currency) surfaces to every later agent and every prompt.
  - **Ledger-backed handoff worklog** (`core/checkpoint.py`): `Checkpoint.render_worklog_block()` gives a dispatched child a bounded `[work_already_done]` digest spanning both tool calls and dispatches, both successes and failures (so children stop repeating a sibling's failed call), each entry carrying its observation ref for `read_observation` expansion and a producer address so a continued session excludes its own prior entries — replacing the racy by-reference contextvar scratchpad.
  - **Conversational continuity for specialists** (`core/checkpoint.py`): `Checkpoint.child_sessions` lets a re-dispatch to a specialist you already talked to continue one session child thread (bounded, recycled after 20 uses) instead of re-discovering context in a fresh thread; concurrent dispatches to the same handle deterministically fall back to isolation.
- **Claude Sonnet 5** (`models/anthropic.py`, `core/llms/bedrock.py`): Anthropic's most agentic Sonnet model (released June 30, 2026, model id `claude-sonnet-5`), succeeding Sonnet 4.6. Adaptive thinking is on by default — manual extended thinking and non-default sampling params (temperature/top_p/top_k) are rejected — so it is registered with `supports_temperature=False` and in `_NO_EXTENDED_THINKING`. 1M context, 128K max output, structured outputs. Introductory pricing $2/$10 per 1M tokens through Aug 31, 2026, reverting to $3/$15. Matching Bedrock entry added.

### Changed
- **Model roster metadata refresh** (`models/anthropic.py`, `models/google.py`, `models/openai.py`): Claude Fable 5 marked generally available again — the June 12 US export-control suspension was lifted July 1, 2026 and calls succeed once more; Gemini 3.1 Pro corrected to a 2M-token context window with tiered pricing ($2/$12 per 1M input/output up to 200K, $4/$18 above); Claude Sonnet 4.6 max output corrected to 128K. The GPT-5.6 (Sol/Terra/Luna) series is intentionally still not registered — it remains a limited preview whose API identifiers are not final and would surface "no access"/"model not found" errors for bring-your-own-key users.
- **Claude Sonnet 4.6 → legacy** (`models/anthropic.py`, `core/llms/bedrock.py`): Marked as succeeded by Claude Sonnet 5.
- **Default sub-agent model rolled to Sonnet 5** (`core/react/configured_subagent.py`, `examples/subagents.py`): the configured sub-agent default moves from `claude-sonnet-4.6` to `claude-sonnet-5`.
- **Atomic per-edge dispatch cap** (`core/react/dispatch.py`): `DispatchCounter.reserve` now takes a `per_handle_limit` checked under the counter's lock (min'd with the global cap), replacing a racy pre-check in the dispatch tool that assumed serial dispatch while the schema is actually parallelizable.

### Fixed
- **Pinned tools survive the in-process tool_search gate** (`core/react/tool_executor.py`): explicitly pinned tools are no longer hidden when the registry is large enough to trigger tool search, so an agent that depends on a pinned tool always sees it.

### Removed
- **Retired OpenAI `o3`** **(breaking for callers pinning this ID)** (`models/openai.py`): Removed the deprecated `o3` reasoning model (deprecated June 11, 2026, API shutdown December 11, 2026) along with its parameter-map and `_REASONING_MODELS` entries.
- **Dead `Checkpoint.transcript` field** (`core/checkpoint.py`): removed a checkpoint field that never had a writer (it always serialized `[]`); `from_dict` now drops the key from old blobs. The conversation record lives in message rows with per-message excerpt/ref timelines.

  #### Migration notes
  - OpenAI: move to the GPT-5.x series (e.g. `gpt-5.5` / `gpt-5.4`) for reasoning workloads.

## [1.10.0] - 2026-06-29

### Added
- **Durable pause/resume with intent-scoped approvals** (`core/checkpoint.py`, `core/interrupt.py`, `core/react/orchestrator.py`): New typed `EstablishedFact` and `PendingInterrupt` models (exported from the package root) make clarification and approval state deterministic — questions are keyed by a stable slug so a re-asked question short-circuits to its known answer instead of pausing again. A single `GraphInterrupt` primitive unifies clarification, tool-approval, and plan-approval flows; the orchestrator catches it, checkpoints the typed interrupt, and surfaces it through one path.
- **Multi-choice clarification questions** (`core/tools/clarification.py`): The clarification tool now carries multiple `ClarificationQuestion` entries per pause, each optionally multi-select, so agents can present structured choices in one round-trip instead of open-ended text prompts.
- **Native Anthropic server-side tool search** (`core/react/tool_executor.py`, `core/tools/tool_search.py`): When the registry is large enough to warrant tool search on an Anthropic model, the full tool list is sent every turn with non-core tools flagged `defer_loading: true`; the API strips deferred tools from the prompt (preserving the cache prefix) and the model discovers them via the server-side regex, replacing the in-process meta-tool on that path.
- **Bounded parallel tool execution** (`core/react/tool_executor.py`): Parallel tool batches are now gated by a semaphore (`MIIFLOW_MAX_PARALLEL_TOOLS`, default 8) — excess calls queue and start as slots free, preventing event-loop, provider-rate-limit, and downstream-API stampedes when the model emits very wide batches.
- **Claude Fable 5** (`models/anthropic.py`): Added Anthropic's newest flagship for demanding reasoning and long-horizon agentic work.

### Changed
- **Model roster refresh** (`models/anthropic.py`, `models/openai.py`, `models/google.py`): Added Claude Fable 5; promoted Gemini 3.1 Pro to GA (`gemini-3.1-pro`, dropping the `-preview` suffix). See **Removed** for retired entries.
- **Anthropic prompt-cache breakpoints** (`providers/anthropic_client.py`, `core/stream_normalizer.py`): Cache-control markers are now placed correctly across multi-turn conversations, plus suggestion-run cost linkage and structured log correlation for observability.
- **Strict tool-schema caps** (`providers/anthropic_client.py`): Union-typed parameters (`anyOf`/`oneOf` or multi-entry `type`) are now counted and bounded to 16 per request alongside the existing 20-tool / 24-optional-param caps; exceeding any cap demotes excess tools to non-strict in precedence order, and a residual "schema too complex" 400 triggers an automatic full demotion fallback.
- **MCP tool registration safety** (`core/tools/registry.py`): An empty allowlist passed to `register_mcp_manager()` now registers all discovered tools (rather than silently hiding them); restriction requires a non-empty set, and a loud warning fires if a manager advertised tools but none registered. MCP/HTTP tools are also now visible to schema lookup, context-injection checks, and `has_tool` queries.

### Fixed
- **Agent and video-fetch reliability** (`providers/anthropic_client.py`, `core/tools/decorators.py`): Corrected strict-default handling and tool-decorator behavior so subscription webhook events and video creative displays surface reliably.
- **Soft-deleted rows double-counted in ad metrics** (`core/react/tool_executor.py`): Pacing and creative-timeseries metrics no longer double-count soft-deleted rows during parallel evaluation.
- **Multi-agent campaign creation/update** (`core/tools/clarification.py`, `core/react/events/bus.py`): Reserved kwargs are scoped to avoid collisions when agents share a tool, and clarification serialization handles both the legacy single-question and new multi-question shapes.

### Removed
- **Retired model entries** **(breaking for callers pinning these IDs)**: Removed Claude `claude-opus-4.5`, `claude-sonnet-4.5`, and `claude-opus-4.1` (`models/anthropic.py`); OpenAI `o4-mini`, `o3-mini`, `o3-pro`, and `gpt-4o-mini` (`models/openai.py`); and Google `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, plus the preview aliases `gemini-3.1-pro-preview` and `gemini-3-flash-preview` (`models/google.py`).

  #### Migration notes
  - Anthropic: move to `claude-fable-5` or `claude-opus-4.8`.
  - OpenAI: move to the GPT-5.x series (or `o3` where a reasoning model is required).
  - Google: move to `gemini-3.5-flash` or `gemini-3.1-pro` (the GA replacement for `gemini-3.1-pro-preview`).

## [1.9.0] - 2026-05-28

### Added
- **New model definitions** (`models/anthropic.py`, `models/openai.py`, `models/google.py`): Claude Opus 4.8 (the new Anthropic flagship, succeeding 4.7), the OpenAI o4-mini reasoning model (200K context), and Google Gemini 3.5 Flash. Parameter configs and successor metadata added for each; older entries (Opus 4.7, Gemini 3.1 Pro, Gemini 3-flash-preview, o3-mini) re-pointed accordingly.
- **`RepeatedToolErrorCondition` safety condition** (`core/react/safety.py`, `core/react/enums.py`): Stops an agent loop when the same tool returns the same error code N times in a row regardless of input — the schema-validation retry loops that `ErrorThresholdCondition` (soft errors never reach step level) and `RepeatedActionsCondition` (input varies each retry) both miss.
- **Structured child-failure propagation** (`core/react/orchestrator.py`, `core/react/configured_subagent.py`, `core/react/events/bus.py`, `core/react/execution/state.py`, `core/subagent.py`): When a sub-agent halts on a safety condition, `_extract_failure_metadata()` walks the last ReAct steps and stashes `last_tool`, `last_tool_error`, `last_tool_input`, and `attempts_seen` on `ExecutionState.failure_metadata`. These surface in the `STOP_CONDITION` event's `failure` field and thread out through `SubAgentResult.failure`, so a parent can report the real cause of a child's failure instead of a generic "ran into repeated issues" message.

### Changed
- **Default client timeout raised 120s → 300s** (`core/client.py`): Removes noisy false-positive timeouts on legitimately long streaming calls.
- **Slow-call observability** (`providers/anthropic_client.py`): A successful Anthropic stream that exceeds 120s now emits a structured `anthropic_stream_slow` warning (model, elapsed seconds, configured timeout), so the timeout bump can be measured against real latency rather than silently masking a regression.

### Fixed
- **MCP/HTTP tools dropped from the tool surface** (`core/react/tool_executor.py`, `core/tools/registry.py`): Lookups that consulted only the registry's `tools` dict silently missed every MCP and HTTP tool. A new `_lookup_any_tool()` helper consults the function/http/mcp dicts uniformly, so MCP and HTTP tools are now visible to `has_tool`, schema lookup, and context-injection checks. `register_mcp_manager()` also gained an optional `allowed_names` filter to register only a subset of discovered MCP tools.
- **MCP/HTTP tools missing schemas via `tool_search`** (`core/react/tool_executor.py`): Schema assembly only handled `FunctionTool`, so MCP/HTTP tools surfaced through `tool_search` were silently dropped from the LLM's tool surface. Any tool exposing `.schema` is now used directly.
- **`RepeatedToolErrorCondition` false-fire on parallel dispatches** (`core/react/safety.py`, `core/react/orchestrator.py`): `_error_key` classified successful tool returns carrying `'error': None` as soft errors, so three or more parallel `dispatch_assistant` calls to the same handle tripped the repeat limit and stopped the root agent before it could process the results. The probe now ignores successful returns, and `_should_stop` logs which safety condition fired.
- **All-failed parallel steps misclassified as runtime failures** (`core/react/orchestrator.py`, `core/react/models.py`, `core/tools/function/function_tool.py`, `core/tools/registry.py`): When every call in a parallel step failed on deterministic input-shape validation (e.g. GAQL preflight), the step was tagged `all_failed` and burned the recovery ladder. An `is_validation_error` marker now propagates from the raised exception's `is_tool_validation_error` attribute; an all-validation-failed step is classified as `schema` so the recovery manager short-circuits — the per-call observations already carry the corrective hint.

## [1.8.0] - 2026-05-18

Folds in the previously unreleased `1.7.1` (a bare version bump from PR #1001) plus the work merged since.

### Added
- **Multi-agent hand-off framework** (`core/subagent.py`, `core/react/dispatch.py`): New `SubAgent` primitive lets a parent agent delegate turns to specialist children with per-edge policy (`when_to_use`, `handoff_schema`, `clarification_policy`, `auto_approve_child_tools`). Dispatch lifecycle and event bubbling are first-class so the streaming layer can surface sub-agent reasoning.
- **`AgentConfig` dataclass** (`core/config.py`): Canonical agent construction. Centralizes client, system prompt, tools, sub-agents, max iterations, etc. The `Agent()` constructor now accepts either the legacy positional args or `config=AgentConfig(...)`.
- **OAuth-enabled MCP servers**: The Anthropic provider client handles OAuth flows for authenticated MCP servers, so agents can call Notion/HubSpot/Stripe/Airtable tools after a one-time consent.
- **Plan-mode tools** (`core/tools/plan_mode.py`): New deferred `enter_plan_mode` / `exit_plan_mode` tools let a model temporarily restrict itself to read-only work while drafting an approach.
- **`tool_search` registry tool**: Agents operating over large tool catalogs can now query for relevant tools by intent rather than seeing every entry up front. Backed by registry-level filter hooks.
- **Data reference helpers** (`core/data_reference.py`): `put_render_data()` / `get_render_data()` share structured payloads across tool calls without re-serializing through the model context.
- **Truncation + recovery events** (`core/react/events/bus.py`): New event types report tool description and parameter truncation, plus error-recovery transitions, so the UI can explain why a tool call was reshaped.
- **Deterministic suggestion scanners + action-taking framework**: New scanner pipeline produces structured suggestions independent of LLM generation, plus an action-taking framework for executing the recommended response.
- **Slack workspace selection**: Slack tool calls can target a specific installed workspace when the org has multiple connections.

### Changed
- **Unified ReAct orchestration**: Planning and parallel multi-agent execution are now emergent behaviors inside the single ReAct loop — `dispatch_assistant` and plan-mode are normal tool calls — rather than separate orchestrators with their own event types. Collapses the agentic-mode surface area significantly.
- **Anthropic responder batching**: Streaming events are batched in `providers/anthropic_client.py` for lower per-chunk overhead.
- **Google Gemini provider** (`providers/gemini_client.py`, `models/google.py`): Migrated from `google-generativeai` to `google-genai`; the SDK is no longer imported at runtime — direct REST via `httpx`. Context window values and model descriptions also refreshed.
- **Tool executor resilience** (`core/tools/exceptions.py`): Clearer exception types for tool failures, truncated schemas, and malformed tool calls; recovery paths route through the new event bus.
- **Tool registry**: Tool descriptions support optional fields and strict-schema compliance.
- **Reasoning panel + dispatch events**: Enhanced dispatch tracing (DD trace context propagation fixed) and sub-agent lifecycle callbacks.
- **Memory + agent consolidation**: Memory orchestration reworked alongside agent consolidation; tool registry and schema normalizer updated to support the new memory tools.

### Fixed
- **Suggested actions error**: Tool preparation failures during suggested-action generation are now handled gracefully instead of aborting the run.
- **Tool feedback to planner**: Tool results now flow into subsequent planning steps as intended.
- **DD tracing in multi-agent dispatch**: Distributed tracing context propagates correctly into sub-agent calls.
- **Asyncio test plumbing**: Tests updated to use `asyncio.run()` instead of deprecated `get_event_loop().run_until_complete()`.

### Removed
- **Legacy orchestrator classes** (breaking for direct importers): `PlanAndExecuteOrchestrator`, `MultiAgentOrchestrator`, `TaskTool` / `TaskToolResult` / `create_task_tool`, and `SubTask` / `Plan` / `PlanExecuteResult` dataclasses are gone — the same outcomes are now expressed via the unified ReAct loop and the new `SubAgent` framework.
- **Legacy `AgentType` enum values** (breaking): `PLAN_AND_EXECUTE`, `PARALLEL_PLAN`, and `MULTI_AGENT` removed; `REACT` is the only mode.
- **`examples/plan_execute.py`**: Replaced by unified ReAct examples.
- **Deprecated OpenAI models**: `gpt-4o` and `gpt-4o-mini` removed from `models/openai.py` (superseded by newer GPT-4.x / GPT-5 variants).

### Migration notes
- If you previously instantiated `Agent(client, ...)`, that signature still works. To opt into the new shape, construct an `AgentConfig(...)` and pass it as the `config=` keyword.
- If you imported `PlanAndExecuteOrchestrator` or `MultiAgentOrchestrator` directly, switch to a single `Agent` with `sub_agents=[SubAgent(...)]` and let the ReAct loop dispatch.
- If you pinned `google-generativeai`, remove the pin — `miiflow-agent` now depends on `google-genai` and uses HTTP directly.

## [1.7.0] - 2026-04-26

### Added
- **Artifact generation** (`miiflow_agent/artifacts/`): `ArtifactResult` marker dataclass for tools that produce downloadable files (PDF, HTML, etc.). Streaming handler detects the `__artifact__` marker, persists the file, and emits a dedicated SSE event — mirroring the `MediaResult` / `VisualizationResult` pattern.
- **Organization-level timezone**: ReAct orchestrator threads an org timezone through prompts so time-sensitive tool calls and answers respect the caller's locale.
- **Generic email tool**: Provider-agnostic email send tool replacing the previous Slack-specific implementation.

### Changed
- **ReAct orchestrator rewrite**: Collapsed the orchestrator from ~440 LOC to a clean action-or-answer loop. Each turn the model emits exactly one of {tool calls → loop continues, text answer → loop exits}; the prompt enforces the invariant. Removed the XML parser layer (`parsing/xml_parser.py`, `_strip_xml_tags_from_answer`) and associated XML integration tests — native tool calling is the only path now.
- **Hybrid memory system + Claude 4.7**: Memory orchestration reworked alongside Claude 4.7 model support in `models/anthropic.py` and `providers/anthropic_client.py`. Tool registry and schema normalizer updated to support the new memory tools.
- **Cost attribution for agent nodes**: Anthropic client cost tracking corrected so workflow agent-node spend is attributed to the right run.
- **Router & client fixes**: Several streaming/tool-call edge cases in the orchestrator (72 lines added in `orchestrator.py`).

### Removed
- **Legacy provider model configurations**: Cleared deprecated entries from `models/anthropic.py`, `models/google.py`, and `models/openai.py` (~270 LOC deleted) — provider model lists now reflect only currently-supported models.
- **XML parser** (`core/react/parsing/xml_parser.py`) and related tests — superseded by native tool calling.

## [1.6.0] - 2026-04-14

### Added
- **`analyze_creative` tool**: Visual LLM analysis of ad creative pixels with URL classifier, dedupe, and provider-aware fallback (Gemini → OpenAI). Supports Meta and Google ad creatives, including video formats.
- **Creative audit foundation**: Tool suite for fetching, ranking, and analyzing Meta + Google creatives — ranking, fatigue detection, carousel expansion, and lazy-loaded media children.
- **`tool_search` tool**: Dynamic tool discovery for agents operating over large tool registries — agents can search for relevant tools by intent rather than seeing the full catalog up front.
- **`tool_filter`**: Registry-level filter hooks so orchestrators can scope the tool catalog per run.
- **Progress tracking** (`core/react/progress.py`): Structured progress events emitted alongside reasoning chunks for long-running runs.
- **Recovery module** (`core/react/recovery.py`): Unified error-recovery path across orchestrators (ReAct, plan-and-execute, multi-agent) with retry and graceful termination.
- **Context compression** (`core/context_compression.py`): Automatic summarization of older turns when approaching context limits.
- **Human-in-the-loop tool approval**: New exceptions and orchestrator hooks let mutating tool calls pause for user approval before executing.

### Changed
- **Streaming restructure**: Reworked streaming plumbing across orchestrators — cleaner event-bus semantics, better separation of reasoning vs. tool vs. progress chunks.
- **Memory system repurpose**: Memory is now guideline-oriented rather than a raw transcript store; orchestrator consults memory via the dynamic tool registry.
- **Prompts & branding**: Updated ReAct / plan-and-execute / multi-agent prompts; refreshed product branding in system messages.
- **Model list updates**: Added latest OpenAI models; refreshed Anthropic and Google model descriptions.
- Clarification flow fixes across multi-agent and plan-and-execute orchestrators.
- Slack workspace selection support: tool calls can target a specific installed workspace.

## [1.5.1] - 2026-03-16

### Added
- **User cancellation support**: `RunContext` now includes a `cancel_event` field and `is_cancelled` property for cooperative cancellation
- **`USER_CANCELLED` stop reason**: New `StopReason` enum value for distinguishing user-initiated stops from other termination conditions

### Changed
- ReAct orchestrator checks for cancellation at each step and exits gracefully
- Plan-and-execute orchestrator checks for cancellation between subtasks and waves
- Multi-agent orchestrator checks for cancellation before subagent launch and before synthesis

## [1.5.0] - 2026-03-08

### Added
- **`MediaResult` type**: New dataclass for image/media tool results that renders as inline markdown (`![alt](url)`) in agent chat, while preserving structured dict output for workflow mode
- **`is_media_result()` helper**: Detection function for `MediaResult` instances and dicts with `__media__` marker

### Changed
- ReAct orchestrator now detects `MediaResult` tool outputs and uses markdown rendering instead of raw `str()` conversion, so generated images display inline in the chat UI
- OpenAI image tool (`OpenAIImageTool`) returns `MediaResult` instead of plain dict
- Stability AI image tool (`StabilityImageTool`) returns `MediaResult` instead of plain dict

## [1.4.0] - 2026-03-04

### Changed
- Bumped `aiohttp` dependency from `^3.9.0` to `^3.13.3` to fix vulnerable dependencies
- Bumped `mcp` optional dependency from `^1.0.0` to `^1.23.0`
- Added CI-based bidirectional sync workflows for standalone repo

## [1.3.0] - 2026-03-04

### Added
- Anthropic extended thinking support (`thinking_enabled`, `budget_tokens` kwargs, `thinking_delta` streaming field)

### Changed
- Simplified ReAct prompts: removed XML `<thinking>`/`<answer>` tags, plain text final answers
- Default `max_steps` increased from 10 to 25
- System prompt priority: framework prompt now takes precedence over user system prompt
- Stream normalizers use keyword arguments for `_build_chunk()`

## [1.2.0] - 2026-03-03

### Added
- **Gemini REST API client**: Rewrote Gemini provider to use direct REST API calls (httpx SSE) instead of the google-generativeai SDK protobuf layer, preserving fields like `thoughtSignature` on `functionCall` parts that the SDK strips out
- **Gemini tool call support**: `GeminiStreamNormalizer` now handles both REST API dict chunks and legacy protobuf chunks, with proper `function_call_metadata` propagation including `thought_signature`
- **New OpenAI model**: o3-pro (200K context, 100K output, $20/$80) — OpenAI's premium reasoning model for maximum accuracy
- ReAct orchestrator now preserves `function_call_metadata` on accumulated tool calls

### Changed
- Updated Anthropic model descriptions to mark older models (Claude Opus 4.5, 4.1, 4; Sonnet 4.5, 4; Claude 3.7 Sonnet; Claude 3.5 Haiku) as legacy with successor references
- Updated Google model descriptions with deprecation/sunset dates (gemini-2.0-flash/flash-lite retiring June 1, 2026; gemini-2.5-pro/flash sunset June 17, 2026)
- Updated OpenAI model descriptions to mark older models (GPT-5, GPT-4o, GPT-4 Turbo, o1, o1-pro, gpt-5-pro) as legacy with successor references

### Removed
- Removed `gemini-3-pro-preview` model (superseded by `gemini-3.1-pro-preview`)

## [1.1.2] - 2026-03-02

### Fixed
- Fixed parallel plan execution: planner now includes available tool names in the planning prompt to prevent hallucinated tool references
- Fixed subtask results being polluted with hallucinated XML tool call tags by stripping them from final answers
- Provider clients (Groq, Mistral, OpenAI, OpenRouter, xAI) now respect `tool_choice` passed via kwargs instead of always defaulting to `"auto"`
- Plan generation now forces the model to use the `create_plan` tool via provider-specific `tool_choice` format

## [1.1.1] - 2026-02-26

### Fixed
- Fixed Anthropic stream normalizer test failures by adding proper `index` and `content_block.type` fields to mock chunks

## [1.1.0] - 2026-02-26

### Added
- **New OpenAI models**: o4-mini (200K context, $1.10/$4.40) and gpt-5-pro (400K context, $15.00/$120.00)
- **New Anthropic models**: claude-opus-4.6 (200K context, 128K output, $5.00/$25.00), claude-sonnet-4.6 (200K context, 64K output, $3.00/$15.00), claude-opus-4.5 (200K context, 64K output, $5.00/$25.00)
- **New Google models**: gemini-3.1-pro-preview (1M context, $2.00/$12.00) and gemini-3-flash-preview (1M context, $0.50/$3.00)

### Changed
- Updated GPT-5, GPT-5.1, GPT-5 Mini, and GPT-5 Nano context windows from 272K to 400K tokens
- Updated o3 pricing from $1.10/$4.40 to $2.00/$8.00 and enabled streaming support
- All new Anthropic models support extended thinking and structured outputs

## [1.0.0] - 2026-01-13

### Changed
- **Package Renamed**: `miiflow-llm` is now `miiflow-agent` to better reflect the package's focus on AI agent orchestration
- **Stable Release**: Marking version 1.0.0 as production-stable
- Updated all imports from `miiflow_llm` to `miiflow_agent`
- Repository URL updated to `https://github.com/Miiflow/miiflow-agent`

### Migration Guide
Update your imports:
```python
# Old
from miiflow_llm import Agent, LLMClient

# New
from miiflow_agent import Agent, LLMClient
```

Update your dependencies:
```bash
# Old
pip install miiflow-llm

# New
pip install miiflow-agent
```

## [0.5.1] - 2026-01-12

### Fixed
- Fixed issue with Bedrock client when using MCP

## [0.5.0] - 2026-01-11

### Added
- **Dynamic Multi-Agent System**: New hierarchical agent spawning system with specialized subagents
  - `TaskTool`: Spawn specialized subagents (explorer, researcher, implementer, reviewer, planner) during execution with nesting depth limits and context isolation
  - `SubAgentRegistry`: Central registry for subagent types with task-based matching and priority sorting
  - `DynamicSubAgentConfig`: Enhanced configuration with per-subagent model selection, tool scoping, and timeout control
  - `ModelSelector`: Intelligent model selection (haiku/sonnet/opus) based on task type, complexity, and budget constraints with cost estimation
- **Tool Execution Callbacks**: New `TOOL_EXECUTED` callback event type with `@on_tool_executed` decorator for tracking tool name, inputs, output, and execution time
- **Post-Tool Response Streaming**: Agent now streams the LLM response after tool execution instead of blocking

### Changed
- Unified Plan & Execute orchestrator: `PlanAndExecuteOrchestrator` now supports `parallel_execution=True` flag, replacing the separate `ParallelPlanOrchestrator`
- Query classifier improvements for better routing of simple use cases
- MCP connection now supports auto-reload for improved development experience

## [0.4.0] - 2025-12-30

### Added
- **Global Callback System**: New `CallbackRegistry` for registering listeners on LLM events (token usage, errors, agent lifecycle). Includes `@on_post_call` decorator and `callback_context` for passing metadata through calls
- **Multi-Agent Orchestrator [beta]**: New orchestrator for parallel subagent execution with lead agent planning and coordination. Supports dynamic team allocation based on query complexity
- **Parallel Plan Orchestrator [beta]**: Wave-based parallel execution of independent subtasks. Topological sorting into execution waves for up to 90% reduction in execution time for parallelizable tasks
- **AG-UI Protocol Support**: Native support for Agent-User Interaction Protocol via optional `agui` extra. New `AGUIEventFactory` for creating standardized AG-UI events
- **Shared Agent State**: Thread-safe shared state module (`SharedAgentState`) for multi-agent coordination following Google ADK patterns

### Changed
- Enhanced event system with new event types for multi-agent and parallel execution workflows
- Improved tool executor with better context handling for nested agent execution

## [0.3.1] - 2025-12-20

### Added
- New xAI Grok models: grok-4-1-fast-reasoning, grok-4-1-fast-non-reasoning, grok-code-fast-1, grok-4-fast-reasoning, grok-4-fast-non-reasoning, grok-4-0709, grok-3, grok-3-mini, grok-2-vision-1212

### Removed
- Deprecated OpenAI models: o1-preview, o1-mini
- Deprecated xAI models: grok-beta, grok-vision-beta

## [0.3.0] - 2025-12-12

### Added
- Plan validation to catch invalid plans early (duplicate tasks, circular dependencies, missing references)
- Real-time streaming during replanning phase so users can see the agent's thinking as it recovers from failures
- Human-readable tool descriptions in events (e.g., "Searching for Tesla stock price" instead of just "search_web")
- Subtask timeout protection to prevent individual tasks from hanging indefinitely (default 120s)
- Context-aware error messages that provide relevant guidance based on what the user was trying to do
- OpenRouter provider support

### Changed
- Simplified API by making ReAct orchestrator the standard for subtask execution (removed `use_react_for_subtasks` flag)
- Richer replanning events with failure context so UIs can show why and how the agent is adapting
- Preserved completed work during replanning to avoid re-executing successful subtasks
- Tool events now include arguments and descriptions for better observability in UIs

### Fixed
- Tool call state sometimes not updating properly in the UI
- Test reliability improvements

## [0.2.0] - 2025-12-05

### Added
- Enhanced system prompts for native tool calling with improved guidance

### Changed
- ReAct orchestrator now exclusively uses native tool calling (removed legacy XML-only path)
- Simplified orchestrator architecture by removing `use_native_tools` parameter
- Enhanced provider implementations (Gemini, OpenAI, Anthropic) for better compatibility
- Improved response handling with XML tag sanitization for cleaner outputs
- Agent.stream() now automatically injects user queries into message context

### Fixed
- Improved null safety in classification logic for edge cases
- Better error handling for unexpected classification responses
- Enhanced schema generation for array types and complex parameters

## [0.1.0] - 2025-11-30

### Added
- Unified interface for 9 LLM providers (OpenAI, Anthropic, Google Gemini, Groq, OpenRouter, Mistral, xAI, Ollama, Bedrock)
- Support for latest models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0)
- Streaming with unified StreamChunk format
- ReAct agents with native tool calling
- Plan & Execute orchestrator for complex multi-step tasks
- Tool system with @tool decorator and automatic schema generation
- Context injection patterns (Pydantic AI compatible)
- Multi-modal support (text + images)
- Async and sync APIs
- Full type hints with generics
- Comprehensive error handling with retry logic
- Token usage tracking and metrics
- Observability support (OpenTelemetry, Prometheus, Arize Phoenix)

### Documentation
- Quickstart guide
- Complete API reference
- Tool tutorial
- Agent tutorial (ReAct + Plan & Execute)
- Provider-specific documentation
- Contributing guidelines
- Code of conduct
