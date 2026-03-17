# Changelog

All notable changes to miiflow-agent will be documented here.

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
