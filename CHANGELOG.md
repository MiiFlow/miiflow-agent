# Changelog

All notable changes to miiflow-agent will be documented here.

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
