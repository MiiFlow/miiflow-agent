# Changelog

All notable changes to miiflow-llm will be documented here.

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
