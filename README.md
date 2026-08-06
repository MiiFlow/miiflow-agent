<p align="center">
  <h1 align="center">miiflow-agent</h1>
  <p align="center">
    <strong>A lightweight, unified Python SDK for LLM providers with built-in agentic patterns</strong>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/miiflow-agent/"><img src="https://img.shields.io/pypi/v/miiflow-agent.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/miiflow-agent/"><img src="https://img.shields.io/pypi/pyversions/miiflow-agent.svg" alt="Python versions"></a>
  <a href="https://github.com/Miiflow/miiflow-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

**miiflow-agent** gives you a unified API across LLM providers, with built-in support for ReAct agents, tool calling, and streaming — all in ~39K lines of focused code.

```python
from miiflow_agent import LLMClient, Message

# Same interface for any provider
client = LLMClient.create("openai", model="gpt-5.6-luna")
response = client.chat([Message.user("Hello!")])

# Switch providers with one line
client = LLMClient.create("anthropic", model="claude-sonnet-5")
```

**Demo of an Agentic Run**


https://github.com/user-attachments/assets/0b5c870a-f9b2-4d55-a829-9d7c000be907


## Why miiflow-agent?

| | miiflow-agent | LangChain | LiteLLM |
|---|:---:|:---:|:---:|
| **Codebase size** | ~39K lines | ~500K lines | ~50K lines |
| **Dependencies** | 9 core | 50+ | 20+ |
| **Built-in agents** | ReAct + sub-agent hand-off | Requires setup | None |
| **Tool system** | @tool decorator | Chains | None |
| **Learning curve** | Hours | Weeks | Hours |
| **Type safety** | Full generics | Partial | Basic |

### The LangChain Problem

LangChain is powerful but complex. For production apps, you often fight its abstractions more than use them. miiflow-agent gives you **what you actually need**:

- **Unified provider interface** — swap OpenAI → Claude → Gemini with one line
- **Agentic patterns built-in** — a single ReAct loop with emergent planning and multi-agent hand-off, not bolted on
- **Simple tool system** — decorate any function with `@tool`
- **Real streaming** — event-based, not just token callbacks
- **Type-safe** — full generics, proper error types

### The LiteLLM Gap

LiteLLM unifies provider APIs but stops there. miiflow-agent adds:

- **ReAct agents** with multi-hop reasoning
- **Sub-agent hand-off** for complex multi-step tasks
- **Tool calling** with automatic schema generation
- **Context injection** (Pydantic AI compatible)

## Installation

```bash
pip install miiflow-agent

# With optional providers
pip install miiflow-agent[groq,google]

# Tracing support (Phoenix / Arize AX)
pip install miiflow-agent[observability]

# Everything
pip install miiflow-agent[all]
```

Requires **Python 3.10+** (the 1.15.0 release dropped 3.9, forced by the `openai` 2.x floor — stay on `1.14.0` if you're pinned to 3.9).

## Quick Start

### Basic Chat

```python
from miiflow_agent import LLMClient, Message

client = LLMClient.create("openai", model="gpt-5.6-luna")
response = client.chat([
    Message.system("You are a helpful assistant."),
    Message.user("What is Python?")
])
print(response.message.content)
```

> `client.chat()` is a sync convenience that calls `asyncio.run()` internally — inside an already-running event loop (Jupyter, an async app), use `await client.achat(...)` instead.

### Streaming

```python
async for chunk in client.astream_chat([Message.user("Tell me a story")]):
    print(chunk.delta, end="", flush=True)
```

### ReAct Agent with Tools

```python
from miiflow_agent import Agent, AgentType, LLMClient, tool

@tool("calculate", "Evaluate mathematical expressions")
def calculate(expression: str) -> str:
    return str(eval(expression))

@tool("search", "Search for information")
def search(query: str) -> str:
    return f"Results for '{query}': ..."

# Create agent
agent = Agent(
    LLMClient.create("openai", model="gpt-5.6-sol"),
    agent_type=AgentType.REACT,
    max_iterations=10
)
agent.add_tool(calculate)
agent.add_tool(search)

# Run with automatic reasoning
result = await agent.run("What is 25 * 4 + the population of France?")
print(result.data)  # Agent reasons, calls tools, synthesizes answer
```

### Context Injection (Pydantic AI Style)

```python
from dataclasses import dataclass
from miiflow_agent import Agent, RunContext, tool

@dataclass
class UserContext:
    user_id: str
    permissions: list[str]

@tool("get_user_data")
def get_user_data(ctx: RunContext[UserContext], field: str) -> str:
    """Fetch data for the current user."""
    if "read" not in ctx.deps.permissions:
        return "Permission denied"
    return f"User {ctx.deps.user_id} data for {field}"

agent: Agent[UserContext, str] = Agent(client, agent_type=AgentType.REACT)
agent.add_tool(get_user_data)

result = await agent.run(
    "What's my account status?",
    deps=UserContext(user_id="alice", permissions=["read"])
)
```

`Agent` is generic over its deps — annotate the variable (`Agent[UserContext, str]`) and pass the runtime value via `run(deps=...)`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Application                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                          LLMClient                               │
│  • Unified interface for all providers                          │
│  • Automatic tool schema generation                             │
│  • Metrics collection & observability                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    Agent      │   │   Provider    │   │    Tools      │
│               │   │   Clients     │   │               │
│ • SINGLE_HOP  │   │               │   │ • @tool       │
│ • REACT       │   │ • OpenAI      │   │ • FunctionTool│
│ • SubAgents   │   │ • Anthropic   │   │ • HTTPTool    │
│               │   │ • Gemini      │   │ • Registry    │
│ ┌───────────┐ │   │ • More...     │   │               │
│ │Orchestrator│ │   │               │   │ ┌───────────┐ │
│ │ • plan    │ │   │               │   │ │ Schemas   │ │
│ │ • handoff │ │   │               │   │ │ • Auto-gen│ │
│ └───────────┘ │   │               │   │ │ • Validate│ │
└───────────────┘   │               │   │ └───────────┘ │
                    └───────────────┘   └───────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │   Message     │
                    │   Unified     │
                    │   Format      │
                    │               │
                    │ • Text        │
                    │ • Images      │
                    │ • Tool calls  │
                    └───────────────┘
```

## Supported Providers

| Provider | Streaming | Tool Calling | Vision | Status |
|----------|:---------:|:------------:|:------:|:------:|
| **OpenAI** | ✅ | ✅ | ✅ | **Stable** |
| **Anthropic** | ✅ | ✅ | ✅ | **Stable** |
| **Google Gemini** | ✅ | ✅ | ✅ | **Stable** |
| Groq | ✅ | ✅ | - | Beta |
| Amazon Bedrock | ✅ | ✅ | ✅ | Beta |
| Mistral | ✅ | ✅ | - | Beta |
| OpenRouter | ✅ | ✅ | ✅ | Beta |
| Ollama | ✅ | ✅ | - | Beta |
| xAI | ✅ | ✅ | - | Beta |

> **Stable** providers are production-tested with full feature support. **Beta** providers are functional but may have edge cases.

> Provider keys for `LLMClient.create()`: `openai`, `anthropic`, `gemini` (not `google` — that's only the pip extra's name), `groq`, `bedrock`, `mistral`, `openrouter`, `ollama`, `xai`.

## Agentic Patterns

miiflow-agent runs a **single, unified ReAct loop**. Each turn the model emits
exactly one of: tool calls (the loop continues) or a text answer (the loop exits).
Planning and parallel multi-agent execution are *emergent behaviors* inside this
one loop — there are no separate "plan & execute" or "multi-agent" orchestrators to
choose between. Complex tasks are handled by letting the model plan over multiple
turns and dispatch work to sub-agents as normal tool calls.

> **Migrating from a pre-1.8 release?** The standalone `PlanAndExecuteOrchestrator`,
> `MultiAgentOrchestrator`, and the `AgentType.PLAN_AND_EXECUTE` / `PARALLEL_PLAN` /
> `MULTI_AGENT` enum values have been removed. Express the same outcomes with a single
> `Agent` plus `sub_agents=[SubAgent(...)]` — see [Sub-agent hand-off](#sub-agent-hand-off)
> below. `AgentType.SINGLE_HOP` (the default) and `AgentType.REACT` are the only modes.

### ReAct (Reasoning + Acting)

The agent thinks step-by-step, deciding when to use tools, and plans across turns
for multi-step tasks:

```python
agent = Agent(client, agent_type=AgentType.REACT)

# Agent internally:
# Thought: I need to search for this information
# Action: search("topic")
# Observation: Results...
# Thought: Now I can answer
# Final Answer: ...
```

### Sub-agent hand-off

For complex work that benefits from specialization, give the agent one or more
sub-agents. The parent's ReAct loop decides *when* to hand off, and the dispatch
happens as an ordinary tool call — no separate orchestrator. Sub-agents are wired
in through `AgentConfig`; the framework synthesizes a `dispatch_assistant`-shaped
tool that routes calls through the dispatch lifecycle (depth, cycle, budget, and
event bubbling):

```python
from miiflow_agent import Agent
from miiflow_agent.core.config import AgentConfig

# `sub_agents` holds objects implementing the SubAgent protocol — each one a
# parent-side "edge" to a specialist child, carrying per-edge policy such as
# `when_to_use`, `handoff_schema`, and `clarification_policy`.
lead = Agent(config=AgentConfig(
    client=client,
    agent_type=AgentType.REACT,  # AgentConfig defaults to SINGLE_HOP
    system_prompt="Coordinate research and writing.",
    sub_agents=[researcher_subagent, writer_subagent],
))

result = await lead.run(
    "Research Python web frameworks, compare them, and write a summary"
)
# The ReAct loop plans, dispatches to each specialist, and synthesizes the result.
```

> When constructing with `config=`, pass everything through `AgentConfig` — mixing
> `config=` with sibling kwargs like `system_prompt=` raises a `ValueError`.
> See `miiflow_agent/core/subagent.py` for the `SubAgent` protocol and its
> per-edge policy fields, and [`examples/subagents.py`](examples/subagents.py)
> for reusable sub-agent config templates and a registry-based dispatch example.

## Event Streaming

Stream real-time events during agent execution:

```python
from miiflow_agent import Agent, AgentType, RunContext
from miiflow_agent.core.react import ReActEventType

agent = Agent(client, agent_type=AgentType.REACT)
context = RunContext(deps=None)

async for event in agent.stream("What is 2+2?", context):
    match event.event_type:
        case ReActEventType.THINKING_CHUNK:
            print(event.data.get("delta", ""), end="")
        case ReActEventType.ACTION_PLANNED:
            print(f"\nCalling: {event.data['action']}")
        case ReActEventType.OBSERVATION:
            print(f"Result: {event.data['observation']}")
        case ReActEventType.FINAL_ANSWER:
            print(f"\nAnswer: {event.data['answer']}")
```

## Observability

Built-in OpenInference tracing for [Phoenix](https://phoenix.arize.com/) and Arize AX. Requires the `observability` extra:

```bash
pip install "miiflow-agent[observability]"
```

```python
# Local / self-hosted Phoenix
from miiflow_agent.core.observability import ObservabilityConfig, enable_phoenix_tracing

enable_phoenix_tracing(ObservabilityConfig.for_local())  # or ObservabilityConfig.from_env()

# Arize AX — credentials are the switch, no flag needed:
#   export ARIZE_SPACE_ID=...  ARIZE_API_KEY=...  [ARIZE_PROJECT_NAME=my-app]
from miiflow_agent.core.observability import setup_opentelemetry_tracing
setup_opentelemetry_tracing()

# All LLM calls are now traced. Wrap your own units of work in an agent span:
from miiflow_agent.core.observability import agent_span

with agent_span("my-run", input_value=prompt, session_id=thread_id):
    result = await agent.run(prompt)
```

The legacy `setup_tracing(phoenix_endpoint=...)` helper still exists but is gated on the `PHOENIX_ENABLED=true` env var — pass `force=True` to bypass the gate.

## Error Handling

Comprehensive error hierarchy:

```python
from miiflow_agent import (
    MiiflowLLMError,    # Base
    ProviderError,      # Provider-specific
    RateLimitError,     # Rate limited
    AuthenticationError, # Invalid API key
    TimeoutError,       # Request timeout
    ToolError,          # Tool execution failed
)

try:
    response = client.chat(messages)
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except AuthenticationError:
    print("Check your API key")
except ProviderError as e:
    print(f"{e.provider} error: {e.message}")
```

`RateLimitError.retry_after` is populated automatically from provider `Retry-After` headers. `ModelError` and `ParsingError` are also exported.

## Documentation

- [Quickstart Guide](docs/quickstart.md) — Get started in 5 minutes
- [API Reference](docs/api.md) — Complete API documentation
- [Tool Tutorial](docs/tutorial-tools.md) — Build custom tools
- [Agent Tutorial](docs/tutorial-agents.md) — Build ReAct agents
- [Provider Guide](docs/providers.md) — Provider-specific configuration
- [Observability](docs/observability.md) — Tracing and debugging

## Contributing

We welcome contributions! Here's how to get started:

```bash
# Clone and install
git clone https://github.com/Miiflow/miiflow-agent.git
cd miiflow-agent
pip install -e ".[all]"

# Run tests
pytest tests/

# Format code
black miiflow_agent/ tests/
isort miiflow_agent/ tests/
```

### Ways to Contribute
- **Report bugs** — Open an issue with reproduction steps
- **Request features** — Describe your use case
- **Add providers** — See [CONTRIBUTING.md](CONTRIBUTING.md) for the provider guide
- **Improve docs** — Fix typos, add examples
- **Write tests** — Increase coverage

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
