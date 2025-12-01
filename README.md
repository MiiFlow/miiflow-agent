<p align="center">
  <h1 align="center">miiflow-llm</h1>
  <p align="center">
    <strong>A lightweight, unified Python SDK for LLM providers with built-in agentic patterns</strong>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/miiflow-llm/"><img src="https://img.shields.io/pypi/v/miiflow-llm.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/miiflow-llm/"><img src="https://img.shields.io/pypi/pyversions/miiflow-llm.svg" alt="Python versions"></a>
  <a href="https://github.com/Miiflow/miiflow-llm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

**miiflow-llm** gives you a unified API across LLM providers, with built-in support for ReAct agents, tool calling, and streaming — all in ~15K lines of focused code.

```python
from miiflow_llm import LLMClient, Message

# Same interface for any provider
client = LLMClient.create("openai", model="gpt-4o-mini")
response = client.chat([Message.user("Hello!")])

# Switch providers with one line
client = LLMClient.create("anthropic", model="claude-sonnet-4-20250514")
```

## Why miiflow-llm?

| | miiflow-llm | LangChain | LiteLLM |
|---|:---:|:---:|:---:|
| **Codebase size** | ~15K lines | ~500K lines | ~50K lines |
| **Dependencies** | 8 core | 50+ | 20+ |
| **Built-in agents** | ReAct, Plan&Execute | Requires setup | None |
| **Tool system** | @tool decorator | Chains | None |
| **Learning curve** | Hours | Weeks | Hours |
| **Type safety** | Full generics | Partial | Basic |

### The LangChain Problem

LangChain is powerful but complex. For production apps, you often fight its abstractions more than use them. miiflow-llm gives you **what you actually need**:

- **Unified provider interface** — swap OpenAI → Claude → Gemini with one line
- **Agentic patterns built-in** — ReAct and Plan & Execute, not bolted on
- **Simple tool system** — decorate any function with `@tool`
- **Real streaming** — event-based, not just token callbacks
- **Type-safe** — full generics, proper error types

### The LiteLLM Gap

LiteLLM unifies provider APIs but stops there. miiflow-llm adds:

- **ReAct agents** with multi-hop reasoning
- **Plan & Execute** for complex multi-step tasks
- **Tool calling** with automatic schema generation
- **Context injection** (Pydantic AI compatible)

## Installation

```bash
pip install miiflow-llm

# With optional providers
pip install miiflow-llm[groq,google]

# Everything
pip install miiflow-llm[all]
```

## Quick Start

### Basic Chat

```python
from miiflow_llm import LLMClient, Message

client = LLMClient.create("openai", model="gpt-4o-mini")
response = client.chat([
    Message.system("You are a helpful assistant."),
    Message.user("What is Python?")
])
print(response.message.content)
```

### Streaming

```python
async for chunk in client.astream_chat([Message.user("Tell me a story")]):
    print(chunk.delta, end="", flush=True)
```

### ReAct Agent with Tools

```python
from miiflow_llm import Agent, AgentType, LLMClient, tool

@tool("calculate", "Evaluate mathematical expressions")
def calculate(expression: str) -> str:
    return str(eval(expression))

@tool("search", "Search for information")
def search(query: str) -> str:
    return f"Results for '{query}': ..."

# Create agent
agent = Agent(
    LLMClient.create("openai", model="gpt-4o"),
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
from miiflow_llm import Agent, RunContext, tool

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

agent = Agent(client, deps_type=UserContext)
agent.add_tool(get_user_data)

result = await agent.run(
    "What's my account status?",
    deps=UserContext(user_id="alice", permissions=["read"])
)
```

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
│ • PLAN_EXEC   │   │ • Anthropic   │   │ • HTTPTool    │
│               │   │ • Gemini      │   │ • Registry    │
│ ┌───────────┐ │   │ • More...     │   │               │
│ │Orchestrator│ │   │               │   │ ┌───────────┐ │
│ │ • ReAct   │ │   │               │   │ │ Schemas   │ │
│ │ • Plan&Ex │ │   │               │   │ │ • Auto-gen│ │
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

## Agentic Patterns

### ReAct vs Plan & Execute

| | ReAct | Plan & Execute |
|---|---|---|
| **Best for** | Simple queries, quick lookups | Complex multi-step tasks |
| **Planning** | Implicit (step-by-step) | Explicit (plan first, then execute) |
| **Adaptability** | High (adjusts each step) | Medium (can replan if needed) |
| **Token usage** | Lower | Higher (planning overhead) |
| **Latency** | Faster for simple tasks | Better for complex tasks |

**Use ReAct when:**
- The task can be solved in 1-3 tool calls
- You need quick, reactive responses
- The path to solution isn't known upfront

**Use Plan & Execute when:**
- The task has multiple dependent steps
- You need to coordinate several tools
- The task benefits from upfront planning (research, analysis, multi-part creation)

### ReAct (Reasoning + Acting)

The agent thinks step-by-step, deciding when to use tools:

```python
agent = Agent(client, agent_type=AgentType.REACT)

# Agent internally:
# Thought: I need to search for this information
# Action: search("topic")
# Observation: Results...
# Thought: Now I can answer
# Final Answer: ...
```

### Plan & Execute

For complex tasks, the agent creates a plan first, then executes each step:

```python
from miiflow_llm.core.react import ReActFactory

orchestrator = ReActFactory.create_plan_execute_orchestrator(
    agent=agent,
    max_replans=2  # Allow 2 re-planning attempts if steps fail
)

result = await orchestrator.execute(
    "Research Python web frameworks, compare them, and write a summary"
)

# Agent internally:
# Plan:
#   1. Search for popular Python web frameworks
#   2. Gather key features of each framework
#   3. Compare performance and use cases
#   4. Write summary with recommendations
#
# Execute Step 1: search("Python web frameworks 2024")
# Execute Step 2: search("Django vs FastAPI vs Flask features")
# ...
# Final Answer: [comprehensive summary]
```

## Event Streaming

Stream real-time events during agent execution:

```python
from miiflow_llm import Agent, AgentType, RunContext
from miiflow_llm.core.react import ReActEventType

agent = Agent(client, agent_type=AgentType.REACT)
context = RunContext(deps=None)

async for event in agent.stream_react("What is 2+2?", context):
    match event.event_type:
        case ReActEventType.THINKING_CHUNK:
            print(event.data.get("delta", ""), end="")
        case ReActEventType.TOOL_START:
            print(f"\nCalling: {event.data['tool_name']}")
        case ReActEventType.OBSERVATION:
            print(f"Result: {event.data['observation']}")
        case ReActEventType.FINAL_ANSWER:
            print(f"\nAnswer: {event.data['answer']}")
```

## Observability

Built-in Phoenix tracing support:

```python
from miiflow_llm.core import setup_tracing

setup_tracing(phoenix_endpoint="http://localhost:6006")

# All LLM calls are now traced
```

## Error Handling

Comprehensive error hierarchy:

```python
from miiflow_llm import (
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

## Error Handling

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
git clone https://github.com/Miiflow/miiflow-llm.git
cd miiflow-llm
pip install -e ".[all]"

# Run tests
pytest tests/

# Format code
black miiflow_llm/ tests/
isort miiflow_llm/ tests/
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
