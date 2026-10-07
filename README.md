<p align="center">
  <h1 align="center">miiflow-agent</h1>
  <p align="center">
    <strong>A model-agnostic agent harness for Python — think Claude Agent SDK, for any LLM</strong>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/miiflow-agent/"><img src="https://img.shields.io/pypi/v/miiflow-agent.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/miiflow-agent/"><img src="https://img.shields.io/pypi/pyversions/miiflow-agent.svg" alt="Python versions"></a>
  <a href="https://github.com/Miiflow/miiflow-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

A model call is easy. The hard part of an agent is the **harness** around it: the loop that decides when to call tools and when to answer, running those tools safely and in parallel, handing work to sub-agents, pausing for a human and resuming later, keeping a long run inside the context window, recovering when a provider errors, and streaming all of it to a UI.

**miiflow-agent is that harness, as a Python library.** It plays the same role as the Claude Agent SDK, but it is not tied to one model family: the same agent runs on Claude, GPT, Gemini or six other providers, and you can switch with one line. It runs Miiflow's production agent traffic.

```python
from miiflow_agent import Agent, AgentType, CallbackEventType, LLMClient, scoped_callbacks, tool

@tool("get_weather", "Current weather for a city", writes=False)
def get_weather(city: str) -> str:
    return f"18°C and clear in {city}"

@tool("book_table", "Reserve a restaurant table", writes=True)
def book_table(restaurant: str, time: str) -> str:
    return f"Booked {restaurant} at {time}"

agent = Agent(
    LLMClient.create("anthropic", model="claude-sonnet-5-5"),  # or "openai", "gemini", ...
    agent_type=AgentType.REACT,
    tools=[get_weather, book_table],
)

# A PRE_TOOL_USE hook is the permission gate: blocking a call pauses the run.
async def ask_before_booking(event):
    if event.tool_name == "book_table":
        event.blocked = True
        event.block_reason = "Bookings need a person's OK"

with scoped_callbacks() as hooks:
    hooks.register(CallbackEventType.PRE_TOOL_USE, ask_before_booking)
    # The loop checks the weather first. Before book_table runs, the agent emits
    # TOOL_APPROVAL_NEEDED and pauses (see Human in the loop).
    result = await agent.run("If it's nice out in Lisbon tonight, book Taberna at 8pm")
```

**Demo of an Agentic Run**


https://github.com/user-attachments/assets/0b5c870a-f9b2-4d55-a829-9d7c000be907


## The harness, piece by piece

If you have used the Claude Agent SDK, the concepts carry over:

| Harness concern | Claude Agent SDK | miiflow-agent |
|---|---|---|
| Agent loop | Claude Code's loop | One ReAct loop (`AgentType.REACT`): think → call tools → observe → answer |
| Custom tools | `@tool` + in-process MCP server | `@tool` on any function; the schema comes from type hints |
| Sub-agents | `agents={...}` | `sub_agents=[...]`, with depth, cycle and budget limits |
| Hooks | `PreToolUse`, `PostToolUse`, ... | `PRE_TOOL_USE` (block or rewrite), `POST_TOOL_USE` (transform, opt-in), `POST_CALL`, ... |
| Permissions and plan mode | Permission modes, `plan` | A `PRE_TOOL_USE` hook that blocks a call pauses the run for approval; `enable_plan_mode=True` keeps a run read-only until the plan is approved |
| Pause and resume | Sessions | A JSON-serializable `Checkpoint` resumed with a `ResumeCommand`, plus multiple-choice clarification questions |
| MCP | MCP servers | Client-side servers (stdio, HTTP, SSE) or provider-native MCP on Anthropic and OpenAI |
| Long runs | Automatic compaction | A context engine that sizes the whole request and compacts older history; tool search over large catalogs |
| Streaming | Typed messages | Typed events for thinking, tool calls, observations, answer tokens and pauses, or AG-UI |
| **Models** | **Claude** | **OpenAI, Anthropic, Gemini, Bedrock, OpenRouter, Groq, Mistral, Ollama, xAI** |

There are also real differences. The Claude Agent SDK drives the Claude Code runtime and ships its filesystem, shell and web tools. miiflow-agent runs in your process and ships no tools that act outside the process (its only built-ins are control tools such as plan mode, clarification and tool search): your agent can only do what the tools you give it do. That fits product agents serving many users, where every capability should be one you chose.

What else the harness takes care of:

- **Safe parallel tool execution** — calls declared `parallelizable=True` overlap; writes and approval-gated calls run one at a time, in order
- **Stop means stop** — a cancelled run abandons the calls it has in flight, except those declared `writes=True`: it lets them finish, so the transcript never calls a change that happened "cancelled"
- **Recovery** — context overflow, truncated output and malformed history go through a recovery ladder before the run fails, and a run halted by a safety limit still answers from the work it completed
- **Observability** — OpenInference tracing for Phoenix and Arize AX, plus token-usage and latency callbacks

## Installation

```bash
pip install miiflow-agent

# Optional providers
pip install "miiflow-agent[groq,google,mistral]"

# MCP servers, image handling, tracing
pip install "miiflow-agent[mcp,images,observability]"

# Everything
pip install "miiflow-agent[all]"
```

Requires **Python 3.10–3.12**.

| Extra | Adds |
|---|---|
| `google` | Gemini (`google-genai`) |
| `groq`, `mistral` | Those providers' SDKs |
| `mcp` | Client-side MCP servers |
| `images` | Pillow — **install this if you send images**; they are validated before upload |
| `pdf`, `multimedia` | PDF text extraction (PyMuPDF), plus Pillow |
| `agui` | AG-UI protocol event output |
| `observability` | OpenTelemetry, Phoenix, OpenInference instrumentors |

API keys are read from `<PROVIDER>_API_KEY` (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) or passed as `api_key=`.

## Quick Start

### Agent with Tools

```python
from miiflow_agent import Agent, AgentType, LLMClient, tool

@tool("calculate", "Evaluate mathematical expressions", writes=False, parallelizable=True)
def calculate(expression: str) -> str:
    return str(eval(expression))

@tool("search", "Search for information", writes=False, parallelizable=True)
def search(query: str) -> str:
    return f"Results for '{query}': ..."

agent = Agent(
    LLMClient.create("openai", model="gpt-6.1-sol"),
    agent_type=AgentType.REACT,
    max_iterations=10,
    tools=[calculate, search],
)

result = await agent.run("What is 25 * 4 + the population of France?")
print(result.data)
```

### Context Injection (Pydantic AI Style)

```python
from dataclasses import dataclass
from miiflow_agent import Agent, AgentType, RunContext, tool

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
    deps=UserContext(user_id="alice", permissions=["read"]),
)
```

`Agent` is generic over its deps — annotate the variable (`Agent[UserContext, str]`) and pass the runtime value via `run(deps=...)`. Pass prior turns with `run(..., message_history=[...])`.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Your Application                         │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                              Agent                               │
│  SINGLE_HOP  │  REACT loop: think → call tools → observe → answer │
│  sub-agent dispatch · approvals · clarifications · plan mode     │
│  context engine · tool search · recovery · safety conditions     │
└──────────┬──────────────────────┬──────────────────────┬─────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│    LLMClient     │   │    ToolRegistry    │   │    Callbacks /   │
│                  │   │                    │   │   Observability  │
│ • 9 providers    │   │ • @tool functions  │   │                  │
│ • stream retry   │   │ • MCP (local and   │   │ • usage, latency │
│ • usage metrics  │   │   provider-native) │   │ • pre/post tool  │
│ • prompt caching │   │ • HTTP tools       │   │ • OpenInference  │
│                  │   │ • tool_search      │   │   spans          │
└──────────────────┘   └────────────────────┘   └──────────────────┘
```

## How the Loop Works

miiflow-agent runs a **single, unified ReAct loop**. Each turn the model emits either tool calls (the loop continues) or a text answer (the loop exits). Planning and multi-agent execution happen *inside* this loop — the model plans over several turns and hands work to sub-agents as ordinary tool calls. `AgentType.SINGLE_HOP` (the default: one call, used for chat-only or `json_schema` output) and `AgentType.REACT` are the only modes.

### Tools

`@tool` builds a schema from the function's type hints and docstring. The flags worth knowing:

| Flag | Effect |
|---|---|
| `writes=True/False` | Declares whether the tool changes outside state. Read-only tools stay callable in plan mode. Leaving it `None` means "unclassified", so a host can enforce coverage. |
| `parallelizable=True` | May run concurrently with other calls from the same turn. A mixed batch runs in order, in stages: consecutive parallelizable calls overlap, while writes, approval-gated and control-flow calls run one at a time in their original position. Set `MIIFLOW_READONLY_PARALLEL=1` to let `writes=False` calls overlap too. |
| `require_approval=True` | Declares that the tool needs a human decision and keeps it out of parallel batches. The flag does not pause the run by itself: a `PRE_TOOL_USE` hook enforces it (see below). |
| `always_load=True`, `search_keywords=[...]` | Keep the tool visible, or make it findable, when tool search hides the rest of a large catalog. |
| `strict=True` | Provider-side strict schemas. Opt-in and for small read tools only — strict grammars have per-request size caps. |

A tool reports failure either by raising or by returning `ToolFailure(error=..., output=...)` from `miiflow_agent.core.tools.schemas`. Ordinary return values are never inspected to guess whether the call failed.

When the catalog grows past `tool_search_threshold`, schemas are hidden behind a `tool_search` meta-tool (or the provider's native tool search on Anthropic) so the prompt stays small.

### Sub-agent hand-off

Give an agent sub-agents and its ReAct loop decides when to hand off. The dispatch is a tool call routed through a lifecycle that enforces depth, cycle and budget limits and bubbles child events up to the parent stream.

```python
from miiflow_agent import Agent, AgentType, LLMClient
from miiflow_agent.core.config import AgentConfig
from miiflow_agent.core.react import ConfiguredSubAgent, DynamicSubAgentConfig

client = LLMClient.create("anthropic", model="claude-sonnet-5-5")

researcher_config = DynamicSubAgentConfig(
    name="researcher",
    description="Finds and summarizes sources on a topic",
    system_prompt="You are a research specialist. Cite your sources.",
    max_steps=8,
)
researcher = ConfiguredSubAgent(
    researcher_config,
    # Its own LLMClient: an Agent's tools live on its client's registry,
    # so agents built from one client share a single tool surface.
    Agent(LLMClient.create("anthropic", model="claude-sonnet-5-5"),
          agent_type=AgentType.REACT,
          system_prompt=researcher_config.system_prompt, tools=[search]),
    when_to_use="Any question that needs outside information",
)

lead = Agent(config=AgentConfig(
    client=client,
    agent_type=AgentType.REACT,  # AgentConfig defaults to SINGLE_HOP
    system_prompt="Coordinate research and writing.",
    sub_agents=[researcher],
))

result = await lead.run("Compare the three most popular Python web frameworks")
```

> When constructing with `config=`, pass everything through `AgentConfig` — mixing `config=` with sibling kwargs like `system_prompt=` raises a `ValueError`. `ConfiguredSubAgent` also takes `forward_transcript_window`, `clarification_policy` and `auto_approve_child_tools`. See [`examples/subagents.py`](examples/subagents.py) for registry-based dispatch with `SubAgentRegistry` and `make_registry_dispatcher_tool`.

### Human in the loop

A run can stop and wait for a person without losing its place:

- **Tool approval** — when a `PRE_TOOL_USE` hook sets `event.blocked = True`, the call does not run: the stream emits `TOOL_APPROVAL_NEEDED` and the run pauses. A hook can also approve with edits by setting `event.inputs_override`.
- **Clarification** — the agent can ask the user one or more (multiple-choice) questions; the stream emits `CLARIFICATION_NEEDED`.
- **Plan mode** — with `enable_plan_mode=True` the model can call `enter_plan_mode`, after which only read-only tools run until the user approves the plan it submits via `exit_plan_mode` (`PLAN_APPROVAL_NEEDED`).

The paused state is captured in a `Checkpoint` — pending interrupts, approved actions, answers the user already gave (`EstablishedFact`), and a ledger of sub-agent dispatches — with `to_dict()` / `from_dict()` so the host can store it as JSON and resume the run later with a `ResumeCommand`. Answered clarifications are not asked again.

### MCP

Two ways to use [Model Context Protocol](https://modelcontextprotocol.io) servers (requires the `mcp` extra):

```python
from miiflow_agent.core.tools.mcp import MCPServerConfig, MCPToolManager, NativeMCPServerConfig

# 1. Client-side: this process connects (stdio, streamable_http or sse) and executes the tools.
manager = MCPToolManager()
manager.add_server(MCPServerConfig(
    name="filesystem",
    transport="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
))
async with manager:
    agent.tool_registry.register_mcp_manager(manager)
    result = await agent.run("List the files in /tmp")

# 2. Provider-native: Anthropic or OpenAI connects to the server and runs the tools.
agent.tool_registry.register_native_mcp_server(NativeMCPServerConfig(
    name="docs",
    url="https://mcp.example.com/mcp",
    authorization_token="...",
))
```

Native servers can be registered `lazy=True` and activated only when needed. On Anthropic a whole server's tools can be deferred behind tool search, and individual tools switched off with `disabled_tools`.

### Context management

Before every LLM call a context engine sizes the **whole** request — system prompt, tool schemas and messages — and compacts older history into a handoff note when the budget is tight. The per-tier breakdown is emitted as a `CONTEXT_BREAKDOWN` event. It is on by default (`context_compression=True`); set `max_context_tokens` to cap the budget, or plug in your own engine with `register_engine(name, factory)` and `context_engine=name`.

## Event Streaming

Stream typed events while an agent runs:

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
        case ReActEventType.FINAL_ANSWER_CHUNK:
            print(event.data.get("delta", ""), end="")
        case ReActEventType.FINAL_ANSWER:
            print(f"\nAnswer: {event.data['answer']}")
```

Other events cover tool execution (`ACTION_EXECUTING`), sub-agent progress (`SUBAGENT_DISPATCH`), pauses (`TOOL_APPROVAL_NEEDED`, `CLARIFICATION_NEEDED`, `PLAN_APPROVAL_NEEDED`), rich tool results (`VISUALIZATION`, `MEDIA`, `ARTIFACT`), and run health (`PROGRESS`, `LLM_TRUNCATED`, `STOP_CONDITION`, `ERROR`). Answer tokens stream as they are generated; if text that looked like the answer turns out to be preamble before a tool call, an `ANSWER_RETRACTED` event tells the UI to demote it to thinking.

Pass `event_format="agui"` (with `thread_id` and `message_id`, and the `agui` extra) to receive [AG-UI](https://docs.ag-ui.com) protocol events instead.

## Hooks and Callbacks

Hook LLM calls and tool execution — for billing, approvals, auditing or output enrichment:

```python
from miiflow_agent import CallbackContext, CallbackEventType, callback_context, on_post_call, scoped_callbacks

@on_post_call
async def record_usage(event):
    print(event.provider, event.model, event.tokens, event.latency_ms, event.ttft_ms)

# Attribute every call in this block to an org / thread
with callback_context(CallbackContext(organization_id="org_123", thread_id="t_1")):
    await agent.run(prompt)

# Register per-run callbacks without touching the global registry
async def approval_gate(event):
    if event.tool_name in DANGEROUS_TOOLS:
        event.blocked = True
        event.block_reason = "Needs approval"

with scoped_callbacks() as cbs:
    cbs.register(CallbackEventType.PRE_TOOL_USE, approval_gate)
    await agent.run(prompt)
```

Event types: `POST_CALL`, `ON_ERROR`, `AGENT_RUN_START`, `AGENT_RUN_END`, `PRE_TOOL_USE` (can block or rewrite inputs), `POST_TOOL_USE` (can transform output; opt-in with `MIIFLOW_EMIT_POST_TOOL_USE=1`) and `TOOL_EXECUTED` (the final outcome). Inside async generators use `callback_context_stream` / `scoped_callbacks_stream`, which stay correct when the stream is advanced from another task.

## Observability

Built-in OpenInference tracing for [Phoenix](https://phoenix.arize.com/) and Arize AX. Requires the `observability` extra.

```python
# Local / self-hosted Phoenix
from miiflow_agent.core.observability import ObservabilityConfig, enable_phoenix_tracing

enable_phoenix_tracing(ObservabilityConfig.for_local())  # or ObservabilityConfig.from_env()

# Arize AX — credentials are the switch, no flag needed:
#   export ARIZE_SPACE_ID=...  ARIZE_API_KEY=...  [ARIZE_PROJECT_NAME=my-app]
from miiflow_agent.core.observability import setup_opentelemetry_tracing
setup_opentelemetry_tracing()

# All LLM calls and tool calls are now traced. Wrap a run in an agent span:
from miiflow_agent.core.observability import agent_span

with agent_span("my-run", input_value=prompt, session_id=thread_id):
    result = await agent.run(prompt)
```

`ObservabilityConfig.from_env()` enables Phoenix only when `PHOENIX_ENABLED=true`. `traced_stream` keeps a span correct around an async generator.

## Error Handling

```python
from miiflow_agent import (
    MiiflowLLMError,      # Base
    ProviderError,        # Provider-specific
    RateLimitError,       # Rate limited
    AuthenticationError,  # Invalid API key
    TimeoutError,         # Request timeout
    ToolError,            # Tool execution failed
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

`RateLimitError.retry_after` is populated from the provider's `Retry-After` header. `ModelError` and `ParsingError` are also exported. Inside an agent run, recoverable provider errors (context overflow, truncated output, malformed history) go through a recovery ladder before the run fails, and a run halted by a safety limit still answers from the work it completed.

## Models and Providers

The harness talks to models through `LLMClient`, which gives every provider the same interface. You can also use it directly, without an agent.

### Chat

```python
from miiflow_agent import LLMClient, Message

client = LLMClient.create("openai", model="gpt-6-luna")
response = client.chat([
    Message.system("You are a helpful assistant."),
    Message.user("What is Python?"),
])
print(response.message.content)
```

> `client.chat()` is a sync convenience that calls `asyncio.run()` internally — inside an already-running event loop (Jupyter, an async app), use `await client.achat(...)` instead.

### Streaming tokens

```python
async for chunk in client.astream_chat([Message.user("Tell me a story")]):
    print(chunk.delta, end="", flush=True)
```

Streams retry transparently until the first chunk arrives (`MIIFLOW_STREAM_RETRY_ATTEMPTS`, default 3) and fail on a stalled connection after `MIIFLOW_STREAM_INACTIVITY_TIMEOUT` seconds (default 300).

### Supported Providers

| Provider | `LLMClient.create` key | Streaming | Tool Calling | Vision | Status |
|----------|------------------------|:---------:|:------------:|:------:|:------:|
| **OpenAI** | `openai` | ✅ | ✅ | ✅ | **Stable** |
| **Anthropic** | `anthropic` | ✅ | ✅ | ✅ | **Stable** |
| **Google Gemini** | `gemini` | ✅ | ✅ | ✅ | **Stable** |
| Amazon Bedrock | `bedrock` | ✅ | ✅ | ✅ | Beta |
| OpenRouter | `openrouter` | ✅ | ✅ | ✅ | Beta |
| Groq | `groq` | ✅ | ✅ | - | Beta |
| Mistral | `mistral` | ✅ | ✅ | - | Beta |
| Ollama | `ollama` | ✅ | ✅ | - | Beta |
| xAI | `xai` | ✅ | ✅ | - | Beta |

> Keys are lowercase and exact — `gemini`, not `google` (that is only the pip extra's name). Bedrock takes `aws_access_key_id`, `aws_secret_access_key` and `region_name` as keyword arguments instead of an API key.

The model catalog in `miiflow_agent/models/` records each model's context window, output cap, prices (including cache-read and cache-write rates), supported parameters and reasoning-effort levels. It is re-audited against the providers' docs regularly; see [CHANGELOG.md](CHANGELOG.md) for what is current, legacy or deprecated.

## Documentation

- [Examples](examples/) — runnable scripts for chat, streaming, tools, context injection, sub-agents and tracing
- [CHANGELOG](CHANGELOG.md) — every release, with migration notes for breaking changes
- [Provider Guide](docs/providers.md) — Provider-specific configuration
- [Observability](docs/observability.md) — Tracing and debugging
- [Quickstart](docs/quickstart.md), [API Reference](docs/api.md), [Tool Tutorial](docs/tutorial-tools.md), [Agent Tutorial](docs/tutorial-agents.md)

## Contributing

```bash
git clone https://github.com/Miiflow/miiflow-agent.git
cd miiflow-agent
poetry install --with dev --all-extras

poetry run pytest tests/
poetry run black miiflow_agent/ tests/
poetry run isort miiflow_agent/ tests/
```

Bug reports, provider additions, docs fixes and tests are all welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the provider guide and detailed guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
