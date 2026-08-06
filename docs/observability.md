# Observability

Track LLM calls and agent execution with [Phoenix](https://phoenix.arize.com/).

## Quick Setup

```bash
pip install "miiflow-agent[observability]"
```

```python
from miiflow_agent.core.observability import ObservabilityConfig, enable_phoenix_tracing

enable_phoenix_tracing(ObservabilityConfig.for_local())

# Use normally - all calls are traced
from miiflow_agent import LLMClient, Message
client = LLMClient.create("openai", model="gpt-5.6-luna")
response = await client.achat([Message.user("Hello")])

# View traces at http://localhost:6006
```

## Configuration

### Environment Variables

```bash
# Phoenix (flag-gated)
export PHOENIX_ENABLED=true
export PHOENIX_ENDPOINT=http://localhost:6006   # or PHOENIX_COLLECTOR_ENDPOINT
export PHOENIX_PROJECT_NAME=my-app              # default: miiflow-agent

# Arize AX (credential-gated — the credentials ARE the switch)
export ARIZE_SPACE_ID=...
export ARIZE_API_KEY=...
export ARIZE_PROJECT_NAME=my-app                # default: miiflow-agent
```

### Programmatic Setup

```python
from miiflow_agent.core.observability import (
    ObservabilityConfig,
    enable_phoenix_tracing,
    setup_opentelemetry_tracing,
)

# From environment (recommended)
enable_phoenix_tracing(ObservabilityConfig.from_env())

# Or point at a specific Phoenix instance
config = ObservabilityConfig.from_env()
config.phoenix_enabled = True
config.phoenix_endpoint = "https://phoenix.yourcompany.com"
enable_phoenix_tracing(config)

# Arize AX (reads ARIZE_* env vars)
setup_opentelemetry_tracing()
```

### Wrapping your own spans

```python
from miiflow_agent.core.observability import agent_span

with agent_span("my-run", input_value=prompt, session_id=thread_id):
    result = await agent.run(prompt)
```

## What Gets Traced

- LLM requests: model, tokens, latency, content
- Agent execution: step-by-step reasoning
- Tool calls: inputs and outputs
- Streaming: real-time chunks

## Phoenix Dashboard

Open http://localhost:6006 to view:

**Traces Tab:**
- Request/response for each LLM call
- Token counts and latency
- Agent reasoning steps
- Tool executions

**Timeline View:**
- See when each step happened
- Identify slow operations
- Track token usage over time

**Search:**
- Filter by provider, model, or time range
- Search trace content
- Find specific agent runs

## Example

```python
from miiflow_agent import LLMClient, Agent, Message
from miiflow_agent.core.tools import tool
import asyncio

@tool("calculate", "Do math")
def calculate(expr: str) -> str:
    return str(eval(expr))

async def main():
    client = LLMClient.create("openai", model="gpt-5.6-luna")
    agent = Agent(client=client)
    agent.add_tool(calculate)

    result = await agent.run("What is 25 * 4?")
    print(result.data)

asyncio.run(main())
# Check Phoenix dashboard for full trace
```

## Troubleshooting

### Phoenix Not Starting

**Check installation:**
```bash
pip install "miiflow-agent[observability]"
# Verify Phoenix installed
python -c "import phoenix; print('Phoenix OK')"
```

**Launch a local Phoenix session (development only):**
```python
from miiflow_agent.core.observability import ObservabilityConfig, enable_phoenix_tracing

enable_phoenix_tracing(ObservabilityConfig.for_local(), launch_local=True)
```

### No Traces Appearing

**1. Check Phoenix is running:**
- Visit http://localhost:6006
- Should see Phoenix UI

**2. Verify instrumentation:**
```python
from miiflow_agent.core.observability.auto_instrumentation import check_instrumentation_status

status = check_instrumentation_status()
for provider, info in status.items():
    print(f"{provider}: {info}")
```

**3. Check dependencies:**
```bash
# Install OpenInference instrumentations
pip install openinference-instrumentation-openai
pip install openinference-instrumentation-anthropic
```

### Common Errors

**"OpenInference instrumentation not available"**
```bash
pip install openinference-instrumentation-openai openinference-instrumentation-anthropic
```

**"Phoenix session setup failed"**
```bash
pip install arize-phoenix
```

**Traces delayed or missing**
- Verify Phoenix endpoint is accessible
- Check firewall/network settings
