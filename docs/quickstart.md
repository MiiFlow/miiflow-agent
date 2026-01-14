# Quickstart

Get started in 5 minutes.

## Install

```bash
pip install miiflow-agent
```

Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
```

## First Chat

The basic pattern: create a client, send messages, get responses.

```python
from miiflow_agent import LLMClient
from miiflow_agent.core import Message

client = LLMClient.create("openai", model="gpt-4o-mini")
response = client.chat([Message.user("What is Rust?")])
print(response.message.content)
```

## Streaming

Same interface, get chunks instead of full response:

```python
for chunk in client.stream_chat([Message.user("Explain async/await")]):
    print(chunk.delta, end="", flush=True)
```

## Switch Providers

Change one line, everything else stays the same:

```python
# OpenAI
client = LLMClient.create("openai", model="gpt-4o-mini")

# Claude
client = LLMClient.create("anthropic", model="claude-3-5-sonnet-20241022")

# Groq (fast inference)
client = LLMClient.create("groq", model="llama-3.3-70b-versatile")

# Same interface for all
response = client.chat([Message.user("Hello")])
```

## Async

```python
import asyncio

async def main():
    client = LLMClient.create("openai", model="gpt-4o-mini")

    response = await client.achat([Message.user("Hi")])
    print(response.message.content)

    async for chunk in client.astream_chat([Message.user("Count to 10")]):
        print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

## Next

- [Tool Tutorial](tutorial-tools.md) - Build tools step-by-step
- [Agent Tutorial](tutorial-agents.md) - Build ReAct agents
- [API Reference](api.md) - Complete reference
