# API Reference

## LLMClient

### `LLMClient.create(provider, model, **kwargs)`

Create a client for any provider.

```python
from miiflow_agent import LLMClient

# Basic
client = LLMClient.create("openai", model="gpt-4o-mini")

# With options
client = LLMClient.create(
    "anthropic",
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=4000
)
```

**Parameters:**
- `provider` (str): `"openai"`, `"anthropic"`, `"gemini"`, `"groq"`, `"openrouter"`, `"mistral"`, `"xai"`, `"ollama"`, `"bedrock"`
- `model` (str): Model name (provider-specific)
- `temperature` (float): 0.0-2.0, default varies by provider
- `max_tokens` (int): Max completion tokens
- `timeout` (float): Request timeout in seconds

### `chat(messages) -> ChatResponse`

Sync chat completion.

```python
from miiflow_agent.core import Message

response = client.chat([
    Message.system("You are a helpful assistant."),
    Message.user("Explain async/await")
])

print(response.message.content)
print(f"Tokens: {response.usage.total_tokens}")
```

**Returns:** `ChatResponse`
- `message`: Message object with content
- `usage`: TokenCount (prompt_tokens, completion_tokens, total_tokens)
- `model`: Model used
- `provider`: Provider name
- `finish_reason`: "stop", "length", etc.

### `achat(messages) -> ChatResponse`

Async chat completion.

```python
import asyncio

async def main():
    response = await client.achat([Message.user("Hi")])
    print(response.message.content)

asyncio.run(main())
```

### `stream_chat(messages) -> Iterator[StreamChunk]`

Sync streaming.

```python
for chunk in client.stream_chat([Message.user("Count to 10")]):
    print(chunk.delta, end="", flush=True)
    if chunk.finish_reason:
        print(f"\nDone: {chunk.usage.total_tokens} tokens")
```

**Returns:** Iterator of `StreamChunk`
- `delta`: New content piece
- `content`: Accumulated content so far
- `finish_reason`: None until done
- `usage`: Token counts (only in final chunk)

### `astream_chat(messages) -> AsyncIterator[StreamChunk]`

Async streaming.

```python
async for chunk in client.astream_chat([Message.user("Hello")]):
    print(chunk.delta, end="", flush=True)
```

## Messages

### `Message.user(content)`

User message.

```python
from miiflow_agent.core import Message

# Text only
msg = Message.user("What is Rust?")

# Multi-modal (list of blocks)
from miiflow_agent.core import TextBlock, ImageBlock

msg = Message.user([
    TextBlock(text="Describe this:"),
    ImageBlock(image_url="https://example.com/img.jpg")
])
```

### `Message.assistant(content)`

Assistant message (for context).

```python
msg = Message.assistant("Rust is a systems programming language.")
```

### `Message.system(content)`

System prompt.

```python
msg = Message.system("You are a Python expert.")
```

### `TextBlock(text)`

Text content block.

```python
from miiflow_agent.core import TextBlock

block = TextBlock(text="What's in this image?")
```

### `ImageBlock(image_url, detail="auto")`

Image content block.

```python
from miiflow_agent.core import ImageBlock

# URL
block = ImageBlock(image_url="https://example.com/photo.jpg")

# Base64
block = ImageBlock(
    image_url="data:image/jpeg;base64,/9j/4AAQ...",
    detail="high"  # "low", "high", "auto"
)
```

## Agent

### `Agent(client, agent_type, system_prompt="")`

Create an agent with tools.

```python
from miiflow_agent import Agent
from miiflow_agent.core import AgentType

agent = Agent(
    client=client,
    agent_type=AgentType.REACT,  # XML-based ReAct
    system_prompt="You are a helpful assistant."
)
```

### `add_tool(tool_func)`

Add a tool to the agent.

```python
from miiflow_agent.core.tools import tool

@tool("search", "Search the web")
def search(query: str) -> str:
    return f"Results for: {query}"

agent.add_tool(search)
```

### `run(query, deps={}) -> AgentResult`

Run the agent (async).

```python
import asyncio

result = asyncio.run(agent.run(
    "What's the weather in Paris?",
    deps={}  # Dependencies for tools
))

print(result.data)  # Final answer
print(result.metadata["react_steps"])  # Execution steps
```

**Returns:** `AgentResult`
- `data`: Final answer string
- `metadata`: Dict with `react_steps` list
- `usage`: Total token usage

## Tools

### `@tool(name, description)`

Decorator to create a tool.

```python
from miiflow_agent.core.tools import tool

@tool("calculate", "Evaluate a math expression")
def calculate(expression: str) -> str:
    """
    Args:
        expression: Math expression like "2 + 2"
    """
    return str(eval(expression))

# Type hints are used for schema generation
@tool("get_user", "Fetch user by ID")
def get_user(user_id: int, include_email: bool = False) -> dict:
    return {"id": user_id, "name": "Alice"}
```

Function signature generates tool schema. Type hints required.

## Exceptions

```python
from miiflow_agent.core.exceptions import (
    ProviderError,          # Base exception
    RateLimitError,         # Rate limited
    InvalidRequestError,    # Bad request
    AuthenticationError,    # Invalid API key
    TimeoutError,          # Request timeout
)

try:
    response = client.chat([Message.user("Hi")])
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
    print(f"Retry after: {e.retry_after}")
except AuthenticationError:
    print("Check your API key")
except ProviderError as e:
    print(f"{e.provider} error: {e.message}")
```

## Types

### ChatResponse

```python
@dataclass
class ChatResponse:
    message: Message           # Assistant's response
    usage: TokenCount         # Token counts
    model: str               # Model used
    provider: str            # Provider name
    finish_reason: str       # "stop", "length", etc.
```

### StreamChunk

```python
@dataclass
class StreamChunk:
    delta: str               # New content
    content: str             # Accumulated content
    finish_reason: Optional[str]  # None until done
    usage: Optional[TokenCount]   # Only in final chunk
```

### TokenCount

```python
@dataclass
class TokenCount:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### AgentResult

```python
@dataclass
class AgentResult:
    data: str                # Final answer
    metadata: dict          # Execution details
    usage: TokenCount       # Total tokens used
```

## Advanced

### Custom Provider Configuration

```python
# OpenAI with custom base URL
client = LLMClient.create(
    "openai",
    model="gpt-4o-mini",
    base_url="https://custom-endpoint.com/v1",
    api_key="custom-key"
)

# Groq with custom timeout
client = LLMClient.create(
    "groq",
    model="llama-3.3-70b-versatile",
    timeout=120.0
)
```

### Context Management

```python
messages = [
    Message.system("You are a Python expert."),
    Message.user("How do I read a file?"),
    Message.assistant("Use open('file.txt') as f: content = f.read()"),
    Message.user("What about writing?")
]

response = client.chat(messages)
```

### Token Counting

```python
response = client.chat([Message.user("Hello")])

print(f"Prompt: {response.usage.prompt_tokens}")
print(f"Completion: {response.usage.completion_tokens}")
print(f"Total: {response.usage.total_tokens}")
```

### Streaming with Error Handling

```python
try:
    for chunk in client.stream_chat([Message.user("Hi")]):
        print(chunk.delta, end="")
except TimeoutError:
    print("Request timed out")
except ProviderError as e:
    print(f"Stream failed: {e.message}")
```
