# Providers

## OpenAI

```python
client = LLMClient.create("openai", model="gpt-4o-mini")
```

**API Key:** `OPENAI_API_KEY`

**Models:** `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `gpt-4-turbo`

**Features:** Vision, structured output

## Anthropic (Claude)

```python
client = LLMClient.create("anthropic", model="claude-3-5-sonnet-20241022")
```

**API Key:** `ANTHROPIC_API_KEY`

**Models:** `claude-3-5-sonnet-20241022`, `claude-4`, `claude-3-haiku-20240307`

**Features:** Vision, 200k context window

## Google (Gemini)

```python
client = LLMClient.create("gemini", model="gemini-1.5-flash")
```

**API Key:** `GEMINI_API_KEY`

**Models:** `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`

**Features:** Vision, audio, video, 1M context (Pro), free tier

**Configuration:**
```python
client = LLMClient.create(
    "gemini",
    model="gemini-1.5-pro",
    temperature=0.7,
    top_p=0.9,
    top_k=40
)
```

---

## Groq

```python
client = LLMClient.create("groq", model="llama-3.3-70b-versatile")
```

**API Key:** `GROQ_API_KEY`

**Models:** `llama-3.3-70b-versatile`, `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`

**Features:** Fast inference, free tier

## OpenRouter

```python
client = LLMClient.create("openrouter", model="anthropic/claude-3.5-sonnet")
```

**API Key:** `OPENROUTER_API_KEY`

**Example Models:** `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`, `google/gemini-pro-1.5`, `meta-llama/llama-3.3-70b-instruct`

**Features:** 200+ models, free tier

## Mistral

```python
client = LLMClient.create("mistral", model="mistral-large-latest")
```

**API Key:** `MISTRAL_API_KEY`

**Models:** `mistral-large-latest`, `mistral-small-latest`, `mistral-nemo`

## XAI (Grok)

```python
client = LLMClient.create("xai", model="grok-beta")
```

**API Key:** `XAI_API_KEY`

**Models:** `grok-beta`

## Configuration

### Custom Base URL

```python
client = LLMClient.create(
    "openai",
    model="gpt-4o-mini",
    base_url="https://custom-proxy.com/v1"
)
```

### Timeout

```python
client = LLMClient.create(
    "anthropic",
    model="claude-3-5-sonnet-20241022",
    timeout=120.0  # seconds
)
```

### Temperature

```python
client = LLMClient.create(
    "openai",
    model="gpt-4o-mini",
    temperature=0.7  # 0.0 = deterministic, 2.0 = creative
)
```

### Max Tokens

```python
client = LLMClient.create(
    "anthropic",
    model="claude-3-5-sonnet-20241022",
    max_tokens=4000
)
```


## Environment Variables

Set all your keys in `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
XAI_API_KEY=xai-...
OPENROUTER_API_KEY=sk-or-...
MISTRAL_API_KEY=...
```

Or set programmatically:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

client = LLMClient.create("openai", model="gpt-4o-mini")
```
