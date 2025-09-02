# MiiFlow LLM

**Production-ready unified LLM API layer** that solves streaming inconsistencies and provides a single interface across 9 providers.

[![Tests](https://img.shields.io/badge/tests-78%20passed-brightgreen)](tests/)
[![Providers](https://img.shields.io/badge/providers-9%20supported-blue)](#supported-providers)
[![Coverage](https://img.shields.io/badge/streaming-unified-orange)](#streaming-interface)

## 🎯 Problem Solved

**GPT-5 Streaming Inconsistency Crisis:**
- **GPT-4**: Uses `response_gen` iterator
- **GPT-5**: Uses `chunk.delta` attributes  
- **Claude**: Uses `chunk.message.content`
- **Groq**: Requires `str(chunk)` fallback
- **Result**: Broken code when switching providers

**MiiFlow LLM Solution:** Single `StreamChunk` format across ALL providers ✅

## 🚀 Quick Start

```python
from miiflow_llm import LLMClient
from miiflow_llm.core import Message

# Same API across ALL 9 providers
client = LLMClient.create("openai", model="gpt-5")
# client = LLMClient.create("anthropic", model="claude-3-5-sonnet-20241022")  
# client = LLMClient.create("gemini", model="gemini-1.5-pro")

# Unified streaming interface
messages = [Message.user("Explain quantum computing")]
async for chunk in client.stream_chat(messages):
    print(chunk.delta, end="")  # Same format everywhere!
```

## ✅ Supported Providers (9 Total)

| Provider | Models | Status | Streaming |
|----------|--------|--------|-----------|
| **OpenAI** | GPT-4, GPT-4o, **GPT-5** | ✅ | ✅ |
| **Anthropic** | Claude 3, Claude 3.5 Sonnet | ✅ | ✅ |
| **Google** | Gemini 1.5 Pro, Flash, Flash-8B | ✅ | ✅ |
| **Groq** | Llama 3.1, Llama 3.3, Mixtral | ✅ | ✅ |
| **xAI** | Grok Beta | ✅ | ✅ |
| **TogetherAI** | Meta Llama, Mixtral, Nous | ✅ | ✅ |
| **OpenRouter** | 200+ models, Free tier | ✅ | ✅ |
| **Mistral** | Mistral Small, Large | ✅ | ✅ |
| **Ollama** | Local models (Llama, etc.) | ✅ | ✅ |

## 🏗️ Architecture

### 1. Unified Streaming Layer
```python
# All providers return identical StreamChunk format
@dataclass
class StreamChunk:
    content: str          # Accumulated content
    delta: str           # New piece of content  
    finish_reason: str   # "stop", "length", etc.
    usage: TokenCount    # Standardized token counts
```

### 2. Provider Stream Normalizer
```python
# Converts provider-specific formats to unified StreamContent
class ProviderStreamNormalizer:
    def normalize_chunk(self, chunk: Any, provider: str) -> StreamContent:
        # Maps: OpenAI chunk.delta -> Anthropic chunk.message -> Unified format
```

### 3. Multi-Modal Message Support
```python  
from miiflow_llm.core import Message, TextBlock, ImageBlock

# Unified message format with image support
message = Message.user([
    TextBlock(text="What's in this image?"),
    ImageBlock(image_url="data:image/jpeg;base64,...", detail="high")
])
```

## 🔧 Advanced Features

### Structured Output with Streaming
```python
from dataclasses import dataclass

@dataclass
class Analysis:
    sentiment: str
    confidence: float
    topics: List[str]

# Get structured output while streaming
async for chunk in client.stream_with_schema(messages, schema=Analysis):
    if chunk.partial_parse:
        print(f"Partial: {chunk.partial_parse}")
    if chunk.structured_output:
        result: Analysis = chunk.structured_output
        break
```

### Metrics & Observability
```python
# Automatic metrics collection
metrics = client.get_metrics()
print(f"Total tokens: {metrics.total_tokens}")
print(f"Average latency: {metrics.avg_latency_ms}ms")
print(f"Success rate: {metrics.success_rate}%")
```

### Error Handling & Retries
```python
from miiflow_llm.core.exceptions import ProviderError, RateLimitError

try:
    response = await client.chat(messages)
except RateLimitError:
    # Automatic exponential backoff with provider-specific handling
    pass
except ProviderError as e:
    print(f"Provider {e.provider} error: {e.message}")
```

## 🧪 Testing & Reliability

**Comprehensive Test Suite: 78 tests passing**
- **Provider Adapters**: Unit tests for all 9 providers
- **Streaming Normalization**: Tests for chunk format consistency  
- **Error Handling**: Robustness testing with network failures
- **Multi-modal**: Image and file input validation
- **Integration**: End-to-end testing with real provider patterns

```bash
python -m pytest tests/  # 78 passed, 0 failed ✅
python test_unified_streaming.py  # Integration test
```

## 📦 Installation & Setup

```bash
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env file
```

**Environment Setup:**
```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
XAI_API_KEY=gsk_...
TOGETHERAI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
MISTRAL_API_KEY=...  # Optional
# OLLAMA_API_KEY not needed for local usage
```

## 🎯 Use Cases

- **Multi-provider Applications**: Switch between providers without code changes
- **A/B Testing**: Compare model performance across providers  
- **Failover Systems**: Automatic fallback when providers are down
- **Cost Optimization**: Route to cheapest provider for each request
- **GPT-5 Migration**: Seamless upgrade from GPT-4 without refactoring

## 🏆 Key Benefits

✅ **Eliminates GPT-5 streaming inconsistency**  
✅ **Single API across 9 providers**  
✅ **Production-ready error handling**  
✅ **Comprehensive test coverage (78 tests)**  
✅ **Multi-modal support (text + images)**  
✅ **Structured output streaming**  
✅ **Built-in metrics & observability**  
✅ **Local model support (Ollama)**  

## 🔮 Future Roadmap

- **Function Calling**: Unified tool interface across providers
- **Batch Processing**: Efficient bulk request handling  
- **Caching Layer**: Response caching with TTL
- **Load Balancing**: Smart request distribution
- **More Providers**: Cohere, AI21, etc.

---

**Built for production LLM applications that demand reliability and consistency across providers.**
