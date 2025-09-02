# MiiFlow LLM

Unified abstraction layer for large language model providers addressing streaming interface inconsistencies and provider-specific API variations.

[![Tests](https://img.shields.io/badge/tests-78%20passed-brightgreen)](tests/)
[![Providers](https://img.shields.io/badge/providers-9%20supported-blue)](#supported-providers)
[![Coverage](https://img.shields.io/badge/streaming-unified-orange)](#streaming-interface)

## Problem Statement

Current LLM integrations suffer from inconsistent streaming response formats across providers:
- **GPT-4**: Uses `response_gen` iterator
- **GPT-5**: Uses `chunk.delta` attributes  
- **Claude**: Uses `chunk.message.content`
- **Groq**: Requires `str(chunk)` fallback
- **Result**: Breaking changes when switching providers

**Solution:** Unified `StreamChunk` format with provider abstraction layer.

## Usage

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

## Supported Providers

| Provider | Models | Status | Streaming |
|----------|--------|--------|-----------|
| **OpenAI** | GPT-4, GPT-4o, GPT-5 | Active | Yes |
| **Anthropic** | Claude 3, Claude 3.5 Sonnet | Active | Yes |
| **Google** | Gemini 1.5 Pro, Flash, Flash-8B | Active | Yes |
| **Groq** | Llama 3.1, Llama 3.3, Mixtral | Active | Yes |
| **xAI** | Grok Beta | Active | Yes |
| **TogetherAI** | Meta Llama, Mixtral, Nous | Active | Yes |
| **OpenRouter** | 200+ models, Free tier | Active | Yes |
| **Mistral** | Mistral Small, Large | Active | Yes |
| **Ollama** | Local models (Llama, etc.) | Active | Yes |

## Architecture

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

## Advanced Features

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

## Testing & Reliability

**Comprehensive Test Suite: 78 tests passing**
- **Provider Adapters**: Unit tests for all 9 providers
- **Streaming Normalization**: Tests for chunk format consistency  
- **Error Handling**: Robustness testing with network failures
- **Multi-modal**: Image and file input validation
- **Integration**: End-to-end testing with real provider patterns

```bash
python -m pytest tests/  # 78 passed, 0 failed
python test_unified_streaming.py  # Integration test
```

## Installation & Setup

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

## Integration with MiiFlow Web

### Installation in MiiFlow Web Project

```bash
# In your miiflow-web project directory
pip install -e ../miiflow-llm  # Local development
# OR
pip install git+https://github.com/MiiFlow/miiflow-llm.git  # Latest from GitHub
```
### Example Run:
```bash
python miiflow-llm/examples/test_unified_streaming.py
```

### Backend Integration

```python
# In your FastAPI/Django backend
from miiflow_llm import LLMClient
from miiflow_llm.core import Message

# Create a service layer for LLM operations
class LLMService:
    def __init__(self):
        self.clients = {}
    
    def get_client(self, provider: str, model: str) -> LLMClient:
        """Get or create LLM client for provider/model combination."""
        key = f"{provider}:{model}"
        if key not in self.clients:
            self.clients[key] = LLMClient.create(provider, model=model)
        return self.clients[key]
    
    async def chat_completion(self, provider: str, model: str, messages: list) -> dict:
        """Handle chat completion request from frontend."""
        client = self.get_client(provider, model)
        
        # Convert dict messages to Message objects
        llm_messages = [Message.user(msg['content']) for msg in messages if msg['role'] == 'user']
        
        response = await client.chat(llm_messages)
        
        return {
            'content': response.message.content,
            'provider': response.provider,
            'model': response.model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }
    
    async def stream_completion(self, provider: str, model: str, messages: list):
        """Handle streaming completion for real-time responses."""
        client = self.get_client(provider, model)
        llm_messages = [Message.user(msg['content']) for msg in messages if msg['role'] == 'user']
        
        async for chunk in client.stream_chat(llm_messages):
            yield {
                'delta': chunk.delta,
                'content': chunk.content,
                'finish_reason': chunk.finish_reason
            }

# FastAPI endpoint example
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
llm_service = LLMService()

@app.post("/api/chat")
async def chat_endpoint(request: dict):
    """Standard chat completion endpoint."""
    return await llm_service.chat_completion(
        provider=request['provider'],
        model=request['model'], 
        messages=request['messages']
    )

@app.post("/api/chat/stream")
async def stream_endpoint(request: dict):
    """Streaming chat completion endpoint."""
    return StreamingResponse(
        llm_service.stream_completion(
            provider=request['provider'],
            model=request['model'],
            messages=request['messages']
        ),
        media_type="text/plain"
    )
```

### Frontend Integration

```javascript
// In your React/Vue/Angular frontend
class LLMClient {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
    }
    
    async chat(provider, model, messages) {
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, model, messages })
        });
        return response.json();
    }
    
    async* streamChat(provider, model, messages) {
        const response = await fetch(`${this.baseUrl}/chat/stream`, {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, model, messages })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = JSON.parse(decoder.decode(value));
            yield chunk;
        }
    }
}

// Usage in React component
const llmClient = new LLMClient();

// Standard completion
const response = await llmClient.chat('openai', 'gpt-4o', [
    { role: 'user', content: 'Hello!' }
]);

// Streaming completion
for await (const chunk of llmClient.streamChat('openai', 'gpt-4o', messages)) {
    console.log(chunk.delta); // Real-time response
}
```

### Configuration for MiiFlow Web

```python
# settings.py or config.py
MIIFLOW_LLM_CONFIG = {
    'default_provider': 'openai',
    'default_model': 'gpt-4o',
    'timeout': 60.0,
    'max_retries': 3,
    'enable_metrics': True,
    'providers': {
        'openai': {
            'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-5'],
            'default_temperature': 0.7
        },
        'anthropic': {
            'models': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
            'default_temperature': 0.7
        },
        'groq': {
            'models': ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile'],
            'default_temperature': 0.7
        }
    }
}
```

## Use Cases

- **Multi-provider Applications**: Switch between providers without code changes
- **A/B Testing**: Compare model performance across providers  
- **Failover Systems**: Automatic fallback when providers are down
- **Cost Optimization**: Route to cheapest provider for each request
- **GPT-5 Migration**: Seamless upgrade from GPT-4 without refactoring

## Key Benefits

- **Eliminates GPT-5 streaming inconsistency**  
- **Single API across 9 providers**  
- **Production-ready error handling**  
- **Comprehensive test coverage (78 tests)**  
- **Multi-modal support (text + images)**  
- **Structured output streaming**  
- **Built-in metrics & observability**  
- **Local model support (Ollama)**  

## Future Roadmap

- **Function Calling**: Unified tool interface across providers
- **Batch Processing**: Efficient bulk request handling  
- **Caching Layer**: Response caching with TTL
- **Load Balancing**: Smart request distribution
- **More Providers**: Cohere, AI21, etc.

---

**Built for production LLM applications that demand reliability and consistency across providers.**
