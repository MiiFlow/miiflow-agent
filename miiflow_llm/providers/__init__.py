"""Provider implementations for different LLM services."""

from typing import Dict, Type
from ..core.client import ModelClient

# Import observability components
try:
    from ..core.observability.auto_instrumentation import setup_openinference_instrumentation
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


def get_provider_client(
    provider: str,
    model: str,
    api_key: str = None,
    **kwargs
) -> ModelClient:
    """Get provider client instance with observability."""

    provider_map: Dict[str, Type[ModelClient]] = {
        "openai": lambda: _import_openai_client(),
        "anthropic": lambda: _import_anthropic_client(),
        "groq": lambda: _import_groq_client(),
        "xai": lambda: _import_xai_client(),
        "gemini": lambda: _import_gemini_client(),
        "together": lambda: _import_together_client(),
        "openrouter": lambda: _import_openrouter_client(),
        "mistral": lambda: _import_mistral_client(),
        "ollama": lambda: _import_ollama_client(),
    }

    if provider not in provider_map:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {list(provider_map.keys())}")

    client_class = provider_map[provider]()

    # Setup auto-instrumentation if available (one-time setup)
    if OBSERVABILITY_AVAILABLE:
        # This is idempotent - safe to call multiple times
        setup_openinference_instrumentation()

    return client_class(model=model, api_key=api_key, **kwargs)


def _import_openai_client():
    """Lazy import OpenAI client."""
    from .openai_client import OpenAIClient
    return OpenAIClient


def _import_anthropic_client():
    """Lazy import Anthropic client."""
    from .anthropic_client import AnthropicClient
    return AnthropicClient


def _import_groq_client():
    """Lazy import Groq client."""
    from .groq_client import GroqClient
    return GroqClient


def _import_xai_client():
    """Lazy import xAI client."""
    from .xai_client import XAIClient
    return XAIClient


def _import_gemini_client():
    """Lazy import Gemini client."""
    from .gemini_client import GeminiClient
    return GeminiClient


def _import_together_client():
    """Lazy import TogetherAI client."""
    from .together_client import TogetherClient
    return TogetherClient


def _import_openrouter_client():
    """Lazy import OpenRouter client."""
    from .openrouter_client import OpenRouterClient
    return OpenRouterClient


def _import_mistral_client():
    """Lazy import Mistral client."""
    from .mistral_client import MistralClient
    return MistralClient


def _import_ollama_client():
    """Lazy import Ollama client."""
    from .ollama_client import OllamaClient
    return OllamaClient