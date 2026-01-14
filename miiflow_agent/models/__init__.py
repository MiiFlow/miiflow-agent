"""Model configurations and capabilities for LLM providers.

This module contains model definitions including capabilities, parameter
constraints, pricing hints, and helper functions for correct API parameter selection.

Model configurations serve as the single source of truth for model capabilities
that can be shared between the SDK and server implementations.
"""

# Anthropic
from .anthropic import ANTHROPIC_MODELS, ANTHROPIC_PARAMETERS
from .anthropic import supports_structured_outputs as anthropic_supports_structured_outputs
from .anthropic import supports_thinking as anthropic_supports_thinking
from .base import ModelConfig, ParameterConfig, ParameterType

# DeepSeek
from .deepseek import DEEPSEEK_MODELS, DEEPSEEK_PARAMETERS
from .deepseek import get_token_param_name as deepseek_get_token_param_name

# Google
from .google import GOOGLE_MODELS, GOOGLE_PARAMETERS
from .google import get_token_param_name as google_get_token_param_name

# Groq
from .groq import GROQ_MODELS, GROQ_PARAMETERS

# Mistral
from .mistral import MISTRAL_MODELS, MISTRAL_PARAMETERS

# Ollama
from .ollama import OLLAMA_MODELS, OLLAMA_PARAMETERS
from .ollama import get_token_param_name as ollama_get_token_param_name

# OpenAI
from .openai import OPENAI_MODELS, OPENAI_PARAMETERS
from .openai import get_token_param_name as openai_get_token_param_name
from .openai import supports_temperature as openai_supports_temperature

# OpenRouter
from .openrouter import OPENROUTER_MODELS, OPENROUTER_PARAMETERS

# xAI
from .xai import XAI_MODELS, XAI_PARAMETERS

__all__ = [
    # Base types
    "ModelConfig",
    "ParameterConfig",
    "ParameterType",
    # OpenAI
    "OPENAI_MODELS",
    "OPENAI_PARAMETERS",
    "openai_get_token_param_name",
    "openai_supports_temperature",
    # Anthropic
    "ANTHROPIC_MODELS",
    "ANTHROPIC_PARAMETERS",
    "anthropic_supports_structured_outputs",
    "anthropic_supports_thinking",
    # Google
    "GOOGLE_MODELS",
    "GOOGLE_PARAMETERS",
    "google_get_token_param_name",
    # Groq
    "GROQ_MODELS",
    "GROQ_PARAMETERS",
    # DeepSeek
    "DEEPSEEK_MODELS",
    "DEEPSEEK_PARAMETERS",
    "deepseek_get_token_param_name",
    # Mistral
    "MISTRAL_MODELS",
    "MISTRAL_PARAMETERS",
    # Ollama
    "OLLAMA_MODELS",
    "OLLAMA_PARAMETERS",
    "ollama_get_token_param_name",
    # OpenRouter
    "OPENROUTER_MODELS",
    "OPENROUTER_PARAMETERS",
    # xAI
    "XAI_MODELS",
    "XAI_PARAMETERS",
]
