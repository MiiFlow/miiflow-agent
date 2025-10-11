"""
Miiflow LLM - A lightweight, unified interface for LLM providers.

This package provides a consistent API for calling multiple LLM providers
with support for streaming, tool calling, and structured output.
"""

from .core.client import LLMClient, ModelClient, ChatResponse, StreamChunk
from .core.message import Message, MessageRole, ContentBlock, TextBlock, ImageBlock
from .core.metrics import LLMMetrics, TokenCount, UsageData, MetricsCollector
from .core.exceptions import (
    MiiflowLLMError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelError,
    TimeoutError,
    ParsingError,
    ToolError,
)

# Agent Interface - Clean abstraction for miiflow-web
from .agents import AgentClient, AgentConfig, create_agent, AgentContext

__version__ = "0.1.0"
__author__ = "Debjyoti Ray"

__all__ = [
    # Core Components
    "LLMClient",
    "ModelClient",
    "ChatResponse",
    "StreamChunk",
    "Message",
    "MessageRole", 
    "ContentBlock",
    "TextBlock",
    "ImageBlock",
    "LLMMetrics",
    "TokenCount",
    "UsageData",
    "MetricsCollector",
    
    # Exceptions
    "MiiflowLLMError",
    "ProviderError", 
    "AuthenticationError",
    "RateLimitError",
    "ModelError",
    "TimeoutError",
    "ParsingError",
    "ToolError",
    
    # Agent Interface - Clean abstraction for miiflow-web
    "AgentClient",
    "AgentConfig", 
    "create_agent",
    "AgentContext",
]
