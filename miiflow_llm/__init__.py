"""
Miiflow LLM - A lightweight, unified interface for LLM providers.

This package provides a consistent API for calling multiple LLM providers
with support for streaming, tool calling, and structured output.
"""

# Agent Interface - Clean abstraction for miiflow-web
from .agents import AgentClient, AgentConfig, AgentContext, create_agent
from .core.client import ChatResponse, LLMClient, ModelClient, StreamChunk
from .core.exceptions import (
    AuthenticationError,
    MiiflowLLMError,
    ModelError,
    ParsingError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ToolError,
)
from .core.message import ContentBlock, ImageBlock, Message, MessageRole, TextBlock
from .core.metrics import LLMMetrics, MetricsCollector, TokenCount, UsageData

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
