"""
Miiflow LLM - A lightweight, unified interface for LLM providers.

This package provides a consistent API for calling multiple LLM providers
with support for streaming, tool calling, structured output, and agentic patterns
like ReAct and Plan & Execute.

Quick Start:
    from miiflow_llm import LLMClient, Message

    # Simple chat
    client = LLMClient.create("openai", model="gpt-4o-mini")
    response = client.chat([Message.user("Hello")])

    # Agent with ReAct
    from miiflow_llm import Agent, AgentType
    from miiflow_llm.core.tools import tool

    @tool("calculate", "Do math")
    def calculate(expression: str) -> str:
        return str(eval(expression))

    agent = Agent(client, agent_type=AgentType.REACT)
    agent.add_tool(calculate)
    result = await agent.run("What is 25 * 4?")
"""

# Core LLM Client
from .core.client import ChatResponse, LLMClient, ModelClient, StreamChunk

# Message types
from .core.message import ContentBlock, ImageBlock, Message, MessageRole, TextBlock

# Agent system
from .core.agent import Agent, AgentType, RunContext, RunResult

# Tool system
from .core.tools import tool, FunctionTool, ToolRegistry

# Metrics
from .core.metrics import LLMMetrics, MetricsCollector, TokenCount, UsageData

# Callbacks
from .core.callbacks import (
    CallbackContext,
    CallbackEvent,
    CallbackEventType,
    CallbackRegistry,
    clear,
    get_global_registry,
    on_agent_run_end,
    on_agent_run_start,
    on_error,
    on_post_call,
    register,
    unregister,
)
from .core.callback_context import (
    callback_context,
    get_callback_context,
    reset_callback_context,
    set_callback_context,
)

# Exceptions
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

__version__ = "0.1.0"
__author__ = "Miiflow Team"

__all__ = [
    # Core LLM Client
    "LLMClient",
    "ModelClient",
    "ChatResponse",
    "StreamChunk",
    # Message types
    "Message",
    "MessageRole",
    "ContentBlock",
    "TextBlock",
    "ImageBlock",
    # Agent system
    "Agent",
    "AgentType",
    "RunContext",
    "RunResult",
    # Tool system
    "tool",
    "FunctionTool",
    "ToolRegistry",
    # Metrics
    "LLMMetrics",
    "TokenCount",
    "UsageData",
    "MetricsCollector",
    # Callbacks
    "CallbackContext",
    "CallbackEvent",
    "CallbackEventType",
    "CallbackRegistry",
    "callback_context",
    "clear",
    "get_callback_context",
    "get_global_registry",
    "on_agent_run_end",
    "on_agent_run_start",
    "on_error",
    "on_post_call",
    "register",
    "reset_callback_context",
    "set_callback_context",
    "unregister",
    # Exceptions
    "MiiflowLLMError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelError",
    "TimeoutError",
    "ParsingError",
    "ToolError",
]
