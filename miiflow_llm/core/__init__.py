"""Core components for Miiflow LLM."""

from .client import LLMClient, ModelClient, ChatResponse, StreamChunk
from .message import Message, MessageRole, ContentBlock, TextBlock, ImageBlock
from .metrics import LLMMetrics, TokenCount, UsageData, MetricsCollector
from .streaming import (
    UnifiedStreamingClient, 
    IncrementalParser, 
    EnhancedStreamChunk
)
from .exceptions import (
    MiiflowLLMError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelError,
    TimeoutError,
    ParsingError,
    ToolError,
)
from .tools import (
    # Production-grade modular tools
    FunctionTool,
    ToolRegistry,
    FunctionOutput,
    ToolResult,
    FunctionType,
    ParameterType,
    PreparedCall,
    ToolPreparationError,
    ToolExecutionError,
    tool,
    detect_function_type,
    get_fun_schema
)
from .agent import (
    # Core agent architecture - Stateless framework
    Agent,
    RunContext,
    RunResult,
    AgentType,
)

__all__ = [
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
    "UnifiedStreamingClient", 
    "IncrementalParser", 
    "EnhancedStreamChunk",
    "MiiflowLLMError",
    "ProviderError",
    "AuthenticationError", 
    "RateLimitError",
    "ModelError",
    "TimeoutError",
    "ParsingError",
    "ToolError",
 
    "FunctionTool",
    "ToolRegistry",
    "SimpleToolRegistry",  # Backward compatibility
    "FunctionOutput", 
    "ToolResult",
    "FunctionType",
    "ParameterType",
    "PreparedCall",
    "ToolPreparationError",
    "ToolExecutionError",
    "tool",
    "detect_function_type",
    "get_fun_schema",
    
    # Core agent architecture - Stateless framework
    "Agent",
    "RunContext", 
    "RunResult",
    "AgentType",
]
