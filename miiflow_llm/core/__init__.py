"""Core components for Miiflow LLM."""

# Initialize observability on module import
try:
    from .observability.auto_instrumentation import enable_phoenix_tracing
    from .observability.config import ObservabilityConfig
    
    # Simple Phoenix setup if enabled via environment
    config = ObservabilityConfig.from_env()
    if config.phoenix_enabled:
        enable_phoenix_tracing(config.phoenix_endpoint)
except ImportError:
    # Observability dependencies not available
    _observability_setup = {"phoenix_enabled": False}
except Exception:
    # Observability setup failed, continue without it
    _observability_setup = {"phoenix_enabled": False}

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

# Observability exports (optional)
try:
    from .observability import (
        ObservabilityConfig,
        TraceContext,
        get_current_trace_context,
    )
    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    _OBSERVABILITY_AVAILABLE = False

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

# Add observability exports if available
if _OBSERVABILITY_AVAILABLE:
    __all__.extend([
        "ObservabilityConfig",
        "TraceContext",
        "get_current_trace_context",
    ])
