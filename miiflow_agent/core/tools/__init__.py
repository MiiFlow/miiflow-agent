"""
MiiFlow LLM Tools Package

A modular, production-ready tool system for function calling and HTTP API integration.
Supports multiple AI providers (OpenAI, Anthropic, Gemini, etc.) with comprehensive
error handling, proxy support, and context injection patterns.

Key Features:
- Function tools with automatic schema generation
- HTTP/REST API tools with proxy support 
- Production-grade registry with allowlist validation
- Multi-provider schema formatting (OpenAI, Anthropic, Gemini, etc.)
- Context injection patterns (Pydantic AI compatible)
- Comprehensive error handling and execution stats
- Easy-to-use decorators for tool definition

Quick Start:
    from miiflow_agent.core.tools import tool, ToolRegistry
    
    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b
    
    registry = ToolRegistry()
    registry.register(get_tool_from_function(add))
"""

# Core classes
from .registry import ToolRegistry
from .function import FunctionTool
from .http import HTTPTool

# MCP (Model Context Protocol) support
from .mcp import (
    MCPTool,
    MCPServerConfig,
    MCPServerConnection,
    StdioMCPConnection,
    SSEMCPConnection,
    StreamableHTTPMCPConnection,
    MCPToolManager,
    create_connection,
)

# Schemas and types
from .schemas import (
    ParameterSchema,
    ToolResult,
    ToolSchema,
    PreparedCall
)
from .types import ToolType, FunctionType, ParameterType

# Decorators and utilities
from .decorators import (
    tool,
    http_tool,
    get_tool_from_function,
    is_tool,
    get_tool_schema,
    auto_register_tools,
)

# Schema utilities
from .schema_utils import (
    get_fun_schema,
    detect_function_type
)

# Exceptions
from .exceptions import (
    ToolPreparationError,
    ToolExecutionError,
    HTTPToolError,
    ProxyError,
    ValidationError,
    MCPConnectionError,
    MCPToolError,
    MCPTimeoutError,
)

# Clarification tool for human-in-the-loop
from .clarification import (
    CLARIFICATION_MARKER,
    CLARIFICATION_TOOL_NAME,
    ClarificationRequest,
    is_clarification_result,
    extract_clarification_data,
    ask_user_clarification,
    create_clarification_tool,
)

# HTTP utilities (for advanced users)
from .http.proxy_utils import (
    get_proxy_config,
    should_use_proxy
)

# Context patterns (for framework integration)
from .function.context_patterns import (
    ContextPattern,
    detect_context_pattern,
    filter_context_params,
    analyze_context_pattern,
    filter_context_from_schema
)

# Version info
__version__ = "0.2.0"
__author__ = "MiiFlow Team"

# Public API exports
__all__ = [
    # Core classes
    "ToolRegistry",
    "FunctionTool",
    "HTTPTool",

    # MCP classes
    "MCPTool",
    "MCPServerConfig",
    "MCPServerConnection",
    "StdioMCPConnection",
    "SSEMCPConnection",
    "StreamableHTTPMCPConnection",
    "MCPToolManager",
    "create_connection",

    # Schemas and types
    "ParameterSchema",
    "ToolResult",
    "ToolSchema",
    "PreparedCall",
    "ToolType",
    "FunctionType",
    "ParameterType",

    # Decorators
    "tool",
    "http_tool",
    "get_tool_from_function",
    "is_tool",
    "get_tool_schema",
    "auto_register_tools",

    # Schema utilities
    "get_fun_schema",
    "get_type_string",
    "extract_parameter_info",

    # Exceptions
    "ToolPreparationError",
    "ToolExecutionError",
    "HTTPToolError",
    "ProxyError",
    "ValidationError",
    "MCPConnectionError",
    "MCPToolError",
    "MCPTimeoutError",

    # HTTP utilities
    "get_proxy_config",
    "should_use_proxy",

    # Context patterns
    "ContextPattern",
    "detect_context_pattern",
    "filter_context_params",

    # Clarification tool
    "CLARIFICATION_MARKER",
    "CLARIFICATION_TOOL_NAME",
    "ClarificationRequest",
    "is_clarification_result",
    "extract_clarification_data",
    "ask_user_clarification",
    "create_clarification_tool",
]


# Module-level convenience functions
def create_registry(allowlist=None, enable_logging=True):
    """
    Convenience function to create a tool registry.

    Args:
        allowlist: Optional list of allowed tool names
        enable_logging: Whether to enable logging

    Returns:
        ToolRegistry instance
    """
    return ToolRegistry(allowlist=allowlist, enable_logging=enable_logging)

# Add convenience functions to __all__
__all__.extend([
    "create_registry",
])

# Package metadata for introspection
__package_info__ = {
    "name": "miiflow-agent-tools",
    "version": __version__,
    "description": "Modular tool system for AI function calling",
    "features": [
        "Function tools with automatic schema generation",
        "HTTP/REST API tools with proxy support",
        "MCP (Model Context Protocol) client support",
        "Multi-provider compatibility (OpenAI, Anthropic, Gemini, etc.)",
        "Production-grade error handling and validation",
        "Context injection patterns",
        "Execution statistics and monitoring",
        "Easy-to-use decorators",
    ],
    "supported_providers": [
        "OpenAI", "Anthropic", "Google Gemini", "Groq",
        "Mistral", "Ollama", "OpenRouter", "xAI"
    ],
    "mcp_transports": [
        "stdio",
        "streamable_http",
        "sse",
    ],
}
