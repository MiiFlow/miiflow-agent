"""Tool-specific exceptions."""


class ToolPreparationError(Exception):
    """Raised when tool preparation fails."""
    pass


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class HTTPToolError(Exception):
    """Raised when HTTP tool operations fail."""
    pass


class ProxyError(Exception):
    """Raised when proxy configuration or usage fails."""
    pass


class ValidationError(Exception):
    """Raised when tool validation fails."""
    pass


# MCP-specific exceptions


class MCPConnectionError(Exception):
    """Raised when MCP server connection fails."""
    pass


class MCPToolError(Exception):
    """Raised when MCP tool execution fails."""
    pass


class MCPTimeoutError(Exception):
    """Raised when MCP operation times out."""
    pass
