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


class MCPAuthRequired(Exception):
    """Raised when an MCP call needs the calling user to (re)authorize.

    Distinct from MCPConnectionError because the failure isn't on the server
    or the network — it's the user-side OAuth state. The agent runtime should
    propagate this as a structured `auth_required` event so the chat UI can
    render a "Reconnect <provider>" button.

    Carries enough context for the UI to identify which server needs reauth
    without round-tripping to the backend.

    Note: the Django side defines its own MCPAuthRequired in
    workflow.services.mcp_oauth that subclasses this one, so a single
    `except MCPAuthRequired` clause catches both.
    """

    def __init__(
        self,
        mcp_server_id: str,
        mcp_server_name: str,
        reason: str = "",
    ):
        self.mcp_server_id = mcp_server_id
        self.mcp_server_name = mcp_server_name
        self.reason = reason
        super().__init__(
            f"MCP server {mcp_server_name!r} requires user authorization"
            + (f": {reason}" if reason else "")
        )
