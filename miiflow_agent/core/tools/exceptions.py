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


def is_tool_validation_error(exc: BaseException) -> bool:
    """True when a tool raised a *declared* input-shape rejection.

    Tools mark exceptions with ``is_tool_validation_error = True`` when the
    failure means "the model must fix its call" (bad GAQL, unknown field, …)
    rather than "the tool malfunctioned". The registry propagates the flag
    into ``ToolResult.metadata['is_validation_error']`` so the recovery
    manager skips the runtime-failure ladder — and the same flag decides how
    the failure is *logged*: a validation rejection is the tool's normal
    contract with the model, not an exception worth an Error Tracking issue.
    """
    return bool(getattr(exc, "is_tool_validation_error", False))


def is_declared_tool_failure(exc: BaseException) -> bool:
    """True when a tool DECLARED this failure — nothing of ours malfunctioned.

    Two markers say so, and they are deliberately separate concepts:

    * ``is_tool_validation_error`` — the MODEL must fix its call. Also steers
      the recovery ladder, so its meaning must not widen.
    * a non-empty ``remedy`` — a PERSON must fix something outside the code:
      a key without the right scope, a connection missing a required field, a
      suspended ad account. Set by the tool that already knows the fix, and
      forwarded to the model by ``_tool_error_payload``.

    Neither is a defect, and neither should mint an Error Tracking issue. A
    failure we can already explain does not need a human to come and diagnose
    it; one we cannot is exactly what an issue is for. Only *logging* reads
    this union — nothing about retries or recovery consults it.
    """
    if is_tool_validation_error(exc):
        return True
    remedy = getattr(exc, "remedy", None)
    return isinstance(remedy, str) and bool(remedy.strip())


def log_tool_failure(logger, message: str, exc: BaseException) -> None:
    """Log a failed tool call at the right severity.

    ``logger.exception`` for genuine malfunctions (traceback attached, so the
    PostHog log bridge captures it as an ``$exception`` and Error Tracking
    alerts). ``logger.warning`` — no traceback — for failures the tool
    declared (see ``is_declared_tool_failure``): a malformed GAQL query the
    model self-corrects on the next turn, a Triple Whale key without the
    Summary Page scope. Reporting those as exceptions pages on-call for model
    typos and for customer configuration.
    """
    if is_declared_tool_failure(exc):
        logger.warning("%s [declared_tool_failure=%s]", message, type(exc).__name__)
    else:
        logger.exception(message)
