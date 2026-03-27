"""Exception classes for ReAct and Plan & Execute systems."""


class ReActParsingError(Exception):
    """Raised when ReAct response cannot be parsed or healed."""

    pass


class ReActExecutionError(Exception):
    """Raised when ReAct execution fails."""

    pass


class SafetyViolationError(Exception):
    """Raised when a safety condition is violated."""

    pass


class ToolApprovalRequired(Exception):
    """Raised when a tool requires user approval before execution.

    This exception is raised by the PRE_TOOL_USE callback when a tool
    is configured to require manual approval. The orchestrator catches
    this and pauses execution, emitting a TOOL_APPROVAL_NEEDED event.
    """

    def __init__(self, tool_name: str, tool_inputs: dict, reason: str = None):
        self.tool_name = tool_name
        self.tool_inputs = tool_inputs
        self.reason = reason
        super().__init__(
            f"Tool '{tool_name}' requires user approval: {reason or 'manual approval configured'}"
        )
