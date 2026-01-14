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
