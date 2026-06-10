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


class ToolInputValidationRejected(Exception):
    """Raised by the PRE_TOOL_USE emit when a callback set
    ``event.validation_error``: the call's inputs can't succeed (e.g. they'd
    bounce off the platform API), so instead of pausing for user approval the
    executor returns a FAILED ToolResult carrying ``reason`` — the model fixes
    the inputs and retries without a wasted approval modal.
    """

    def __init__(self, tool_name: str, tool_inputs: dict, reason: str = None):
        self.tool_name = tool_name
        self.tool_inputs = tool_inputs
        self.reason = reason or "Tool inputs failed validation"
        super().__init__(f"Tool '{tool_name}' inputs rejected: {self.reason}")


class PlanApprovalRequired(Exception):
    """Raised by the ``exit_plan_mode`` tool to pause the run for user
    approval of the proposed plan.

    The orchestrator catches this and emits a ``PLAN_APPROVAL_NEEDED``
    event, then halts the loop the same way it does for
    ``ToolApprovalRequired``. The server-side streaming service
    persists the pending plan to thread metadata; the user's
    approve/reject decision arrives as the next user message and the
    Django adapter resumes the run with ``permission_mode`` flipped to
    ``"default"`` (on approve) or kept as ``"plan"`` plus rejection
    feedback injected (on reject).

    Attributes:
        plan_text: The proposed plan text (markdown) the user is
            being asked to approve.
        tool_call_id: The id of the ``exit_plan_mode`` tool call;
            preserved so the next-turn tool result block round-trips
            correctly when the run resumes.
    """

    def __init__(self, plan_text: str, tool_call_id: str | None = None):
        self.plan_text = plan_text
        self.tool_call_id = tool_call_id
        super().__init__(
            f"Plan approval required (plan length: {len(plan_text)} chars)"
        )
