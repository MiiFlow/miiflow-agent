"""
Clarification tool for asking users questions during agent execution.

This tool allows agents to explicitly request user input when they need
clarification to proceed. When called, the orchestrator will pause execution
and send the clarification request to the frontend.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .decorators import tool
from .schemas import ToolResult


# Marker constant used to identify clarification requests in tool results
CLARIFICATION_MARKER = "__CLARIFICATION_REQUEST__"

# The name of the clarification tool - used for detection in orchestrators
CLARIFICATION_TOOL_NAME = "ask_user_clarification"


@dataclass
class ClarificationRequest:
    """Data structure for a clarification request."""

    question: str
    options: Optional[List[str]] = None
    context: Optional[str] = None
    allow_free_text: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "marker": CLARIFICATION_MARKER,
            "question": self.question,
            "options": self.options,
            "context": self.context,
            "allow_free_text": self.allow_free_text,
        }


def is_clarification_result(result: ToolResult) -> bool:
    """
    Check if a tool result is a clarification request.

    Args:
        result: The tool result to check

    Returns:
        True if the result is a clarification request
    """
    if not result or not result.success:
        return False

    if result.name != CLARIFICATION_TOOL_NAME:
        return False

    output = result.output
    if isinstance(output, dict):
        return output.get("marker") == CLARIFICATION_MARKER

    return False


def extract_clarification_data(result: ToolResult) -> Optional[ClarificationRequest]:
    """
    Extract clarification data from a tool result.

    Args:
        result: The tool result containing clarification data

    Returns:
        ClarificationRequest if valid, None otherwise
    """
    if not is_clarification_result(result):
        return None

    output = result.output
    if isinstance(output, dict):
        return ClarificationRequest(
            question=output.get("question", ""),
            options=output.get("options"),
            context=output.get("context"),
            allow_free_text=output.get("allow_free_text", True),
        )

    return None


@tool(
    name=CLARIFICATION_TOOL_NAME,
    description=(
        "Ask the user a question when you need more information to complete the task. "
        "Use this tool when the user's request is ambiguous, missing critical details, "
        "or when you need to confirm assumptions before proceeding. "
        "The user will be prompted to respond, and execution will resume with their answer."
    ),
)
def ask_user_clarification(
    question: str,
    options: Optional[List[str]] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask the user for clarification.

    This tool pauses agent execution and prompts the user to provide
    additional information. Use it when:
    - The user's request is ambiguous
    - Critical information is missing
    - You need to confirm assumptions before proceeding
    - Multiple valid approaches exist and user preference matters

    Args:
        question: The question to ask the user. Be specific and clear
            about what information you need.
        options: Optional list of predefined answer choices. If provided,
            the user can select from these options or provide custom input.
            Use this for multiple-choice scenarios.
        reason: Optional explanation of why this clarification is needed
            and how the answer will be used.

    Returns:
        A clarification request marker that signals the orchestrator to
        pause and request user input.

    Example:
        >>> ask_user_clarification(
        ...     question="What time horizon are you investing for?",
        ...     options=["Short-term (1-3 years)", "Medium-term (3-7 years)", "Long-term (7+ years)"],
        ...     reason="This will help me tailor investment recommendations to your goals."
        ... )
    """
    request = ClarificationRequest(
        question=question,
        options=options,
        context=reason,  # Map reason to context for internal use
        allow_free_text=True,
    )
    return request.to_dict()


def create_clarification_tool():
    """
    Create and return a FunctionTool instance for the clarification tool.

    Returns:
        FunctionTool instance ready to be registered with a ToolRegistry
    """
    from .function import FunctionTool

    return FunctionTool(ask_user_clarification)
