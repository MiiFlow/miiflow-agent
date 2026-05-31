"""
Clarification tool for asking users multiple-choice questions during agent execution.

This tool lets an agent pause and ask the user one or more **multiple-choice**
questions. When called, the orchestrator pauses execution and sends the
clarification request to the frontend, which renders each question as a
single- or multi-select chooser.

Design rule (intentional, see ``ask_user_clarification``):
    This tool is for *structured choices only*. Every question MUST ship a
    concrete list of ``options``. If the agent wants to ask an open-ended
    question, it should just ask it in its normal reply instead of calling
    this tool.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .decorators import tool
from .schemas import ToolResult


# Marker constant used to identify clarification requests in tool results
CLARIFICATION_MARKER = "__CLARIFICATION_REQUEST__"

# The name of the clarification tool - used for detection in orchestrators
CLARIFICATION_TOOL_NAME = "ask_user_clarification"


@dataclass
class ClarificationQuestion:
    """A single multiple-choice question within a clarification request."""

    question: str
    options: List[str] = field(default_factory=list)
    # When True the user may pick several options (checkboxes); otherwise a
    # single choice (radio).
    multi_select: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "options": list(self.options or []),
            "multi_select": bool(self.multi_select),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClarificationQuestion":
        return cls(
            question=data.get("question", ""),
            options=list(data.get("options") or []),
            multi_select=bool(data.get("multi_select", False)),
        )


@dataclass
class ClarificationRequest:
    """A set of multiple-choice questions to put to the user in one pause."""

    questions: List[ClarificationQuestion] = field(default_factory=list)
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "marker": CLARIFICATION_MARKER,
            "questions": [q.to_dict() for q in self.questions],
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClarificationRequest":
        raw_questions = data.get("questions")
        if raw_questions is None:
            # Tolerate the legacy single-question shape so in-flight threads
            # and old persisted clarifications still parse.
            legacy_options = data.get("options") or []
            raw_questions = [
                {
                    "question": data.get("question", ""),
                    "options": legacy_options,
                    "multi_select": False,
                }
            ]
        return cls(
            questions=[ClarificationQuestion.from_dict(q) for q in raw_questions],
            context=data.get("context"),
        )


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
        return ClarificationRequest.from_dict(output)

    return None


def _normalize_questions(questions: Any) -> List[ClarificationQuestion]:
    """Coerce the LLM-supplied ``questions`` payload into typed questions.

    Drops questions that have no concrete options — this tool is for
    multiple-choice only; open-ended questions belong in the agent's reply.
    """
    normalized: List[ClarificationQuestion] = []
    for item in questions or []:
        if not isinstance(item, dict):
            continue
        q = ClarificationQuestion.from_dict(item)
        if not q.question or len(q.options) < 1:
            continue
        normalized.append(q)
    return normalized


@tool(
    name=CLARIFICATION_TOOL_NAME,
    description=(
        "Ask the user one or more MULTIPLE-CHOICE questions when you need more "
        "information to proceed. Pass a list of questions; each is rendered as a "
        "single- or multi-select chooser and shown together.\n\n"
        "Every question MUST include a concrete `options` list. If you want to ask "
        "an OPEN-ENDED question, do NOT use this tool — just ask it directly in your "
        "normal reply. Only use this tool when you can offer the user concrete "
        "choices to pick from."
    ),
)
def ask_user_clarification(
    questions: List[Dict[str, Any]],
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask the user one or more multiple-choice questions.

    Args:
        questions: A list of question objects. Each object is
            ``{"question": str, "options": [str, ...], "multi_select": bool}``.
            ``options`` is required and must contain concrete choices.
            Set ``multi_select`` True to let the user pick several options.
        context: Optional shared explanation of why you need this information
            and how the answers will be used.

    Returns:
        A clarification request marker that signals the orchestrator to
        pause and request user input.

    Example:
        >>> ask_user_clarification(
        ...     questions=[
        ...         {
        ...             "question": "What time horizon are you investing for?",
        ...             "options": ["Short-term", "Medium-term", "Long-term"],
        ...         },
        ...         {
        ...             "question": "Which asset classes interest you?",
        ...             "options": ["Stocks", "Bonds", "Crypto", "Real estate"],
        ...             "multi_select": True,
        ...         },
        ...     ],
        ...     context="This tailors the recommendations to your goals.",
        ... )
    """
    request = ClarificationRequest(
        questions=_normalize_questions(questions),
        context=context,
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
