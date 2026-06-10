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
TOOL_APPROVAL_MARKER = "__TOOL_APPROVAL_REQUEST__"

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
    # Optional stable slug identifying the *subject* of this question (e.g.
    # ``"daily_budget"``). Used as the established-facts key so a re-ask of the
    # same subject short-circuits to the known answer deterministically. When
    # omitted, a canonical slug of the question text is used as the key.
    key: Optional[str] = None
    # When True the answer panel renders an inline "Other" text input so the
    # user can type a custom value submitted together with the option answers
    # — use instead of an "I'll type my own" option (which costs an extra
    # round trip to collect the actual value).
    allow_free_text: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "question": self.question,
            "options": list(self.options or []),
            "multi_select": bool(self.multi_select),
            "key": self.key,
        }
        if self.allow_free_text:
            data["allow_free_text"] = True
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClarificationQuestion":
        return cls(
            question=data.get("question", ""),
            options=list(data.get("options") or []),
            multi_select=bool(data.get("multi_select", False)),
            key=data.get("key"),
            allow_free_text=bool(data.get("allow_free_text", False)),
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

    Detection is by the private ``CLARIFICATION_MARKER`` in the output, NOT by tool
    name — so a clarification raised *inside a dispatched sub-agent* and surfaced back
    through the ``dispatch_assistant`` observation pauses the parent the same way a
    direct ``ask_user_clarification`` call does (Phase 2, R1). The marker is unique to
    clarification requests, so dropping the name gate is safe.

    Args:
        result: The tool result to check

    Returns:
        True if the result is a clarification request
    """
    if not result or not result.success:
        return False

    output = result.output
    if isinstance(output, dict):
        return output.get("marker") == CLARIFICATION_MARKER

    return False


def is_tool_approval_result(result: ToolResult) -> bool:
    """Check if a tool result is a surfaced child tool-approval request."""
    if not result or not result.success:
        return False

    output = result.output
    if isinstance(output, dict):
        return output.get("marker") == TOOL_APPROVAL_MARKER

    return False


def child_clarification_observation(
    questions: List[Dict[str, Any]],
    context: Optional[str],
    dispatch_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the parent-facing observation for a clarification surfaced by a sub-agent.

    Carries the clarification marker + the child's questions (so the parent's pause
    detection fires and the established-facts short-circuit composes) alongside the
    normal dispatch metadata (handle/subagent_id/status) so consumers of the dispatch
    observation don't need to fork.
    """
    payload: Dict[str, Any] = {
        "marker": CLARIFICATION_MARKER,
        "questions": list(questions or []),
        "context": context,
    }
    if dispatch_meta:
        for k, v in dispatch_meta.items():
            payload.setdefault(k, v)
    return payload


def child_tool_approval_observation(
    *,
    tool_name: str,
    tool_inputs: Dict[str, Any],
    tool_call_id: Optional[str],
    interrupt_id: Optional[str],
    reason: Optional[str] = None,
    tool_description: Optional[str] = None,
    tool_schema: Optional[Dict[str, Any]] = None,
    dispatch_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the parent-facing observation for a child approval pause.

    The marker is consumed by the parent orchestrator as control flow. It is not
    a prompt hint for the model; the run pauses before the model sees another
    reasoning turn.
    """
    payload: Dict[str, Any] = {
        "marker": TOOL_APPROVAL_MARKER,
        "type": "tool_approval",
        "tool_name": tool_name,
        "tool_inputs": dict(tool_inputs or {}),
        "tool_call_id": tool_call_id,
        "interrupt_id": interrupt_id,
        "reason": reason,
        "tool_description": tool_description or "",
        "tool_schema": tool_schema or {},
    }
    if dispatch_meta:
        for k, v in dispatch_meta.items():
            payload.setdefault(k, v)
    return payload


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
        "ASK EVERYTHING AT ONCE: gather ALL the information you'll need for the whole "
        "task in this SINGLE call. Do NOT drip-feed questions across turns — it is poor "
        "UX and asking again after the user already acted (e.g. after an approval) is "
        "especially bad. Think through every parameter the task requires, then ask for "
        "all of them together now.\n\n"
        "Give each question a stable `key` — a short slug naming its subject (e.g. "
        "\"daily_budget\", \"target_geo\"). The system uses it to remember the answer so "
        "you are never asked to re-clarify something already settled.\n\n"
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
            ``{"question": str, "options": [str, ...], "multi_select": bool,
            "key": str}``. ``options`` is required and must contain concrete
            choices. Set ``multi_select`` True to let the user pick several
            options. ``key`` is a short stable slug naming the question's subject
            (e.g. ``"daily_budget"``) — supply it so an already-answered subject is
            never re-asked. Ask for EVERYTHING you need in one call.
        context: Optional shared explanation of why you need this information
            and how the answers will be used.

    Returns:
        A clarification request marker that signals the orchestrator to
        pause and request user input.

    Example:
        >>> ask_user_clarification(
        ...     questions=[
        ...         {
        ...             "key": "time_horizon",
        ...             "question": "What time horizon are you investing for?",
        ...             "options": ["Short-term", "Medium-term", "Long-term"],
        ...         },
        ...         {
        ...             "key": "asset_classes",
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
