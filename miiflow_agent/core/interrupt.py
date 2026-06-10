"""Unified human-in-the-loop Interrupt primitive + deterministic established-facts logic.

This is the runtime half of Pillar 2 (the persisted half is ``PendingInterrupt`` in
``checkpoint.py``). It collapses the three historically-divergent pause paths —
clarification, tool approval, plan approval — into one primitive an agent raises with a
``GraphInterrupt``, which the orchestrator catches, checkpoints, and surfaces as a single
``INTERRUPT`` event.

It also houses the *deterministic* established-facts machinery (R4) that replaces the
server's 50%-token-overlap loop detector:

  * ``question_key`` — a stable identity for a clarification question (model-supplied
    ``key`` slug, else a deterministic canonicalization of the question text). NOT fuzzy
    matching: identical questions map to identical keys, full stop.
  * ``partition_questions_by_facts`` — split a clarification round into already-resolved
    (short-circuit to the known answer) vs genuinely-new (must pause).
  * ``render_established_facts_block`` — the "already settled — do not ask again" block
    injected into *every* agent's prompt. This is the behavioral guarantee against
    re-asking; the short-circuit is the deterministic backstop.

Everything here is pure except ``mint_interrupt_id`` (uuid). No fuzzy matching, no
thresholds — the only heuristic the old system had is gone.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .checkpoint import EstablishedFact, PendingInterrupt

INTERRUPT_KIND_CLARIFICATION = "clarification"
INTERRUPT_KIND_TOOL_APPROVAL = "tool_approval"
INTERRUPT_KIND_PLAN_APPROVAL = "plan_approval"

VALID_INTERRUPT_KINDS = frozenset(
    {
        INTERRUPT_KIND_CLARIFICATION,
        INTERRUPT_KIND_TOOL_APPROVAL,
        INTERRUPT_KIND_PLAN_APPROVAL,
    }
)

# Hard circuit-breaker (NOT a content heuristic): an absolute ceiling on how many
# clarification pauses one turn may raise, so a pathological model can't loop forever
# even though the established-facts injection should make re-asking unnecessary.
MAX_CLARIFICATION_INTERRUPTS_PER_TURN = 5


class GraphInterrupt(Exception):
    """Raised by any agent to pause the run for human input.

    Carries the typed ``PendingInterrupt`` the orchestrator checkpoints and surfaces.
    The orchestrator — not the model — owns continuation: on resume it re-drives the
    suspended tool call and the originating ``interrupt()`` call returns the user value.
    """

    def __init__(self, interrupt: PendingInterrupt):
        self.interrupt = interrupt
        super().__init__(f"interrupt:{interrupt.kind}:{interrupt.interrupt_id}")


def mint_interrupt_id(kind: str, tool_call_id: Optional[str] = None) -> str:
    """Stable, collision-free interrupt id.

    Prefers the tool_call_id (already unique per call and convenient for transcript
    pairing); falls back to a uuid. The ``kind`` prefix keeps ids self-describing in logs.
    """
    suffix = tool_call_id or uuid.uuid4().hex
    return f"int_{kind}_{suffix}"


# --- established-facts key derivation ----------------------------------------------

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Deterministic canonicalization of free text → a stable key.

    Lowercase, collapse any run of non-alphanumerics to a single underscore, trim.
    This is exact canonicalization, not similarity: "What daily budget?" →
    ``what_daily_budget``. Reworded questions get *different* keys by design — the
    injection (not fuzzy matching) is what prevents re-asking.
    """
    return _SLUG_STRIP.sub("_", (text or "").strip().lower()).strip("_")


def question_key(question: Dict[str, Any]) -> str:
    """Stable identity for a clarification question.

    Uses the model-supplied ``key`` slug when present (semantic, survives rewording),
    else a deterministic slug of the question text (exact floor). Either way the result
    is deterministic — the same input always yields the same key.
    """
    explicit = (question or {}).get("key")
    if explicit and isinstance(explicit, str) and explicit.strip():
        return _slugify(explicit)
    return _slugify((question or {}).get("question", ""))


def partition_questions_by_facts(
    questions: List[Dict[str, Any]],
    facts_by_key: Dict[str, EstablishedFact],
) -> Tuple[List[Tuple[Dict[str, Any], Any]], List[Dict[str, Any]]]:
    """Split a clarification round into (already-resolved, still-unresolved).

    Returns ``(resolved, unresolved)`` where ``resolved`` is a list of
    ``(question, known_answer)`` to short-circuit, and ``unresolved`` is the questions
    that genuinely require a pause. Exact key match only — deterministic.
    """
    resolved: List[Tuple[Dict[str, Any], Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for q in questions or []:
        fact = facts_by_key.get(question_key(q))
        if fact is not None:
            resolved.append((q, fact.answer))
        else:
            unresolved.append(q)
    return resolved, unresolved


@dataclass
class ClarificationDecision:
    """The deterministic outcome of checking a clarification round against settled facts.

    ``should_pause=False`` means every asked question is already answered — the run
    continues and the model is handed ``resolved_observation`` (the known answers) as the
    tool result instead of pausing. ``should_pause=True`` carries ``pause_questions`` —
    the genuinely-unresolved subset (or the full set when no facts are in play).
    """

    should_pause: bool
    pause_questions: List[Dict[str, Any]] = field(default_factory=list)
    resolved_observation: Optional[str] = None


def decide_clarification(
    question_dicts: List[Dict[str, Any]],
    facts_by_key: Dict[str, EstablishedFact],
    interrupt_count: int = 0,
    max_interrupts: int = MAX_CLARIFICATION_INTERRUPTS_PER_TURN,
) -> ClarificationDecision:
    """Decide whether a clarification round should pause, and on which questions.

    Pure and deterministic — this is the orchestrator's short-circuit logic (R4) lifted
    out so it can be tested without standing up the whole ReAct loop. Three outcomes:

      1. Facts cover *every* question → no pause; hand the model the known answers.
      2. ``interrupt_count`` has hit the hard cap → no pause; force the model to proceed
         with defaults. This is the content-free circuit-breaker that replaces the
         old token-overlap loop detector — a predictable ceiling, not a heuristic.
      3. Otherwise pause on the unresolved subset (or the full set when no facts apply).
    """
    resolved, unresolved = partition_questions_by_facts(question_dicts, facts_by_key)

    if facts_by_key and resolved and not unresolved:
        # Return the known answer(s) as a plain tool-result the model consumes and keeps
        # working from — the SAME shape a normal answer arrives in. It must NOT read like
        # a user-facing message: a chatty "these were already answered, do NOT ask again"
        # directive gets echoed verbatim to the user as the final reply (observed leak).
        # Keep it terse data + a continue nudge so the model proceeds with the task.
        answer_lines = "\n".join(
            f"{(q.get('question') or '').rstrip(' ?')}: {ans}" for q, ans in resolved
        )
        return ClarificationDecision(
            should_pause=False,
            resolved_observation=(
                answer_lines
                + "\n\n(You already have these answers — do not call ask_user_clarification "
                "for them. Continue the task now; do not repeat this note to the user.)"
            ),
        )

    if interrupt_count >= max_interrupts:
        return ClarificationDecision(
            should_pause=False,
            resolved_observation=(
                f"You have asked the user for clarification {interrupt_count} times "
                "already in a row. Do NOT ask again. Proceed now using the information "
                "provided plus sensible defaults (e.g. last 30 days for an unspecified "
                "date range) and produce the result."
            ),
        )

    pause_questions = unresolved if (facts_by_key and resolved) else question_dicts
    return ClarificationDecision(should_pause=True, pause_questions=pause_questions)


def render_established_facts_block(facts: List[EstablishedFact]) -> str:
    """Render the "already settled — do not ask again" prompt block.

    Injected into every agent's system prompt (root + sub) so settled answers are
    salient and identical across the whole dispatch tree. Empty string when there are
    no facts, so callers can unconditionally concatenate.
    """
    if not facts:
        return ""
    lines = ["# Already settled — do NOT ask the user about these again:"]
    for f in facts:
        label = f.question_text or f.key
        answer = f.answer
        if isinstance(answer, (list, tuple)):
            answer = ", ".join(str(a) for a in answer)
        lines.append(f"- {label}: {answer}")
    return "\n".join(lines)
