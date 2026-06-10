"""Tests for the deterministic established-facts logic (R4) + interrupt primitive.

These pin the property that replaces the old 50%-token-overlap loop detector: question
identity is an *exact* deterministic key, the short-circuit is an exact lookup, and the
injected facts block is stable. No fuzzy matching, no thresholds.
"""

from miiflow_agent.core.checkpoint import EstablishedFact, PendingInterrupt
from miiflow_agent.core.interrupt import (
    GraphInterrupt,
    decide_clarification,
    mint_interrupt_id,
    partition_questions_by_facts,
    question_key,
    render_established_facts_block,
)


def test_question_key_prefers_model_slug_then_text():
    assert question_key({"key": "daily_budget", "question": "What daily budget?"}) == "daily_budget"
    # No explicit key → canonical slug of the text (deterministic, not fuzzy).
    assert question_key({"question": "What daily budget?"}) == "what_daily_budget"
    # Same text always maps to the same key.
    assert question_key({"question": "What daily budget?"}) == question_key(
        {"question": "what  DAILY   budget!!!"}
    )
    # A model key survives rewording the prose.
    assert question_key({"key": "daily_budget", "question": "budget per day?"}) == "daily_budget"


def test_partition_short_circuits_resolved_questions_only():
    facts = {
        "daily_budget": EstablishedFact(key="daily_budget", answer="$50/day"),
    }
    questions = [
        {"key": "daily_budget", "question": "What daily budget?"},
        {"key": "geo", "question": "Which geos?"},
    ]
    resolved, unresolved = partition_questions_by_facts(questions, facts)

    assert len(resolved) == 1
    assert resolved[0][1] == "$50/day"  # known answer surfaced
    assert len(unresolved) == 1
    assert unresolved[0]["key"] == "geo"  # genuinely-new question still pauses


def test_partition_all_resolved_means_no_pause():
    facts = {
        "daily_budget": EstablishedFact(key="daily_budget", answer="$50/day"),
        "geo": EstablishedFact(key="geo", answer="California"),
    }
    questions = [{"key": "daily_budget"}, {"key": "geo"}]
    resolved, unresolved = partition_questions_by_facts(questions, facts)
    assert len(resolved) == 2 and unresolved == []


def test_render_facts_block_is_stable_and_lists_answers():
    facts = [
        EstablishedFact(key="daily_budget", answer="$50/day", question_text="What daily budget?"),
        EstablishedFact(key="geos", answer=["CA", "TX"], question_text="Which geos?"),
    ]
    block = render_established_facts_block(facts)
    assert "do NOT ask" in block
    assert "What daily budget?: $50/day" in block
    assert "Which geos?: CA, TX" in block  # list answers joined deterministically
    # Empty input → empty string so callers can unconditionally concatenate.
    assert render_established_facts_block([]) == ""


def test_decide_clarification_short_circuits_when_all_resolved():
    facts = {
        "daily_budget": EstablishedFact(key="daily_budget", answer="$50/day", question_text="What daily budget?"),
    }
    decision = decide_clarification([{"key": "daily_budget", "question": "What daily budget?"}], facts)
    assert decision.should_pause is False
    # The observation carries the answer (so the model continues) but must NOT contain
    # the leakable "these were already answered, do NOT ask again. Proceed using:" prose
    # the model parroted to the user.
    assert "What daily budget: $50/day" in decision.resolved_observation
    assert "Proceed using" not in decision.resolved_observation
    assert "already answered earlier in this conversation" not in decision.resolved_observation
    assert "Continue the task now" in decision.resolved_observation


def test_decide_clarification_pauses_on_unresolved_subset():
    facts = {"daily_budget": EstablishedFact(key="daily_budget", answer="$50/day")}
    decision = decide_clarification(
        [
            {"key": "daily_budget", "question": "What daily budget?"},
            {"key": "geo", "question": "Which geos?"},
        ],
        facts,
    )
    assert decision.should_pause is True
    # Only the genuinely-new question is surfaced to the user.
    assert [q["key"] for q in decision.pause_questions] == ["geo"]


def test_decide_clarification_no_facts_pauses_on_full_set_legacy():
    questions = [{"key": "a", "question": "A?"}, {"key": "b", "question": "B?"}]
    decision = decide_clarification(questions, {})
    assert decision.should_pause is True
    assert decision.pause_questions == questions  # unchanged legacy behaviour


def test_decide_clarification_circuit_breaker_forces_proceed():
    # Content-free hard cap: after enough consecutive rounds, stop pausing even
    # when the question is genuinely unresolved (no fact to short-circuit on).
    questions = [{"key": "geo", "question": "Which geos?"}]
    decision = decide_clarification(questions, {}, interrupt_count=5, max_interrupts=5)
    assert decision.should_pause is False
    assert "Do NOT ask again" in decision.resolved_observation


def test_decide_clarification_under_cap_still_pauses():
    questions = [{"key": "geo", "question": "Which geos?"}]
    decision = decide_clarification(questions, {}, interrupt_count=2, max_interrupts=5)
    assert decision.should_pause is True


def test_mint_interrupt_id_prefers_tool_call_id():
    assert mint_interrupt_id("clarification", "tc_9") == "int_clarification_tc_9"
    # Without a tool_call_id it is still unique + self-describing.
    other = mint_interrupt_id("tool_approval")
    assert other.startswith("int_tool_approval_") and len(other) > len("int_tool_approval_")


def test_graph_interrupt_carries_pending_interrupt():
    pi = PendingInterrupt(interrupt_id="int_1", kind="clarification", raised_by_path=["root"])
    err = GraphInterrupt(pi)
    assert err.interrupt is pi
    assert "int_1" in str(err)
