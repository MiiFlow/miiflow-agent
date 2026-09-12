"""The clarification tool must never produce a question-less clarification request.

A request with no questions renders as an empty panel, so the user is offered no way
to answer and the thread parks forever. The tool therefore fails the call instead,
with a message the model can act on.

The malformed payloads here are the real ones from production: on 2026-09-06 and
2026-09-12 the model sent `questions` as a JSON *string* whose content was itself
invalid JSON, with a `"multi_select": false` pair sitting as a bare array element
between two question objects.
"""

from miiflow_agent.core.tools.clarification import (
    CLARIFICATION_MARKER,
    _normalize_questions,
    ask_user_clarification,
)
from miiflow_agent.core.tools.schemas import ToolFailure


# The exact shape observed in thread_0rylPSJaVYR34FAEAXicwSPC, trimmed.
MALFORMED_PRODUCTION_PAYLOAD = (
    '[{"key":"doc_purpose","question":"What should this agent do with your Google '
    'Doc?","options":["Summarize it","Watch it for changes"]},"multi_select":false,'
    '{"key":"cadence","question":"How often should the agent run?","options":'
    '["Daily","Weekly"]}]'
)


def test_the_malformed_production_payload_fails_instead_of_asking_nothing():
    result = ask_user_clarification(questions=MALFORMED_PRODUCTION_PAYLOAD, context="")

    assert isinstance(result, ToolFailure)
    assert result.error_type == "invalid_questions"
    # The model is told the actual defect, not just that something was wrong.
    assert "not valid JSON" in result.error


def test_a_questions_array_sent_as_valid_json_text_is_repaired():
    # Models routinely stringify structured params. When the text parses, that is a
    # repair, not a rejection — the user still gets asked.
    result = ask_user_clarification(
        questions='[{"key":"geo","question":"Which geo?","options":["US","CA"]}]'
    )

    assert result["marker"] == CLARIFICATION_MARKER
    assert [q["key"] for q in result["questions"]] == ["geo"]


def test_an_options_less_question_fails_and_says_to_ask_in_prose():
    result = ask_user_clarification(
        questions=[{"key": "goal", "question": "What are you trying to achieve?"}]
    )

    assert isinstance(result, ToolFailure)
    assert "open-ended" in result.error


def test_an_empty_list_fails():
    result = ask_user_clarification(questions=[])

    assert isinstance(result, ToolFailure)


def test_a_well_formed_call_is_unaffected():
    result = ask_user_clarification(
        questions=[
            {"key": "geo", "question": "Which geo?", "options": ["US", "CA"]},
            {
                "key": "channels",
                "question": "Which channels?",
                "options": ["Search", "Social"],
                "multi_select": True,
            },
        ],
        context="Sizing the launch.",
    )

    assert result["marker"] == CLARIFICATION_MARKER
    assert len(result["questions"]) == 2
    assert result["questions"][1]["multi_select"] is True
    assert result["context"] == "Sizing the launch."


def test_partial_payloads_keep_the_usable_questions_and_report_the_rest():
    # One bad entry must not cost the user the questions that were fine.
    good = {"key": "geo", "question": "Which geo?", "options": ["US"]}
    normalized, problems = _normalize_questions([good, "multi_select", {"question": ""}])

    assert [q.key for q in normalized] == ["geo"]
    assert len(problems) == 2


def test_a_string_is_never_walked_character_by_character():
    # The original defect: iterating the string yielded characters, every one was
    # dropped as "not a dict", and the empty result was presented as a success.
    normalized, problems = _normalize_questions(MALFORMED_PRODUCTION_PAYLOAD)

    assert normalized == []
    # One problem naming the payload — not one per character.
    assert len(problems) == 1
