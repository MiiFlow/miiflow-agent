"""A media block the provider cannot decode poisons every resend — repair it.

Production fingerprint (2026-08-18, thread_ZFuH2vZySdM7RzDMEIbsokgv): view_media
put an xlsx into a tool_result as an image block; Anthropic answered
``400 messages.1.content.5.image.source.base64.data: The file format is
invalid or unsupported``; the recovery ladder re-sent the identical history
four times (guidance, compaction, tool exclusion) and the run halted on
"Too many consecutive errors (3)". Nothing on the ladder edits history.

Drives the REAL ``_repair_rejected_request`` on a stand-in self, same pattern
as test_answer_after_halt.
"""

import asyncio
from types import SimpleNamespace

from miiflow_agent.core.message import (
    DocumentBlock,
    ImageBlock,
    Message,
    MessageRole,
    TextBlock,
    VideoBlock,
)
from miiflow_agent.core.react.message_repair import (
    is_unsupported_media_error,
    strip_unprocessable_media,
)
from miiflow_agent.core.react.models import ReActStep
from miiflow_agent.core.react.orchestrator import ReActOrchestrator

ANTHROPIC_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'messages.1.content.5.image.source.base64.data: The file format is "
    "invalid or unsupported'}, 'request_id': 'req_011CeAshG1WVYkKhsNnpPV4u'}"
)


def _call(call_id, name="view_media"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _poisoned_history():
    return [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="check the daily pacing file uploaded"),
        Message(role=MessageRole.ASSISTANT, content="", tool_calls=[_call("t1", "read_file")]),
        Message(role=MessageRole.TOOL, content="# plan.xlsx (binary)", tool_call_id="t1"),
        Message(role=MessageRole.ASSISTANT, content="", tool_calls=[_call("t2")]),
        Message(
            role=MessageRole.TOOL,
            content=[
                TextBlock(text="Injected 1 media item(s) for visual analysis"),
                ImageBlock(image_url="https://x.supabase.co/s3/plan.xlsx?X-Amz-Signature=abc"),
                TextBlock(text="Focus for visual analysis: pacing"),
            ],
            tool_call_id="t2",
        ),
    ]


class TestDetection:
    def test_anthropic_wording(self):
        assert is_unsupported_media_error(Exception(ANTHROPIC_400))
        assert is_unsupported_media_error(
            Exception("400 messages.3.content.0.image.source.url: Could not fetch the URL")
        )
        assert is_unsupported_media_error(
            Exception("400 messages.2.content.1.image: image exceeds 5 MB maximum")
        )

    def test_openai_and_gemini_wordings(self):
        assert is_unsupported_media_error(
            Exception("Error code: 400 - {'error': {'code': 'invalid_image_format'}}")
        )
        assert is_unsupported_media_error(
            Exception("400 Unsupported MIME type: application/vnd.ms-excel")
        )

    def test_unrelated_errors_do_not_match(self):
        assert not is_unsupported_media_error(Exception("prompt is too long"))
        assert not is_unsupported_media_error(Exception("credit balance is too low"))
        assert not is_unsupported_media_error(
            Exception("tool_use ids were found without tool_result blocks")
        )
        assert not is_unsupported_media_error(None)


class TestStrip:
    def test_replaces_media_blocks_of_the_last_media_message(self):
        messages = _poisoned_history()
        repaired, anomalies = strip_unprocessable_media(messages)

        assert len(anomalies) == 1
        assert "image" in anomalies[0] and "plan.xlsx" in anomalies[0]
        # Input untouched.
        assert isinstance(messages[5].content[1], ImageBlock)
        # Same shape: same length, same tool pairing, only that message changed.
        assert len(repaired) == len(messages)
        assert repaired[:5] == messages[:5]
        fixed = repaired[5]
        assert fixed.role == MessageRole.TOOL and fixed.tool_call_id == "t2"
        assert not any(isinstance(b, ImageBlock) for b in fixed.content)
        note = fixed.content[1]
        assert isinstance(note, TextBlock)
        assert "could not process" in note.text
        assert "read_file" in note.text
        # The text neighbours survive verbatim.
        assert fixed.content[0].text.startswith("Injected 1 media item")
        assert fixed.content[2].text.startswith("Focus for visual analysis")

    def test_targets_the_most_recent_media_message_only(self):
        messages = _poisoned_history()
        messages.append(Message(role=MessageRole.ASSISTANT, content="", tool_calls=[_call("t3")]))
        messages.append(
            Message(
                role=MessageRole.TOOL,
                content=[VideoBlock(video_url="https://cdn/clip.mp4")],
                tool_call_id="t3",
            )
        )
        repaired, anomalies = strip_unprocessable_media(messages)
        assert len(anomalies) == 1 and "video" in anomalies[0]
        assert isinstance(repaired[5].content[1], ImageBlock)  # earlier one kept
        assert isinstance(repaired[7].content[0], TextBlock)

    def test_data_uri_and_document_sources_are_summarised(self):
        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    ImageBlock(image_url="data:application/octet-stream;base64,AAAA"),
                    DocumentBlock(document_url="https://cdn/brief.docx", document_type="docx"),
                ],
            )
        ]
        repaired, anomalies = strip_unprocessable_media(messages)
        assert len(anomalies) == 2
        assert "data:application/octet-stream;base64" in anomalies[0]
        assert "AAAA" not in anomalies[0]
        assert "brief.docx" in anomalies[1]
        assert all(isinstance(b, TextBlock) for b in repaired[0].content)

    def test_no_media_means_no_change(self):
        messages = [
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.ASSISTANT, content=[TextBlock(text="hello")]),
        ]
        repaired, anomalies = strip_unprocessable_media(messages)
        assert anomalies == []
        assert repaired is messages


def _repair(error, messages, *, media_repairs=0, structural_attempted=False):
    step = ReActStep(step_number=3, thought="")
    step.error = f"Step execution failed: {error}"
    state = SimpleNamespace(
        media_repairs=media_repairs,
        structural_repair_attempted=structural_attempted,
    )
    context = SimpleNamespace(messages=messages)
    orch = SimpleNamespace(MAX_MEDIA_REPAIRS=ReActOrchestrator.MAX_MEDIA_REPAIRS)
    resend = asyncio.run(
        ReActOrchestrator._repair_rejected_request(orch, step, context, state)
    )
    return resend, state, context


class TestRepairRejectedRequest:
    def test_media_400_strips_and_resends(self):
        resend, state, context = _repair(ANTHROPIC_400, _poisoned_history())
        assert resend is True
        assert state.media_repairs == 1
        assert not any(
            isinstance(b, ImageBlock)
            for m in context.messages
            if isinstance(m.content, list)
            for b in m.content
        )

    def test_media_repairs_are_capped(self):
        history = _poisoned_history()
        resend, state, context = _repair(
            ANTHROPIC_400, history, media_repairs=ReActOrchestrator.MAX_MEDIA_REPAIRS
        )
        assert resend is False
        assert context.messages is history  # untouched: ladder's turn now

    def test_media_400_with_nothing_to_strip_falls_through(self):
        history = [Message(role=MessageRole.USER, content="plain")]
        resend, state, context = _repair(ANTHROPIC_400, history)
        assert resend is False
        assert state.media_repairs == 0

    def test_structural_400_repairs_once(self):
        history = [
            Message(role=MessageRole.USER, content="go"),
            Message(role=MessageRole.ASSISTANT, content="", tool_calls=[_call("a", "x")]),
            Message(role=MessageRole.ASSISTANT, content="continuing"),
        ]
        err = "400 messages.1: tool_use ids were found without tool_result blocks: a"
        resend, state, context = _repair(err, history)
        assert resend is True
        assert state.structural_repair_attempted is True
        assert context.messages[2].role == MessageRole.TOOL
        # Second time round it does not loop.
        resend2, _, _ = _repair(err, context.messages, structural_attempted=True)
        assert resend2 is False

    def test_other_errors_are_not_repaired(self):
        resend, state, context = _repair("rate_limit_error", _poisoned_history())
        assert resend is False
        assert isinstance(context.messages[5].content[1], ImageBlock)
