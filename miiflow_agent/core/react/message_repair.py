"""Repair of structurally invalid message histories.

Providers enforce hard invariants on the wire format — most importantly that
every assistant ``tool_calls`` entry is answered by a matching TOOL message,
and that no TOOL message arrives without its ``tool_use``. A history can
violate these after an interrupted run, a crashed tool batch, or an external
mutation of the thread, and the provider then rejects *every* subsequent
request with a 400: the session is permanently stuck.

Rather than papering over each producer bug at its source (they keep coming —
several incident docstrings in this package cite exactly these 400s), the
orchestrator gets one recovery path: when a provider rejects the request with
a structural error, repair the history once and resend. Every repair is
reported, never silent — a repair that fires repeatedly is a producer bug to
fix, and the log line is how it gets found. Patterned after kimi-code's
strict projection + ProjectionAnomaly design.
"""

import logging
from typing import List, Optional, Tuple

from ..message import Message, MessageRole

logger = logging.getLogger(__name__)


# Known wordings of structural 400s across providers. Sniffing strings is the
# portable lowest common denominator (each provider raises its own exception
# type); a false negative just means the error propagates as before.
_STRUCTURAL_HINTS = (
    # Anthropic
    "tool_use ids were found without tool_result",
    "unexpected tool_use_id",
    "tool_result block(s) provided when previous message",
    "must be followed by a user message with a corresponding tool_result",
    # Anthropic, duplicate answer: `messages.N.content.M: each tool_use must
    # have a single result. Found multiple tool_result blocks with id ...`.
    # Seen 2026-08-11 on a web thread whose PERSISTED history carried two
    # results for one call — every later turn 400'd until compaction happened
    # to drop the pair, because this wording was not recognised as structural.
    "each tool_use must have a single result",
    "found multiple `tool_result` blocks",
    "found multiple tool_result blocks",
    # OpenAI
    "must be followed by tool messages",
    "did not have response messages",
    "must be a response to a preceeding message with 'tool_calls'",
    "must be a response to a preceding message with 'tool_calls'",
    "no tool call in previous message",
    # Gemini
    "function response parts must come immediately after",
    "please ensure that function call turn comes immediately after",
)

_SYNTHESIZED_RESULT = (
    "Tool execution was interrupted before its result was recorded. "
    "Do not assume the tool completed successfully; re-run it if its "
    "result matters."
)


def is_structural_message_error(error: BaseException) -> bool:
    """Heuristically classify ``error`` as a structural-history 400."""
    if error is None:
        return False
    text = (str(error) or "").lower()
    return any(hint in text for hint in _STRUCTURAL_HINTS)


def _call_ids(message: Message) -> List[str]:
    ids = []
    for call in message.tool_calls or []:
        if isinstance(call, dict) and call.get("id"):
            ids.append(str(call["id"]))
    return ids


def repair_tool_pairing(
    messages: List[Message],
) -> Tuple[List[Message], List[str]]:
    """Restore the tool_use ↔ tool_result pairing invariant.

    Two repairs, mirroring the two 400s providers actually raise:

    * an orphan TOOL message (its tool_use is gone) is dropped;
    * a DUPLICATE TOOL message (its tool_use was already answered) is dropped
      — providers accept exactly one result per call;
    * an unanswered assistant tool_call gets a synthesized TOOL result saying
      the execution was interrupted, so the model knows not to trust it.

    Returns ``(repaired, anomalies)`` — ``anomalies`` is a list of
    human-readable descriptions, empty when the history was already valid.
    The input list is never mutated.
    """
    repaired: List[Message] = []
    anomalies: List[str] = []

    pending: List[str] = []  # unanswered call ids from the latest assistant
    answered: set = set()  # call ids already answered in this answer window

    def close_pending() -> None:
        for call_id in pending:
            anomalies.append(f"synthesized missing tool_result for {call_id}")
            repaired.append(
                Message(
                    role=MessageRole.TOOL,
                    content=_SYNTHESIZED_RESULT,
                    tool_call_id=call_id,
                )
            )
        pending.clear()
        answered.clear()

    for message in messages:
        if message.role == MessageRole.TOOL:
            call_id: Optional[str] = message.tool_call_id
            if call_id is not None and call_id in pending:
                pending.remove(call_id)
                answered.add(call_id)
                repaired.append(message)
            elif call_id is not None and call_id in answered:
                anomalies.append(
                    f"dropped duplicate tool_result {call_id!r} (already answered)"
                )
            else:
                anomalies.append(
                    f"dropped orphan tool_result {call_id!r} (no matching tool_use)"
                )
            continue

        # Any non-TOOL message ends the answer window of the previous
        # assistant turn: whatever is still unanswered gets synthesized
        # BEFORE this message so results stay adjacent to their calls.
        close_pending()

        if message.role == MessageRole.ASSISTANT and message.tool_calls:
            pending.extend(_call_ids(message))
        repaired.append(message)

    close_pending()
    return repaired, anomalies


# ── Unprocessable media ─────────────────────────────────────────────────────
#
# The second way a history poisons every subsequent request: a media block the
# provider cannot decode. Seen 2026-08-18 (thread_ZFuH2vZySdM7RzDMEIbsokgv):
# view_media pushed an xlsx into a tool_result as an image, Anthropic answered
# `400 messages.1.content.5.image.source.base64.data: The file format is
# invalid or unsupported`, and the recovery ladder re-sent the identical block
# four times (guidance, compaction, tool exclusion) before halting. Nothing on
# the ladder edits the history; this does — the offending block is replaced by
# a text note the model can act on, and the request goes out again.

_UNSUPPORTED_MEDIA_HINTS = (
    # Anthropic
    "image.source.base64.data",
    "image.source.url",
    "document.source.base64.data",
    "document.source.url",
    "file format is invalid or unsupported",
    "could not process image",
    "image exceeds",  # size limits: 5MB / 8000px
    "could not fetch",  # URL source the provider cannot download
    "unsupported image type",
    "invalid image",
    # OpenAI
    "invalid_image_format",
    "invalid_image_url",
    "unsupported image",
    "image_parse_error",
    "error while downloading",
    # Gemini
    "unsupported mime type",
    "unable to process input image",
)

_REMOVED_MEDIA_NOTE = (
    "[{label} removed: the model provider could not process it "
    "({source}). It is not a viewable image/video for this model. If it is a "
    "document (spreadsheet, PDF, Word), read it as text with read_file on its "
    "workspace path instead of view_media.]"
)


def is_unsupported_media_error(error: BaseException) -> bool:
    """True when the provider rejected a media block in the request."""
    if error is None:
        return False
    text = (str(error) or "").lower()
    return any(hint in text for hint in _UNSUPPORTED_MEDIA_HINTS)


def _is_media_block(block: object) -> bool:
    return type(block).__name__ in ("ImageBlock", "VideoBlock", "DocumentBlock")


def _media_source(block: object) -> str:
    for attr in ("image_url", "video_url", "document_url"):
        value = getattr(block, attr, None)
        if value:
            if value.startswith("data:"):
                # "data:application/octet-stream;base64" — the mimetype is the
                # useful part; the payload is not.
                return value.split(",", 1)[0][:60]
            return value.split("?", 1)[0][-160:]
    return "unknown source"


def strip_unprocessable_media(
    messages: List[Message],
) -> Tuple[List[Message], List[str]]:
    """Replace the media blocks of the LAST media-bearing message with a note.

    The provider's error names a block by wire index, which does not map back
    to ``messages`` (system prompt lifted out, tool results re-nested), so the
    repair targets the most recent message that carries any image/video/
    document block: media enters history one tool call at a time, and the
    request that first failed is the one right after the poison arrived. If a
    second bad block sits further back, the next rejection strips that one —
    the caller bounds how many times this may run.

    Returns ``(repaired, anomalies)``; ``anomalies`` is empty (and ``repaired``
    is the input list) when no message holds a media block. The input is never
    mutated.
    """
    from ..message import TextBlock

    target = None
    for index in range(len(messages) - 1, -1, -1):
        content = messages[index].content
        if isinstance(content, list) and any(_is_media_block(b) for b in content):
            target = index
            break
    if target is None:
        return messages, []

    anomalies: List[str] = []
    original = messages[target]
    new_content = []
    for block in original.content:
        if _is_media_block(block):
            label = type(block).__name__.replace("Block", "").lower()
            source = _media_source(block)
            anomalies.append(f"removed {label} block ({source})")
            new_content.append(
                TextBlock(text=_REMOVED_MEDIA_NOTE.format(label=label, source=source))
            )
        else:
            new_content.append(block)

    repaired_message = Message(
        role=original.role,
        content=new_content,
        name=original.name,
        tool_calls=original.tool_calls,
        tool_call_id=original.tool_call_id,
        timestamp=original.timestamp,
        metadata=original.metadata,
    )
    repaired = list(messages)
    repaired[target] = repaired_message
    return repaired, anomalies
