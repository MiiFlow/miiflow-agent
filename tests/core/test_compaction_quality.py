"""Tests for compaction quality: head-of-task preservation and the handoff
summary.

The original task statement (first USER message) is what naive oldest-first
dropping removes first, and the summary that replaces dropped history used to
be capped at a flat 500 tokens regardless of how much it replaced. Compaction
now pins the task statement verbatim and writes a scaled handoff note.
"""

import asyncio
from types import SimpleNamespace

import pytest

from miiflow_agent.core.context_compression import (
    _HEAD_USER_MESSAGE_MAX_CHARS,
    _SUMMARY_MAX_TOKENS,
    _SUMMARY_MIN_TOKENS,
    _SUMMARY_RETRY_TOKENS,
    CompressionStrategy,
    ContextCompressor,
)
from miiflow_agent.core.message import Message, MessageRole

TASK = "Analyze the Q3 ad spend for account act_12345 and prepare a report"


def _history(n_filler=30, filler_size=400):
    msgs = [
        Message(role=MessageRole.SYSTEM, content="System prompt"),
        Message(role=MessageRole.USER, content=TASK),
    ]
    for i in range(n_filler):
        msgs.append(Message(role=MessageRole.ASSISTANT, content=f"step {i} " * (filler_size // 8)))
        msgs.append(Message(role=MessageRole.USER, content=f"ok {i} " * (filler_size // 8)))
    return msgs


class _SummarizerClient:
    """Captures the summarization call and returns a canned handoff note.

    `responses` (optional) is a queue of (content, finish_reason) pairs so a
    test can script a truncated first answer followed by a full one.
    """

    def __init__(self, responses=None):
        self.calls = []
        self._responses = list(responses or [])

    async def achat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self._responses:
            content, finish = self._responses.pop(0)
        else:
            content, finish = "HANDOFF NOTE", "end_turn"
        return SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish
        )


class TestTruncateHeadPreservation:
    async def test_task_statement_survives_truncation(self):
        compressor = ContextCompressor(
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.TRUNCATE,
        )
        result = await compressor.compress_if_needed(_history(), preserve_recent=4)

        assert result.was_compressed
        contents = [
            m.content for m in result.messages if isinstance(m.content, str)
        ]
        assert any(TASK in c for c in contents)
        # The task statement comes before the compaction marker.
        task_idx = next(i for i, c in enumerate(contents) if TASK in c)
        marker_idx = next(
            i for i, c in enumerate(contents) if "Context compressed" in c
        )
        assert task_idx < marker_idx

    async def test_task_statement_not_duplicated_when_recent(self):
        compressor = ContextCompressor(
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.TRUNCATE,
        )
        # Small history: the head user message lands inside the kept tail.
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content=TASK),
            Message(role=MessageRole.ASSISTANT, content="working " * 600),
        ]
        result = await compressor.compress_if_needed(msgs, preserve_recent=4)
        occurrences = sum(
            1
            for m in result.messages
            if isinstance(m.content, str) and TASK in m.content
        )
        assert occurrences == 1

    async def test_huge_task_statement_is_clamped(self):
        compressor = ContextCompressor(
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.TRUNCATE,
        )
        msgs = _history()
        msgs[1] = Message(role=MessageRole.USER, content="T" * 50_000)
        result = await compressor.compress_if_needed(msgs, preserve_recent=4)

        head = next(
            m
            for m in result.messages
            if isinstance(m.content, str) and m.content.startswith("T" * 100)
        )
        assert len(head.content) < _HEAD_USER_MESSAGE_MAX_CHARS + 500


class TestHandoffSummary:
    async def test_summary_is_a_handoff_note_with_scaled_budget(self):
        client = _SummarizerClient()
        compressor = ContextCompressor(
            client=client,
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.SUMMARIZE,
        )
        result = await compressor.compress_if_needed(
            _history(n_filler=140, filler_size=1600), preserve_recent=4
        )

        assert result.was_compressed
        assert len(client.calls) == 1
        call = client.calls[0]
        prompt = call["messages"][0].content
        assert "handoff note" in prompt
        assert "SETTLED" in prompt and "OPEN" in prompt
        # Budget scales with dropped volume instead of the old flat cap.
        assert _SUMMARY_MIN_TOKENS <= call["max_tokens"] <= _SUMMARY_MAX_TOKENS
        assert call["max_tokens"] > _SUMMARY_MIN_TOKENS
        # The summariser call is bare: no agent tools, no MCP connectors, and
        # thinking off — otherwise adaptive thinking eats the token budget on
        # Claude 5 models and the note comes back empty.
        assert call["_formatted_tools"] == []
        assert call["mcp_servers"] is None
        assert call["thinking_disabled"] is True

        # The note lands in a marker that frames it as established work.
        note_msg = next(
            m
            for m in result.messages
            if isinstance(m.content, str) and "HANDOFF NOTE" in m.content
        )
        assert "verify against it" in note_msg.content

    async def test_task_statement_kept_verbatim_not_summarized(self):
        client = _SummarizerClient()
        compressor = ContextCompressor(
            client=client,
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.SUMMARIZE,
        )
        result = await compressor.compress_if_needed(_history(), preserve_recent=4)

        # Preserved verbatim in the output...
        assert any(
            isinstance(m.content, str) and m.content == TASK
            for m in result.messages
        )
        # ...and excluded from the summarizer's input.
        prompt = client.calls[0]["messages"][0].content
        assert TASK not in prompt

    async def test_tool_call_names_surface_in_summarizer_input(self):
        client = _SummarizerClient()
        compressor = ContextCompressor(
            client=client,
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.SUMMARIZE,
        )
        msgs = _history(n_filler=20)
        msgs.insert(
            5,
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {"id": "t1", "type": "function", "function": {"name": "meta_ads_insights", "arguments": "{}"}}
                ],
            ),
        )
        msgs.insert(
            6,
            Message(role=MessageRole.TOOL, content="spend: $1234", tool_call_id="t1"),
        )
        await compressor.compress_if_needed(msgs, preserve_recent=4)

        prompt = client.calls[0]["messages"][0].content
        assert "meta_ads_insights" in prompt

    async def test_summarizer_failure_falls_back_gracefully(self):
        class _FailingClient:
            async def achat(self, messages, **kwargs):
                raise RuntimeError("summarizer down")

        compressor = ContextCompressor(
            client=_FailingClient(),
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.SUMMARIZE,
        )
        result = await compressor.compress_if_needed(_history(), preserve_recent=4)
        assert result.was_compressed
        assert any(
            isinstance(m.content, str) and "details unavailable" in m.content
            for m in result.messages
        )


class TestHandoffSummaryTruncation:
    async def test_empty_note_at_max_tokens_is_retried_with_larger_budget(self):
        # First answer: thinking consumed the whole budget, no text at all.
        client = _SummarizerClient(
            responses=[("", "max_tokens"), ("REAL HANDOFF NOTE", "end_turn")]
        )
        compressor = ContextCompressor(
            client=client,
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.SUMMARIZE,
        )
        result = await compressor.compress_if_needed(_history(), preserve_recent=4)

        assert len(client.calls) == 2
        assert client.calls[1]["max_tokens"] == _SUMMARY_RETRY_TOKENS
        assert any(
            isinstance(m.content, str) and "REAL HANDOFF NOTE" in m.content
            for m in result.messages
        )
        # And never the placeholder — an empty note must not be accepted.
        assert not any(
            isinstance(m.content, str) and "context unavailable" in m.content
            for m in result.messages
        )

    async def test_truncated_but_nonempty_note_is_kept_without_retry(self):
        client = _SummarizerClient(responses=[("PARTIAL NOTE", "max_tokens")])
        compressor = ContextCompressor(
            client=client,
            max_context_tokens=2000,
            compression_threshold=0.5,
            strategy=CompressionStrategy.SUMMARIZE,
        )
        result = await compressor.compress_if_needed(_history(), preserve_recent=4)

        assert len(client.calls) == 1
        assert any(
            isinstance(m.content, str) and "PARTIAL NOTE" in m.content
            for m in result.messages
        )


class TestNothingToCompact:
    """A forced compaction on a short history must be a no-op, not a note.

    2026-08-18 (thread_ZFuH2vZySdM7RzDMEIbsokgv): the recovery ladder forced a
    compaction on a 6-message run. Everything sat inside the preserved-recent
    window, so the summariser was handed ZERO messages, came back with
    "the conversation history is empty… wait for the user's first message",
    and that note was inserted into the live conversation as compressed
    history — reported as a successful compaction.
    """

    def _short_history(self):
        return [
            Message(role=MessageRole.SYSTEM, content="System prompt"),
            Message(role=MessageRole.USER, content=TASK),
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{"id": "t1", "function": {"name": "read_memory", "arguments": "{}"}}],
            ),
            Message(role=MessageRole.TOOL, content="# plan.xlsx (binary)", tool_call_id="t1"),
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{"id": "t2", "function": {"name": "view_media", "arguments": "{}"}}],
            ),
            Message(role=MessageRole.TOOL, content="Injected 1 media item", tool_call_id="t2"),
        ]

    async def test_summarize_with_nothing_older_than_the_window_is_a_noop(self):
        client = _SummarizerClient()
        compressor = ContextCompressor(
            client=client,
            # Tiny budget so the size check triggers, exactly like force=True.
            max_context_tokens=1,
            compression_threshold=1.0,
            strategy=CompressionStrategy.SUMMARIZE,
        )
        history = self._short_history()

        result = await compressor.compress_if_needed(history, preserve_recent=4)

        assert client.calls == [], "no summariser call over an empty transcript"
        assert result.was_compressed is False
        assert result.messages is history
        assert result.compressed_count == len(history)

    async def test_auto_strategy_reports_the_noop_too(self):
        client = _SummarizerClient()
        compressor = ContextCompressor(
            client=client,
            max_context_tokens=1,
            compression_threshold=1.0,
            strategy=CompressionStrategy.AUTO,
        )
        result = await compressor.compress_if_needed(self._short_history(), preserve_recent=4)
        assert result.was_compressed is False
        assert client.calls == []
