"""Context compression for managing conversation history within token limits.

Provides multiple strategies for compressing message history when it approaches
context window limits, inspired by Claude Code's multi-level compaction system.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .message import Message, MessageRole
from .streaming import TRUNCATED_FINISH_REASONS

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Strategy for compressing message history."""

    NONE = "none"  # No compression
    TRUNCATE = "truncate"  # Drop oldest messages (keep system + recent)
    SUMMARIZE = "summarize"  # LLM-summarize old messages
    AUTO = "auto"  # Summarize if over threshold, truncate as fallback


# Approximate tokens per character (conservative estimate for mixed content)
_CHARS_PER_TOKEN = 4


def _estimate_message_tokens(message: Message) -> int:
    """Estimate token count for a single message."""
    content = message.content or ""
    # Add overhead for role, formatting
    overhead = 4
    return len(content) // _CHARS_PER_TOKEN + overhead


def _estimate_tokens(messages: List[Message]) -> int:
    """Estimate total token count for a list of messages."""
    return sum(_estimate_message_tokens(m) for m in messages)


# Hard cap on a single message's string content during compression. A single
# oversized tool result (e.g. a raw google_ads_query / meta_ads_insights dump)
# can blow the whole request past the model's context window on its own — and
# because it usually lands in the preserved-recent window, dropping *other*
# messages never shrinks it. When we're already compressing, clamp any one
# message to this many characters (~50k tokens) so the request can fit.
_MAX_SINGLE_MESSAGE_CHARS = 200_000

# The first USER message is the original task statement — the one thing the
# model must never lose sight of, and exactly what naive oldest-first dropping
# removes first. Both strategies preserve it verbatim (clamped to ~2k tokens)
# ahead of the compaction marker/summary.
_HEAD_USER_MESSAGE_MAX_CHARS = 8_000

# Per-message clip when formatting history for the summarizer. Large enough to
# keep a tool result's shape and key numbers, small enough that a long run
# still fits in the summarizer's own context.
_SUMMARY_INPUT_CLIP_CHARS = 2_000

# Summary output budget scales with how much is being replaced: a flat cap
# (formerly 500 tokens) meant a 100-message run compacted into the same space
# as a 10-message one, discarding almost everything that made the run useful.
#
# The floor is 2,000, not 500, because `max_tokens` is a hard cap on thinking
# PLUS text: on the thinking-by-default Claude 5 models the 500-token floor was
# consumed entirely by the thinking block — production traces (2026-08) showed
# 20 of 24 handoff notes returned `stop_reason=max_tokens` with an empty text
# body, so every compacted run continued from a blank note. Thinking is now
# disabled for this call where the API allows it, and the floor gives real
# headroom where it does not (Fable 5).
_SUMMARY_MIN_TOKENS = 2_000
# The ceiling is deliberately generous: prod (2026-08) showed essentially
# every compaction finishing at the old 6,000-token cap — the prompt asks for
# verbatim identifiers across the whole dropped history, so a capped note
# ends mid-sentence and the tail of the handoff is lost. Compaction fires a
# handful of times per DAY fleet-wide and output tokens are only billed as
# generated, so a higher ceiling costs nothing unless the model actually has
# more to say — which is exactly the content the cap was discarding.
_SUMMARY_MAX_TOKENS = 12_000
# One retry with this budget when the note comes back EMPTY at `max_tokens`
# (thinking consumed the whole budget on models where it can't be disabled) —
# a truncated note is a partial handoff, but no note at all is a hard reset.
_SUMMARY_RETRY_TOKENS = 24_000


# Provider spellings of "output was cut off by max_tokens" — one shared set,
# so a provider added to the loop is added here too.
_TRUNCATED_FINISH_REASONS = TRUNCATED_FINISH_REASONS


def _response_text(response: Any) -> str:
    """The text of a ChatResponse, tolerating list-of-blocks content."""
    content = getattr(getattr(response, "message", None), "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return ""


def _head_user_message(messages: List[Message]) -> Optional[Message]:
    """The first USER message — the original task statement, if any."""
    for m in messages:
        if m.role == MessageRole.USER:
            return m
    return None


def _group_with_tool_pairs(messages: List[Message]) -> List[List[Message]]:
    """Group messages into atomic units that must not be split.

    An assistant message carrying ``tool_calls`` is grouped together with the
    consecutive ``TOOL`` (tool_result) messages that answer it. Every other
    message is its own singleton group. Dropping or keeping whole groups
    preserves Anthropic's invariant that each ``tool_use`` block is immediately
    followed by its matching ``tool_result`` block — splitting a group is
    exactly what produced the "tool_use ids were found without tool_result
    blocks" / "unexpected tool_use_id in tool_result blocks" 400s.
    """
    groups: List[List[Message]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if m.role == MessageRole.ASSISTANT and m.tool_calls:
            j = i + 1
            while j < n and messages[j].role == MessageRole.TOOL:
                j += 1
            groups.append(messages[i:j])
            i = j
        else:
            groups.append([m])
            i += 1
    return groups


def _clamp_message(message: Message, max_chars: int) -> Message:
    """Return ``message`` unchanged, or a copy with its string content clamped.

    Only clamps plain string content (tool results / text). List-of-blocks
    content (images, documents) is left alone — those aren't the source of the
    multi-megabyte text blowups this guards against. Returns a NEW Message when
    clamping so the caller never mutates the shared conversation history.
    """
    content = message.content
    if not isinstance(content, str) or len(content) <= max_chars:
        return message
    head = content[:max_chars]
    clamped = (
        f"{head}\n\n[... {len(content) - max_chars} characters truncated to fit "
        f"the context window. Re-run the tool with a narrower scope (fewer rows, "
        f"a shorter date range, or specific fields) if you need the full data.]"
    )
    return Message(
        role=message.role,
        content=clamped,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=message.tool_calls,
        timestamp=message.timestamp,
        metadata=message.metadata,
    )


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    messages: List[Message]
    was_compressed: bool
    original_count: int
    compressed_count: int
    estimated_tokens_before: int
    estimated_tokens_after: int


class ContextCompressor:
    """Compresses message history to fit within token budgets.

    Supports multiple strategies:
    - TRUNCATE: Drop oldest non-system messages, keep recent ones
    - SUMMARIZE: Use LLM to summarize old messages into a single message
    - AUTO: Try summarize first, fall back to truncate
    """

    def __init__(
        self,
        client=None,
        max_context_tokens: Optional[int] = None,
        compression_threshold: float = 0.75,
        strategy: CompressionStrategy = CompressionStrategy.AUTO,
        token_fn: Optional[Callable[[List[Message]], int]] = None,
    ):
        """Initialize context compressor.

        Args:
            client: LLMClient instance (required for SUMMARIZE strategy)
            max_context_tokens: Maximum context tokens. Defaults to 128000.
            compression_threshold: Compress when usage exceeds this fraction (0-1).
            strategy: Compression strategy to use.
            token_fn: Message-token estimator. Defaults to the local chars/4
                heuristic so standalone use keeps working unchanged.
                ``DefaultContextEngine`` injects a calibrated counter here, so
                the truncation loop converges on the same number the engine
                used to make the compress/don't-compress decision. Without
                that, the two disagree and truncation can stop one group short
                of what the engine needed.
        """
        self.client = client
        self.max_context_tokens = max_context_tokens or 128000
        self.compression_threshold = compression_threshold
        self.strategy = strategy
        self._token_fn = token_fn or _estimate_tokens

    def estimate_tokens(self, messages: List[Message]) -> int:
        """Estimate token count for messages."""
        return self._token_fn(messages)

    async def compress_if_needed(
        self, messages: List[Message], preserve_recent: int = 4
    ) -> CompressionResult:
        """Compress messages if they exceed the threshold.

        Args:
            messages: Current message history.
            preserve_recent: Number of recent messages to always preserve.

        Returns:
            CompressionResult with compressed messages if needed.
        """
        estimated_tokens = self._token_fn(messages)
        threshold_tokens = int(self.max_context_tokens * self.compression_threshold)

        if estimated_tokens <= threshold_tokens:
            return CompressionResult(
                messages=messages,
                was_compressed=False,
                original_count=len(messages),
                compressed_count=len(messages),
                estimated_tokens_before=estimated_tokens,
                estimated_tokens_after=estimated_tokens,
            )

        logger.info(
            f"Context compression triggered: ~{estimated_tokens} tokens "
            f"exceeds threshold of ~{threshold_tokens} "
            f"({len(messages)} messages, strategy={self.strategy.value})"
        )

        if self.strategy == CompressionStrategy.NONE:
            return CompressionResult(
                messages=messages,
                was_compressed=False,
                original_count=len(messages),
                compressed_count=len(messages),
                estimated_tokens_before=estimated_tokens,
                estimated_tokens_after=estimated_tokens,
            )

        if self.strategy == CompressionStrategy.TRUNCATE:
            compressed = self._truncate(messages, preserve_recent, threshold_tokens)
        elif self.strategy == CompressionStrategy.SUMMARIZE:
            compressed = await self._summarize(messages, preserve_recent)
        else:  # AUTO
            if self.client:
                try:
                    compressed = await self._summarize(messages, preserve_recent)
                except Exception as e:
                    logger.warning(f"Summarization failed, falling back to truncate: {e}")
                    compressed = self._truncate(messages, preserve_recent, threshold_tokens)
            else:
                compressed = self._truncate(messages, preserve_recent, threshold_tokens)

        # The strategies hand back the SAME list object when they had nothing
        # to do (too few messages, everything inside the preserved window).
        # That is not a compression, and reporting it as one is what made a
        # no-op forced compaction look like a successful recovery step.
        was_compressed = compressed is not messages

        # Final safety net: a single oversized message (e.g. a multi-megabyte
        # tool result) can keep the request over budget even after dropping
        # every other message — and it usually sits in the preserved-recent
        # window, so truncation alone never reaches it. Clamp individual message
        # contents when we're still over threshold. Runs on every strategy /
        # early-return path, so it's the one place the giant-result case is
        # guaranteed to be handled.
        if self._token_fn(compressed) > threshold_tokens:
            clamped = [
                _clamp_message(m, _MAX_SINGLE_MESSAGE_CHARS) for m in compressed
            ]
            if any(c is not m for c, m in zip(clamped, compressed)):
                was_compressed = True
                compressed = clamped

        after_tokens = self._token_fn(compressed)
        logger.info(
            f"Compressed {len(messages)} messages (~{estimated_tokens} tokens) "
            f"to {len(compressed)} messages (~{after_tokens} tokens)"
        )

        return CompressionResult(
            messages=compressed,
            was_compressed=was_compressed,
            original_count=len(messages),
            compressed_count=len(compressed),
            estimated_tokens_before=estimated_tokens,
            estimated_tokens_after=after_tokens,
        )

    def _truncate(
        self, messages: List[Message], preserve_recent: int, target_tokens: int
    ) -> List[Message]:
        """Drop oldest non-system messages to fit within token budget.

        Truncation operates on whole tool_use/tool_result *groups* (see
        ``_group_with_tool_pairs``) so it never severs a tool_use block from its
        tool_result — that split is what made the Anthropic API reject the
        compacted request. As a last resort, when even the preserved tail is
        over budget (typically a single multi-megabyte tool result), individual
        message contents are clamped via ``_clamp_message``.
        """
        if len(messages) <= preserve_recent + 1:
            return messages

        # Always keep system messages and the most recent messages
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in messages if m.role != MessageRole.SYSTEM]

        if len(non_system) <= preserve_recent:
            return messages

        # Group so tool_use ↔ tool_result stays atomic, then keep whole trailing
        # groups until we've preserved at least ``preserve_recent`` messages.
        groups = _group_with_tool_pairs(non_system)
        kept_groups: List[List[Message]] = []
        kept_count = 0
        for group in reversed(groups):
            kept_groups.insert(0, group)
            kept_count += len(group)
            if kept_count >= preserve_recent:
                break

        # The boundary marker is a USER message; a kept window that opens on a
        # TOOL message would be an orphan tool_result (its tool_use was dropped)
        # → "unexpected tool_use_id in tool_result blocks". Drop such leading
        # groups until the window opens on a non-TOOL message.
        while kept_groups and kept_groups[0] and kept_groups[0][0].role == MessageRole.TOOL:
            kept_groups.pop(0)

        head_user = _head_user_message(non_system)

        def build(groups: List[List[Message]], dropped: int) -> List[Message]:
            kept_ids = {id(m) for g in groups for m in g}
            result = list(system_msgs)
            # Re-pin the original task statement when dropping would lose it.
            # It is a singleton USER message, so re-inserting it cannot sever
            # a tool_use/tool_result pair.
            if head_user is not None and id(head_user) not in kept_ids:
                result.append(
                    _clamp_message(head_user, _HEAD_USER_MESSAGE_MAX_CHARS)
                )
            result.append(
                Message(
                    role=MessageRole.USER,
                    content=(
                        f"[Context compressed: {dropped} earlier messages were "
                        "removed to fit context limits. Recent conversation "
                        "follows.]"
                    ),
                )
            )
            for group in groups:
                result.extend(group)
            return result

        dropped_count = len(non_system) - sum(len(g) for g in kept_groups)
        result = build(kept_groups, dropped_count)

        # If still over budget, progressively drop whole groups from the front
        # (after the boundary marker). Dropping a group keeps pairs intact.
        while self._token_fn(result) > target_tokens and len(kept_groups) > 1:
            kept_groups.pop(0)
            while kept_groups and kept_groups[0] and kept_groups[0][0].role == MessageRole.TOOL:
                kept_groups.pop(0)
            dropped_count = len(non_system) - sum(len(g) for g in kept_groups)
            result = build(kept_groups, dropped_count)

        # A single preserved message (usually a giant tool result) may still
        # blow the budget; ``compress_if_needed`` applies the oversized-message
        # clamp as the final safety net across all strategies.
        return result

    async def _request_summary(
        self, summary_prompt: str, max_tokens: int, *, dropped_messages: int = 0
    ) -> str:
        """One summariser call, shaped so the note cannot silently come back empty.

        Runs inside a `context.compaction` CHAIN span so the note's budget,
        finish reason and size are visible in the trace next to the LLM call
        that produced it — the empty-note regression was only found by
        reading raw LLM spans, which is not a place anyone looks routinely.

        * No tools and no MCP servers: the note is prose for the model's own
          future self; the agent's tool schemas and connectors only add tokens
          and change how the model answers.
        * Thinking disabled where the API allows it: adaptive thinking is on by
          default on Claude 5 and shares `max_tokens` with the text.
        * `stop_reason=max_tokens` with no text is retried once with a much
          larger budget; a still-truncated note is kept (partial beats blank)
          and logged so the compaction span shows it.
        """
        from .observability.spans import agent_span, set_span_attribute

        summary_message = [Message(role=MessageRole.USER, content=summary_prompt)]
        summary_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "_formatted_tools": [],
            "mcp_servers": None,
            "thinking_disabled": True,
        }
        with agent_span(
            "context.compaction",
            kind="CHAIN",
            **{
                "compaction.dropped_messages": dropped_messages,
                "compaction.max_tokens": max_tokens,
            },
        ) as span:
            response = await self.client.achat(
                messages=summary_message, max_tokens=max_tokens, **summary_kwargs
            )
            text = _response_text(response)
            finish = getattr(response, "finish_reason", None)
            retried = False
            if not text.strip() and finish in _TRUNCATED_FINISH_REASONS:
                retried = True
                logger.warning(
                    "[COMPACTION] handoff note empty at max_tokens=%d (finish=%s); "
                    "retrying with max_tokens=%d",
                    max_tokens,
                    finish,
                    _SUMMARY_RETRY_TOKENS,
                )
                response = await self.client.achat(
                    messages=summary_message,
                    max_tokens=_SUMMARY_RETRY_TOKENS,
                    **summary_kwargs,
                )
                text = _response_text(response)
                finish = getattr(response, "finish_reason", None)
            if finish in _TRUNCATED_FINISH_REASONS:
                logger.warning(
                    "[COMPACTION] handoff note truncated (finish=%s, chars=%d)",
                    finish,
                    len(text),
                )
            set_span_attribute(span, "compaction.retried", retried)
            set_span_attribute(span, "compaction.finish_reason", finish)
            set_span_attribute(span, "compaction.summary_chars", len(text))
            set_span_attribute(span, "compaction.truncated", finish in _TRUNCATED_FINISH_REASONS)
            # The budget the FINAL response was generated under — the attr set
            # at span start records only the first attempt's budget, which
            # understates a retried call.
            set_span_attribute(
                span,
                "compaction.max_tokens",
                _SUMMARY_RETRY_TOKENS if retried else max_tokens,
            )
            # No local clip: the SDK's span limit (spans.attribute_value_limit,
            # 32k default) owns attribute bounding. The old 4,000-char clip
            # made every complete note LOOK cut off in Arize, which is how
            # "notes still truncated" survived two audits after the empty-note
            # bug was actually fixed.
            set_span_attribute(span, "output.value", text)
        return text or "Previous conversation context unavailable."

    async def _summarize(
        self, messages: List[Message], preserve_recent: int
    ) -> List[Message]:
        """Summarize older messages using LLM."""
        if not self.client:
            raise ValueError("LLMClient required for summarize strategy")

        if len(messages) <= preserve_recent + 1:
            return messages

        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in messages if m.role != MessageRole.SYSTEM]

        if len(non_system) <= preserve_recent:
            return messages

        # Split into old (to summarize) and recent (to keep). Select the recent
        # window by whole tool_use/tool_result groups so we never keep an orphan
        # tool_result whose tool_use got folded into the summary — that split is
        # what the Anthropic API rejects.
        groups = _group_with_tool_pairs(non_system)
        kept_groups: List[List[Message]] = []
        kept_count = 0
        for group in reversed(groups):
            kept_groups.insert(0, group)
            kept_count += len(group)
            if kept_count >= preserve_recent:
                break
        while kept_groups and kept_groups[0] and kept_groups[0][0].role == MessageRole.TOOL:
            kept_groups.pop(0)
        recent = [m for group in kept_groups for m in group]
        recent_set = {id(m) for m in recent}

        # The original task statement is preserved verbatim, so it is neither
        # summarized nor dropped — paraphrasing the one message that defines
        # success is how compacted runs drift off-goal.
        head_user = _head_user_message(non_system)
        if head_user is not None and id(head_user) in recent_set:
            head_user = None
        excluded = recent_set | ({id(head_user)} if head_user is not None else set())
        old_messages = [m for m in non_system if id(m) not in excluded]

        # Nothing sits between the task statement and the recent window: there
        # is nothing to fold into a note. Returning unchanged (same list
        # object, so the caller reports was_compressed=False) beats what used
        # to happen — a summariser call over an empty transcript that came
        # back "the conversation history is empty… wait for the user's first
        # message", which was then INSERTED into the live conversation
        # (2026-08-18, forced compaction on a 6-message run).
        if not old_messages:
            logger.info(
                "[COMPACTION] nothing to compact: %d messages, all within the "
                "preserved window (preserve_recent=%d)",
                len(messages),
                preserve_recent,
            )
            return messages

        # Format old messages for summarization
        formatted = []
        for msg in old_messages:
            role_label = msg.role.value.upper()
            content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                calls = ", ".join(
                    str((c.get("function") or {}).get("name", "?"))
                    for c in msg.tool_calls
                    if isinstance(c, dict)
                )
                content = f"[called tools: {calls}] {content}"
            # Clip very long individual messages, keeping enough to retain a
            # tool result's shape and key numbers.
            if len(content) > _SUMMARY_INPUT_CLIP_CHARS:
                content = content[:_SUMMARY_INPUT_CLIP_CHARS] + "..."
            formatted.append(f"{role_label}: {content}")

        conversation_text = "\n".join(formatted)

        # A handoff note beats a summary: the model that continues the run is
        # the audience, and what it needs is exact identifiers and an explicit
        # settled/open split — not prose about what the conversation covered.
        summary_prompt = (
            "You are the assistant in the conversation below, which is about "
            "to have its older messages compacted away. Write a handoff note "
            "to yourself so you can continue seamlessly.\n"
            "- State the user's goal and the current subtask.\n"
            "- Record exact identifiers verbatim: file paths, commands, IDs, "
            "URLs, names, and numbers, including the key results of tool "
            "calls. Never paraphrase an identifier.\n"
            "- Separate SETTLED (decisions made, results obtained, questions "
            "answered) from OPEN (pending work, unverified assumptions, "
            "unanswered questions).\n"
            "- End with the immediate next step.\n"
            "Write in first person. Facts only — no meta commentary about "
            "summarizing.\n\n"
            f"Conversation ({len(old_messages)} messages):\n{conversation_text}"
        )

        # Budget scales with how much history the note replaces.
        dropped_chars = sum(
            len(m.content) if isinstance(m.content, str) else 0 for m in old_messages
        )
        summary_max_tokens = min(
            _SUMMARY_MAX_TOKENS, max(_SUMMARY_MIN_TOKENS, dropped_chars // 100)
        )

        try:
            summary = await self._request_summary(
                summary_prompt, summary_max_tokens, dropped_messages=len(old_messages)
            )
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            summary = f"[{len(old_messages)} earlier messages summarized - details unavailable due to error]"

        # Build result
        result = list(system_msgs)
        if head_user is not None:
            result.append(_clamp_message(head_user, _HEAD_USER_MESSAGE_MAX_CHARS))
        result.append(
            Message(
                role=MessageRole.USER,
                content=(
                    f"[Context compressed: {len(old_messages)} earlier messages "
                    "were replaced by this handoff note. Facts and results in "
                    "it are already established — verify against it instead of "
                    "re-doing work it records as done.]\n"
                    f"{summary}"
                ),
            )
        )
        result.extend(recent)
        return result
