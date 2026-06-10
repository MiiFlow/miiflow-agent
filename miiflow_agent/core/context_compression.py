"""Context compression for managing conversation history within token limits.

Provides multiple strategies for compressing message history when it approaches
context window limits, inspired by Claude Code's multi-level compaction system.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

from .message import Message, MessageRole

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
    ):
        """Initialize context compressor.

        Args:
            client: LLMClient instance (required for SUMMARIZE strategy)
            max_context_tokens: Maximum context tokens. Defaults to 128000.
            compression_threshold: Compress when usage exceeds this fraction (0-1).
            strategy: Compression strategy to use.
        """
        self.client = client
        self.max_context_tokens = max_context_tokens or 128000
        self.compression_threshold = compression_threshold
        self.strategy = strategy

    def estimate_tokens(self, messages: List[Message]) -> int:
        """Estimate token count for messages."""
        return _estimate_tokens(messages)

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
        estimated_tokens = _estimate_tokens(messages)
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

        # Final safety net: a single oversized message (e.g. a multi-megabyte
        # tool result) can keep the request over budget even after dropping
        # every other message — and it usually sits in the preserved-recent
        # window, so truncation alone never reaches it. Clamp individual message
        # contents when we're still over threshold. Runs on every strategy /
        # early-return path, so it's the one place the giant-result case is
        # guaranteed to be handled.
        if _estimate_tokens(compressed) > threshold_tokens:
            compressed = [
                _clamp_message(m, _MAX_SINGLE_MESSAGE_CHARS) for m in compressed
            ]

        after_tokens = _estimate_tokens(compressed)
        logger.info(
            f"Compressed {len(messages)} messages (~{estimated_tokens} tokens) "
            f"to {len(compressed)} messages (~{after_tokens} tokens)"
        )

        return CompressionResult(
            messages=compressed,
            was_compressed=True,
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

        dropped_count = len(non_system) - sum(len(g) for g in kept_groups)

        # Build result with boundary marker
        result = list(system_msgs)
        result.append(
            Message(
                role=MessageRole.USER,
                content=f"[Context compressed: {dropped_count} earlier messages were removed to fit context limits. Recent conversation follows.]",
            )
        )
        for group in kept_groups:
            result.extend(group)

        # If still over budget, progressively drop whole groups from the front
        # (after the boundary marker). Dropping a group keeps pairs intact.
        while _estimate_tokens(result) > target_tokens and len(kept_groups) > 1:
            kept_groups.pop(0)
            while kept_groups and kept_groups[0] and kept_groups[0][0].role == MessageRole.TOOL:
                kept_groups.pop(0)
            result = list(system_msgs)
            result.append(
                Message(
                    role=MessageRole.USER,
                    content=f"[Context compressed: earlier messages were removed to fit context limits. Recent conversation follows.]",
                )
            )
            for group in kept_groups:
                result.extend(group)

        # A single preserved message (usually a giant tool result) may still
        # blow the budget; ``compress_if_needed`` applies the oversized-message
        # clamp as the final safety net across all strategies.
        return result

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
        old_messages = [m for m in non_system if id(m) not in recent_set]

        # Format old messages for summarization
        formatted = []
        for msg in old_messages:
            role_label = msg.role.value.upper()
            content = msg.content or ""
            # Truncate very long individual messages
            if len(content) > 1000:
                content = content[:1000] + "..."
            formatted.append(f"{role_label}: {content}")

        conversation_text = "\n".join(formatted)

        summary_prompt = (
            "Summarize this conversation history concisely. "
            "Preserve key facts, decisions, tool results, and context needed "
            "to continue the conversation. Be brief but complete.\n\n"
            f"Conversation ({len(old_messages)} messages):\n{conversation_text}"
        )

        try:
            response = await self.client.achat(
                messages=[Message(role=MessageRole.USER, content=summary_prompt)],
                temperature=0.0,
                max_tokens=500,
            )
            summary = response.message.content or "Previous conversation context unavailable."
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            summary = f"[{len(old_messages)} earlier messages summarized - details unavailable due to error]"

        # Build result
        result = list(system_msgs)
        result.append(
            Message(
                role=MessageRole.USER,
                content=f"[Context compressed: {len(old_messages)} messages summarized]\n{summary}",
            )
        )
        result.extend(recent)
        return result
