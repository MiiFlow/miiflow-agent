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
        """Drop oldest non-system messages to fit within token budget."""
        if len(messages) <= preserve_recent + 1:
            return messages

        # Always keep system messages and the most recent messages
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in messages if m.role != MessageRole.SYSTEM]

        if len(non_system) <= preserve_recent:
            return messages

        # Keep the most recent messages
        recent = non_system[-preserve_recent:]
        dropped_count = len(non_system) - preserve_recent

        # Build result with boundary marker
        result = list(system_msgs)
        result.append(
            Message(
                role=MessageRole.USER,
                content=f"[Context compressed: {dropped_count} earlier messages were removed to fit context limits. Recent conversation follows.]",
            )
        )
        result.extend(recent)

        # If still over budget, progressively drop more
        while _estimate_tokens(result) > target_tokens and len(result) > len(system_msgs) + 2:
            # Remove the message right after the boundary marker
            for i, msg in enumerate(result):
                if msg.role != MessageRole.SYSTEM and "[Context compressed:" not in (msg.content or ""):
                    result.pop(i)
                    break
            else:
                break

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

        # Split into old (to summarize) and recent (to keep)
        old_messages = non_system[:-preserve_recent]
        recent = non_system[-preserve_recent:]

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
