"""XML-based ReAct parser with streaming support.

This parser handles XML-tagged ReAct responses in a streaming fashion,
allowing for incremental parsing without buffering the entire response.

Supported formats:
    <thinking>
    Reasoning about the task
    </thinking>

    <tool_call name="tool_name">
    {"param1": "value1"}
    </tool_call>

    <answer>
    Final answer content
    </answer>

Also suppresses (when native tool calling is active):
    <function_calls>
    <invoke name="tool_name">
    <parameter name="param">value</parameter>
    </invoke>
    </function_calls>
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# XML patterns to suppress from display (redundant when native tool calling is active)
# These are stripped from the buffer to prevent them from appearing in final output
SUPPRESS_PATTERNS = [
    # Anthropic's XML function calling format
    (r'<function_calls>.*?</function_calls>', 'function_calls'),
    # Alternative formats
    (r'<tool_use>.*?</tool_use>', 'tool_use'),
]

# Tags that require holdback during streaming (wait for complete tag before processing)
HOLDBACK_TAGS = [
    '<function_calls',
    '</function_calls',
    '<invoke',
    '</invoke',
    '<tool_use',
    '</tool_use',
    '<',
    '</',
]


class ParseEventType(Enum):
    """Types of parse events emitted during streaming."""
    THINKING = "thinking"
    THINKING_COMPLETE = "thinking_complete"
    TOOL_CALL = "tool_call"
    ANSWER_START = "answer_start"
    ANSWER_CHUNK = "answer_chunk"
    ANSWER_COMPLETE = "answer_complete"


@dataclass
class ParseEvent:
    """Event emitted during streaming parse."""
    event_type: ParseEventType
    data: Dict[str, Any]


class XMLReActParser:
    """Streaming XML-based ReAct parser.

    This parser processes XML-tagged ReAct responses incrementally,
    allowing for real-time event emission without waiting for complete response.

    It also suppresses redundant XML formats (like <function_calls>) that some
    LLMs output even when using native tool calling APIs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset parser state for new response."""
        self.buffer = ""
        self.current_thinking = ""
        self.current_answer = ""
        self.in_thinking = False
        self.in_answer = False
        self.thinking_complete = False
        self.answer_started = False
        self.has_parsed_content = False  # Track if we've parsed any valid content
        self.in_suppressed_block = False  # Track if we're inside a block being suppressed

    def _suppress_redundant_xml(self) -> bool:
        """Suppress redundant XML blocks from the buffer.

        When LLMs use native tool calling but also output XML descriptions,
        we need to strip the XML to prevent it from appearing in the output.

        Returns:
            True if content was suppressed, False otherwise
        """
        suppressed = False

        for pattern, tag_name in SUPPRESS_PATTERNS:
            match = re.search(pattern, self.buffer, re.DOTALL | re.IGNORECASE)
            if match:
                # Remove the matched content from buffer
                self.buffer = self.buffer[:match.start()] + self.buffer[match.end():]
                logger.debug(f"Suppressed <{tag_name}> XML block ({match.end() - match.start()} chars)")
                suppressed = True

        return suppressed

    def _has_partial_suppress_tag(self) -> bool:
        """Check if buffer ends with a partial tag that should be suppressed.

        This prevents emitting partial XML tags like '<function_ca' as text.

        Returns:
            True if we should hold back the buffer
        """
        # Check for potential partial tags at the end of buffer
        for tag in HOLDBACK_TAGS:
            # Check if buffer ends with start of this tag
            for i in range(1, len(tag) + 1):
                if self.buffer.endswith(tag[:i]):
                    return True

        return False

    def parse_streaming(self, chunk: str) -> Iterator[ParseEvent]:
        """Parse XML chunks incrementally and yield parse events.

        Args:
            chunk: New text chunk to parse

        Yields:
            ParseEvent: Events detected in the chunk
        """
        self.buffer += chunk

        # Suppress redundant XML blocks (e.g., <function_calls> when using native tool calling)
        # This must happen before other processing to prevent partial XML from being emitted
        self._suppress_redundant_xml()

        # Check if we're waiting for a potentially suppressible tag to complete
        # This prevents emitting partial tags like "<function_ca" as text
        if self._has_partial_suppress_tag() and not self.in_thinking and not self.in_answer:
            # Check if we have an incomplete <function_calls> or similar block
            if '<function_calls' in self.buffer and '</function_calls>' not in self.buffer:
                return  # Wait for complete block before processing
            if '<invoke' in self.buffer and '</invoke>' not in self.buffer:
                return  # Wait for complete block

        # Handle thinking section with proper streaming
        if not self.thinking_complete:
            if self.in_thinking:
                # We're already inside <thinking>, look for closing tag
                closing_match = re.search(r'</thinking>', self.buffer, re.IGNORECASE)

                if closing_match:
                    # Found closing tag - emit remaining content and complete
                    remaining_content = self.buffer[:closing_match.start()]

                    if remaining_content:
                        self.current_thinking += remaining_content
                        yield ParseEvent(
                            event_type=ParseEventType.THINKING,
                            data={"delta": remaining_content}
                        )

                    self.thinking_complete = True
                    self.in_thinking = False
                    self.has_parsed_content = True

                    yield ParseEvent(
                        event_type=ParseEventType.THINKING_COMPLETE,
                        data={"thought": self.current_thinking.strip()}
                    )

                    # Remove processed content from buffer
                    self.buffer = self.buffer[closing_match.end():]
                else:
                    # No closing tag yet - stream content with holdback
                    holdback = 11  # Length of </thinking>

                    if len(self.buffer) > holdback:
                        # Emit everything except last 11 chars
                        chunk_to_emit = self.buffer[:-holdback]
                        self.current_thinking += chunk_to_emit
                        self.buffer = self.buffer[-holdback:]

                        if chunk_to_emit:
                            yield ParseEvent(
                                event_type=ParseEventType.THINKING,
                                data={"delta": chunk_to_emit}
                            )
                    # else: buffer too small, wait for more chunks

            elif '<thinking>' in self.buffer:
                # Found opening tag, start thinking mode
                self.in_thinking = True
                start_idx = self.buffer.find('<thinking>') + len('<thinking>')
                self.buffer = self.buffer[start_idx:]

                # Check immediately if there's a closing tag in the same buffer
                closing_match = re.search(r'</thinking>', self.buffer, re.IGNORECASE)
                if closing_match:
                    # Complete thinking in one chunk
                    thinking_content = self.buffer[:closing_match.start()]
                    self.current_thinking = thinking_content

                    if thinking_content:
                        yield ParseEvent(
                            event_type=ParseEventType.THINKING,
                            data={"delta": thinking_content}
                        )

                    self.thinking_complete = True
                    self.in_thinking = False
                    self.has_parsed_content = True

                    yield ParseEvent(
                        event_type=ParseEventType.THINKING_COMPLETE,
                        data={"thought": thinking_content.strip()}
                    )

                    self.buffer = self.buffer[closing_match.end():]

        # Try to extract tool calls
        tool_call_match = re.search(
            r'<tool_call\s+name=["\']([^"\']+)["\']>(.*?)</tool_call>',
            self.buffer,
            re.DOTALL | re.IGNORECASE
        )
        if tool_call_match:
            tool_name = tool_call_match.group(1).strip()
            tool_params_str = tool_call_match.group(2).strip()

            # Parse JSON parameters
            try:
                tool_params = json.loads(tool_params_str) if tool_params_str else {}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool parameters as JSON: {e}")
                logger.debug(f"Raw parameters: {tool_params_str}")
                # Try to fix common issues
                tool_params_str = tool_params_str.strip()
                if not tool_params_str:
                    tool_params = {}
                else:
                    # Attempt to parse again with more lenient approach
                    try:
                        # Remove any markdown code block markers
                        tool_params_str = re.sub(r'```(?:json)?\s*', '', tool_params_str)
                        tool_params_str = re.sub(r'```\s*', '', tool_params_str)
                        tool_params = json.loads(tool_params_str)
                    except json.JSONDecodeError:
                        logger.error(f"Could not parse tool parameters, using empty dict")
                        tool_params = {}

            self.has_parsed_content = True
            yield ParseEvent(
                event_type=ParseEventType.TOOL_CALL,
                data={
                    "tool_name": tool_name,
                    "parameters": tool_params
                }
            )
            # Remove processed tool call from buffer
            self.buffer = self.buffer[tool_call_match.end():]

        # Detect answer start
        if not self.answer_started:
            answer_start_match = re.search(r'<answer>', self.buffer, re.IGNORECASE)
            if answer_start_match:
                self.answer_started = True
                self.in_answer = True
                self.has_parsed_content = True
                # Remove everything up to and including <answer> tag
                self.buffer = self.buffer[answer_start_match.end():]
                yield ParseEvent(
                    event_type=ParseEventType.ANSWER_START,
                    data={}
                )

        # If in answer mode, emit answer chunks
        if self.in_answer and self.buffer:
            # Check if answer is complete
            answer_end_match = re.search(r'</answer>', self.buffer, re.IGNORECASE)
            if answer_end_match:
                # Emit final chunk before closing tag
                final_chunk = self.buffer[:answer_end_match.start()]
                if final_chunk:
                    self.current_answer += final_chunk
                    yield ParseEvent(
                        event_type=ParseEventType.ANSWER_CHUNK,
                        data={"delta": final_chunk}
                    )

                self.in_answer = False
                yield ParseEvent(
                    event_type=ParseEventType.ANSWER_COMPLETE,
                    data={"answer": self.current_answer}
                )
                self.buffer = self.buffer[answer_end_match.end():]
            else:
                # Emit chunk but keep last few chars in buffer
                # (in case </answer> tag is split across chunks)
                if len(self.buffer) > 10:
                    chunk_to_emit = self.buffer[:-10]
                    self.current_answer += chunk_to_emit
                    self.buffer = self.buffer[-10:]
                    yield ParseEvent(
                        event_type=ParseEventType.ANSWER_CHUNK,
                        data={"delta": chunk_to_emit}
                    )

    def finalize(self) -> Iterator[ParseEvent]:
        """Finalize parsing and flush any remaining buffered content.

        IMPORTANT: Call this method after streaming ends to ensure all content
        is properly emitted. This handles the case where the LLM doesn't output
        a closing </answer> tag - the held-back buffer content needs to be flushed.

        Yields:
            ParseEvent: Any remaining events from buffered content
        """
        # Final cleanup: suppress any remaining redundant XML blocks
        # This catches incomplete blocks that were waiting for closing tags
        self._suppress_redundant_xml()

        # Also strip any partial/incomplete suppressed tags from the buffer
        # This prevents partial XML like "<function_calls" from appearing in output
        self.buffer = re.sub(r'<function_calls[^>]*$', '', self.buffer)
        self.buffer = re.sub(r'<invoke[^>]*$', '', self.buffer)
        self.buffer = re.sub(r'<tool_use[^>]*$', '', self.buffer)
        self.buffer = re.sub(r'<parameter[^>]*$', '', self.buffer)

        # If we're in answer mode and have buffered content, flush it
        if self.in_answer and self.buffer:
            # Check one more time for closing tag
            answer_end_match = re.search(r'</answer>', self.buffer, re.IGNORECASE)
            if answer_end_match:
                # Found closing tag - emit final chunk and complete
                final_chunk = self.buffer[:answer_end_match.start()]
                if final_chunk:
                    self.current_answer += final_chunk
                    yield ParseEvent(
                        event_type=ParseEventType.ANSWER_CHUNK,
                        data={"delta": final_chunk}
                    )
                self.in_answer = False
                yield ParseEvent(
                    event_type=ParseEventType.ANSWER_COMPLETE,
                    data={"answer": self.current_answer}
                )
                self.buffer = self.buffer[answer_end_match.end():]
            else:
                # No closing tag found - flush remaining buffer as final chunk
                # This handles LLMs that don't output </answer> properly
                if self.buffer:
                    self.current_answer += self.buffer
                    yield ParseEvent(
                        event_type=ParseEventType.ANSWER_CHUNK,
                        data={"delta": self.buffer}
                    )
                    self.buffer = ""

                # Emit ANSWER_COMPLETE with whatever we accumulated
                # (even without closing tag, we need to signal completion)
                self.in_answer = False
                yield ParseEvent(
                    event_type=ParseEventType.ANSWER_COMPLETE,
                    data={"answer": self.current_answer}
                )

        # If we're in thinking mode and have buffered content, flush it
        elif self.in_thinking and self.buffer:
            # Check for closing tag
            closing_match = re.search(r'</thinking>', self.buffer, re.IGNORECASE)
            if closing_match:
                remaining_content = self.buffer[:closing_match.start()]
                if remaining_content:
                    self.current_thinking += remaining_content
                    yield ParseEvent(
                        event_type=ParseEventType.THINKING,
                        data={"delta": remaining_content}
                    )
                self.thinking_complete = True
                self.in_thinking = False
                self.has_parsed_content = True
                yield ParseEvent(
                    event_type=ParseEventType.THINKING_COMPLETE,
                    data={"thought": self.current_thinking.strip()}
                )
                self.buffer = self.buffer[closing_match.end():]
            else:
                # No closing tag - flush remaining as thinking
                if self.buffer:
                    self.current_thinking += self.buffer
                    yield ParseEvent(
                        event_type=ParseEventType.THINKING,
                        data={"delta": self.buffer}
                    )
                    self.buffer = ""

                self.thinking_complete = True
                self.in_thinking = False
                self.has_parsed_content = True
                yield ParseEvent(
                    event_type=ParseEventType.THINKING_COMPLETE,
                    data={"thought": self.current_thinking.strip()}
                )

    def parse_complete(self, response: str) -> Dict[str, Any]:
        """Parse a complete XML response (non-streaming).

        Args:
            response: Complete XML response text

        Returns:
            Dictionary with parsed fields
        """
        self.reset()

        # Strip redundant XML blocks before parsing
        # This removes <function_calls> and similar blocks that are redundant
        # when native tool calling is active
        cleaned_response = response
        for pattern, tag_name in SUPPRESS_PATTERNS:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)

        result = {
            "thought": None,
            "action_type": None,
            "action": None,
            "action_input": None,
            "answer": None,
            "original_response": response  # Keep original for debugging
        }

        # Extract thinking (use cleaned_response to avoid parsing suppressed content)
        thinking_match = re.search(
            r'<thinking>(.*?)</thinking>',
            cleaned_response,
            re.DOTALL | re.IGNORECASE
        )
        if thinking_match:
            result["thought"] = thinking_match.group(1).strip()

        # Extract tool call
        tool_call_match = re.search(
            r'<tool_call\s+name=["\']([^"\']+)["\']>(.*?)</tool_call>',
            cleaned_response,
            re.DOTALL | re.IGNORECASE
        )
        if tool_call_match:
            result["action_type"] = "tool_call"
            result["action"] = tool_call_match.group(1).strip()
            tool_params_str = tool_call_match.group(2).strip()

            try:
                result["action_input"] = json.loads(tool_params_str) if tool_params_str else {}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool parameters: {tool_params_str}")
                result["action_input"] = {}

        # Extract answer
        answer_match = re.search(
            r'<answer>(.*?)</answer>',
            cleaned_response,
            re.DOTALL | re.IGNORECASE
        )
        if answer_match:
            result["action_type"] = "final_answer"
            result["answer"] = answer_match.group(1).strip()

        # Validate we found something
        if not result["thought"] and not result["answer"]:
            raise ValueError(
                f"No valid ReAct XML tags found in response. "
                f"Expected <thinking>, <tool_call>, or <answer> tags. "
                f"Response: {cleaned_response[:200]}..."
            )

        return result
