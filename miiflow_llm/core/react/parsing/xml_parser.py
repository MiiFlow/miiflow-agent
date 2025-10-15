"""XML-based ReAct parser with streaming support.

This parser handles XML-tagged ReAct responses in a streaming fashion,
allowing for incremental parsing without buffering the entire response.

Format:
    <thinking>
    Reasoning about the task
    </thinking>

    <tool_call name="tool_name">
    {"param1": "value1"}
    </tool_call>

    <answer>
    Final answer content
    </answer>
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


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

    def parse_streaming(self, chunk: str) -> Iterator[ParseEvent]:
        """Parse XML chunks incrementally and yield parse events.

        Args:
            chunk: New text chunk to parse

        Yields:
            ParseEvent: Events detected in the chunk
        """
        self.buffer += chunk

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

    def parse_complete(self, response: str) -> Dict[str, Any]:
        """Parse a complete XML response (non-streaming).

        Args:
            response: Complete XML response text

        Returns:
            Dictionary with parsed fields
        """
        self.reset()

        result = {
            "thought": None,
            "action_type": None,
            "action": None,
            "action_input": None,
            "answer": None,
            "original_response": response
        }

        # Extract thinking
        thinking_match = re.search(
            r'<thinking>(.*?)</thinking>',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if thinking_match:
            result["thought"] = thinking_match.group(1).strip()

        # Extract tool call
        tool_call_match = re.search(
            r'<tool_call\s+name=["\']([^"\']+)["\']>(.*?)</tool_call>',
            response,
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
            response,
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
                f"Response: {response[:200]}..."
            )

        return result
