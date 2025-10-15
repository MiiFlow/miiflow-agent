"""ReAct response parser - simplified wrapper around XML parser.

This module provides backward compatibility while delegating to the XML parser.
"""

import logging
from typing import Any, Dict

from .data import ParseResult
from .parsing.xml_parser import XMLReActParser

logger = logging.getLogger(__name__)


class ReActParsingError(Exception):
    """Raised when ReAct response cannot be parsed."""
    pass


class ReActParser:
    """Simplified ReAct parser that delegates to XML parser.

    This class maintains backward compatibility while using the new
    XML-based parsing infrastructure.
    """

    def __init__(self, strict_validation: bool = True):
        self.strict_validation = strict_validation
        self.xml_parser = XMLReActParser()

    def parse_response(self, response: str, step_number: int) -> ParseResult:
        """Parse LLM response into structured ReActStep.

        Args:
            response: Complete XML response text
            step_number: Current step number (for logging)

        Returns:
            ParseResult with extracted fields

        Raises:
            ReActParsingError: If parsing fails
        """
        try:
            parsed_data = self.xml_parser.parse_complete(response)

            # Convert to ParseResult format
            return ParseResult(
                thought=parsed_data.get("thought") or "",
                action_type=parsed_data.get("action_type") or "unknown",
                action=parsed_data.get("action"),
                action_input=parsed_data.get("action_input"),
                answer=parsed_data.get("answer"),
                original_response=response,
                was_healed=False,
                healing_applied="",
                confidence=1.0
            )

        except Exception as e:
            logger.error(f"Failed to parse response at step {step_number}: {e}")
            raise ReActParsingError(
                f"Failed to parse ReAct response: {str(e)}. "
                f"Response preview: {response[:200]}..."
            )

    def reset(self):
        """Reset parser state for new streaming response."""
        self.xml_parser.reset()

    def parse_streaming(self, chunk: str):
        """Parse XML chunks incrementally (delegates to XML parser)."""
        return self.xml_parser.parse_streaming(chunk)

    @property
    def has_parsed_content(self) -> bool:
        """Check if any content was successfully parsed."""
        return self.xml_parser.has_parsed_content

    @property
    def buffer(self) -> str:
        """Access the internal buffer."""
        return self.xml_parser.buffer
