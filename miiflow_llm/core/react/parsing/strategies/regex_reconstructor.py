"""Clean and effective regex-based JSON reconstruction strategy."""

import json
import re
import logging

from .json_extractor import HealingStrategy
from ...data import ReActParsingError

logger = logging.getLogger(__name__)


class RegexReconstructor(HealingStrategy):
    """Reconstruct JSON from natural language response using regex patterns."""

    def _execute_healing(self, response: str, context: dict) -> str:
        """Reconstruct JSON from natural language response."""
        # Extract thought
        thought_patterns = [
            r'(?:thought|thinking|reasoning)[:=]\s*["\']?([^"\'.\n]+)["\']?',
            r'I (?:think|believe|need to|should|will)\s+([^.\n]+)',
            r'(?:My reasoning|The plan|Next step):\s*([^.\n]+)'
        ]

        thought = "No clear thought identified"
        for pattern in thought_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                thought = match.group(1).strip()
                break

        # Detect if this is a final answer
        final_answer_indicators = [
            r'(?:final|my|the) answer(?:\s+is)?[:=]\s*([^.\n]+)',
            r'(?:conclusion|result|solution)[:=]\s*([^.\n]+)',
            r'(?:in summary|to summarize|therefore)[:.,]\s*([^.\n]+)'
        ]

        for pattern in final_answer_indicators:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                return json.dumps({
                    "thought": thought,
                    "action_type": "final_answer",
                    "answer": answer
                })

        # Look for tool calls
        tool_patterns = [
            r'(?:use|call|execute).*?(?:tool|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'(?:action|tool)[:=]\s*([a-zA-Z_][a-zA-Z0-9_]*)',
        ]

        for pattern in tool_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                action = match.group(1).strip()
                # Try to extract parameters
                action_input = {}
                param_pattern = r'(?:parameters?|args?|input)[:=]\s*(\{.*?\})'
                param_match = re.search(param_pattern, response, re.IGNORECASE | re.DOTALL)
                if param_match:
                    try:
                        action_input = json.loads(param_match.group(1))
                    except:
                        action_input = {"query": response}  # Fallback

                return json.dumps({
                    "thought": thought,
                    "action_type": "tool_call",
                    "action": action,
                    "action_input": action_input or {"query": thought}
                })

        # Default: treat as final answer
        return json.dumps({
            "thought": thought,
            "action_type": "final_answer",
            "answer": response[:200] + ("..." if len(response) > 200 else "")
        })

    def get_name(self) -> str:
        """Get strategy name."""
        return "regex_reconstruction"
