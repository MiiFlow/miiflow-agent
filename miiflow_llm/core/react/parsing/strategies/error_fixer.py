"""Common JSON error fixing strategy."""

import re

from .json_extractor import HealingStrategy
from ...data import ReActParsingError
from .json_extractor import JsonBlockExtractor

class CommonErrorFixer(HealingStrategy):
    """Fix common JSON formatting issues."""

    def _execute_healing(self, response: str, context: dict) -> str:
        """Fix common JSON formatting errors."""
        extractor = JsonBlockExtractor()

        try:
            json_str = extractor.heal(response)
        except (ReActParsingError, Exception):
            json_str = response

        fixes = [
            (r',(\s*[}\]])', r'\1'),
            (r"'([^']*)':", r'"\1":'),
            (r":\s*'([^']*)'", r': "\1"'),
            (r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
            (r'\bTrue\b', 'true'),
            (r'\bFalse\b', 'false'),
            (r'\bNone\b', 'null'),
        ]

        for pattern, replacement in fixes:
            json_str = re.sub(pattern, replacement, json_str)

        return json_str
