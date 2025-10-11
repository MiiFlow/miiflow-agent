"""Clean and effective JSON extraction strategies."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ...data import ReActParsingError

logger = logging.getLogger(__name__)




class HealingStrategy(ABC):
    """Clean base class for parsing healing strategies."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def _execute_healing(self, response: str, context: Dict[str, Any]) -> str:
        """Core healing logic - implemented by subclasses."""
        pass
    
    def heal(self, response: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute healing strategy with effective error handling."""
        context = context or {}
        
        try:
            self._validate_input(response)
            healed_content = self._execute_healing(response, context)
            try:
                json.loads(healed_content)
                return healed_content
            except json.JSONDecodeError as e:
                raise ReActParsingError(f"Strategy produced invalid JSON: {e}")
            
        except Exception as e:
            logger.error(f"Strategy {self.name} failed: {e}", exc_info=True)
            raise ReActParsingError(f"Strategy {self.name} failed: {e}")
    
    def _validate_input(self, response: str) -> None:
        """Validate input parameters."""
        if not isinstance(response, str):
            raise ValueError(f"Response must be string, got {type(response)}")
        if not response.strip():
            raise ValueError("Response cannot be empty")
        if len(response) > 50000:  # Reasonable limit
            raise ValueError(f"Response too long: {len(response)} characters")
    
    def get_name(self) -> str:
        """Get strategy name for compatibility."""
        return self.name


class JsonBlockExtractor(HealingStrategy):
    """Production-grade JSON block extractor with multiple extraction patterns."""
    
    def __init__(self):
        super().__init__("JsonBlockExtractor")
        # Ordered from most specific to most general
        self.extraction_patterns = [
            (r'```(?:json|javascript|js)\s*\n(.*?)\n```', "markdown_json"),
            (r'```\s*\n(\{.*?\})\s*\n```', "markdown_generic"),
            (r'`(\{[^`]+\})`', "inline_code"),
            (r'(?:^|\n)\s*(\{(?:[^{}]|{[^{}]*})*\})\s*(?:\n|$)', "delimited_json"),
            (r'(\{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*\})', "general_json"),
        ]
    
    def _execute_healing(self, response: str, context: Dict[str, Any]) -> str:
        """Extract JSON using multiple sophisticated patterns."""
        candidates = []
        
        for pattern, pattern_name in self.extraction_patterns:
            matches = self._find_json_candidates(response, pattern, pattern_name)
            candidates.extend(matches)
        
        if not candidates:
            raise ReActParsingError("No JSON candidates found in response")
        
        best_candidate = self._select_best_candidate(candidates, response)
        
        logger.debug(f"Selected JSON candidate using pattern: {best_candidate['pattern_name']}")
        return best_candidate['content']
    
    def _find_json_candidates(self, response: str, pattern: str, pattern_name: str) -> List[Dict[str, Any]]:
        """Find JSON candidates using a specific pattern."""
        import re
        candidates = []
        
        matches = re.finditer(pattern, response, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        for match in matches:
            content = match.group(1).strip()
            if self._is_valid_json_candidate(content):
                candidates.append({
                    'content': content,
                    'pattern_name': pattern_name
                })
        
        return candidates
    
    def _is_valid_json_candidate(self, content: str) -> bool:
        """Quick validation to filter obvious non-JSON candidates."""
        if not content or len(content) < 10:
            return False
        content = content.strip()
        return content.startswith('{') and content.endswith('}')
    
    def _select_best_candidate(self, candidates: List[Dict[str, Any]], original: str) -> Dict[str, Any]:
        """Select best JSON candidate """
        if len(candidates) == 1:
            return candidates[0]
        for candidate in candidates:
            try:
                json.loads(candidate['content'])
                return candidate
            except json.JSONDecodeError:
                continue
        return candidates[0]
