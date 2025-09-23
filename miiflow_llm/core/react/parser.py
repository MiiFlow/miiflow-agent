"""ReAct response parser with self-healing capabilities."""

import json
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .data import ReActStep, REACT_RESPONSE_SCHEMA, ParseResult, ReActParsingError
from .parsing.strategies.json_extractor import HealingStrategy, JsonBlockExtractor
from .parsing.strategies.error_fixer import CommonErrorFixer
from .parsing.strategies.regex_reconstructor import RegexReconstructor

logger = logging.getLogger(__name__)


class ReActParser:
    """Parser for ReAct responses with pluggable healing strategies."""

    def __init__(self, strict_validation: bool = True, strategies: List[HealingStrategy] = None):
        self.strict_validation = strict_validation
        self.healing_attempts = 0
        self.strategies = strategies or [
            JsonBlockExtractor(),
            CommonErrorFixer(),
            RegexReconstructor(),
        ]
    
    def parse_response(self, response: str, step_number: int) -> ParseResult:
        """Parse LLM response into structured ReActStep."""
        original_response = response.strip()
        self.healing_attempts = 0
        try:
            return self._parse_json(original_response, original_response)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Initial parsing failed for step {step_number}: {e}")
        for strategy in self.strategies:
            self.healing_attempts += 1
            try:
                healing_result = strategy.heal(response)

                # Handle new HealingResult format
                if hasattr(healing_result, 'content'):
                    healed_json = healing_result.content
                    confidence = healing_result.confidence
                    strategy_name = healing_result.strategy_name
                else:
                    # Fallback for old string format
                    healed_json = healing_result
                    confidence = max(0.5, 1.0 - (self.healing_attempts * 0.2))
                    strategy_name = strategy.get_name()

                if not healing_result.success if hasattr(healing_result, 'success') else True:
                    continue

                result = self._parse_json(healed_json, original_response)
                result.was_healed = True
                result.healing_applied = strategy_name
                result.confidence = confidence

                logger.info(f"Successfully healed response for step {step_number} using {strategy_name}")
                return result

            except Exception as e:
                strategy_name = strategy.get_name() if hasattr(strategy, 'get_name') else str(strategy)
                logger.debug(f"Strategy {strategy_name} failed for step {step_number}: {e}")
                continue

        # All strategies failed
        raise ReActParsingError(
            f"Failed to parse ReAct response after trying {len(self.strategies)} healing strategies. "
            f"Original response: {original_response[:200]}..."
        )
    
    def _parse_json(self, json_str: str, original: str) -> ParseResult:
        """Parse JSON string into ParseResult."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ReActParsingError(f"Invalid JSON: {e}")
        
        # Validate required fields
        if "thought" not in data:
            raise ReActParsingError("Missing required field: thought")
        if "action_type" not in data:
            raise ReActParsingError("Missing required field: action_type")
        
        action_type = data["action_type"]
        if action_type not in ["tool_call", "final_answer"]:
            raise ReActParsingError(f"Invalid action_type: {action_type}")
        
        # Validate conditional fields
        if action_type == "tool_call":
            if "action" not in data:
                raise ReActParsingError("Missing 'action' for tool_call")
            if "action_input" not in data:
                raise ReActParsingError("Missing 'action_input' for tool_call")
        elif action_type == "final_answer":
            if "answer" not in data:
                raise ReActParsingError("Missing 'answer' for final_answer")
        
        return ParseResult(
            thought=str(data["thought"]).strip(),
            action_type=action_type,
            action=data.get("action"),
            action_input=data.get("action_input"),
            answer=data.get("answer"),
            original_response=original
        )

    
    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate parsed data against ReAct schema."""
        try:
            import jsonschema
            jsonschema.validate(data, REACT_RESPONSE_SCHEMA)
            return True
        except ImportError:
            logger.warning("jsonschema not available, skipping schema validation")
            return True
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
            return False