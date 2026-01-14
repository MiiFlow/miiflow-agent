"""Unified streaming with structured output parsing (clean architecture)."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union, AsyncGenerator
from datetime import datetime

from .message import Message
from .metrics import TokenCount
from .exceptions import ParsingError


@dataclass
class StreamChunk:
    """Chunk from a streaming response."""
    content: str
    delta: str
    finish_reason: Optional[str] = None
    usage: Optional[TokenCount] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class EnhancedStreamChunk:
    """Enhanced stream chunk with structured parsing support."""
    
    content: str
    delta: str = ""
    is_complete: bool = False
    partial_parse: Optional[Dict[str, Any]] = None
    structured_output: Optional[Any] = None
    finish_reason: Optional[str] = None
    usage: Optional[TokenCount] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.delta and self.content:
            self.delta = self.content


class IncrementalParser:
    """Parses structured output incrementally from streaming content."""
    
    def __init__(self, schema: Optional[Type] = None):
        self.schema = schema
        self.buffer = ""
        self.partial_objects: List[Dict[str, Any]] = []
    
    def try_parse_partial(self, new_content: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse partial structured output from accumulated content."""
        self.buffer += new_content
        
        # Try to extract complete JSON objects
        complete_objects = self._extract_complete_json_objects(self.buffer)
        
        if complete_objects:
            # Return the most recent complete object
            latest_object = complete_objects[-1]
            self.partial_objects.extend(complete_objects)
            return latest_object
        
        # Try to parse incomplete JSON for preview
        return self._attempt_partial_json_parse(self.buffer)
    
    def finalize_parse(self, complete_text: str) -> Optional[Any]:
        """Final parsing attempt with fallback strategies."""
        if complete_text.strip():
            self.buffer = complete_text
        
        # Strategy 1: Try direct JSON parsing
        try:
            return json.loads(self.buffer.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from text
        json_match = self._extract_json_from_text(self.buffer)
        if json_match:
            try:
                return json.loads(json_match)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Use regex patterns for common structures
        return self._fallback_regex_parse(self.buffer)
    
    def _extract_complete_json_objects(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete JSON objects from text buffer."""
        objects = []
        
        # Find potential JSON objects using braces
        brace_stack = []
        start_pos = None
        
        for i, char in enumerate(text):
            if char == '{':
                if not brace_stack:
                    start_pos = i
                brace_stack.append(char)
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_pos is not None:
                        # Found complete object
                        json_str = text[start_pos:i+1]
                        try:
                            obj = json.loads(json_str)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            continue
        
        return objects
    
    def _attempt_partial_json_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse incomplete JSON for preview."""
        # Look for partial objects that might be parseable
        text = text.strip()
        
        # Simple heuristic: if we have opening brace and some content
        if text.startswith('{') and len(text) > 2:
            # Try to close the JSON and parse
            attempts = [
                text + '}',
                text + '"}',
                text + '"}}'
            ]
            
            for attempt in attempts:
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON content from mixed text."""
        # Pattern to find JSON objects in text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            # Return the longest match (most likely to be complete)
            return max(matches, key=len)
        
        return None
    
    def _fallback_regex_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Fallback parsing using regex patterns."""
        result = {}
        
        # Common patterns for key-value extraction
        patterns = [
            r'"(\w+)":\s*"([^"]+)"',  # "key": "value"
            r'"(\w+)":\s*(\d+(?:\.\d+)?)',  # "key": number
            r'"(\w+)":\s*(true|false|null)',  # "key": boolean/null
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                # Convert value types
                if value.lower() == 'true':
                    result[key] = True
                elif value.lower() == 'false':
                    result[key] = False
                elif value.lower() == 'null':
                    result[key] = None
                elif value.replace('.', '').isdigit():
                    result[key] = float(value) if '.' in value else int(value)
                else:
                    result[key] = value
        
        return result if result else None


class UnifiedStreamingClient:
    """Unified streaming client with structured output support."""
    
    def __init__(self, client):
        self.client = client
    
    async def stream_with_schema(
        self,
        messages: List[Message],
        schema: Optional[Type] = None,
        **kwargs
    ) -> AsyncGenerator[EnhancedStreamChunk, None]:
        """Stream with incremental structured output parsing.
        
        Note: Client is expected to already normalize chunks to StreamChunk format.
        """
        
        parser = IncrementalParser(schema) if schema else None
        buffer = ""
        
        try:
            async for stream_chunk in self.client.astream_chat(messages, **kwargs):
                buffer += stream_chunk.delta if stream_chunk.delta else ""
                
                partial_parse = None
                if parser and stream_chunk.delta:
                    partial_parse = parser.try_parse_partial(stream_chunk.delta)
                
                yield EnhancedStreamChunk(
                    content=buffer,
                    delta=stream_chunk.delta,
                    is_complete=stream_chunk.finish_reason is not None,
                    partial_parse=partial_parse,
                    finish_reason=stream_chunk.finish_reason,
                    usage=stream_chunk.usage,
                    tool_calls=stream_chunk.tool_calls,
                    metadata={
                        "provider": self.client.provider_name,
                        "buffer_length": len(buffer),
                        "has_partial_parse": partial_parse is not None
                    }
                )
                
                if stream_chunk.finish_reason:
                    break
            
            if parser and buffer:
                final_result = parser.finalize_parse(buffer)
                yield EnhancedStreamChunk(
                    content=buffer,
                    delta="",
                    is_complete=True,
                    structured_output=final_result,
                    metadata={
                        "final_parse": True,
                        "parse_success": final_result is not None
                    }
                )
                
        except Exception as e:
            yield EnhancedStreamChunk(
                content="",
                delta="",
                is_complete=True,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
