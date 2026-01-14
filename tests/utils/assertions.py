"""Custom assertions for testing."""

import json
from typing import Any, Dict, List
import pytest


def assert_valid_json(content: str) -> Dict[str, Any]:
    """Validate and parse JSON content."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON: {e}\nContent: {content[:200]}...")


def assert_schema_compliance(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate data against JSON schema."""
    required_fields = schema.get("required", [])
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    properties = schema.get("properties", {})
    for field, props in properties.items():
        if field in data:
            expected_type = props.get("type")
            actual_value = data[field]
            
            if expected_type == "string":
                assert isinstance(actual_value, str), f"Field '{field}' should be string"
            elif expected_type == "number":
                assert isinstance(actual_value, (int, float)), f"Field '{field}' should be number"
            elif expected_type == "integer":
                assert isinstance(actual_value, int), f"Field '{field}' should be integer"
            elif expected_type == "boolean":
                assert isinstance(actual_value, bool), f"Field '{field}' should be boolean"
            elif expected_type == "array":
                assert isinstance(actual_value, list), f"Field '{field}' should be array"
            elif expected_type == "object":
                assert isinstance(actual_value, dict), f"Field '{field}' should be object"


def assert_has_tool_calls(response, min_calls: int = 1) -> None:
    """Validate response contains tool calls."""
    assert hasattr(response.message, 'tool_calls'), "Response missing tool_calls"
    assert response.message.tool_calls is not None, "tool_calls is None"
    assert len(response.message.tool_calls) >= min_calls, \
        f"Expected >={min_calls} tool calls, got {len(response.message.tool_calls)}"


def assert_streaming_chunks(chunks: List, min_chunks: int = 1) -> None:
    """Validate streaming chunks."""
    assert len(chunks) >= min_chunks, f"Expected >={min_chunks} chunks, got {len(chunks)}"
    
    if chunks:
        final_chunk = chunks[-1]
        assert hasattr(final_chunk, 'finish_reason'), "Final chunk missing finish_reason"
        assert final_chunk.finish_reason is not None, "Final chunk finish_reason is None"


def assert_content_accumulation(chunks: List) -> None:
    """Validate content accumulates correctly in streaming."""
    previous_content = ""
    for i, chunk in enumerate(chunks):
        if chunk.content:
            assert chunk.content.startswith(previous_content), \
                f"Chunk {i}: Content accumulation failed"
            previous_content = chunk.content


def assert_response_valid(response, check_usage: bool = True) -> None:
    """Validate basic response structure."""
    assert response is not None, "Response is None"
    assert hasattr(response, 'message'), "Response missing message"
    assert response.message is not None, "Response message is None"
    assert hasattr(response.message, 'content'), "Message missing content"
    
    if check_usage:
        assert hasattr(response, 'usage'), "Response missing usage"
        if response.usage:
            assert response.usage.total_tokens > 0, "Usage tokens should be > 0"
