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


def assert_output_cap_map_matches_catalog(models, parameters, field_name: str) -> None:
    """Every catalogued model declares its own output cap, never the fallback.

    Each provider module keeps a hand-written per-model ``max_value`` map
    beside its catalog, and every one of those maps has a ``default`` that
    matches no model. A model added to the catalog but left out of the map
    inherits that default; `ImportLLMModelsFromProviders.read_value` falls back
    to it and `load_llm_models` writes it to the `LLMModel` row as the model's
    real limit. NOTHING FAILS when that happens, which is how two GPT-6.2
    entries shipped briefly capped at 100,000 against their real 128,000.

    The defaults miss in BOTH directions, so the symptom is provider-specific:
    OpenAI's 100000 and Anthropic's 8192 are below their catalogues and
    TRUNCATE answers, while Google's 65536 is above Carbon's and Argon's 32768
    and OVER-ADVERTISES, turning requests the UI permits into provider 400s.
    Either way the number is wrong and silent, which is why this is a test
    rather than a comment.

    Shared rather than copied per module so that adding a provider is the only
    thing a future audit has to get right — near-identical copies of this body
    in each provider's test module were what let Groq be left out. Groq is the
    one provider in this package NOT checked here, because its map already
    diverges; `test_groq_output_cap_map_still_diverges_exactly_as_recorded` in
    `tests/models/test_output_cap_coverage.py` pins that divergence and records
    what finishing it requires.

    Args:
        models: the provider's catalog, ``{name: ModelConfig}``.
        parameters: the provider's ``ParameterConfig`` list.
        field_name: the token parameter whose ``max_value`` map to check
            (``max_tokens``, ``max_completion_tokens`` or
            ``max_output_tokens``).
    """
    parameter = next(p for p in parameters if p.field_name == field_name)
    limits = parameter.max_value
    assert isinstance(limits, dict), (
        f"{field_name} carries no per-model map, so every model shares one "
        f"bound: {limits!r}"
    )
    for name, config in models.items():
        assert name in limits, (
            f"{name} is in the catalog but not in the {field_name} map, so it "
            f"inherits the {limits.get('default')!r} default against its real "
            f"{config.maximum_output_tokens}"
        )
        assert limits[name] == config.maximum_output_tokens, (
            f"{name}: {field_name} map says {limits[name]}, catalog says "
            f"{config.maximum_output_tokens}"
        )
