"""Tests for the centralized JSON schema normalizer."""

import copy

import pytest

from miiflow_agent.core.schema_normalizer import SchemaMode, normalize_json_schema


class TestSchemaModeStrict:
    """Tests for STRICT mode (OpenAI, OpenRouter, Mistral, xAI)."""

    def test_adds_additional_properties_false_to_object(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = normalize_json_schema(schema, SchemaMode.STRICT)
        assert result["additionalProperties"] is False

    def test_recursively_processes_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        result = normalize_json_schema(schema, SchemaMode.STRICT)
        assert result["additionalProperties"] is False
        assert result["properties"]["user"]["additionalProperties"] is False

    def test_recursively_processes_array_items(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                }
            },
        }
        result = normalize_json_schema(schema, SchemaMode.STRICT)
        assert result["additionalProperties"] is False
        assert result["properties"]["items"]["items"]["additionalProperties"] is False

    def test_ensure_all_required_adds_all_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
        }
        result = normalize_json_schema(schema, SchemaMode.STRICT, ensure_all_required=True)
        assert set(result["required"]) == {"name", "age", "email"}

    def test_ensure_all_required_merges_with_existing(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = normalize_json_schema(schema, SchemaMode.STRICT, ensure_all_required=True)
        assert set(result["required"]) == {"name", "age"}

    def test_ensure_all_required_on_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                    },
                }
            },
        }
        result = normalize_json_schema(schema, SchemaMode.STRICT, ensure_all_required=True)
        assert set(result["required"]) == {"user"}
        assert set(result["properties"]["user"]["required"]) == {"first_name", "last_name"}

    def test_empty_properties_no_empty_required(self):
        schema = {"type": "object", "properties": {}}
        result = normalize_json_schema(schema, SchemaMode.STRICT, ensure_all_required=True)
        # Should not add empty required array for empty properties
        assert "required" not in result or result.get("required") == []

    def test_does_not_modify_original_schema(self):
        original = {"type": "object", "properties": {"name": {"type": "string"}}}
        original_copy = copy.deepcopy(original)
        normalize_json_schema(original, SchemaMode.STRICT)
        assert original == original_copy


class TestSchemaModeLoose:
    """Tests for LOOSE mode (Anthropic fallback)."""

    def test_sets_additional_properties_true(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}},
        }
        result = normalize_json_schema(schema, SchemaMode.LOOSE)
        assert result["additionalProperties"] is True

    def test_only_modifies_existing_additional_properties(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = normalize_json_schema(schema, SchemaMode.LOOSE)
        # Should not add additionalProperties if it wasn't there
        assert "additionalProperties" not in result

    def test_recursively_processes_nested_objects(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "user": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        result = normalize_json_schema(schema, SchemaMode.LOOSE)
        assert result["additionalProperties"] is True
        assert result["properties"]["user"]["additionalProperties"] is True

    def test_handles_combinators(self):
        schema = {
            "allOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"a": {"type": "string"}},
                }
            ]
        }
        result = normalize_json_schema(schema, SchemaMode.LOOSE)
        assert result["allOf"][0]["additionalProperties"] is True


class TestSchemaModeNativeStrict:
    """Tests for NATIVE_STRICT mode (Anthropic structured outputs)."""

    def test_adds_additional_properties_false(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = normalize_json_schema(schema, SchemaMode.NATIVE_STRICT)
        assert result["additionalProperties"] is False

    def test_handles_all_of_combinator(self):
        schema = {
            "allOf": [{"type": "object", "properties": {"a": {"type": "string"}}}]
        }
        result = normalize_json_schema(schema, SchemaMode.NATIVE_STRICT)
        assert result["allOf"][0]["additionalProperties"] is False

    def test_handles_any_of_combinator(self):
        schema = {
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "integer"}}},
            ]
        }
        result = normalize_json_schema(schema, SchemaMode.NATIVE_STRICT)
        assert result["anyOf"][0]["additionalProperties"] is False
        assert result["anyOf"][1]["additionalProperties"] is False

    def test_handles_one_of_combinator(self):
        schema = {
            "oneOf": [{"type": "object", "properties": {"x": {"type": "boolean"}}}]
        }
        result = normalize_json_schema(schema, SchemaMode.NATIVE_STRICT)
        assert result["oneOf"][0]["additionalProperties"] is False


class TestSchemaModeGeminiCompat:
    """Tests for GEMINI_COMPAT mode."""

    def test_removes_additional_properties(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}},
        }
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert "additionalProperties" not in result

    def test_removes_schema_field(self):
        schema = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert "$schema" not in result

    def test_removes_definitions(self):
        schema = {
            "type": "object",
            "definitions": {"Name": {"type": "string"}},
            "properties": {"name": {"type": "string"}},
        }
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert "definitions" not in result

    def test_removes_defs(self):
        schema = {
            "type": "object",
            "$defs": {"Name": {"type": "string"}},
            "properties": {"name": {"type": "string"}},
        }
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert "$defs" not in result

    def test_removes_default(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "default": "John"}},
        }
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert "default" not in result["properties"]["name"]

    def test_converts_array_type_to_single_type(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": ["string", "null"]}},
        }
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert result["properties"]["name"]["type"] == "string"

    def test_converts_null_only_type_to_string(self):
        schema = {"type": "object", "properties": {"empty": {"type": ["null"]}}}
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert result["properties"]["empty"]["type"] == "string"

    def test_preserves_required_array(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert result["required"] == ["name"]

    def test_recursively_processes_nested_structures(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "user": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": ["string", "null"], "default": "Unknown"}
                    },
                }
            },
        }
        result = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
        assert "additionalProperties" not in result
        assert "additionalProperties" not in result["properties"]["user"]
        assert result["properties"]["user"]["properties"]["name"]["type"] == "string"
        assert "default" not in result["properties"]["user"]["properties"]["name"]


class TestSchemaModePassthrough:
    """Tests for PASSTHROUGH mode."""

    def test_returns_schema_unchanged(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}},
        }
        result = normalize_json_schema(schema, SchemaMode.PASSTHROUGH)
        assert result == schema

    def test_returns_deep_copy_by_default(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = normalize_json_schema(schema, SchemaMode.PASSTHROUGH)
        result["properties"]["name"]["type"] = "integer"
        assert schema["properties"]["name"]["type"] == "string"


class TestInPlaceModification:
    """Tests for in_place parameter."""

    def test_in_place_modifies_original(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = normalize_json_schema(schema, SchemaMode.STRICT, in_place=True)
        assert result is schema
        assert schema["additionalProperties"] is False

    def test_not_in_place_preserves_original(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = normalize_json_schema(schema, SchemaMode.STRICT, in_place=False)
        assert result is not schema
        assert "additionalProperties" not in schema


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_non_dict_input_returns_as_is(self):
        assert normalize_json_schema(None, SchemaMode.STRICT) is None
        assert normalize_json_schema("string", SchemaMode.STRICT) == "string"
        assert normalize_json_schema(123, SchemaMode.STRICT) == 123

    def test_top_level_array_schema(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
            },
        }
        result = normalize_json_schema(schema, SchemaMode.STRICT)
        assert result["items"]["additionalProperties"] is False

    def test_deeply_nested_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {"value": {"type": "string"}},
                                }
                            },
                        }
                    },
                }
            },
        }
        result = normalize_json_schema(schema, SchemaMode.STRICT)
        assert result["additionalProperties"] is False
        assert result["properties"]["level1"]["additionalProperties"] is False
        assert result["properties"]["level1"]["properties"]["level2"]["additionalProperties"] is False
        assert (
            result["properties"]["level1"]["properties"]["level2"]["properties"]["level3"][
                "additionalProperties"
            ]
            is False
        )

    def test_schema_with_no_type_field(self):
        schema = {"properties": {"name": {"type": "string"}}}
        result = normalize_json_schema(schema, SchemaMode.STRICT)
        # Should still process properties even without type field
        assert result["properties"]["name"]["type"] == "string"
