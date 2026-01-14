"""Centralized JSON schema normalization for LLM provider compatibility.

This module provides a unified approach to normalizing JSON schemas for different
LLM providers. Each provider has specific requirements for schema structure,
and this module handles those transformations in a consistent way.

Responsibilities:
- Provider-specific transformations (additionalProperties, required arrays, etc.)
- NOT responsible for custom type mappings (TIMESTAMP->string) - that's handled upstream

Usage:
    from miiflow_agent.core.schema_normalizer import normalize_json_schema, SchemaMode

    # For OpenAI
    normalized = normalize_json_schema(schema, SchemaMode.STRICT)

    # For OpenRouter (stricter requirements)
    normalized = normalize_json_schema(schema, SchemaMode.STRICT, ensure_all_required=True)

    # For Gemini
    normalized = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)
"""

import copy
from enum import Enum, auto
from typing import Any, Dict, List


class SchemaMode(Enum):
    """Defines how JSON schemas should be normalized for different provider requirements.

    Modes:
        STRICT: OpenAI-compatible strict mode
            - additionalProperties: false on all objects
            - Optionally ensures 'required' array includes all properties
            - Used by: OpenAI, OpenRouter, Mistral, xAI

        LOOSE: Relaxed constraints for better model compliance
            - additionalProperties: true on all objects
            - Used by: Anthropic fallback for older models

        NATIVE_STRICT: Anthropic native structured outputs
            - additionalProperties: false on all objects
            - Handles allOf/anyOf/oneOf combinators
            - Used by: Anthropic with structured outputs (Claude 3.5+)

        GEMINI_COMPAT: Google Gemini compatibility mode
            - Removes unsupported fields: additionalProperties, $schema,
              definitions, $defs, default
            - Converts array types ["string", "null"] to single types
            - Used by: Gemini

        PASSTHROUGH: No transformation applied
            - Returns schema as-is (deep copy unless in_place=True)
            - Used for testing or pre-normalized schemas
    """

    STRICT = auto()
    LOOSE = auto()
    NATIVE_STRICT = auto()
    GEMINI_COMPAT = auto()
    PASSTHROUGH = auto()


def normalize_json_schema(
    schema: Dict[str, Any],
    mode: SchemaMode,
    *,
    ensure_all_required: bool = False,
    in_place: bool = False,
) -> Dict[str, Any]:
    """Normalize a JSON schema according to the specified mode.

    Args:
        schema: The JSON schema to normalize
        mode: The normalization mode to apply
        ensure_all_required: If True, adds all properties to 'required' array.
                            Only applies to STRICT and NATIVE_STRICT modes.
                            Needed for OpenRouter strict mode.
        in_place: If True, modifies the schema in place. If False (default),
                  returns a deep copy with modifications.

    Returns:
        Normalized JSON schema (either modified copy or original if in_place=True)

    Example:
        # For OpenAI
        normalized = normalize_json_schema(schema, SchemaMode.STRICT)

        # For OpenRouter (stricter)
        normalized = normalize_json_schema(
            schema,
            SchemaMode.STRICT,
            ensure_all_required=True
        )

        # For Gemini
        normalized = normalize_json_schema(schema, SchemaMode.GEMINI_COMPAT)

        # For Anthropic older models
        normalized = normalize_json_schema(schema, SchemaMode.LOOSE)
    """
    if not isinstance(schema, dict):
        return schema

    # Make a deep copy unless in_place is True
    if not in_place:
        schema = copy.deepcopy(schema)

    if mode == SchemaMode.PASSTHROUGH:
        return schema
    elif mode == SchemaMode.STRICT:
        _normalize_strict(schema, ensure_all_required=ensure_all_required)
        return schema
    elif mode == SchemaMode.NATIVE_STRICT:
        _normalize_native_strict(schema, ensure_all_required=ensure_all_required)
        return schema
    elif mode == SchemaMode.LOOSE:
        _normalize_loose(schema)
        return schema
    elif mode == SchemaMode.GEMINI_COMPAT:
        # GEMINI_COMPAT builds a new dict (removes fields), so we return that
        return _normalize_gemini_compat(schema)
    else:
        raise ValueError(f"Unknown schema mode: {mode}")


def _normalize_strict(obj: Dict[str, Any], ensure_all_required: bool = False) -> None:
    """Apply STRICT mode transformations in-place.

    - Sets additionalProperties: false on all object types
    - Optionally ensures all properties are in 'required' array
    - Recursively processes: properties, items
    """
    if not isinstance(obj, dict):
        return

    if obj.get("type") == "object":
        obj["additionalProperties"] = False

        # Optionally add all properties to required array
        if ensure_all_required and "properties" in obj and obj["properties"]:
            existing_required = set(obj.get("required", []))
            all_props = set(obj["properties"].keys())
            # Merge: keep existing required and add any missing properties
            obj["required"] = list(existing_required | all_props)

    # Recursively process nested schemas
    if "properties" in obj and isinstance(obj["properties"], dict):
        for prop_value in obj["properties"].values():
            _normalize_strict(prop_value, ensure_all_required=ensure_all_required)

    if "items" in obj and isinstance(obj["items"], dict):
        _normalize_strict(obj["items"], ensure_all_required=ensure_all_required)


def _normalize_native_strict(obj: Dict[str, Any], ensure_all_required: bool = False) -> None:
    """Apply NATIVE_STRICT mode transformations in-place.

    Same as STRICT but also handles allOf/anyOf/oneOf combinators.
    Used for Anthropic native structured outputs.

    - Sets additionalProperties: false on all object types
    - Optionally ensures all properties are in 'required' array
    - Recursively processes: properties, items, allOf, anyOf, oneOf
    """
    if not isinstance(obj, dict):
        return

    if obj.get("type") == "object":
        obj["additionalProperties"] = False

        # Optionally add all properties to required array
        if ensure_all_required and "properties" in obj and obj["properties"]:
            existing_required = set(obj.get("required", []))
            all_props = set(obj["properties"].keys())
            obj["required"] = list(existing_required | all_props)

    # Recursively process nested schemas
    if "properties" in obj and isinstance(obj["properties"], dict):
        for prop_value in obj["properties"].values():
            _normalize_native_strict(prop_value, ensure_all_required=ensure_all_required)

    if "items" in obj and isinstance(obj["items"], dict):
        _normalize_native_strict(obj["items"], ensure_all_required=ensure_all_required)

    # Handle combinators (Anthropic-specific requirement)
    for combinator in ["allOf", "anyOf", "oneOf"]:
        if combinator in obj and isinstance(obj[combinator], list):
            for item in obj[combinator]:
                if isinstance(item, dict):
                    _normalize_native_strict(item, ensure_all_required=ensure_all_required)


def _normalize_loose(obj: Dict[str, Any]) -> None:
    """Apply LOOSE mode transformations in-place.

    - Sets additionalProperties: true on all object types
    - Recursively processes: properties, items, allOf, anyOf, oneOf
    """
    if not isinstance(obj, dict):
        return

    if obj.get("type") == "object" and "additionalProperties" in obj:
        obj["additionalProperties"] = True

    # Recursively process nested schemas
    if "properties" in obj and isinstance(obj["properties"], dict):
        for prop_value in obj["properties"].values():
            _normalize_loose(prop_value)

    if "items" in obj and isinstance(obj["items"], dict):
        _normalize_loose(obj["items"])

    # Handle combinators
    for combinator in ["allOf", "anyOf", "oneOf"]:
        if combinator in obj and isinstance(obj[combinator], list):
            for item in obj[combinator]:
                if isinstance(item, dict):
                    _normalize_loose(item)


def _normalize_gemini_compat(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Apply GEMINI_COMPAT mode transformations, returning new dict.

    - Removes unsupported fields: additionalProperties, $schema,
      definitions, $defs, default
    - Converts array types to single types (e.g., ["string", "null"] -> "string")
    - Recursively processes all nested structures
    """
    if not isinstance(obj, dict):
        return obj

    # Fields that Gemini doesn't support
    unsupported_fields = {
        "additionalProperties",
        "$schema",
        "definitions",
        "$defs",
        "default",
        # Numeric constraints not supported by Gemini Schema
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        # String constraints not supported by Gemini Schema
        "minLength",
        "maxLength",
        "pattern",
        # Array constraints not supported by Gemini Schema
        "minItems",
        "maxItems",
        "uniqueItems",
        # Object constraints not supported by Gemini Schema
        "minProperties",
        "maxProperties",
        # Other unsupported fields
        "examples",
        "const",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$id",
        "$ref",
        "$comment",
    }

    normalized: Dict[str, Any] = {}

    for key, value in obj.items():
        # Skip unsupported fields
        if key in unsupported_fields:
            continue

        if key == "type":
            # Convert array types to single type
            if isinstance(value, list):
                # Filter out "null" and take the first non-null type
                non_null_types = [t for t in value if t != "null"]
                if non_null_types:
                    normalized[key] = non_null_types[0]
                else:
                    # If only "null", default to "string"
                    normalized[key] = "string"
            else:
                normalized[key] = value

        elif key == "properties" and isinstance(value, dict):
            # Recursively normalize nested properties
            normalized[key] = {
                prop_key: _normalize_gemini_compat(prop_value)
                for prop_key, prop_value in value.items()
            }

        elif key == "items" and isinstance(value, dict):
            # Recursively normalize array items
            normalized[key] = _normalize_gemini_compat(value)

        elif key == "required" and isinstance(value, list):
            # Keep required fields list as-is
            normalized[key] = value

        elif isinstance(value, dict):
            # Recursively normalize nested objects
            normalized[key] = _normalize_gemini_compat(value)

        elif isinstance(value, list):
            # Recursively normalize lists
            normalized[key] = [
                _normalize_gemini_compat(item) if isinstance(item, dict) else item
                for item in value
            ]

        else:
            normalized[key] = value

    return normalized
