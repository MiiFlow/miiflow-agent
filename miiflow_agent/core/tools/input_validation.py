"""Constraint enforcement for tool inputs.

``ParameterSchema`` carries type, enum, minimum/maximum, pattern, and item
types — all of which were emitted to providers but never enforced client-side
(validation was presence-only). Providers only enforce them in strict mode,
so a model could pass ``limit="abc"`` and the failure surfaced wherever the
tool body happened to crash, with whatever message it happened to produce.

This module enforces exactly what ``ParameterSchema`` can express — nothing
more, so it stays complete by construction without a JSON-Schema dependency.
Philosophy matches the rest of the dispatch path: tolerant where intent is
unambiguous (numeric strings coerce, ints pass as numbers), loud and
model-readable where it isn't. All problems are collected into one error so
the model fixes everything in a single retry.

Enforcement is OPT-IN (``MIIFLOW_STRICT_TOOL_VALIDATION=1``). Deployed
schemas carry constraints that were declared but never enforced — a stale
enum, an ``integer`` param whose tool body happily takes ``"10,20"`` — and
providers only check them in strict mode, so turning enforcement on at
library-upgrade time rejects calls that worked yesterday, with the defect
in the schema rather than the call (retries cannot self-correct). Until a
deployment audits its schemas and opts in, violations are logged and the
value passes through, and only the two base-era rules apply: presence of
required parameters, and unknown parameters dropped before dispatch.

Coercion is separate from enforcement. A coercion that SUCCEEDS is a
repair, not a rejection, so it applies in both modes: `limit="10"` reaches
the tool as ``10`` and ``rows='[{...}]'`` as a list. Withholding a
successful coercion until strict mode only moved the failure downstream to
whatever could not name it.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .types import ParameterType

logger = logging.getLogger(__name__)


def strict_validation_enabled() -> bool:
    """Whether schema constraints are enforced (vs. logged)."""
    return os.getenv("MIIFLOW_STRICT_TOOL_VALIDATION", "0") == "1"

#: Types that are strings on the wire regardless of their semantic label.
_STRING_TYPES = (ParameterType.STRING, ParameterType.MEDIA, ParameterType.TEXT)


def _type_label(param_type: ParameterType) -> str:
    if param_type in _STRING_TYPES:
        return "string"
    return param_type.value


def _parse_json_string(value: Any, expected: type) -> Tuple[Any, bool]:
    """Decode a JSON-encoded string into ``expected``; return (value, ok).

    Models routinely hand structured parameters over as JSON *text* rather
    than as structure — ``rows="[{...}]"``, ``breakdowns='["age","gender"]'``.
    Scalars have always been coerced from their string form here; arrays and
    objects were not, so the string flowed through untouched and the failure
    surfaced somewhere downstream that could not name it (for `render_table`,
    a zod error in the browser). Same narrow rule as the scalar branches: the
    text must parse, and it must parse to the declared type.
    """
    if not isinstance(value, str):
        return value, False
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value, False
    if isinstance(parsed, expected):
        return parsed, True
    return value, False


def _coerce(param_type: ParameterType, value: Any) -> Tuple[Any, bool]:
    """Try to coerce ``value`` to ``param_type``; return (value, ok).

    Coercion is deliberately narrow — only conversions with unambiguous
    intent: numeric strings for numeric params, "true"/"false" for booleans,
    int where a number is expected. Anything else is a real type error the
    model should hear about.
    """
    if param_type in _STRING_TYPES:
        return (value, True) if isinstance(value, str) else (value, False)

    if param_type == ParameterType.INTEGER:
        if isinstance(value, bool):
            return value, False
        if isinstance(value, int):
            return value, True
        if isinstance(value, float) and value.is_integer():
            return int(value), True
        if isinstance(value, str):
            try:
                return int(value.strip()), True
            except ValueError:
                return value, False
        return value, False

    if param_type == ParameterType.NUMBER:
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            return value, True
        if isinstance(value, str):
            try:
                return float(value.strip()), True
            except ValueError:
                return value, False
        return value, False

    if param_type == ParameterType.BOOLEAN:
        if isinstance(value, bool):
            return value, True
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true", True
        return value, False

    if param_type == ParameterType.ARRAY:
        if isinstance(value, list):
            return value, True
        return _parse_json_string(value, list)

    if param_type == ParameterType.OBJECT:
        if isinstance(value, dict):
            return value, True
        return _parse_json_string(value, dict)

    if param_type == ParameterType.NULL:
        return (value, value is None)

    return value, True  # unknown label: don't block on what we can't check


def _check_constraints(schema, value: Any) -> List[str]:
    """Enum / range / pattern / item-type checks for an already-typed value."""
    problems: List[str] = []

    if schema.enum is not None and value not in schema.enum:
        allowed = ", ".join(repr(v) for v in schema.enum)
        problems.append(f"must be one of [{allowed}], got {value!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if schema.minimum is not None and value < schema.minimum:
            problems.append(f"must be >= {schema.minimum}, got {value}")
        if schema.maximum is not None and value > schema.maximum:
            problems.append(f"must be <= {schema.maximum}, got {value}")

    if schema.pattern is not None and isinstance(value, str):
        try:
            # JSON Schema `pattern` is an unanchored regex search.
            if re.search(schema.pattern, value) is None:
                problems.append(
                    f"must match pattern {schema.pattern!r}, got {value!r}"
                )
        except re.error:  # a bad pattern is a tool-author bug, not the model's
            logger.warning(
                "unparseable pattern %r on parameter %r; skipping check",
                schema.pattern,
                schema.name,
            )

    if isinstance(value, list) and schema.items:
        item_type_str = (
            schema.items.get("type") if isinstance(schema.items, dict) else None
        )
        if item_type_str:
            try:
                item_type = ParameterType(item_type_str)
            except ValueError:
                item_type = None
            if item_type is not None:
                for index, element in enumerate(value):
                    _, ok = _coerce(item_type, element)
                    if not ok:
                        problems.append(
                            f"element {index} must be {item_type_str}, "
                            f"got {type(element).__name__} {element!r}"
                        )
                        break  # one element message is enough to self-correct
    return problems


def validate_inputs_against(
    parameters: Dict[str, Any],
    kwargs: Dict[str, Any],
    *,
    enforce: Optional[bool] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate ``kwargs`` against a ``ParameterSchema`` map.

    Returns ``(validated, errors)``. Unknown parameters are DROPPED from
    ``validated`` (base behavior — models routinely hallucinate stray
    kwargs, and passing them to ``fn(**validated)`` turns a working call
    into a TypeError). Explicit ``None`` always passes through: whether the
    tool body accepts it is the body's contract, and rejecting it broke
    calls that worked before enforcement existed.

    ``enforce=None`` reads ``MIIFLOW_STRICT_TOOL_VALIDATION`` (default off).
    Off: presence of required params is the only hard rule; type/enum/range/
    pattern violations are logged and the call proceeds. On: violations are
    collected into errors — all of them, so the model fixes everything in a
    single retry. Unambiguous coercions apply in both modes; only whether a
    violation *raises* depends on ``enforce``.
    """
    if enforce is None:
        enforce = strict_validation_enabled()

    validated: Dict[str, Any] = {}
    errors: List[str] = []

    for name, schema in parameters.items():
        if name not in kwargs:
            if schema.required:
                errors.append(f"Missing required parameter: {name}")
            continue

        value = kwargs[name]
        if value is None:
            validated[name] = value
            continue

        coerced, ok = _coerce(schema.type, value)
        if not ok:
            message = (
                f"Parameter '{name}' must be {_type_label(schema.type)}, "
                f"got {type(value).__name__} {value!r}"
            )
            if enforce:
                errors.append(message)
            else:
                logger.warning("tool input violates schema (not enforced): %s", message)
                validated[name] = value
            continue

        constraint_problems = _check_constraints(schema, coerced)
        if constraint_problems:
            if enforce:
                errors.extend(
                    f"Parameter '{name}' {problem}" for problem in constraint_problems
                )
            else:
                for problem in constraint_problems:
                    logger.warning(
                        "tool input violates schema (not enforced): "
                        "parameter %r %s",
                        name,
                        problem,
                    )
                validated[name] = coerced
            continue

        validated[name] = coerced

    return validated, errors
