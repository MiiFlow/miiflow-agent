"""Tests pinning that __description is *required* on every tool schema.

The UI renders the model-provided phrase as a status label and (when
approval is needed) as the consent question. Marking __description
required guarantees every tool call carries a user-readable label
instead of falling back to bare tool names like ``tool_search`` or
``search_memory``.

Earlier the field was optional because a too-vague schema description
led the model to either skip it or emit perfunctory labels ("Calling
tool"). The schema description now pins __description to THIS specific
call's action with per-tool examples, which removes the perfunctory-
label failure mode and makes the required contract net-positive.
"""

from unittest.mock import MagicMock

from miiflow_agent.core.react.tool_executor import AgentToolExecutor


def _executor():
    """Build an AgentToolExecutor with a stub agent (we only exercise the
    schema-injection helper, no tool execution)."""
    agent = MagicMock()
    agent.tool_registry = MagicMock()
    return AgentToolExecutor(agent)


def test_description_param_property_is_present():
    """The UI label hook is still wired into every tool schema."""
    executor = _executor()
    schema = executor._inject_description_param({
        "name": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    })
    props = schema["parameters"]["properties"]
    assert "__description" in props
    assert props["__description"]["type"] == "string"
    # Description should explicitly tell the model the field is required.
    assert "required" in props["__description"]["description"].lower()


def test_description_param_is_required():
    """Required-by-default ensures every tool call gets a user-readable
    label rather than a bare tool name in the UI."""
    executor = _executor()
    schema = executor._inject_description_param({
        "name": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    })
    assert "__description" in schema["parameters"]["required"]


def test_existing_required_fields_preserved():
    """Marking __description required must not drop the tool's own
    required parameters."""
    executor = _executor()
    schema = executor._inject_description_param({
        "name": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    })
    assert "q" in schema["parameters"]["required"]
    assert "__description" in schema["parameters"]["required"]


def test_description_not_double_added_when_already_required():
    """Re-injecting on a schema that already has __description required
    should not duplicate the entry."""
    executor = _executor()
    schema = executor._inject_description_param({
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["__description", "q"],
        },
    })
    required = schema["parameters"]["required"]
    assert required.count("__description") == 1
    assert "q" in required


def test_schema_without_required_array_gets_one():
    """If a tool's schema has no required[] (uncommon but possible),
    inject one with __description in it."""
    executor = _executor()
    schema = executor._inject_description_param({
        "name": "noop",
        "parameters": {"type": "object", "properties": {}},
    })
    assert schema["parameters"]["required"] == ["__description"]
