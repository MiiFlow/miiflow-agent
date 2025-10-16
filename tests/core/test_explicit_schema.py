"""Tests for explicit parameter schema support in @tool decorator."""

import pytest
from miiflow_llm.core.tools import tool, ParameterSchema, get_tool_from_function, get_tool_schema
from miiflow_llm.core.tools.types import ParameterType


def test_explicit_schema_overrides_reflection():
    """Test that explicit schema takes precedence over reflection."""

    @tool(
        name="test_tool",
        description="Test tool with explicit schema",
        parameters={
            "status": ParameterSchema(
                name="status",
                type=ParameterType.STRING,
                description="Status value",
                required=True,
                enum=["active", "inactive", "pending"]
            ),
            "count": ParameterSchema(
                name="count",
                type=ParameterType.INTEGER,
                description="Count value",
                required=False,
                default=10,
                minimum=1,
                maximum=100
            )
        }
    )
    def my_tool(ctx, status: str, count: int = 10):
        """This docstring should be ignored for params."""
        return {"status": status, "count": count}

    # Get tool and check schema
    tool_obj = get_tool_from_function(my_tool)
    assert tool_obj is not None

    schema = tool_obj.schema
    assert schema.name == "test_tool"
    assert schema.description == "Test tool with explicit schema"
    assert len(schema.parameters) == 2

    # Check enum parameter
    status_param = schema.parameters["status"]
    assert status_param.type == ParameterType.STRING
    assert status_param.enum == ["active", "inactive", "pending"]
    assert status_param.required == True
    assert status_param.description == "Status value"

    # Check integer parameter with range
    count_param = schema.parameters["count"]
    assert count_param.type == ParameterType.INTEGER
    assert count_param.default == 10
    assert count_param.minimum == 1
    assert count_param.maximum == 100
    assert count_param.required == False
    assert count_param.description == "Count value"


def test_reflection_fallback_when_no_explicit_schema():
    """Test that reflection still works when no explicit schema provided."""

    @tool(description="Test tool with reflection")
    def my_tool(ctx, name: str, age: int = 25):
        """
        Args:
            name: Person's name
            age: Person's age
        """
        return {"name": name, "age": age}

    tool_obj = get_tool_from_function(my_tool)
    schema = tool_obj.schema

    assert len(schema.parameters) == 2
    assert "name" in schema.parameters
    assert "age" in schema.parameters

    # Check that reflection picked up types
    name_param = schema.parameters["name"]
    assert name_param.type == ParameterType.STRING
    assert name_param.required == True

    age_param = schema.parameters["age"]
    assert age_param.type == ParameterType.INTEGER
    assert age_param.default == 25
    assert age_param.required == False


def test_partial_explicit_schema():
    """Test that we can specify explicit schema for some params and rely on reflection for others."""
    # Note: Currently it's all-or-nothing, but this documents the expected behavior

    @tool(
        parameters={
            "mode": ParameterSchema(
                name="mode",
                type=ParameterType.STRING,
                description="Operation mode",
                required=True,
                enum=["read", "write", "delete"]
            )
        }
    )
    def my_tool(ctx, mode: str):
        return {"mode": mode}

    tool_obj = get_tool_from_function(my_tool)
    schema = tool_obj.schema

    # Should only have the explicitly defined parameter
    assert len(schema.parameters) == 1
    assert "mode" in schema.parameters


def test_enum_in_provider_formats():
    """Test that enum is properly included in provider-specific formats."""

    @tool(
        parameters={
            "mode": ParameterSchema(
                name="mode",
                type=ParameterType.STRING,
                description="Operation mode",
                required=True,
                enum=["read", "write", "delete"]
            )
        }
    )
    def my_tool(ctx, mode: str):
        return {"mode": mode}

    tool_obj = get_tool_from_function(my_tool)

    # Test OpenAI format
    openai_format = tool_obj.schema.to_provider_format("openai")
    assert "function" in openai_format
    assert "parameters" in openai_format["function"]
    mode_param = openai_format["function"]["parameters"]["properties"]["mode"]
    assert mode_param["enum"] == ["read", "write", "delete"]
    assert mode_param["type"] == "string"

    # Test Anthropic format
    anthropic_format = tool_obj.schema.to_provider_format("anthropic")
    assert "input_schema" in anthropic_format
    mode_param = anthropic_format["input_schema"]["properties"]["mode"]
    assert mode_param["enum"] == ["read", "write", "delete"]

    # Test Gemini format
    gemini_format = tool_obj.schema.to_provider_format("gemini")
    assert "parameters" in gemini_format
    mode_param = gemini_format["parameters"]["properties"]["mode"]
    assert mode_param["enum"] == ["read", "write", "delete"]


def test_tags_with_explicit_schema():
    """Test that tags work correctly with explicit schemas."""

    @tool(
        tags=["crm", "account_management"],
        parameters={
            "entity_type": ParameterSchema(
                name="entity_type",
                type=ParameterType.STRING,
                description="Entity type",
                required=True,
                enum=["account", "contact"]
            )
        }
    )
    def my_tool(ctx, entity_type: str):
        return {"entity_type": entity_type}

    schema = get_tool_schema(my_tool)
    assert schema is not None
    assert "tags" in schema.metadata
    assert schema.metadata["tags"] == ["crm", "account_management"]


def test_return_schema():
    """Test that return_schema is properly stored in metadata."""

    return_spec = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "results": {"type": "array"}
        }
    }

    @tool(
        return_schema=return_spec,
        parameters={
            "query": ParameterSchema(
                name="query",
                type=ParameterType.STRING,
                description="Query string",
                required=True
            )
        }
    )
    def my_tool(ctx, query: str):
        return {"success": True, "results": []}

    schema = get_tool_schema(my_tool)
    assert schema is not None
    assert "return_schema" in schema.metadata
    assert schema.metadata["return_schema"] == return_spec


def test_complex_crm_tool_example():
    """Test a realistic CRM tool with complex explicit schema."""

    @tool(
        name="Query CRM Data",
        description="Query accounts, contacts, events, or signals with flexible JSON filters",
        tags=["crm", "account_management"],
        parameters={
            "entity_type": ParameterSchema(
                name="entity_type",
                type=ParameterType.STRING,
                description="Type of entity to query",
                required=True,
                enum=["account", "contact", "event", "signal"]
            ),
            "filters_json": ParameterSchema(
                name="filters_json",
                type=ParameterType.STRING,
                description='JSON string with filters. Example: {"name": "Acme"}',
                required=False
            ),
            "limit": ParameterSchema(
                name="limit",
                type=ParameterType.INTEGER,
                description="Maximum number of results",
                required=False,
                default=20,
                minimum=1,
                maximum=100
            ),
            "sort_by": ParameterSchema(
                name="sort_by",
                type=ParameterType.STRING,
                description="Field to sort by",
                required=False
            )
        }
    )
    def query_crm_data(ctx, entity_type: str, filters_json: str = None, limit: int = 20, sort_by: str = None):
        return {"entity_type": entity_type, "limit": limit}

    tool_obj = get_tool_from_function(query_crm_data)
    schema = tool_obj.schema

    # Verify schema structure
    assert schema.name == "Query CRM Data"
    assert len(schema.parameters) == 4
    assert "tags" in schema.metadata
    assert "crm" in schema.metadata["tags"]

    # Verify enum works
    entity_param = schema.parameters["entity_type"]
    assert entity_param.enum == ["account", "contact", "event", "signal"]

    # Verify range constraints
    limit_param = schema.parameters["limit"]
    assert limit_param.minimum == 1
    assert limit_param.maximum == 100
    assert limit_param.default == 20

    # Test OpenAI format includes all constraints
    openai_format = schema.to_provider_format("openai")
    params = openai_format["function"]["parameters"]

    # Check required fields
    assert set(params["required"]) == {"entity_type"}

    # Check enum is present
    assert params["properties"]["entity_type"]["enum"] == ["account", "contact", "event", "signal"]

    # Check range constraints
    assert params["properties"]["limit"]["minimum"] == 1
    assert params["properties"]["limit"]["maximum"] == 100
    assert params["properties"]["limit"]["default"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
