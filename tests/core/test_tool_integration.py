"""Tests for the tool schema integration system."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from miiflow_llm.core.tools import (
    ToolSchema, ParameterSchema, ToolType, ToolRegistry, 
    FunctionTool, tool, ParameterType
)
from miiflow_llm.core.client import LLMClient
from miiflow_llm.core.message import Message, MessageRole
from miiflow_llm.providers.openai_client import OpenAIClient
from miiflow_llm.providers.anthropic_client import AnthropicClient
from miiflow_llm.providers.gemini_client import GeminiClient


class TestToolSchemaConversion:
    """Test universal tool schema conversion system."""
    
    def test_universal_schema_generation(self):
        """Test ToolSchema.to_universal_schema() method."""
        parameters = {
            "query": ParameterSchema(
                name="query",
                type=ParameterType.STRING, 
                description="Search query",
                required=True
            ),
            "limit": ParameterSchema(
                name="limit",
                type=ParameterType.INTEGER,
                description="Result limit",
                required=False,
                default=10
            )
        }
        
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.FUNCTION,
            parameters=parameters
        )
        
        universal = schema.to_universal_schema()
        
        assert universal["name"] == "test_tool"
        assert universal["description"] == "A test tool"
        assert universal["parameters"]["type"] == "object"
        assert "query" in universal["parameters"]["properties"]
        assert "limit" in universal["parameters"]["properties"]
        assert universal["parameters"]["required"] == ["query"]
        assert universal["parameters"]["properties"]["limit"]["default"] == 10
    
    def test_provider_specific_conversion(self):
        """Test provider-specific schema conversion using client methods."""
        parameters = {
            "location": ParameterSchema(
                name="location",
                type=ParameterType.STRING,
                description="City name",
                required=True
            )
        }
        
        schema = ToolSchema(
            name="weather_tool",
            description="Get weather info",
            tool_type=ToolType.FUNCTION,
            parameters=parameters
        )
        
        universal = schema.to_universal_schema()
        
        # Test OpenAI conversion
        openai_client = OpenAIClient(model="gpt-4o-mini", api_key="test")
        openai_format = openai_client.convert_schema_to_provider_format(universal)
        
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "weather_tool"
        assert openai_format["function"]["description"] == "Get weather info"
        assert "location" in openai_format["function"]["parameters"]["properties"]
        assert openai_format["function"]["parameters"]["required"] == ["location"]
        
        # Test Anthropic conversion
        anthropic_client = AnthropicClient(model="claude-3-haiku-20240307", api_key="test")
        anthropic_format = anthropic_client.convert_schema_to_provider_format(universal)
        
        assert anthropic_format["name"] == "weather_tool"
        assert anthropic_format["description"] == "Get weather info"
        assert anthropic_format["input_schema"]["type"] == "object"
        assert "location" in anthropic_format["input_schema"]["properties"]
        assert anthropic_format["input_schema"]["required"] == ["location"]
        
        # Test Gemini conversion
        gemini_client = GeminiClient(model="gemini-1.5-flash", api_key="test")
        gemini_format = gemini_client.convert_schema_to_provider_format(universal)
        
        assert gemini_format["name"] == "weather_tool"
        assert gemini_format["description"] == "Get weather info"
        assert "location" in gemini_format["parameters"]["properties"]


class TestEnhancedToolRegistry:
    """Test the enhanced ToolRegistry with universal conversion."""
    
    def test_get_tool_schemas_with_provider(self):
        """Test ToolRegistry.get_tool_schemas() with provider parameter."""
        registry = ToolRegistry()
        
        # Create a simple test function with explicit name parameter
        @tool("test_func", "Test function")
        def test_function(param: str) -> str:
            return f"Result: {param}"
        
        # Register the function directly - the decorator should use the explicit name
        func_tool = FunctionTool(test_function, name="test_func")
        registry.register(func_tool)
        
        # Test different provider formats
        openai_schemas = registry.get_schemas("openai")
        assert len(openai_schemas) == 1
        assert openai_schemas[0]["type"] == "function"
        assert openai_schemas[0]["function"]["name"] == "test_func"
        
        anthropic_schemas = registry.get_schemas("anthropic")
        assert len(anthropic_schemas) == 1
        assert anthropic_schemas[0]["name"] == "test_func"
        assert "input_schema" in anthropic_schemas[0]
        
        gemini_schemas = registry.get_schemas("gemini")
        assert len(gemini_schemas) == 1
        assert gemini_schemas[0]["name"] == "test_func"
        assert "parameters" in gemini_schemas[0]


class TestLLMClientToolIntegration:
    """Test tool integration in LLMClient."""
    
    def test_extract_tool_name_helper(self):
        """Test _extract_tool_name helper method."""
        # Create a real LLMClient for testing
        mock_client = MagicMock()
        mock_client.provider_name = "openai"
        llm_client = LLMClient(mock_client)
        
        # OpenAI format
        openai_schema = {
            "type": "function",
            "function": {"name": "test_tool"}
        }
        assert llm_client._extract_tool_name(openai_schema) == "test_tool"
        
        # Anthropic/Gemini format
        anthropic_schema = {
            "name": "another_tool",
            "description": "Test"
        }
        assert llm_client._extract_tool_name(anthropic_schema) == "another_tool"


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_cross_provider_compatibility(self):
        """Test that tool schemas work across different providers."""
        # Create a tool with explicit name
        @tool("universal_tool", "Works with any provider")
        def universal_function(input_text: str, count: int = 1) -> str:
            return input_text * count
        
        tool_instance = FunctionTool(universal_function, name="universal_tool")
        registry = ToolRegistry()
        registry.register(tool_instance)
        
        # Test across multiple providers  
        providers_to_test = ["openai", "anthropic", "gemini"]
        
        for provider in providers_to_test:
            schemas = registry.get_schemas(provider)
            assert len(schemas) == 1
            
            schema = schemas[0]
            # Each provider format should contain the tool name
            if provider == "openai":
                assert schema["function"]["name"] == "universal_tool"
            else:
                assert schema["name"] == "universal_tool"
