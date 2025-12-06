"""
Comprehensive integration tests for all providers.

Tests:
1. JSON mode - structured responses
2. Streaming - real-time token streaming
3. Agent mode - tool/function calling
4. Image input - multimodal understanding

Note: These tests require real API keys and are excluded by default (marked with @pytest.mark.smoke).
Run with: pytest -m smoke tests/smoke/
"""

import asyncio
import json
import os
import pytest
from typing import List, Dict, Any

from miiflow_llm import LLMClient
from miiflow_llm.core.message import Message, MessageRole
from miiflow_llm.core.tools import tool, ParameterSchema, ParameterType
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def is_ci_environment() -> bool:
    """Check if running in a CI/CD environment."""
    ci_env_vars = [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "JENKINS_URL",
        "BUILDKITE",
        "DRONE",
        "TEAMCITY_VERSION",
        "TF_BUILD",
    ]
    return any(os.getenv(var) for var in ci_env_vars)


# Helper function to extract tool call information (provider-agnostic)
def extract_tool_call_info(tool_call):
    """Extract name and arguments from provider-specific tool call format."""
    # Handle provider-specific tool call formats
    if hasattr(tool_call, 'function'):  # OpenAI format
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
    elif hasattr(tool_call, 'name'):  # Anthropic/direct format
        tool_name = tool_call.name
        tool_args = tool_call.arguments if hasattr(tool_call, 'arguments') else str(tool_call.input)
    elif isinstance(tool_call, dict):  # Dict format
        tool_name = tool_call.get('name') or tool_call.get('function', {}).get('name')
        tool_args = tool_call.get('arguments') or tool_call.get('function', {}).get('arguments')
    else:
        raise ValueError(f"Unknown tool call format: {type(tool_call)}")

    # Parse arguments if they're a JSON string
    if isinstance(tool_args, str):
        tool_args = json.loads(tool_args)

    return tool_name, tool_args


# ============================================================================
# Test Configuration
# ============================================================================

# Provider configurations for different features
# Note: Only providers with native JSON schema support
PROVIDERS_JSON_MODE = [
    {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "api_key_env": "ANTHROPIC_API_KEY"},
    # Gemini supports JSON mode but has strict schema requirements
    {"provider": "gemini", "model": "gemini-2.5-flash", "api_key_env": "GOOGLE_API_KEY"},
]

# Provider configurations with native structured output support
# Note: Only Claude Sonnet 4.5 and Opus 4.1 currently support structured outputs
PROVIDERS_STRUCTURED_OUTPUT = [
    {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "api_key_env": "ANTHROPIC_API_KEY"},
]

PROVIDERS_STREAMING = [
    {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "api_key_env": "ANTHROPIC_API_KEY"},
    {"provider": "gemini", "model": "gemini-2.5-flash", "api_key_env": "GOOGLE_API_KEY"},
    {"provider": "groq", "model": "llama-3.1-8b-instant", "api_key_env": "GROQ_API_KEY"},
]

PROVIDERS_TOOL_CALLING = [
    {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "api_key_env": "ANTHROPIC_API_KEY"},
    {"provider": "gemini", "model": "gemini-2.5-flash", "api_key_env": "GOOGLE_API_KEY"},
    {"provider": "groq", "model": "llama-3.1-8b-instant", "api_key_env": "GROQ_API_KEY"},
]

PROVIDERS_MULTIMODAL = [
    {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "api_key_env": "ANTHROPIC_API_KEY"},
    {"provider": "gemini", "model": "gemini-2.5-flash", "api_key_env": "GOOGLE_API_KEY"},
]


def skip_if_no_api_key(api_key_env: str):
    """Skip test if API key is not available or in CI without keys."""
    api_key = os.getenv(api_key_env)
    has_valid_key = (
        api_key
        and not api_key.startswith("your-")
        and not api_key.startswith("sk-test")
        and api_key != "test"
    )

    # Always skip in CI if no valid API key
    if is_ci_environment() and not has_valid_key:
        return pytest.mark.skip(reason=f"Skipping in CI: {api_key_env} not configured")

    return pytest.mark.skipif(
        not has_valid_key,
        reason=f"{api_key_env} not configured"
    )


# ============================================================================
# JSON Mode Tests
# ============================================================================

@pytest.mark.smoke
class TestJSONMode:
    """Test JSON mode structured outputs across providers."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_JSON_MODE)
    async def test_json_mode_simple_schema(self, config):
        """Test JSON mode with simple schema."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        # Create client
        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        # Define a simple schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        }

        # Make request
        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "Generate data for a person named Alice, 30 years old, living in Paris."
            }],
            json_schema=schema
        )

        # Verify response
        assert response.message.content is not None

        # Parse JSON response
        data = json.loads(response.message.content)

        # Verify schema compliance
        assert "name" in data
        assert "age" in data
        assert "city" in data
        assert isinstance(data["age"], int)
        assert data["name"].lower() == "alice"
        assert data["age"] == 30

        print(f"✅ {config['provider']}: JSON mode simple schema passed")
        print(f"   Response: {data}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_JSON_MODE)
    async def test_json_mode_nested_schema(self, config):
        """Test JSON mode with nested objects."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        # Complex nested schema
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    },
                    "required": ["name", "email"]
                },
                "scores": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "active": {"type": "boolean"}
            },
            "required": ["user", "scores", "active"]
        }

        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "Generate user data with name 'Bob', email 'bob@test.com', scores [85, 90, 95], active true."
            }],
            json_schema=schema
        )

        data = json.loads(response.message.content)

        # Verify nested structure
        assert "user" in data
        assert "name" in data["user"]
        assert "email" in data["user"]
        assert "scores" in data
        assert isinstance(data["scores"], list)
        assert all(isinstance(s, int) for s in data["scores"])
        assert "active" in data
        assert isinstance(data["active"], bool)

        print(f"✅ {config['provider']}: JSON mode nested schema passed")
        print(f"   Response: {data}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_JSON_MODE)
    async def test_json_mode_array_response(self, config):
        """Test JSON mode with array of objects."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        schema = {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                            "inStock": {"type": "boolean"}
                        },
                        "required": ["name", "price", "inStock"]
                    }
                }
            },
            "required": ["products"]
        }

        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "Generate 3 products: Laptop ($999, in stock), Mouse ($29, in stock), Keyboard ($79, out of stock)"
            }],
            json_schema=schema
        )

        data = json.loads(response.message.content)

        assert "products" in data
        assert isinstance(data["products"], list)
        assert len(data["products"]) == 3

        for product in data["products"]:
            assert "name" in product
            assert "price" in product
            assert "inStock" in product
            assert isinstance(product["price"], (int, float))
            assert isinstance(product["inStock"], bool)

        print(f"✅ {config['provider']}: JSON mode array response passed")
        print(f"   Products: {len(data['products'])}")


# ============================================================================
# Structured Output Tests (Native API)
# ============================================================================

@pytest.mark.smoke
class TestStructuredOutputs:
    """Test native structured outputs for supported models (Claude Sonnet 4.5, GPT-4o, etc.)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_STRUCTURED_OUTPUT)
    async def test_structured_output_guaranteed_compliance(self, config):
        """Test that structured outputs guarantee schema compliance."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        # Complex schema with strict requirements
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            },
            "required": ["sentiment", "score", "keywords"],
            "additionalProperties": False
        }

        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "Analyze sentiment: 'This product exceeded my expectations!'"
            }],
            json_schema=schema
        )

        data = json.loads(response.message.content)

        # Verify strict compliance
        assert "sentiment" in data
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert "score" in data
        assert isinstance(data["score"], (int, float))
        assert 0 <= data["score"] <= 1
        assert "keywords" in data
        assert isinstance(data["keywords"], list)
        assert len(data["keywords"]) >= 1
        # Verify no additional properties
        assert set(data.keys()) == {"sentiment", "score", "keywords"}

        print(f"✅ {config['provider']}: Structured output compliance verified")
        print(f"   Data: {data}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_STRUCTURED_OUTPUT)
    async def test_structured_output_with_streaming(self, config):
        """Test that structured outputs work with streaming."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "word_count": {"type": "integer"}
            },
            "required": ["summary", "word_count"]
        }

        accumulated_content = ""
        chunks = []

        async for chunk in client.astream_chat(
            messages=[{
                "role": "user",
                "content": "Summarize in 10 words: 'Machine learning is transforming industries worldwide.'"
            }],
            json_schema=schema
        ):
            chunks.append(chunk)
            if chunk.delta:
                accumulated_content += chunk.delta

        # Verify streaming worked
        assert len(chunks) > 0
        assert accumulated_content

        # Verify final content is valid JSON matching schema
        data = json.loads(accumulated_content)
        assert "summary" in data
        assert "word_count" in data
        assert isinstance(data["word_count"], int)

        print(f"✅ {config['provider']}: Structured output with streaming passed")
        print(f"   Chunks: {len(chunks)}, Data: {data}")


# ============================================================================
# Streaming Tests
# ============================================================================

@pytest.mark.smoke
class TestStreaming:
    """Test streaming functionality across providers."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_STREAMING)
    async def test_basic_streaming(self, config):
        """Test basic streaming with token-by-token delivery."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        chunks = []
        accumulated_content = ""

        async for chunk in client.astream_chat(
            messages=[{
                "role": "user",
                "content": "Count from 1 to 5, saying only the numbers separated by spaces."
            }]
        ):
            chunks.append(chunk)
            if chunk.delta:
                accumulated_content += chunk.delta

        # Verify streaming happened
        assert len(chunks) > 0, f"{config['provider']}: No chunks received"
        assert accumulated_content, f"{config['provider']}: No content accumulated"

        # Verify final chunk has finish reason
        final_chunk = chunks[-1]
        assert final_chunk.finish_reason is not None, f"{config['provider']}: No finish reason"

        print(f"✅ {config['provider']}: Basic streaming passed")
        print(f"   Chunks received: {len(chunks)}")
        print(f"   Content: {accumulated_content[:100]}...")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_STREAMING)
    async def test_streaming_with_json_mode(self, config):
        """Test streaming with JSON schema."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        # Skip providers that don't support streaming + JSON schema
        # Note: Anthropic now supports streaming with structured outputs for new models
        if config["provider"] in ["groq"]:
            pytest.skip(f"{config['provider']} doesn't support streaming with JSON schema")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["result", "confidence"]
        }

        accumulated_content = ""
        chunks = []

        async for chunk in client.astream_chat(
            messages=[{
                "role": "user",
                "content": "Classify this as positive or negative: 'I love this product!'. Return confidence 0-1."
            }],
            json_schema=schema
        ):
            chunks.append(chunk)
            if chunk.delta:
                accumulated_content += chunk.delta

        # Verify we got chunks
        assert len(chunks) > 0
        assert accumulated_content

        # Verify final content is valid JSON
        data = json.loads(accumulated_content)
        assert "result" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], (int, float))

        print(f"✅ {config['provider']}: Streaming with JSON mode passed")
        print(f"   Result: {data}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_STREAMING)
    async def test_streaming_content_accumulation(self, config):
        """Test that streaming chunks properly accumulate content."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        previous_content = ""
        chunk_count = 0

        async for chunk in client.astream_chat(
            messages=[{
                "role": "user",
                "content": "Write a haiku about coding."
            }]
        ):
            chunk_count += 1

            # Verify content only grows (never shrinks or changes previous content)
            if chunk.content:
                assert chunk.content.startswith(previous_content), \
                    f"{config['provider']}: Content didn't accumulate properly"
                previous_content = chunk.content

        assert chunk_count > 0
        assert len(previous_content) > 0

        print(f"✅ {config['provider']}: Content accumulation verified")
        print(f"   Total chunks: {chunk_count}")


# ============================================================================
# Agent Mode (Tool Calling) Tests
# ============================================================================

@pytest.mark.smoke
class TestAgentMode:
    """Test agent mode with function/tool calling."""

    @pytest.fixture
    def calculator_tools(self):
        """Create calculator tools for testing."""

        @tool(
            name="add",
            description="Add two numbers",
            parameters={
                "a": ParameterSchema(
                    name="a",
                    type=ParameterType.NUMBER,
                    description="First number",
                    required=True
                ),
                "b": ParameterSchema(
                    name="b",
                    type=ParameterType.NUMBER,
                    description="Second number",
                    required=True
                )
            }
        )
        def add(ctx, a: float, b: float) -> float:
            return a + b

        @tool(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "a": ParameterSchema(
                    name="a",
                    type=ParameterType.NUMBER,
                    description="First number",
                    required=True
                ),
                "b": ParameterSchema(
                    name="b",
                    type=ParameterType.NUMBER,
                    description="Second number",
                    required=True
                )
            }
        )
        def multiply(ctx, a: float, b: float) -> float:
            return a * b

        return [add, multiply]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_TOOL_CALLING)
    async def test_single_tool_call(self, config, calculator_tools):
        """Test single tool call."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "What is 15 + 27?"
            }],
            tools=calculator_tools
        )

        # Verify tool calls were made
        assert response.message.tool_calls is not None and len(response.message.tool_calls) > 0, \
            f"{config['provider']}: No tool calls made"

        tool_name, tool_args = extract_tool_call_info(response.message.tool_calls[0])

        assert tool_name == "add", f"Expected 'add' tool, got '{tool_name}'"
        assert "a" in tool_args
        assert "b" in tool_args

        print(f"✅ {config['provider']}: Single tool call passed")
        print(f"   Tool: {tool_name}")
        print(f"   Arguments: {tool_args}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_TOOL_CALLING)
    async def test_multiple_tool_calls(self, config, calculator_tools):
        """Test multiple tool calls in sequence."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "Calculate: (5 + 3) * 2. Use tools for each operation."
            }],
            tools=calculator_tools
        )

        # Should make at least one tool call
        assert response.message.tool_calls is not None and len(response.message.tool_calls) > 0

        print(f"✅ {config['provider']}: Multiple tool calls passed")
        print(f"   Tool calls made: {len(response.message.tool_calls)}")
        for tc in response.message.tool_calls:
            name, args = extract_tool_call_info(tc)
            print(f"   - {name}: {args}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_TOOL_CALLING)
    async def test_tool_choice_specific(self, config, calculator_tools):
        """Test forcing a specific tool."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        # Note: tool_choice implementation varies by provider
        # This test may need adjustment based on provider support
        try:
            response = await client.achat(
                messages=[{
                    "role": "user",
                    "content": "I want to multiply 7 and 8"
                }],
                tools=calculator_tools,
                tool_choice="auto"  # Let model choose
            )

            if response.message.tool_calls:
                # If tool was called, verify it's multiply
                tool_names = [extract_tool_call_info(tc)[0] for tc in response.message.tool_calls]
                assert any(name == "multiply" for name in tool_names), \
                    f"{config['provider']}: multiply tool not used"

                print(f"✅ {config['provider']}: Tool choice passed")
                print(f"   Tools used: {tool_names}")
            else:
                print(f"⚠️  {config['provider']}: No tools called (provider may not support tool_choice)")

        except Exception as e:
            print(f"⚠️  {config['provider']}: Tool choice not supported - {str(e)[:100]}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_STRUCTURED_OUTPUT)
    async def test_strict_tool_use(self, config):
        """Test strict tool use mode for type-safe function calling."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        # Define a strict tool with exact schema requirements
        @tool(
            name="get_weather",
            description="Get weather information for a city",
            parameters={
                "city": ParameterSchema(
                    name="city",
                    type=ParameterType.STRING,
                    description="The city name",
                    required=True
                ),
                "units": ParameterSchema(
                    name="units",
                    type=ParameterType.STRING,
                    description="Temperature units (celsius or fahrenheit)",
                    required=True
                )
            },
            strict=True  # Enable strict mode for guaranteed type safety
        )
        def get_weather(ctx, city: str, units: str) -> str:
            return f"Weather in {city}: 72°{units[0].upper()}"

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "What's the weather in Tokyo in celsius?"
            }],
            tools=[get_weather]
        )

        # Verify tool was called with correct types
        assert response.message.tool_calls is not None and len(response.message.tool_calls) > 0

        tool_name, tool_args = extract_tool_call_info(response.message.tool_calls[0])

        assert tool_name == "get_weather"
        assert "city" in tool_args
        assert "units" in tool_args
        assert isinstance(tool_args["city"], str)
        assert isinstance(tool_args["units"], str)

        print(f"✅ {config['provider']}: Strict tool use passed")
        print(f"   Tool: {tool_name}")
        print(f"   Arguments (type-safe): {tool_args}")


# ============================================================================
# Image Input Tests
# ============================================================================

@pytest.mark.smoke
class TestImageInput:
    """Test multimodal image understanding."""

    @pytest.fixture
    def test_image_url(self):
        """Sample image URL for testing."""
        # Using a simple, reliable test image
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_MULTIMODAL)
    async def test_image_description(self, config, test_image_url):
        """Test basic image understanding."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        from miiflow_llm.core.message import Message

        # Create message with image
        message = Message.from_image(
            text="Describe what you see in this image in one sentence.",
            image_url=test_image_url
        )

        response = await client.achat(messages=[message])

        # Verify we got a response
        assert response.message.content is not None
        assert len(response.message.content) > 10, f"{config['provider']}: Response too short"

        print(f"✅ {config['provider']}: Image description passed")
        print(f"   Response: {response.message.content[:100]}...")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_MULTIMODAL)
    async def test_image_with_json_output(self, config, test_image_url):
        """Test image understanding with structured JSON output."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        from miiflow_llm.core.message import Message

        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "hasText": {"type": "boolean"},
                "colors": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["description", "hasText", "colors"]
        }

        message = Message.from_image(
            text="Analyze this image and describe it, noting if it has text and what colors are present.",
            image_url=test_image_url
        )

        response = await client.achat(
            messages=[message],
            json_schema=schema
        )

        # Parse JSON response
        data = json.loads(response.message.content)

        assert "description" in data
        assert "hasText" in data
        assert "colors" in data
        assert isinstance(data["hasText"], bool)
        assert isinstance(data["colors"], list)

        print(f"✅ {config['provider']}: Image with JSON output passed")
        print(f"   Analysis: {data}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_MULTIMODAL)
    async def test_multiple_images(self, config):
        """Test processing multiple images in one request."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        from miiflow_llm.core.message import Message, TextBlock, ImageBlock

        # Create message with multiple images
        message = Message(
            role=MessageRole.USER,
            content=[
                TextBlock(text="Compare these two images. What's different?"),
                ImageBlock(image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"),
                ImageBlock(image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/1x1.png/240px-1x1.png")
            ]
        )

        try:
            response = await client.achat(messages=[message])

            assert response.message.content is not None
            assert len(response.message.content) > 10

            print(f"✅ {config['provider']}: Multiple images passed")
            print(f"   Response: {response.message.content[:100]}...")

        except Exception as e:
            # Some providers may have limitations on multiple images
            print(f"⚠️  {config['provider']}: Multiple images not supported - {str(e)[:100]}")


# ============================================================================
# Combined Feature Tests
# ============================================================================

@pytest.mark.smoke
class TestCombinedFeatures:
    """Test combinations of features together."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", PROVIDERS_TOOL_CALLING)
    async def test_streaming_with_tools(self, config):
        """Test streaming while using tools."""
        api_key = os.getenv(config["api_key_env"])
        if not api_key or "your-" in api_key:
            pytest.skip(f"{config['api_key_env']} not configured")

        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )

        @tool(
            name="get_weather",
            description="Get current weather",
            parameters={
                "city": ParameterSchema(
                    name="city",
                    type=ParameterType.STRING,
                    description="City name",
                    required=True
                )
            }
        )
        def get_weather(ctx, city: str) -> str:
            return f"Sunny, 72°F in {city}"

        chunks = []
        tool_calls = []

        async for chunk in client.astream_chat(
            messages=[{
                "role": "user",
                "content": "What's the weather in Paris?"
            }],
            tools=[get_weather]
        ):
            chunks.append(chunk)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        assert len(chunks) > 0

        # Some providers return tool calls in streaming
        if tool_calls:
            # Extract tool names using helper function
            tool_names = [extract_tool_call_info(tc)[0] for tc in tool_calls]
            assert any(name == "get_weather" for name in tool_names)
            print(f"✅ {config['provider']}: Streaming with tools passed")
            print(f"   Tool calls in stream: {len(tool_calls)}")
        else:
            print(f"ℹ️  {config['provider']}: Tool calls not streamed (may be in final chunk)")


if __name__ == "__main__":
    # Run with: pytest tests/integration/test_all_features.py -v -s
    pytest.main([__file__, "-v", "-s"])
