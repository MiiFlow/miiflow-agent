#!/usr/bin/env env python
"""
Demo script to test Anthropic's new structured output API support.

This script tests:
1. Native structured output with Claude Sonnet 4.5 (new API)
2. Fallback tool-based approach with older models
3. Streaming with structured outputs
4. Strict tool use mode

Usage:
    export CLAUDE_API_KEY=your_api_key
    poetry run python test_structured_output_demo.py
"""

import asyncio
import json
import os
from typing import Dict, Any

from miiflow_agent import LLMClient
from miiflow_agent.core.tools import tool, ParameterSchema, ParameterType


# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_json(data: Dict[str, Any], label: str = "Response"):
    """Print formatted JSON."""
    print(f"{Colors.OKBLUE}{label}:{Colors.ENDC}")
    print(json.dumps(data, indent=2))


async def test_native_structured_output():
    """Test 1: Native structured output with Claude Sonnet 4.5."""
    print_section("Test 1: Native Structured Output (Claude Sonnet 4.5)")

    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable not set")
        return False

    try:
        # Create client with new model
        client = LLMClient.create(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key=api_key
        )

        print_info("Using model: claude-sonnet-4-5-20250929 (supports native structured outputs)")

        # Define a strict schema
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "summary": {
                    "type": "string"
                }
            },
            "required": ["sentiment", "confidence", "keywords", "summary"],
            "additionalProperties": False
        }

        print_info("Testing sentiment analysis with strict schema...")

        # Make request
        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "Analyze the sentiment of this review: 'This product is amazing! It exceeded all my expectations and I highly recommend it to everyone.'"
            }],
            json_schema=schema
        )

        # Parse response
        data = json.loads(response.message.content)

        # Verify schema compliance
        assert "sentiment" in data, "Missing 'sentiment' field"
        assert data["sentiment"] in ["positive", "negative", "neutral"], "Invalid sentiment value"
        assert "confidence" in data, "Missing 'confidence' field"
        assert isinstance(data["confidence"], (int, float)), "confidence must be a number"
        assert "keywords" in data, "Missing 'keywords' field"
        assert isinstance(data["keywords"], list), "keywords must be an array"
        assert "summary" in data, "Missing 'summary' field"
        assert set(data.keys()) == {"sentiment", "confidence", "keywords", "summary"}, "Extra fields present"

        print_success("Schema compliance verified!")
        print_json(data, "Structured Output")

        # Check metadata
        print(f"\n{Colors.OKCYAN}Token Usage:{Colors.ENDC}")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

        return True

    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_mode():
    """Test 2: Fallback tool-based approach with older model."""
    print_section("Test 2: Fallback Mode (Claude 3.5 Haiku)")

    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable not set")
        return False

    try:
        # Create client with older model
        client = LLMClient.create(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            api_key=api_key
        )

        print_info("Using model: claude-3-5-haiku-20241022 (uses tool-based fallback)")

        # Same schema as before
        schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "total_count": {"type": "integer"}
            },
            "required": ["category", "items", "total_count"]
        }

        print_info("Testing data extraction with fallback mode...")

        # Make request
        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "List 5 programming languages in the 'web development' category."
            }],
            json_schema=schema
        )

        # Parse response
        data = json.loads(response.message.content)

        # Verify schema compliance
        assert "category" in data, "Missing 'category' field"
        assert "items" in data, "Missing 'items' field"
        assert isinstance(data["items"], list), "items must be an array"
        assert "total_count" in data, "Missing 'total_count' field"
        assert isinstance(data["total_count"], int), "total_count must be an integer"

        print_success("Fallback mode works correctly!")
        print_json(data, "Structured Output (via tool-based fallback)")

        return True

    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_structured_output():
    """Test 3: Streaming with structured outputs."""
    print_section("Test 3: Streaming with Structured Outputs")

    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable not set")
        return False

    try:
        client = LLMClient.create(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key=api_key
        )

        print_info("Using model: claude-sonnet-4-5-20250929 with streaming")

        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["title", "description", "tags"]
        }

        print_info("Streaming structured response...")
        print(f"{Colors.OKCYAN}Stream output:{Colors.ENDC} ", end="", flush=True)

        accumulated_content = ""
        chunk_count = 0

        async for chunk in client.astream_chat(
            messages=[{
                "role": "user",
                "content": "Generate metadata for an article about 'AI in Healthcare'"
            }],
            json_schema=schema
        ):
            chunk_count += 1
            if chunk.delta:
                print(".", end="", flush=True)
                accumulated_content += chunk.delta

        print()  # New line after dots

        # Verify streaming worked
        assert chunk_count > 0, "No chunks received"
        assert accumulated_content, "No content accumulated"

        # Parse final content
        data = json.loads(accumulated_content)

        # Verify schema
        assert "title" in data
        assert "description" in data
        assert "tags" in data
        assert isinstance(data["tags"], list)

        print_success(f"Streaming completed! Received {chunk_count} chunks")
        print_json(data, "Final Structured Output")

        return True

    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_strict_tool_use():
    """Test 4: Strict tool use mode."""
    print_section("Test 4: Strict Tool Use (Type-Safe Function Calling)")

    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable not set")
        return False

    try:
        # Define a strict tool
        @tool(
            name="calculate_shipping",
            description="Calculate shipping cost based on weight and destination",
            parameters={
                "weight_kg": ParameterSchema(
                    name="weight_kg",
                    type=ParameterType.NUMBER,
                    description="Package weight in kilograms",
                    required=True
                ),
                "destination": ParameterSchema(
                    name="destination",
                    type=ParameterType.STRING,
                    description="Destination country code (e.g., US, UK, JP)",
                    required=True
                ),
                "express": ParameterSchema(
                    name="express",
                    type=ParameterType.BOOLEAN,
                    description="Whether to use express shipping",
                    required=True
                )
            },
            strict=True  # Enable strict mode
        )
        def calculate_shipping(ctx, weight_kg: float, destination: str, express: bool) -> str:
            base_cost = weight_kg * 5
            if express:
                base_cost *= 2
            return f"Shipping to {destination}: ${base_cost:.2f}"

        client = LLMClient.create(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key=api_key
        )

        print_info("Using model: claude-sonnet-4-5-20250929 with strict tool")
        print_info("Testing type-safe function calling...")

        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "Calculate shipping cost for a 2.5kg package to Japan with express delivery"
            }],
            tools=[calculate_shipping]
        )

        # Verify tool was called
        assert response.message.tool_calls is not None, "No tool calls made"
        assert len(response.message.tool_calls) > 0, "No tool calls made"

        tool_call = response.message.tool_calls[0]

        # Extract tool info
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments")
        else:
            tool_name = getattr(tool_call, "name", None)
            tool_args = getattr(tool_call, "arguments", None)

        assert tool_name == "calculate_shipping", f"Wrong tool called: {tool_name}"

        # Verify type safety
        assert "weight_kg" in tool_args, "Missing weight_kg parameter"
        assert "destination" in tool_args, "Missing destination parameter"
        assert "express" in tool_args, "Missing express parameter"

        assert isinstance(tool_args["weight_kg"], (int, float)), "weight_kg must be a number"
        assert isinstance(tool_args["destination"], str), "destination must be a string"
        assert isinstance(tool_args["express"], bool), "express must be a boolean"

        print_success("Strict tool use verified - all parameters are type-safe!")
        print(f"\n{Colors.OKBLUE}Tool Call Details:{Colors.ENDC}")
        print(f"  Function: {tool_name}")
        print(f"  Arguments:")
        print(f"    weight_kg: {tool_args['weight_kg']} (type: {type(tool_args['weight_kg']).__name__})")
        print(f"    destination: {tool_args['destination']} (type: {type(tool_args['destination']).__name__})")
        print(f"    express: {tool_args['express']} (type: {type(tool_args['express']).__name__})")

        return True

    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_complex_nested_schema():
    """Test 5: Complex nested schema with arrays and objects."""
    print_section("Test 5: Complex Nested Schema")

    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable not set")
        return False

    try:
        client = LLMClient.create(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key=api_key
        )

        print_info("Using model: claude-sonnet-4-5-20250929")

        # Complex nested schema
        schema = {
            "type": "object",
            "properties": {
                "company": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "founded": {"type": "integer"}
                    },
                    "required": ["name", "founded"]
                },
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                            "in_stock": {"type": "boolean"}
                        },
                        "required": ["name", "price", "in_stock"]
                    }
                },
                "employee_count": {"type": "integer"}
            },
            "required": ["company", "products", "employee_count"],
            "additionalProperties": False
        }

        print_info("Testing complex nested schema...")

        response = await client.achat(
            messages=[{
                "role": "user",
                "content": "Generate data for a tech company named 'TechCorp' founded in 2010 with 50 employees. Include 3 products: 'Widget Pro' ($99.99, in stock), 'Gadget Plus' ($149.99, in stock), 'Tool Master' ($199.99, out of stock)."
            }],
            json_schema=schema
        )

        data = json.loads(response.message.content)

        # Verify nested structure
        assert "company" in data
        assert "name" in data["company"]
        assert "founded" in data["company"]
        assert isinstance(data["company"]["founded"], int)

        assert "products" in data
        assert isinstance(data["products"], list)
        assert len(data["products"]) == 3

        for product in data["products"]:
            assert "name" in product
            assert "price" in product
            assert "in_stock" in product
            assert isinstance(product["price"], (int, float))
            assert isinstance(product["in_stock"], bool)

        assert "employee_count" in data
        assert isinstance(data["employee_count"], int)

        print_success("Complex nested schema validated!")
        print_json(data, "Nested Structured Output")

        return True

    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                  Anthropic Structured Output API Test Suite                  ║")
    print("║                                                                               ║")
    print("║  Testing native structured outputs, fallback mode, streaming, and strict     ║")
    print("║  tool use with Anthropic's new API (beta: structured-outputs-2025-11-13)     ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")

    # Check API key
    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("Error: CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable not set")
        print_info("Please set it with: export CLAUDE_API_KEY=your_api_key")
        return

    print_info(f"API Key found: {api_key[:8]}...{api_key[-4:]}")

    # Run tests
    results = []

    tests = [
        ("Native Structured Output", test_native_structured_output),
        ("Fallback Mode", test_fallback_mode),
        ("Streaming", test_streaming_structured_output),
        ("Strict Tool Use", test_strict_tool_use),
        ("Complex Nested Schema", test_complex_nested_schema),
    ]

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print_error("\nTest interrupted by user")
            break
        except Exception as e:
            print_error(f"Unexpected error in {test_name}: {str(e)}")
            results.append((test_name, False))

    # Print summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{Colors.BOLD}")
    if passed == total:
        print(f"{Colors.OKGREEN}All tests passed! ({passed}/{total}){Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}Some tests failed: {passed}/{total} passed{Colors.ENDC}")

    print(f"\n{Colors.OKCYAN}Implementation details:{Colors.ENDC}")
    print(f"  • Native API: Used for Claude Sonnet 4.5 and Opus 4.1")
    print(f"  • Fallback: Tool-based approach for older models")
    print(f"  • Streaming: Now supported with structured outputs!")
    print(f"  • Strict mode: Type-safe function calling with guaranteed compliance")
    print()


if __name__ == "__main__":
    asyncio.run(main())
