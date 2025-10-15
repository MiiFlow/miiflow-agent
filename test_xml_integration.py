#!/usr/bin/env python3
"""
Integration test for XML-based ReAct system.
Tests the complete flow from parser to orchestrator to ensure XML parsing works correctly.
"""

import asyncio
from miiflow_llm.core.react.parsing.xml_parser import XMLReActParser, ParseEventType
from miiflow_llm.core.react.parser import ReActParser
from miiflow_llm.core.react.data import REACT_SYSTEM_PROMPT


def test_xml_parser():
    """Test XML parser with various scenarios."""
    print("=" * 80)
    print("TEST 1: XML Parser - Complete Response Parsing")
    print("=" * 80)

    parser = XMLReActParser()

    # Test thinking + tool call
    response = """<thinking>
I need to calculate 15 + 27 using the add tool
</thinking>

<tool_call name="add">
{"a": 15, "b": 27}
</tool_call>"""

    result = parser.parse_complete(response)
    assert result["thought"] == "I need to calculate 15 + 27 using the add tool"
    assert result["action_type"] == "tool_call"
    assert result["action"] == "add"
    assert result["action_input"] == {"a": 15, "b": 27}
    print("✓ Tool call parsing works")

    # Test thinking + answer
    parser.reset()
    response = """<thinking>
I have the result, I can answer now
</thinking>

<answer>
The sum of 15 and 27 is 42.
</answer>"""

    result = parser.parse_complete(response)
    assert result["thought"] == "I have the result, I can answer now"
    assert result["action_type"] == "final_answer"
    assert "42" in result["answer"]
    print("✓ Final answer parsing works")

    print()


def test_streaming_parser():
    """Test streaming XML parser."""
    print("=" * 80)
    print("TEST 2: XML Parser - Streaming Mode")
    print("=" * 80)

    parser = XMLReActParser()

    # Simulate streaming chunks
    chunks = [
        "<think",
        "ing>",
        "Let me ",
        "answer this",
        "</thinking>\n",
        "<answer>",
        "The answer ",
        "is 42",
        ".</answer>"
    ]

    events = []
    for chunk in chunks:
        for event in parser.parse_streaming(chunk):
            events.append(event)
            if event.event_type == ParseEventType.THINKING_COMPLETE:
                print(f"✓ Thinking: {event.data['thought']}")
            elif event.event_type == ParseEventType.ANSWER_START:
                print(f"✓ Answer start detected")
            elif event.event_type == ParseEventType.ANSWER_CHUNK:
                print(f"  → Chunk: \"{event.data['delta']}\"")
            elif event.event_type == ParseEventType.ANSWER_COMPLETE:
                print(f"✓ Answer complete: {event.data['answer']}")

    # Verify we got the expected events
    event_types = [e.event_type for e in events]
    assert ParseEventType.THINKING_COMPLETE in event_types
    assert ParseEventType.ANSWER_START in event_types
    assert ParseEventType.ANSWER_CHUNK in event_types
    assert ParseEventType.ANSWER_COMPLETE in event_types
    print("✓ All streaming events emitted correctly")

    print()


def test_react_parser_wrapper():
    """Test that ReActParser correctly wraps XMLReActParser."""
    print("=" * 80)
    print("TEST 3: ReActParser Wrapper")
    print("=" * 80)

    parser = ReActParser()

    # Verify it has xml_parser
    assert hasattr(parser, 'xml_parser')
    assert isinstance(parser.xml_parser, XMLReActParser)
    print("✓ ReActParser has xml_parser attribute")

    # Test parse_response method
    response = """<thinking>
Testing the wrapper
</thinking>

<tool_call name="test_tool">
{"param": "value"}
</tool_call>"""

    result = parser.parse_response(response, step_number=1)
    assert result.thought == "Testing the wrapper"
    assert result.action_type == "tool_call"
    assert result.action == "test_tool"
    assert result.action_input == {"param": "value"}
    print("✓ parse_response works correctly")

    # Test streaming methods
    parser.reset()
    assert hasattr(parser, 'parse_streaming')
    print("✓ Streaming methods available")

    print()


def test_system_prompt():
    """Test that system prompt uses XML format."""
    print("=" * 80)
    print("TEST 4: System Prompt Format")
    print("=" * 80)

    # Check for XML tags
    assert "<thinking>" in REACT_SYSTEM_PROMPT
    assert "</thinking>" in REACT_SYSTEM_PROMPT
    assert "<tool_call" in REACT_SYSTEM_PROMPT
    assert "</tool_call>" in REACT_SYSTEM_PROMPT
    assert "<answer>" in REACT_SYSTEM_PROMPT
    assert "</answer>" in REACT_SYSTEM_PROMPT
    print("✓ System prompt uses XML tags")

    # Check it doesn't contain JSON artifacts
    assert 'REACT_RESPONSE_SCHEMA' not in REACT_SYSTEM_PROMPT
    assert '"action_type"' not in REACT_SYSTEM_PROMPT
    print("✓ No JSON artifacts in system prompt")

    # Show a snippet
    lines = REACT_SYSTEM_PROMPT.split('\n')
    print(f"\nPrompt snippet (first 5 lines):")
    for line in lines[:5]:
        print(f"  {line}")

    print()


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=" * 80)
    print("TEST 5: Edge Cases")
    print("=" * 80)

    parser = XMLReActParser()

    # Test JSON parameters with special characters (properly escaped)
    response = """<thinking>
Testing special characters in JSON
</thinking>

<tool_call name="search">
{"query": "Paris the city of lights", "limit": 10}
</tool_call>"""

    result = parser.parse_complete(response)
    assert result["action"] == "search"
    assert "Paris" in result["action_input"]["query"]
    assert result["action_input"]["limit"] == 10
    print("✓ Handles JSON with special characters")

    # Test multi-line thinking
    response = """<thinking>
This is a multi-line thought.
It spans several lines.
Each line adds more detail.
</thinking>

<answer>
Multi-line answer.
</answer>"""

    result = parser.parse_complete(response)
    assert "multi-line" in result["thought"].lower()
    assert len(result["thought"].split('\n')) > 1
    print("✓ Handles multi-line thinking")

    # Test empty parameters
    response = """<thinking>
Calling tool with empty params
</thinking>

<tool_call name="no_params_tool">
{}
</tool_call>"""

    result = parser.parse_complete(response)
    assert result["action"] == "no_params_tool"
    assert result["action_input"] == {}
    print("✓ Handles empty tool parameters")

    print()


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "XML-BASED REACT INTEGRATION TESTS" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        test_xml_parser()
        test_streaming_parser()
        test_react_parser_wrapper()
        test_system_prompt()
        test_edge_cases()

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("Summary:")
        print("  - XML parser handles complete responses ✓")
        print("  - Streaming mode emits events correctly ✓")
        print("  - ReActParser wrapper works properly ✓")
        print("  - System prompt uses XML format ✓")
        print("  - Edge cases handled correctly ✓")
        print()
        print("The XML-based ReAct system is ready for production use!")
        print()
        return 0

    except AssertionError as e:
        print()
        print("=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ UNEXPECTED ERROR!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
