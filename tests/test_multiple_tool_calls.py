#!/usr/bin/env python3
"""Test multiple tool calls in a single response."""

import asyncio
from miiflow_agent.core.react.parsing.xml_parser import XMLReActParser, ParseEventType

async def test_multiple_tool_calls():
    """Test parsing a response with multiple tool calls."""

    # Simulate LLM response with thinking + 2 tool calls
    response = """<thinking>
Let me start with the multiplication step.
</thinking>

<tool_call name="Multiply">
{"a": 2, "b": 2}
</tool_call>

<tool_call name="Add">
{"a": 9, "b": 6}
</tool_call>"""

    parser = XMLReActParser()

    # Simulate streaming by feeding response in chunks
    chunks = [response[i:i+10] for i in range(0, len(response), 10)]

    print("Simulating streaming parse...")
    print(f"Total chunks: {len(chunks)}\n")

    tool_calls_found = []

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {repr(chunk)}")
        for event in parser.parse_streaming(chunk):
            if event.event_type == ParseEventType.THINKING:
                print(f"  → THINKING chunk: {repr(event.data['delta'][:30])}")
            elif event.event_type == ParseEventType.THINKING_COMPLETE:
                print(f"  → THINKING COMPLETE")
            elif event.event_type == ParseEventType.TOOL_CALL:
                tool_name = event.data['tool_name']
                params = event.data['parameters']
                tool_calls_found.append(tool_name)
                print(f"  → TOOL_CALL: {tool_name} with {params}")

    print(f"\n{'='*60}")
    print(f"After streaming:")
    print(f"  has_parsed_content: {parser.has_parsed_content}")
    print(f"  buffer: {repr(parser.buffer)}")
    print(f"  tool_calls_found: {tool_calls_found}")

    # Try to flush remaining buffer
    print(f"\nFlushing remaining buffer with empty chunk...")
    for event in parser.parse_streaming(""):
        if event.event_type == ParseEventType.TOOL_CALL:
            tool_name = event.data['tool_name']
            tool_calls_found.append(tool_name)
            print(f"  → TOOL_CALL from flush: {tool_name}")

    print(f"\n{'='*60}")
    print(f"Final state:")
    print(f"  has_parsed_content: {parser.has_parsed_content}")
    print(f"  buffer: {repr(parser.buffer)}")
    print(f"  tool_calls_found: {tool_calls_found}")

    if len(tool_calls_found) == 2:
        print(f"\n✅ SUCCESS: Found both tool calls")
    else:
        print(f"\n❌ FAILURE: Expected 2 tool calls, found {len(tool_calls_found)}")

if __name__ == "__main__":
    asyncio.run(test_multiple_tool_calls())
