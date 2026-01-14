"""Streaming example.

This example demonstrates real-time token streaming:
- Async streaming with chunk-by-chunk output
- Handling streaming events
- Building responsive UIs with streaming
"""

import asyncio
from miiflow_agent import LLMClient, Message


async def basic_streaming():
    """Basic streaming - print tokens as they arrive."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    print("Streaming response: ", end="", flush=True)

    async for chunk in client.astream_chat([
        Message.user("Write a haiku about programming.")
    ]):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)

    print("\n")


async def streaming_with_metadata():
    """Streaming with usage tracking."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    full_content = ""
    final_usage = None

    async for chunk in client.astream_chat([
        Message.user("Explain recursion in 3 sentences.")
    ]):
        if chunk.delta:
            full_content += chunk.delta
            print(chunk.delta, end="", flush=True)

        # Last chunk contains usage info
        if chunk.finish_reason:
            final_usage = chunk.usage

    print(f"\n\nFull response length: {len(full_content)} chars")
    if final_usage:
        print(f"Tokens used: {final_usage.total_tokens}")


async def streaming_multiple_messages():
    """Stream responses for multiple messages."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    questions = [
        "What is 2+2?",
        "What is the capital of France?",
        "Name a primary color.",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        print("A: ", end="", flush=True)

        async for chunk in client.astream_chat([Message.user(question)]):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)

        print()


async def streaming_with_system_prompt():
    """Streaming with custom system prompt."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    messages = [
        Message.system("You are a pirate. Respond in pirate speak."),
        Message.user("Tell me about the weather today."),
    ]

    print("Pirate response: ", end="", flush=True)

    async for chunk in client.astream_chat(messages):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    print("=== Basic Streaming ===")
    asyncio.run(basic_streaming())

    print("=== Streaming with Metadata ===")
    asyncio.run(streaming_with_metadata())

    print("=== Multiple Streaming Messages ===")
    asyncio.run(streaming_multiple_messages())

    print("=== Streaming with System Prompt ===")
    asyncio.run(streaming_with_system_prompt())
