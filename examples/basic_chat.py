"""Basic chat completion example.

This example demonstrates the simplest usage of miiflow-agent:
- Creating a client for any provider
- Sending messages and getting responses
- Handling both sync and async patterns
"""

import asyncio
from miiflow_agent import LLMClient, Message


def sync_example():
    """Synchronous chat example."""
    # Create client (uses OPENAI_API_KEY env var)
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Simple chat
    response = client.chat([Message.user("What is Python in one sentence?")])
    print(f"Response: {response.message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")


async def async_example():
    """Asynchronous chat example."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    # Async chat
    response = await client.achat([Message.user("What is Rust in one sentence?")])
    print(f"Response: {response.message.content}")


async def multi_turn_conversation():
    """Multi-turn conversation example."""
    client = LLMClient.create("openai", model="gpt-4o-mini")

    messages = [
        Message.system("You are a helpful coding assistant."),
        Message.user("What's the best way to handle errors in Python?"),
    ]

    # First turn
    response = await client.achat(messages)
    print(f"Assistant: {response.message.content}\n")

    # Continue conversation
    messages.append(Message.assistant(response.message.content))
    messages.append(Message.user("Can you show me a simple example?"))

    response = await client.achat(messages)
    print(f"Assistant: {response.message.content}")


def switch_providers():
    """Example showing how to switch between providers."""
    # Same interface, different providers
    providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        # ("groq", "llama-3.1-8b-instant"),  # Uncomment if you have GROQ_API_KEY
    ]

    for provider, model in providers:
        try:
            client = LLMClient.create(provider, model=model)
            response = client.chat([Message.user("Say hello in 5 words or less.")])
            print(f"{provider}: {response.message.content}")
        except Exception as e:
            print(f"{provider}: Skipped ({e})")


if __name__ == "__main__":
    print("=== Sync Example ===")
    sync_example()

    print("\n=== Async Example ===")
    asyncio.run(async_example())

    print("\n=== Multi-turn Conversation ===")
    asyncio.run(multi_turn_conversation())

    print("\n=== Provider Switching ===")
    switch_providers()
