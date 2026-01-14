"""Basic usage: chat, streaming, async."""

import asyncio
from miiflow_agent import LLMClient
from miiflow_agent.core import Message


def sync_chat():
    client = LLMClient.create("openai", model="gpt-4o-mini")
    response = client.chat([Message.user("What is Python?")])
    print(f"Response: {response.message.content[:80]}...")
    print(f"Tokens: {response.usage.total_tokens}\n")


def sync_streaming():
    client = LLMClient.create("openai", model="gpt-4o-mini")
    print("Streaming: ", end="")
    for chunk in client.stream_chat([Message.user("Count 1 to 5")]):
        print(chunk.delta, end="", flush=True)
    print("\n")


async def async_chat():
    client = LLMClient.create("openai", model="gpt-4o-mini")
    response = await client.achat([Message.user("What is async/await?")])
    print(f"Async response: {response.message.content[:80]}...\n")


async def async_streaming():
    client = LLMClient.create("openai", model="gpt-4o-mini")
    print("Async streaming: ", end="")
    async for chunk in client.astream_chat([Message.user("Say hello")]):
        print(chunk.delta, end="", flush=True)
    print("\n")


def conversation_history():
    client = LLMClient.create("openai", model="gpt-4o-mini")
    messages = [
        Message.user("My name is Alice"),
        Message.assistant("Hello Alice!"),
        Message.user("What's my name?")
    ]
    response = client.chat(messages)
    print(f"Conversation: {response.message.content}")


if __name__ == "__main__":
    sync_chat()
    sync_streaming()
    conversation_history()
    asyncio.run(async_chat())
    asyncio.run(async_streaming())
