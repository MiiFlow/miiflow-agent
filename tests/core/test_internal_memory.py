"""Test internal episodic memory functionality."""

import pytest
from dotenv import load_dotenv
from miiflow_llm.core import LLMClient, Agent


@pytest.fixture
def llm_client():
    load_dotenv()
    return LLMClient.create(provider="openai", model="gpt-4o-mini")


@pytest.mark.asyncio
async def test_conversation_memory(llm_client):
    agent = Agent(llm_client, max_stored_threads=5, max_messages_per_thread=50)
    
    result1 = await agent.run("Hi, my name is Alice.", thread_id="user_alice")
    result2 = await agent.run("What's my name?", thread_id="user_alice")
    result3 = await agent.run("Do you know my name?", thread_id="user_bob")
    
    alice_remembered = "alice" in result2.data.lower()
    bob_unknown = "don't know" in result3.data.lower() or "no" in result3.data.lower()
    
    assert alice_remembered
    assert bob_unknown
    assert len(agent._thread_conversations) == 2


@pytest.mark.asyncio 
async def test_memory_cleanup(llm_client):
    agent = Agent(llm_client, max_stored_threads=2, max_messages_per_thread=4)
    
    for i in range(4):
        await agent.run(f"Hello from thread {i}", thread_id=f"test_thread_{i}")
    
    assert len(agent._thread_conversations) <= 2
    
    thread_ids = list(agent._thread_conversations.keys())
    assert "test_thread_2" in thread_ids or "test_thread_3" in thread_ids


@pytest.mark.asyncio
async def test_message_limit_per_thread(llm_client):
    agent = Agent(llm_client, max_stored_threads=5, max_messages_per_thread=3)
    
    for i in range(5):
        await agent.run(f"Message {i}", thread_id="test_thread")
    
    stored_messages = agent._thread_conversations["test_thread"]
    assert len(stored_messages) <= 3


@pytest.mark.asyncio
async def test_internal_memory_priority(llm_client):
    agent = Agent(llm_client, max_stored_threads=5, max_messages_per_thread=50)
    
    result1 = await agent.run("My favorite color is blue", thread_id="priority_test")
    assert "priority_test" in agent._thread_conversations
    
    result2 = await agent.run("What's my favorite color?", thread_id="priority_test")
    assert "blue" in result2.data.lower()
