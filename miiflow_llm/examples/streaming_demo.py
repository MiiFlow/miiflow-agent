"""Demo showing unified streaming with structured output."""

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from miiflow_llm.core import LLMClient, Message


@dataclass 
class PersonInfo:
    """Example structured output schema."""
    name: str
    age: int
    occupation: str
    hobbies: List[str]
    bio: Optional[str] = None


async def test_streaming_normalization():
    """Test streaming normalization across different providers."""
    
    providers_to_test = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-haiku-20240307"), 
        ("groq", "llama-3.1-8b-instant"),
        ("xai", "grok-beta"),
    ]
    
    messages = [
        Message.system("You are a helpful assistant. Respond with structured data about a fictional person."),
        Message.user("Tell me about a fictional software engineer named Alice. Format as JSON with keys: name, age, occupation, hobbies (array), bio.")
    ]
    
    for provider, model in providers_to_test:
        print(f"\n=== Testing {provider} ({model}) ===")
        
        try:
            client = LLMClient.create(provider=provider, model=model)
            
            # Test 1: Regular streaming
            print("\n1. Regular Streaming:")
            buffer = ""
            async for chunk in client.stream_chat(messages):
                buffer += chunk.delta
                print(f"Delta: '{chunk.delta}'")
                if chunk.finish_reason:
                    print(f"Finish reason: {chunk.finish_reason}")
                    break
            print(f"Complete response: {buffer}")
            
            # Test 2: Streaming with schema
            print("\n2. Streaming with Schema:")
            async for chunk in client.stream_with_schema(messages, schema=PersonInfo):
                print(f"Content: '{chunk.delta}'")
                if chunk.partial_parse:
                    print(f"Partial parse: {chunk.partial_parse}")
                if chunk.structured_output:
                    print(f"Final structured output: {chunk.structured_output}")
                if chunk.is_complete:
                    break
                    
        except Exception as e:
            print(f"Error with {provider}: {e}")
        
        print("-" * 50)


async def test_incremental_parsing():
    """Test incremental JSON parsing during streaming."""
    
    # Simulate streaming chunks that build up a JSON object
    chunks = [
        '{"name": "Alice',
        ' Johnson", "age":',
        ' 28, "occupation":',
        ' "Software Engineer",',
        ' "hobbies": ["coding",',
        ' "hiking", "reading"],',
        ' "bio": "Passionate about',
        ' AI and machine learning."}'
    ]
    
    from miiflow_llm.core.streaming import IncrementalParser
    
    parser = IncrementalParser(schema=PersonInfo)
    
    print("=== Testing Incremental Parsing ===")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}: '{chunk}'")
        
        partial_result = parser.try_parse_partial(chunk)
        if partial_result:
            print(f"Partial parse successful: {partial_result}")
        else:
            print("No partial parse yet")
    
    # Final parse
    final_result = parser.finalize_parse("")
    print(f"\nFinal result: {final_result}")


async def test_error_handling():
    """Test error handling in streaming normalization."""
    
    from miiflow_llm.core.streaming import ProviderStreamNormalizer
    
    normalizer = ProviderStreamNormalizer()
    
    print("=== Testing Error Handling ===")
    
    # Test with various malformed chunks
    test_cases = [
        (None, "openai"),
        ("plain string", "anthropic"),
        ({"unexpected": "format"}, "groq"),
        (123, "generic")
    ]
    
    for chunk, provider in test_cases:
        try:
            result = normalizer.normalize_chunk(chunk, provider)
            print(f"Provider {provider}, Chunk {chunk} -> {result.content[:50]}...")
        except Exception as e:
            print(f"Provider {provider}, Chunk {chunk} -> Error: {e}")


if __name__ == "__main__":
    print("Miiflow LLM Streaming Demo")
    print("=" * 40)
    
    # Run tests
    asyncio.run(test_incremental_parsing())
    asyncio.run(test_error_handling())
    
    # Only run provider tests if API keys are available
    try:
        asyncio.run(test_streaming_normalization())
    except Exception as e:
        print(f"Provider tests skipped: {e}")
        print("Set API keys in .env file to test streaming normalization")
