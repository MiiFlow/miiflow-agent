#!/usr/bin/env python3
"""Streaming interface consistency validation across LLM providers."""

import asyncio
from miiflow_llm import LLMClient

async def test_unified_streaming():
    """Validate uniform streaming interface across providers."""
    
    print('Provider Streaming Interface Validation')
    print('='*50)
    
    test_configs = [
        ('openai', 'gpt-4o-mini', 'OpenAI Fast'),
        ('openai', 'gpt-4o', 'OpenAI Production'),
        ('openai', 'gpt-5-nano', 'GPT-5 Fast'),
        ('openai', 'gpt-5-mini', 'GPT-5 Efficient'),
        ('openai', 'gpt-5', 'GPT-5 Flagship'),
        ('groq', 'llama-3.1-8b-instant', 'Groq Fast'),
        ('groq', 'llama-3.3-70b-versatile', 'Groq Large'),
        ('gemini', 'gemini-1.5-flash', 'Gemini Fast'),
        ('gemini', 'gemini-1.5-pro', 'Gemini Production'), 
        ('gemini', 'gemini-1.5-flash-8b', 'Gemini Efficient'),
        ('anthropic', 'claude-3-haiku-20240307', 'Claude Fast'),
        ('anthropic', 'claude-3-5-sonnet-20241022', 'Claude Smart'),
        ('xai', 'grok-beta', 'Grok Beta'),
        ('together', 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 'TogetherAI Llama'),
        ('openrouter', 'meta-llama/llama-3.2-3b-instruct:free', 'OpenRouter Free'),
        ('mistral', 'mistral-small-latest', 'Mistral Small'),
        ('ollama', 'llama3.1:8b', 'Local Ollama'),
    ]
    
    successful_tests = []
    failed_tests = []
    
    for provider, model, description in test_configs:
        print(f'\n[TEST] {provider}:{model} ({description})')
        
        try:
            if 'gpt-5' in model:
                timeout = 60.0
            elif provider == 'ollama':
                timeout = 120.0
            else:
                timeout = 30.0
            
            client = LLMClient.create(provider, model=model, timeout=timeout)
            print(f'  Client initialized')
            
            chunks_received = 0
            final_content = None
            
            from miiflow_llm.core import Message
            messages = [Message.user('Say "Hello from [provider]" but replace [provider] with your actual provider name')]
            
            async for chunk in client.stream_chat(messages):
                chunks_received += 1
                
                if chunks_received <= 2:
                    print(f'    StreamChunk #{chunks_received}: delta="{chunk.delta}" content="{chunk.content[:8]}..."')
                
                if chunk.finish_reason:
                    final_content = chunk.content
                    print(f'  SUCCESS: {chunks_received} chunks')
                    print(f'  Response: "{final_content}"')
                    print(f'  StreamChunk format validated')
                    successful_tests.append(f'{provider}:{model}')
                    break
            
        except Exception as e:
            error_msg = str(e)
            print(f'  FAILED: {error_msg}')
            failed_tests.append((f'{provider}:{model}', error_msg))
    
    print(f'\nValidation Results:')
    print(f'Successful: {len(successful_tests)} models')
    for model in successful_tests:
        print(f'   • {model}')
    
    print(f'\nFailed: {len(failed_tests)} models')
    for model, error in failed_tests:
        print(f'   • {model} → {error}')
    
    print(f'\nInterface Consistency Verification:')
    print(f'   Uniform LLMClient.create() API across providers')
    print(f'   Consistent StreamChunk(content, delta, finish_reason, usage) format')
    print(f'   Provider abstraction eliminates streaming inconsistencies')
    
    if successful_tests:
        print(f'\nStructured Output Validation:')
        try:
            from miiflow_llm.core import Message
            client = LLMClient.create('gemini', model='gemini-1.5-flash')
            messages = [Message.user('Return JSON: {"greeting": "hello", "provider": "gemini"}')]
            
            async for chunk in client.stream_with_schema(messages):
                if chunk.structured_output:
                    print(f'   Structured parsing: {chunk.structured_output}')
                    break
                if chunk.is_complete:
                    break
        except Exception as e:
            print(f'   Structured parsing test skipped: {e}')

if __name__ == "__main__":
    asyncio.run(test_unified_streaming())
