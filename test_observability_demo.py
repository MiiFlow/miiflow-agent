#!/usr/bin/env python3
"""
Demo script to test the cleaned-up observability system with real LLM calls.
This will demonstrate Phoenix tracing in action!
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the miiflow_llm directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from miiflow_llm.core.observability import (
    enable_phoenix_tracing,
    launch_local_phoenix,
    get_logger,
    check_instrumentation_status,
    ObservabilityConfig
)
from miiflow_llm.core.client import LLMClient
from miiflow_llm.core.message import Message, MessageRole


def main():
    """Demo the observability system with real LLM calls."""
    print("🚀 Miiflow LLM Observability Demo")
    print("=" * 50)
    
    
    print("\n2.  Starting Local Phoenix Session...")
    session = launch_local_phoenix()
    
    if not session:
        print(" Failed to start Phoenix. Make sure Phoenix is installed:")
        print("   pip install arize-phoenix")
        return
    
    print(f" Phoenix started: {session.url}")
    
    # 3. Enable tracing 
    print("\n3.  Enabling Phoenix Tracing...")
    # Use the Phoenix session URL directly (it already includes the correct port)
    from urllib.parse import urlparse
    parsed_url = urlparse(session.url)
    endpoint = f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port}"
    success = enable_phoenix_tracing(endpoint=endpoint)
    
    if not success:
        print(" Failed to enable Phoenix tracing")
        return
    
    print(" Phoenix tracing enabled!")
    
    # 4. Get trace-aware logger
    logger = get_logger(__name__)
    logger.info("Observability demo started", demo_version="1.0", phoenix_url=session.url)
    
    # 5. Check if we have API keys for testing
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not openai_key and not anthropic_key:
        logger.info("Test log message with tracing", 
                   test_data={"model": "test", "tokens": 100},
                   operation="demo_test")
        
      
    else:
        # 6. Make actual LLM calls with tracing
        print(f"\n4. 🤖 Making traced LLM calls...")
        
        # Choose provider based on available keys
        if openai_key:
            provider = "openai"
            model = "gpt-3.5-turbo"
        else:
            provider = "anthropic" 
            model = "claude-3-haiku-20240307"
            
        print(f"   Using {provider} with model {model}")
        
        try:
            # Create client using the class method
            client = LLMClient.create(provider=provider, model=model)
            
            # Create test messages
            messages = [
                Message(role=MessageRole.USER, content="Hello! Please say 'Observability test successful!' and explain what tracing means in 1 sentence.")
            ]
            
            logger.info("Making LLM call", provider=provider, model=model, message_count=len(messages))
            
            # Make the call (this should be traced!)
            start_time = time.time()
            response = client.chat(messages)
            duration = (time.time() - start_time) * 1000
            
            logger.info("LLM call completed", 
                       provider=provider, 
                       model=model,
                       response_length=len(response.message.content) if hasattr(response, 'message') and hasattr(response.message, 'content') else 0,
                       duration_ms=duration)
            
            print(f"\n📝 LLM Response:")
            print(f"   {response.message.content if hasattr(response, 'message') and hasattr(response.message, 'content') else str(response)}")
            
            
        except Exception as e:
            logger.error("LLM call failed", error=str(e), provider=provider, model=model)
            
    try:
        input("\nPress Enter to stop Phoenix and exit...")
    except KeyboardInterrupt:
        print("\n👋 Stopping demo...")


if __name__ == "__main__":
    main()
