#!/usr/bin/env python3
"""
Test image vision workflow using multimedia support.
Demonstrates that the same framework works for images and PDFs.

Usage:
    python test_image_vision.py [IMAGE_PATH]

"""

import asyncio
import base64
import sys
import time
from pathlib import Path

from miiflow_llm.agents import create_agent, AgentConfig, AgentContext
from miiflow_llm.core.agent import AgentType
from miiflow_llm.core.message import Message


def get_image_path() -> str:
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if Path(image_path).exists() and Path(image_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            return image_path
        else:
            print(f"Error: File '{image_path}' not found or not a supported image format")
    
    
    return None


async def test_image_vision_workflow(image_path: str):
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine MIME type based on file extension
        ext = Path(image_path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif ext == '.png':
            mime_type = "image/png"
        elif ext == '.gif':
            mime_type = "image/gif"
        elif ext == '.webp':
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"  # default
            
        image_data_uri = f"data:{mime_type};base64,{image_b64}"
    
    
    providers_to_test = [
        {"provider": "openai", "model": "gpt-4o-mini", "description": "OpenAI GPT-4o Vision"},
        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "description": "Anthropic Claude Vision"},
    ]
    
    vision_query = "What do you see in this image? Please describe it in detail."
    
    for config in providers_to_test:
        provider = config["provider"]
        model = config["model"]
        description = config["description"]
        
        print(f"🔄 Testing {provider.upper()}: {description}")
        print("-" * 50)
        
        try:
            agent = create_agent(AgentConfig(
                provider=provider,
                model=model,
                agent_type=AgentType.SINGLE_HOP,
                tools=[],
                system_prompt="You are an expert at analyzing and describing images. Provide detailed, accurate descriptions of what you see."
            ))
            
            print(f"Vision agent initialized: {provider.upper()} ({model})")
            
            # Create message with image using our multimedia framework
            message = Message.from_image(
                text=vision_query,
                image_url=image_data_uri,
                detail="high"
            )
            
            print(f"Analyzing image with {provider.upper()}...")
            context = AgentContext()
            start_time = time.time()
            
            result = await agent.run(
                prompt=vision_query,
                context=context,
                message_history=[message]
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            response = result.get('response', str(result))
            
            print(f"SUCCESS: {processing_time:.2f}s")
            print(f"Response: {len(response)} chars")
            print(f"Vision Analysis:")
            print(f"   {response}")
            
        except Exception as e:
            print(f"FAILED: {str(e)}")
        
        

async def test_multimodal_combination(image_path: str):
    try:
        agent = create_agent(AgentConfig(
            provider="openai",  # OpenAI supports both vision and documents
            model="gpt-4o-mini",
            agent_type=AgentType.SINGLE_HOP,
            tools=[],
            system_prompt="You are a multimodal AI assistant capable of analyzing both documents and images. Provide comprehensive analysis."
        ))
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            
            # Determine MIME type based on file extension
            ext = Path(image_path).suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                mime_type = "image/jpeg"
            elif ext == '.png':
                mime_type = "image/png"
            elif ext == '.gif':
                mime_type = "image/gif"
            elif ext == '.webp':
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # default
                
            image_data_uri = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        
        message = Message.from_image(
            text="Analyze this image and tell me what you see. Be detailed and descriptive.",
            image_url=image_data_uri,
            detail="high"
        )
        
        print(" Processing image with multimodal agent...")
        
        start_time = time.time()
        result = await agent.run(
            prompt="Describe this image in detail",
            context=AgentContext(),
            message_history=[message]
        )
        end_time = time.time()
        
        response = result.get('response', str(result))
        print(f" Multimodal analysis complete: {end_time - start_time:.2f}s")
        print(f" Analysis Result:")
        print(f"   {response}")
        
    except Exception as e:
        print(f" Multimodal test failed: {e}")


async def main():
    """Run image vision workflow tests."""
    image_path = get_image_path()
    if not image_path:
        return
    
    print(f"Using image: {image_path}")
    
    # Test image vision across providers
    await test_image_vision_workflow(image_path)
    
    # Test multimodal combination with the same image
    await test_multimodal_combination(image_path)


if __name__ == "__main__":
    asyncio.run(main())
