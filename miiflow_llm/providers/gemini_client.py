"""Google Gemini client implementation."""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..core.client import ModelClient
from ..core.message import Message, MessageRole, TextBlock, ImageBlock
from ..core.metrics import TokenCount
from ..core.exceptions import ProviderError, AuthenticationError, ModelError
from .stream_normalizer import get_stream_normalizer


class GeminiClient(ModelClient):
    """Google Gemini client implementation."""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai is required for Gemini. Install with: pip install google-generativeai"
            )
        
        super().__init__(model, api_key, timeout, max_retries, **kwargs)
        
        if not api_key:
            raise AuthenticationError("Gemini API key is required")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize the model
        try:
            self.client = genai.GenerativeModel(model_name=model)
        except Exception as e:
            raise ModelError(f"Failed to initialize Gemini model {model}: {e}")
        
        self.stream_normalizer = get_stream_normalizer("gemini")
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        self.provider_name = "gemini"
    
    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Gemini format."""
        return {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"]
        }
    
    def _convert_messages_to_gemini_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format."""
        gemini_messages = []
        
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                if gemini_messages and gemini_messages[-1]["role"] == "user":
                    gemini_messages[-1]["parts"][0]["text"] = f"System: {message.content}\n\n{gemini_messages[-1]['parts'][0]['text']}"
                else:
                    gemini_messages.append({
                        "role": "user", 
                        "parts": [{"text": f"System: {message.content}"}]
                    })
            elif message.role == MessageRole.USER:
                parts = []
                
                if isinstance(message.content, str):
                    parts.append({"text": message.content})
                elif isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append({"text": block.text})
                        elif isinstance(block, ImageBlock):
                            # Gemini supports images, but format may vary
                            parts.append({"text": f"[Image: {block.image_url}]"})
                
                gemini_messages.append({"role": "user", "parts": parts})
                
            elif message.role == MessageRole.ASSISTANT:
                gemini_messages.append({
                    "role": "model", 
                    "parts": [{"text": message.content}]
                })
        
        return gemini_messages
    
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """Send chat completion request to Gemini."""
        try:
            gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            if len(gemini_messages) == 1:
                prompt = gemini_messages[0]["parts"][0]["text"]
            else:
                prompt_parts = []
                for msg in gemini_messages:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    text = msg["parts"][0]["text"]
                    prompt_parts.append(f"{role}: {text}")
                prompt = "\n\n".join(prompt_parts)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens or 8192,
            )
            
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            if response.candidates and response.candidates[0].content.parts:
                content = response.candidates[0].content.parts[0].text
            else:
                content = ""
            
            usage = TokenCount()
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = TokenCount(
                    prompt_tokens=getattr(response.usage_metadata, 'prompt_token_count', 0),
                    completion_tokens=getattr(response.usage_metadata, 'candidates_token_count', 0),
                    total_tokens=getattr(response.usage_metadata, 'total_token_count', 0)
                )
            
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content
            )
            
            from ..core.client import ChatResponse
            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None
            )
            
        except Exception as e:
            raise ProviderError(f"Gemini API error: {e}", provider="gemini")
    
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator:
        """Send streaming chat completion request to Gemini."""
        try:
            gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            if len(gemini_messages) == 1:
                prompt = gemini_messages[0]["parts"][0]["text"]
            else:
                prompt_parts = []
                for msg in gemini_messages:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    text = msg["parts"][0]["text"]
                    prompt_parts.append(f"{role}: {text}")
                prompt = "\n\n".join(prompt_parts)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens or 8192,
            )
            
            response_stream = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                stream=True
            )
            
            accumulated_content = ""
            
            for chunk in response_stream:
                normalized_chunk = self.stream_normalizer.normalize(chunk)
                
                if normalized_chunk.delta:
                    accumulated_content += normalized_chunk.delta
                
                normalized_chunk.content = accumulated_content
                
                yield normalized_chunk
            
        except Exception as e:
            raise ProviderError(f"Gemini streaming error: {e}", provider="gemini")


# Available Gemini models (updated for current API)
GEMINI_MODELS = {
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-pro-latest": "gemini-1.5-pro-latest", 
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
    # Note: gemini-pro is deprecated, use gemini-1.5-pro instead
}
