"""Google Gemini client implementation."""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
    from google.generativeai.types import FunctionDeclaration, Tool

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..core.client import ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError
from ..core.message import ImageBlock, Message, MessageRole, TextBlock
from ..core.metrics import TokenCount
from ..core.streaming import StreamChunk
from ..utils.image import image_url_to_bytes


class GeminiClient(ModelClient):
    """Google Gemini client implementation."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs,
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai is required for Gemini. Install with: pip install google-generativeai"
            )

        super().__init__(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )

        if not api_key:
            raise AuthenticationError("Gemini API key is required")

        # Configure Gemini with REST transport (avoids gRPC connection issues)
        genai.configure(api_key=api_key, transport='rest')

        # Initialize the model
        try:
            self.client = genai.GenerativeModel(model_name=model)
        except Exception as e:
            raise ModelError(f"Failed to initialize Gemini model {model}: {e}")

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        self.provider_name = "gemini"

        # Streaming state
        self._accumulated_content = ""

    def _reset_stream_state(self):
        """Reset streaming state for a new streaming session."""
        self._accumulated_content = ""

    def _normalize_stream_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize Gemini streaming format to unified StreamChunk."""
        content = ""
        delta = ""
        finish_reason = None
        usage = None

        try:
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]

                if hasattr(candidate, 'content') and candidate.content.parts:
                    delta = candidate.content.parts[0].text
                    content = delta

                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = candidate.finish_reason.name

            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage = TokenCount(
                    prompt_tokens=getattr(chunk.usage_metadata, 'prompt_token_count', 0) or 0,
                    completion_tokens=getattr(chunk.usage_metadata, 'candidates_token_count', 0) or 0,
                    total_tokens=getattr(chunk.usage_metadata, 'total_token_count', 0) or 0
                )

        except AttributeError:
            if hasattr(chunk, 'text'):
                content = chunk.text
                delta = content
            else:
                content = str(chunk) if chunk else ""
                delta = content

        return StreamChunk(
            content=content,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=None
        )

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Gemini format."""
        return {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
        }

    def _normalize_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize JSON schema for Gemini API compatibility.

        Gemini has strict requirements:
        - Does NOT support array types like ["string", "null"]
        - Does NOT support "additionalProperties" field
        - Only accepts single type values

        This provides a safety net for schemas from any source.
        """
        if not isinstance(schema, dict):
            return schema

        normalized = {}

        # Fields that Gemini doesn't support
        unsupported_fields = ["additionalProperties", "$schema", "definitions", "$defs"]

        for key, value in schema.items():
            # Skip unsupported fields
            if key in unsupported_fields:
                continue

            if key == "type":
                # Convert array types to single type
                if isinstance(value, list):
                    # Filter out "null" and take the first non-null type
                    non_null_types = [t for t in value if t != "null"]
                    if non_null_types:
                        normalized[key] = non_null_types[0]
                    else:
                        # If only "null", default to "string"
                        normalized[key] = "string"
                else:
                    normalized[key] = value

            elif key == "properties" and isinstance(value, dict):
                # Recursively normalize nested properties
                normalized[key] = {
                    prop_key: self._normalize_schema_for_gemini(prop_value)
                    for prop_key, prop_value in value.items()
                }

            elif key == "items" and isinstance(value, dict):
                # Recursively normalize array items
                normalized[key] = self._normalize_schema_for_gemini(value)

            elif key == "required" and isinstance(value, list):
                # Keep required fields list
                normalized[key] = value

            elif isinstance(value, dict):
                # Recursively normalize nested objects
                normalized[key] = self._normalize_schema_for_gemini(value)

            elif isinstance(value, list):
                # Recursively normalize lists
                normalized[key] = [
                    self._normalize_schema_for_gemini(item) if isinstance(item, dict) else item
                    for item in value
                ]

            else:
                normalized[key] = value

        return normalized

    async def _convert_messages_to_gemini_format(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format (async to support URL downloads).

        Consolidates consecutive USER messages into a single message, ensuring images
        come before text (as required by Gemini API).
        """
        gemini_messages = []

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                if gemini_messages and gemini_messages[-1]["role"] == "user":
                    gemini_messages[-1]["parts"][0][
                        "text"
                    ] = f"System: {message.content}\n\n{gemini_messages[-1]['parts'][0]['text']}"
                else:
                    gemini_messages.append(
                        {"role": "user", "parts": [{"text": f"System: {message.content}"}]}
                    )
            elif message.role == MessageRole.USER:
                parts = []

                if isinstance(message.content, str):
                    parts.append({"text": message.content})
                elif isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append({"text": block.text})
                        elif isinstance(block, ImageBlock):
                            # Handle image blocks: convert to bytes for Gemini API
                            try:
                                # Use unified utility to convert any image URL format to bytes
                                image_bytes, mime_type = await image_url_to_bytes(
                                    block.image_url, timeout=self.timeout
                                )
                                parts.append(
                                    {"inline_data": {"mime_type": mime_type, "data": image_bytes}}
                                )
                            except Exception as e:
                                # If conversion fails, add as text placeholder
                                parts.append(
                                    {"text": f"[Image failed to load: {block.image_url}. Error: {str(e)}]"}
                                )

                # Consolidate consecutive USER messages (common pattern from LLMNode)
                # Gemini requires images before text in the same message
                if gemini_messages and gemini_messages[-1]["role"] == "user":
                    # Merge with previous user message: images first, then text
                    existing_parts = gemini_messages[-1]["parts"]

                    # Separate images and text from both messages
                    all_images = [p for p in existing_parts if "inline_data" in p]
                    all_text = [p for p in existing_parts if "text" in p]
                    all_images.extend([p for p in parts if "inline_data" in p])
                    all_text.extend([p for p in parts if "text" in p])

                    # Combine: images first, then text
                    gemini_messages[-1]["parts"] = all_images + all_text
                else:
                    # New user message
                    gemini_messages.append({"role": "user", "parts": parts})

            elif message.role == MessageRole.ASSISTANT:
                gemini_messages.append({"role": "model", "parts": [{"text": message.content}]})

        return gemini_messages

    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Send chat completion request to Gemini."""
        try:
            gemini_messages = await self._convert_messages_to_gemini_format(messages)

            # Determine if we have multimodal content
            has_multimodal = False
            for msg in gemini_messages:
                for part in msg.get("parts", []):
                    if "inline_data" in part:
                        has_multimodal = True
                        break
                if has_multimodal:
                    break

            # Prepare content for API call
            if len(gemini_messages) == 1 and not has_multimodal:
                # Simple single message with text only
                prompt = gemini_messages[0]["parts"][0]["text"]
            elif len(gemini_messages) == 1 and has_multimodal:
                # Single message with multimodal content - pass parts directly
                prompt = gemini_messages[0]["parts"]
            else:
                # Multiple messages - use chat interface for multi-turn conversations
                # For now, flatten to text (TODO: use proper chat API for multi-turn)
                prompt_parts = []
                for msg in gemini_messages:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    # Extract text from parts
                    text_parts = [p.get("text", "[non-text content]") for p in msg["parts"]]
                    text = " ".join(text_parts)
                    prompt_parts.append(f"{role}: {text}")
                prompt = "\n\n".join(prompt_parts)

            # Build generation config
            generation_config_params = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or 8192,
            }

            # Add JSON schema support (CANNOT be used with tools!)
            if json_schema:
                if tools:
                    raise ProviderError(
                        "Gemini does not support JSON schema with function calling. "
                        "Use either json_schema OR tools, not both.",
                        provider="gemini",
                    )
                generation_config_params["response_mime_type"] = "application/json"
                # Normalize schema for Gemini compatibility
                generation_config_params["response_schema"] = self._normalize_schema_for_gemini(
                    json_schema
                )

            generation_config = genai.GenerationConfig(**generation_config_params)

            # Prepare tools for Gemini (if provided)
            gemini_tools = None
            if tools:
                # Gemini expects tools wrapped in a Tool object
                
                function_declarations = []
                for tool in tools:
                    func_decl = FunctionDeclaration(
                        name=tool["name"],
                        description=tool["description"],
                        parameters=tool["parameters"]
                    )
                    function_declarations.append(func_decl)

                gemini_tools = [Tool(function_declarations=function_declarations)]

            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                tools=gemini_tools,
            )

            content = ""
            tool_calls = []

            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        func_call = part.function_call
                        tool_call = {
                            "name": func_call.name,
                            "arguments": dict(func_call.args) if func_call.args else {}
                        }
                        tool_calls.append(tool_call)
                    elif hasattr(part, 'text') and part.text:
                        content += part.text

            usage = TokenCount()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = TokenCount(
                    prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                    completion_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
                    total_tokens=getattr(response.usage_metadata, "total_token_count", 0),
                )

            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls if tool_calls else None
            )

            from ..core.client import ChatResponse

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=(
                    response.candidates[0].finish_reason.name if response.candidates else None
                ),
            )

        except Exception as e:
            raise ProviderError(f"Gemini API error: {e}", provider="gemini")

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator:
        """Send streaming chat completion request to Gemini."""
        try:
            gemini_messages = await self._convert_messages_to_gemini_format(messages)

            # Determine if we have multimodal content
            has_multimodal = False
            for msg in gemini_messages:
                for part in msg.get("parts", []):
                    if "inline_data" in part:
                        has_multimodal = True
                        break
                if has_multimodal:
                    break

            # Prepare content for API call
            if len(gemini_messages) == 1 and not has_multimodal:
                # Simple single message with text only
                prompt = gemini_messages[0]["parts"][0]["text"]
            elif len(gemini_messages) == 1 and has_multimodal:
                # Single message with multimodal content - pass parts directly
                prompt = gemini_messages[0]["parts"]
            else:
                # Multiple messages - use chat interface for multi-turn conversations
                # For now, flatten to text (TODO: use proper chat API for multi-turn)
                prompt_parts = []
                for msg in gemini_messages:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    # Extract text from parts
                    text_parts = [p.get("text", "[non-text content]") for p in msg["parts"]]
                    text = " ".join(text_parts)
                    prompt_parts.append(f"{role}: {text}")
                prompt = "\n\n".join(prompt_parts)

            # Build generation config
            generation_config_params = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or 8192,
            }
            if json_schema:
                if tools:
                    raise ProviderError(
                        "Gemini does not support JSON mode with function calling. "
                        "Use either json_mode/json_schema OR tools, not both.",
                        provider="gemini",
                    )
                generation_config_params["response_mime_type"] = "application/json"
                # Normalize schema for Gemini compatibility
                generation_config_params["response_schema"] = self._normalize_schema_for_gemini(
                    json_schema
                )

            generation_config = genai.GenerationConfig(**generation_config_params)

            response_stream = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                stream=True,
            )

            # Reset stream state for new streaming session
            self._reset_stream_state()

            for chunk in response_stream:
                normalized_chunk = self._normalize_stream_chunk(chunk)

                if normalized_chunk.delta:
                    self._accumulated_content += normalized_chunk.delta

                normalized_chunk.content = self._accumulated_content

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
