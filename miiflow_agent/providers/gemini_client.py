"""Google Gemini client implementation."""

import asyncio
import re
import warnings
from typing import Any, AsyncIterator, Dict, List, Optional


def _sanitize_tool_name_for_gemini(name: str) -> str:
    """Sanitize tool name to match Gemini's function name requirements.

    Gemini requires function names to:
    - Start with a letter or underscore
    - Contain only: a-z, A-Z, 0-9, _, ., :, -
    - Maximum length of 64 characters
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_.\-:]", "_", name)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Ensure starts with letter or underscore
    if sanitized and not re.match(r"^[a-zA-Z_]", sanitized):
        sanitized = "_" + sanitized
    # Strip trailing underscores and limit length
    sanitized = sanitized.rstrip("_")[:64]
    return sanitized or "_unnamed"

try:
    import google.generativeai as genai
    from google.generativeai.types import (
        FunctionDeclaration,
        HarmBlockThreshold,
        HarmCategory,
        Tool,
    )

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Suppress warnings about unrecognized FinishReason enum values from proto-plus.
# Gemini 2.5 models return new enum values (like 12, 15) that aren't yet in the
# google-generativeai SDK's protobuf definitions. Our code handles these gracefully
# via _get_finish_reason_name(), so the warnings are just noise.
# See: https://github.com/langchain-ai/langchain-google/issues/1268
warnings.filterwarnings(
    "ignore",
    message=r"Unrecognized FinishReason enum value: \d+",
    category=UserWarning,
    module=r"proto\.marshal\.rules\.enums",
)

from ..core.client import ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError
from ..core.message import DocumentBlock, ImageBlock, Message, MessageRole, TextBlock
from ..core.metrics import TokenCount
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.stream_normalizer import GeminiStreamNormalizer
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
            model=model, api_key=api_key, timeout=timeout, max_retries=max_retries, **kwargs
        )

        if not api_key:
            raise AuthenticationError("Gemini API key is required")

        # Configure Gemini with REST transport (avoids gRPC connection issues)
        genai.configure(api_key=api_key, transport="rest")

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

        # Stream normalizer for unified streaming handling
        self._stream_normalizer = GeminiStreamNormalizer()

        # Tool name mapping: sanitized_name -> original_name
        self._tool_name_mapping: Dict[str, str] = {}

    def _get_finish_reason_name(self, finish_reason: Any) -> Optional[str]:
        """Safely extract finish_reason name, handling both enum and int values.

        Gemini 2.5 models may return new undocumented finish_reason enum values
        (like 12) that come as raw integers instead of enum objects.
        """
        if finish_reason is None:
            return None
        # If it's an enum with a name attribute, use that
        if hasattr(finish_reason, "name"):
            return finish_reason.name
        # If it's an integer (unrecognized enum value), convert to string
        if isinstance(finish_reason, int):
            return f"UNKNOWN_{finish_reason}"
        # Fallback: convert to string
        return str(finish_reason)

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Gemini format."""
        # Normalize parameters to remove unsupported fields (default, additionalProperties, etc.)
        normalized_parameters = normalize_json_schema(schema["parameters"], SchemaMode.GEMINI_COMPAT)
        return {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": normalized_parameters,
        }

    def _extract_system_instruction(self, messages: List[Message]) -> Optional[str]:
        """Extract system instruction from messages.

        Gemini supports system instructions via the system_instruction parameter.
        This extracts and combines all system messages into a single instruction.
        """
        system_parts = []
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                if isinstance(message.content, str):
                    system_parts.append(message.content)
                elif isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            system_parts.append(block.text)

        if system_parts:
            return "\n\n".join(system_parts)
        return None

    async def _convert_messages_to_gemini_format(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format (async to support URL downloads).

        Consolidates consecutive USER messages into a single message, ensuring images
        come before text (as required by Gemini API).

        Note: System messages are handled separately via _extract_system_instruction()
        and passed to the model via the system_instruction parameter.
        """
        gemini_messages = []

        for message in messages:
            # Skip system messages - they're handled via system_instruction parameter
            if message.role == MessageRole.SYSTEM:
                continue
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
                                    {
                                        "text": f"[Image failed to load: {block.image_url}. Error: {str(e)}]"
                                    }
                                )
                        elif isinstance(block, DocumentBlock):
                            # Handle document blocks: extract text and add as text content
                            # Gemini doesn't have native document support like Anthropic,
                            # so we extract PDF text similar to OpenAI's approach
                            try:
                                from ..utils.pdf_extractor import extract_pdf_text_simple

                                pdf_text = extract_pdf_text_simple(block.document_url)

                                filename_info = f" [{block.filename}]" if block.filename else ""
                                pdf_content = f"[PDF Document{filename_info}]\n\n{pdf_text}"

                                parts.append({"text": pdf_content})
                            except Exception as e:
                                # If extraction fails, add error as text placeholder
                                filename_info = f" {block.filename}" if block.filename else ""
                                parts.append(
                                    {"text": f"[Error processing PDF{filename_info}: {str(e)}]"}
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
                parts = []

                # Add text content if present
                if message.content:
                    if isinstance(message.content, str):
                        parts.append({"text": message.content})
                    elif isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                parts.append({"text": block.text})

                # Add function calls if present (for multi-turn with tool use)
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        # Handle both dict format and object format
                        if isinstance(tool_call, dict):
                            func_name = tool_call.get("function", {}).get("name") or tool_call.get(
                                "name"
                            )
                            func_args = tool_call.get("function", {}).get(
                                "arguments"
                            ) or tool_call.get("arguments", {})
                            # Parse arguments if they're a string
                            if isinstance(func_args, str):
                                import json

                                try:
                                    func_args = json.loads(func_args)
                                except json.JSONDecodeError:
                                    func_args = {}
                        else:
                            # Object with attributes
                            func_name = getattr(tool_call, "name", None) or getattr(
                                getattr(tool_call, "function", None), "name", None
                            )
                            func_args = getattr(tool_call, "arguments", {}) or getattr(
                                getattr(tool_call, "function", None), "arguments", {}
                            )

                        if func_name:
                            parts.append(
                                {
                                    "function_call": {
                                        "name": func_name,
                                        "args": func_args if isinstance(func_args, dict) else {},
                                    }
                                }
                            )

                # Only add if we have parts
                if parts:
                    gemini_messages.append({"role": "model", "parts": parts})
                elif not message.content and not message.tool_calls:
                    # Empty assistant message - add placeholder
                    gemini_messages.append({"role": "model", "parts": [{"text": ""}]})

            elif message.role == MessageRole.TOOL:
                # Tool results need to be sent as function_response with role "user"
                # Extract tool name from tool_call_id or use a default
                tool_name = getattr(message, "name", None) or "tool_result"

                # Parse the content as the result
                result_content = (
                    message.content if isinstance(message.content, str) else str(message.content)
                )

                # Gemini expects function responses as user messages with function_response parts
                # Check if last message was also a tool response - consolidate them
                if gemini_messages and gemini_messages[-1]["role"] == "user":
                    # Check if it contains function_response parts
                    has_function_response = any(
                        "function_response" in p for p in gemini_messages[-1]["parts"]
                    )
                    if has_function_response:
                        # Add to existing function response message
                        gemini_messages[-1]["parts"].append(
                            {
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": result_content},
                                }
                            }
                        )
                    else:
                        # New message for function response
                        gemini_messages.append(
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "function_response": {
                                            "name": tool_name,
                                            "response": {"result": result_content},
                                        }
                                    }
                                ],
                            }
                        )
                else:
                    gemini_messages.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "function_response": {
                                        "name": tool_name,
                                        "response": {"result": result_content},
                                    }
                                }
                            ],
                        }
                    )

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
            # Extract system instruction from messages
            system_instruction = self._extract_system_instruction(messages)

            # Convert remaining messages to Gemini format
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
            # Gemini's generate_content supports multi-turn by passing list of messages
            # with 'role' ('user' or 'model') and 'parts' keys
            if len(gemini_messages) == 0:
                raise ProviderError(
                    "No user or assistant messages provided to Gemini", provider="gemini"
                )
            elif len(gemini_messages) == 1:
                # Single message - can pass just the parts for simplicity
                parts = gemini_messages[0]["parts"]
                if has_multimodal:
                    prompt = parts
                elif len(parts) == 1:
                    # Single text part - can pass as string
                    prompt = parts[0]["text"]
                else:
                    # Multiple text parts - join them together
                    prompt = "\n\n".join(p["text"] for p in parts if "text" in p)
            else:
                # Multiple messages - pass as list for proper multi-turn conversation
                # Gemini expects: [{'role': 'user'|'model', 'parts': [...]}]
                prompt = gemini_messages

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
                generation_config_params["response_schema"] = normalize_json_schema(
                    json_schema, SchemaMode.GEMINI_COMPAT
                )

            generation_config = genai.GenerationConfig(**generation_config_params)

            # Prepare tools for Gemini (if provided)
            gemini_tools = None
            if tools:
                # Clear previous mapping
                self._tool_name_mapping.clear()

                # Gemini expects tools wrapped in a Tool object
                function_declarations = []
                for tool in tools:
                    original_name = tool["name"]
                    sanitized_name = _sanitize_tool_name_for_gemini(original_name)

                    # Store mapping if name was changed
                    if sanitized_name != original_name:
                        self._tool_name_mapping[sanitized_name] = original_name

                    # Normalize parameters to remove unsupported fields (minimum, maximum, etc.)
                    normalized_parameters = normalize_json_schema(
                        tool["parameters"], SchemaMode.GEMINI_COMPAT
                    )
                    func_decl = FunctionDeclaration(
                        name=sanitized_name,
                        description=tool["description"],
                        parameters=normalized_parameters,
                    )
                    function_declarations.append(func_decl)
                gemini_tools = [Tool(function_declarations=function_declarations)]

            # Create model with system instruction if provided
            # Gemini requires system_instruction to be set at model creation time
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_instruction,
                )
            else:
                model = self.client

            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings,
                tools=gemini_tools,
            )

            content = ""
            tool_calls = []

            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        func_call = part.function_call
                        # Map sanitized name back to original
                        tool_name = self._tool_name_mapping.get(func_call.name, func_call.name)
                        # Format as OpenAI-compatible tool call structure
                        tool_call = {
                            "id": f"gemini_{tool_name}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": dict(func_call.args) if func_call.args else {},
                            },
                        }
                        tool_calls.append(tool_call)
                    elif hasattr(part, "text") and part.text:
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
                tool_calls=tool_calls if tool_calls else None,
            )

            from ..core.client import ChatResponse

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=(
                    self._get_finish_reason_name(response.candidates[0].finish_reason)
                    if response.candidates
                    else None
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
            # Extract system instruction from messages
            system_instruction = self._extract_system_instruction(messages)

            # Convert remaining messages to Gemini format
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
            # Gemini's generate_content supports multi-turn by passing list of messages
            # with 'role' ('user' or 'model') and 'parts' keys
            if len(gemini_messages) == 0:
                raise ProviderError(
                    "No user or assistant messages provided to Gemini", provider="gemini"
                )
            elif len(gemini_messages) == 1:
                # Single message - can pass just the parts for simplicity
                parts = gemini_messages[0]["parts"]
                if has_multimodal:
                    prompt = parts
                elif len(parts) == 1:
                    # Single text part - can pass as string
                    prompt = parts[0]["text"]
                else:
                    # Multiple text parts - join them together
                    prompt = "\n\n".join(p["text"] for p in parts if "text" in p)
            else:
                # Multiple messages - pass as list for proper multi-turn conversation
                # Gemini expects: [{'role': 'user'|'model', 'parts': [...]}]
                prompt = gemini_messages

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
                generation_config_params["response_schema"] = normalize_json_schema(
                    json_schema, SchemaMode.GEMINI_COMPAT
                )

            generation_config = genai.GenerationConfig(**generation_config_params)

            # Prepare tools for Gemini (if provided)
            gemini_tools = None
            if tools:
                # Clear previous mapping
                self._tool_name_mapping.clear()

                function_declarations = []
                for tool in tools:
                    original_name = tool["name"]
                    sanitized_name = _sanitize_tool_name_for_gemini(original_name)

                    # Store mapping if name was changed
                    if sanitized_name != original_name:
                        self._tool_name_mapping[sanitized_name] = original_name

                    # Normalize parameters to remove unsupported fields (minimum, maximum, etc.)
                    normalized_parameters = normalize_json_schema(
                        tool["parameters"], SchemaMode.GEMINI_COMPAT
                    )
                    func_decl = FunctionDeclaration(
                        name=sanitized_name,
                        description=tool["description"],
                        parameters=normalized_parameters,
                    )
                    function_declarations.append(func_decl)
                gemini_tools = [Tool(function_declarations=function_declarations)]

            # Create model with system instruction if provided
            # Gemini requires system_instruction to be set at model creation time
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_instruction,
                )
            else:
                model = self.client

            # Build kwargs for generate_content - only include tools if provided
            generate_kwargs = {
                "generation_config": generation_config,
                "safety_settings": self.safety_settings,
                "stream": True,
            }
            if gemini_tools:
                generate_kwargs["tools"] = gemini_tools

            try:
                response_stream = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    **generate_kwargs,
                )
            except Exception as stream_error:
                # Streaming failed - try non-streaming to get actual error message
                try:
                    generate_kwargs["stream"] = False
                    await asyncio.to_thread(
                        model.generate_content,
                        prompt,
                        **generate_kwargs,
                    )
                except Exception as detail_error:
                    raise ProviderError(
                        f"Gemini API error: {detail_error}", provider="gemini"
                    )
                # If non-streaming succeeds, re-raise original streaming error
                raise ProviderError(
                    f"Gemini streaming error: {stream_error}", provider="gemini"
                )

            # Reset stream state for new streaming session
            self._stream_normalizer.reset_state()

            for chunk in response_stream:
                normalized_chunk = self._stream_normalizer.normalize_chunk(chunk)

                # Map tool names back to original names
                if normalized_chunk.tool_calls:
                    for tool_call in normalized_chunk.tool_calls:
                        # Name is inside the "function" object
                        func_data = tool_call.get("function", {})
                        if func_data.get("name"):
                            original_name = self._tool_name_mapping.get(
                                func_data["name"], func_data["name"]
                            )
                            func_data["name"] = original_name

                yield normalized_chunk

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Gemini streaming error: {e}", provider="gemini")
