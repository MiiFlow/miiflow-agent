"""Google Gemini client implementation.

Uses direct REST API calls via httpx instead of the google-generativeai SDK
for API communication. This allows preserving fields like `thoughtSignature`
on functionCall parts, which the SDK's protobuf layer strips out.

The SDK is still imported for GEMINI_AVAILABLE detection only.
"""

import base64
import json
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx


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
    import google.generativeai as genai  # noqa: F401 – presence check only

    GEMINI_AVAILABLE = True
except ImportError:
    # SDK is no longer required for API calls (we use httpx REST API directly),
    # but we keep it as an install-time dependency marker.
    GEMINI_AVAILABLE = True

from ..core.client import ModelClient
from ..core.exceptions import AuthenticationError, ProviderError
from ..core.message import DocumentBlock, ImageBlock, Message, MessageRole, TextBlock
from ..core.metrics import TokenCount
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.stream_normalizer import GeminiStreamNormalizer
from ..core.streaming import StreamChunk
from ..utils.image import image_url_to_bytes

# Base URL for Gemini REST API
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Safety settings in REST format
_SAFETY_SETTINGS_REST = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    parts = snake_str.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _convert_to_rest_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert internal snake_case message dicts to REST API camelCase format.

    Handles: function_call -> functionCall, function_response -> functionResponse,
    inline_data -> inlineData, mime_type -> mimeType.
    Preserves thought_signature -> thoughtSignature on functionCall parts.
    """
    rest_messages = []
    for msg in messages:
        rest_msg = {"role": msg["role"], "parts": []}
        for part in msg.get("parts", []):
            if "text" in part:
                rest_msg["parts"].append({"text": part["text"]})
            elif "function_call" in part:
                fc = part["function_call"]
                rest_fc: Dict[str, Any] = {"name": fc["name"], "args": fc.get("args", {})}
                rest_part: Dict[str, Any] = {"functionCall": rest_fc}
                # thoughtSignature is a sibling of functionCall in the part
                if "thought_signature" in fc:
                    rest_part["thoughtSignature"] = fc["thought_signature"]
                rest_msg["parts"].append(rest_part)
            elif "function_response" in part:
                fr = part["function_response"]
                rest_msg["parts"].append(
                    {"functionResponse": {"name": fr["name"], "response": fr.get("response", {})}}
                )
            elif "inline_data" in part:
                id_part = part["inline_data"]
                rest_msg["parts"].append(
                    {"inlineData": {"mimeType": id_part["mime_type"], "data": id_part["data"]}}
                )
            else:
                # Pass through unknown parts as-is
                rest_msg["parts"].append(part)
        rest_messages.append(rest_msg)
    return rest_messages


def _parse_rest_response(
    data: Dict[str, Any], tool_name_mapping: Dict[str, str]
) -> Tuple[str, List[Dict[str, Any]], TokenCount, Optional[str]]:
    """Parse REST API response JSON.

    Returns (content, tool_calls, usage, finish_reason).
    Extracts thoughtSignature from functionCall parts and stores in function_call_metadata.
    """
    content = ""
    tool_calls: List[Dict[str, Any]] = []
    finish_reason = None

    candidates = data.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                gemini_name = fc.get("name", "")
                original_name = tool_name_mapping.get(gemini_name, gemini_name)
                tool_call: Dict[str, Any] = {
                    "id": f"gemini_{original_name}",
                    "type": "function",
                    "function": {
                        "name": original_name,
                        "arguments": fc.get("args", {}),
                    },
                    "function_call_metadata": {
                        "gemini_function_name": gemini_name,
                    },
                }
                # thoughtSignature is a sibling of functionCall in the part,
                # not nested inside functionCall.
                if "thoughtSignature" in part:
                    tool_call["function_call_metadata"]["thought_signature"] = part[
                        "thoughtSignature"
                    ]
                tool_calls.append(tool_call)
            elif "text" in part:
                content += part["text"]

        finish_reason = candidate.get("finishReason")

    usage_meta = data.get("usageMetadata", {})
    usage = TokenCount(
        prompt_tokens=usage_meta.get("promptTokenCount", 0) or 0,
        completion_tokens=usage_meta.get("candidatesTokenCount", 0) or 0,
        total_tokens=usage_meta.get("totalTokenCount", 0) or 0,
    )

    return content, tool_calls, usage, finish_reason


class GeminiClient(ModelClient):
    """Google Gemini client implementation using direct REST API calls."""

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

        self.provider_name = "gemini"

        # Stream normalizer for unified streaming handling
        self._stream_normalizer = GeminiStreamNormalizer()

        # Tool name mapping: sanitized_name -> original_name
        self._tool_name_mapping: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # REST API URL builders
    # ------------------------------------------------------------------

    def _build_rest_url(self, streaming: bool = False) -> str:
        """Build the REST API endpoint URL."""
        # Strip "models/" prefix if present — the SDK accepted both formats
        # but the REST URL already includes /models/ in the path.
        model = self.model
        if model.startswith("models/"):
            model = model[len("models/"):]
        if streaming:
            return (
                f"{_GEMINI_API_BASE}/models/{model}:streamGenerateContent"
                f"?alt=sse&key={self.api_key}"
            )
        return (
            f"{_GEMINI_API_BASE}/models/{model}:generateContent"
            f"?key={self.api_key}"
        )

    # ------------------------------------------------------------------
    # Request body builders
    # ------------------------------------------------------------------

    def _build_generation_config_rest(
        self,
        temperature: float,
        max_tokens: Optional[int],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build generationConfig in REST format."""
        config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens or 8192,
        }
        if json_schema:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = normalize_json_schema(json_schema, SchemaMode.GEMINI_COMPAT)
        return config

    def _build_tools_rest(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build tools/functionDeclarations in REST format."""
        self._tool_name_mapping.clear()

        function_declarations = []
        for tool in tools:
            original_name = tool["name"]
            sanitized_name = _sanitize_tool_name_for_gemini(original_name)
            if sanitized_name != original_name:
                self._tool_name_mapping[sanitized_name] = original_name

            normalized_parameters = normalize_json_schema(
                tool["parameters"], SchemaMode.GEMINI_COMPAT
            )
            function_declarations.append(
                {
                    "name": sanitized_name,
                    "description": tool["description"],
                    "parameters": normalized_parameters,
                }
            )
        return [{"functionDeclarations": function_declarations}]

    def _build_rest_request_body(
        self,
        gemini_messages: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the full REST API request body."""
        rest_contents = _convert_to_rest_format(gemini_messages)

        body: Dict[str, Any] = {
            "contents": rest_contents,
            "generationConfig": generation_config,
            "safetySettings": _SAFETY_SETTINGS_REST,
        }
        if tools:
            body["tools"] = tools
        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        return body

    # ------------------------------------------------------------------
    # Helpers shared between achat/astream_chat
    # ------------------------------------------------------------------

    def _get_finish_reason_name(self, finish_reason: Any) -> Optional[str]:
        """Safely extract finish_reason name, handling both enum and string values."""
        if finish_reason is None:
            return None
        if isinstance(finish_reason, str):
            return finish_reason
        if hasattr(finish_reason, "name"):
            return finish_reason.name
        if isinstance(finish_reason, int):
            return f"UNKNOWN_{finish_reason}"
        return str(finish_reason)

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to Gemini format."""
        normalized_parameters = normalize_json_schema(schema["parameters"], SchemaMode.GEMINI_COMPAT)
        return {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": normalized_parameters,
        }

    def _extract_system_instruction(self, messages: List[Message]) -> Optional[str]:
        """Extract system instruction from messages."""
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

        Image data is base64-encoded for the REST API.

        Note: System messages are handled separately via _extract_system_instruction()
        and passed to the model via the systemInstruction parameter.
        """
        gemini_messages: List[Dict[str, Any]] = []

        for message in messages:
            # Skip system messages - they're handled via system_instruction parameter
            if message.role == MessageRole.SYSTEM:
                continue
            elif message.role == MessageRole.USER:
                parts: List[Dict[str, Any]] = []

                if isinstance(message.content, str):
                    parts.append({"text": message.content})
                elif isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append({"text": block.text})
                        elif isinstance(block, ImageBlock):
                            try:
                                image_bytes, mime_type = await image_url_to_bytes(
                                    block.image_url, timeout=self.timeout, resize=True
                                )
                                # REST API needs base64 string
                                b64_data = base64.b64encode(image_bytes).decode("utf-8")
                                parts.append(
                                    {"inline_data": {"mime_type": mime_type, "data": b64_data}}
                                )
                            except Exception as e:
                                parts.append(
                                    {
                                        "text": f"[Image failed to load: {block.image_url}. Error: {str(e)}]"
                                    }
                                )
                        elif isinstance(block, DocumentBlock):
                            try:
                                filename_info = f" [{block.filename}]" if block.filename else ""
                                if block.document_type == "pdf":
                                    from ..utils.pdf_extractor import extract_pdf_text_simple

                                    text = extract_pdf_text_simple(block.document_url)
                                    doc_content = f"[PDF Document{filename_info}]\n\n{text}"
                                else:
                                    resp = httpx.get(
                                        block.document_url, timeout=30, follow_redirects=True
                                    )
                                    resp.raise_for_status()
                                    text = resp.content.decode("utf-8", errors="replace")
                                    doc_content = f"[Document{filename_info}]\n\n{text}"
                                parts.append({"text": doc_content})
                            except Exception as e:
                                filename_info = f" {block.filename}" if block.filename else ""
                                parts.append(
                                    {"text": f"[Error processing document{filename_info}: {str(e)}]"}
                                )

                # Consolidate consecutive USER messages
                if gemini_messages and gemini_messages[-1]["role"] == "user":
                    existing_parts = gemini_messages[-1]["parts"]
                    all_images = [p for p in existing_parts if "inline_data" in p]
                    all_text = [p for p in existing_parts if "text" in p]
                    all_images.extend([p for p in parts if "inline_data" in p])
                    all_text.extend([p for p in parts if "text" in p])
                    gemini_messages[-1]["parts"] = all_images + all_text
                else:
                    gemini_messages.append({"role": "user", "parts": parts})

            elif message.role == MessageRole.ASSISTANT:
                parts = []

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
                        if isinstance(tool_call, dict):
                            func_name = tool_call.get("function", {}).get(
                                "name"
                            ) or tool_call.get("name")
                            func_args = tool_call.get("function", {}).get(
                                "arguments"
                            ) or tool_call.get("arguments", {})
                            if isinstance(func_args, str):
                                try:
                                    func_args = json.loads(func_args)
                                except json.JSONDecodeError:
                                    func_args = {}

                            # Use gemini_function_name from metadata if available
                            metadata = tool_call.get("function_call_metadata", {})
                            gemini_name = metadata.get("gemini_function_name", func_name)
                        else:
                            func_name = getattr(tool_call, "name", None) or getattr(
                                getattr(tool_call, "function", None), "name", None
                            )
                            func_args = getattr(tool_call, "arguments", {}) or getattr(
                                getattr(tool_call, "function", None), "arguments", {}
                            )
                            metadata = {}
                            gemini_name = func_name

                        if func_name:
                            fc_part: Dict[str, Any] = {
                                "name": gemini_name,
                                "args": func_args if isinstance(func_args, dict) else {},
                            }
                            # Preserve thought_signature from metadata
                            thought_sig = metadata.get("thought_signature")
                            if thought_sig:
                                fc_part["thought_signature"] = thought_sig
                            parts.append({"function_call": fc_part})

                if parts:
                    gemini_messages.append({"role": "model", "parts": parts})
                elif not message.content and not message.tool_calls:
                    gemini_messages.append({"role": "model", "parts": [{"text": ""}]})

            elif message.role == MessageRole.TOOL:
                tool_name = getattr(message, "name", None) or "tool_result"
                result_content = (
                    message.content if isinstance(message.content, str) else str(message.content)
                )

                if gemini_messages and gemini_messages[-1]["role"] == "user":
                    has_function_response = any(
                        "function_response" in p for p in gemini_messages[-1]["parts"]
                    )
                    if has_function_response:
                        gemini_messages[-1]["parts"].append(
                            {
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": result_content},
                                }
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

    # ------------------------------------------------------------------
    # achat – non-streaming
    # ------------------------------------------------------------------

    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Send chat completion request to Gemini via REST API."""
        try:
            system_instruction = self._extract_system_instruction(messages)
            gemini_messages = await self._convert_messages_to_gemini_format(messages)

            if not gemini_messages:
                raise ProviderError(
                    "No user or assistant messages provided to Gemini", provider="gemini"
                )

            # Validate json_schema + tools mutual exclusivity
            if json_schema and tools:
                raise ProviderError(
                    "Gemini does not support JSON schema with function calling. "
                    "Use either json_schema OR tools, not both.",
                    provider="gemini",
                )

            gen_config = self._build_generation_config_rest(temperature, max_tokens, json_schema)
            rest_tools = self._build_tools_rest(tools) if tools else None
            body = self._build_rest_request_body(
                gemini_messages, gen_config, rest_tools, system_instruction
            )

            url = self._build_rest_url(streaming=False)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=body)
                if resp.status_code != 200:
                    raise ProviderError(
                        f"Gemini API error ({resp.status_code}): {resp.text}",
                        provider="gemini",
                    )
                data = resp.json()

            content, tool_calls, usage, finish_reason = _parse_rest_response(
                data, self._tool_name_mapping
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
                finish_reason=self._get_finish_reason_name(finish_reason),
            )

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Gemini API error: {e}", provider="gemini")

    # ------------------------------------------------------------------
    # astream_chat – SSE streaming
    # ------------------------------------------------------------------

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator:
        """Send streaming chat completion request to Gemini via REST SSE API."""
        try:
            system_instruction = self._extract_system_instruction(messages)
            gemini_messages = await self._convert_messages_to_gemini_format(messages)

            if not gemini_messages:
                raise ProviderError(
                    "No user or assistant messages provided to Gemini", provider="gemini"
                )

            if json_schema and tools:
                raise ProviderError(
                    "Gemini does not support JSON mode with function calling. "
                    "Use either json_mode/json_schema OR tools, not both.",
                    provider="gemini",
                )

            gen_config = self._build_generation_config_rest(temperature, max_tokens, json_schema)
            rest_tools = self._build_tools_rest(tools) if tools else None
            body = self._build_rest_request_body(
                gemini_messages, gen_config, rest_tools, system_instruction
            )

            url = self._build_rest_url(streaming=True)

            # Reset stream state for new streaming session
            self._stream_normalizer.reset_state()

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=body) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        raise ProviderError(
                            f"Gemini API error ({resp.status_code}): {error_body.decode()}",
                            provider="gemini",
                        )

                    async for chunk_dict in self._parse_sse_stream(resp):
                        # Feed dict chunks to normalizer
                        normalized_chunk = self._stream_normalizer.normalize_chunk(chunk_dict)

                        # Map tool names back to original names
                        if normalized_chunk.tool_calls:
                            for tool_call in normalized_chunk.tool_calls:
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

    async def _parse_sse_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[Dict[str, Any]]:
        """Parse SSE stream from Gemini REST API.

        Yields parsed JSON dicts from `data: {...}` lines.
        """
        buffer = ""
        async for line in response.aiter_lines():
            line = line.strip()
            if line.startswith("data: "):
                json_str = line[6:]  # strip "data: " prefix
                if json_str:
                    try:
                        yield json.loads(json_str)
                    except json.JSONDecodeError:
                        # Accumulate partial JSON across lines (shouldn't happen
                        # with Gemini SSE, but be defensive)
                        buffer += json_str
                        try:
                            yield json.loads(buffer)
                            buffer = ""
                        except json.JSONDecodeError:
                            pass
            elif not line and buffer:
                # Empty line after accumulated buffer
                try:
                    yield json.loads(buffer)
                except json.JSONDecodeError:
                    pass
                buffer = ""
