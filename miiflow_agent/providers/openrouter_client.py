"""OpenRouter client implementation using direct API calls."""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError, RateLimitError
from ..core.exceptions import TimeoutError as MiiflowTimeoutError
from ..core.message import Message, MessageRole
from ..core.metrics import TokenCount
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.streaming import StreamChunk


class OpenRouterClient(ModelClient):
    """OpenRouter client implementation using direct API calls."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        app_name: Optional[str] = None,
        app_url: Optional[str] = None,
        **kwargs,
    ):
        if not api_key:
            raise AuthenticationError("OpenRouter API key is required", provider="openrouter")

        super().__init__(
            model=model, api_key=api_key, timeout=timeout, max_retries=max_retries, **kwargs
        )

        self.api_key = api_key
        self.app_name = app_name
        self.app_url = app_url
        self.provider_name = "openrouter"

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_name:
            headers["X-Title"] = self.app_name
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        return headers

    def _convert_message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert Message to OpenRouter message format."""
        msg_dict: Dict[str, Any] = {"role": message.role.value}

        if message.content:
            # Handle multimodal content
            if isinstance(message.content, list):
                content_parts = []
                for part in message.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_parts.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            content_parts.append(part)
                    else:
                        content_parts.append({"type": "text", "text": str(part)})
                msg_dict["content"] = content_parts
            else:
                msg_dict["content"] = message.content

        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id if hasattr(tc, "id") else tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": (
                            tc.function.name
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("name", "")
                        ),
                        "arguments": (
                            tc.function.arguments
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("arguments", "")
                        ),
                    },
                }
                for tc in message.tool_calls
            ]

        if message.tool_call_id:
            msg_dict["tool_call_id"] = message.tool_call_id

        if message.name:
            msg_dict["name"] = message.name

        return msg_dict

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenRouter format."""
        return [self._convert_message_to_dict(msg) for msg in messages]

    def _parse_error_response(self, response: httpx.Response) -> str:
        """Parse error response to get detailed error message."""
        try:
            data = response.json()
            if isinstance(data, dict):
                # OpenRouter error format: {"error": {"message": "...", "code": ..., "metadata": {"raw": "..."}}}
                if "error" in data:
                    error_obj = data["error"]
                    if isinstance(error_obj, dict):
                        # Check for nested provider error in metadata.raw
                        metadata = error_obj.get("metadata", {})
                        raw_error = metadata.get("raw") if isinstance(metadata, dict) else None
                        provider_name = (
                            metadata.get("provider_name", "") if isinstance(metadata, dict) else ""
                        )

                        if raw_error:
                            try:
                                # Parse the nested JSON error from the provider
                                raw_data = json.loads(raw_error)
                                if isinstance(raw_data, dict) and "error" in raw_data:
                                    nested_error = raw_data["error"]
                                    if isinstance(nested_error, dict):
                                        nested_msg = nested_error.get("message", "")
                                        if nested_msg:
                                            prefix = f"[{provider_name}] " if provider_name else ""
                                            return f"{prefix}{nested_msg}"
                            except (json.JSONDecodeError, ValueError):
                                # If raw isn't valid JSON, use it directly
                                prefix = f"[{provider_name}] " if provider_name else ""
                                return f"{prefix}{raw_error}"

                        # Fall back to top-level message
                        msg = error_obj.get("message", "")
                        code = error_obj.get("code", "")
                        if msg:
                            return f"{msg} (code: {code})" if code else msg
                    elif isinstance(error_obj, str):
                        return error_obj
                # Alternative format
                if "message" in data:
                    return data["message"]
        except (json.JSONDecodeError, ValueError):
            pass
        return response.text or f"HTTP {response.status_code}"

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error response and raise appropriate exception."""
        error_msg = self._parse_error_response(response)
        status_code = response.status_code

        if status_code == 401:
            raise AuthenticationError(error_msg, self.provider_name)
        elif status_code == 429:
            raise RateLimitError(error_msg, self.provider_name)
        elif status_code == 400:
            raise ModelError(error_msg, self.model)
        elif status_code == 404:
            raise ModelError(f"Model not found: {error_msg}", self.model)
        else:
            raise ProviderError(
                f"OpenRouter API error ({status_code}): {error_msg}", self.provider_name
            )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True
    )
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to OpenRouter."""
        try:
            openrouter_messages = self._convert_messages(messages)

            # Build request payload
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": openrouter_messages,
                "temperature": temperature,
            }

            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            # Add structured output support via response_format
            if json_schema:
                normalized_schema = normalize_json_schema(
                    json_schema, SchemaMode.STRICT, ensure_all_required=True
                )
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "strict": True,
                        "schema": normalized_schema,
                    },
                }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )

            if response.status_code != 200:
                self._handle_error_response(response)

            data = response.json()

            content = ""
            tool_calls = None

            if data.get("choices") and len(data["choices"]) > 0:
                choice = data["choices"][0]
                message_data = choice.get("message", {})
                content = message_data.get("content") or ""
                tool_calls = message_data.get("tool_calls")

            usage = TokenCount()
            if data.get("usage"):
                usage = TokenCount(
                    prompt_tokens=data["usage"].get("prompt_tokens", 0),
                    completion_tokens=data["usage"].get("completion_tokens", 0),
                    total_tokens=data["usage"].get("total_tokens", 0),
                )

            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            )

            finish_reason = None
            if data.get("choices") and len(data["choices"]) > 0:
                finish_reason = data["choices"][0].get("finish_reason")

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=finish_reason,
                metadata={"response_id": data.get("id")} if data.get("id") else {},
            )

        except httpx.TimeoutException as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except (AuthenticationError, RateLimitError, ModelError, ProviderError) as e:
            raise
        except Exception as e:
            raise ProviderError(
                f"OpenRouter API error: {str(e)}", self.provider_name, original_error=e
            )

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming chat completion request to OpenRouter."""
        try:
            openrouter_messages = self._convert_messages(messages)

            # Build request payload
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": openrouter_messages,
                "temperature": temperature,
                "stream": True,
            }

            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            # Add structured output support via response_format
            if json_schema:
                normalized_schema = normalize_json_schema(
                    json_schema, SchemaMode.STRICT, ensure_all_required=True
                )
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "strict": True,
                        "schema": normalized_schema,
                    },
                }

            # Track tool call state for streaming
            current_tool_calls: Dict[int, Dict[str, Any]] = {}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        # Read the full response for error handling
                        await response.aread()
                        self._handle_error_response(response)

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if not chunk_data.get("choices"):
                            continue

                        choice = chunk_data["choices"][0]
                        delta = choice.get("delta", {})

                        # Extract content delta
                        content_delta = delta.get("content")

                        # Handle tool calls in streaming
                        tool_calls = None
                        if delta.get("tool_calls"):
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in current_tool_calls:
                                    current_tool_calls[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc.get("id"):
                                    current_tool_calls[idx]["id"] = tc["id"]
                                if tc.get("function"):
                                    if tc["function"].get("name"):
                                        current_tool_calls[idx]["function"]["name"] += tc[
                                            "function"
                                        ]["name"]
                                    if tc["function"].get("arguments"):
                                        current_tool_calls[idx]["function"]["arguments"] += tc[
                                            "function"
                                        ]["arguments"]

                        finish_reason = choice.get("finish_reason")

                        # Build tool calls list if we have accumulated any
                        if finish_reason and current_tool_calls:
                            tool_calls = list(current_tool_calls.values())

                        # Only yield if there's content or metadata
                        if content_delta or tool_calls or finish_reason:
                            yield StreamChunk(
                                delta=content_delta,
                                tool_calls=tool_calls,
                                finish_reason=finish_reason,
                            )

        except httpx.TimeoutException as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except (AuthenticationError, RateLimitError, ModelError, ProviderError):
            raise
        except Exception as e:
            raise ProviderError(
                f"OpenRouter streaming error: {str(e)}", self.provider_name, original_error=e
            )
