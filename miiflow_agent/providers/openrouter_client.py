"""OpenRouter client implementation using direct API calls."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import (
    AuthenticationError,
    ModelError,
    ProviderError,
    RateLimitError,
    is_retryable_error,
)
from ..core.exceptions import TimeoutError as MiiflowTimeoutError
from ..core.message import (
    DocumentBlock,
    ImageBlock,
    Message,
    MessageRole,
    TextBlock,
    VideoBlock,
)
from ..core.metrics import TokenCount
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.streaming import StreamChunk
from ..models.openrouter import is_openrouter_model_allowed




class OpenRouterClient(ModelClient):
    """OpenRouter client implementation using direct API calls."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        app_name: Optional[str] = None,
        app_url: Optional[str] = None,
        **kwargs,
    ):
        if not is_openrouter_model_allowed(model):
            raise ModelError(
                f"OpenRouter model '{model}' is not allowed; "
                "allowed families are DeepSeek, GLM, and Grok",
                model,
            )

        if not api_key:
            raise AuthenticationError("OpenRouter API key is required", provider="openrouter")

        super().__init__(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
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

        if isinstance(message.content, list):
            content_parts = []
            for part in message.content:
                if isinstance(part, TextBlock):
                    content_parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageBlock):
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": part.image_url,
                                "detail": part.detail,
                            },
                        }
                    )
                elif isinstance(part, DocumentBlock):
                    filename = part.filename or f"document.{part.document_type}"
                    content_parts.append(
                        {
                            "type": "file",
                            "file": {
                                "filename": filename,
                                "file_data": part.document_url,
                            },
                        }
                    )
                elif isinstance(part, VideoBlock):
                    content_parts.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": part.video_url},
                        }
                    )
                elif isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        content_parts.append({"type": "text", "text": part.get("text", "")})
                    elif part_type == "image_url":
                        image_url = part.get("image_url", {})
                        if isinstance(image_url, str):
                            image_url = {"url": image_url}
                        if isinstance(image_url, dict):
                            content_parts.append({"type": "image_url", "image_url": image_url})
                    elif part_type in {"document", "file"}:
                        file_data = part.get("file")
                        if not isinstance(file_data, dict):
                            file_data = {
                                "filename": part.get("filename", "document.pdf"),
                                "file_data": part.get("document_url", ""),
                            }
                        content_parts.append({"type": "file", "file": file_data})
                    elif part_type == "video_url":
                        video_url = part.get("video_url", {})
                        if isinstance(video_url, str):
                            video_url = {"url": video_url}
                        if isinstance(video_url, dict):
                            content_parts.append({"type": "video_url", "video_url": video_url})
                else:
                    content_parts.append({"type": "text", "text": str(part)})
            msg_dict["content"] = content_parts
        else:
            msg_dict["content"] = message.content

        if message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                function = (
                    tool_call.function
                    if hasattr(tool_call, "function")
                    else tool_call.get("function", {})
                )
                arguments = (
                    function.arguments
                    if hasattr(function, "arguments")
                    else function.get("arguments", "")
                )
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                tool_calls.append(
                    {
                        "id": (
                            tool_call.id if hasattr(tool_call, "id") else tool_call.get("id", "")
                        ),
                        "type": "function",
                        "function": {
                            "name": (
                                function.name
                                if hasattr(function, "name")
                                else function.get("name", "")
                            ),
                            "arguments": arguments,
                        },
                    }
                )
            msg_dict["tool_calls"] = tool_calls

        if message.tool_call_id:
            msg_dict["tool_call_id"] = message.tool_call_id

        if message.name:
            msg_dict["name"] = message.name

        return msg_dict

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to OpenRouter format."""
        return [self._convert_message_to_dict(msg) for msg in messages]

    @staticmethod
    def _parse_usage(usage_data: Any) -> TokenCount:
        """Convert OpenRouter's OpenAI-compatible usage dictionary."""
        if not isinstance(usage_data, dict):
            return TokenCount()

        def as_int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        prompt_tokens = as_int(usage_data.get("prompt_tokens"))
        completion_tokens = as_int(usage_data.get("completion_tokens"))
        prompt_details = usage_data.get("prompt_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        completion_details = usage_data.get("completion_tokens_details")
        if not isinstance(completion_details, dict):
            completion_details = {}

        return TokenCount(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                as_int(usage_data.get("total_tokens")) or prompt_tokens + completion_tokens
            ),
            cache_read_tokens=as_int(prompt_details.get("cached_tokens")),
            reasoning_tokens=as_int(completion_details.get("reasoning_tokens")),
        )

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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
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
                payload["tool_choice"] = kwargs.pop("tool_choice", "auto")

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

            usage = self._parse_usage(data.get("usage"))

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
        except (AuthenticationError, RateLimitError, ModelError, ProviderError):
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
                payload["tool_choice"] = kwargs.pop("tool_choice", "auto")

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
            accumulated_content = ""

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

                        if isinstance(chunk_data.get("error"), dict):
                            error = chunk_data["error"]
                            error_message = str(error.get("message", "OpenRouter stream failed"))
                            try:
                                error_code = int(error.get("code"))
                            except (TypeError, ValueError):
                                error_code = 0
                            if error_code == 401:
                                raise AuthenticationError(error_message, self.provider_name)
                            if error_code == 429:
                                raise RateLimitError(error_message, self.provider_name)
                            if error_code in {400, 404}:
                                raise ModelError(error_message, self.model)
                            raise ProviderError(error_message, self.provider_name)

                        usage_data = chunk_data.get("usage")
                        usage = (
                            self._parse_usage(usage_data) if isinstance(usage_data, dict) else None
                        )

                        if not chunk_data.get("choices"):
                            if usage is not None:
                                yield StreamChunk(
                                    content=accumulated_content,
                                    delta="",
                                    usage=usage,
                                )
                            continue

                        choice = chunk_data["choices"][0]
                        delta = choice.get("delta", {})

                        # Extract content delta
                        content_delta = delta.get("content") or ""
                        accumulated_content += content_delta

                        # Reasoning delta. Every model this client admits is a
                        # reasoning model and every fallback config sets
                        # `reasoning=True`, so the thinking phase is not an edge
                        # case here — it is the start of most turns. Reading
                        # only `content` meant those chunks carried nothing, the
                        # yield guard below skipped them, and the stream sat
                        # silent for the whole phase while the reasoning text
                        # was dropped on the floor. OpenRouter spells it
                        # `reasoning`; some upstreams pass `reasoning_content`
                        # through, so both are accepted.
                        thinking_delta = (
                            delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or ""
                        )

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
                        if (
                            content_delta
                            or thinking_delta
                            or tool_calls
                            or finish_reason
                            or usage is not None
                        ):
                            yield StreamChunk(
                                content=accumulated_content,
                                delta=content_delta,
                                thinking_delta=thinking_delta or None,
                                tool_calls=tool_calls,
                                finish_reason=finish_reason,
                                usage=usage,
                            )

        except httpx.TimeoutException as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except (AuthenticationError, RateLimitError, ModelError, ProviderError):
            raise
        except Exception as e:
            raise ProviderError(
                f"OpenRouter streaming error: {str(e)}",
                self.provider_name,
                original_error=e,
            )
