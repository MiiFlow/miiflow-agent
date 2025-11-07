"""OpenAI provider implementation."""

import asyncio
import copy
from typing import Any, AsyncIterator, Dict, List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.client import ChatResponse, ModelClient
from ..core.exceptions import AuthenticationError, ModelError, ProviderError, RateLimitError
from ..core.exceptions import TimeoutError as MiiflowTimeoutError
from ..core.message import DocumentBlock, ImageBlock, Message, MessageRole, TextBlock
from ..core.metrics import TokenCount, UsageData
from ..core.streaming import StreamChunk


class OpenAIStreaming:
    """Mixin for OpenAI-compatible streaming format normalization."""

    def _reset_stream_state(self):
        """Reset streaming state for a new streaming session."""
        self._accumulated_content = ""
        self._accumulated_tool_calls = {}  # index -> {id, type, function: {name, arguments}}

    def _normalize_stream_chunk(self, chunk: Any) -> StreamChunk:
        """Normalize OpenAI streaming format to unified StreamChunk."""
        delta = ""
        finish_reason = None
        tool_calls = None
        usage = None

        try:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]

                if hasattr(choice, 'delta') and choice.delta:
                    # Handle text content
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        delta = choice.delta.content
                        self._accumulated_content += delta

                    # Handle tool call deltas - convert to standard dict format
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        normalized_tool_calls = []

                        for tool_call_delta in choice.delta.tool_calls:
                            idx = tool_call_delta.index if hasattr(tool_call_delta, 'index') else 0

                            # Initialize accumulator for this index
                            if idx not in self._accumulated_tool_calls:
                                self._accumulated_tool_calls[idx] = {
                                    'id': None,
                                    'type': 'function',
                                    'function': {'name': None, 'arguments': ''}
                                }

                            # Update ID if present
                            if hasattr(tool_call_delta, 'id') and tool_call_delta.id:
                                self._accumulated_tool_calls[idx]['id'] = tool_call_delta.id

                            # Update function name and arguments
                            if hasattr(tool_call_delta, 'function') and tool_call_delta.function:
                                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                    self._accumulated_tool_calls[idx]['function']['name'] = tool_call_delta.function.name

                                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                                    self._accumulated_tool_calls[idx]['function']['arguments'] += tool_call_delta.function.arguments

                            # Emit the current state as a dict
                            normalized_tool_calls.append(self._accumulated_tool_calls[idx].copy())

                        tool_calls = normalized_tool_calls

                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason

            if hasattr(chunk, 'usage') and chunk.usage:
                usage = TokenCount(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                )

        except AttributeError:
            delta = str(chunk) if chunk else ""
            self._accumulated_content += delta

        return StreamChunk(
            content=self._accumulated_content,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls
        )


class OpenAIClient(OpenAIStreaming, ModelClient):
    """OpenAI provider client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key,
            **kwargs
        )
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.provider_name = "openai"

        # Initialize streaming state
        self._accumulated_content = ""
        self._accumulated_tool_calls = {}

    def _clean_json_schema(self, obj: Any) -> None:
        if not isinstance(obj, dict):
            return obj

        if obj.get("type") == "object":
            obj["additionalProperties"] = False

        if "properties" in obj:
            for prop_key, prop_value in obj["properties"].items():
                obj["properties"][prop_key] = self._clean_json_schema(prop_value)

        if "items" in obj:
            obj["items"] = self._clean_json_schema(obj["items"])

        return obj

    def _has_additional_properties(self, obj: Any) -> bool:
        if not isinstance(obj, dict):
            return True

        if obj.get("type") == "object" and "additionalProperties" not in obj:
            return False

        if "properties" in obj:
            for prop_value in obj["properties"].values():
                if not self._has_additional_properties(prop_value):
                    return False

        if "items" in obj:
            if not self._has_additional_properties(obj["items"]):
                return False

        return True

    def _add_additional_properties(self, obj: Any) -> None:
        if not isinstance(obj, dict):
            return

        if obj.get("type") == "object" and "additionalProperties" not in obj:
            obj["additionalProperties"] = False

        if "properties" in obj:
            for prop_value in obj["properties"].values():
                self._add_additional_properties(prop_value)

        if "items" in obj:
            self._add_additional_properties(obj["items"])

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        return OpenAIClient.convert_schema_to_openai_format(schema)

    @staticmethod
    def convert_schema_to_openai_format(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to OpenAI format (static for reuse by compatible providers)."""
        return {"type": "function", "function": schema}

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        return OpenAIClient.convert_message_to_openai_format(message)

    @staticmethod
    def convert_message_to_openai_format(message: Message) -> Dict[str, Any]:
        """Convert universal Message to OpenAI format (static for reuse by compatible providers)."""
        openai_message = {"role": message.role.value}

        if isinstance(message.content, str):
            openai_message["content"] = message.content
        else:
            content_list = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": block.image_url, "detail": block.detail},
                        }
                    )
                elif isinstance(block, DocumentBlock):
                    try:
                        from ..utils.pdf_extractor import extract_pdf_text_simple

                        pdf_text = extract_pdf_text_simple(block.document_url)

                        filename_info = f" [{block.filename}]" if block.filename else ""
                        pdf_content = f"[PDF Document{filename_info}]\n\n{pdf_text}"

                        content_list.append({"type": "text", "text": pdf_content})
                    except Exception as e:
                        filename_info = f" {block.filename}" if block.filename else ""
                        error_content = f"[Error processing PDF{filename_info}: {str(e)}]"
                        content_list.append({"type": "text", "text": error_content})

            openai_message["content"] = content_list

        if message.name:
            openai_message["name"] = message.name
        if message.tool_call_id:
            openai_message["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            openai_message["tool_calls"] = message.tool_calls

        return openai_message

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
        """Send chat completion request to OpenAI."""
        try:
            openai_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": openai_messages,
            }

            if not self.model.startswith("gpt-5"):
                request_params["temperature"] = temperature

            if max_tokens:
                request_params["max_tokens"] = max_tokens
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if json_schema:

                json_schema = self._clean_json_schema(json_schema)

                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": json_schema,
                    },
                }

            response = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params), timeout=self.timeout
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            response_message = Message(
                role=MessageRole.ASSISTANT, content=content, tool_calls=choice.message.tool_calls
            )

            usage = TokenCount(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

            return ChatResponse(
                message=response_message,
                usage=usage,
                model=self.model,
                provider=self.provider_name,
                finish_reason=choice.finish_reason,
                metadata={"response_id": response.id},
            )

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, "retry-after", None)
            raise RateLimitError(
                str(e), self.provider_name, retry_after=retry_after, original_error=e
            )
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}", self.provider_name, original_error=e)

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        try:
            openai_messages = [self.convert_message_to_provider_format(msg) for msg in messages]

            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "stream": True,
            }

            if not self.model.startswith("gpt-5"):
                request_params["temperature"] = temperature

            if max_tokens:
                request_params["max_tokens"] = max_tokens
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if json_schema:
                json_schema = self._clean_json_schema(json_schema)

                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": json_schema,
                    },
                }

            stream = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params), timeout=self.timeout
            )

            # Reset stream state for new streaming session
            self._reset_stream_state()

            async for chunk in stream:
                if not chunk.choices:
                    continue

                normalized_chunk = self._normalize_stream_chunk(chunk)

                # Only yield if there's content or metadata
                if (
                    normalized_chunk.delta
                    or normalized_chunk.tool_calls
                    or normalized_chunk.finish_reason
                ):
                    yield normalized_chunk

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            retry_after = getattr(e.response.headers, "retry-after", None)
            raise RateLimitError(
                str(e), self.provider_name, retry_after=retry_after, original_error=e
            )
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError("Streaming request timed out", self.timeout, original_error=e)
        except Exception as e:
            raise ProviderError(
                f"OpenAI streaming error: {str(e)}", self.provider_name, original_error=e
            )
