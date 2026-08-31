"""OpenAI provider implementation."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import openai
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
from ..core.metrics import TokenCount, UsageData
from ..core.schema_normalizer import SchemaMode, normalize_json_schema
from ..core.stream_normalizer import OpenAIStreamNormalizer
from ..core.streaming import StreamChunk
from ..models.openai import (
    get_token_param_name,
    is_gpt56_model,
    normalize_model_name,
    requires_responses_api,
    supports_native_mcp,
    supports_reasoning_effort,
    supports_json_mode,
    supports_sampling_penalties,
    supports_streaming,
    supports_temperature,
    supports_verbosity,
)

if TYPE_CHECKING:
    from ..core.tools.mcp import NativeMCPServerConfig


def _sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to match OpenAI's pattern: ^[a-zA-Z0-9_-]+$"""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized[:64]  # OpenAI has a 64 char limit for function names


#: Shared with ``OpenAIStreamNormalizer`` — see ``TokenCount.from_openai_usage``.
extract_usage = TokenCount.from_openai_usage
logger = logging.getLogger(__name__)


_CHAT_COMPLETIONS_PASSTHROUGH = frozenset(
    {
        "logit_bias",
        "logprobs",
        "metadata",
        "modalities",
        "moderation",
        "n",
        "parallel_tool_calls",
        "prediction",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "top_logprobs",
        "top_p",
        "user",
        "web_search_options",
    }
)
_RESPONSES_PASSTHROUGH = frozenset(
    {
        "background",
        "context_management",
        "conversation",
        "include",
        "instructions",
        "max_tool_calls",
        "metadata",
        "moderation",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "safety_identifier",
        "service_tier",
        "store",
        "stream_options",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    }
)
_RESPONSES_ONLY_KWARGS = frozenset(
    {
        "background",
        "context_management",
        "conversation",
        "include",
        "instructions",
        "max_tool_calls",
        "previous_response_id",
        "prompt",
        "reasoning_context",
        "reasoning_generate_summary",
        "reasoning_summary",
        "truncation",
    }
)


class OpenAIClient(ModelClient):
    """OpenAI provider client."""

    # Class-level mapping shared across instances for tool name resolution
    # Maps sanitized names back to original names
    _tool_name_mapping: Dict[str, str] = {}

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.fine_tuned_model = kwargs.pop("fine_tuned_model", None) or None
        super().__init__(model=model, api_key=api_key, **kwargs)
        from .sdk_client_cache import get_or_create_sdk_client

        # Reuse the SDK client (and its warm TLS connection pool) across
        # turns — see sdk_client_cache. OpenAIClient itself stays per-turn.
        self.client = get_or_create_sdk_client(
            "openai", api_key, lambda: openai.AsyncOpenAI(api_key=api_key)
        )
        self.provider_name = "openai"

        # Stream normalizer for unified streaming handling
        # Note: Pass class-level mapping for tool name restoration
        self._stream_normalizer = OpenAIStreamNormalizer(
            OpenAIClient._tool_name_mapping
        )

    def _supports_native_mcp(self) -> bool:
        """Check if current model supports native MCP via Responses API."""
        return supports_native_mcp(self.model)

    def _request_model(self, kwargs: Dict[str, Any]) -> str:
        """Resolve a valid request model, including legacy/fine-tune aliases."""
        configured = (
            kwargs.get("fine_tuned_model") or self.fine_tuned_model or self.model
        )
        return normalize_model_name(configured)

    def _should_use_responses_api(
        self,
        *,
        tools: Optional[List[Dict[str, Any]]],
        mcp_servers: Optional[List["NativeMCPServerConfig"]],
        kwargs: Dict[str, Any],
    ) -> bool:
        """Choose the endpoint from model and requested feature capabilities."""
        if kwargs.get("use_responses_api"):
            return True
        if any(kwargs.get(key) is not None for key in _RESPONSES_ONLY_KWARGS):
            return True
        if requires_responses_api(self.model):
            return True
        if kwargs.get("reasoning_mode") == "pro":
            return True
        if mcp_servers and self._supports_native_mcp():
            return True
        # GPT-5.6 tool calling belongs on Responses. It preserves reasoning
        # controls and avoids Chat Completions' tools+reasoning incompatibility.
        if tools and is_gpt56_model(self.model):
            return True
        if (
            tools
            and kwargs.get("reasoning_effort")
            and normalize_model_name(self.model).lower().startswith("gpt-5")
        ):
            return True
        return False

    @staticmethod
    def _copy_allowed_kwargs(
        request_params: Dict[str, Any], kwargs: Dict[str, Any], allowed: frozenset[str]
    ) -> None:
        for key in allowed:
            if key in kwargs and kwargs[key] is not None:
                request_params[key] = kwargs[key]

    def _build_responses_request(
        self,
        *,
        messages: List[Message],
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[List[Dict[str, Any]]],
        json_schema: Optional[Dict[str, Any]],
        mcp_servers: Optional[List["NativeMCPServerConfig"]],
        kwargs: Dict[str, Any],
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Build one Responses request for both streaming and non-streaming.

        Keeping this mapping in one place prevents endpoint parameters from
        drifting between ``achat`` and ``astream_chat``.
        """
        response_tools: List[Dict[str, Any]] = []
        for tool in tools or []:
            if tool.get("type") == "function" and "function" in tool:
                response_tools.append({"type": "function", **tool["function"]})
            else:
                response_tools.append(tool)
        for server in mcp_servers or []:
            response_tools.append(server.to_openai_format())

        request_params: Dict[str, Any] = {
            "model": self._request_model(kwargs),
            "input": self._convert_messages_to_responses_input(messages),
        }
        if stream:
            request_params["stream"] = True
        if response_tools:
            request_params["tools"] = response_tools
            tool_choice = kwargs.get("tool_choice", "auto")
            if (
                isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"
                and isinstance(tool_choice.get("function"), dict)
            ):
                tool_choice = {
                    "type": "function",
                    "name": tool_choice["function"].get("name", ""),
                }
            request_params["tool_choice"] = tool_choice
        if max_tokens is not None:
            request_params["max_output_tokens"] = max_tokens
        if supports_temperature(self.model):
            request_params["temperature"] = temperature

        reasoning: Dict[str, Any] = {}
        reasoning_keys = {
            "reasoning_effort": "effort",
            "reasoning_mode": "mode",
            "reasoning_context": "context",
            "reasoning_summary": "summary",
            "reasoning_generate_summary": "generate_summary",
        }
        for source, target in reasoning_keys.items():
            if kwargs.get(source) is not None:
                reasoning[target] = kwargs[source]
        if self.model.lower() == "gpt-5.6-sol-pro":
            reasoning["mode"] = "pro"
        if reasoning and supports_reasoning_effort(self.model):
            request_params["reasoning"] = reasoning

        text_config: Dict[str, Any] = {}
        if json_schema:
            text_config["format"] = {
                "type": "json_schema",
                "name": "response_schema",
                "schema": normalize_json_schema(json_schema, SchemaMode.STRICT),
                "strict": True,
            }
        if supports_verbosity(self.model) and kwargs.get("verbosity"):
            text_config["verbosity"] = kwargs["verbosity"]
        if text_config:
            request_params["text"] = text_config

        self._copy_allowed_kwargs(request_params, kwargs, _RESPONSES_PASSTHROUGH)
        for unsupported in ("frequency_penalty", "presence_penalty"):
            if unsupported in kwargs and kwargs[unsupported] not in (None, 0, 0.0):
                logger.warning(
                    "Ignoring %s for %s because the Responses API does not accept it",
                    unsupported,
                    self.model,
                )
        return request_params

    async def _create_response(self, request_params: Dict[str, Any]) -> Any:
        """Create a Response and wait for background work when requested."""
        response = await asyncio.wait_for(
            self.client.responses.create(**request_params), timeout=self.timeout
        )
        if not request_params.get("background"):
            return response

        terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        while getattr(response, "status", None) not in terminal_statuses:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.sleep(min(0.5, remaining))
            response = await asyncio.wait_for(
                self.client.responses.retrieve(response.id), timeout=remaining
            )

        if getattr(response, "status", None) in {"failed", "cancelled"}:
            error = getattr(response, "error", None)
            message = getattr(error, "message", None) or str(error or response.status)
            raise ModelError(f"OpenAI background response {message}", self.model)
        return response

    def _apply_reasoning_effort(
        self,
        request_params: Dict[str, Any],
        kwargs: Dict[str, Any],
        *,
        has_tools: bool,
    ) -> None:
        """Forward a configured ``reasoning_effort`` onto the Chat Completions
        request, honoring OpenAI's two constraints:

        1. Only reasoning / GPT-5 models accept the parameter at all — sending
           it to a standard model is a 400.
        2. OpenAI rejects ``reasoning_effort`` when function ``tools`` are also
           present on ``/v1/chat/completions`` ("Function tools with
           reasoning_effort are not supported for <model> in
           /v1/chat/completions"). On tool-calling turns we therefore drop it
           and let the model fall back to its default effort rather than 400.
        """
        reasoning_effort = kwargs.get("reasoning_effort")
        if not reasoning_effort:
            return
        if not supports_reasoning_effort(self.model):
            return
        if has_tools:
            logger.debug(
                "Dropping reasoning_effort=%s for %s: not supported alongside "
                "function tools on /v1/chat/completions",
                reasoning_effort,
                self.model,
            )
            return
        request_params["reasoning_effort"] = reasoning_effort

    @staticmethod
    def _is_function_tools_reasoning_effort_error(
        error: openai.BadRequestError,
    ) -> bool:
        """Recognize OpenAI's Chat Completions tools/effort incompatibility.

        Keep this deliberately narrower than a generic ``reasoning_effort``
        check: other 400s are caller errors and must still fail immediately.
        """
        message = str(error).lower()
        return (
            "function tools with reasoning_effort are not supported" in message
            and "/v1/chat/completions" in message
        )

    async def _create_chat_completion(self, request_params: Dict[str, Any]) -> Any:
        """Create a Chat Completion with one compatibility fallback.

        ``_apply_reasoning_effort`` normally prevents the unsupported
        function-tools combination before it reaches OpenAI. If request
        shaping outside that guard ever reintroduces it, the provider rejects
        the request before doing any work, so it is safe to retry once with
        the optional effort override removed.
        """
        try:
            return await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=self.timeout,
            )
        except openai.BadRequestError as error:
            if (
                "reasoning_effort" not in request_params
                or not self._is_function_tools_reasoning_effort_error(error)
            ):
                raise

            fallback_params = dict(request_params)
            fallback_params.pop("reasoning_effort")
            return await asyncio.wait_for(
                self.client.chat.completions.create(**fallback_params),
                timeout=self.timeout,
            )

    def convert_schema_to_provider_format(
        self, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert universal schema to OpenAI format with name sanitization."""
        original_name = schema["name"]
        sanitized_name = _sanitize_tool_name(original_name)

        # Track mapping for restoring original names from tool call responses
        if sanitized_name != original_name:
            OpenAIClient._tool_name_mapping[sanitized_name] = original_name

        sanitized_schema = {**schema, "name": sanitized_name}
        return {"type": "function", "function": sanitized_schema}

    @staticmethod
    def convert_schema_to_openai_format(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to OpenAI format with name sanitization.

        Note: For proper name mapping support, use convert_schema_to_provider_format instead.
        """
        original_name = schema["name"]
        sanitized_name = _sanitize_tool_name(original_name)

        # Track mapping for restoring original names from tool call responses
        if sanitized_name != original_name:
            OpenAIClient._tool_name_mapping[sanitized_name] = original_name

        sanitized_schema = {**schema, "name": sanitized_name}
        return {"type": "function", "function": sanitized_schema}

    def convert_message_to_provider_format(self, message: Message) -> Dict[str, Any]:
        return OpenAIClient.convert_message_to_openai_format(message)

    @staticmethod
    def convert_message_to_openai_format(message: Message) -> Dict[str, Any]:
        """Convert universal Message to OpenAI format (static for reuse by compatible providers)."""
        openai_message = {"role": message.role.value}

        # OpenAI's Chat Completions API requires tool-role messages to have
        # string content — it rejects an array of blocks. When a tool returns
        # multimodal content (e.g. LlmBlockInjection), collapse to a text
        # summary that references the media URLs. OpenAI users miss the visual
        # analysis but get a coherent response instead of a 400 error.
        if (
            message.role == MessageRole.TOOL
            and message.tool_call_id
            and isinstance(message.content, list)
        ):
            text_parts: List[str] = []
            dropped_images: List[str] = []
            dropped_videos: List[str] = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        text_parts.append(block.text)
                elif isinstance(block, ImageBlock):
                    dropped_images.append(block.image_url)
                elif isinstance(block, VideoBlock):
                    dropped_videos.append(block.video_url)
            if dropped_images:
                text_parts.append(
                    "[NOTE: OpenAI tool messages cannot carry image content. "
                    f"{len(dropped_images)} image(s) referenced by URL: {dropped_images}]"
                )
            if dropped_videos:
                text_parts.append(
                    f"[NOTE: OpenAI cannot view videos in tool messages. "
                    f"{len(dropped_videos)} video(s) referenced: {dropped_videos}]"
                )
            openai_message["content"] = (
                "\n".join(text_parts) if text_parts else "[empty result]"
            )
            openai_message["tool_call_id"] = message.tool_call_id
            if message.name:
                openai_message["name"] = message.name
            return openai_message

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
                            "image_url": {
                                "url": block.image_url,
                                "detail": block.detail,
                            },
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
                            import httpx

                            resp = httpx.get(
                                block.document_url, timeout=30, follow_redirects=True
                            )
                            resp.raise_for_status()
                            text = resp.content.decode("utf-8", errors="replace")
                            doc_content = f"[Document{filename_info}]\n\n{text}"
                        content_list.append({"type": "text", "text": doc_content})
                    except Exception as e:
                        filename_info = f" {block.filename}" if block.filename else ""
                        error_content = (
                            f"[Error processing document{filename_info}: {str(e)}]"
                        )
                        content_list.append({"type": "text", "text": error_content})

            openai_message["content"] = content_list

        if message.name:
            openai_message["name"] = message.name
        if message.tool_call_id:
            openai_message["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            # Sanitize tool names in tool_calls for OpenAI compatibility
            sanitized_tool_calls = []
            for tc in message.tool_calls:
                # Provider-executed MCP calls (`mcp_call` output items on the
                # Responses API) have no representation in Chat Completions:
                # the shape only knows `type: "function"`, and nothing ran on
                # our side so there is no tool message to answer the call —
                # which the API rejects. Drop them. Emitting them as ordinary
                # function calls would be worse than losing the transcript
                # entry: it tells the model an MCP tool is client-side, so it
                # reissues it locally and gets "tool not found" from a registry
                # that never holds native-MCP tools.
                if isinstance(tc, dict) and tc.get("type") == "mcp_function":
                    continue
                sanitized_tc = copy.deepcopy(tc) if isinstance(tc, dict) else tc
                if isinstance(sanitized_tc, dict) and "function" in sanitized_tc:
                    original_name = sanitized_tc["function"].get("name", "")
                    sanitized_tc["function"]["name"] = _sanitize_tool_name(
                        original_name
                    )
                    # Normalize: Anthropic-style dict arguments → JSON string for OpenAI API
                    args = sanitized_tc["function"].get("arguments")
                    if isinstance(args, dict):
                        sanitized_tc["function"]["arguments"] = json.dumps(args)
                sanitized_tool_calls.append(sanitized_tc)
            if sanitized_tool_calls:
                openai_message["tool_calls"] = sanitized_tool_calls

        return openai_message

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
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to OpenAI.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            tools: Tool schemas for function calling
            json_schema: JSON schema for structured output
            mcp_servers: Optional list of MCP server configs for native MCP support.
                        When provided, uses Responses API instead of Chat Completions.
            **kwargs: Additional parameters passed to the API

        Returns:
            ChatResponse with assistant message and metadata
        """
        if json_schema and not supports_json_mode(self.model):
            raise ModelError(
                f"Structured outputs are not supported by {self.model}", self.model
            )
        if self._should_use_responses_api(
            tools=tools, mcp_servers=mcp_servers, kwargs=kwargs
        ):
            return await self._achat_responses_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                json_schema=json_schema,
                mcp_servers=mcp_servers,
                **kwargs,
            )

        # Use standard Chat Completions API
        try:
            # convert_message_to_provider_format may do blocking I/O for
            # DocumentBlocks (httpx.get, PDF extraction) — offload to a worker
            # thread so the event loop stays free under ASGI servers.
            openai_messages = await asyncio.to_thread(
                lambda: [
                    self.convert_message_to_provider_format(msg) for msg in messages
                ]
            )

            request_params: Dict[str, Any] = {
                "model": self._request_model(kwargs),
                "messages": openai_messages,
            }

            if supports_temperature(self.model):
                request_params["temperature"] = temperature

            if max_tokens is not None:
                request_params[get_token_param_name(self.model)] = max_tokens
            else:
                # Provide sensible default when not specified
                request_params[get_token_param_name(self.model)] = 16384
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

            if json_schema:
                normalized_schema = normalize_json_schema(
                    json_schema, SchemaMode.STRICT
                )
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": normalized_schema,
                    },
                }

            if supports_sampling_penalties(self.model):
                for key in ("frequency_penalty", "presence_penalty"):
                    if key in kwargs and kwargs[key] is not None:
                        request_params[key] = kwargs[key]
            if supports_verbosity(self.model) and kwargs.get("verbosity"):
                request_params["verbosity"] = kwargs["verbosity"]

            self._copy_allowed_kwargs(
                request_params, kwargs, _CHAT_COMPLETIONS_PASSTHROUGH
            )
            self._apply_reasoning_effort(request_params, kwargs, has_tools=bool(tools))

            response = await self._create_chat_completion(request_params)

            choice = response.choices[0]
            content = choice.message.content or ""

            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=choice.message.tool_calls,
            )

            usage = extract_usage(getattr(response, "usage", None))

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
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError(
                "Request timed out", self.timeout, original_error=e
            )
        except Exception as e:
            raise ProviderError(
                f"OpenAI API error: {str(e)}", self.provider_name, original_error=e
            )

    async def _achat_responses_api(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Chat using OpenAI Responses API (supports native MCP).

        The Responses API is a different API surface from Chat Completions,
        designed for agentic workflows with native MCP server support.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Regular function tools
            json_schema: Optional structured-output schema
            mcp_servers: MCP server configurations for native MCP

        Returns:
            ChatResponse with assistant message and metadata
        """
        try:
            request_params = self._build_responses_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                json_schema=json_schema,
                mcp_servers=mcp_servers,
                kwargs=kwargs,
            )
            response = await self._create_response(request_params)

            return self._parse_responses_api_response(response)

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError(
                "Request timed out", self.timeout, original_error=e
            )
        except ModelError:
            raise
        except Exception as e:
            raise ProviderError(
                f"OpenAI Responses API error: {str(e)}",
                self.provider_name,
                original_error=e,
            )

    @staticmethod
    def _convert_blocks_to_responses_content(
        blocks: List[Any],
    ) -> List[Dict[str, Any]]:
        """Convert universal multimodal blocks to Responses content items."""
        content_parts: List[Dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, TextBlock):
                content_parts.append({"type": "input_text", "text": block.text})
            elif isinstance(block, ImageBlock):
                content_parts.append(
                    {
                        "type": "input_image",
                        "image_url": block.image_url,
                        "detail": block.detail,
                    }
                )
            elif isinstance(block, DocumentBlock):
                # `filename` is NOT a free-standing label on an input_file --
                # it is the name that goes WITH inline bytes, and the Responses
                # API rejects it beside any other file source:
                #
                #   400 mutually_exclusive_parameters -- "Ensure you are only
                #   providing one of: 'file_id' or 'filename'."
                #
                # (Verified against the live API on 2026-08-31: `file_url` +
                # `filename` 400s with exactly that message; `file_url` alone
                # succeeds. The message names `file_id` whatever the other
                # source actually was, which is why it reads as unrelated to
                # the request we sent.) So the filename rides with `file_data`
                # and only with `file_data`.
                #
                # For a URL it becomes a text label instead of being dropped:
                # the URL is a storage key, not the name the user knows the
                # file by (`enhanced_response_generator` fills `filename` from
                # `attachment.original_filename` precisely because the two
                # differ), and the Anthropic and Gemini paths already label
                # documents this way.
                file_part: Dict[str, Any] = {"type": "input_file"}
                if block.document_url.startswith("data:"):
                    file_part["file_data"] = block.document_url
                    if block.filename:
                        file_part["filename"] = block.filename
                else:
                    file_part["file_url"] = block.document_url
                    if block.filename:
                        content_parts.append(
                            {
                                "type": "input_text",
                                "text": f"[Document: {block.filename}]",
                            }
                        )
                content_parts.append(file_part)
            elif isinstance(block, VideoBlock):
                content_parts.append(
                    {
                        "type": "input_text",
                        "text": (
                            "[OpenAI cannot view this video; reference URL: "
                            f"{block.video_url}]"
                        ),
                    }
                )
        return content_parts

    def _convert_messages_to_responses_input(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Convert messages to Responses API input format.

        The Responses API uses a different input structure than Chat Completions.
        It expects an array of input items rather than messages.
        """
        input_items: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System messages become system items
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                input_items.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": content,
                    }
                )
            elif msg.role == MessageRole.USER:
                # User messages
                if isinstance(msg.content, str):
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": msg.content,
                        }
                    )
                else:
                    content_parts = self._convert_blocks_to_responses_content(
                        msg.content
                    )
                    input_items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": content_parts,
                        }
                    )
            elif msg.role == MessageRole.ASSISTANT:
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if content:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        }
                    )
                # Responses represents assistant function calls as peer input
                # items, not as a ``tool_calls`` property on the message.
                for tool_call in msg.tool_calls or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    if not isinstance(function, dict):
                        function = {
                            "name": getattr(function, "name", ""),
                            "arguments": getattr(function, "arguments", "{}"),
                        }
                    call_id = tool_call.get("call_id") or tool_call.get("id") or ""
                    arguments = function.get("arguments", "{}")
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": function.get("name", ""),
                            "arguments": arguments,
                        }
                    )
            elif msg.role == MessageRole.TOOL:
                # Tool results
                output: Any
                if isinstance(msg.content, str):
                    output = msg.content
                else:
                    output = self._convert_blocks_to_responses_content(msg.content)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": output,
                    }
                )

        return input_items

    def _parse_responses_api_response(self, response: Any) -> ChatResponse:
        """Parse Responses API output format.

        The Responses API returns output items instead of choices/messages.
        """
        content = ""
        tool_calls = []

        # Extract content and tool calls from output items
        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None)

            if item_type == "message":
                # Text message content
                for part in getattr(item, "content", []):
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        content += getattr(part, "text", "")
                    elif part_type == "text":
                        content += getattr(part, "text", "")

            elif item_type == "function_call":
                # Regular function call
                sanitized_name = getattr(item, "name", "")
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", ""),
                        "type": "function",
                        "function": {
                            "name": self._tool_name_mapping.get(
                                sanitized_name, sanitized_name
                            ),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                    }
                )

            elif item_type == "mcp_call":
                # Native MCP call result
                tool_calls.append(
                    {
                        "id": getattr(item, "id", ""),
                        "type": "mcp_function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                        "server_label": getattr(item, "server_label", None),
                    }
                )

        # Extract usage
        usage = extract_usage(getattr(response, "usage", None))

        response_message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

        return ChatResponse(
            message=response_message,
            usage=usage,
            model=self.model,
            provider=self.provider_name,
            finish_reason=getattr(response, "status", "stop"),
            metadata={"response_id": getattr(response, "id", "")},
        )

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from OpenAI.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            tools: Tool schemas for function calling
            json_schema: JSON schema for structured output
            mcp_servers: Optional list of MCP server configs for native MCP support.
                        When provided, uses Responses API streaming instead of Chat Completions.
            **kwargs: Additional parameters passed to the API

        Yields:
            StreamChunk with delta content and metadata
        """
        if json_schema and not supports_json_mode(self.model):
            raise ModelError(
                f"Structured outputs are not supported by {self.model}", self.model
            )
        use_responses_api = self._should_use_responses_api(
            tools=tools, mcp_servers=mcp_servers, kwargs=kwargs
        )
        if use_responses_api and (
            not supports_streaming(self.model) or kwargs.get("background")
        ):
            response = await self._achat_responses_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                json_schema=json_schema,
                mcp_servers=mcp_servers,
                **kwargs,
            )
            yield StreamChunk(
                content=response.message.content,
                delta=response.message.content,
                finish_reason=response.finish_reason,
                usage=response.usage,
                tool_calls=response.message.tool_calls,
            )
            return

        if use_responses_api:
            async for chunk in self._astream_chat_responses_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                json_schema=json_schema,
                mcp_servers=mcp_servers,
                **kwargs,
            ):
                yield chunk
            return

        # Use standard Chat Completions API streaming
        try:
            # Offload sync DocumentBlock I/O / PDF extraction off the event loop.
            openai_messages = await asyncio.to_thread(
                lambda: [
                    self.convert_message_to_provider_format(msg) for msg in messages
                ]
            )

            request_params: Dict[str, Any] = {
                "model": self._request_model(kwargs),
                "messages": openai_messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            if supports_temperature(self.model):
                request_params["temperature"] = temperature

            if max_tokens is not None:
                request_params[get_token_param_name(self.model)] = max_tokens
            else:
                # Provide sensible default when not specified
                request_params[get_token_param_name(self.model)] = 16384
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

            if json_schema:
                normalized_schema = normalize_json_schema(
                    json_schema, SchemaMode.STRICT
                )
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_schema",
                        "schema": normalized_schema,
                    },
                }

            if supports_sampling_penalties(self.model):
                for key in ("frequency_penalty", "presence_penalty"):
                    if key in kwargs and kwargs[key] is not None:
                        request_params[key] = kwargs[key]
            if supports_verbosity(self.model) and kwargs.get("verbosity"):
                request_params["verbosity"] = kwargs["verbosity"]

            self._copy_allowed_kwargs(
                request_params, kwargs, _CHAT_COMPLETIONS_PASSTHROUGH
            )
            self._apply_reasoning_effort(request_params, kwargs, has_tools=bool(tools))

            stream = await self._create_chat_completion(request_params)

            # Reset stream state for new streaming session
            self._stream_normalizer.reset_state()

            async for chunk in stream:
                # Handle final usage-only chunk (has usage but empty choices)
                # This is sent when stream_options.include_usage=True
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = extract_usage(chunk.usage)
                        yield StreamChunk(
                            content=self._stream_normalizer._state.accumulated_content,
                            delta="",
                            finish_reason=None,
                            usage=usage,
                        )
                    continue

                normalized_chunk = self._stream_normalizer.normalize_chunk(chunk)

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
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError(
                "Streaming request timed out", self.timeout, original_error=e
            )
        except Exception as e:
            raise ProviderError(
                f"OpenAI streaming error: {str(e)}",
                self.provider_name,
                original_error=e,
            )

    async def _astream_chat_responses_api(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        mcp_servers: Optional[List["NativeMCPServerConfig"]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat using OpenAI Responses API (supports native MCP).

        The Responses API supports streaming with a different event format.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            tools: Regular function tools
            json_schema: Optional structured-output schema
            mcp_servers: MCP server configurations for native MCP

        Yields:
            StreamChunk with delta content and metadata
        """
        try:
            request_params = self._build_responses_request(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                json_schema=json_schema,
                mcp_servers=mcp_servers,
                kwargs=kwargs,
                stream=True,
            )

            # Use Responses API streaming
            stream = await asyncio.wait_for(
                self.client.responses.create(**request_params), timeout=self.timeout
            )

            accumulated_content = ""

            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    # Text content delta
                    delta = getattr(event, "delta", "")
                    accumulated_content += delta
                    yield StreamChunk(
                        content=accumulated_content,
                        delta=delta,
                        finish_reason=None,
                    )

                elif event_type == "response.function_call_arguments.done":
                    # Function call complete
                    sanitized_name = getattr(event, "name", "")
                    yield StreamChunk(
                        content=accumulated_content,
                        delta="",
                        finish_reason=None,
                        tool_calls=[
                            {
                                "id": getattr(event, "call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": self._tool_name_mapping.get(
                                        sanitized_name, sanitized_name
                                    ),
                                    "arguments": getattr(event, "arguments", "{}"),
                                },
                            }
                        ],
                    )

                elif event_type == "response.mcp_call.done":
                    # MCP call complete
                    yield StreamChunk(
                        content=accumulated_content,
                        delta="",
                        finish_reason=None,
                        tool_calls=[
                            {
                                "id": getattr(event, "id", ""),
                                "type": "mcp_function",
                                "function": {
                                    "name": getattr(event, "name", ""),
                                    "arguments": getattr(event, "arguments", "{}"),
                                },
                                "server_label": getattr(event, "server_label", None),
                            }
                        ],
                    )

                elif event_type == "response.done":
                    # Response complete
                    response_data = getattr(event, "response", None)
                    usage = None
                    if response_data:
                        usage_data = getattr(response_data, "usage", None)
                        if usage_data:
                            usage = extract_usage(usage_data)

                    yield StreamChunk(
                        content=accumulated_content,
                        delta="",
                        finish_reason="stop",
                        usage=usage,
                    )

        except openai.AuthenticationError as e:
            raise AuthenticationError(str(e), self.provider_name, original_error=e)
        except openai.RateLimitError as e:
            raise RateLimitError(str(e), self.provider_name, original_error=e)
        except openai.BadRequestError as e:
            raise ModelError(str(e), self.model, original_error=e)
        except asyncio.TimeoutError as e:
            raise MiiflowTimeoutError(
                "Streaming request timed out", self.timeout, original_error=e
            )
        except Exception as e:
            raise ProviderError(
                f"OpenAI Responses API streaming error: {str(e)}",
                self.provider_name,
                original_error=e,
            )
