"""Core LLM client interface and base implementations."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)

from .exceptions import MiiflowLLMError, TimeoutError
from .message import Message, MessageRole
from .metrics import MetricsCollector, TokenCount, UsageData
from .streaming import StreamChunk
from .tools import FunctionTool, ToolRegistry

if TYPE_CHECKING:
    from .callbacks import CallbackRegistry


@dataclass
class ChatResponse:
    """Response from a chat completion request."""

    message: Message
    usage: TokenCount
    model: str
    provider: str
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None


@runtime_checkable
class ModelClientProtocol(Protocol):
    """Protocol defining the interface for LLM provider clients."""

    model: str
    api_key: Optional[str]
    timeout: float
    max_retries: int
    metrics_collector: MetricsCollector
    provider_name: str

    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send async chat completion request."""
        ...

    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send async streaming chat completion request."""
        ...

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send sync chat completion request."""
        ...

    def stream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Send sync streaming chat completion request."""
        ...


class ModelClient(ABC):
    """Abstract base class for LLM provider clients."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        metrics_collector: Optional[MetricsCollector] = None,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.provider_name = self.__class__.__name__.replace("Client", "").lower()

    def convert_schema_to_provider_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert universal schema to provider-specific format."""
        # Default implementation - subclasses should override for provider-specific formats
        return schema

    def supports_vision(self) -> bool:
        """Check if the model supports vision/image inputs.

        Default implementation assumes all models support vision.
        This method exists for future compatibility if vision checks are needed.
        """
        return True

    @abstractmethod
    async def achat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send async chat completion request."""
        pass

    @abstractmethod
    async def astream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send async streaming chat completion request."""
        pass

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send sync chat completion request."""
        return asyncio.run(self.achat(messages, temperature, max_tokens, tools, **kwargs))

    def stream_chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Send sync streaming chat completion request."""

        async def _async_stream():
            async for chunk in self.astream_chat(
                messages, temperature, max_tokens, tools, **kwargs
            ):
                yield chunk

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_gen = _async_stream()
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def _record_metrics(self, usage: UsageData) -> None:
        """Record usage metrics."""
        if self.metrics_collector:
            self.metrics_collector.record_usage(usage)


class LLMClient:
    """Main LLM client with provider management."""

    def __init__(
        self,
        client: ModelClient,
        metrics_collector: Optional[MetricsCollector] = None,
        tool_registry: Optional[ToolRegistry] = None,
        callback_registry: Optional["CallbackRegistry"] = None,
    ):
        self.client = client
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.client.metrics_collector = self.metrics_collector
        self.tool_registry = tool_registry or ToolRegistry()

        # Callback support - uses instance registry or falls back to global
        self._callback_registry = callback_registry

        # Initialize unified streaming client
        self._unified_streaming_client = None

    def _supports_native_mcp(self) -> bool:
        """Check if the current provider supports native MCP.

        Native MCP allows the provider to connect directly to MCP servers
        and execute tools server-side, rather than client-side handling.

        Returns:
            True if provider supports native MCP
        """
        # Check if provider client has _supports_native_mcp method
        if hasattr(self.client, "_supports_native_mcp"):
            return self.client._supports_native_mcp()

        # Check provider name
        provider = getattr(self.client, "provider_name", "").lower()
        return provider in ("anthropic", "openai")

    @property
    def callback_registry(self) -> "CallbackRegistry":
        """Get the callback registry (instance-level or global)."""
        if self._callback_registry:
            return self._callback_registry
        from .callbacks import get_global_registry

        return get_global_registry()

    async def _emit_callback(self, event: "CallbackEvent") -> None:
        """Emit a callback event."""
        await self.callback_registry.emit(event)

    @classmethod
    def create(
        cls, provider: str, model: str, api_key: Optional[str] = None, **kwargs
    ) -> "LLMClient":
        """Create client for specified provider."""
        from ..providers import get_provider_client

        # Bedrock uses AWS credentials instead of API key
        if provider.lower() == "bedrock":
            # Skip API key check for Bedrock - it uses AWS credentials
            client = get_provider_client(provider=provider, model=model, api_key=None, **kwargs)
            return cls(client)

        if api_key is None:
            from ..utils.env import get_api_key, load_env_file

            load_env_file()
            api_key = get_api_key(provider)
            if api_key is None and provider.lower() != "ollama":
                raise ValueError(
                    f"No API key found for {provider}. Set {provider.upper()}_API_KEY in .env or pass api_key parameter."
                )

        client = get_provider_client(provider=provider, model=model, api_key=api_key, **kwargs)

        return cls(client)

    # Async methods
    async def achat(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        tools: Optional[List[FunctionTool]] = None,
        _formatted_tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send async chat completion request.

        Args:
            messages: List of messages to send.
            tools: List of FunctionTool objects to make available.
            _formatted_tools: Pre-formatted tool schemas (internal use by AgentToolExecutor).
                             If provided, skips tool formatting.
            **kwargs: Additional arguments passed to the provider.

        Returns:
            ChatResponse with the model's response.
        """
        from .callback_context import get_callback_context
        from .callbacks import CallbackEvent, CallbackEventType

        normalized_messages = self._normalize_messages(messages)

        # Use pre-formatted tools if provided, otherwise format from tools
        formatted_tools = _formatted_tools
        if formatted_tools is None:
            if tools:
                for tool in tools:
                    self.tool_registry.register(tool)
                tool_names = [
                    (
                        getattr(tool, "_function_tool", tool).name
                        if hasattr(getattr(tool, "_function_tool", tool), "name")
                        else getattr(tool, "__name__", str(tool))
                    )
                    for tool in tools
                ]
                all_schemas = self.tool_registry.get_schemas(self.client.provider_name, self.client)
                formatted_tools = [s for s in all_schemas if self._extract_tool_name(s) in tool_names]
            elif self.tool_registry.tools:
                formatted_tools = self.tool_registry.get_schemas(self.client.provider_name, self.client)

        # Check for native MCP servers and pass to provider if supported
        if self.tool_registry.has_native_mcp_servers() and self._supports_native_mcp():
            kwargs["mcp_servers"] = self.tool_registry.get_native_mcp_configs()

        # Get callback context
        ctx = get_callback_context()

        start_time = time.time()
        try:
            response = await self.client.achat(normalized_messages, tools=formatted_tools, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Record successful usage
            self._record_usage(
                normalized_messages, response.usage, time.time() - start_time, success=True
            )

            # Emit POST_CALL callback
            post_event = CallbackEvent(
                event_type=CallbackEventType.POST_CALL,
                provider=self.client.provider_name,
                model=self.client.model,
                tokens=response.usage,
                latency_ms=latency_ms,
                context=ctx,
                success=True,
            )
            await self._emit_callback(post_event)

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Record failed usage
            self._record_usage(
                normalized_messages, TokenCount(), time.time() - start_time, success=False
            )

            # Emit ON_ERROR callback
            error_event = CallbackEvent(
                event_type=CallbackEventType.ON_ERROR,
                provider=self.client.provider_name,
                model=self.client.model,
                error=e,
                error_type=type(e).__name__,
                latency_ms=latency_ms,
                context=ctx,
                success=False,
            )
            await self._emit_callback(error_event)

            raise

    # Sync wrapper methods
    def chat(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send sync chat completion request."""
        return asyncio.run(self.achat(messages, tools=tools, **kwargs))

    async def astream_chat(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        tools: Optional[List[FunctionTool]] = None,
        _formatted_tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send async streaming chat completion request.

        Args:
            messages: List of messages to send.
            tools: List of FunctionTool objects to make available.
            _formatted_tools: Pre-formatted tool schemas (internal use by AgentToolExecutor).
                             If provided, skips tool formatting.
            **kwargs: Additional arguments passed to the provider.

        Yields:
            StreamChunk objects with content deltas and usage information.
        """
        from .callback_context import get_callback_context
        from .callbacks import CallbackEvent, CallbackEventType

        normalized_messages = self._normalize_messages(messages)

        # Use pre-formatted tools if provided, otherwise format from tools
        formatted_tools = _formatted_tools
        if formatted_tools is None:
            if tools:
                for tool in tools:
                    self.tool_registry.register(tool)
                tool_names = [
                    (
                        getattr(tool, "_function_tool", tool).name
                        if hasattr(getattr(tool, "_function_tool", tool), "name")
                        else getattr(tool, "__name__", str(tool))
                    )
                    for tool in tools
                ]
                all_schemas = self.tool_registry.get_schemas(self.client.provider_name, self.client)
                formatted_tools = [s for s in all_schemas if self._extract_tool_name(s) in tool_names]
            elif self.tool_registry.tools:
                formatted_tools = self.tool_registry.get_schemas(self.client.provider_name, self.client)

        # Check for native MCP servers and pass to provider if supported
        if self.tool_registry.has_native_mcp_servers() and self._supports_native_mcp():
            kwargs["mcp_servers"] = self.tool_registry.get_native_mcp_configs()

        # Get callback context
        ctx = get_callback_context()

        start_time = time.time()
        total_tokens = TokenCount()
        callback_emitted = False
        error_occurred = None

        try:
            async for chunk in self.client.astream_chat(
                normalized_messages, tools=formatted_tools, **kwargs
            ):
                if chunk.usage:
                    total_tokens += chunk.usage
                yield chunk

        except Exception as e:
            error_occurred = e
            raise

        finally:
            # Always emit callback when generator closes (success, break, or error)
            if not callback_emitted:
                latency_ms = (time.time() - start_time) * 1000

                if error_occurred:
                    # Record failed streaming usage
                    self._record_usage(
                        normalized_messages, total_tokens, time.time() - start_time, success=False
                    )

                    # Emit ON_ERROR callback
                    error_event = CallbackEvent(
                        event_type=CallbackEventType.ON_ERROR,
                        provider=self.client.provider_name,
                        model=self.client.model,
                        tokens=total_tokens,
                        error=error_occurred,
                        error_type=type(error_occurred).__name__,
                        latency_ms=latency_ms,
                        context=ctx,
                        success=False,
                    )
                    # Use sync emit in finally block since we can't await
                    self.callback_registry.emit_sync(error_event)
                else:
                    # Record successful streaming usage
                    self._record_usage(
                        normalized_messages, total_tokens, time.time() - start_time, success=True
                    )

                    # Emit POST_CALL callback
                    post_event = CallbackEvent(
                        event_type=CallbackEventType.POST_CALL,
                        provider=self.client.provider_name,
                        model=self.client.model,
                        tokens=total_tokens,
                        latency_ms=latency_ms,
                        context=ctx,
                        success=True,
                    )
                    # Use sync emit in finally block since we can't await
                    self.callback_registry.emit_sync(post_event)

                callback_emitted = True

    def stream_chat(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Send sync streaming chat completion request."""

        async def _async_stream():
            async for chunk in self.astream_chat(messages, tools=tools, **kwargs):
                yield chunk

        # Convert async generator to sync generator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_gen = _async_stream()
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def _normalize_messages(
        self, messages: Union[List[Dict[str, Any]], List[Message]]
    ) -> List[Message]:
        """Normalize message format."""
        if not messages:
            return []

        if isinstance(messages[0], dict):
            return [
                Message(
                    role=MessageRole(msg["role"]),
                    content=msg["content"],
                    name=msg.get("name"),
                    tool_call_id=msg.get("tool_call_id"),
                    tool_calls=msg.get("tool_calls"),
                )
                for msg in messages
            ]

        return messages

    def _record_usage(
        self, messages: List[Message], tokens: TokenCount, latency: float, success: bool
    ) -> None:
        """Record usage metrics."""
        usage = UsageData(
            provider=self.client.provider_name,
            model=self.client.model,
            operation="chat",
            tokens=tokens,
            latency_ms=latency * 1000,
            success=success,
            metadata={
                "message_count": len(messages),
                "has_tools": any(msg.tool_calls for msg in messages),
            },
        )

        self.metrics_collector.record_usage(usage)

    def _extract_tool_name(self, schema: Dict[str, Any]) -> str:
        """Extract tool name from provider-specific schema."""
        if "function" in schema:
            # OpenAI format
            return schema["function"]["name"]
        elif "name" in schema:
            # Anthropic/Gemini format
            return schema["name"]
        else:
            raise ValueError(f"Unable to extract tool name from schema: {schema}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics_collector.get_metrics()

    async def stream_with_schema(
        self,
        messages: Union[List[Dict[str, Any]], List[Message]],
        schema: Optional[Type] = None,
        **kwargs,
    ):
        """Stream with structured output parsing support."""
        from .streaming import UnifiedStreamingClient

        if self._unified_streaming_client is None:
            self._unified_streaming_client = UnifiedStreamingClient(self.client)

        normalized_messages = self._normalize_messages(messages)

        async for chunk in self._unified_streaming_client.stream_with_schema(
            normalized_messages, schema, **kwargs
        ):
            yield chunk
