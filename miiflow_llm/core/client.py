"""Core LLM client interface and base implementations."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
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
from .tools import FunctionTool, ToolRegistry


@dataclass
class ChatResponse:
    """Response from LLM chat completion."""

    message: Message
    usage: TokenCount
    model: str
    provider: str
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class StreamChunk:
    """Streaming response chunk."""

    content: str
    delta: str
    finish_reason: Optional[str] = None
    usage: Optional[TokenCount] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


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
    ):
        self.client = client
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.client.metrics_collector = self.metrics_collector
        self.tool_registry = tool_registry or ToolRegistry()

        # Initialize unified streaming client
        self._unified_streaming_client = None

    @classmethod
    def create(
        cls, provider: str, model: str, api_key: Optional[str] = None, **kwargs
    ) -> "LLMClient":
        """Create client for specified provider."""
        from ..providers import get_provider_client

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
        **kwargs,
    ) -> ChatResponse:
        """Send async chat completion request."""
        normalized_messages = self._normalize_messages(messages)

        formatted_tools = None
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

        start_time = time.time()
        try:
            response = await self.client.achat(normalized_messages, tools=formatted_tools, **kwargs)

            # Record successful usage
            self._record_usage(
                normalized_messages, response.usage, time.time() - start_time, success=True
            )

            return response

        except Exception as e:
            # Record failed usage
            self._record_usage(
                normalized_messages, TokenCount(), time.time() - start_time, success=False
            )
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
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Send async streaming chat completion request."""
        normalized_messages = self._normalize_messages(messages)

        formatted_tools = None
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

        start_time = time.time()
        total_tokens = TokenCount()

        try:
            async for chunk in self.client.astream_chat(
                normalized_messages, tools=formatted_tools, **kwargs
            ):
                if chunk.usage:
                    total_tokens += chunk.usage
                yield chunk

            # Record successful streaming usage
            self._record_usage(
                normalized_messages, total_tokens, time.time() - start_time, success=True
            )

        except Exception as e:
            # Record failed streaming usage
            self._record_usage(
                normalized_messages, total_tokens, time.time() - start_time, success=False
            )
            raise

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
