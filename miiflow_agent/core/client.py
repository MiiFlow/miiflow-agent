"""Core LLM client interface and base implementations."""

import asyncio
import logging
import os
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

from .exceptions import (
    MiiflowLLMError,
    TimeoutError,
    is_retryable_error,
    retry_delay_seconds,
)
from .message import Message, MessageRole
from .metrics import MetricsCollector, TokenCount, UsageData
from .streaming import StreamChunk
from .tools import FunctionTool, ToolRegistry

if TYPE_CHECKING:
    from .callbacks import CallbackEvent, CallbackRegistry

logger = logging.getLogger(__name__)


def _tools_fingerprint(formatted_tools: Optional[List[Any]]) -> Optional[str]:
    """Short digest of the tools array as it will go on the wire.

    Deliberately ORDER-SENSITIVE (list order preserved): the tools array is
    the first prompt-cache tier and byte order is part of the prefix, so a
    reordering must change the fingerprint even when the set is identical.

    Deliberately SHALLOW: hashing (name, description length, top-level key
    set) per entry catches the drift classes actually observed — tools
    appearing/disappearing (connect stubs, skill activation), reordering,
    description edits — without json.dumps'ing a 100K+-char schema array on
    the event loop before every stream round (this runs per LLM call on the
    TTFT path). A deep schema edit with an unchanged description length is
    invisible here; accepted trade. Best-effort — telemetry never fails a
    call.

    Entries marked ``defer_loading`` are EXCLUDED from the digest: the
    Anthropic API strips deferred tools from the prompt (native tool
    search), so churn confined to them — an MCP manager reconnect moving a
    server's block to the tail, a keyword-suffix description edit — cannot
    bust the cache. Hashing them anyway made prod drift stats cry wolf:
    turns with a changed hash, an identical name set, and a FULL cache hit.
    A tool FLIPPING between deferred and loaded still changes the digest,
    because the entry enters or leaves the hashed subset — which matches
    the prefix actually changing.
    """
    fingerprint, _ = _tools_fingerprint_and_names(formatted_tools)
    return fingerprint


def _tools_fingerprint_and_names(
    formatted_tools: Optional[List[Any]],
) -> "tuple[Optional[str], Optional[List[str]]]":
    """`_tools_fingerprint` plus the tool names behind it, in wire order.

    The names make a changed fingerprint diffable: consumers persist them
    sparsely (first call of a turn / on fingerprint change) so drift can be
    attributed to the tools that appeared, vanished, or moved. Same
    best-effort contract — (None, None) on any failure.
    """
    if not formatted_tools:
        return None, None
    try:
        import hashlib

        parts = []
        names = []
        for tool in formatted_tools:
            if not isinstance(tool, dict):
                parts.append(type(tool).__name__)
                names.append(type(tool).__name__)
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
            name = tool.get("name") or fn.get("name") or tool.get("type", "?")
            names.append(str(name))
            if tool.get("defer_loading"):
                # Stripped from the prompt by the API — not part of the
                # cached prefix, so not part of the digest. Still listed in
                # `names` so set-level drift stays diffable.
                continue
            desc = tool.get("description") or fn.get("description") or ""
            parts.append(f"{name}:{len(str(desc))}:{','.join(sorted(tool.keys()))}")
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
        return digest, names
    except Exception:  # noqa: BLE001
        return None, None


def _format_tokens(tokens: TokenCount) -> str:
    """Compact token summary for log lines: in/out/total (+cache split)."""
    if tokens is None:
        return "in=?/out=?/total=?"
    in_t = getattr(tokens, "prompt_tokens", None)
    out_t = getattr(tokens, "completion_tokens", None)
    total_t = getattr(tokens, "total_tokens", None)
    base = f"in={in_t}/out={out_t}/total={total_t}"
    cache_r = getattr(tokens, "cache_read_tokens", 0) or 0
    cache_w = getattr(tokens, "cache_write_tokens", 0) or 0
    if cache_r or cache_w:
        return f"{base} cache_read={cache_r}/cache_write={cache_w}"
    return base


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
        timeout: float = 300.0,
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

        # Retries for astream_chat, applied only until the first chunk
        # arrives (see astream_chat). 0 disables.
        try:
            self._max_stream_retries = max(
                0, int(os.getenv("MIIFLOW_STREAM_RETRY_ATTEMPTS", "3"))
            )
        except ValueError:
            self._max_stream_retries = 3

        # Per-chunk inactivity deadline for astream_chat. The providers'
        # own timeout guards stream *open* only, so a connection that opened
        # and then stalled mid-token used to hang the run forever — the
        # loop-level MaxTimeCondition only runs between steps and cannot
        # interrupt a hung await. Default 300s (matching ModelClient.timeout):
        # reasoning-heavy models can legitimately emit nothing for minutes
        # before the first token, and a tighter default turned those silent
        # periods into failures with re-billed retries. <=0 disables.
        try:
            self._stream_inactivity_timeout = float(
                os.getenv("MIIFLOW_STREAM_INACTIVITY_TIMEOUT", "300")
            )
        except ValueError:
            self._stream_inactivity_timeout = 300.0

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
        from .callbacks import get_active_registry

        return get_active_registry()

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
    # Provider-neutral control kwargs callers may pass to achat/astream_chat.
    # Only the providers listed handle them; every other provider forwards
    # unknown kwargs into its SDK call (Mistral) or its options map (Ollama),
    # so they must be dropped here rather than trusted to be ignored.
    _PROVIDERS_HANDLING_THINKING_DISABLED = frozenset({"anthropic", "bedrock"})

    def _strip_control_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Remove control kwargs the bound provider does not understand.

        * ``thinking_disabled`` — kept for providers that pop it.
        * ``mcp_servers=None`` — the caller's opt-out of registry injection
          (see above) has done its job once we get here; forwarding a literal
          None would reach SDKs that reject unknown keyword arguments.
        """
        provider = getattr(self.client, "provider_name", None)
        if (
            "thinking_disabled" in kwargs
            and provider not in self._PROVIDERS_HANDLING_THINKING_DISABLED
        ):
            kwargs.pop("thinking_disabled")
        if "mcp_servers" in kwargs and kwargs["mcp_servers"] is None:
            kwargs.pop("mcp_servers")

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
            elif self.tool_registry.tools or self.tool_registry.http_tools or self.tool_registry.mcp_tools:
                from .tools.tool_search import get_enabled_tool_names_ordered, is_session_active

                if is_session_active() and self.tool_registry.should_use_tool_search():
                    formatted_tools = self.tool_registry.get_filtered_schemas(
                        self.client.provider_name,
                        self.client,
                        enabled_names=get_enabled_tool_names_ordered(),
                    )
                else:
                    formatted_tools = self.tool_registry.get_schemas(
                        self.client.provider_name, self.client
                    )

        # Check for native MCP servers and pass to provider if supported. An
        # explicit `mcp_servers=` from the caller (including `None`, meaning
        # "no server-side tools on this call") wins over the registry: a
        # compaction summary or a classifier must not carry the agent's MCP
        # connectors — they cost tokens, change model behaviour, and ship the
        # servers' auth headers with a call that has no use for them.
        if (
            "mcp_servers" not in kwargs
            and self.tool_registry.has_native_mcp_servers()
            and self._supports_native_mcp()
        ):
            kwargs["mcp_servers"] = self.tool_registry.get_native_mcp_configs()
        self._strip_control_kwargs(kwargs)

        # Get callback context
        ctx = get_callback_context()

        start_time = time.time()
        try:
            response = await self.client.achat(normalized_messages, tools=formatted_tools, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "[LLM_CALL] provider=%s model=%s mode=achat status=ok "
                "latency_ms=%.0f tokens=%s tools=%d",
                self.client.provider_name,
                self.client.model,
                latency_ms,
                _format_tokens(response.usage),
                len(formatted_tools or []),
            )

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

            logger.warning(
                "[LLM_CALL] provider=%s model=%s mode=achat status=error "
                "latency_ms=%.0f error=%s tools=%d",
                self.client.provider_name,
                self.client.model,
                latency_ms,
                type(e).__name__,
                len(formatted_tools or []),
            )

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

    async def _guard_stream_inactivity(
        self, stream: AsyncIterator[StreamChunk]
    ) -> AsyncIterator[StreamChunk]:
        """Yield from ``stream``, bounding the wait for each chunk.

        A stall before the first chunk raises a retryable TimeoutError (the
        retry loop in astream_chat reopens the stream); a stall after content
        has flowed surfaces the same error to the caller instead of hanging
        the run forever.
        """
        timeout = self._stream_inactivity_timeout
        if not timeout or timeout <= 0:
            async for chunk in stream:
                yield chunk
            return

        iterator = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                # Release the underlying HTTP resources before surfacing;
                # the generator was cancelled mid-__anext__ by wait_for.
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001 — already failing
                        pass
                raise TimeoutError(
                    f"LLM stream produced no data for {timeout:.0f}s "
                    f"(provider={self.client.provider_name}, "
                    f"model={self.client.model})",
                    timeout,
                )
            yield chunk

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
            elif self.tool_registry.tools or self.tool_registry.http_tools or self.tool_registry.mcp_tools:
                from .tools.tool_search import get_enabled_tool_names_ordered, is_session_active

                if is_session_active() and self.tool_registry.should_use_tool_search():
                    formatted_tools = self.tool_registry.get_filtered_schemas(
                        self.client.provider_name,
                        self.client,
                        enabled_names=get_enabled_tool_names_ordered(),
                    )
                else:
                    formatted_tools = self.tool_registry.get_schemas(
                        self.client.provider_name, self.client
                    )

        # Check for native MCP servers and pass to provider if supported. An
        # explicit `mcp_servers=` from the caller (including `None`, meaning
        # "no server-side tools on this call") wins over the registry: a
        # compaction summary or a classifier must not carry the agent's MCP
        # connectors — they cost tokens, change model behaviour, and ship the
        # servers' auth headers with a call that has no use for them.
        if (
            "mcp_servers" not in kwargs
            and self.tool_registry.has_native_mcp_servers()
            and self._supports_native_mcp()
        ):
            kwargs["mcp_servers"] = self.tool_registry.get_native_mcp_configs()
        self._strip_control_kwargs(kwargs)

        # Get callback context
        ctx = get_callback_context()

        start_time = time.time()
        total_tokens = TokenCount()
        callback_emitted = False
        error_occurred = None
        tools_fingerprint, tools_names = _tools_fingerprint_and_names(formatted_tools)

        # Timing decomposition (kimi-code's StreamDecodeStats insight): a slow
        # step is only diagnosable when the wait splits into build (local
        # serialization before the request), TTFT (network + server queue to
        # first token, excluding retry backoff), and stream (token
        # generation). One latency_ms number cannot tell "our schemas are
        # huge" from "the provider is overloaded" from "the answer was long".
        first_open_at: Optional[float] = None
        open_at: Optional[float] = None
        first_chunk_at: Optional[float] = None
        attempts_used = 0

        try:
            # Transport retry, but ONLY until the first chunk arrives. Before
            # that point no tokens have been produced and no deltas have
            # reached the consumer, so reopening the stream is free; after it,
            # a blind retry would replay content the consumer already saw.
            # This is the loop's only hot path (achat has tenacity; this had
            # nothing), so a transient 429/529/5xx used to kill the whole run.
            attempt = 0
            while True:
                saw_first_chunk = False
                open_at = time.time()
                if first_open_at is None:
                    first_open_at = open_at
                try:
                    provider_stream = self.client.astream_chat(
                        normalized_messages, tools=formatted_tools, **kwargs
                    )
                    async for chunk in self._guard_stream_inactivity(provider_stream):
                        if not saw_first_chunk:
                            saw_first_chunk = True
                            first_chunk_at = time.time()
                        if chunk.usage:
                            total_tokens += chunk.usage
                        yield chunk
                    break
                except Exception as stream_error:
                    attempt += 1
                    attempts_used = attempt
                    if (
                        saw_first_chunk
                        or attempt > self._max_stream_retries
                        or not is_retryable_error(stream_error)
                    ):
                        raise
                    delay = retry_delay_seconds(stream_error, attempt)
                    logger.warning(
                        "[LLM_CALL] provider=%s model=%s stream open failed "
                        "(%s: %s); retry %d/%d in %.1fs",
                        self.client.provider_name,
                        self.client.model,
                        type(stream_error).__name__,
                        str(stream_error)[:200],
                        attempt,
                        self._max_stream_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)

        except Exception as e:
            error_occurred = e
            raise

        finally:
            # Always emit callback when generator closes (success, break, or error)
            if not callback_emitted:
                now = time.time()
                latency_ms = (now - start_time) * 1000

                # The decomposition (see the tracking vars above). build =
                # normalization + tool formatting before the first open; ttft
                # is measured from the LAST open so retry backoff doesn't
                # pollute it; stream = first chunk → close.
                build_ms = (
                    (first_open_at - start_time) * 1000 if first_open_at else None
                )
                ttft_ms = (
                    (first_chunk_at - open_at) * 1000
                    if first_chunk_at and open_at
                    else None
                )
                stream_ms = (now - first_chunk_at) * 1000 if first_chunk_at else None

                def _ms(value: Optional[float]) -> str:
                    return f"{value:.0f}" if value is not None else "na"

                timing_suffix = (
                    f" build_ms={_ms(build_ms)} ttft_ms={_ms(ttft_ms)} "
                    f"stream_ms={_ms(stream_ms)} retries={attempts_used}"
                )

                if error_occurred:
                    logger.warning(
                        "[LLM_CALL] provider=%s model=%s mode=astream status=error "
                        "latency_ms=%.0f tokens=%s error=%s tools=%d tools_hash=%s%s",
                        self.client.provider_name,
                        self.client.model,
                        latency_ms,
                        _format_tokens(total_tokens),
                        type(error_occurred).__name__,
                        len(formatted_tools or []),
                        tools_fingerprint,
                        timing_suffix,
                    )

                    # Record failed streaming usage
                    self._record_usage(
                        normalized_messages, total_tokens, now - start_time, success=False
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
                        ttft_ms=ttft_ms,
                        stream_ms=stream_ms,
                        request_build_ms=build_ms,
                        transport_retries=attempts_used,
                        tools_fingerprint=tools_fingerprint,
                        tools_names=tools_names,
                        context=ctx,
                        success=False,
                    )
                    # Use sync emit in finally block since we can't await
                    self.callback_registry.emit_sync(error_event)
                else:
                    logger.info(
                        "[LLM_CALL] provider=%s model=%s mode=astream status=ok "
                        "latency_ms=%.0f tokens=%s tools=%d tools_hash=%s%s",
                        self.client.provider_name,
                        self.client.model,
                        latency_ms,
                        _format_tokens(total_tokens),
                        len(formatted_tools or []),
                        tools_fingerprint,
                        timing_suffix,
                    )

                    # Record successful streaming usage
                    self._record_usage(
                        normalized_messages, total_tokens, now - start_time, success=True
                    )

                    # Emit POST_CALL callback
                    post_event = CallbackEvent(
                        event_type=CallbackEventType.POST_CALL,
                        provider=self.client.provider_name,
                        model=self.client.model,
                        tokens=total_tokens,
                        latency_ms=latency_ms,
                        ttft_ms=ttft_ms,
                        stream_ms=stream_ms,
                        request_build_ms=build_ms,
                        transport_retries=attempts_used,
                        tools_fingerprint=tools_fingerprint,
                        tools_names=tools_names,
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
