"""MCP server connection classes for different transport types."""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..exceptions import MCPConnectionError, MCPTimeoutError

if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger(__name__)

# Owner tasks of live and closing connections. The event loop holds tasks only
# weakly, so a cancelled owner still tearing down must be referenced somewhere.
_OWNER_TASKS: "set[asyncio.Task]" = set()


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.

    Attributes:
        name: Unique identifier for the server (used for tool namespacing)
        transport: Transport type - "stdio", "streamable_http", or "sse"

        Stdio-specific:
            command: Command to execute (e.g., "npx", "python")
            args: Command arguments
            env: Environment variables for the subprocess

        HTTP-specific:
            url: Server URL (e.g., "http://localhost:8000/mcp")
            headers: HTTP headers (e.g., for authentication)

        Connection settings:
            timeout: Connection/operation timeout in seconds
            auto_reconnect: Whether to auto-reconnect on disconnect
            max_retries: Maximum reconnection attempts
    """

    name: str
    transport: str  # "stdio", "streamable_http", "sse"

    # Stdio-specific
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    # HTTP-specific
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    # Connection settings
    timeout: float = 30.0
    auto_reconnect: bool = True
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration based on transport type."""
        transport = self.transport.lower()

        if transport == "stdio":
            if not self.command:
                raise ValueError("Stdio transport requires 'command' in config")
        elif transport in ("streamable_http", "http", "sse"):
            if not self.url:
                raise ValueError(f"{transport} transport requires 'url' in config")
        else:
            raise ValueError(f"Unsupported transport: {transport}")


@dataclass
class NativeMCPServerConfig:
    """Configuration for native provider MCP support (server-side execution).

    This is used when the LLM provider (Anthropic, OpenAI) handles MCP
    connections and tool execution directly, rather than the client.

    Attributes:
        name: Unique identifier for the server
        url: MCP server URL (must be accessible from the provider's servers)
        authorization_token: Optional bearer token for authentication (Anthropic)
        allowed_tools: Optional list of tool names to enable (filter)
        headers: Optional HTTP headers for authentication (OpenAI)
        require_approval: Tool approval mode for OpenAI: "never", "always"
        tool_configuration: Extra per-server tool settings. On Anthropic these
            are folded into the ``mcp_toolset`` entry (``enabled`` →
            ``default_config.enabled``, ``allowed_tools`` → an allowlist in
            ``configs``); the deprecated ``mcp_servers[].tool_configuration``
            wire field is never sent.
        known_tools: Tool names this server is known to serve, for local
            DIAGNOSTICS only — never sent to the provider. A native-MCP tool is
            not in the local registry (the provider connects to the server
            itself), so `ToolRegistry.execute_safe` would otherwise answer a
            misrouted call with "not found" plus a list of every *local* tool,
            which reads as "the integration is gone". Populated by the host
            from whatever it has cached; an empty/None list only costs the
            sharper message.
        description: What the server is for, in the host's words — for the
            host's own prompt, never on the wire. Under deferral the model no
            longer sees the server's tool descriptions up front, so this is
            the one line that tells it the capability exists at all.
        disabled_tools: Tool names the host has switched OFF for this server
            (rendered as ``configs[name].enabled: false`` on Anthropic).
            The provider hosts a connector tool's definition, so this is the
            host's only lever over which of them the model can ever load —
            e.g. a definition too large to sit in context. Wins over
            ``allowed_tools`` for a name in both.
    """

    name: str
    url: str
    authorization_token: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    headers: Optional[Dict[str, str]] = None
    require_approval: str = "never"  # OpenAI: "never", "always"
    tool_configuration: Optional[Dict[str, Any]] = None
    known_tools: Optional[List[str]] = None
    description: Optional[str] = None
    disabled_tools: Optional[List[str]] = None

    def to_anthropic_format(self) -> Dict[str, Any]:
        """The ``mcp_servers[]`` entry: connection details only.

        Tool selection lives in the paired ``mcp_toolset`` (see
        :meth:`to_anthropic_toolset`), not here — under
        ``mcp-client-2025-11-20`` the old ``tool_configuration`` field on the
        server entry is deprecated, and every server must be referenced by
        exactly one toolset in ``tools``.
        """
        config: Dict[str, Any] = {
            "type": "url",
            "url": self.url,
            "name": self.name,
        }
        if self.authorization_token:
            config["authorization_token"] = self.authorization_token
        return config

    def to_anthropic_toolset(self, *, defer_loading: bool = False) -> Dict[str, Any]:
        """The ``tools[]`` entry that enables this server's tools on Anthropic.

        ``defer_loading=True`` marks every tool on the server as deferred:
        the API keeps the definitions server-side and the model reaches
        them through the tool-search tool, exactly like a deferred local
        tool. This is the ONLY deferral mechanism for connector tools — the
        per-tool ``defer_loading`` flag on local schemas never applies to
        them, which is how a whole-server attachment (263 tools, ~660K
        tokens of schema on one production org) shipped into every request
        uncached and cost ~30 s of first-token latency per new thread. Only
        meaningful when the request also carries a ``tool_search_tool_*``
        entry; the caller decides that from the tools array it is sending.

        ``allowed_tools`` becomes the documented allowlist shape
        (``default_config.enabled: false`` + per-tool ``enabled: true``);
        an explicit ``tool_configuration.enabled`` sets the default;
        ``disabled_tools`` become per-tool ``enabled: false`` and win over
        the allowlist.
        """
        toolset: Dict[str, Any] = {
            "type": "mcp_toolset",
            "mcp_server_name": self.name,
        }
        default_config: Dict[str, Any] = {}
        configs: Dict[str, Dict[str, Any]] = {}

        extra = dict(self.tool_configuration or {})
        if "enabled" in extra:
            default_config["enabled"] = bool(extra["enabled"])
        allowed = list(self.allowed_tools or []) or list(extra.get("allowed_tools") or [])
        if allowed:
            default_config["enabled"] = False
            for tool_name in allowed:
                configs[str(tool_name)] = {"enabled": True}
        for tool_name in self.disabled_tools or []:
            configs[str(tool_name)] = {"enabled": False}
        if defer_loading:
            default_config["defer_loading"] = True

        if default_config:
            toolset["default_config"] = default_config
        if configs:
            toolset["configs"] = configs
        return toolset

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI MCP tool format (for Responses API)."""
        # OpenAI requires server_label to start with a letter and only contain
        # letters, digits, '-' and '_'. Sanitize the name accordingly.
        sanitized_label = re.sub(r"[^a-zA-Z0-9_-]", "_", self.name)
        # Ensure it starts with a letter
        if sanitized_label and not sanitized_label[0].isalpha():
            sanitized_label = "mcp_" + sanitized_label

        config: Dict[str, Any] = {
            "type": "mcp",
            "server_label": sanitized_label,
            "server_url": self.url,
            "require_approval": self.require_approval,
        }
        if self.allowed_tools:
            config["allowed_tools"] = self.allowed_tools
        if self.headers:
            config["headers"] = self.headers
        return config


class MCPServerConnection(ABC):
    """Abstract base class for MCP server connections.

    Provides a common interface for connecting to MCP servers via different
    transport mechanisms (stdio, HTTP, SSE).

    The transport and the `ClientSession` run inside one `async with` in a task
    this connection owns, from `connect()` until `disconnect()`. Both are anyio
    task groups: entered in the caller's task, a failure in one of their
    background tasks (a refused port, a dropped stream) cancels the caller's
    task with a bare `CancelledError` that `except Exception` misses (a chat
    turn was cancelled instead of skipping a closed custom server), and
    `disconnect()` from another task cannot exit the cancel scope at all.
    Subclasses only say how to open their transport.
    """

    #: Transport name used in log and error messages.
    transport_label = "MCP"

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Optional[ClientSession] = None
        self._connected = False
        self._tools_cache: Optional[List[Any]] = None
        self._owner: Optional[asyncio.Task] = None
        self._stop: Optional[asyncio.Event] = None
        self._ready: Optional[asyncio.Future] = None

    @property
    def name(self) -> str:
        """Server name from config."""
        return self.config.name

    @property
    def is_connected(self) -> bool:
        """Whether the connection is active."""
        return self._connected

    @abstractmethod
    def _open_transport(self) -> Any:
        """The transport's async context manager, yielding (read, write, ...).

        Raises:
            MCPConnectionError: If the MCP package is not installed
        """

    async def connect(self) -> None:
        """Establish connection to the MCP server.

        Raises:
            MCPTimeoutError: If initialization times out
            MCPConnectionError: If the connection fails for any other reason
        """
        if self._connected:
            logger.warning(
                f"Already connected to {self.transport_label} MCP server: {self.config.name}"
            )
            return
        loop = asyncio.get_running_loop()
        in_flight = self._ready
        if in_flight is not None and not in_flight.done() and in_flight.get_loop() is loop:
            # A concurrent connect() shares the attempt already running.
            await self._await_attempt(self._owner, self._stop, in_flight)
            return
        try:
            from mcp import ClientSession
        except ImportError:
            raise MCPConnectionError(
                "MCP package not installed. Install with: pip install mcp"
            )
        transport = self._open_transport()

        ready: asyncio.Future = loop.create_future()
        # After a cancelled connect() nobody awaits the failure; read it here so
        # asyncio does not log it as never retrieved.
        ready.add_done_callback(lambda f: f.cancelled() or f.exception())
        stop = asyncio.Event()

        async def own_session() -> None:
            failure: Optional[BaseException] = None
            try:
                async with transport as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await asyncio.wait_for(
                            session.initialize(), timeout=self.config.timeout
                        )
                        ready.set_result(session)
                        await stop.wait()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                # Includes the CancelledError of the transport's own cancel
                # scope: it ends this task, never the caller's.
                failure = e
            finally:
                if self._owner is asyncio.current_task():
                    self._connected = False
                if not ready.done():
                    ready.set_exception(self._connect_error(failure))
                elif (
                    failure is not None
                    and not stop.is_set()
                    # A cancel is a deliberate close (the loop that opened the
                    # connection ended), not a server failure.
                    and not isinstance(failure, asyncio.CancelledError)
                ):
                    logger.warning(
                        f"{self.transport_label} MCP server '{self.config.name}' "
                        f"connection closed: {failure!r}"
                    )

        owner = asyncio.create_task(
            own_session(), name=f"mcp-connection:{self.config.name}"
        )
        _OWNER_TASKS.add(owner)
        owner.add_done_callback(_OWNER_TASKS.discard)
        # A task cancelled before its first step never runs its `finally`.
        owner.add_done_callback(
            lambda _: ready.done() or ready.set_exception(self._connect_error(None))
        )
        self._owner, self._stop, self._ready = owner, stop, ready
        try:
            session = await self._await_attempt(owner, stop, ready)
        except BaseException:
            # A failed connect has already ended the owner task. A cancel of
            # this caller abandons the attempt (a concurrent connect() sharing
            # it is told it was disconnected), so nothing stays open.
            if self._owner is owner:
                self._owner = self._stop = self._ready = None
            if not owner.done():
                stop.set()
                owner.cancel()
            raise
        self.session = session
        self._connected = True
        logger.info(
            f"Connected to {self.transport_label} MCP server: {self.config.name}"
        )

    async def _await_attempt(
        self,
        owner: Optional[asyncio.Task],
        stop: Optional[asyncio.Event],
        ready: asyncio.Future,
    ) -> Any:
        """The session of a connect attempt.

        The attempt's own error is raised as is; a disconnect() that ran
        meanwhile (it sets `stop`) wins over both a session and an error.
        """
        try:
            # Shielded so a cancel of one waiter does not cancel `ready` for
            # the others.
            session = await asyncio.shield(ready)
        except Exception as e:
            if stop is None or not stop.is_set():
                raise
            raise self._disconnected_while_connecting() from e
        if owner is None or stop is None or stop.is_set():
            raise self._disconnected_while_connecting()
        if owner.done():
            # The transport failed between handing over the session and now.
            raise self._connect_error(None)
        return session

    def _disconnected_while_connecting(self) -> MCPConnectionError:
        return MCPConnectionError(
            f"Disconnected from {self.transport_label} MCP server "
            f"'{self.config.name}' while connecting"
        )

    def _connect_error(self, failure: Optional[BaseException]) -> Exception:
        leaves = _leaf_errors(failure) if failure else []
        if any(isinstance(e, asyncio.TimeoutError) for e in leaves):
            error: Exception = MCPTimeoutError(
                f"Timeout connecting to {self.transport_label} MCP server: {self.config.name}"
            )
        elif len(leaves) == 1 and isinstance(leaves[0], MCPConnectionError):
            return leaves[0]
        else:
            reason = _describe_failure(failure) if failure else "connection closed"
            error = MCPConnectionError(
                f"Failed to connect to {self.transport_label} MCP server "
                f"'{self.config.name}': {reason}"
            )
        error.__cause__ = failure
        return error

    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        await self._stop_owner()
        logger.info(
            f"Disconnected from {self.transport_label} MCP server: {self.config.name}"
        )

    async def _stop_owner(self) -> None:
        owner, stop, ready = self._owner, self._stop, self._ready
        self._owner = self._stop = self._ready = None
        self._connected = False
        self.session = None
        if owner is None or owner.done():
            return
        owner_loop = owner.get_loop()
        if owner_loop is not asyncio.get_running_loop():
            # Opened on another event loop (async_to_sync starts one per call):
            # its task cannot be awaited from here. Ask that loop to close it if
            # it still runs; a closed loop already cancelled it.
            if not owner_loop.is_closed():
                owner_loop.call_soon_threadsafe(stop.set)
            return
        if not ready.done():
            # Still connecting: there is no session to close gracefully, and
            # waiting would last until initialize times out.
            stop.set()
            owner.cancel()
            return
        stop.set()
        try:
            # Closing the session and transport is bounded like any other call.
            await asyncio.wait_for(asyncio.shield(owner), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            owner.cancel()
        except asyncio.CancelledError:
            owner.cancel()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the MCP server.

        Returns:
            List of MCP tool definitions

        Raises:
            MCPConnectionError: If not connected to server
        """
        if not self._connected or not self.session:
            raise MCPConnectionError(
                f"Not connected to MCP server: {self.config.name}"
            )

        try:
            result = await asyncio.wait_for(
                self.session.list_tools(),
                timeout=self.config.timeout,
            )
            self._tools_cache = result.tools
            return result.tools
        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f"Timeout listing tools from MCP server: {self.config.name}"
            )

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            name: Tool name (original name, not namespaced)
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            MCPConnectionError: If not connected to server
            MCPTimeoutError: If operation times out
        """
        if not self._connected or not self.session:
            raise MCPConnectionError(
                f"Not connected to MCP server: {self.config.name}"
            )

        try:
            return await asyncio.wait_for(
                self.session.call_tool(name, arguments=arguments),
                timeout=self.config.timeout,
            )
        except asyncio.TimeoutError:
            raise MCPTimeoutError(
                f"Timeout calling tool '{name}' on MCP server: {self.config.name}"
            )

    async def ensure_connected(self) -> None:
        """Ensure connection is active, reconnect if needed.

        Raises:
            MCPConnectionError: If reconnection fails after max retries
        """
        if self.is_connected:
            return

        if not self.config.auto_reconnect:
            raise MCPConnectionError(
                f"Not connected to MCP server: {self.config.name}"
            )

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                await self.connect()
                return
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed for "
                        f"'{self.config.name}', retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)

        raise MCPConnectionError(
            f"Failed to connect to MCP server '{self.config.name}' "
            f"after {self.config.max_retries} attempts: {last_error}"
        )

    @asynccontextmanager
    async def managed_connection(self):
        """Context manager for managed connection lifecycle.

        Usage:
            async with connection.managed_connection():
                tools = await connection.list_tools()
        """
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()


def _leaf_errors(exc: BaseException) -> List[BaseException]:
    """The errors inside (possibly nested) exception groups.

    A failure inside an anyio task group arrives as "unhandled errors in a
    TaskGroup (1 sub-exception)", with the real error one level down.
    """
    inner = getattr(exc, "exceptions", None)
    if inner:
        return [leaf for e in inner for leaf in _leaf_errors(e)]
    return [exc]


def _describe_failure(exc: BaseException) -> str:
    """A message naming the leaf errors, e.g. "ConnectError: ..."."""
    described = []
    for leaf in _leaf_errors(exc):
        message = str(leaf).strip()
        name = type(leaf).__name__
        described.append(f"{name}: {message}" if message else name)
    return "; ".join(described)


class StdioMCPConnection(MCPServerConnection):
    """MCP connection via stdio transport.

    Spawns a subprocess and communicates via stdin/stdout.
    """

    transport_label = "stdio"

    def _open_transport(self) -> Any:
        try:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise MCPConnectionError(
                "MCP package not installed. Install with: pip install mcp"
            )
        return stdio_client(
            StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env,
            )
        )


class StreamableHTTPMCPConnection(MCPServerConnection):
    """MCP connection via Streamable HTTP transport.

    Recommended for production use. Connects to an HTTP endpoint
    with streaming support.
    """

    transport_label = "Streamable HTTP"

    def _open_transport(self) -> Any:
        try:
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            raise MCPConnectionError(
                "MCP package not installed. Install with: pip install mcp"
            )
        return streamablehttp_client(self.config.url, headers=self.config.headers)


class SSEMCPConnection(MCPServerConnection):
    """MCP connection via SSE (Server-Sent Events) transport.

    Note: SSE is deprecated in favor of Streamable HTTP.
    This is provided for backward compatibility.
    """

    transport_label = "SSE"

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        logger.warning(
            "SSE transport is deprecated. Consider using streamable_http instead."
        )

    def _open_transport(self) -> Any:
        try:
            from mcp.client.sse import sse_client
        except ImportError:
            raise MCPConnectionError(
                "MCP package not installed. Install with: pip install mcp"
            )
        return sse_client(self.config.url, headers=self.config.headers)


def create_connection(config: MCPServerConfig) -> MCPServerConnection:
    """Factory function to create appropriate connection type.

    Args:
        config: Server configuration with transport type

    Returns:
        Appropriate MCPServerConnection subclass instance

    Raises:
        ValueError: If transport type is not supported
    """
    transport = config.transport.lower()

    if transport == "stdio":
        return StdioMCPConnection(config)
    elif transport in ("streamable_http", "http"):
        return StreamableHTTPMCPConnection(config)
    elif transport == "sse":
        return SSEMCPConnection(config)
    else:
        raise ValueError(f"Unsupported MCP transport: {transport}")
