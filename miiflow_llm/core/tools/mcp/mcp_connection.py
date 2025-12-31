"""MCP server connection classes for different transport types."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..exceptions import MCPConnectionError, MCPTimeoutError

if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger(__name__)


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
        tool_configuration: Additional tool configuration options
    """

    name: str
    url: str
    authorization_token: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    headers: Optional[Dict[str, str]] = None
    require_approval: str = "never"  # OpenAI: "never", "always"
    tool_configuration: Optional[Dict[str, Any]] = None

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic MCP server format."""
        config: Dict[str, Any] = {
            "type": "url",
            "url": self.url,
            "name": self.name,
        }
        if self.authorization_token:
            config["authorization_token"] = self.authorization_token
        if self.allowed_tools:
            config["tool_configuration"] = {"allowed_tools": self.allowed_tools}
        if self.tool_configuration:
            # Merge with existing tool_configuration
            existing = config.get("tool_configuration", {})
            config["tool_configuration"] = {**existing, **self.tool_configuration}
        return config

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI MCP tool format (for Responses API)."""
        config: Dict[str, Any] = {
            "type": "mcp",
            "server_label": self.name,
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
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Optional[ClientSession] = None
        self._connected = False
        self._tools_cache: Optional[List[Any]] = None

    @property
    def name(self) -> str:
        """Server name from config."""
        return self.config.name

    @property
    def is_connected(self) -> bool:
        """Whether the connection is active."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the MCP server."""
        pass

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


class StdioMCPConnection(MCPServerConnection):
    """MCP connection via stdio transport.

    Spawns a subprocess and communicates via stdin/stdout.
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._client_context = None
        self._session_context = None

    async def connect(self) -> None:
        """Connect to MCP server via stdio."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise MCPConnectionError(
                "MCP package not installed. Install with: pip install mcp"
            )

        if self._connected:
            logger.warning(f"Already connected to stdio MCP server: {self.config.name}")
            return

        try:
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env,
            )

            # Store context managers for cleanup
            self._client_context = stdio_client(server_params)
            read, write = await self._client_context.__aenter__()

            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()

            await asyncio.wait_for(
                self.session.initialize(),
                timeout=self.config.timeout,
            )

            self._connected = True
            logger.info(f"Connected to stdio MCP server: {self.config.name}")

        except asyncio.TimeoutError:
            await self._cleanup_contexts()
            raise MCPTimeoutError(
                f"Timeout connecting to stdio MCP server: {self.config.name}"
            )
        except Exception as e:
            await self._cleanup_contexts()
            raise MCPConnectionError(
                f"Failed to connect to stdio MCP server '{self.config.name}': {e}"
            )

    async def disconnect(self) -> None:
        """Disconnect from stdio MCP server."""
        await self._cleanup_contexts()
        self._connected = False
        self.session = None
        logger.info(f"Disconnected from stdio MCP server: {self.config.name}")

    async def _cleanup_contexts(self) -> None:
        """Clean up context managers."""
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing session context: {e}")
            self._session_context = None

        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing client context: {e}")
            self._client_context = None


class StreamableHTTPMCPConnection(MCPServerConnection):
    """MCP connection via Streamable HTTP transport.

    Recommended for production use. Connects to an HTTP endpoint
    with streaming support.
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._client_context = None
        self._session_context = None

    async def connect(self) -> None:
        """Connect to MCP server via Streamable HTTP."""
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            raise MCPConnectionError(
                "MCP package not installed. Install with: pip install mcp"
            )

        if self._connected:
            logger.warning(
                f"Already connected to Streamable HTTP MCP server: {self.config.name}"
            )
            return

        try:
            self._client_context = streamablehttp_client(
                self.config.url,
                headers=self.config.headers,
            )
            read_stream, write_stream, _ = await self._client_context.__aenter__()

            self._session_context = ClientSession(read_stream, write_stream)
            self.session = await self._session_context.__aenter__()

            await asyncio.wait_for(
                self.session.initialize(),
                timeout=self.config.timeout,
            )

            self._connected = True
            logger.info(
                f"Connected to Streamable HTTP MCP server: {self.config.name}"
            )

        except asyncio.TimeoutError:
            await self._cleanup_contexts()
            raise MCPTimeoutError(
                f"Timeout connecting to Streamable HTTP MCP server: {self.config.name}"
            )
        except Exception as e:
            await self._cleanup_contexts()
            raise MCPConnectionError(
                f"Failed to connect to Streamable HTTP MCP server "
                f"'{self.config.name}': {e}"
            )

    async def disconnect(self) -> None:
        """Disconnect from Streamable HTTP MCP server."""
        await self._cleanup_contexts()
        self._connected = False
        self.session = None
        logger.info(
            f"Disconnected from Streamable HTTP MCP server: {self.config.name}"
        )

    async def _cleanup_contexts(self) -> None:
        """Clean up context managers."""
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing session context: {e}")
            self._session_context = None

        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing client context: {e}")
            self._client_context = None


class SSEMCPConnection(MCPServerConnection):
    """MCP connection via SSE (Server-Sent Events) transport.

    Note: SSE is deprecated in favor of Streamable HTTP.
    This is provided for backward compatibility.
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._client_context = None
        self._session_context = None
        logger.warning(
            "SSE transport is deprecated. Consider using streamable_http instead."
        )

    async def connect(self) -> None:
        """Connect to MCP server via SSE."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise MCPConnectionError(
                "MCP package not installed. Install with: pip install mcp"
            )

        if self._connected:
            logger.warning(f"Already connected to SSE MCP server: {self.config.name}")
            return

        try:
            self._client_context = sse_client(
                self.config.url,
                headers=self.config.headers,
            )
            read_stream, write_stream = await self._client_context.__aenter__()

            self._session_context = ClientSession(read_stream, write_stream)
            self.session = await self._session_context.__aenter__()

            await asyncio.wait_for(
                self.session.initialize(),
                timeout=self.config.timeout,
            )

            self._connected = True
            logger.info(f"Connected to SSE MCP server: {self.config.name}")

        except asyncio.TimeoutError:
            await self._cleanup_contexts()
            raise MCPTimeoutError(
                f"Timeout connecting to SSE MCP server: {self.config.name}"
            )
        except Exception as e:
            await self._cleanup_contexts()
            raise MCPConnectionError(
                f"Failed to connect to SSE MCP server '{self.config.name}': {e}"
            )

    async def disconnect(self) -> None:
        """Disconnect from SSE MCP server."""
        await self._cleanup_contexts()
        self._connected = False
        self.session = None
        logger.info(f"Disconnected from SSE MCP server: {self.config.name}")

    async def _cleanup_contexts(self) -> None:
        """Clean up context managers."""
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing session context: {e}")
            self._session_context = None

        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing client context: {e}")
            self._client_context = None


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
