"""MCP Tool Manager for managing multiple MCP server connections."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .mcp_connection import MCPServerConfig, MCPServerConnection, create_connection
from .mcp_tool import MCPTool
from ..exceptions import MCPConnectionError

logger = logging.getLogger(__name__)


class MCPToolManager:
    """Manages multiple MCP server connections and their tools.

    Provides a unified interface for:
    - Configuring and connecting to multiple MCP servers
    - Discovering and registering tools from all servers
    - Executing tools across servers

    Usage:
        manager = MCPToolManager()

        # Add servers
        manager.add_server(MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        ))

        # Connect and discover tools
        async with manager:
            tools = manager.get_all_tools()
            result = await manager.execute_tool(
                "filesystem__read_file",
                path="/etc/hosts"
            )
    """

    def __init__(self):
        """Initialize the MCP tool manager."""
        self._servers: Dict[str, MCPServerConfig] = {}
        self._connections: Dict[str, MCPServerConnection] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether all servers are connected."""
        return self._connected

    @property
    def server_names(self) -> List[str]:
        """List of configured server names."""
        return list(self._servers.keys())

    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration.

        Args:
            config: Server configuration

        Raises:
            ValueError: If server with same name already exists
        """
        if config.name in self._servers:
            raise ValueError(f"Server '{config.name}' already registered")

        self._servers[config.name] = config
        logger.info(f"Added MCP server config: {config.name} ({config.transport})")

    def remove_server(self, name: str) -> None:
        """Remove an MCP server configuration.

        Args:
            name: Server name to remove
        """
        if name in self._servers:
            del self._servers[name]

        if name in self._connections:
            del self._connections[name]

        # Remove tools from this server
        tools_to_remove = [
            tool_name
            for tool_name, tool in self._tools.items()
            if tool.server_name == name
        ]
        for tool_name in tools_to_remove:
            del self._tools[tool_name]

        logger.info(f"Removed MCP server: {name}")

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers.

        Raises:
            MCPConnectionError: If any server fails to connect
        """
        errors = []

        for name, config in self._servers.items():
            try:
                connection = create_connection(config)
                await connection.connect()
                self._connections[name] = connection

                # Discover and register tools
                tools = await connection.list_tools()
                for mcp_tool in tools:
                    tool = MCPTool(mcp_tool, connection, name)
                    self._tools[tool.name] = tool
                    logger.debug(f"Registered MCP tool: {tool.name}")

                logger.info(
                    f"Connected to MCP server '{name}': {len(tools)} tools discovered"
                )

            except Exception as e:
                error_msg = f"Failed to connect to MCP server '{name}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            # Disconnect any successful connections
            await self._disconnect_all_silent()
            raise MCPConnectionError(
                f"Failed to connect to MCP servers:\n" + "\n".join(errors)
            )

        self._connected = True

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        await self._disconnect_all_silent()
        self._connected = False
        logger.info("Disconnected from all MCP servers")

    async def _disconnect_all_silent(self) -> None:
        """Disconnect from all servers, logging but not raising errors."""
        for name, connection in list(self._connections.items()):
            try:
                await connection.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from '{name}': {e}")

        self._connections.clear()
        self._tools.clear()

    async def __aenter__(self) -> MCPToolManager:
        """Async context manager entry."""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        await self.disconnect_all()
        return False

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific MCP tool by name.

        Args:
            name: Namespaced tool name (e.g., "server__tool")

        Returns:
            MCPTool instance or None if not found
        """
        return self._tools.get(name)

    def get_all_tools(self) -> List[MCPTool]:
        """Get all registered MCP tools.

        Returns:
            List of all MCPTool instances
        """
        return list(self._tools.values())

    def get_server_tools(self, server_name: str) -> List[MCPTool]:
        """Get all tools from a specific server.

        Args:
            server_name: Name of the MCP server

        Returns:
            List of MCPTool instances from that server
        """
        return [
            tool for tool in self._tools.values() if tool.server_name == server_name
        ]

    def list_tool_names(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of namespaced tool names
        """
        return list(self._tools.keys())

    async def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute an MCP tool by name.

        Args:
            name: Namespaced tool name
            **kwargs: Tool arguments

        Returns:
            ToolResult from the execution

        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            available = self.list_tool_names()
            raise ValueError(
                f"MCP tool not found: '{name}'. Available: {available}"
            )

        return await tool.execute(**kwargs)

    def get_connection(self, server_name: str) -> Optional[MCPServerConnection]:
        """Get the connection for a specific server.

        Args:
            server_name: Name of the MCP server

        Returns:
            MCPServerConnection instance or None
        """
        return self._connections.get(server_name)

    async def refresh_tools(self, server_name: Optional[str] = None) -> None:
        """Refresh tool list from servers.

        Args:
            server_name: Specific server to refresh, or None for all
        """
        servers_to_refresh = (
            [server_name] if server_name else list(self._connections.keys())
        )

        for name in servers_to_refresh:
            connection = self._connections.get(name)
            if not connection:
                continue

            # Remove existing tools from this server
            tools_to_remove = [
                tool_name
                for tool_name, tool in self._tools.items()
                if tool.server_name == name
            ]
            for tool_name in tools_to_remove:
                del self._tools[tool_name]

            # Re-discover tools
            try:
                tools = await connection.list_tools()
                for mcp_tool in tools:
                    tool = MCPTool(mcp_tool, connection, name)
                    self._tools[tool.name] = tool

                logger.info(f"Refreshed tools from '{name}': {len(tools)} tools")
            except Exception as e:
                logger.error(f"Failed to refresh tools from '{name}': {e}")

    def get_schemas(self, provider: str) -> List[Dict[str, Any]]:
        """Get all tool schemas in provider format.

        Args:
            provider: Provider name (openai, anthropic, gemini, etc.)

        Returns:
            List of provider-formatted tool schemas
        """
        return [tool.to_provider_format(provider) for tool in self._tools.values()]

    def __repr__(self) -> str:
        return (
            f"MCPToolManager(servers={len(self._servers)}, "
            f"tools={len(self._tools)}, connected={self._connected})"
        )
