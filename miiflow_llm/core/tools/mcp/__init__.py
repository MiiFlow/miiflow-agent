"""
MCP (Model Context Protocol) tools package.

Provides client support for connecting to external MCP servers and integrating
their tools with the miiflow-llm tool system.

Supported transports:
- stdio: Local subprocess communication
- streamable_http: HTTP with streaming support (recommended for production)
- sse: HTTP Server-Sent Events (deprecated, use streamable_http)

Usage:
    from miiflow_llm.core.tools.mcp import MCPToolManager, MCPServerConfig

    manager = MCPToolManager()
    manager.add_server(MCPServerConfig(
        name="filesystem",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    ))

    async with manager:
        tools = manager.get_all_tools()
        result = await manager.execute_tool("filesystem__read_file", path="/etc/hosts")
"""

from .mcp_connection import (
    MCPServerConfig,
    NativeMCPServerConfig,
    MCPServerConnection,
    StdioMCPConnection,
    SSEMCPConnection,
    StreamableHTTPMCPConnection,
    create_connection,
)
from .mcp_tool import MCPTool
from .mcp_manager import MCPToolManager

__all__ = [
    # Configuration
    "MCPServerConfig",
    "NativeMCPServerConfig",
    # Connection classes
    "MCPServerConnection",
    "StdioMCPConnection",
    "SSEMCPConnection",
    "StreamableHTTPMCPConnection",
    "create_connection",
    # Tool wrapper
    "MCPTool",
    # Manager
    "MCPToolManager",
]
