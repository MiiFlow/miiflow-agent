"""Clean tool executor adapter."""

import logging
from typing import List

from ..tools import ToolResult

logger = logging.getLogger(__name__)


class AgentToolExecutor:
    """Tool execution adapter following Django Manager pattern."""

    def __init__(self, agent):
        self.agent = agent
        self._tool_registry = agent.tool_registry
        self._client = agent.client

    async def execute_tool(self, tool_name: str, inputs: dict, context=None) -> ToolResult:
        """Execute tool with context injection if context is provided."""
        if context is not None:
            return await self._tool_registry.execute_safe_with_context(tool_name, context, **inputs)
        return await self._tool_registry.execute_safe(tool_name, **inputs)

    async def execute_without_tools(self, messages: List, temperature: float = None):
        """Execute LLM call with tools temporarily disabled."""
        saved_state = self._save_tool_state()

        try:
            self._disable_all_tools()
            return await self._client.achat(
                messages=messages, temperature=temperature or self.agent.temperature
            )
        finally:
            self._restore_tool_state(saved_state)

    async def stream_without_tools(self, messages: List, temperature: float = None):
        """Stream LLM call with tools temporarily disabled."""
        saved_state = self._save_tool_state()

        try:
            self._disable_all_tools()
            async for chunk in self._client.astream_chat(
                messages=messages, temperature=temperature or self.agent.temperature
            ):
                yield chunk
        finally:
            self._restore_tool_state(saved_state)

    def list_tools(self) -> List[str]:
        return self._tool_registry.list_tools()

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_registry.tools

    def get_tool_schema(self, tool_name: str) -> dict:
        tool = self._tool_registry.tools.get(tool_name)
        return tool.schema.to_universal_schema() if tool else {}

    def tool_needs_context(self, tool_name: str) -> bool:
        """Check if a tool requires context injection."""
        tool = self._tool_registry.tools.get(tool_name)
        if not tool:
            return False
        # Check if tool has context_injection attribute and if pattern is not 'none'
        if hasattr(tool, 'context_injection'):
            pattern = tool.context_injection.get('pattern', 'none')
            return pattern in ('first_param', 'keyword')
        return False

    def build_tools_description(self) -> str:
        """Format all tools for system prompt."""
        if not self.list_tools():
            return "No tools available."

        descriptions = []
        for tool_name in sorted(self.list_tools()):
            schema = self.get_tool_schema(tool_name)
            tool_desc = self._format_tool_description(tool_name, schema)
            descriptions.append(tool_desc)

            # DEBUG: Log each tool description
            logger.info(f"[TOOL SCHEMA] {tool_desc}")

        tools_text = "\n".join(descriptions)
        logger.info(f"[TOOLS DESCRIPTION] Built tools description with {len(descriptions)} tools")
        return tools_text

    def _save_tool_state(self) -> dict:
        return {
            "tools": dict(self._client.tool_registry.tools),
            "http_tools": dict(self._client.tool_registry.http_tools),
        }

    def _restore_tool_state(self, state: dict) -> None:
        self._client.tool_registry.tools = state["tools"]
        self._client.tool_registry.http_tools = state["http_tools"]

    def _disable_all_tools(self) -> None:
        self._client.tool_registry.tools = {}
        self._client.tool_registry.http_tools = {}

    def _format_tool_description(self, tool_name: str, schema: dict) -> str:
        """Format tool description for system prompt with detailed parameter information."""
        desc = schema.get("description", "No description available")
        params = schema.get("parameters", {}).get("properties", {})

        if not params:
            return f"- {tool_name}(): {desc}"

        # Format parameters with type and enum information
        param_descriptions = []
        for param_name, param_schema in params.items():
            param_type = param_schema.get("type", "any")

            # If enum exists, show the allowed values
            if "enum" in param_schema and param_schema["enum"]:
                enum_values = param_schema["enum"]
                if len(enum_values) <= 5:  # Show all values if reasonable
                    enum_str = "|".join(f'"{v}"' for v in enum_values)
                    param_descriptions.append(f'{param_name}: {param_type}({enum_str})')
                else:  # Just indicate there are allowed values
                    param_descriptions.append(f'{param_name}: {param_type}(allowed values defined)')
            else:
                param_descriptions.append(f'{param_name}: {param_type}')

        params_str = ', '.join(param_descriptions)
        return f"- {tool_name}({params_str}): {desc}"
