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
                messages=messages,
                temperature=temperature or self.agent.temperature,
                max_tokens=self.agent.max_tokens,
            )
        finally:
            self._restore_tool_state(saved_state)

    async def stream_without_tools(self, messages: List, temperature: float = None):
        """Stream LLM call with tools temporarily disabled."""
        saved_state = self._save_tool_state()

        try:
            self._disable_all_tools()
            async for chunk in self._client.astream_chat(
                messages=messages,
                temperature=temperature or self.agent.temperature,
                max_tokens=self.agent.max_tokens,
            ):
                yield chunk
        finally:
            self._restore_tool_state(saved_state)

    async def execute_with_tools(self, messages: List, temperature: float = None):
        """Execute LLM call WITH native tools enabled."""
        tools = self._build_native_tool_schemas()

        # Use LLMClient with pre-formatted tools via _formatted_tools parameter
        # This ensures callbacks fire while avoiding tool re-formatting
        return await self._client.achat(
            messages=messages,
            _formatted_tools=tools,
            temperature=temperature or self.agent.temperature,
            max_tokens=self.agent.max_tokens,
        )

    async def stream_with_tools(self, messages: List, temperature: float = None):
        """Stream LLM call WITH native tools enabled."""
        tools = self._build_native_tool_schemas()

        # Use LLMClient with pre-formatted tools via _formatted_tools parameter
        # This ensures callbacks fire while avoiding tool re-formatting
        async for chunk in self._client.astream_chat(
            messages=messages,
            _formatted_tools=tools,
            temperature=temperature or self.agent.temperature,
            max_tokens=self.agent.max_tokens,
        ):
            yield chunk

    def _build_native_tool_schemas(self) -> List:
        """Build tool schemas in native provider format.

        Converts universal schemas to provider-specific format
        (OpenAI, Anthropic, Gemini, etc.)
        """
        from ..tools import FunctionTool

        native_schemas = []

        for tool_name in self.list_tools():
            tool = self._tool_registry.tools.get(tool_name)
            if not tool:
                continue

            # Get universal schema from tool
            if isinstance(tool, FunctionTool):
                universal_schema = tool.schema.to_universal_schema()
            else:
                universal_schema = self.get_tool_schema(tool_name)

            # Filter out context parameters from schema
            # (context is injected, not exposed to LLM)
            filtered_schema = self._filter_context_params(tool_name, universal_schema)

            # Inject __description parameter for LLM to provide human-readable descriptions
            filtered_schema = self._inject_description_param(filtered_schema)

            # Convert to provider-specific format
            provider_schema = self._client.client.convert_schema_to_provider_format(filtered_schema)
            native_schemas.append(provider_schema)

        logger.debug(
            f"Built {len(native_schemas)} native tool schemas for provider {self._client.client.provider_name}"
        )

        # Debug: Log the actual schemas being sent
        import json

        logger.debug(
            f"Tool schemas being sent to provider:\n{json.dumps(native_schemas, indent=2, default=str)}"
        )

        return native_schemas

    def _filter_context_params(self, tool_name: str, schema: dict) -> dict:
        """Remove context parameters from schema (they're injected, not LLM-provided)."""
        tool = self._tool_registry.tools.get(tool_name)
        if not tool or not hasattr(tool, "context_injection"):
            return schema

        context_pattern = tool.context_injection.get("pattern", "none")

        # If tool has context injection, remove the context parameter from schema
        if context_pattern == "first_param":
            # First param is context - already handled by FunctionTool schema generation
            # FunctionTool.schema already excludes first param if it's context
            return schema

        elif context_pattern == "keyword":
            # Remove 'context' or 'ctx' keyword parameter from properties
            filtered_schema = schema.copy()
            if "parameters" in filtered_schema and "properties" in filtered_schema["parameters"]:
                properties = filtered_schema["parameters"]["properties"].copy()
                properties.pop("context", None)
                properties.pop("ctx", None)
                filtered_schema["parameters"]["properties"] = properties

                # Also remove from required list
                if "required" in filtered_schema["parameters"]:
                    required = [
                        r
                        for r in filtered_schema["parameters"]["required"]
                        if r not in ("context", "ctx")
                    ]
                    filtered_schema["parameters"]["required"] = required

            return filtered_schema

        return schema

    def _inject_description_param(self, schema: dict) -> dict:
        """Inject __description parameter into tool schema.

        This parameter requires the LLM to provide a human-readable description
        of what it's doing with each tool call (e.g., 'Searching for Tesla news').
        """
        import copy

        schema = copy.deepcopy(schema)
        if "parameters" not in schema:
            schema["parameters"] = {"type": "object", "properties": {}, "required": []}

        # Ensure properties dict exists
        if "properties" not in schema["parameters"]:
            schema["parameters"]["properties"] = {}

        # Add __description as a required parameter
        schema["parameters"]["properties"]["__description"] = {
            "type": "string",
            "description": "Brief, user-friendly description of what you're doing with this tool call (e.g., 'Searching for Tesla stock price' not just 'search_web')"
        }

        # Make it required
        if "required" not in schema["parameters"]:
            schema["parameters"]["required"] = []
        if "__description" not in schema["parameters"]["required"]:
            schema["parameters"]["required"].insert(0, "__description")

        return schema

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
        if hasattr(tool, "context_injection"):
            pattern = tool.context_injection.get("pattern", "none")
            return pattern in ("first_param", "keyword")
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

        tools_text = "\n".join(descriptions)

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
                    param_descriptions.append(f"{param_name}: {param_type}({enum_str})")
                else:  # Just indicate there are allowed values
                    param_descriptions.append(f"{param_name}: {param_type}(allowed values defined)")
            else:
                param_descriptions.append(f"{param_name}: {param_type}")

        params_str = ", ".join(param_descriptions)
        return f"- {tool_name}({params_str}): {desc}"
