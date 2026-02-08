"""Tool registry for managing available tools."""

from typing import Any

from src.tools.base import Tool
from src.tools.milvus_search import MilvusSearchTool
from src.tools.weather import WeatherTool

# Registry of all available tools
_AVAILABLE_TOOLS: dict[str, Tool] = {
    "get_weather": WeatherTool(),
    "milvus_search": MilvusSearchTool(),
}


class ToolRegistry:
    """Registry for managing and executing tools."""

    def __init__(self, tools: dict[str, Tool] | None = None) -> None:
        """Initialize the tool registry.

        Args:
            tools: Optional dictionary of tool name to Tool instance mappings.
                  If None, uses the default set of tools.
        """
        self._tools = tools if tools is not None else _AVAILABLE_TOOLS.copy()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get OpenAI tool definitions for all registered tools.

        Returns:
            List of tool definitions in OpenAI format.
        """
        definitions: list[dict[str, Any]] = []
        for tool in self._tools.values():
            tool_def = tool.get_definition()
            # ChatCompletionToolParam is a TypedDict, convert to plain dict
            if isinstance(tool_def, dict):
                definitions.append(dict(tool_def))  # type: ignore[arg-type]
            else:
                # If it's a Pydantic model or similar, try to convert
                if hasattr(tool_def, "model_dump"):
                    definitions.append(tool_def.model_dump())
                elif hasattr(tool_def, "dict"):
                    definitions.append(tool_def.dict())
                else:
                    # Manual conversion
                    definitions.append(
                        {
                            "type": getattr(tool_def, "type", "function"),
                            "function": getattr(tool_def, "function", {}),
                        }
                    )
        return definitions

    def get_tool_definition(self, tool_name: str) -> dict[str, Any]:
        """Get the OpenAI tool definition for a specific tool.

        Args:
            tool_name: The name of the tool to get the definition for.

        Returns:
            The tool definition in OpenAI format.
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")
        return self._tools[tool_name].get_definition()

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
        """Execute a tool by name with the given arguments.

        Args:
            tool_name: The name of the tool to execute.
            arguments: The arguments for the tool.
            metadata: Optional metadata for tool initialization (user-specific data).

        Returns:
            The result of the tool execution as a JSON string.

        Raises:
            KeyError: If the tool is not registered.
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")

        tool = self._tools[tool_name]
        # Pass metadata through arguments if the tool needs it
        # Tools can check for a special '_metadata' key or use it directly
        if metadata:
            # Add metadata to arguments so tools can access it
            # Tools that need user-specific initialization can check for this
            execution_args = arguments.copy()
            execution_args["_metadata"] = metadata
            return await tool.execute(execution_args)
        return await tool.execute(arguments)

    def register_tool(self, tool_name: str, tool: Tool) -> None:
        """Register a new tool.

        Args:
            tool_name: The name of the tool (must match the name in get_definition).
            tool: The Tool instance to register.
        """
        self._tools[tool_name] = tool


# Singleton registry instance
tool_registry = ToolRegistry()
