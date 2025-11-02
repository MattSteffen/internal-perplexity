"""Base protocol for tools."""

from typing import Any, Protocol

from openai.types.chat import ChatCompletionToolParam


class Tool(Protocol):
    """Protocol for tools that can be called by the LLM."""

    def get_definition(self) -> ChatCompletionToolParam:
        """Get the OpenAI tool definition for this tool.

        Returns:
            ChatCompletionToolParam with name, type, and function schema.
        """
        ...

    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with the given arguments.

        Args:
            arguments: The arguments for the tool, parsed from JSON.

        Returns:
            The result of the tool execution as a JSON string.
        """
        ...
