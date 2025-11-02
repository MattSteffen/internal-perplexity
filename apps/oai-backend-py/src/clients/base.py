"""Base protocol for chat completion clients."""

from collections.abc import AsyncIterator
from typing import Any, Protocol

from openai.types.chat import ChatCompletion, ChatCompletionChunk


class ChatCompletionClient(Protocol):
    """Protocol for clients that can handle chat completions."""

    async def create_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion.

        Args:
            model: The model identifier.
            messages: List of message dictionaries with 'role' and 'content'.
            stream: Whether to stream the response.
            **kwargs: Additional parameters.

        Returns:
            ChatCompletion if stream=False, AsyncIterator[ChatCompletionChunk] if stream=True.
        """
        ...
