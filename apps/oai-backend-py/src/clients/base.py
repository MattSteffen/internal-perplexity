"""Base protocols for chat completion and embedding clients."""

from collections.abc import AsyncIterator
from typing import Any, Protocol

from openai.types import CreateEmbeddingResponse
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


class EmbeddingClient(Protocol):
    """Protocol for clients that can handle embeddings."""

    async def create_embedding(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings.

        Args:
            model: The model identifier.
            input: Input text(s) to embed.
            **kwargs: Additional parameters.

        Returns:
            CreateEmbeddingResponse with embeddings.
        """
        ...
