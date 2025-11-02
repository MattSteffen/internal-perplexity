"""Ollama client implementation using OpenAI-compatible API."""

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.config import settings


class OllamaClient:
    """Ollama client wrapper using OpenAI-compatible API."""

    def __init__(self) -> None:
        """Initialize the Ollama client."""
        self._client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key=settings.api_key,
        )

    async def create_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion using Ollama.

        Args:
            model: The model identifier.
            messages: List of message dictionaries.
            stream: Whether to stream the response.
            **kwargs: Additional parameters passed to the OpenAI client.

        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk].

        Raises:
            APIError: If the API call fails.
        """
        # Build request body
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        # API errors will be raised and handled by the endpoint
        if stream:
            # Return the async stream directly
            stream_response = await self._client.chat.completions.create(**body)
            return stream_response  # type: ignore[no-any-return]
        else:
            # Return the completion directly
            completion_response = await self._client.chat.completions.create(**body)
            return completion_response  # type: ignore[no-any-return]

    async def create_embedding(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """Create embeddings using Ollama.

        Args:
            model: The model identifier.
            input: Input text(s) to embed.
            **kwargs: Additional parameters passed to the OpenAI client.

        Returns:
            CreateEmbeddingResponse with embeddings.

        Raises:
            APIError: If the API call fails.
        """
        # Build request body
        body: dict[str, Any] = {
            "model": model,
            "input": input,
            **kwargs,
        }

        # API errors will be raised and handled by the endpoint
        embedding_response = await self._client.embeddings.create(**body)
        return embedding_response  # type: ignore[no-any-return]


# Singleton instance
ollama_client = OllamaClient()
