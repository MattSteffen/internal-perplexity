"""Base protocol for agents."""

from typing import Any, Protocol

from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion
from pymilvus import MilvusClient  # type: ignore


class Agent(Protocol):
    """Protocol for agents that handle agent-specific requests."""

    async def create_completion(
        self,
        agent_name: str,
        body: dict[str, Any],
        user: dict[str, Any] | None = None,
        milvus_client: MilvusClient | None = None,
    ) -> StreamingResponse | ChatCompletion:
        """Create an agent completion.

        Args:
            agent_name: The name of the agent being invoked.
            body: Request body parsed from JSON.
            user: Optional authenticated user context.
            milvus_client: Optional Milvus client instance.

        Returns:
            StreamingResponse for streaming or ChatCompletion for non-streaming.
        """
        ...
